"""Train the v53 MiMoMix language model to actually generate text.

`docs/V53_MIMOMIX_ARCHITECTURE.md` is explicit that its own numbers prove the
mechanisms work and nothing about quality, because the backends are randomly
initialised: *"Their text is noise by design."* This script removes that caveat
for one corpus. It trains `MiMoMixModel` -- hybrid SWA/global attention with
learnable sinks, auxiliary-loss-free sparse MoE, multi-token prediction, and the
recursive thinking core -- on real dialogue, and writes a checkpoint the API can
serve.

What the result is, stated before any number: a **small domain-specific chat
model**. The corpus is 120,000 templated coding-assistant turns with 292 distinct
word types. The model learns to hold a turn in that register. It has no world
knowledge, because there is none in the data to learn, and a word outside the
vocabulary can never be generated.

Held-out loss is measured on rows the model never trained on, but the corpus is
templated and only 37,543 of its 120,000 responses are distinct, so a validation
row's response may still appear in training. That makes validation perplexity a
measure of fit to the template distribution, not of generalisation to unseen
language. The receipt records it under that name.

Usage::

    python source/train_mimomix_talk.py --steps 4000
    python source/train_mimomix_talk.py --steps 200 --pairs 4000 --run_name smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_decoding as decoding  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
from device_utils import resolve_device  # noqa: E402
from mimomix_core import MiMoMixConfig, MiMoMixModel, SparseMoEFeedForward  # noqa: E402

CHECKPOINT_SCHEMA = "supermix-v57-talk-checkpoint-v1"
RECEIPT_SCHEMA = "supermix-v57-talk-benchmark-v1"

#: Prompts used to watch the model learn to talk. They are held fixed across the
#: whole run so successive samples are comparable.
PROBE_PROMPTS = (
    "hello",
    "can you help me with tests",
    "why is my script failing",
    "what is your name",
    "write a unit test for login",
)


def save_talk_checkpoint(
    path: Path,
    model: MiMoMixModel,
    tokenizer: text_utils.WordTokenizer,
    extra: Optional[Dict[str, Any]] = None,
    optimiser: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> None:
    """Persist config, weights and tokenizer together.

    The tokenizer travels with the weights on purpose: a checkpoint whose
    vocabulary is missing is not reloadable, and one loaded against the wrong
    vocabulary produces confident nonsense rather than an error.

    `optimiser` and `scheduler` are optional and, when given, make the checkpoint
    a true resume point rather than just a set of weights. Restoring weights
    alone restarts AdamW's moments and the learning-rate schedule, which is not
    free: continuing v62 that way sent dev loss from 0.8919 up to 1.0036 and cost
    roughly 1,500 steps re-warming before it recovered. On a multi-day run that
    is hours of wasted compute, so the state is written when available.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "config": model.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "tokenizer": tokenizer.to_dict(),
        "extra": extra or {},
    }
    if optimiser is not None:
        payload["optimiser_state"] = optimiser.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()

    # Write beside the target and rename, rather than over it.
    #
    # `torch.save` truncates the destination first, so a process that dies
    # partway through leaves a corrupt file where the checkpoint was. That is
    # not hypothetical here: v64 and v74 both segfaulted at an eval/checkpoint
    # boundary, and for a `.partial.pt` the file being overwritten is the only
    # thing standing between a crash and losing the whole run -- v74 was 9.2
    # hours in. `os.replace` is atomic on POSIX and on Windows, so the previous
    # checkpoint survives intact until the new one is completely written.
    staging = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, staging)
        os.replace(staging, path)
    except BaseException:
        # Includes KeyboardInterrupt: leaving a stray .tmp beside the
        # checkpoint would be read by nothing but would confuse the next
        # person to look in the directory.
        staging.unlink(missing_ok=True)
        raise


def load_talk_checkpoint(path, map_location: str = "cpu"):
    """Rebuild `(model, tokenizer, payload)` from a v57 checkpoint."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint")
    config = MiMoMixConfig(**payload["config"])
    model = MiMoMixModel(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokenizer = text_utils.WordTokenizer.from_dict(payload["tokenizer"])
    return model, tokenizer, payload


@torch.no_grad()
def generate_reply(
    model: MiMoMixModel,
    tokenizer: text_utils.WordTokenizer,
    prompt: str,
    max_new_tokens: int = 48,
    speculative: bool = True,
    thinking_cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate one reply, optionally through MTP self-speculative decoding.

    Speculative decoding here is not a quality choice: for greedy decoding it is
    provably token-identical to one-at-a-time generation, so it only changes how
    many trunk forwards the reply costs.
    """

    model.eval()
    ids, _ = tokenizer.encode_turn(prompt, None)
    prompt_ids = torch.tensor([ids], dtype=torch.long)
    started = time.perf_counter()
    generate = decoding.speculative_generate if speculative else decoding.greedy_generate
    result = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=text_utils.EOS,
        thinking_cycles=thinking_cycles,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    reply = tokenizer.decode(result.new_tokens[0].tolist()).strip()
    return {
        "prompt": prompt,
        "reply": reply,
        "tokens": int(result.new_tokens.shape[1]),
        "latency_ms": round(elapsed, 2),
        "acceptance_length": round(result.stats.acceptance_length, 4),
        "trunk_forwards": int(result.stats.verify_forwards),
    }


@torch.no_grad()
def evaluate(
    model: MiMoMixModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Held-out loss over reply tokens only, plus perplexity and bits/token."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for index in range(0, inputs.shape[0], batch_size):
        # Widened per batch; the corpus itself is stored compactly.
        batch_x = inputs[index : index + batch_size].long()
        batch_y = labels[index : index + batch_size].long()
        out = model(batch_x, return_mtp=False)
        shift_logits = out.logits[:, :-1]
        shift_labels = batch_y[:, 1:]
        counted = int((shift_labels != -100).sum())
        if counted == 0:
            continue
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss)
        total_tokens += counted
    mean = total_loss / max(1, total_tokens)
    return {
        "loss": round(mean, 6),
        "perplexity": round(math.exp(min(20.0, mean)), 4),
        "bits_per_token": round(mean / math.log(2), 6),
        "scored_tokens": total_tokens,
    }


@torch.no_grad()
def routing_report(
    model: MiMoMixModel, inputs: torch.Tensor, batch_size: int = 8, batches: int = 8
) -> Dict[str, Any]:
    """Accumulate expert load over several batches, not one forward.

    The bias rule is a control loop and it rings, so a single batch measures the
    phase of the oscillation. Worse, reading it after a *generation* call reports
    a batch of one with a handful of tokens, where most experts are idle as
    arithmetic rather than starved.
    """

    layers = [m for m in model.modules() if isinstance(m, SparseMoEFeedForward)]
    if not layers:
        return {"moe_layers": 0}
    model.eval()
    totals = [torch.zeros(layer.n_routed) for layer in layers]
    counted = 0
    for index in range(0, min(inputs.shape[0], batch_size * batches), batch_size):
        # Widened per batch; the corpus is stored in a compact integer type.
        model(inputs[index : index + batch_size].long(), return_mtp=False)
        for position, layer in enumerate(layers):
            totals[position] += layer.last_expert_load.detach().cpu()
        counted += 1

    rows = []
    for position, load in enumerate(totals):
        mass = float(load.sum())
        share = (load / mass) if mass > 0 else load
        n = int(load.numel())
        entropy = float(-(share * share.clamp_min(1e-12).log()).sum())
        rows.append(
            {
                "layer": position,
                "normalised_entropy": round(entropy / math.log(n), 6) if n > 1 else 1.0,
                "starved_experts": int((load == 0).sum()),
            }
        )
    return {
        "moe_layers": len(rows),
        "batches_accumulated": counted,
        "mean_normalised_entropy": round(
            sum(r["normalised_entropy"] for r in rows) / len(rows), 6
        ),
        "total_starved": sum(r["starved_experts"] for r in rows),
        "per_layer": rows,
    }


# ---------------------------------------------------------------------------
# Configuration wiring (v82)
# ---------------------------------------------------------------------------
#
# Before v82 `build_config` set 22 of MiMoMixConfig's fields and hardcoded three
# more (`rope_scaling="none"`, and `native_context`/`max_position_embeddings`
# pinned to `--sequence_length`). Everything else -- 33 fields, including
# `router_score_function`, `n_shared_experts`, `n_dense_layers`, every
# differential/MoD/MLA knob and the whole rope-extension policy -- could not be
# reached from either trainer, because `build_config` is the single chokepoint
# both of them call. v80 therefore ran `router_score_function="softmax"` not by
# choice but because nothing could pass anything else.
#
# The rule for this table: **every default here reproduces the v80 run exactly.**
# `test_train_v82.py::test_v80_config_reproduced_from_flag_defaults` asserts
# that against the config stored inside `output/v80_omni/v80_omni.pt`, and that
# assertion is the one worth keeping if all the others were deleted.


def parse_layer_list(value: Any) -> Optional[Tuple[int, ...]]:
    """``"1,3"`` -> ``(1, 3)``; ``None``/``""``/``"none"`` -> ``None``.

    Used for ``--global_layers``, where ``None`` means "keep the uniform
    ``hybrid_ratio`` interleave", which is what every run up to v80 did.
    """

    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return tuple(int(part) for part in re.split(r"[,\s]+", text) if part)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "none"):
        return None
    return float(value)


#: ``config field name -> (args attribute, transform)``.
#:
#: A transform of ``None`` means "copy the attribute through". Fields whose
#: value is not a plain copy (negated store_true flags, the
#: ``sequence_length`` fallbacks) carry a callable that takes the whole
#: ``args`` namespace.
_CONFIG_FROM_ARGS: Dict[str, Any] = {
    "hidden_size": "hidden_size",
    "n_layers": "n_layers",
    "n_heads": "n_heads",
    "n_kv_heads": "n_kv_heads",
    "head_dim": "head_dim",
    "intermediate_size": "intermediate_size",
    "dropout": "dropout",
    "rms_norm_eps": "rms_norm_eps",
    "tie_word_embeddings": lambda a: not a.no_tie_word_embeddings,
    "sliding_window": "sliding_window",
    "hybrid_ratio": "hybrid_ratio",
    "final_layer_global": lambda a: not a.no_final_layer_global,
    "attention_sink": lambda a: not a.no_attention_sink,
    "rope_theta": "rope_theta",
    "rope_local_theta": lambda a: _optional_float(a.rope_local_theta),
    "rope_scale_local": "rope_scale_local",
    # Before v82 both were pinned to --sequence_length. `None` keeps that,
    # which is why the defaults are None rather than the dataclass values.
    "native_context": lambda a: int(a.native_context or a.sequence_length),
    "max_position_embeddings": lambda a: int(
        a.max_position_embeddings or a.sequence_length
    ),
    "rope_scaling": "rope_scaling",
    "yarn_beta_fast": "yarn_beta_fast",
    "yarn_beta_slow": "yarn_beta_slow",
    "rotary_dim": lambda a: (None if a.rotary_dim is None else int(a.rotary_dim)),
    "qk_norm": "qk_norm",
    "attention_output_gate": "attention_output_gate",
    "attention_sink_kinds": "attention_sink_kinds",
    "global_layers": lambda a: parse_layer_list(a.global_layers),
    "use_moe": lambda a: not a.no_moe,
    "n_dense_layers": "n_dense_layers",
    "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts",
    "moe_top_k": "moe_top_k",
    "moe_intermediate_size": "moe_intermediate_size",
    "router_bias_update_speed": "router_bias_update_speed",
    "router_bias_auto_update": "router_bias_auto_update",
    "router_z_loss_coef": "router_z_loss_coef",
    "router_balance_loss_coef": "router_balance_loss_coef",
    "router_score_function": "router_score_function",
    "norm_topk_prob": lambda a: not a.no_norm_topk_prob,
    "moe_balance_scope": "moe_balance_scope",
    "n_mtp_layers": "n_mtp_layers",
    "mtp_loss_weight": "mtp_loss_weight",
    "use_thinking_core": lambda a: not a.no_thinking_core,
    "thinking_latent_dim": "thinking_latent_dim",
    "thinking_cycles": "thinking_cycles",
    "thinking_max_cycles": "thinking_max_cycles",
    "thinking_inner_steps": "thinking_inner_steps",
    "thinking_residual_init": "thinking_residual_init",
    "ponder_loss_weight": "ponder_loss_weight",
    "consistency_loss_weight": "consistency_loss_weight",
    "use_differential_attention": "use_differential_attention",
    "differential_lambda_init": "differential_lambda_init",
    "differential_output_norm": lambda a: not a.no_differential_output_norm,
    "differential_noise_ratio": "differential_noise_ratio",
    "use_mod": "use_mod",
    "mod_capacity_ratio": "mod_capacity_ratio",
    "mod_causal_predictor": lambda a: not a.no_mod_causal_predictor,
    "mod_predictor_loss_coef": "mod_predictor_loss_coef",
    "use_mla": "use_mla",
    "mla_latent_dim": "mla_latent_dim",
    "mla_pe_dim": "mla_pe_dim",
    "mla_global_only": lambda a: not a.no_mla_global_only,
    "use_cns_core": "cns_core",
    "cns_nodes": "cns_nodes",
    "cns_steps": "cns_steps",
    "cns_after_layer": "cns_after_layer",
    "cns_io": "cns_io",
    "cns_graph": "cns_graph",
    "cns_wiring": "cns_wiring",
    # v93 (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md D2/D3). Every default is
    # the v91 value, so a v89/v91 command still builds the identical config.
    # `grow_layers` is deliberately NOT here: it must be applied to the built
    # config through `pin_layout_for_growth` (it changes n_layers and pins
    # global_layers), which the generalisation trainer does after build_config.
    "cns_spare_nodes": "cns_spare_nodes",
    "cns_read_layers": lambda a: parse_layer_list(a.cns_read_layers) or (),
    "cns_write_layers": lambda a: parse_layer_list(a.cns_write_layers) or (),
    "cns_to_thinking": "cns_to_thinking",
    "cns_temporal": "cns_temporal",
    "moe_spare_experts": "moe_spare_experts",
}


def config_field_names() -> frozenset:
    """Names `MiMoMixConfig` actually accepts, read from the dataclass."""

    return frozenset(f.name for f in dataclasses.fields(MiMoMixConfig))


def resolved_config_kwargs(
    args: argparse.Namespace, vocab_size: int
) -> Tuple[Dict[str, Any], List[str]]:
    """``(every requested field, names this MiMoMixConfig cannot accept)``.

    The first element is unfiltered on purpose: `build_config` needs to compare
    a dropped field's requested value against its default before deciding
    whether dropping it is harmless. The drop list exists so a flag the
    installed `mimomix_core` cannot honour fails loudly instead of being
    silently ignored.
    """

    known = config_field_names()
    requested: Dict[str, Any] = {"vocab_size": vocab_size}
    for field_name, source in _CONFIG_FROM_ARGS.items():
        if callable(source):
            requested[field_name] = source(args)
        else:
            requested[field_name] = getattr(args, source)
    dropped = sorted(name for name in requested if name not in known)
    return requested, dropped


def build_config(args: argparse.Namespace, vocab_size: int) -> MiMoMixConfig:
    """Every `MiMoMixConfig` field, from the flags, with v80 defaults.

    A field this build of `mimomix_core` does not have is dropped -- but only
    when the flag is still at its default. Dropping a value the operator
    actually asked for would train a configuration nobody chose and report it
    as the one they picked, which is the failure mode this whole release is
    about.
    """

    requested, dropped = resolved_config_kwargs(args, vocab_size)
    if dropped:
        defaults, _ = resolved_config_kwargs(
            build_parser().parse_args([]), vocab_size
        )
        overridden = [
            name for name in dropped if requested[name] != defaults.get(name)
        ]
        if overridden:
            raise SystemExit(
                "this build of mimomix_core.MiMoMixConfig has no field(s) "
                f"{overridden}, but they were set on the command line. Training "
                "would silently ignore them. Update mimomix_core.py or drop the "
                "flags."
            )
    known = config_field_names()
    return MiMoMixConfig(
        **{name: value for name, value in requested.items() if name in known}
    )


# ---------------------------------------------------------------------------
# v82 training-loop instruments
# ---------------------------------------------------------------------------

#: Default cap on the mid-run accuracy probe's generation length.
#:
#: The old hardcoded 64 was not large enough for the tasks the probe samples,
#: and the shortfall was not marginal. Measured with the v80 tokenizer over a
#: full scan of all 911,478 rows of `datasets/v80/v80_combined.jsonl` (the
#: forty longest responses per task, tokenized exactly), the longest reply each
#: task requires is:
#:
#:     momentum 99   work 97   arithmetic_series 97   wave_speed 96
#:     electrical_power 92   force 89   kinetic_energy 85   voltage 64
#:     average 45   combination 42   percent 41   two_step 37
#:     acceleration 33   power 33   molarity 28   sequence 26
#:     word_problem 26   algebra_one_step 25   multiplication 21   division 20
#:
#: Seven of twenty-two tasks need more than 64 tokens. Sampling the generators
#: directly (400 draws per task) agrees: arithmetic_series replies are median
#: 82 / max 85 with 100% over 64, combination median 61 / max 66 with 22.5%
#: over 64, kinetic_energy median 55 / max 57. So a v80-shaped model scored
#: 0.00 on arithmetic_series whatever it had learned, because the probe cut the
#: answer off before the number.
#:
#: 112 = the measured 99 plus 13 tokens of headroom, rounded to a multiple of
#: 16. It is a *cap*, not a cost: generation stops at EOS, so a model that has
#: learned to end its turn pays the reply length and nothing more. The cap only
#: binds on replies that never terminate, which is exactly when a probe should
#: give up. Note the honest edge: the longest measured prompt is 30 tokens, so
#: a worst-case 112-token reply reaches position 142 against v80's 128-token
#: trained context. RoPE extrapolates rather than raising there, and the
#: alternative -- capping below the answer -- is the bug being fixed.
DEFAULT_PROBE_MAX_NEW_TOKENS = 112


def mtp_weight_at(
    step: int,
    total_steps: int,
    start_weight: float,
    final_weight: Optional[float],
    warmup_fraction: float,
) -> float:
    """MTP loss weight for `step` under the DeepSeek-V3 / MiMo schedule.

    HYPOTHESIS, NOT A MEASURED GAIN. DeepSeek-V3 reports decaying the MTP loss
    weight from 0.3 to 0.1 partway through pretraining, and MiMo reports no
    benefit from extra pretraining depths. Neither has been run on this stack;
    no Supermix run has yet used a schedule at all. `final_weight=None` (the
    default) returns `start_weight` at every step, which is exactly what v80
    trained under.

    The shape is a linear ramp from 0 to `start_weight` over the first
    `warmup_fraction` of the run, then a linear decay to `final_weight` across
    the rest.
    """

    if final_weight is None and warmup_fraction <= 0.0:
        return float(start_weight)
    total = max(1, int(total_steps))
    progress = min(1.0, max(0.0, step / total))
    warmup = max(0.0, min(1.0, float(warmup_fraction)))
    if warmup > 0.0 and progress < warmup:
        return float(start_weight) * (progress / warmup)
    if final_weight is None:
        return float(start_weight)
    span = 1.0 - warmup
    if span <= 0.0:
        return float(final_weight)
    tail = (progress - warmup) / span
    return float(start_weight) + (float(final_weight) - float(start_weight)) * tail


#: Parameter names that get no weight decay under ``--decay_mode no_norm_bias``.
#:
#: AdamW's decay is a pull toward zero applied every step. On a matrix that is
#: a norm-preserving regulariser; on a *gain* (RMSNorm's `weight`, initialised
#: to 1) it is a pull toward a degenerate layer, and on a calibration scalar
#: (`log_temperature`, `residual_scale`) it is a pull toward a specific,
#: meaningful value nobody chose. v80 decayed all of them at 0.01, including
#: the thinking core's residual gate -- the single scalar the whole recursive
#: core is multiplied by.
#:
#: HYPOTHESIS, NOT A MEASURED GAIN: excluding them is standard practice, but no
#: matched pair has been run on this stack. `--decay_mode all` is the default
#: and reproduces v80 exactly.
def parameter_groups(
    model: torch.nn.Module, weight_decay: float, decay_mode: str = "all"
) -> List[Dict[str, Any]]:
    """AdamW parameter groups for `--decay_mode`.

    `all` returns a single group, which is byte-identical to passing the
    model's parameters straight to AdamW -- the same tensors in the same order
    with the same hyperparameters -- so a v80 command produces the same
    optimiser state dict shape it always did.
    """

    if decay_mode == "all":
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]
    if decay_mode != "no_norm_bias":
        raise ValueError(f"unknown decay_mode {decay_mode!r}")

    decayed, undecayed, undecayed_names = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        leaf = name.rsplit(".", 1)[-1]
        # 1-D tensors are norms, biases, sinks, router gates and scalars. The
        # named exceptions are the calibration/gate scalars, which are 0-D.
        if parameter.ndim <= 1 or leaf in (
            "bias",
            "residual_scale",
            "log_temperature",
            "expert_bias",
            "lambda_init",
        ):
            undecayed.append(parameter)
            undecayed_names.append(name)
        else:
            decayed.append(parameter)
    groups = [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
    groups[1]["_names"] = undecayed_names  # for the receipt; AdamW ignores it
    return groups


def sample_batch_indices(
    generator: torch.Generator,
    population: int,
    batch_size: int,
    repeat_subset_size: int = 0,
    repeat_subset_probability: float = 0.0,
) -> torch.Tensor:
    """Row indices for one batch, optionally two-set (Charton & Kempe 2024).

    HYPOTHESIS, NOT A MEASURED GAIN. Charton & Kempe (2024) report a model
    trained on 25M GCD examples repeated ~24x learning 62 GCDs where unlimited
    fresh data learned 27. Nothing like it has been run here.

    What *is* measured is that repetition is nearly free on this corpus: v80
    drew 18,000 x 16 = 288,000 blocks with replacement from 866,142, so the
    expected unseen fraction is exp(-288000/866142) = exp(-0.3325) = 0.717.
    v80 saw at most 28.3% of its own training data, so a run that revisits a
    fixed subset is not giving up coverage it was ever going to have.

    `repeat_subset_size = 0` (the default) reproduces v80's plain uniform
    with-replacement sampling exactly, including the generator's draw sequence.
    """

    if repeat_subset_size <= 0 or repeat_subset_probability <= 0.0:
        return torch.randint(0, population, (batch_size,), generator=generator)
    size = min(int(repeat_subset_size), population)
    draws = torch.randint(0, population, (batch_size,), generator=generator)
    coin = torch.rand((batch_size,), generator=generator)
    from_subset = torch.randint(0, size, (batch_size,), generator=generator)
    return torch.where(coin < repeat_subset_probability, from_subset, draws)


def count_zero_expert_grads(model: torch.nn.Module) -> Dict[str, int]:
    """MoE parameters whose gradient is exactly zero after backward.

    MiMo names "num-zeros" as its MoE stability early-warning signal: an expert
    that stops receiving gradient has left the routing distribution and is not
    coming back on its own. v80 finished with 30 of 144 expert slots at zero
    load and nobody knew until the run was over.

    Counted over the *expert* parameters only, and only where a gradient
    exists. Cost measured on this box at the v80 shape: 3.0 ms per call against
    a 2.0 s step, i.e. 0.15% if called every step and immaterial at the
    per-eval cadence it is actually called at.
    """

    zero_parameters = 0
    zero_elements = 0
    total_parameters = 0
    total_elements = 0
    for module in model.modules():
        if not isinstance(module, SparseMoEFeedForward):
            continue
        for _, parameter in module.named_parameters(recurse=True):
            if parameter.grad is None:
                continue
            total_parameters += 1
            total_elements += parameter.numel()
            zeros = int((parameter.grad == 0).sum())
            zero_elements += zeros
            if zeros == parameter.numel():
                zero_parameters += 1
    return {
        "expert_tensors": total_parameters,
        "expert_tensors_with_zero_grad": zero_parameters,
        "expert_elements": total_elements,
        "expert_elements_with_zero_grad": zero_elements,
    }


def router_bias_report(model: torch.nn.Module) -> Dict[str, Any]:
    """Per-layer expert-bias magnitude for the aux-loss-free balance rule.

    The rule adds `expert_bias` to the router's affinity scores before top-k.
    Under `router_score_function="softmax"` those scores sum to 1 across 48
    experts, so their mean is 0.0208 -- and v80's biases reached 10.4 to 11.7.
    Selection was therefore decided by the bias, not by the router, and 30 of
    144 expert slots starved anyway. Nothing in the v80 receipt records this;
    this is the instrument that would have.
    """

    rows: List[Dict[str, Any]] = []
    for index, module in enumerate(
        m for m in model.modules() if isinstance(m, SparseMoEFeedForward)
    ):
        bias = module.expert_bias.detach()
        rows.append(
            {
                "layer": index,
                "max_abs_bias": round(float(bias.abs().max()), 6),
                "bias_range": round(float(bias.max() - bias.min()), 6),
                "mean_bias": round(float(bias.mean()), 6),
                "starved_experts": int((module.last_expert_load.detach() == 0).sum()),
            }
        )
    if not rows:
        return {"moe_layers": 0}
    return {
        "moe_layers": len(rows),
        "max_abs_bias": max(r["max_abs_bias"] for r in rows),
        "max_bias_range": max(r["bias_range"] for r in rows),
        "per_layer": rows,
    }


#: Hyperparameters a receipt must record for `--compare` to be able to tell two
#: runs apart.
#:
#: Before v82 the receipt held seven keys, and `compare()` checks *only* the
#: keys that are there -- so two runs that differed in `--select_on`,
#: `--digit_tokens`, `--turn_aligned_packing`, `--min_response_characters`,
#: `--corpus_jsonl`, `--amp`, `--eval_every`, `--accuracy_every`,
#: `--accuracy_problems`, `--max_vocab`, `--pct_start` or the probe cap
#: compared as "matched". Every one of those changes what the model is or what
#: the numbers mean.
RECORDED_HYPERPARAMETERS: Tuple[str, ...] = (
    "steps",
    "batch_size",
    "sequence_length",
    "lr",
    "weight_decay",
    "seed",
    "split_seed",
    "pct_start",
    "max_vocab",
    "eval_every",
    "eval_batch_size",
    "accuracy_every",
    "accuracy_problems",
    "probe_max_new_tokens",
    "select_on",
    "turn_aligned_packing",
    "digit_tokens",
    "reverse_digits",
    "min_response_characters",
    "corpus_jsonl",
    "database",
    "amp",
    "decay_mode",
    "repeat_subset_fraction",
    "repeat_subset_prob",
    "mtp_loss_weight",
    "mtp_loss_weight_final",
    "mtp_weight_warmup_fraction",
    "restore_best",
    "start_step",
    "init_from",
    "arm",
    # v93 neurogenesis. `recorded_hyperparameters` skips names a trainer's
    # parser lacks, so the generalisation-only flags (grow_every ...) are
    # listed here once and recorded wherever they exist.
    "cns_spare_nodes",
    "cns_read_layers",
    "cns_write_layers",
    "cns_to_thinking",
    "cns_temporal",
    "moe_spare_experts",
    "grow_layers",
    "extend_vocab",
    "max_new_tokens_vocab",
    "grow_every",
    "grow_modules",
    "grow_edges",
    "grow_taps",
    "grow_experts",
    "prune_threshold",
    "witness_rows",
    "ablation_rows",
)


def recorded_hyperparameters(args: argparse.Namespace) -> Dict[str, Any]:
    """The subset of `args` a receipt records, skipping flags this trainer lacks."""

    payload: Dict[str, Any] = {}
    for name in RECORDED_HYPERPARAMETERS:
        if hasattr(args, name):
            value = getattr(args, name)
            payload[name] = value if _json_safe(value) else str(value)
    return payload


def _json_safe(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def resolved_flags(args: argparse.Namespace) -> Dict[str, Any]:
    """Every parsed flag, JSON-safe. The run's full resolved configuration."""

    return {
        name: (value if _json_safe(value) else str(value))
        for name, value in sorted(vars(args).items())
    }


def tokenizer_options(args: argparse.Namespace) -> Dict[str, Any]:
    """Tokenizer kwargs the installed `WordTokenizer` actually accepts.

    `--digit_tokens` and `--reverse_digits` are trainer flags; the second is
    new in v82 and older checkouts of `mimomix_text` do not have it. Filtering
    against the real signature keeps the trainer runnable against either, and
    both default to False, so the v80 tokenizer is unchanged.
    """

    import inspect

    wanted = {
        "digit_tokens": bool(getattr(args, "digit_tokens", False)),
        "reverse_digits": bool(getattr(args, "reverse_digits", False)),
    }
    accepted = inspect.signature(text_utils.WordTokenizer.build).parameters
    missing = [name for name, value in wanted.items() if value and name not in accepted]
    if missing:
        raise SystemExit(
            f"this build of mimomix_text.WordTokenizer.build has no {missing} "
            "parameter(s); the flag would be silently ignored."
        )
    return {name: value for name, value in wanted.items() if name in accepted}


def corpus_fingerprint(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Path, byte size and sha256 of the corpus file, for `--compare`.

    Two runs whose receipts agree on every hyperparameter can still have been
    trained on different data, and until v82 nothing in the receipt could tell
    you. Hashing 217 MB takes about a second.
    """

    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(file_path),
        "exists": True,
        "bytes": file_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def response_token_report(
    rows: Sequence[Any],
    tokenizer: text_utils.WordTokenizer,
    sample: int = 20000,
) -> Dict[str, Dict[str, Any]]:
    """Reply-token length statistics per task over a sample of corpus rows.

    `rows` is either a sequence of ``(user, assistant)`` pairs -- which have no
    task label, so everything lands under ``"(all rows)"`` -- or a sequence of
    dicts with ``task``/``assistant`` keys.

    The count is ``len(encode(assistant)) + 1``: the reply tokens plus the EOS
    the model must emit to stop. That is exactly the budget
    `generate_reply(..., max_new_tokens)` allocates.

    Sampling is a fixed stride rather than a random draw so the check is
    deterministic: a startup guard that warns on one run and not the next is
    worse than no guard.
    """

    lengths: Dict[str, List[int]] = {}
    stride = max(1, len(rows) // max(1, sample))
    for index in range(0, len(rows), stride):
        row = rows[index]
        if isinstance(row, dict):
            task = str(row.get("task") or "(untagged)")
            assistant = str(row.get("assistant", ""))
        else:
            task = "(all rows)"
            assistant = str(row[1])
        if not assistant:
            continue
        lengths.setdefault(task, []).append(len(tokenizer.encode(assistant)) + 1)
    report: Dict[str, Dict[str, Any]] = {}
    for task, values in lengths.items():
        values.sort()
        report[task] = {
            "rows": len(values),
            "median": values[len(values) // 2],
            "p95": values[max(0, int(0.95 * len(values)) - 1)],
            "max": values[-1],
        }
    return report


def check_probe_token_budget(
    report: Dict[str, Dict[str, Any]],
    cap: int,
    strict: bool = False,
    routinely: float = 0.5,
) -> Dict[str, Any]:
    """Refuse or warn when the corpus needs more tokens than the probe allows.

    This is the guard that stops CONFIRMED BUG E recurring, and it is the same
    class of failure as v67 silently dropping the `average` rows: a measurement
    that cannot see a task reports zero for it and nobody notices for a
    release. A task whose *median* reply already exceeds the cap can never be
    scored correctly, so that is what `routinely` keys on -- a p95 over the cap
    is a warning, a median over it is a structural blindness.
    """

    blind = []
    at_risk = []
    for task, stats in sorted(report.items()):
        if stats["median"] > cap:
            blind.append((task, stats))
        elif stats["p95"] > cap:
            at_risk.append((task, stats))
    result = {
        "probe_max_new_tokens": int(cap),
        "tasks_measured": len(report),
        "tasks_truncated_at_median": [t for t, _ in blind],
        "tasks_truncated_at_p95": [t for t, _ in at_risk],
        "per_task": report,
        "ok": not blind,
    }
    if blind or at_risk:
        print()
        print("!! PROBE TOKEN BUDGET WARNING " + "!" * 40)
        print(f"   --probe_max_new_tokens is {cap}.")
        for task, stats in blind:
            print(
                f"   BLIND   {task:<24} median {stats['median']:>4}  p95 "
                f"{stats['p95']:>4}  max {stats['max']:>4}  -- more than half of "
                "this task's replies cannot finish inside the cap, so its probe "
                "accuracy is structurally 0.00 whatever the model learned"
            )
        for task, stats in at_risk:
            print(
                f"   AT RISK {task:<24} median {stats['median']:>4}  p95 "
                f"{stats['p95']:>4}  max {stats['max']:>4}"
            )
        print("!" * 70, flush=True)
    if blind and strict:
        raise SystemExit(
            f"--strict: {len(blind)} task(s) have median replies longer than "
            f"--probe_max_new_tokens {cap}: {[t for t, _ in blind]}. Raise the "
            "cap or drop --strict."
        )
    return result


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    if args.torch_threads:
        torch.set_num_threads(max(1, args.torch_threads))
    device, device_info = resolve_device(args.device, preference=args.device_preference)

    corpus = text_utils.load_chat_pairs(
        args.database, limit=args.pairs, validation_fraction=args.validation_fraction, seed=args.seed
    )
    # Fields are counted separately, never concatenated: joining them would only
    # ever show the tokenizer a reply's first word with a leading space, and the
    # sentence-initial form would be missing from the vocabulary.
    tokenizer = text_utils.WordTokenizer.build(
        (field for pair in corpus.train for field in pair),
        max_vocab=args.max_vocab,
        **tokenizer_options(args),
    )
    sample_texts = [u for u, _ in corpus.validation[:400]] + [a for _, a in corpus.validation[:400]]
    text_utils.assert_roundtrip(tokenizer, sample_texts[:200])
    coverage = tokenizer.vocabulary_report(sample_texts)

    train_x, train_y = text_utils.build_training_tensors(
        corpus.train, tokenizer, args.sequence_length
    )
    val_x, val_y = text_utils.build_training_tensors(
        corpus.validation, tokenizer, args.sequence_length
    )

    if int(getattr(args, "grow_layers", 0)) > 0:
        # Depth growth is a warm-start operation (zero the appended blocks
        # after loading a source checkpoint); this trainer has no --init_from,
        # so the flag would build a model nobody asked for. Refuse rather
        # than silently ignore it.
        raise SystemExit(
            "--grow_layers is a warm-start operation of "
            "train_mimomix_generalisation.py (--init_from); this trainer "
            "trains from scratch, so drop the flag or raise --n_layers."
        )
    config = build_config(args, tokenizer.vocab_size)
    model = MiMoMixModel(config).to(device)
    parameters = model.parameter_report()

    print(f"v57 MiMoMix talk | corpus {corpus.source}")
    print(f"  pairs        {len(corpus.train):,} train / {len(corpus.validation):,} validation")
    print(f"  vocabulary   {tokenizer.vocab_size} types, coverage {coverage['coverage']:.4f}")
    print(f"  sequences    {tuple(train_x.shape)} train / {tuple(val_x.shape)} validation")
    print(f"  parameters   {parameters['total']:,} total / {parameters['active_per_token']:,} active")
    print(f"  layout       {''.join('G' if k=='global' else 'L' for k in model.layout)}"
          f"  window {config.sliding_window}")
    print(f"  device       {device_info.get('resolved', device)}", flush=True)

    groups = parameter_groups(model, args.weight_decay, args.decay_mode)
    optimiser = torch.optim.AdamW(
        [{k: v for k, v in g.items() if not k.startswith("_")} for g in groups],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=max(1, args.steps), pct_start=args.pct_start
    )
    repeat_subset_size = int(args.repeat_subset_fraction * train_x.shape[0])

    history: List[Dict[str, Any]] = []
    samples_over_time: List[Dict[str, Any]] = []
    generator = torch.Generator().manual_seed(args.seed)
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    started = time.perf_counter()
    running = 0.0
    seen = 0

    for step in range(1, args.steps + 1):
        model.train()
        model.config.mtp_loss_weight = mtp_weight_at(
            step, args.steps, args.mtp_loss_weight,
            args.mtp_loss_weight_final, args.mtp_weight_warmup_fraction,
        )
        pick = sample_batch_indices(
            generator, train_x.shape[0], args.batch_size,
            repeat_subset_size, args.repeat_subset_prob,
        )
        batch_x = train_x[pick].to(device)
        batch_y = train_y[pick].to(device)
        out = model(batch_x, labels=batch_y)
        optimiser.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        model.step_router_bias()  # one bias update per optimizer step
        running += float(out.lm_loss.detach())
        seen += 1

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val_x, val_y, args.eval_batch_size)
            row = {
                "step": step,
                "train_lm_loss": round(running / max(1, seen), 6),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
                **metrics,
            }
            history.append(row)
            running, seen = 0.0, 0
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            probe = generate_reply(model, tokenizer, PROBE_PROMPTS[0], args.sample_tokens)
            samples_over_time.append({"step": step, **probe})
            print(
                f"step {step:>5}/{args.steps}  train {row['train_lm_loss']:.4f}  "
                f"val {metrics['loss']:.4f}  ppl {metrics['perplexity']:.2f}  "
                f"{row['elapsed_seconds']:.0f}s   reply: {probe['reply'][:60]!r}",
                flush=True,
            )

    if best_state is not None and args.restore_best:
        model.load_state_dict(best_state)

    final = evaluate(model, val_x, val_y, args.eval_batch_size)
    conversations = [
        generate_reply(model, tokenizer, prompt, args.sample_tokens) for prompt in PROBE_PROMPTS
    ]
    # Greedy and speculative must emit identical text; only the cost differs.
    parity = generate_reply(model, tokenizer, PROBE_PROMPTS[1], args.sample_tokens, speculative=False)
    speculative = [c for c in conversations if c["prompt"] == PROBE_PROMPTS[1]][0]

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"{args.run_name}.pt"
    report: Dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "model": "v57_mimomix_talk",
        "architecture": "mimomix_core.MiMoMixModel (v53), trained on text",
        "corpus": {
            **corpus.to_dict(),
            "note": (
                "templated coding-assistant dialogue; 37,543 distinct responses over "
                "120,000 rows, so validation perplexity measures fit to the template "
                "distribution, not generalisation to unseen language"
            ),
        },
        "tokenizer": coverage,
        "config": config.to_dict(),
        "parameters": parameters,
        "hyperparameters": recorded_hyperparameters(args),
        "resolved_flags": resolved_flags(args),
        "device": str(device_info.get("resolved", device)),
        "train_seconds": round(time.perf_counter() - started, 1),
        "history": history,
        "sample_progression": samples_over_time,
        "final_validation": final,
        "uniform_baseline_loss": round(math.log(tokenizer.vocab_size), 6),
        "conversations": conversations,
        "decoding_parity": {
            "prompt": PROBE_PROMPTS[1],
            "greedy_reply": parity["reply"],
            "speculative_reply": speculative["reply"],
            "identical": parity["reply"] == speculative["reply"],
            "greedy_trunk_forwards": parity["trunk_forwards"],
            "speculative_trunk_forwards": speculative["trunk_forwards"],
            "acceptance_length": speculative["acceptance_length"],
        },
        "routing": routing_report(model, val_x, args.eval_batch_size),
        "checkpoint_path": str(checkpoint_path),
    }
    checks = {
        "learned_something": final["loss"] < 0.5 * math.log(tokenizer.vocab_size),
        "produces_non_empty_replies": all(c["reply"].strip() for c in conversations),
        "speculative_matches_greedy": report["decoding_parity"]["identical"],
        "no_starved_experts": report["routing"].get("total_starved", 0) == 0,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())

    save_talk_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        extra={
            "run_name": args.run_name,
            "validation_loss": final["loss"],
            "perplexity": final["perplexity"],
            "created_at": report["created_at"],
        },
    )
    atomic_json(output_dir / "talk_results.json", report)
    return report


def print_summary(report: Dict[str, Any]) -> None:
    final = report["final_validation"]
    print()
    print("== v57 MiMoMix talk ==")
    print(f"  parameters        {report['parameters']['total']:,} "
          f"({report['parameters']['active_per_token']:,} active/token)")
    print(f"  vocabulary        {report['tokenizer']['vocab_size']} types "
          f"(coverage {report['tokenizer']['coverage']:.4f})")
    print(f"  validation loss   {final['loss']:.4f}  perplexity {final['perplexity']:.2f}  "
          f"({final['bits_per_token']:.3f} bits/token)")
    print(f"  uniform baseline  {report['uniform_baseline_loss']:.4f}")
    parity = report["decoding_parity"]
    print(f"  MTP decoding      acceptance {parity['acceptance_length']:.3f}, "
          f"{parity['speculative_trunk_forwards']} vs {parity['greedy_trunk_forwards']} forwards, "
          f"identical to greedy: {parity['identical']}")
    routing = report["routing"]
    if routing.get("moe_layers"):
        print(f"  routing entropy   {routing['mean_normalised_entropy']:.3f}, "
              f"starved {routing['total_starved']}")
    print()
    print("  conversations:")
    for row in report["conversations"]:
        print(f"    user  {row['prompt']}")
        print(f"    model {row['reply'][:150]}")
    print()
    for name, passed in report["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  checkpoint  {report['checkpoint_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MiMoMix to generate text")
    parser.add_argument("--database", default=str(SOURCE_DIR.parent / "databases" / "llm_chat.db"))
    parser.add_argument("--pairs", type=int, default=None, help="limit rows read from the database")
    parser.add_argument("--validation_fraction", type=float, default=0.02)
    parser.add_argument("--max_vocab", type=int, default=16384)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--sample_tokens", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--pct_start", type=float, default=0.1)
    parser.add_argument("--restore_best", action="store_true", default=True)
    parser.add_argument("--no_restore_best", dest="restore_best", action="store_false")

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=None,
                        help="per-head width; None (default) is hidden_size // n_heads")
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6)
    parser.add_argument("--no_tie_word_embeddings", action="store_true")
    parser.add_argument("--sliding_window", type=int, default=64)
    parser.add_argument("--hybrid_ratio", type=int, default=3)
    parser.add_argument("--no_final_layer_global", action="store_true")
    parser.add_argument("--no_attention_sink", action="store_true")
    parser.add_argument("--no_moe", action="store_true")
    parser.add_argument("--n_routed_experts", type=int, default=8)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--moe_intermediate_size", type=int, default=128)
    parser.add_argument("--n_mtp_layers", type=int, default=2)
    parser.add_argument("--mtp_loss_weight", type=float, default=0.3)
    parser.add_argument("--no_thinking_core", action="store_true")
    parser.add_argument("--thinking_cycles", type=int, default=2)
    parser.add_argument("--thinking_max_cycles", type=int, default=4)
    parser.add_argument(
        "--thinking_residual_init",
        type=float,
        default=0.0,
        help=(
            "initial value of the scalar gate the recursive thinking core is "
            "multiplied by. 0.0 (default) reproduces v57/v58 exactly and leaves "
            "the core inert; above zero gives it a gradient path from step 0"
        ),
    )
    # v91: the male-CNS connectome core. Off by default; every flag below is
    # inert unless --cns_core is given.
    parser.add_argument(
        "--cns_core",
        action="store_true",
        help=(
            "graft mimomix_core.ConnectomeCore: a Dale's-law recurrent branch "
            "wired like the Janelia male CNS v1.0 connectome. Needs --cns_graph"
        ),
    )
    parser.add_argument("--cns_graph", default="",
                        help="module graph npz from `malecns_connectome.py modules`")
    parser.add_argument("--cns_wiring", choices=("connectome", "rewired"), default="connectome",
                        help="'rewired' installs the degree-preserving null from the same npz")
    parser.add_argument("--cns_nodes", type=int, default=512)
    parser.add_argument("--cns_steps", type=int, default=6)
    parser.add_argument("--cns_after_layer", type=int, default=2)
    parser.add_argument("--cns_io", choices=("afferent_efferent", "all"), default="afferent_efferent")
    parser.add_argument("--cns_spectral_radius", type=float, default=0.9,
                        help="both wirings are rescaled to this spectral radius at install")
    # v93: capacity slots, multi-site taps, temporal core, expert slots, depth
    # growth (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md D2-D4). All inert by
    # default; each maps to a MiMoMixConfig field whose default is the v91
    # behaviour, except --grow_layers which the generalisation trainer applies
    # through pin_layout_for_growth after build_config.
    growth = parser.add_argument_group("v93 growth capacity")
    growth.add_argument(
        "--cns_spare_nodes", type=int, default=0,
        help=(
            "dead module slots at the end of --cns_nodes (which is then the "
            "capacity); the graph npz must carry cns_nodes - cns_spare_nodes "
            "modules. 0 (default) is v91"
        ),
    )
    growth.add_argument(
        "--cns_read_layers", type=int, nargs="+", default=None,
        help=(
            "trunk blocks the core reads (each through its own norm and "
            "read_in); the core runs after the deepest. Default: the single "
            "--cns_after_layer site"
        ),
    )
    growth.add_argument(
        "--cns_write_layers", type=int, nargs="+", default=None,
        help=(
            "trunk blocks after which gate_j * read_out_j is added, each with "
            "its own zero-initialised gate; every one must be >= the deepest "
            "read layer. Default: the single --cns_after_layer site"
        ),
    )
    growth.add_argument(
        "--cns_to_thinking", action="store_true",
        help="bond the core to the recursive thinking core's input (zero gate at birth)",
    )
    growth.add_argument(
        "--cns_temporal", action="store_true",
        help=(
            "carry the core's rate across positions as a causal scan instead of "
            "restarting it per token; the state rides in past_key_values"
        ),
    )
    growth.add_argument(
        "--moe_spare_experts", type=int, default=0,
        help=(
            "dead expert slots appended to every MoE layer (router logits -inf "
            "until born). 0 (default) is byte-identical to v89"
        ),
    )
    growth.add_argument(
        "--grow_layers", type=int, default=0,
        help=(
            "append this many dense identity blocks at the end of the stack "
            "(n_layers grows by this, global_layers is pinned to the existing "
            "layout); after a warm start their o_proj/down_proj are zeroed so "
            "the grown model's logits equal the source's. 0 (default) is off"
        ),
    )

    # --- v82: the 33 MiMoMixConfig fields no trainer could previously set ---
    #
    # Every default below reproduces the v80 configuration exactly. Where a
    # flag names a research result, that result is a hypothesis this stack has
    # not tested; the citation says whose result it is, not what Supermix
    # measured.
    attention = parser.add_argument_group("attention (v82)")
    attention.add_argument(
        "--qk_norm",
        action="store_true",
        help=(
            "RMSNorm per-head q and k before RoPE. OLMo 2 (arXiv 2501.00656), "
            "Gemma 3 and Qwen3 all ship it; the SmolLM-360M controlled ablation "
            "(arXiv 2512.12167) reports final loss 6.334 without against 2.496 "
            "with at LR 1e-3. UNTESTED HERE -- v80 trained without it"
        ),
    )
    attention.add_argument(
        "--attention_output_gate",
        action="store_true",
        help=(
            "head-wise elementwise sigmoid gate on the SDPA output (Qwen, "
            "arXiv 2505.06708, NeurIPS 2025 oral): 1.7B/400B tokens, average "
            "PPL 7.499 -> 7.404, first-token attention mass 46.7%% -> 4.8%%. "
            "UNTESTED HERE"
        ),
    )
    attention.add_argument(
        "--attention_sink_kinds",
        choices=("all", "swa"),
        default="all",
        help=(
            "which layers carry the learnable sink bias. 'all' (default) is "
            "what every run up to v80 did. 'swa' follows MiMo-V2-Flash, which "
            "sets add_swa_attention_sink_bias true and "
            "add_full_attention_sink_bias false; their 32B W=128 ablation reads "
            "MMLU 54.9 without sink, 58.3 with SWA-only, 57.3 all-global. "
            "UNTESTED HERE"
        ),
    )
    attention.add_argument(
        "--rotary_dim",
        type=int,
        default=None,
        help=(
            "rotate only the first N of head_dim (partial RoPE). None (default) "
            "rotates the whole head, which is what v80 did. MiMo-V2-Flash "
            "rotates 64 of 192, Qwen3-Next 64 of 256, DeepSeek MLA decouples a "
            "rope subspace. Adoption evidence only -- no isolated small-scale "
            "ablation exists, and none has been run here"
        ),
    )
    attention.add_argument(
        "--global_layers",
        default=None,
        help=(
            "comma-separated layer indices to make global, instead of the "
            "uniform --hybrid_ratio interleave (Jet-Nemotron PostNAS, arXiv "
            "2508.15884). None (default) keeps the interleave v80 used"
        ),
    )
    attention.add_argument("--rope_theta", type=float, default=10000.0)
    attention.add_argument("--rope_local_theta", default=10000.0,
                           help="RoPE base for sliding-window layers; 'none' reuses --rope_theta")
    attention.add_argument("--rope_scale_local", action="store_true")
    attention.add_argument(
        "--rope_scaling",
        choices=("none", "ntk", "yarn"),
        default="none",
        help=(
            "context-extension policy. Hardcoded to 'none' before v82, so "
            "'none' is the default and reproduces every published run; "
            "MiMoMixConfig's own default is 'yarn' and was unreachable"
        ),
    )
    attention.add_argument("--yarn_beta_fast", type=float, default=32.0)
    attention.add_argument("--yarn_beta_slow", type=float, default=1.0)
    attention.add_argument(
        "--native_context",
        type=int,
        default=None,
        help="None (default) pins it to --sequence_length, as before v82",
    )
    attention.add_argument(
        "--max_position_embeddings",
        type=int,
        default=None,
        help="None (default) pins it to --sequence_length, as before v82",
    )

    variants = parser.add_argument_group("attention variants (v82)")
    variants.add_argument("--use_mla", action="store_true")
    variants.add_argument("--mla_latent_dim", type=int, default=32)
    variants.add_argument("--mla_pe_dim", type=int, default=16)
    variants.add_argument(
        "--no_mla_global_only",
        action="store_true",
        help=(
            "apply MLA to sliding-window layers too. 'mla_global_only' was read "
            "via getattr in mimomix_core but was not a MiMoMixConfig field, so "
            "it could never be set to anything"
        ),
    )
    variants.add_argument("--use_differential_attention", action="store_true")
    variants.add_argument("--differential_lambda_init", type=float, default=0.8)
    variants.add_argument("--no_differential_output_norm", action="store_true")
    variants.add_argument("--differential_noise_ratio", type=int, default=1)
    variants.add_argument("--use_mod", action="store_true")
    variants.add_argument("--mod_capacity_ratio", type=float, default=0.5)
    variants.add_argument("--mod_predictor_loss_coef", type=float, default=1e-2)
    variants.add_argument(
        "--no_mod_causal_predictor",
        action="store_true",
        help=(
            "keep the non-causal top-k MoD router. The causal predictor exists "
            "because the top-k selection reads the whole sequence: measured, "
            "changing token 7 of 8 moved positions 0-3 by 0.519"
        ),
    )

    routing = parser.add_argument_group("MoE routing (v82)")
    routing.add_argument(
        "--router_score_function",
        choices=("softmax", "sigmoid"),
        default="softmax",
        help=(
            "router affinity. 'softmax' is the default and is what v80 ran -- "
            "not by choice, but because build_config never passed this field. "
            "DeepSeek-V3 (arXiv 2412.19437, Table 5) uses sigmoid affinity with "
            "top-k normalisation. UNTESTED HERE"
        ),
    )
    routing.add_argument("--router_balance_loss_coef", type=float, default=1e-3)
    routing.add_argument("--router_z_loss_coef", type=float, default=1e-3)
    routing.add_argument("--router_bias_update_speed", type=float, default=1e-3)
    routing.add_argument("--router_bias_auto_update", action="store_true")
    routing.add_argument("--no_norm_topk_prob", action="store_true")
    routing.add_argument(
        "--moe_balance_scope",
        choices=("batch", "sequence"),
        default="batch",
        help=(
            "scope of the complementary balance loss. 'batch' (default) is "
            "v80's behaviour; 'sequence' is DeepSeek-V3 4.5.3's sequence-wise "
            "auxiliary loss, which they run at 1e-4. UNTESTED HERE"
        ),
    )
    routing.add_argument("--n_shared_experts", type=int, default=1)
    routing.add_argument("--n_dense_layers", type=int, default=1)

    thinking = parser.add_argument_group("thinking core (v82)")
    thinking.add_argument("--thinking_inner_steps", type=int, default=2)
    thinking.add_argument("--thinking_latent_dim", type=int, default=64)
    thinking.add_argument("--ponder_loss_weight", type=float, default=1e-2)
    thinking.add_argument("--consistency_loss_weight", type=float, default=1e-2)

    schedule = parser.add_argument_group("training schedule (v82)")
    schedule.add_argument(
        "--mtp_loss_weight_final",
        type=float,
        default=None,
        help=(
            "decay the MTP loss weight to this value across the run. None "
            "(default) holds --mtp_loss_weight constant, which is v80. "
            "DeepSeek-V3 decays 0.3 -> 0.1. UNTESTED HERE"
        ),
    )
    schedule.add_argument(
        "--mtp_weight_warmup_fraction",
        type=float,
        default=0.0,
        help=(
            "ramp the MTP weight in linearly over this fraction of the run. "
            "0.0 (default) is v80: full weight from step 1"
        ),
    )
    schedule.add_argument(
        "--decay_mode",
        choices=("all", "no_norm_bias"),
        default="all",
        help=(
            "which parameters AdamW decays. 'all' (default) reproduces v80, "
            "which applied weight_decay 0.01 to every parameter including "
            "RMSNorm gains, biases, the router's expert_bias, the thinking "
            "core's residual_scale and log_temperature. 'no_norm_bias' puts "
            "1-D tensors and calibration scalars in a zero-decay group"
        ),
    )
    schedule.add_argument(
        "--repeat_subset_fraction",
        type=float,
        default=0.0,
        help=(
            "size of the fixed repeated subset, as a fraction of the training "
            "rows (Charton & Kempe 2024 two-set training). 0.0 (default) is "
            "v80's uniform with-replacement sampling"
        ),
    )
    schedule.add_argument(
        "--repeat_subset_prob",
        type=float,
        default=0.0,
        help="probability that a given batch slot is drawn from the repeated subset",
    )
    schedule.add_argument(
        "--reverse_digits",
        action="store_true",
        help=(
            "tokenize numbers least-significant-digit first. Lee et al. 2023 "
            "(arXiv 2307.03381) report a 10.6M NanoGPT reaching 100%% on "
            "3-digit addition at ~2.5k samples with reversal and never without. "
            "UNTESTED HERE. Requires --digit_tokens to mean anything"
        ),
    )
    schedule.add_argument(
        "--probe_max_new_tokens",
        type=int,
        default=DEFAULT_PROBE_MAX_NEW_TOKENS,
        help=(
            "generation cap for the mid-run accuracy probe. Was hardcoded to "
            "64, which is below the median reply length of seven of the "
            "twenty-two tasks in the v80 corpus (momentum 83, arithmetic_series "
            "93, work 86, wave_speed 84, electrical_power 76, force 78, "
            "kinetic_energy 76) and below 100%% of arithmetic_series replies, "
            f"so those tasks read 0.00 whatever the model learned. Default "
            f"{DEFAULT_PROBE_MAX_NEW_TOKENS} = the measured 99-token maximum "
            "plus headroom"
        ),
    )
    schedule.add_argument(
        "--strict",
        action="store_true",
        help=(
            "refuse to start when a task's median reply is longer than "
            "--probe_max_new_tokens, instead of warning"
        ),
    )

    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_preference", default="cuda,npu,xpu,dml,mps,cpu")
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--run_name", default="v57_talk")
    parser.add_argument("--output_dir", default=str(SOURCE_DIR.parent / "output" / "v57_talk"))
    parser.add_argument("--enforce_gates", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print_summary(report)
    if args.enforce_gates and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
