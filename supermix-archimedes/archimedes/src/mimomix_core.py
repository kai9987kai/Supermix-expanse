"""MiMoMix v53 neural core: hybrid attention + sparse MoE + MTP + verified recursion.

This module is the model half of the Supermix v53 "MiMoMix" line. It fuses three
lineages into one compact, CPU-runnable PyTorch model:

* **Xiaomi MiMo (V2-Flash / V2.5-Pro) structural techniques.**
  - Hybrid attention: local sliding-window attention (SWA) interleaved with full
    global attention (GA) at a configurable ratio (MiMo-V2-Flash uses 5:1 with a
    128-token window; MiMo-V2.5-Pro uses 6:1). Only the GA layers keep an
    unbounded KV cache, which is where the ~7x KV-cache reduction comes from.
  - A learnable per-head **attention sink** bias, so a head can dump probability
    mass into a null slot instead of being forced to attend somewhere. This is
    the "off-by-one"/softmax-with-sink formulation used by StreamingLLM-style
    stabilisation and by recent open-weight models.
  - **Auxiliary-loss-free load balancing** for the sparse MoE: a per-expert
    routing bias is nudged by a sign rule so selection stays balanced without
    a gradient-carrying balance loss distorting the language objective, plus a
    router z-loss to stop router logits drifting large.
  - **Multi-Token Prediction (MTP)** modules, trained as extra causal depths and
    reusable at inference as the draft model for self-speculative decoding.
  - RoPE with a **progressive context-extension** schedule (none / NTK-aware /
    YaRN), matching the 32K-native to long-context extension pattern.

* **Supermix v51/v52 cognition.** The recurrent latent thinking core keeps
  weight-tied refinement, ACT-style halting with a ponder cost, deep supervision
  over per-cycle decodes, trainable temperature calibration, and the supervised
  quality/continue verifier that gates escalation. See
  ``source/model_variants.py::CognitiveLeapV52ExpertHead`` for the classifier-side
  ancestor whose contract this core deliberately mirrors.

* **AI-Dem-Lab instrumentation.** Every forward pass publishes a JSON-safe
  telemetry dict (router occupancy, sink mass, halting depth, calibrated
  entropy, MTP depth losses). ``mimomix_observatory.py`` turns that stream into
  novelty/stability/resonance measurements and a feedback signal.

Scope honesty: this is a small, self-contained research model. It implements the
*mechanisms* named above and tests them; it is not a reproduction of any
frontier checkpoint, and none of the numbers published by those model families
transfer to it.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "MiMoMixConfig",
    "MiMoMixOutput",
    "RMSNorm",
    "RotaryEmbedding",
    "HybridAttention",
    "DifferentialHybridAttention",
    "MultiLatentAttention",
    "MixtureOfDepthsRouter",
    "DenseFeedForward",
    "SparseMoEFeedForward",
    "MiMoMixBlock",
    "MultiTokenPredictionModule",
    "RecursiveThinkingCore",
    "MultimodalProjectionHead",
    "MiMoMixModel",
    "attention_layout",
    "pin_layout_for_growth",
    "build_mimomix",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MiMoMixConfig:
    """Every structural knob for a MiMoMix model.

    Defaults are deliberately tiny so the whole stack trains and tests on CPU in
    seconds. Ratios and window semantics follow the MiMo hybrid-attention
    description; the sizes do not.
    """

    vocab_size: int = 512
    hidden_size: int = 128
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: Optional[int] = None
    intermediate_size: int = 256
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    # --- hybrid attention -------------------------------------------------
    sliding_window: int = 128
    #: number of consecutive SWA layers per one global layer (MiMo uses 5 or 6)
    hybrid_ratio: int = 5
    #: force the final layer to be global so the last hidden state can see
    #: the whole prefix even when the ratio would have made it local
    final_layer_global: bool = True
    attention_sink: bool = True

    # --- rope / context extension ----------------------------------------
    #: RoPE base for the *global* layers, which are the ones that must reach
    #: the full context.
    rope_theta: float = 10000.0
    #: RoPE base for the *local* (sliding-window) layers. Gemma 3 and MiMo
    #: decouple these: a layer that can only ever see `sliding_window` tokens
    #: has no long-range dependency to encode, so a large base or a context
    #: extension applied to it is at best wasted and at worst harmful. ``None``
    #: reuses ``rope_theta``, which is the single-table behaviour.
    rope_local_theta: Optional[float] = 10000.0
    #: Apply the context-extension policy to local layers too. Off by default,
    #: for the reason above.
    rope_scale_local: bool = False
    native_context: int = 256
    max_position_embeddings: int = 1024
    #: "none" | "ntk" | "yarn"
    rope_scaling: str = "yarn"
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    #: Rotate only the leading ``rotary_dim`` components of each head and pass
    #: the rest through unchanged ("partial RoPE"). ``None`` rotates the whole
    #: head, which is what every checkpoint written before v82 did.
    #:
    #: Adoption evidence only, no isolated small-scale ablation: MiMo-V2-Flash
    #: rotates 64 of 192, Qwen3-Next 64 of 256, and DeepSeek's MLA carries a
    #: decoupled rope sub-vector for the same reason. Whether it helps *this*
    #: model is untested -- treat it as a hypothesis for a future run.
    rotary_dim: Optional[int] = None

    # --- attention refinements (all default-off, all untested here) --------
    #: RMSNorm on the per-head q and k *before* the rotary is applied, the
    #: OLMo 2 (arXiv 2501.00656) / Gemma 3 / Qwen3 placement. The SmolLM-360M
    #: controlled ablation (arXiv 2512.12167) reports final loss 6.334 without
    #: vs 2.496 with at LR 1e-3. No Supermix run has measured it.
    qk_norm: bool = False
    #: Head-wise elementwise sigmoid gate on the SDPA output before ``o_proj``
    #: (Qwen, NeurIPS 2025 oral, arXiv 2505.06708; 1.7B/400B tokens: avg PPL
    #: 7.499 -> 7.404, first-token attention mass 46.7% -> 4.8%). The gate bias
    #: starts at +4 with a zero weight, so a freshly enabled gate is a constant
    #: 0.982 scaling -- near-identity, so switching it on does not destroy a
    #: warm start. Unmeasured here.
    attention_output_gate: bool = False
    #: Which layer kinds get a learnable attention sink. ``"all"`` is today's
    #: behaviour. ``"swa"`` matches MiMo-V2-Flash's released config
    #: (add_swa_attention_sink_bias true, add_full_attention_sink_bias false).
    attention_sink_kinds: str = "all"
    #: Explicit tuple of layer indices that are global, overriding
    #: ``hybrid_ratio`` in :func:`attention_layout` (Jet-Nemotron PostNAS,
    #: arXiv 2508.15884, places global layers by search rather than uniformly).
    #: ``None`` keeps the uniform interleave exactly as before.
    global_layers: Optional[Tuple[int, ...]] = None

    # --- sparse MoE -------------------------------------------------------
    use_moe: bool = True
    #: dense layers at the bottom of the stack (MoE models commonly keep the
    #: first block(s) dense for stability)
    n_dense_layers: int = 1
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_top_k: int = 2
    moe_intermediate_size: int = 64
    #: aux-loss-free load balancing update speed (gamma); 0 disables the rule.
    #: Keep it small. The sign rule is a control loop, and an oversized step
    #: overshoots the balance point every update and oscillates instead of
    #: converging -- which reads as *worse* balance, not faster balance.
    router_bias_update_speed: float = 1e-3
    #: Fire the bias update inside forward() instead of on an explicit
    #: MiMoMixModel.step_router_bias() call. Off by default: firing per-forward
    #: makes the effective gamma depend on gradient-accumulation depth and on
    #: how many times the controller probes a request. See
    #: SparseMoEFeedForward.update_router_bias.
    router_bias_auto_update: bool = False
    router_z_loss_coef: float = 1e-3
    #: small complementary sequence-level balance loss; the bias rule does the
    #: heavy lifting, this only discourages within-sequence collapse
    router_balance_loss_coef: float = 1e-3
    router_score_function: str = "softmax"  # "softmax" | "sigmoid"
    norm_topk_prob: bool = True
    #: Scope of the complementary balance loss. ``"batch"`` flattens ``(B, T)``
    #: and is what every run up to and including v80 computed, despite the
    #: docstring calling it sequence-level. ``"sequence"`` computes the load and
    #: mean routing probability per sequence and averages over the batch, which
    #: is the DeepSeek-V3 4.5.3 formulation. Default stays ``"batch"`` so v80
    #: reproduces bit-for-bit.
    moe_balance_scope: str = "batch"

    # --- multi-token prediction ------------------------------------------
    n_mtp_layers: int = 2
    mtp_loss_weight: float = 0.3

    # --- recursive thinking core -----------------------------------------
    use_thinking_core: bool = True
    thinking_latent_dim: int = 64
    thinking_cycles: int = 3
    thinking_max_cycles: int = 8
    thinking_inner_steps: int = 2
    #: Initial value of ``RecursiveThinkingCore.residual_scale``, the single
    #: scalar the whole recursive core is multiplied by.
    #:
    #: The default 0.0 reproduces every checkpoint trained before v59 exactly.
    #: It is also why the core is inert in those checkpoints: the gate multiplies
    #: the core's own gradient, so starting at zero leaves the mechanism with
    #: almost no path to learn along, and after 1,000 steps v58's gate had
    #: reached only 6.41e-04 -- small enough that closing it entirely changes
    #: zero of 12,192 held-out predictions (see docs/V59_MECHANISM_CAUSALITY.md).
    #: Set this above zero to give the core a gradient path from step 0.
    thinking_residual_init: float = 0.0
    ponder_loss_weight: float = 1e-2
    consistency_loss_weight: float = 1e-2

    # --- differential attention (Microsoft ICLR 2025) ---------------------
    use_differential_attention: bool = False
    differential_lambda_init: float = 0.8
    #: Apply the reference DIFF-Transformer per-head sublayer normalisation and
    #: the ``(1 - lambda_init)`` rescale that this implementation omitted before
    #: v82. Defaulting this to ``True`` is safe *because no trained checkpoint
    #: in ``output/`` sets ``use_differential_attention``* -- verified by
    #: loading every ``output/**/*.pt`` and inspecting its saved config; the
    #: flag is absent or false in all of them. Unmeasured on this model.
    differential_output_norm: bool = True
    #: Signal:noise head allocation (GDA-style asymmetry). ``1`` is today's
    #: symmetric behaviour: every head gets its own noise map. ``R > 1`` groups
    #: ``R`` signal heads onto one shared noise map by averaging their noise
    #: queries. This is an *approximation* of the published parameterisation
    #: (which uses narrower dedicated noise projections); it was chosen so that
    #: ``R == 1`` reproduces the current weights and KV-cache layout exactly.
    #: Its benefit is unmeasured.
    differential_noise_ratio: int = 1

    # --- mixture-of-depths (Google DeepMind 2024) ------------------------
    use_mod: bool = False
    mod_capacity_ratio: float = 0.5
    #: Train a per-token causal predictor of top-k membership (arXiv 2404.02258
    #: sec. 3.5) and use it for selection whenever the block cannot see the
    #: whole sequence -- which is every cached decode step. Without it the
    #: per-call ``ceil(seq_len * ratio)`` capacity selects *every* token at
    #: ``seq_len == 1``, so a cached decode takes a different path through the
    #: block than the full forward did. Setting this to ``False`` restores that
    #: (broken) pre-v82 behaviour.
    mod_causal_predictor: bool = True
    #: Weight of the predictor's BCE loss in the model's aux-loss total.
    mod_predictor_loss_coef: float = 1e-2

    # --- multi-latent attention (DeepSeek-V3 MLA) -------------------------
    use_mla: bool = False
    mla_latent_dim: int = 32
    mla_pe_dim: int = 16
    #: Use MLA only on the global layers, leaving SWA layers on plain grouped
    #: query attention. Before v82 this was read through ``getattr`` with a
    #: default of ``True`` but was not a field, so it could never be set; it is
    #: now a real field with that same default.
    mla_global_only: bool = True

    # --- multimodal projection (Xiaomi MiMo-V2.5) -------------------------
    use_multimodal: bool = False
    multimodal_input_dim: int = 128

    # --- v91 connectome core (Janelia male CNS v1.0) ---------------------
    #: Graft a recurrent side branch whose wiring, Dale signs and initial
    #: weights come from the fly male CNS connectome. See ConnectomeCore.
    use_cns_core: bool = False
    #: Number of connectome modules (cell-type clusters) in the core.
    cns_nodes: int = 512
    #: Recurrent update steps per token.
    cns_steps: int = 6
    #: The core's output joins the residual stream after this block index.
    cns_after_layer: int = 2
    #: "afferent_efferent": text enters through sensory/visual/ascending
    #: modules and is read out of descending/motor ones, so information must
    #: cross the wiring. "all": every module is read in and read out.
    cns_io: str = "afferent_efferent"
    #: Provenance only -- the graph itself is carried in the state_dict
    #: buffers, so a checkpoint loads without the npz.
    cns_graph: str = ""
    #: "connectome" or "rewired" (degree-preserving null). Provenance only.
    cns_wiring: str = "connectome"

    # --- v93 neurogenesis: capacity slots, multi-site taps, temporal core --
    # Every field below defaults to the v91 behaviour, so a v91 checkpoint's
    # config (which lacks them) rebuilds the identical module and loads
    # strictly. See docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md, D2-D4.
    #: Dead module slots allocated at the END of the ``cns_nodes`` capacity.
    #: ``cns_nodes`` is the capacity: ``load_graph`` requires the npz to carry
    #: exactly ``cns_nodes - cns_spare_nodes`` modules and leaves the spare
    #: slots inert until ``split_module`` fills one.
    cns_spare_nodes: int = 0
    #: Trunk blocks whose output the core reads (each through its own norm and
    #: ``read_in``); the core runs once after the deepest. ``()`` means
    #: ``(cns_after_layer,)`` -- the single v91 site. Lists are accepted and
    #: normalised to a sorted tuple.
    cns_read_layers: Tuple[int, ...] = ()
    #: Trunk blocks after which ``gate_j * read_out_j(read)`` is added; every
    #: one must be >= the deepest read layer. ``()`` means ``(cns_after_layer,)``.
    cns_write_layers: Tuple[int, ...] = ()
    #: Add ``thinking_gate * to_thinking(read)`` to the recursive thinking
    #: core's latent input (zero gate at birth, so exact until trained).
    cns_to_thinking: bool = False
    #: Carry the rate vector across positions as a causal scan (``cns_steps``
    #: inner updates per position) instead of restarting it at every token.
    #: Cached decoding then carries the per-position state history as one
    #: extra ``past_key_values`` entry after the layer entries.
    cns_temporal: bool = False

    # --- v93 MoE expert slots ---------------------------------------------
    #: Dead expert slots per MoE layer, appended after ``n_routed_experts``.
    #: Their router logits are set to -inf before the softmax, so with 0 the
    #: module is byte-identical to v89's. ``birth_expert`` fills one.
    moe_spare_experts: int = 0

    # --- v93 depth growth ---------------------------------------------------
    #: Number of blocks appended at the END of the stack as identity blocks.
    #: When > 0, ``n_layers`` already INCLUDES them, the new blocks are dense
    #: (never MoE / MoD), and ``global_layers`` must be pinned so the existing
    #: layout does not re-derive under ``hybrid_ratio``. The trainer obtains
    #: such a config with :func:`pin_layout_for_growth` and, after the warm
    #: start, calls :meth:`MiMoMixModel.zero_new_blocks`.
    grow_layers: int = 0

    #: Config keys seen by :meth:`from_dict` that this build does not know.
    #: Deliberately *not* annotated, so it is a plain class attribute and not a
    #: dataclass field -- ``to_dict()`` must stay a pure round-trip.
    unknown_keys = ()

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.n_heads != 0:
                raise ValueError("hidden_size must divide n_heads when head_dim is unset")
            self.head_dim = self.hidden_size // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.use_differential_attention and self.head_dim % 2 != 0:
            raise ValueError("head_dim must be divisible by 2 for differential attention")
        if self.use_mod and not (0.0 < self.mod_capacity_ratio <= 1.0):
            raise ValueError("mod_capacity_ratio must be in (0, 1]")
        if self.use_mla and self.mla_latent_dim <= 0:
            raise ValueError("mla_latent_dim must be positive")
        if self.use_mla and self.use_differential_attention:
            # MiMoMixBlock has always picked differential first and silently
            # dropped MLA. Say so instead of pretending both are active.
            raise ValueError(
                "use_differential_attention and use_mla are mutually exclusive: "
                "MiMoMixBlock builds DifferentialHybridAttention first, so MLA "
                "would be silently ignored. Pick one."
            )
        if self.use_mla:
            if self.mla_pe_dim <= 0 or self.mla_pe_dim % 2 != 0:
                raise ValueError("mla_pe_dim must be a positive even number")
            if self.mla_pe_dim > int(self.head_dim):
                raise ValueError(
                    f"mla_pe_dim ({self.mla_pe_dim}) must be <= head_dim ({self.head_dim}); "
                    "the decoupled rope sub-vector cannot be wider than the head"
                )
        if self.rotary_dim is not None:
            rot = int(self.rotary_dim)
            if rot <= 0 or rot % 2 != 0:
                raise ValueError("rotary_dim must be a positive even number")
            if rot > int(self.head_dim):
                raise ValueError(
                    f"rotary_dim ({rot}) must be <= head_dim ({self.head_dim})"
                )
            if self.rope_scaling == "ntk" and rot <= 2:
                raise ValueError("rotary_dim must be > 2 when rope_scaling='ntk'")
            self.rotary_dim = rot
        if self.attention_sink_kinds not in {"all", "swa"}:
            raise ValueError(
                f"unknown attention_sink_kinds: {self.attention_sink_kinds!r} (expected 'all' or 'swa')"
            )
        if self.moe_balance_scope not in {"batch", "sequence"}:
            raise ValueError(
                f"unknown moe_balance_scope: {self.moe_balance_scope!r} (expected 'batch' or 'sequence')"
            )
        if int(self.differential_noise_ratio) < 1:
            raise ValueError("differential_noise_ratio must be >= 1")
        if self.use_differential_attention and int(self.differential_noise_ratio) > 1:
            ratio = int(self.differential_noise_ratio)
            if self.n_heads % ratio != 0:
                raise ValueError(
                    f"differential_noise_ratio ({ratio}) must divide n_heads ({self.n_heads})"
                )
        if self.global_layers is not None:
            indices = tuple(sorted({int(i) for i in self.global_layers}))
            for index in indices:
                if not (0 <= index < int(self.n_layers)):
                    raise ValueError(
                        f"global_layers index {index} out of range for n_layers={self.n_layers}"
                    )
            self.global_layers = indices
        if self.rope_scaling not in {"none", "ntk", "yarn"}:
            raise ValueError(f"unknown rope_scaling: {self.rope_scaling!r}")
        if self.router_score_function not in {"softmax", "sigmoid"}:
            raise ValueError(f"unknown router_score_function: {self.router_score_function!r}")
        if self.use_moe and self.moe_top_k > self.n_routed_experts:
            raise ValueError("moe_top_k cannot exceed n_routed_experts")
        if self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")
        if self.hybrid_ratio < 0:
            raise ValueError("hybrid_ratio must be >= 0")
        # v93 growth fields. Normalised and validated whether or not the core
        # is on, so to_dict/from_dict round-trip the same shapes either way.
        self.cns_read_layers = self._layer_tuple("cns_read_layers", self.cns_read_layers)
        self.cns_write_layers = self._layer_tuple("cns_write_layers", self.cns_write_layers)
        if int(self.cns_spare_nodes) < 0:
            raise ValueError("cns_spare_nodes must be >= 0")
        if int(self.moe_spare_experts) < 0:
            raise ValueError("moe_spare_experts must be >= 0")
        if not (0 <= int(self.grow_layers) < int(self.n_layers)):
            raise ValueError(
                f"grow_layers {self.grow_layers} must be in [0, n_layers={self.n_layers})"
            )
        if int(self.grow_layers) > 0 and self.global_layers is None:
            raise ValueError(
                "grow_layers > 0 requires global_layers to be pinned to the pre-growth "
                "layout (use pin_layout_for_growth); otherwise hybrid_ratio re-derives "
                "the kinds of the existing blocks"
            )
        if self.use_cns_core:
            if self.cns_nodes < 2 or self.cns_steps < 1:
                raise ValueError("cns_nodes must be >= 2 and cns_steps >= 1")
            if int(self.cns_spare_nodes) >= int(self.cns_nodes) - 1:
                raise ValueError(
                    f"cns_spare_nodes {self.cns_spare_nodes} leaves fewer than 2 live modules "
                    f"of cns_nodes={self.cns_nodes}"
                )
            if not (0 <= int(self.cns_after_layer) < int(self.n_layers)):
                raise ValueError(
                    f"cns_after_layer {self.cns_after_layer} out of range for n_layers={self.n_layers}"
                )
            if self.cns_io not in {"afferent_efferent", "all"}:
                raise ValueError(f"unknown cns_io: {self.cns_io!r}")
            if self.cns_wiring not in {"connectome", "rewired"}:
                raise ValueError(f"unknown cns_wiring: {self.cns_wiring!r}")
            reads, writes = self.cns_read_sites, self.cns_write_sites
            if min(writes) < max(reads):
                raise ValueError(
                    f"every cns write layer {writes} must be >= the deepest read layer "
                    f"{max(reads)}: the core runs once, after that block"
                )
            if self.cns_to_thinking and not self.use_thinking_core:
                raise ValueError("cns_to_thinking requires use_thinking_core")

    def _layer_tuple(self, name: str, value: object) -> Tuple[int, ...]:
        """Normalise a layer-index list/tuple to a sorted, de-duplicated tuple."""

        if value is None:
            return ()
        if isinstance(value, (int,)):
            value = (value,)
        indices = tuple(sorted({int(i) for i in value}))  # type: ignore[union-attr]
        for index in indices:
            if not (0 <= index < int(self.n_layers)):
                raise ValueError(f"{name} index {index} out of range for n_layers={self.n_layers}")
        return indices

    @property
    def cns_read_sites(self) -> Tuple[int, ...]:
        """Effective read layers: ``cns_read_layers`` or ``(cns_after_layer,)``."""

        return tuple(self.cns_read_layers) or (int(self.cns_after_layer),)

    @property
    def cns_write_sites(self) -> Tuple[int, ...]:
        """Effective write layers: ``cns_write_layers`` or ``(cns_after_layer,)``."""

        return tuple(self.cns_write_layers) or (int(self.cns_after_layer),)

    @property
    def cns_run_after_layer(self) -> int:
        """The block after which the core runs: the deepest read site."""

        return int(self.cns_read_sites[-1])

    @property
    def context_scale(self) -> float:
        """Extension factor from the natively trained context to the target."""

        return max(1.0, float(self.max_position_embeddings) / float(self.native_context))

    #: Tuple-valued v93 fields, written by :meth:`to_dict` as JSON-safe lists.
    #: ``global_layers`` is deliberately not in this set: it has been written
    #: as a tuple (or ``None``) since v82 and checkpoint consumers compare
    #: ``to_dict()`` against the stored payload byte-for-byte.
    _LIST_FIELDS = ("cns_read_layers", "cns_write_layers")

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for name in self._LIST_FIELDS:
            payload[name] = [int(i) for i in payload[name]]
        return payload

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in dataclass_fields(cls))

    @classmethod
    def from_dict(
        cls, payload: Dict[str, object], warn: bool = True
    ) -> "MiMoMixConfig":
        """Build a config from a saved payload, *reporting* unknown keys.

        ``MiMoMixConfig(**payload["config"])`` raises ``TypeError`` the moment a
        checkpoint carries a field this build does not know about, which makes
        every checkpoint written by a newer trainer unloadable by older-but-
        otherwise-compatible code. This accepts them instead: unknown keys are
        dropped, recorded on the returned object as ``unknown_keys`` (a plain
        attribute, not a dataclass field, so ``to_dict`` is unaffected), and --
        unless ``warn=False`` -- emitted once as a ``UserWarning``.

        Dropping a key is not free: if the unknown key was a *structural* knob
        the resulting model will have a different shape and the ``state_dict``
        load will fail loudly. That is the intended failure mode -- loud, not
        silent.
        """

        known = set(cls.field_names())
        accepted = {k: v for k, v in payload.items() if k in known}
        unknown = tuple(sorted(k for k in payload if k not in known))
        if "global_layers" in accepted and accepted["global_layers"] is not None:
            accepted["global_layers"] = tuple(int(i) for i in accepted["global_layers"])  # type: ignore[arg-type]
        config = cls(**accepted)  # type: ignore[arg-type]
        config.unknown_keys = unknown
        if unknown and warn:
            warnings.warn(
                "MiMoMixConfig.from_dict ignored unknown config keys: "
                + ", ".join(unknown),
                UserWarning,
                stacklevel=2,
            )
        return config


def attention_layout(
    n_layers: int,
    hybrid_ratio: int,
    final_layer_global: bool = True,
    global_layers: Optional[Sequence[int]] = None,
) -> Tuple[str, ...]:
    """Return ``("swa", "swa", ..., "global")`` for each layer index.

    With ``hybrid_ratio == r`` every ``r``-th layer is global, giving the
    ``r:1`` local:global interleave MiMo describes. ``hybrid_ratio == 0``
    means "all global" (a plain dense-attention model).

    ``global_layers`` overrides the ratio entirely with an explicit set of
    global layer indices -- the placement-by-search idea from Jet-Nemotron
    PostNAS (arXiv 2508.15884), where *which* layers are global mattered as
    much as how many. ``final_layer_global`` is still honoured on top of it.
    Passing ``None`` (the default) leaves the ratio behaviour untouched.
    """

    if global_layers is not None:
        chosen = {int(i) for i in global_layers}
        layout = ["global" if i in chosen else "swa" for i in range(n_layers)]
        if final_layer_global and layout:
            layout[-1] = "global"
        return tuple(layout)
    if hybrid_ratio <= 0:
        return tuple("global" for _ in range(n_layers))
    layout: List[str] = []
    for index in range(n_layers):
        is_global = ((index + 1) % (hybrid_ratio + 1)) == 0
        layout.append("global" if is_global else "swa")
    if final_layer_global and layout:
        layout[-1] = "global"
    return tuple(layout)


def pin_layout_for_growth(config: MiMoMixConfig, k: int) -> MiMoMixConfig:
    """Config for the same model with ``k`` identity blocks appended (v93 D4).

    Returns a new config with ``n_layers += k``, ``grow_layers += k`` and
    ``global_layers`` pinned to the *current* layout's global indices, so the
    existing blocks keep their kinds (with ``n_layers=5, hybrid_ratio=3`` the
    v89 layout is ``[swa, swa, swa, global, global]``; re-deriving at 6 would
    turn block 4 into ``swa`` while its weights still load, which is a silent
    behaviour change). The appended blocks are dense (``MiMoMixBlock`` reads
    ``grow_layers`` and forces ``force_dense`` on the last ``grow_layers``
    indices); the last one is global whenever ``final_layer_global`` is set,
    the others ``swa``.

    The trainer calls this on the warm-start config, builds the model, loads
    the source checkpoint non-strictly (the new ``layers.{n}.*`` keys are the
    only missing ones) and then calls :meth:`MiMoMixModel.zero_new_blocks`
    so the grown model's logits equal the source's exactly.
    """

    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return MiMoMixConfig(**config.to_dict())
    current = attention_layout(
        config.n_layers, config.hybrid_ratio, config.final_layer_global,
        getattr(config, "global_layers", None),
    )
    pinned = tuple(i for i, kind in enumerate(current) if kind == "global")
    payload = config.to_dict()
    payload["n_layers"] = int(config.n_layers) + k
    payload["global_layers"] = pinned
    payload["grow_layers"] = int(getattr(config, "grow_layers", 0)) + k
    grown = MiMoMixConfig(**payload)
    grown.unknown_keys = tuple(getattr(config, "unknown_keys", ()))
    return grown


# ---------------------------------------------------------------------------
# Normalisation and rotary embeddings
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


def _yarn_correction_dim(num_rotations: float, dim: int, base: float, native_context: int) -> float:
    return (dim * math.log(native_context / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(
    beta_fast: float, beta_slow: float, dim: int, base: float, native_context: int
) -> Tuple[int, int]:
    low = math.floor(_yarn_correction_dim(beta_fast, dim, base, native_context))
    high = math.ceil(_yarn_correction_dim(beta_slow, dim, base, native_context))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(low: float, high: float, dim: int) -> torch.Tensor:
    if abs(high - low) < 1e-3:
        high = low + 1e-3
    ramp = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return ramp.clamp(0.0, 1.0)


class RotaryEmbedding(nn.Module):
    """RoPE with an explicit progressive context-extension policy.

    ``rope_scaling``:

    ``none``
        Plain RoPE. Positions beyond the trained context extrapolate badly.
    ``ntk``
        NTK-aware base rescaling: ``theta' = theta * s ** (d / (d - 2))``. One
        cheap knob, no retraining required, mild degradation on short context.
    ``yarn``
        Per-frequency-band interpolation: high-frequency dimensions (short
        wavelength relative to the native context) are left alone, low-frequency
        dimensions are fully interpolated by ``s``, and a linear ramp blends the
        band in between. YaRN additionally scales attention logits by
        ``0.1 * ln(s) + 1`` to compensate for the entropy change; that factor is
        exposed as :attr:`attention_scaling` and applied by the attention layer.

    **Partial RoPE.** The table is built over ``rotary_dim`` components, not
    necessarily the whole head. ``rotary_dim=None`` falls back to
    ``config.rotary_dim`` and then to ``head_dim``, which is the full rotation
    every pre-v82 checkpoint used. :func:`apply_rotary` rotates exactly the
    leading ``cos.shape[-1]`` components of ``x`` and passes the tail through,
    so a table narrower than the head is a *correct partial rotation* rather
    than a slice of a wider table. Slicing a wider table is not a rotation at
    all: ``cat([freqs, freqs])`` pairs component ``i`` with component
    ``i + head_dim/2``, so truncating to ``d`` re-pairs ``i`` with ``i + d/2``,
    which carries a different frequency. That was the pre-v82 MLA bug.
    """

    def __init__(
        self,
        config: MiMoMixConfig,
        kind: str = "global",
        rotary_dim: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.kind = kind
        self.head_dim = int(config.head_dim)
        requested = rotary_dim if rotary_dim is not None else getattr(config, "rotary_dim", None)
        dim = self.head_dim if requested is None else int(requested)
        if dim <= 0 or dim % 2 != 0 or dim > self.head_dim:
            raise ValueError(
                f"rotary_dim must be a positive even number <= head_dim ({self.head_dim}), got {dim}"
            )
        self.rotary_dim = dim
        if kind == "swa" and config.rope_local_theta is not None:
            base = float(config.rope_local_theta)
        else:
            base = float(config.rope_theta)
        scale = config.context_scale
        if kind == "swa" and not config.rope_scale_local:
            scale = 1.0
        attention_scaling = 1.0

        if config.rope_scaling == "ntk" and scale > 1.0:
            base = base * (scale ** (dim / (dim - 2)))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        elif config.rope_scaling == "yarn" and scale > 1.0:
            pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (scale * pos_freqs)
            low, high = _yarn_correction_range(
                config.yarn_beta_fast, config.yarn_beta_slow, dim, base, config.native_context
            )
            # 1 => keep the untouched extrapolation frequency, 0 => interpolate
            extrapolation_factor = 1.0 - _yarn_linear_ramp(low, high, dim // 2)
            inv_freq = (
                inv_freq_interpolation * (1.0 - extrapolation_factor)
                + inv_freq_extrapolation * extrapolation_factor
            )
            attention_scaling = 0.1 * math.log(scale) + 1.0
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = float(attention_scaling)
        self.effective_base = float(base)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``positions`` is ``(T,)`` -> ``(cos, sin)`` each ``(T, rotary_dim)``."""

        freqs = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` is ``(B, H, T, D)``; ``cos``/``sin`` are ``(T, R)`` with ``R <= D``.

    The leading ``R`` components of ``x`` are rotated as a proper 2-D rotation
    (component ``i`` paired with ``i + R/2``, both carrying frequency ``i``);
    the remaining ``D - R`` components are passed through untouched. When
    ``R == D`` this is exactly the full rotation, so every pre-v82 call site is
    unchanged bit-for-bit.

    Passing a ``cos`` that was *sliced* from a wider table is still wrong and
    always was -- the pairing no longer matches the frequency. Build a table of
    the width you intend to rotate.
    """

    rot = cos.shape[-1]
    width = x.shape[-1]
    if rot > width:
        raise ValueError(f"rotary table width {rot} exceeds the head width {width}")
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
    if rot == width:
        return x * cos + _rotate_half(x) * sin
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    rotated = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat([rotated, x_pass], dim=-1)


# ---------------------------------------------------------------------------
# Hybrid attention
# ---------------------------------------------------------------------------


def _sink_enabled(config: MiMoMixConfig, kind: str) -> bool:
    """Whether this layer kind carries a learnable attention-sink logit.

    ``attention_sink_kinds="all"`` (the default, and every pre-v82 checkpoint)
    gives every layer a sink. ``"swa"`` gives it to the local layers only,
    which is MiMo-V2-Flash's released configuration
    (``add_swa_attention_sink_bias`` true, ``add_full_attention_sink_bias``
    false). Their 32B W=128 ablation reports MMLU 54.9 without a sink, 58.3
    with the SWA-only sink and 57.3 all-global; none of that transfers to this
    model and nothing here has been measured.
    """

    if not bool(config.attention_sink):
        return False
    kinds = str(getattr(config, "attention_sink_kinds", "all"))
    if kinds == "swa":
        return kind == "swa"
    return True


class AttentionOutputGate(nn.Module):
    """Head-wise elementwise sigmoid gate applied before ``o_proj``.

    ``out <- out * sigmoid(W_g x)`` with ``W_g`` producing one scalar per
    (head, head-dim) slot from the *block input*, the SDPA-output placement
    from Qwen's gated-attention study (NeurIPS 2025 oral, arXiv 2505.06708).
    Their 1.7B/400B-token run reports average PPL 7.499 -> 7.404 and attention
    mass on the first token falling 46.7% -> 4.8%.

    The weight starts at zero and the bias at ``+4``, so a freshly enabled gate
    is the constant ``sigmoid(4) = 0.982`` -- near-identity, which is what lets
    the flag be switched on over a warm start without destroying it. That init
    is restored by :meth:`reset_special_parameters` after the model's global
    ``_init_weights`` sweep, which would otherwise overwrite it.

    No Supermix run has measured this. It is a hypothesis for a future run.
    """

    def __init__(self, hidden_size: int, n_heads: int, head_dim: int, bias_init: float = 4.0):
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.bias_init = float(bias_init)
        self.proj = nn.Linear(hidden_size, self.n_heads * self.head_dim, bias=True)
        self.reset_special_parameters()

    def reset_special_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, self.bias_init)

    def forward(self, attn_out: torch.Tensor, block_input: torch.Tensor) -> torch.Tensor:
        """``attn_out`` and ``block_input`` are both ``(B, T, n_heads*head_dim)``-
        and ``(B, T, hidden)``-shaped respectively."""

        gate = torch.sigmoid(self.proj(block_input))
        return attn_out * gate.to(attn_out.dtype)


class HybridAttention(nn.Module):
    """Grouped-query attention that is either sliding-window or global.

    The ``kind`` is fixed per layer by :func:`attention_layout`. An SWA layer
    only ever needs the last ``sliding_window`` keys, so its cache is trimmed;
    a global layer keeps everything. That asymmetry is the whole point of the
    hybrid design and is asserted by the KV-cache tests.

    When ``attention_sink`` is on, one learnable per-head logit is concatenated
    to the score row before the softmax and then dropped from the weights. The
    head can therefore emit a near-zero output instead of being forced to
    normalise over real tokens -- the standard fix for the "attention sink"
    pathology that otherwise pins mass on position 0.

    ``attention_sink_kinds="swa"`` restricts the sink to the local layers,
    matching MiMo-V2-Flash's released config. ``config.qk_norm`` and
    ``config.attention_output_gate`` add the OLMo 2 / Qwen refinements; both are
    default-off and neither has been measured on this model.
    """

    def __init__(self, config: MiMoMixConfig, layer_index: int, kind: str):
        super().__init__()
        if kind not in {"swa", "global"}:
            raise ValueError(f"unknown attention kind: {kind!r}")
        self.config = config
        self.layer_index = int(layer_index)
        self.kind = kind
        self.n_heads = int(config.n_heads)
        self.n_kv_heads = int(config.n_kv_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = int(config.head_dim)
        self.scaling = self.head_dim ** -0.5
        self.window = int(config.sliding_window) if kind == "swa" else None

        # Overwritten by MiMoMixModel with the rotary policy's logit temperature.
        self._attention_scaling = 1.0

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if bool(getattr(config, "qk_norm", False)):
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        if bool(getattr(config, "attention_output_gate", False)):
            self.out_gate = AttentionOutputGate(config.hidden_size, self.n_heads, self.head_dim)
        else:
            self.out_gate = None

        if _sink_enabled(config, kind):
            # Starts at 0 => the sink initially holds softmax weight exp(0)=1
            # relative to the (scaled) real logits, a mild, learnable floor.
            self.sink = nn.Parameter(torch.zeros(self.n_heads))
        else:
            self.register_parameter("sink", None)

        self.register_buffer("last_sink_mass", torch.zeros(1), persistent=False)

    def cache_span(self, slack: int = 0) -> Optional[int]:
        """How many past keys this layer must retain, or ``None`` for all.

        ``slack`` keeps extra entries beyond the strict window. Speculative
        decoding needs it: rejecting ``r`` drafted tokens means rolling the
        cache back by ``r``, and a cache trimmed to exactly ``window`` would
        have already discarded the keys that rollback brings back into range.
        Holding ``window + slack`` makes any rollback of at most ``slack``
        entries exact.
        """

        if self.kind == "global":
            return None
        return int(self.window) + max(0, int(slack))

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        b, h, t, d = x.shape
        return x[:, :, None].expand(b, h, self.n_rep, t, d).reshape(b, h * self.n_rep, t, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_slack: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Returns ``(output, present_kv)``.

        ``query_positions`` are the absolute positions of ``hidden_states``.
        ``key_positions`` are the absolute positions of the *cached* keys (empty
        tensor when there is no cache); the new keys extend it.
        ``attention_mask`` is an optional ``(B, K_total)`` bool tensor which is
        ``True`` for real (non-padding) key positions.
        """

        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            # OLMo 2 / Gemma 3 / Qwen3 placement: normalise the per-head q and k
            # *before* the rotary, so the rotation still acts on unit-scale
            # vectors and the norm cannot undo a position-dependent phase.
            q = self.q_norm(q)
            k = self.k_norm(k)

        q_cos, q_sin = cos, sin
        q = apply_rotary(q, q_cos, q_sin)
        k = apply_rotary(k, q_cos, q_sin)

        if past_kv is not None and past_kv[0].numel() > 0:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
            all_key_positions = torch.cat([key_positions, query_positions], dim=0)
        else:
            all_key_positions = query_positions

        present: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if use_cache:
            span = self.cache_span(cache_slack)
            if span is not None and k.shape[2] > span:
                present = (k[:, :, -span:].detach(), v[:, :, -span:].detach())
            else:
                present = (k.detach(), v.detach())

        key_repeat = self._repeat_kv(k)
        value_repeat = self._repeat_kv(v)

        scores = torch.matmul(q, key_repeat.transpose(2, 3)) * self.scaling
        # YaRN's logit temperature compensation, a no-op for other schedules.
        scores = scores * self._attention_scaling

        causal = query_positions.view(-1, 1) >= all_key_positions.view(1, -1)
        if self.window is not None:
            causal = causal & (query_positions.view(-1, 1) - all_key_positions.view(1, -1) < self.window)
        allowed = causal.view(1, 1, q_len, -1)
        if attention_mask is not None:
            allowed = allowed & attention_mask.view(bsz, 1, 1, -1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg_inf)

        if self.sink is not None:
            sink = self.sink.view(1, self.n_heads, 1, 1).expand(bsz, self.n_heads, q_len, 1)
            scores = torch.cat([sink.to(scores.dtype), scores], dim=-1)
            weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
            sink_mass = weights[..., 0]
            weights = weights[..., 1:]
            self.last_sink_mass = sink_mass.detach().mean().reshape(1)
        else:
            # A fully masked row cannot happen under causal masking (a query can
            # always see itself), so a plain softmax is safe here.
            weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
            self.last_sink_mass = torch.zeros(1, device=q.device)

        weights = self.dropout(weights)
        out = torch.matmul(weights, value_repeat)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.n_heads * self.head_dim)
        if self.out_gate is not None:
            out = self.out_gate(out, hidden_states)
        return self.o_proj(out), present


# ---------------------------------------------------------------------------
# Differential Hybrid Attention (Microsoft ICLR 2025 oral)
# ---------------------------------------------------------------------------


class DifferentialHybridAttention(nn.Module):
    """Differential Attention (Microsoft ICLR 2025 oral) with SWA/Global hybrid design.

    Splits query and key representations into two halves (Q1, Q2) and (K1, K2).
    Computes two softmax attention maps and takes their difference:
        DiffMap = Softmax(Q1 K1^T / sqrt(d/2)) - lambda * Softmax(Q2 K2^T / sqrt(d/2))
    This subtractive operation acts like noise-cancelling headphones for attention,
    eliminating irrelevant background activations and sharpening retrieval.

    Two v82 additions, both governed by config flags:

    * ``differential_output_norm`` (default ``True``) restores the reference
      formulation's per-head sublayer normalisation and ``(1 - lambda_init)``
      rescale, which this implementation previously omitted. Changing this
      default is safe only because **no trained checkpoint sets
      ``use_differential_attention``** -- verified by loading every
      ``output/**/*.pt`` and reading its stored config. Set it to ``False`` to
      recover the pre-v82 arithmetic exactly.
    * ``differential_noise_ratio`` (default ``1``) allocates one shared noise
      map to every ``R`` signal heads instead of one each, by averaging the
      noise queries within a group. ``R == 1`` is the previous behaviour with
      identical weights. This is an approximation of the published GDA
      parameterisation, not a reimplementation of it, and its effect on this
      model is unmeasured.
    """

    def __init__(self, config: MiMoMixConfig, layer_index: int, kind: str):
        super().__init__()
        if kind not in {"swa", "global"}:
            raise ValueError(f"unknown attention kind: {kind!r}")
        self.config = config
        self.layer_index = int(layer_index)
        self.kind = kind
        self.n_heads = int(config.n_heads)
        self.n_kv_heads = int(config.n_kv_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = int(config.head_dim)
        self.sub_dim = self.head_dim // 2
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be divisible by 2 for differential attention")
        self.scaling = self.sub_dim ** -0.5
        self.window = int(config.sliding_window) if kind == "swa" else None
        self._attention_scaling = 1.0

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Learnable lambda per head: lambda = exp(-softplus(lambda_param))
        init_lambda = float(getattr(config, "differential_lambda_init", 0.8))
        init_lambda = max(1e-4, min(0.999, init_lambda))
        target_softplus = -math.log(init_lambda)
        if target_softplus > 0:
            p = math.log(max(1e-6, math.exp(target_softplus) - 1.0))
        else:
            p = 0.0
        self.lambda_param = nn.Parameter(torch.full((self.n_heads,), p, dtype=torch.float32))
        self.lambda_init = init_lambda

        self.noise_ratio = max(1, int(getattr(config, "differential_noise_ratio", 1)))
        if self.n_heads % self.noise_ratio != 0:
            raise ValueError(
                f"differential_noise_ratio ({self.noise_ratio}) must divide n_heads ({self.n_heads})"
            )
        self.n_noise_heads = self.n_heads // self.noise_ratio

        self.output_norm: Optional[RMSNorm]
        if bool(getattr(config, "differential_output_norm", True)):
            # Reference DIFF Transformer: per-head sublayer norm on the
            # differential output, then a (1 - lambda_init) rescale so the
            # sublayer's output magnitude matches a plain-attention one.
            self.output_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.output_rescale = 1.0 - init_lambda
        else:
            self.output_norm = None
            self.output_rescale = 1.0

        if bool(getattr(config, "qk_norm", False)):
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        if bool(getattr(config, "attention_output_gate", False)):
            self.out_gate = AttentionOutputGate(config.hidden_size, self.n_heads, self.head_dim)
        else:
            self.out_gate = None

        if _sink_enabled(config, kind):
            self.sink1 = nn.Parameter(torch.zeros(self.n_heads))
            self.sink2 = nn.Parameter(torch.zeros(self.n_noise_heads))
        else:
            self.register_parameter("sink1", None)
            self.register_parameter("sink2", None)

        self.register_buffer("last_sink_mass", torch.zeros(1), persistent=False)
        self.register_buffer("last_lambda", torch.zeros(self.n_heads), persistent=False)

    def cache_span(self, slack: int = 0) -> Optional[int]:
        if self.kind == "global":
            return None
        return int(self.window) + max(0, int(slack))

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        b, h, t, d = x.shape
        return x[:, :, None].expand(b, h, self.n_rep, t, d).reshape(b, h * self.n_rep, t, d)

    def _match_kv_heads(self, x: torch.Tensor, target_heads: int) -> torch.Tensor:
        """Expand (or, for a narrow noise branch, group-average) kv heads."""

        b, h, t, d = x.shape
        if h == target_heads:
            return x
        if target_heads > h:
            if target_heads % h != 0:
                raise ValueError(f"cannot expand {h} kv heads to {target_heads}")
            rep = target_heads // h
            return x[:, :, None].expand(b, h, rep, t, d).reshape(b, h * rep, t, d)
        if h % target_heads != 0:
            raise ValueError(f"cannot group {h} kv heads down to {target_heads}")
        return x.view(b, target_heads, h // target_heads, t, d).mean(dim=2)

    @property
    def lambda_weight(self) -> torch.Tensor:
        """Bounded per-head differential cancellation weight in (0, 1)."""
        return torch.exp(-F.softplus(self.lambda_param))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_slack: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if past_kv is not None and past_kv[0].numel() > 0:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
            all_key_positions = torch.cat([key_positions, query_positions], dim=0)
        else:
            all_key_positions = query_positions

        present: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if use_cache:
            span = self.cache_span(cache_slack)
            if span is not None and k.shape[2] > span:
                present = (k[:, :, -span:].detach(), v[:, :, -span:].detach())
            else:
                present = (k.detach(), v.detach())

        # Split into differential sub-heads
        q1 = q[..., :self.sub_dim]
        q2 = q[..., self.sub_dim:]
        k1 = k[..., :self.sub_dim]
        k2 = k[..., self.sub_dim:]

        if self.noise_ratio > 1:
            # GDA-style asymmetry: R signal heads share one noise map. The
            # shared noise query is the mean of the group's noise queries, so
            # every q2 parameter still receives gradient (no dead weights).
            b, _, t, d = q2.shape
            q2 = q2.view(b, self.n_noise_heads, self.noise_ratio, t, d).mean(dim=2)

        k1_repeat = self._repeat_kv(k1)
        k2_repeat = self._match_kv_heads(k2, self.n_noise_heads)
        value_repeat = self._repeat_kv(v)

        scores1 = torch.matmul(q1, k1_repeat.transpose(2, 3)) * self.scaling * self._attention_scaling
        scores2 = torch.matmul(q2, k2_repeat.transpose(2, 3)) * self.scaling * self._attention_scaling

        causal = query_positions.view(-1, 1) >= all_key_positions.view(1, -1)
        if self.window is not None:
            causal = causal & (query_positions.view(-1, 1) - all_key_positions.view(1, -1) < self.window)
        allowed = causal.view(1, 1, q_len, -1)
        if attention_mask is not None:
            allowed = allowed & attention_mask.view(bsz, 1, 1, -1)
        neg_inf = torch.finfo(scores1.dtype).min
        scores1 = scores1.masked_fill(~allowed, neg_inf)
        scores2 = scores2.masked_fill(~allowed, neg_inf)

        if self.sink1 is not None and self.sink2 is not None:
            sink1 = self.sink1.view(1, self.n_heads, 1, 1).expand(bsz, self.n_heads, q_len, 1)
            sink2 = self.sink2.view(1, self.n_noise_heads, 1, 1).expand(
                bsz, self.n_noise_heads, q_len, 1
            )
            scores1 = torch.cat([sink1.to(scores1.dtype), scores1], dim=-1)
            scores2 = torch.cat([sink2.to(scores2.dtype), scores2], dim=-1)
            map1 = F.softmax(scores1.float(), dim=-1).to(q.dtype)
            map2 = F.softmax(scores2.float(), dim=-1).to(q.dtype)
            sink_mass = (map1[..., 0].mean() + map2[..., 0].mean()) * 0.5
            self.last_sink_mass = sink_mass.detach().reshape(1)
            map1 = map1[..., 1:]
            map2 = map2[..., 1:]
        else:
            map1 = F.softmax(scores1.float(), dim=-1).to(q.dtype)
            map2 = F.softmax(scores2.float(), dim=-1).to(q.dtype)
            self.last_sink_mass = torch.zeros(1, device=q.device)

        if self.noise_ratio > 1:
            map2 = map2.repeat_interleave(self.noise_ratio, dim=1)

        lambda_head = self.lambda_weight.view(1, self.n_heads, 1, 1).to(q.dtype)
        self.last_lambda = self.lambda_weight.detach()

        diff_weights = map1 - lambda_head * map2
        weights = self.dropout(diff_weights)
        out = torch.matmul(weights, value_repeat)
        if self.output_norm is not None:
            # Reference DIFF-Transformer sublayer norm, applied per head, then
            # the (1 - lambda_init) rescale. Pre-v82 this was omitted entirely.
            out = self.output_norm(out) * self.output_rescale
        out = out.transpose(1, 2).reshape(bsz, q_len, self.n_heads * self.head_dim)
        if self.out_gate is not None:
            out = self.out_gate(out, hidden_states)
        return self.o_proj(out), present


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V3 MLA)
# ---------------------------------------------------------------------------


class MultiLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) for deep KV-cache compression.

    Compresses Keys and Values jointly into a compact low-rank latent vector
    c_kv, and handles rotary positional embeddings via a dedicated decoupled
    k_pe / q_pe projection. This drastically shrinks the KV-cache memory
    footprint on long contexts without losing full multi-head expressive power.

    **v82 fix.** The decoupled pe sub-vector is ``mla_pe_dim`` wide, which is
    usually narrower than ``head_dim``. Before v82 this class sliced
    ``cos[:, :pe_dim]`` out of the *trunk's* ``head_dim``-wide table. That is
    not a rotation: the trunk table is ``cat([freqs, freqs])`` over
    ``head_dim/2`` frequencies, so component ``i`` is meant to pair with
    ``i + head_dim/2``; truncating to ``pe_dim`` pairs it with
    ``i + pe_dim/2`` instead, which carries a *different* frequency. Measured
    consequences on the pre-fix code: the rotation changed the vector norm by
    2.2866 (a rotation must change it by 0), and one fixed relative offset
    scored ``+3.9137`` at absolute positions (5, 2) but ``+7.5882`` at
    (15, 12) -- i.e. the "relative" position encoding was not relative. This
    class now owns a :class:`RotaryEmbedding` built over ``pe_dim``
    frequencies, so the pe rotation is a genuine rotation.

    The layer keeps using the trunk's YaRN logit temperature
    (``_attention_scaling``, assigned by :class:`MiMoMixModel`); only the
    frequency table is decoupled.
    """

    def __init__(self, config: MiMoMixConfig, layer_index: int, kind: str = "global"):
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        self.kind = kind
        self.n_heads = int(config.n_heads)
        self.head_dim = int(config.head_dim)
        self.latent_dim = int(getattr(config, "mla_latent_dim", 32))
        self.pe_dim = int(getattr(config, "mla_pe_dim", 16))
        self.scaling = (self.head_dim + self.pe_dim) ** -0.5
        self.window = int(config.sliding_window) if kind == "swa" else None
        self._attention_scaling = 1.0

        self.q_content_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.q_pe_proj = nn.Linear(config.hidden_size, self.n_heads * self.pe_dim, bias=False)

        self.kv_down_proj = nn.Linear(config.hidden_size, self.latent_dim, bias=False)
        self.kv_norm = RMSNorm(self.latent_dim, config.rms_norm_eps)
        self.k_up_proj = nn.Linear(self.latent_dim, self.n_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(self.latent_dim, self.n_heads * self.head_dim, bias=False)
        self.k_pe_proj = nn.Linear(config.hidden_size, self.pe_dim, bias=False)

        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if self.pe_dim > self.head_dim or self.pe_dim <= 0 or self.pe_dim % 2 != 0:
            raise ValueError(
                f"mla_pe_dim ({self.pe_dim}) must be a positive even number <= head_dim ({self.head_dim})"
            )
        # A rope table of exactly pe_dim frequencies. See the class docstring:
        # slicing the trunk's head_dim-wide table here was Bug A.
        self.pe_rotary = RotaryEmbedding(config, kind=kind, rotary_dim=self.pe_dim)

        if bool(getattr(config, "qk_norm", False)):
            # The pe halves carry the position signal and are left alone; the
            # content halves are the ones whose scale the OLMo 2 placement is
            # about. k_content is produced by the up-projection from the latent,
            # so its norm sits after that projection.
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        if bool(getattr(config, "attention_output_gate", False)):
            self.out_gate = AttentionOutputGate(config.hidden_size, self.n_heads, self.head_dim)
        else:
            self.out_gate = None

        if _sink_enabled(config, kind):
            self.sink = nn.Parameter(torch.zeros(self.n_heads))
        else:
            self.register_parameter("sink", None)

        self.register_buffer("last_sink_mass", torch.zeros(1), persistent=False)

    def cache_span(self, slack: int = 0) -> Optional[int]:
        if self.kind == "global":
            return None
        return int(self.window) + max(0, int(slack))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_slack: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        q_content = self.q_content_proj(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        q_pe = self.q_pe_proj(hidden_states).view(bsz, q_len, self.n_heads, self.pe_dim).transpose(1, 2)

        # Own table over pe_dim frequencies -- NOT a slice of the trunk's table.
        pe_cos, pe_sin = self.pe_rotary(query_positions)
        if self.q_norm is not None:
            q_content = self.q_norm(q_content)
        q_pe = apply_rotary(q_pe, pe_cos, pe_sin)

        c_kv = self.kv_norm(self.kv_down_proj(hidden_states))  # (B, q_len, latent_dim)
        k_pe = self.k_pe_proj(hidden_states).view(bsz, q_len, 1, self.pe_dim).transpose(1, 2)  # (B, 1, q_len, pe_dim)
        k_pe = apply_rotary(k_pe, pe_cos, pe_sin)

        if past_kv is not None and past_kv[0].numel() > 0:
            c_kv_all = torch.cat([past_kv[0], c_kv], dim=1)
            k_pe_all = torch.cat([past_kv[1], k_pe], dim=2)
            all_key_positions = torch.cat([key_positions, query_positions], dim=0)
        else:
            c_kv_all = c_kv
            k_pe_all = k_pe
            all_key_positions = query_positions

        present: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if use_cache:
            span = self.cache_span(cache_slack)
            if span is not None and c_kv_all.shape[1] > span:
                present = (c_kv_all[:, -span:].detach(), k_pe_all[:, :, -span:].detach())
            else:
                present = (c_kv_all.detach(), k_pe_all.detach())

        total_len = c_kv_all.shape[1]
        k_content = self.k_up_proj(c_kv_all).view(bsz, total_len, self.n_heads, self.head_dim).transpose(1, 2)
        if self.k_norm is not None:
            k_content = self.k_norm(k_content)
        v = self.v_up_proj(c_kv_all).view(bsz, total_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores_content = torch.matmul(q_content, k_content.transpose(2, 3))
        scores_pe = torch.matmul(q_pe, k_pe_all.expand(bsz, self.n_heads, total_len, self.pe_dim).transpose(2, 3))
        scores = (scores_content + scores_pe) * self.scaling * self._attention_scaling

        causal = query_positions.view(-1, 1) >= all_key_positions.view(1, -1)
        if self.window is not None:
            causal = causal & (query_positions.view(-1, 1) - all_key_positions.view(1, -1) < self.window)
        allowed = causal.view(1, 1, q_len, -1)
        if attention_mask is not None:
            allowed = allowed & attention_mask.view(bsz, 1, 1, -1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg_inf)

        if self.sink is not None:
            sink = self.sink.view(1, self.n_heads, 1, 1).expand(bsz, self.n_heads, q_len, 1)
            scores = torch.cat([sink.to(scores.dtype), scores], dim=-1)
            weights = F.softmax(scores.float(), dim=-1).to(q_content.dtype)
            sink_mass = weights[..., 0]
            weights = weights[..., 1:]
            self.last_sink_mass = sink_mass.detach().mean().reshape(1)
        else:
            weights = F.softmax(scores.float(), dim=-1).to(q_content.dtype)
            self.last_sink_mass = torch.zeros(1, device=q_content.device)

        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.n_heads * self.head_dim)
        if self.out_gate is not None:
            out = self.out_gate(out, hidden_states)
        return self.o_proj(out), present


# ---------------------------------------------------------------------------
# Feed-forward: dense and sparse-MoE
# ---------------------------------------------------------------------------


class DenseFeedForward(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SparseMoEFeedForward(nn.Module):
    """Top-k sparse MoE with auxiliary-loss-free load balancing.

    Routing follows the DeepSeek-V3 / *Auxiliary-Loss-Free Load Balancing*
    recipe:

    1. score every expert from the token, ``s_i = softmax(W x)_i`` (or sigmoid);
    2. **select** the top-k experts by ``s_i + b_i`` where ``b_i`` is a
       non-gradient per-expert bias;
    3. **weight** the selected experts by the raw ``s_i`` (the bias never enters
       the forward value, so it cannot distort the model's function);
    4. after the step, nudge ``b_i`` by ``gamma * sign(mean_load - load_i)``, so
       overloaded experts become less attractive next step.

    Two gradient-carrying regularisers remain, both small:

    * a **router z-loss** ``mean(logsumexp(logits)^2)`` that keeps router logits
      from drifting into a numerically bad regime;
    * a light **balance loss** ``N * sum(f_i * P_i)``. ``moe_balance_scope``
      selects its scope. ``"batch"`` (the default, and what every run up to and
      including v80 actually computed despite older docstrings calling it
      sequence-level) flattens ``(B, T)`` into one pool. ``"sequence"``
      computes ``f`` and ``P`` per sequence and averages over the batch, which
      is the DeepSeek-V3 4.5.3 complementary sequence-wise term. The bias rule
      does the real work either way; the effect of the scope on this model is
      unmeasured.

    ``n_shared_experts`` experts are always applied. Shared experts capture the
    common transformation so the routed experts can specialise -- fine-grained
    expert segmentation in the DeepSeekMoE sense.

    **v93 expert slots.** ``moe_spare_experts`` extra slots are allocated at
    construction so no tensor changes shape mid-run (the optimiser, scheduler
    and strict checkpoint load are untouched by growth). A persistent
    ``expert_alive`` mask says which slots are in use; a dead slot's router
    logit and selection score are set to ``-inf`` before the softmax, so its
    score is exactly 0, it can never enter the top-k, and the balance loss is
    taken over the alive count. With ``moe_spare_experts == 0`` no masking
    runs at all and the module is byte-identical to v89's.
    """

    #: Buffers introduced by v93. A checkpoint written before them lacks the
    #: key; :meth:`_load_from_state_dict` fills it with the construction-time
    #: default instead of reporting it missing, so v89/v91 checkpoints still
    #: load strictly.
    _V93_BUFFERS = ("expert_alive",)

    def __init__(self, config: MiMoMixConfig):
        super().__init__()
        self.config = config
        self.n_live_experts = int(config.n_routed_experts)
        self.n_spare = int(getattr(config, "moe_spare_experts", 0))
        self.n_routed = self.n_live_experts + self.n_spare
        self.top_k = int(config.moe_top_k)
        self.n_shared = int(config.n_shared_experts)
        self.score_function = config.router_score_function
        self.norm_topk_prob = bool(config.norm_topk_prob)
        self.update_speed = float(config.router_bias_update_speed)
        self.balance_scope = str(getattr(config, "moe_balance_scope", "batch"))

        self.gate = nn.Linear(config.hidden_size, self.n_routed, bias=False)
        self.experts = nn.ModuleList(
            [
                DenseFeedForward(config.hidden_size, config.moe_intermediate_size, config.dropout)
                for _ in range(self.n_routed)
            ]
        )
        if self.n_shared > 0:
            self.shared_expert = DenseFeedForward(
                config.hidden_size,
                config.moe_intermediate_size * self.n_shared,
                config.dropout,
            )
        else:
            self.shared_expert = None

        self.auto_update_bias = bool(config.router_bias_auto_update)

        # Selection bias is state, not a parameter: it is updated by a rule, not
        # by the optimizer, and it must survive a checkpoint round trip.
        self.register_buffer("expert_bias", torch.zeros(self.n_routed), persistent=True)
        # v93 slot mask: the first n_routed_experts slots are born alive, the
        # spares dead. uint8 rather than bool so the checkpoint is portable.
        alive = torch.zeros(self.n_routed, dtype=torch.uint8)
        alive[: self.n_live_experts] = 1
        self.register_buffer("expert_alive", alive, persistent=True)
        self.register_buffer("last_expert_load", torch.zeros(self.n_routed), persistent=False)
        # Load accumulated since the last bias step. See update_router_bias().
        self.register_buffer("pending_load", torch.zeros(self.n_routed), persistent=False)
        self.register_buffer("pending_batches", torch.zeros(()), persistent=False)
        self.register_buffer("last_router_z_loss", torch.zeros(()), persistent=False)
        self.register_buffer("last_router_balance_loss", torch.zeros(()), persistent=False)
        self.register_buffer("last_router_entropy", torch.zeros(()), persistent=False)
        # v93 growth statistics (dev-load accumulation, eval mode only).
        self.collect_stats = False
        self.register_buffer("stat_load_sum", torch.zeros(self.n_routed), persistent=False)
        self.register_buffer("stat_batches", torch.zeros(()), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Fill v93 buffers missing from an older checkpoint with their defaults.

        ``nn.Module.load_state_dict`` hands every module a *copy* of the
        payload, so adding the key here changes nothing the caller holds, and
        the key then loads like any other -- strict mode neither reports it
        missing nor unexpected. The default is the construction-time value,
        which for a v89/v91 config (``moe_spare_experts == 0``) is all-alive.
        """

        for name in self._V93_BUFFERS:
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.score_function == "sigmoid":
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)

    # -- v93 slot bookkeeping ----------------------------------------------

    def alive_count(self) -> int:
        return int(self.expert_alive.sum())

    def _all_alive(self) -> bool:
        """True when no slot is dead, i.e. the pre-v93 code path applies exactly."""

        return not bool((self.expert_alive == 0).any())

    @torch.no_grad()
    def birth_expert(self, parent: int, noise: float = 0.01) -> Optional[int]:
        """Fill the first dead slot with a copy of expert ``parent`` (v93 D3).

        The child gets the parent's expert weights, router row and selection
        bias, each weight tensor perturbed by ``noise * std(tensor)`` Gaussian
        noise so the twins can diverge. Returns the child's index, or ``None``
        when no slot is free or ``parent`` is dead.

        This is **not** function-preserving: the child scores like its parent
        from the first token, so tokens that routed to the parent now see two
        near-identical candidates in the top-k. The controller measures the
        witness-batch loss before and after and logs it (D6).
        """

        parent = int(parent)
        if not (0 <= parent < self.n_routed) or int(self.expert_alive[parent]) == 0:
            return None
        free = torch.nonzero(self.expert_alive == 0, as_tuple=False).flatten()
        if free.numel() == 0:
            return None
        child = int(free[0])
        source = self.experts[parent]
        target = self.experts[child]
        for p_src, p_dst in zip(source.parameters(), target.parameters()):
            p_dst.copy_(p_src)
            if noise > 0:
                p_dst.add_(torch.randn_like(p_dst) * (float(noise) * float(p_src.std())))
        row = self.gate.weight[parent]
        self.gate.weight[child].copy_(row)
        if noise > 0:
            self.gate.weight[child].add_(torch.randn_like(row) * (float(noise) * float(row.std())))
        self.expert_bias[child] = self.expert_bias[parent]
        self.expert_alive[child] = 1
        self.pending_load[child] = 0.0
        self.last_expert_load[child] = 0.0
        return child

    @torch.no_grad()
    def kill_expert(self, index: int) -> bool:
        """Mask slot ``index`` off and reset it (zero weights, zero bias).

        Refused (``False``) when the slot is already dead or when killing it
        would leave fewer than ``top_k`` alive experts, since the router must
        always be able to fill its top-k.
        """

        index = int(index)
        if not (0 <= index < self.n_routed) or int(self.expert_alive[index]) == 0:
            return False
        if self.alive_count() - 1 < self.top_k:
            return False
        self.expert_alive[index] = 0
        for parameter in self.experts[index].parameters():
            parameter.zero_()
        self.gate.weight[index].zero_()
        self.expert_bias[index] = 0.0
        self.pending_load[index] = 0.0
        self.last_expert_load[index] = 0.0
        return True

    def reset_stats(self) -> None:
        self.stat_load_sum.zero_()
        self.stat_batches.zero_()

    def growth_statistics(self) -> Dict[str, object]:
        """Dev-pass expert load, accumulated in eval mode while ``collect_stats``.

        ``mean_load`` is the per-slot fraction of tokens routed there, averaged
        over accumulated batches (length ``n_routed``, dead slots read 0).
        """

        batches = float(self.stat_batches)
        mean_load = self.stat_load_sum / max(batches, 1.0)
        return {
            "mean_load": mean_load.detach().clone(),
            "batches": int(batches),
            "alive": self.expert_alive.detach().clone().bool(),
            "n_alive": self.alive_count(),
        }

    @torch.no_grad()
    def update_router_bias(self) -> bool:
        """Apply one bias update from the load accumulated since the last call.

        **This is deliberately not done inside** :meth:`forward`. The bias rule
        is a control loop with step size ``gamma``, and firing it once per
        forward makes the effective step depend on how many forwards happen per
        optimizer step:

        * gradient accumulation over ``N`` micro-batches fires it ``N`` times,
          multiplying gamma by ``N``;
        * the thinking controller probes the same request at several budgets,
          firing it once per probe;
        * any evaluation pass left in train mode corrupts routing state.

        An oversized effective step does not merely converge faster -- the sign
        rule overshoots the balance point every update and **rings**, so the
        router looks *more* collapsed, not less. Measured on the benchmark task,
        ``gamma=1e-3`` converges while ``gamma>=5e-2`` starves most experts.

        Call this once per optimizer step, after ``optimizer.step()``. Returns
        ``False`` if there was nothing to apply.
        """

        if self.update_speed <= 0.0 or float(self.pending_batches) <= 0.0:
            return False
        load = self.pending_load / self.pending_batches
        if self._all_alive():
            target = load.mean()
            # sign(+) => this expert is under-loaded => raise its bias
            self.expert_bias += self.update_speed * torch.sign(target - load)
        else:
            # v93: dead slots carry load 0 and must neither drag the target
            # down (which would read every alive expert as overloaded) nor
            # have their own bias walked; only alive slots take part.
            alive = self.expert_alive.bool()
            target = load[alive].mean()
            step = self.update_speed * torch.sign(target - load)
            self.expert_bias += torch.where(alive, step, torch.zeros_like(step))
        self.pending_load.zero_()
        self.pending_batches.zero_()
        return True

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``token_mask`` is an optional ``(B, T)`` bool tensor of tokens that
        actually reach the residual stream.

        Mixture-of-Depths gates a token's MLP contribution to zero, but the
        expert is still selected for it, so before v82 gated-out tokens drove
        ``expert_load``, ``pending_load`` and the balance loss exactly as hard
        as tokens the block actually used. Passing the MoD gate mask here
        excludes them from all three. It does **not** stop the experts running
        on those tokens -- see :class:`MixtureOfDepthsRouter` on why there is no
        FLOP saving here.
        """

        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        n_tokens = flat.shape[0]
        flat_mask: Optional[torch.Tensor] = None
        if token_mask is not None:
            flat_mask = token_mask.reshape(-1).to(dtype=torch.bool, device=flat.device)
            if flat_mask.numel() != n_tokens:
                raise ValueError(
                    f"token_mask has {flat_mask.numel()} entries for {n_tokens} tokens"
                )

        logits = self.gate(flat)
        all_alive = self._all_alive()
        n_balance = self.n_routed
        dead_slots: Optional[torch.Tensor] = None
        if not all_alive:
            # v93 slot mask. -inf before the softmax makes a dead slot's score
            # exactly 0 (and drops it from the z-loss's logsumexp); the
            # selection score is masked too because v89's expert_bias sits at
            # 16.5..17.7, so a 0 score plus a stale bias could still win top-k.
            dead_slots = (self.expert_alive == 0).unsqueeze(0)
            logits = logits.masked_fill(dead_slots, float("-inf"))
            n_balance = self.alive_count()
        scores = self._scores(logits)
        selection_scores = scores + self.expert_bias.to(scores.dtype).unsqueeze(0)
        if dead_slots is not None:
            selection_scores = selection_scores.masked_fill(dead_slots, float("-inf"))
        _, expert_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        gate_weights = scores.gather(-1, expert_indices)
        if self.norm_topk_prob and self.top_k > 1:
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        one_hot = F.one_hot(expert_indices, num_classes=self.n_routed).sum(dim=1)  # (N, E)
        for expert_id, expert in enumerate(self.experts):
            token_ids = torch.nonzero(one_hot[:, expert_id], as_tuple=False).flatten()
            if token_ids.numel() == 0:
                continue
            expert_out = expert(flat.index_select(0, token_ids))
            weight = (gate_weights * (expert_indices == expert_id)).sum(dim=-1)
            weight = weight.index_select(0, token_ids).unsqueeze(-1)
            # `index_add_` requires the source to match the accumulator's dtype
            # exactly. Under autocast the experts return bf16/fp16 while
            # `output` was allocated from `flat` in fp32, so the cast is what
            # lets mixed precision run at all -- without it the MoE path raises
            # on the first step.
            contribution = expert_out * weight.to(expert_out.dtype)
            output.index_add_(0, token_ids, contribution.to(output.dtype))

        if self.shared_expert is not None:
            output = output + self.shared_expert(flat)

        # --- routing telemetry and regularisers ---------------------------
        occupancy = one_hot.float()
        probabilities = scores.float()
        if flat_mask is not None:
            keep = flat_mask.float().unsqueeze(-1)
            denom = keep.sum().clamp_min(1.0)
            load = (occupancy * keep).sum(dim=0) / denom
            mean_prob = (probabilities * keep).sum(dim=0) / denom
        else:
            load = occupancy.mean(dim=0)  # fraction of tokens per expert
            mean_prob = probabilities.mean(dim=0)
        self.last_expert_load = load.detach()
        self.last_router_entropy = (
            -(mean_prob * mean_prob.clamp_min(1e-9).log()).sum().detach()
        )
        z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        if self.balance_scope == "sequence" and len(original_shape) == 3:
            bsz, seq_len = int(original_shape[0]), int(original_shape[1])
            per_seq_occupancy = occupancy.view(bsz, seq_len, self.n_routed)
            per_seq_prob = probabilities.view(bsz, seq_len, self.n_routed)
            if flat_mask is not None:
                keep_seq = flat_mask.view(bsz, seq_len, 1).float()
                denom_seq = keep_seq.sum(dim=1).clamp_min(1.0)
                seq_load = (per_seq_occupancy * keep_seq).sum(dim=1) / denom_seq
                seq_prob = (per_seq_prob * keep_seq).sum(dim=1) / denom_seq
            else:
                seq_load = per_seq_occupancy.mean(dim=1)
                seq_prob = per_seq_prob.mean(dim=1)
            balance_loss = float(n_balance) * (seq_load * seq_prob).sum(dim=-1).mean()
        else:
            balance_loss = float(n_balance) * torch.sum(load * mean_prob)
        self.last_router_z_loss = z_loss.detach()
        self.last_router_balance_loss = balance_loss.detach()
        if self.training:
            self._aux_loss = (
                self.config.router_z_loss_coef * z_loss
                + self.config.router_balance_loss_coef * balance_loss
            )
            with torch.no_grad():
                self.pending_load += load.detach()
                self.pending_batches += 1.0
            if self.auto_update_bias:
                self.update_router_bias()
        else:
            self._aux_loss = logits.new_zeros(())
            if self.collect_stats:
                # v93: dev-pass load for the neurogenesis controller. Eval
                # mode only, so a training forward never pollutes it and the
                # routing state contract (pending_load untouched in eval) holds.
                with torch.no_grad():
                    self.stat_load_sum += load.detach()
                    self.stat_batches += 1.0

        return output.reshape(original_shape)

    def aux_loss(self) -> torch.Tensor:
        value = getattr(self, "_aux_loss", None)
        if value is None:
            return self.expert_bias.new_zeros(())
        return value


# ---------------------------------------------------------------------------
# Mixture-of-Depths router (Google DeepMind 2024)
# ---------------------------------------------------------------------------


class MixtureOfDepthsRouter(nn.Module):
    """Mixture-of-Depths (MoD) token-level router (arXiv 2404.02258).

    **This does not save compute here, and it never did.** The block gates only
    the *residual contribution* of the attention and MLP sublayers; both
    sublayers still execute for every token, on the full sequence. There is no
    gather/scatter and no shortened sequence, so FLOPs are unchanged and
    wall-clock is slightly *worse* than not using it. Treat this as a routing /
    regularisation study -- a learned per-token gate on a block's output -- and
    not as a compute saving. Telemetry reports ``mod_compute_saved: false`` to
    keep that visible at the point of use.

    Two v82 correctness fixes (both measured against the pre-v82 code):

    *Capacity was per-call.* ``capacity = ceil(seq_len * ratio)`` is computed
    from *this call's* ``seq_len``, so at ``seq_len == 1`` -- every cached
    decode step -- capacity 1 >= 1 selected every token. Measured skip ratio was
    0.500 at T=8 and T=4 but 0.000 at T=1, and full-forward vs incremental
    decode logits differed by 8.1e-2 against a 1.2e-7 non-MoD baseline.

    *Top-k is not causal.* Selecting the top ``capacity`` tokens of the whole
    sequence lets a later token change an earlier token's routing: measured,
    changing token 7 of 8 moved positions 0-3 by 0.519.

    The paper's own remedy is implemented here: keep top-k for the training
    forward, and train a small per-token **causal predictor** (a linear head,
    BCE against top-k membership) that decides selection whenever the block
    cannot see the whole sequence. To make batched and incremental decode agree
    token-for-token, the predictor is used for *all* eval-mode selection, not
    only cached steps -- otherwise a full eval forward would still take the
    top-k path while its incremental replay took the predictor path, and the
    two would disagree exactly as before.

    Setting ``mod_causal_predictor=False`` restores the pre-v82 behaviour,
    including the mismatch.
    """

    def __init__(
        self,
        hidden_size: int,
        capacity_ratio: float = 0.5,
        causal_predictor: bool = True,
        predictor_loss_coef: float = 1e-2,
    ):
        super().__init__()
        self.capacity_ratio = float(capacity_ratio)
        self.router_proj = nn.Linear(hidden_size, 1, bias=False)
        self.use_causal_predictor = bool(causal_predictor)
        self.predictor_loss_coef = float(predictor_loss_coef)
        if self.use_causal_predictor:
            self.predictor = nn.Linear(hidden_size, 1, bias=True)
        else:
            self.predictor = None
        self.register_buffer("last_skip_ratio", torch.zeros(1), persistent=False)
        self.register_buffer("last_predictor_loss", torch.zeros(()), persistent=False)
        self.register_buffer("last_predictor_agreement", torch.zeros(()), persistent=False)
        self.last_selection_mode = "topk"

    def _clear_aux(self, reference: torch.Tensor) -> None:
        self._aux_loss = reference.new_zeros(())

    def aux_loss(self) -> torch.Tensor:
        """Predictor BCE, already multiplied by ``mod_predictor_loss_coef``."""

        value = getattr(self, "_aux_loss", None)
        if value is None:
            return self.router_proj.weight.new_zeros(())
        return value

    def gate_mask(self, x: torch.Tensor, incremental: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(selected_mask, weights)``, both ``(B, T)``.

        ``selected_mask`` is bool; ``weights`` is ``sigmoid(router_logits)``.
        ``incremental`` says the caller cannot see the whole sequence (a cache
        is attached, or the query block is shorter than the context).
        """

        bsz, seq_len, _ = x.shape
        router_logits = self.router_proj(x).squeeze(-1)  # (B, T)
        weights = torch.sigmoid(router_logits)
        capacity = max(1, min(seq_len, int(math.ceil(seq_len * self.capacity_ratio))))

        predictor_logits: Optional[torch.Tensor] = None
        if self.predictor is not None:
            predictor_logits = self.predictor(x).squeeze(-1)

        # top-k membership over the visible sequence: the training target and,
        # in training mode, the selection itself.
        topk_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=x.device)
        if capacity >= seq_len:
            topk_mask[:] = True
        else:
            _, indices = torch.topk(weights, capacity, dim=-1, sorted=False)
            topk_mask.scatter_(1, indices, True)

        use_predictor = self.predictor is not None and (incremental or not self.training)
        if use_predictor:
            assert predictor_logits is not None
            selected_mask = predictor_logits > 0.0
            self.last_selection_mode = "predictor"
        else:
            selected_mask = topk_mask
            self.last_selection_mode = "topk"

        if self.training and predictor_logits is not None:
            loss = F.binary_cross_entropy_with_logits(
                predictor_logits, topk_mask.to(predictor_logits.dtype)
            )
            self._aux_loss = self.predictor_loss_coef * loss
            self.last_predictor_loss = loss.detach()
        else:
            self._clear_aux(weights)

        # Agreement between the causal predictor and the top-k target it was
        # trained against. This is the diagnostic that says whether the
        # predictor is a usable stand-in at decode time, so it is exactly the
        # number wanted under eval() -- and it used to be computed only inside
        # the training branch above, leaving every eval and benchmark snapshot
        # reporting a stale value or the 0.0 it was initialised with. The aux
        # loss stays training-only; this does not need to be. `topk_mask` is
        # already computed on every call, so the cost is one comparison.
        if predictor_logits is not None:
            self.last_predictor_agreement = (
                ((predictor_logits > 0.0) == topk_mask).float().mean().detach()
            )

        skipped = (~selected_mask).float().mean()
        self.last_skip_ratio = skipped.detach().reshape(1)
        return selected_mask, weights

    def forward(
        self, x: torch.Tensor, incremental: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy tuple interface: ``(selected_indices, weights, skip_ratio)``.

        Kept because the existing tests and any external caller use it. It can
        only return a rectangular index tensor, which is why the block itself
        uses :meth:`gate_mask` -- the predictor selects a different number of
        tokens per row.
        """

        bsz, seq_len, _ = x.shape
        selected_mask, weights = self.gate_mask(x, incremental=incremental)
        counts = selected_mask.sum(dim=-1)
        width = int(counts.max().item()) if counts.numel() else 0
        width = max(1, width)
        # Pad short rows by repeating their first selected index; the mask is
        # the authoritative object, this is a convenience view.
        selected_indices = torch.zeros(bsz, width, dtype=torch.long, device=x.device)
        for row in range(bsz):
            idx = torch.nonzero(selected_mask[row], as_tuple=False).flatten()
            if idx.numel() == 0:
                idx = torch.zeros(1, dtype=torch.long, device=x.device)
            if idx.numel() < width:
                idx = torch.cat([idx, idx[-1:].expand(width - idx.numel())])
            selected_indices[row] = idx[:width]
        return selected_indices, weights, self.last_skip_ratio


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------


class MiMoMixBlock(nn.Module):
    def __init__(self, config: MiMoMixConfig, layer_index: int, kind: str, force_dense: bool = False):
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        self.kind = kind
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if getattr(config, "use_differential_attention", False):
            self.self_attn = DifferentialHybridAttention(config, layer_index, kind)
        elif getattr(config, "use_mla", False) and (kind == "global" or not getattr(config, "mla_global_only", True)):
            self.self_attn = MultiLatentAttention(config, layer_index, kind)
        else:
            self.self_attn = HybridAttention(config, layer_index, kind)

        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        is_moe = config.use_moe and layer_index >= config.n_dense_layers and not force_dense
        self.mlp: nn.Module
        if is_moe:
            self.mlp = SparseMoEFeedForward(config)
        else:
            self.mlp = DenseFeedForward(config.hidden_size, config.intermediate_size, config.dropout)
        self.is_moe = is_moe

        self.use_mod = bool(getattr(config, "use_mod", False) and layer_index >= config.n_dense_layers and not force_dense)
        if self.use_mod:
            self.mod_router = MixtureOfDepthsRouter(
                config.hidden_size,
                getattr(config, "mod_capacity_ratio", 0.5),
                causal_predictor=bool(getattr(config, "mod_causal_predictor", True)),
                predictor_loss_coef=float(getattr(config, "mod_predictor_loss_coef", 1e-2)),
            )
        else:
            self.mod_router = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        past_kv=None,
        attention_mask=None,
        use_cache: bool = False,
        cache_slack: int = 0,
    ):
        residual = hidden_states
        attn_out, present = self.self_attn(
            self.input_norm(hidden_states),
            cos,
            sin,
            query_positions,
            key_positions,
            past_kv=past_kv,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache_slack=cache_slack,
        )

        if self.use_mod and self.mod_router is not None:
            # "Cannot see the whole sequence" == a cache is attached, or this is
            # a single-token decode step. In eval mode the router uses its
            # causal predictor regardless, so a batched forward and its
            # incremental replay take the same path.
            cached = past_kv is not None and past_kv[0].numel() > 0
            incremental = bool(cached or hidden_states.shape[1] == 1)
            selected_mask, weights = self.mod_router.gate_mask(
                hidden_states, incremental=incremental
            )
            gate = (weights * selected_mask.to(weights.dtype)).unsqueeze(-1).to(hidden_states.dtype)

            mid_states = residual + gate * attn_out
            mlp_input = self.post_attn_norm(mid_states)
            if self.is_moe:
                # Gated-out tokens contribute nothing to the residual, so they
                # must not shape the router's load statistics either.
                mlp_out = self.mlp(mlp_input, token_mask=selected_mask)
            else:
                mlp_out = self.mlp(mlp_input)
            out_states = mid_states + gate * mlp_out
            return out_states, present
        else:
            hidden_states = residual + attn_out
            hidden_states = hidden_states + self.mlp(self.post_attn_norm(hidden_states))
            return hidden_states, present


# ---------------------------------------------------------------------------
# Multi-token prediction
# ---------------------------------------------------------------------------


class MultiTokenPredictionModule(nn.Module):
    """One MTP depth, in the sequential DeepSeek-V3 style.

    Depth ``k`` predicts token ``t + k + 1`` from the hidden state of depth
    ``k - 1`` combined with the *embedding of the token it already knows*:

    ``h'_i = W [ norm(h_i^{k-1}) ; norm(emb(x_{i+k})) ]`` followed by one
    transformer block. Embedding and output head are shared with the trunk, so a
    depth costs one block, not one model.

    At inference the same modules become the draft model for self-speculative
    decoding (see :mod:`mimomix_decoding`).
    """

    def __init__(self, config: MiMoMixConfig, depth: int):
        super().__init__()
        self.depth = int(depth)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        # MTP depths use global attention and a *dense* FFN. MiMo describes the
        # MTP module as lightweight with dense FFNs; a routed depth would also
        # make the draft path's cost data-dependent, which defeats the point.
        self.block = MiMoMixBlock(
            config, layer_index=config.n_layers + depth, kind="global", force_dense=True
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        shifted_embeddings: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_positions: torch.Tensor,
        attention_mask=None,
    ) -> torch.Tensor:
        merged = torch.cat(
            [self.hidden_norm(prev_hidden), self.embed_norm(shifted_embeddings)], dim=-1
        )
        hidden = self.proj(merged)
        empty_positions = query_positions.new_empty((0,))
        hidden, _ = self.block(
            hidden,
            cos,
            sin,
            query_positions,
            empty_positions,
            past_kv=None,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self.final_norm(hidden)


# ---------------------------------------------------------------------------
# Recursive thinking core (Supermix v51/v52 heritage)
# ---------------------------------------------------------------------------


class RecursiveThinkingCore(nn.Module):
    """Weight-tied latent refinement with ACT halting and a quality verifier.

    This is the sequence-model port of ``CognitiveLeapV52ExpertHead``. It runs
    on the trunk's final hidden state and:

    * refines a latent ``z`` for up to ``max_cycles`` weight-tied cycles;
    * emits a halting probability per cycle (ACT / PonderNet), producing a
      halting-weighted mixture of the per-cycle residuals plus a ponder cost;
    * carries a trainable temperature so the head's confidence is calibratable;
    * emits ``p(correct)`` and ``p(continue)`` from a supervised quality head.

    The verifier is *advisory*: it reports whether more compute is warranted.
    It does not silently mutate the returned hidden state beyond a bounded,
    near-zero-initialised residual, so a freshly built model reproduces the
    trunk output up to that residual.
    """

    def __init__(self, config: MiMoMixConfig):
        super().__init__()
        hidden = int(config.hidden_size)
        latent = int(config.thinking_latent_dim)
        self.config = config
        self.n_cycles = int(config.thinking_cycles)
        self.max_cycles = int(config.thinking_max_cycles)
        self.inner_steps = int(config.thinking_inner_steps)

        self.to_latent = nn.Sequential(RMSNorm(hidden, config.rms_norm_eps), nn.Linear(hidden, latent))
        self.cell = nn.GRUCell(latent, latent)
        self.refine = nn.Sequential(
            nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, latent)
        )
        self.halt_head = nn.Linear(latent, 1)
        self.to_residual = nn.Linear(latent, hidden, bias=False)
        self.residual_scale = nn.Parameter(
            torch.full((), float(getattr(config, "thinking_residual_init", 0.0)))
        )
        self.log_temperature = nn.Parameter(torch.zeros(()))

        self.quality_encoder = nn.Sequential(nn.Linear(latent + 3, latent), nn.GELU())
        self.quality_head = nn.Linear(latent, 2)
        self.reset_special_parameters()

        self.register_buffer("last_cycles_used", torch.zeros(()), persistent=False)
        self.register_buffer("last_ponder_cost", torch.zeros(()), persistent=False)
        self.register_buffer("last_consistency_loss", torch.zeros(()), persistent=False)
        self.register_buffer("last_quality_score", torch.full((), 0.5), persistent=False)
        self.register_buffer("last_continue_probability", torch.full((), 0.5), persistent=False)
        self.last_exit_reason = "not_run"
        self._last_quality_logits: Optional[torch.Tensor] = None

    def reset_special_parameters(self) -> None:
        """The core's deliberate initialisation, re-appliable after a sweep.

        ``quality_head`` starts at exactly zero so a fresh verifier reports
        p=0.5 and cannot bias anything before it is supervised, and
        ``to_residual`` starts at std 0.01 so the recursive residual is small
        but not degenerate. :meth:`MiMoMixModel._restore_special_inits` calls
        this after ``apply(_init_weights)``, which would otherwise replace both
        with the generic ``N(0, 0.02)``.
        """

        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)
        nn.init.normal_(self.to_residual.weight, std=0.01)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature.clamp(min=-2.3, max=2.3))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cycles: Optional[int] = None,
        adaptive: bool = False,
        halt_threshold: float = 0.95,
        stability_tol: float = 1e-3,
        stability_patience: int = 2,
        extra_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``extra_input`` (v93 thinking bond) is an optional ``(B*T, H)``
        tensor added to the trunk state *before* ``to_latent`` only. It shapes
        what the recursive core thinks about and never enters the returned
        residual directly, so the bond reaches the logits solely through the
        core's own residual path (the one whose gate opened in v89,
        ``residual_scale = -0.111``). ``None`` is byte-identical to before.
        """

        budget = self.n_cycles if cycles is None else int(cycles)
        budget = max(1, min(budget, self.max_cycles))

        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        latent_input = flat if extra_input is None else flat + extra_input.reshape(-1, shape[-1])
        z = self.to_latent(latent_input)
        state = torch.zeros_like(z)

        cumulative_halt = torch.zeros(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)
        residual_mixture = torch.zeros_like(flat)
        ponder = torch.zeros(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)
        consistency = flat.new_zeros(())
        previous_residual: Optional[torch.Tensor] = None
        stable_steps = 0
        cycles_used = 0
        exit_reason = "budget_exhausted"

        for cycle in range(budget):
            cycles_used = cycle + 1
            for _ in range(self.inner_steps):
                state = self.cell(self.refine(z), state)
            z = z + state

            halt_logit = self.halt_head(state)
            halt_p = torch.sigmoid(halt_logit)
            remaining = (1.0 - cumulative_halt).clamp_min(0.0)
            # PonderNet-style: this cycle claims halt_p of the mass still in
            # flight. Whatever is still unclaimed when the loop ends is given to
            # the final residual below, so the mixture stays convex no matter
            # which exit fired.
            step_weight = remaining * halt_p
            cumulative_halt = cumulative_halt + step_weight
            step_residual = self.to_residual(z)
            residual_mixture = residual_mixture + step_weight * step_residual
            ponder = ponder + remaining

            if previous_residual is not None:
                delta = (step_residual - previous_residual).pow(2).mean()
                consistency = consistency + delta
                if adaptive and float(delta.detach()) <= stability_tol:
                    stable_steps += 1
                    if stable_steps >= stability_patience:
                        exit_reason = "prediction_stability"
                        previous_residual = step_residual
                        break
                else:
                    stable_steps = 0
            previous_residual = step_residual

            if adaptive and float(cumulative_halt.mean().detach()) >= halt_threshold:
                exit_reason = "halting_threshold"
                break

        # Any unspent halting mass falls to the last residual so the mixture is
        # a proper convex combination regardless of where the loop exited.
        leftover = (1.0 - cumulative_halt).clamp_min(0.0)
        if previous_residual is not None:
            residual_mixture = residual_mixture + leftover * previous_residual

        scale = self.residual_scale + (1e-4 if self.training else 0.0)
        refined = flat + scale * residual_mixture

        confidence = torch.sigmoid(residual_mixture.abs().mean(dim=-1, keepdim=True))
        depth_feature = torch.full_like(confidence, float(cycles_used) / float(self.max_cycles))
        halt_feature = cumulative_halt
        quality_state = self.quality_encoder(
            torch.cat([z, confidence, depth_feature, halt_feature], dim=-1)
        )
        # Temperature scaling on the verifier's own logits. This is the one
        # calibration knob the controller trusts, and it is trained by
        # verifier_loss -- deliberately not by the language-model objective, so
        # calibrating the verifier cannot drift the token distribution.
        quality_logits = self.quality_head(quality_state) / self.temperature
        quality_probs = torch.sigmoid(quality_logits)

        self.last_cycles_used = torch.tensor(float(cycles_used), device=flat.device)
        self.last_ponder_cost = ponder.mean().detach()
        self.last_consistency_loss = (consistency / max(1, cycles_used - 1)).detach()
        self.last_quality_score = quality_probs[:, 0].mean().detach()
        self.last_continue_probability = quality_probs[:, 1].mean().detach()
        self.last_exit_reason = exit_reason
        # Retained for verifier_loss(...); cleared outside training so an
        # inference pass never pins a graph.
        self._last_quality_logits = quality_logits if self.training else None

        info = {
            "quality_logits": quality_logits,
            "quality_probability": quality_probs[:, 0],
            "continue_probability": quality_probs[:, 1],
            "ponder_cost": ponder.mean(),
            "consistency_loss": consistency / max(1, cycles_used - 1),
            "cycles_used": torch.tensor(float(cycles_used), device=flat.device),
        }
        return refined.reshape(shape), info


# ---------------------------------------------------------------------------
# v91 connectome core (Janelia FlyEM male CNS v1.0)
# ---------------------------------------------------------------------------

#: Module roles that receive the token's input / whose activity is read out
#: under ``cns_io="afferent_efferent"``. Mirrors the fly: signals enter through
#: sensory, visual and ascending populations and leave through descending and
#: motor/efferent ones.
CNS_AFFERENT_ROLES = ("sensory", "visual", "ascending")
CNS_EFFERENT_ROLES = ("descending", "output")


class ConnectomeCore(nn.Module):
    """A recurrent side branch wired like the male fly central nervous system.

    Each node is a *module*: a cluster of male-CNS cell types that share a
    functional role and a Dale sign (see ``source/malecns_connectome.py``). The
    recurrent matrix is constrained the way connectome-constrained models of
    the fly are (Lappalainen et al. 2024; Shiu et al. 2024):

    * **topology is fixed** -- only module pairs connected by at least 1% of
      the postsynaptic module's input in the real connectome can interact;
    * **signs are fixed by Dale's law** -- a module's outgoing weights all
      carry the sign of its neurotransmitter (ACh +, GABA/Glu/histamine -);
    * **magnitudes are learned**, starting from the measured input fractions,
      rescaled to a fixed spectral radius so the connectome and its rewired
      null start in the same dynamical regime.

    Per token (no mixing across positions, so KV-cached decoding is exact)::

        u   = in_mask * W_in norm(x)
        r_0 = 0
        r_k = (1 - a) r_{k-1} + a relu(W r_{k-1} + u + b),   k = 1..steps
        x'  = x + g * W_out rms_norm(out_mask * r_steps)

    ``g`` is a per-channel gate that starts at exactly zero, so grafting the
    core into a trained checkpoint changes none of its outputs until training
    opens the gate (function-preserving growth, as in ReZero). Whether the gate
    *does* open is itself a measurement -- v58's thinking-core gate reached
    only 6.4e-4 in 1,000 steps (docs/V59_MECHANISM_CAUSALITY.md) -- so the
    trainer reports it and ablates the core at the end.

    **v93 (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md, D2).** Everything below is
    default-off, and with the defaults the module's parameters, buffers,
    forward numerics and telemetry keys are those of v91:

    * *Two hemispheres in one block matrix.* ``hemisphere`` (n,) int8 comes
      from the npz's ``module_side``; ``W`` is then ``[[LL, LR], [RL, RR]]``
      and ``ablate_cross`` zeroes the off-diagonal (commissural) blocks.
    * *Capacity slots.* ``cns_nodes`` is the capacity; the last
      ``cns_spare_nodes`` slots are born dead (``alive == 0``): mask row and
      column 0, in/out masks 0, ``read_in`` row and ``read_out`` column 0, and
      their rates are forced to 0, so they are exactly inert until
      :meth:`split_module` fills one. ``born_step`` and ``edge_grown`` /
      ``tap_grown`` record provenance for the neurogenesis ablation.
    * *Multi-site taps.* The primary read site (the deepest read layer) uses
      ``norm``/``read_in`` and the primary write site (the shallowest write
      layer) ``read_out``/``gate``, so the v91 keys are unchanged; every
      additional read site gets ``extra_norms[j]``/``extra_read_in[j]``, every
      additional write site ``extra_read_out[j]``/``extra_gates[j]``, and the
      thinking bond ``to_thinking``/``thinking_gate``. Every gate is zero at
      birth, so the graft is function-preserving at every site.
    * *Temporal recurrence.* ``cns_temporal`` carries the rate across
      positions (causal scan, ``cns_steps`` inner updates per position); the
      model threads the per-position state through ``past_key_values`` so
      cached decoding stays exact.
    * *Growth primitives* (:meth:`split_module`, :meth:`grow_edge`,
      :meth:`open_tap`, :meth:`prune_edges`, :meth:`kill_module`) edit slots
      in place; no tensor ever changes shape. ``reset_special_parameters`` is
      never called after growth (it would rebuild the logits from
      ``init_fraction``, which grown edges do not have).
    * *Statistics* for growth decisions accumulate only in eval mode while
      ``collect_stats`` is set; see :meth:`growth_statistics`.
    """

    #: Persistent buffers introduced by v93. A v91 checkpoint lacks them;
    #: :meth:`_load_from_state_dict` fills each missing one with its
    #: construction-time default so the strict load still succeeds.
    _V93_BUFFERS = ("hemisphere", "alive", "born_step", "edge_grown", "tap_grown")

    def __init__(self, config: MiMoMixConfig):
        super().__init__()
        hidden = int(config.hidden_size)
        n = int(config.cns_nodes)
        self.n_nodes = n
        self.spare_nodes = int(getattr(config, "cns_spare_nodes", 0))
        self.steps = int(config.cns_steps)
        self.io = str(config.cns_io)
        self.temporal = bool(getattr(config, "cns_temporal", False))
        self.read_sites: Tuple[int, ...] = tuple(config.cns_read_sites)
        self.write_sites: Tuple[int, ...] = tuple(config.cns_write_sites)
        self.norm = RMSNorm(hidden, config.rms_norm_eps)
        self.read_in = nn.Linear(hidden, n, bias=False)
        self.read_out = nn.Linear(n, hidden, bias=False)
        self.edge_logit = nn.Parameter(torch.full((n, n), -10.0))
        self.node_bias = nn.Parameter(torch.zeros(n))
        self.leak_logit = nn.Parameter(torch.zeros(n))
        self.gate = nn.Parameter(torch.zeros(hidden))
        # The graph. Buffers, so a checkpoint carries its own wiring.
        self.register_buffer("mask", torch.zeros(n, n), persistent=True)
        self.register_buffer("sign", torch.ones(n), persistent=True)
        self.register_buffer("in_mask", torch.ones(n), persistent=True)
        self.register_buffer("out_mask", torch.ones(n), persistent=True)
        self.register_buffer("init_fraction", torch.zeros(n, n), persistent=True)
        self.register_buffer("spectral_scale", torch.ones(()), persistent=True)
        self.register_buffer("last_out_activity", torch.zeros(()), persistent=False)
        self.register_buffer("last_active_fraction", torch.zeros(()), persistent=False)
        # v93 slot provenance. Spare slots sit at the END of the capacity.
        alive = torch.ones(n, dtype=torch.uint8)
        if self.spare_nodes > 0:
            alive[n - self.spare_nodes:] = 0
            with torch.no_grad():
                self.in_mask[n - self.spare_nodes:] = 0.0
                self.out_mask[n - self.spare_nodes:] = 0.0
        self.register_buffer("hemisphere", torch.zeros(n, dtype=torch.int8), persistent=True)
        self.register_buffer("alive", alive, persistent=True)
        self.register_buffer("born_step", torch.zeros(n, dtype=torch.int32), persistent=True)
        self.register_buffer("edge_grown", torch.zeros(n, n, dtype=torch.uint8), persistent=True)
        # tap_grown[:, 0] = afferent tap opened by growth, [:, 1] = efferent.
        self.register_buffer("tap_grown", torch.zeros(n, 2, dtype=torch.uint8), persistent=True)
        # v93 additional sites. Empty containers add no state_dict keys.
        self.extra_norms = nn.ModuleList(
            [RMSNorm(hidden, config.rms_norm_eps) for _ in self.read_sites[:-1]]
        )
        self.extra_read_in = nn.ModuleList(
            [nn.Linear(hidden, n, bias=False) for _ in self.read_sites[:-1]]
        )
        self.extra_read_out = nn.ModuleList(
            [nn.Linear(n, hidden, bias=False) for _ in self.write_sites[1:]]
        )
        self.extra_gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden)) for _ in self.write_sites[1:]]
        )
        if bool(getattr(config, "cns_to_thinking", False)):
            self.to_thinking = nn.Linear(n, hidden, bias=False)
            self.thinking_gate = nn.Parameter(torch.zeros(hidden))
        else:
            self.to_thinking = None
            self.thinking_gate = None
        # v93 growth statistics (eval mode only, while collect_stats is set).
        self.collect_stats = False
        self.register_buffer("stat_rate_sum", torch.zeros(n), persistent=False)
        self.register_buffer("stat_rate_sq_sum", torch.zeros(n), persistent=False)
        self.register_buffer("stat_rate_cov_sum", torch.zeros(n, n), persistent=False)
        self.register_buffer("stat_rate_resid_sum", torch.zeros(n), persistent=False)
        self.register_buffer("stat_resid_sum", torch.zeros(()), persistent=False)
        self.register_buffer("stat_resid_sq_sum", torch.zeros(()), persistent=False)
        self.register_buffer("stat_tokens", torch.zeros(()), persistent=False)
        # v93 analysis switches (plain attributes: never saved, default off).
        self.ablate_cross = False
        self.ablate_side: Optional[int] = None
        self.ablate_grown = False
        #: What the most recent growth primitive changed, for the event log.
        self.last_event: Dict[str, object] = {}
        self.graph_info: Dict[str, object] = {"loaded": False}

    # -- v93 compatibility ---------------------------------------------------

    @property
    def single_site(self) -> bool:
        """True when the core is exactly v91's: one read site that is also the
        only write site, per-token dynamics, no thinking bond. The model then
        calls :meth:`forward` as v91 did (which also keeps
        ``v91_analysis.CoreMode``'s forward monkeypatch working)."""

        return (
            len(self.read_sites) == 1
            and self.write_sites == self.read_sites
            and not self.temporal
            and self.to_thinking is None
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Fill v93 buffers missing from a v91 checkpoint with their defaults.

        ``nn.Module.load_state_dict`` hands each module a copy of the payload,
        so adding keys here is invisible to the caller and strict mode reports
        nothing. Defaults: ``hemisphere`` 0, ``alive`` 1 for every slot the
        config says is live (all of them for a v91 config, which has no
        spares), ``born_step`` 0, ``edge_grown`` / ``tap_grown`` 0.
        """

        for name in self._V93_BUFFERS:
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset_special_parameters(self) -> None:
        """Gates at exactly zero; edges at their measured (rescaled) strengths.

        Re-applied by :meth:`MiMoMixModel._restore_special_inits` after the
        blanket ``N(0, 0.02)`` sweep, and again by :meth:`load_graph`. Dead
        slots also get their ``read_in`` row and ``read_out`` column zeroed.
        Never call this after growth: grown edges have ``init_fraction`` 0 and
        would be reset to softplus^-1(1e-6).
        """

        with torch.no_grad():
            self.gate.zero_()
            self.node_bias.zero_()
            self.leak_logit.zero_()
            for gate in self.extra_gates:
                gate.zero_()
            if self.thinking_gate is not None:
                self.thinking_gate.zero_()
            target = (self.init_fraction * self.spectral_scale).clamp_min(1e-6)
            # softplus^-1(y) = log(expm1(y)); masked entries sit far below zero.
            inverse = torch.log(torch.expm1(target))
            self.edge_logit.copy_(torch.where(self.mask > 0, inverse, torch.full_like(inverse, -10.0)))
            dead = self.alive == 0
            if bool(dead.any()):
                self._zero_slot_taps(dead)

    @torch.no_grad()
    def _zero_slot_taps(self, slots: torch.Tensor) -> None:
        """Zero every read-in row and read-out column of the boolean ``slots``."""

        self.read_in.weight[slots] = 0.0
        self.read_out.weight[:, slots] = 0.0
        for tap in self.extra_read_in:
            tap.weight[slots] = 0.0
        for tap in self.extra_read_out:
            tap.weight[:, slots] = 0.0
        if self.to_thinking is not None:
            self.to_thinking.weight[:, slots] = 0.0

    @torch.no_grad()
    def load_graph(self, path: str, wiring: str = "connectome", spectral_radius: float = 0.9) -> Dict[str, object]:
        """Install a module graph written by ``malecns_connectome.py modules``.

        ``wiring="rewired"`` installs the degree-preserving Maslov-Sneppen null
        from the same file: same modules, same signs, same in/out degrees, same
        per-edge strengths (carried by the presynaptic end), same self-loops --
        only *which* module talks to which differs.

        v93: the npz must carry exactly ``cns_nodes - cns_spare_nodes``
        modules; they fill the leading slots and the spares stay dead. A
        ``module_side`` array (the ``hemispheres`` command) sets
        ``hemisphere``; a v91 file leaves it 0. The spectral scaling is
        computed over the live block only, exactly as v91 computed it over
        the whole (then all-live) matrix.
        """

        import numpy as np

        data = np.load(path, allow_pickle=True)
        n = int(len(data["module_sign"]))
        live = self.n_nodes - self.spare_nodes
        if n != live:
            if self.spare_nodes == 0:
                raise ValueError(f"{path} has {n} modules but cns_nodes={self.n_nodes}")
            raise ValueError(
                f"{path} has {n} modules but cns_nodes={self.n_nodes} minus "
                f"cns_spare_nodes={self.spare_nodes} leaves {live} live slots"
            )
        fraction = data["edge_fraction"].astype(np.float64)
        if wiring == "connectome":
            post, pre = data["edge_post"], data["edge_pre"]
        elif wiring == "rewired":
            post, pre = data["rewired_post"], data["rewired_pre"]
            # The stratified null (malecns_connectome.stratified_rewire) moves
            # each fraction with its target module; files written before it
            # have no rewired_fraction and keep the fraction on the edge.
            if "rewired_fraction" in data.files:
                fraction = data["rewired_fraction"].astype(np.float64)
        else:
            raise ValueError(f"unknown wiring {wiring!r}")
        dense = np.zeros((n, n), dtype=np.float64)
        dense[post, pre] = fraction
        mask = (dense > 0).astype(np.float64)
        sign = data["module_sign"].astype(np.float64)
        # Match the dynamical regime across wirings: scale |W| to a common
        # spectral radius. Row sums of the real graph are <= 1 by construction
        # but the rewired graph's are not, so without this the null would start
        # in a different (possibly unstable) regime and the comparison would
        # measure that instead of the wiring.
        rho = float(np.max(np.abs(np.linalg.eigvals(dense))))
        scale = spectral_radius / rho if rho > 0 else 1.0
        roles = [str(r) for r in data["module_role"]]
        if self.io == "afferent_efferent":
            in_mask = np.array([r in CNS_AFFERENT_ROLES for r in roles], dtype=np.float64)
            out_mask = np.array([r in CNS_EFFERENT_ROLES for r in roles], dtype=np.float64)
        else:
            in_mask = np.ones(n)
            out_mask = np.ones(n)
        capacity = self.n_nodes
        full_mask = np.zeros((capacity, capacity), dtype=np.float64)
        full_mask[:n, :n] = mask
        full_dense = np.zeros((capacity, capacity), dtype=np.float64)
        full_dense[:n, :n] = dense
        full_sign = np.ones(capacity, dtype=np.float64)
        full_sign[:n] = sign
        full_in = np.zeros(capacity, dtype=np.float64)
        full_in[:n] = in_mask
        full_out = np.zeros(capacity, dtype=np.float64)
        full_out[:n] = out_mask
        self.mask.copy_(torch.from_numpy(full_mask))
        self.sign.copy_(torch.from_numpy(full_sign))
        self.in_mask.copy_(torch.from_numpy(full_in))
        self.out_mask.copy_(torch.from_numpy(full_out))
        self.init_fraction.copy_(torch.from_numpy(full_dense))
        self.spectral_scale.fill_(scale)
        self.alive.zero_()
        self.alive[:n] = 1
        self.born_step.zero_()
        self.edge_grown.zero_()
        self.tap_grown.zero_()
        self.hemisphere.zero_()
        sides: Optional[np.ndarray] = None
        if "module_side" in data.files:
            sides = np.asarray(data["module_side"]).astype(np.int64)
            if sides.shape != (n,):
                raise ValueError(f"{path}: module_side has shape {sides.shape}, expected ({n},)")
            self.hemisphere[:n] = torch.from_numpy(sides.astype(np.int8))
        self.reset_special_parameters()
        self.graph_info = {
            "loaded": True,
            "path": str(path),
            "wiring": wiring,
            "nodes": n,
            "edges": int(mask.sum()),
            "self_loops": int(np.trace(mask)),
            "excitatory_nodes": int((sign > 0).sum()),
            "inhibitory_nodes": int((sign < 0).sum()),
            "input_nodes": int(in_mask.sum()),
            "output_nodes": int(out_mask.sum()),
            "unscaled_spectral_radius": round(rho, 6),
            "spectral_scale": round(scale, 6),
            "target_spectral_radius": spectral_radius,
        }
        if self.spare_nodes > 0:
            self.graph_info["capacity"] = int(capacity)
            self.graph_info["spare_nodes"] = int(self.spare_nodes)
        if sides is not None:
            self.graph_info["hemisphere_nodes"] = [int((sides == 0).sum()), int((sides == 1).sum())]
            self.graph_info["edges_by_block_at_load"] = self._edges_by_block()
        return dict(self.graph_info)

    # -- the dynamics ---------------------------------------------------------

    def _active(self) -> Optional[torch.Tensor]:
        """Float ``(n,)`` vector of modules that take part, or ``None`` when all
        do (the v91 path: no extra multiply at all)."""

        active = self.alive
        if self.ablate_grown:
            active = active * (self.born_step == 0).to(active.dtype)
        if self.ablate_side is not None:
            active = active * (self.hemisphere != int(self.ablate_side)).to(active.dtype)
        if bool(active.all()):
            return None
        return active.to(dtype=torch.float32)

    def _tap_masks(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Effective ``(in_mask, out_mask, active)`` after slots and ablations."""

        in_mask, out_mask = self.in_mask, self.out_mask
        if self.ablate_grown:
            in_mask = in_mask * (1 - self.tap_grown[:, 0].to(in_mask.dtype))
            out_mask = out_mask * (1 - self.tap_grown[:, 1].to(out_mask.dtype))
        active = self._active()
        if active is not None:
            in_mask = in_mask * active
            out_mask = out_mask * active
        return in_mask.to(dtype), out_mask.to(dtype), active

    def weight(self) -> torch.Tensor:
        """Signed recurrent matrix ``W[post, pre]`` (Dale sign per column).

        ``ablate_grown`` removes every grown edge; ``ablate_cross`` removes
        the commissural (LR and RL) blocks. Neither touches the parameters.
        """

        mask = self.mask
        if self.ablate_grown:
            mask = mask * (1 - self.edge_grown.to(mask.dtype))
        if self.ablate_cross:
            same_side = self.hemisphere.unsqueeze(1) == self.hemisphere.unsqueeze(0)
            mask = mask * same_side.to(mask.dtype)
        return mask * F.softplus(self.edge_logit) * self.sign.unsqueeze(0)

    def drive(self, hiddens: Sequence[torch.Tensor]) -> torch.Tensor:
        """Afferent drive ``(B*T, n)`` from the read-site hidden states.

        ``hiddens`` are the trunk states after each read layer, shallowest
        first; the last is the primary site (``norm``/``read_in``), the others
        go through ``extra_norms[j]``/``extra_read_in[j]``.
        """

        if len(hiddens) != len(self.read_sites):
            raise ValueError(
                f"core reads {len(self.read_sites)} sites {self.read_sites} but got {len(hiddens)} hidden states"
            )
        primary = hiddens[-1]
        flat = primary.reshape(-1, primary.shape[-1])
        in_mask, _, _ = self._tap_masks(flat.dtype)
        total = self.read_in(self.norm(flat)) * in_mask
        for index, hidden in enumerate(hiddens[:-1]):
            tap_flat = hidden.reshape(-1, hidden.shape[-1])
            total = total + self.extra_read_in[index](self.extra_norms[index](tap_flat)) * in_mask
        return total + self.node_bias

    def _per_token_rates(
        self, drive: torch.Tensor, w_t: torch.Tensor, alpha: torch.Tensor, active: Optional[torch.Tensor]
    ) -> torch.Tensor:
        rate = torch.zeros_like(drive)
        for _ in range(self.steps):
            rate = (1.0 - alpha) * rate + alpha * F.relu(rate @ w_t + drive)
            if active is not None:
                rate = rate * active
        return rate

    def _temporal_rates(
        self,
        drive: torch.Tensor,
        state: Optional[torch.Tensor],
        w_t: torch.Tensor,
        alpha: torch.Tensor,
        active: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Causal scan over ``drive`` ``(B, T, n)`` from ``state`` ``(B, n)``.

        Returns the state after every position, ``(B, T, n)``. Position ``t``
        sees positions ``< t`` only through the carried state, so a cached
        decode that resumes from the last state reproduces the full scan.
        """

        bsz, seq_len, n = drive.shape
        if state is None:
            state = drive.new_zeros(bsz, n)
        else:
            state = state.to(dtype=drive.dtype)
        states: List[torch.Tensor] = []
        for t in range(seq_len):
            u = drive[:, t]
            for _ in range(self.steps):
                state = (1.0 - alpha) * state + alpha * F.relu(state @ w_t + u)
                if active is not None:
                    state = state * active
            states.append(state)
        return torch.stack(states, dim=1)

    def rates(self, hiddens: Sequence[torch.Tensor], state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Raw module rates ``(B*T, n)`` (per-token) or ``(B, T, n)`` (temporal)."""

        drive = self.drive(hiddens)
        w_t = self.weight().to(drive.dtype).t()
        alpha = torch.sigmoid(self.leak_logit).to(drive.dtype)
        _, _, active = self._tap_masks(drive.dtype)
        if self.temporal:
            primary = hiddens[-1]
            bsz, seq_len = int(primary.shape[0]), int(primary.shape[1])
            return self._temporal_rates(drive.view(bsz, seq_len, self.n_nodes), state, w_t, alpha, active)
        return self._per_token_rates(drive, w_t, alpha, active)

    def readout(self, rate: torch.Tensor) -> torch.Tensor:
        """RMS-normalised efferent read ``(B*T, n)`` from raw rates ``(B*T, n)``."""

        _, out_mask, _ = self._tap_masks(rate.dtype)
        read = rate * out_mask
        n_out = out_mask.sum().clamp_min(1.0)
        self.last_out_activity = read.detach().abs().sum(-1).mean() / n_out
        self.last_active_fraction = (rate.detach() > 0).float().mean()
        # RMS-normalise over the read-out modules only. Measured on the v89
        # graft, 6 steps through the real wiring deliver a mean efferent rate
        # of ~0.01 against a drive of ~0.36, so an unnormalised read-out hands
        # the zero-initialised gate a gradient ~30x too small to ever open.
        read = read * torch.rsqrt(read.pow(2).sum(-1, keepdim=True) / n_out + 1e-6)
        return read

    def write(self, site: int, hidden_states: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
        """``hidden + gate_site * read_out_site(read)`` for write site ``site``
        (an index into ``write_sites``; 0 is the primary ``read_out``/``gate``)."""

        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        if site == 0:
            gate, projection = self.gate, self.read_out
        else:
            gate, projection = self.extra_gates[site - 1], self.extra_read_out[site - 1]
        out = flat + gate.to(flat.dtype) * projection(read)
        return out.reshape(shape)

    def thinking_input(self, read: torch.Tensor) -> Optional[torch.Tensor]:
        """``thinking_gate * to_thinking(read)`` ``(B*T, H)``, or ``None`` without the bond."""

        if self.to_thinking is None:
            return None
        return self.thinking_gate.to(read.dtype) * self.to_thinking(read)

    def compute(
        self,
        hiddens: Sequence[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        keep_states: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the core once and return ``(read, state_history)``.

        ``read`` is the normalised efferent read ``(B*T, n)`` that every write
        site and the thinking bond consume. ``state`` is the carried rate
        ``(B, n)`` for the temporal core (``None`` starts from rest).
        ``state_history`` is ``None`` for the per-token core; for the temporal
        core it is the last ``keep_states`` per-position states ``(B, T', n)``
        (all of them when ``keep_states`` is ``None``), which the model stores
        in ``past_key_values``.
        """

        rate = self.rates(hiddens, state)
        history: Optional[torch.Tensor] = None
        if self.temporal:
            history = rate
            if keep_states is not None and history.shape[1] > int(keep_states):
                history = history[:, -int(keep_states):]
            rate = rate.reshape(-1, self.n_nodes)
        read = self.readout(rate)
        if self.collect_stats and not self.training:
            primary = hiddens[-1]
            self._accumulate_stats(rate, primary.reshape(-1, primary.shape[-1]))
        return read, history

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The v91 single-site graft: ``x + gate * read_out(read(x))``.

        Valid only for a single-site core (``single_site``); the model drives
        a multi-site or temporal core through :meth:`compute`/:meth:`write`.
        """

        if len(self.read_sites) != 1:
            raise RuntimeError(
                "ConnectomeCore.forward is the single-site path; a core with "
                f"read sites {self.read_sites} is driven through compute()/write()"
            )
        read, _ = self.compute([hidden_states])
        return self.write(0, hidden_states, read)

    # -- v93 growth statistics ------------------------------------------------

    @torch.no_grad()
    def _accumulate_stats(self, rate: torch.Tensor, primary_flat: torch.Tensor) -> None:
        r = rate.detach().float()
        self.stat_rate_sum += r.sum(0)
        self.stat_rate_sq_sum += (r * r).sum(0)
        self.stat_rate_cov_sum += r.t() @ r
        resid = primary_flat.detach().float().pow(2).mean(-1).sqrt()  # trunk residual RMS per token
        self.stat_rate_resid_sum += (r * resid.unsqueeze(1)).sum(0)
        self.stat_resid_sum += resid.sum()
        self.stat_resid_sq_sum += (resid * resid).sum()
        self.stat_tokens += float(r.shape[0])

    def reset_stats(self) -> None:
        for name in (
            "stat_rate_sum", "stat_rate_sq_sum", "stat_rate_cov_sum", "stat_rate_resid_sum",
            "stat_resid_sum", "stat_resid_sq_sum", "stat_tokens",
        ):
            getattr(self, name).zero_()

    def growth_statistics(self) -> Dict[str, object]:
        """Moments accumulated over dev tokens for the neurogenesis controller.

        ``rate_cov`` is ``E[r r^T] - E[r] E[r]^T`` ``(n, n)`` (entry
        ``[post, pre]`` is the pair's covariance; symmetric), ``resid_cov`` is
        ``E[r s] - E[r] E[s]`` ``(n,)`` against the trunk residual RMS ``s`` at
        the primary read site, for tap decisions. Dead slots read 0.
        """

        tokens = float(self.stat_tokens)
        denom = max(tokens, 1.0)
        mean = self.stat_rate_sum / denom
        var = self.stat_rate_sq_sum / denom - mean * mean
        cov = self.stat_rate_cov_sum / denom - torch.outer(mean, mean)
        resid_mean = self.stat_resid_sum / denom
        resid_var = self.stat_resid_sq_sum / denom - resid_mean * resid_mean
        resid_cov = self.stat_rate_resid_sum / denom - mean * resid_mean
        return {
            "tokens": int(tokens),
            "mean_rate": mean.detach().clone(),
            "rate_var": var.detach().clone(),
            "rate_cov": cov.detach().clone(),
            "resid_cov": resid_cov.detach().clone(),
            "resid_mean": float(resid_mean),
            "resid_var": float(resid_var),
            "alive": self.alive.detach().clone().bool(),
        }

    # -- v93 growth primitives (D6 items 2-4) ---------------------------------

    def free_slots(self) -> List[int]:
        """Indices of dead slots, lowest first."""

        return torch.nonzero(self.alive == 0, as_tuple=False).flatten().tolist()

    @staticmethod
    def _halve_logits(logits: torch.Tensor) -> torch.Tensor:
        """softplus^-1(softplus(x) / 2): the logit whose strength is half."""

        return torch.log(torch.expm1((F.softplus(logits) / 2.0).clamp_min(1e-6)))

    @torch.no_grad()
    def split_module(self, parent: int, step: int) -> Optional[int]:
        """Module mitosis: fill a free slot with a child of ``parent``.

        The child copies the parent's incoming edges (and their grown flags),
        tap masks, sign, hemisphere, leak, bias and ``read_in`` rows. Parent
        and child then each keep HALF the parent's outgoing strengths and half
        its ``read_out`` columns (including the thinking bond), so at the
        split instant every other module's post-synaptic drive and the
        unnormalised read-out are conserved: the twins start with identical
        rates (identical rows, including the halved self-loop split across
        both columns), and ``W[i,p]/2 r + W[i,p]/2 r == W[i,p] r``. The
        read-out RMS normaliser's module count changes by one when the parent
        is an efferent module, which is the only non-exact part -- the
        controller measures it on the witness batch (D6 item 2).

        Returns the child's index, or ``None`` when ``parent`` is dead or no
        slot is free. ``last_event`` records what changed.
        """

        parent = int(parent)
        if not (0 <= parent < self.n_nodes) or int(self.alive[parent]) == 0:
            self.last_event = {"event": "split_module", "parent": parent, "child": None, "refused": "parent dead"}
            return None
        free = self.free_slots()
        if not free:
            self.last_event = {"event": "split_module", "parent": parent, "child": None, "refused": "no free slot"}
            return None
        child = int(free[0])
        # (1) the child's incoming row is the parent's, including the parent's
        #     self-loop, which lands at column `parent`.
        self.mask[child] = self.mask[parent]
        self.edge_logit[child] = self.edge_logit[parent]
        self.init_fraction[child] = self.init_fraction[parent]
        self.edge_grown[child] = self.edge_grown[parent]
        # (2) halve the parent's outgoing column where an edge exists.
        has_edge = self.mask[:, parent] > 0
        self.edge_logit[:, parent] = torch.where(
            has_edge, self._halve_logits(self.edge_logit[:, parent]), self.edge_logit[:, parent]
        )
        self.init_fraction[:, parent] = self.init_fraction[:, parent] / 2.0
        # (3) the child's outgoing column is the (halved) parent column; after
        #     (1) that column already holds the child's copy of the self-loop,
        #     so W[c, c] == W[p, c] == W[c, p] == W[p, p] == old W[p, p] / 2.
        self.mask[:, child] = self.mask[:, parent]
        self.edge_logit[:, child] = self.edge_logit[:, parent]
        self.init_fraction[:, child] = self.init_fraction[:, parent]
        self.edge_grown[:, child] = self.edge_grown[:, parent]
        # node attributes
        self.sign[child] = self.sign[parent]
        self.hemisphere[child] = self.hemisphere[parent]
        self.in_mask[child] = self.in_mask[parent]
        self.out_mask[child] = self.out_mask[parent]
        self.tap_grown[child] = self.tap_grown[parent]
        self.node_bias[child] = self.node_bias[parent]
        self.leak_logit[child] = self.leak_logit[parent]
        self.alive[child] = 1
        self.born_step[child] = int(step)
        # taps: afferent rows copied, efferent columns halved and shared
        self.read_in.weight[child] = self.read_in.weight[parent]
        for tap in self.extra_read_in:
            tap.weight[child] = tap.weight[parent]
        for projection in [self.read_out, *self.extra_read_out] + (
            [self.to_thinking] if self.to_thinking is not None else []
        ):
            projection.weight[:, parent] = projection.weight[:, parent] / 2.0
            projection.weight[:, child] = projection.weight[:, parent]
        self.last_event = {
            "event": "split_module", "parent": parent, "child": child, "step": int(step),
            "in_edges": int(self.mask[child].sum()), "out_edges": int(self.mask[:, child].sum()),
            "hemisphere": int(self.hemisphere[child]),
            "afferent": bool(self.in_mask[child] > 0), "efferent": bool(self.out_mask[child] > 0),
        }
        return child

    @torch.no_grad()
    def grow_edge(self, post: int, pre: int, step: int, logit: float = -7.0) -> bool:
        """Synaptogenesis: open the edge ``pre -> post`` at ``logit``.

        The edge's sign is the presynaptic module's Dale sign (``weight()``
        applies it per column), so every pair is Dale-legal. Refused
        (``False``) for an existing edge, a dead endpoint or a self-loop --
        a grown self-loop on an excitatory module is unbounded positive
        feedback, which the real graph's self-loops were measured to avoid.
        The default logit -7 is a strength of 9.1e-4 (D6 item 3).
        """

        post, pre = int(post), int(pre)
        if post == pre or not (0 <= post < self.n_nodes and 0 <= pre < self.n_nodes):
            return False
        if int(self.alive[post]) == 0 or int(self.alive[pre]) == 0 or float(self.mask[post, pre]) > 0:
            return False
        self.mask[post, pre] = 1.0
        self.edge_logit[post, pre] = float(logit)
        self.edge_grown[post, pre] = 1
        self.last_event = {
            "event": "grow_edge", "post": post, "pre": pre, "step": int(step), "logit": float(logit),
            "strength": float(F.softplus(torch.tensor(float(logit)))),
            "sign": int(self.sign[pre]),
            "block": self._block_name(int(self.hemisphere[post]), int(self.hemisphere[pre])),
        }
        return True

    @torch.no_grad()
    def open_tap(self, module: int, kind: str) -> bool:
        """Give ``module`` an afferent (``"in"``) or efferent (``"out"``) tap.

        The corresponding ``read_in`` row (every read site) or ``read_out``
        column (every write site and the thinking bond) is zeroed, so the new
        tap contributes nothing until trained. An efferent tap also raises the
        read-out RMS normaliser's module count by one, which rescales the
        existing read by ``sqrt(n_out / (n_out + 1))`` -- small (0.5% at 100
        efferent modules) but not exact; D6 logs the witness delta.
        Refused (``False``) when the module is dead or already has the tap.
        """

        module = int(module)
        if kind not in {"in", "out"}:
            raise ValueError(f"kind must be 'in' or 'out', got {kind!r}")
        if not (0 <= module < self.n_nodes) or int(self.alive[module]) == 0:
            return False
        if kind == "in":
            if float(self.in_mask[module]) > 0:
                return False
            self.in_mask[module] = 1.0
            self.read_in.weight[module] = 0.0
            for tap in self.extra_read_in:
                tap.weight[module] = 0.0
            self.tap_grown[module, 0] = 1
        else:
            if float(self.out_mask[module]) > 0:
                return False
            self.out_mask[module] = 1.0
            self.read_out.weight[:, module] = 0.0
            for tap in self.extra_read_out:
                tap.weight[:, module] = 0.0
            if self.to_thinking is not None:
                self.to_thinking.weight[:, module] = 0.0
            self.tap_grown[module, 1] = 1
        self.last_event = {"event": "open_tap", "module": module, "kind": kind}
        return True

    @torch.no_grad()
    def prune_edges(self, threshold: float, grown_only: bool = False) -> int:
        """Apoptosis of edges whose strength ``softplus(logit)`` fell below
        ``threshold``. ``grown_only`` spares the real connectome edges. Pruned
        entries return to the unmasked state (mask 0, logit -10, grown 0).
        Returns how many were removed; ``last_event`` splits the count into
        grown and real edges.
        """

        strength = F.softplus(self.edge_logit)
        weak = (self.mask > 0) & (strength < float(threshold))
        grown = self.edge_grown.bool()
        if grown_only:
            weak = weak & grown
        grown_pruned = int((weak & grown).sum())
        count = int(weak.sum())
        if count:
            self.mask[weak] = 0.0
            self.edge_logit[weak] = -10.0
            self.edge_grown[weak] = 0
            self.init_fraction[weak] = 0.0
        self.last_event = {
            "event": "prune_edges", "threshold": float(threshold), "grown_only": bool(grown_only),
            "pruned": count, "grown_pruned": grown_pruned, "real_pruned": count - grown_pruned,
        }
        return count

    @torch.no_grad()
    def kill_module(self, index: int) -> bool:
        """Return slot ``index`` to the dead state: every edge in and out
        removed, taps closed, tap rows/columns zeroed, provenance cleared.
        Refused (``False``) when already dead."""

        index = int(index)
        if not (0 <= index < self.n_nodes) or int(self.alive[index]) == 0:
            return False
        in_edges, out_edges = int(self.mask[index].sum()), int(self.mask[:, index].sum())
        self.alive[index] = 0
        self.born_step[index] = 0
        self.hemisphere[index] = 0
        self.mask[index] = 0.0
        self.mask[:, index] = 0.0
        self.edge_logit[index] = -10.0
        self.edge_logit[:, index] = -10.0
        self.edge_grown[index] = 0
        self.edge_grown[:, index] = 0
        self.init_fraction[index] = 0.0
        self.init_fraction[:, index] = 0.0
        self.in_mask[index] = 0.0
        self.out_mask[index] = 0.0
        self.tap_grown[index] = 0
        self.sign[index] = 1.0
        self.node_bias[index] = 0.0
        self.leak_logit[index] = 0.0
        slot = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.alive.device)
        slot[index] = True
        self._zero_slot_taps(slot)
        self.last_event = {"event": "kill_module", "module": index, "in_edges": in_edges, "out_edges": out_edges}
        return True

    # -- telemetry -------------------------------------------------------------

    @staticmethod
    def _block_name(post_side: int, pre_side: int) -> str:
        return ("L" if post_side == 0 else "R") + ("L" if pre_side == 0 else "R")

    def _edges_by_block(self) -> Dict[str, int]:
        """Installed edges per hemisphere block, keyed ``LL``/``RR``/``LR``/``RL``
        (post side first: ``LR`` is a right module driving a left one)."""

        mask = self.mask > 0
        side = self.hemisphere.to(torch.int64)
        post_left = (side == 0).unsqueeze(1)
        pre_left = (side == 0).unsqueeze(0)
        return {
            "LL": int((mask & post_left & pre_left).sum()),
            "RR": int((mask & ~post_left & ~pre_left).sum()),
            "LR": int((mask & post_left & ~pre_left).sum()),
            "RL": int((mask & ~post_left & pre_left).sum()),
        }

    def telemetry(self) -> Dict[str, object]:
        alive = self.alive.bool()
        snapshot: Dict[str, object] = {
            **self.graph_info,
            # Read from the buffers, so this is right for a model restored from
            # a checkpoint (where load_graph never ran) as well as a fresh one.
            "edges_installed": int(self.mask.sum()),
            "gate_mean_abs": float(self.gate.detach().abs().mean()),
            "gate_max_abs": float(self.gate.detach().abs().max()),
            "out_activity": float(self.last_out_activity),
            "active_fraction": float(self.last_active_fraction),
            "mean_leak": float(torch.sigmoid(self.leak_logit.detach()).mean()),
            # v93
            "alive_nodes": int(alive.sum()),
            "dead_nodes": int((~alive).sum()),
            "grown_nodes": int(((self.born_step > 0) & alive).sum()),
            "grown_edges": int(self.edge_grown.sum()),
            "edges_by_block": self._edges_by_block(),
            "hemisphere_nodes": [
                int(((self.hemisphere == 0) & alive).sum()), int(((self.hemisphere == 1) & alive).sum()),
            ],
            "taps_in": int(((self.in_mask > 0) & alive).sum()),
            "taps_out": int(((self.out_mask > 0) & alive).sum()),
            "taps_in_grown": int((self.tap_grown[:, 0] > 0).sum()),
            "taps_out_grown": int((self.tap_grown[:, 1] > 0).sum()),
            "read_layers": list(self.read_sites),
            "write_layers": list(self.write_sites),
            "temporal": bool(self.temporal),
            "extra_gates": [
                {
                    "layer": int(layer),
                    "gate_mean_abs": float(gate.detach().abs().mean()),
                    "gate_max_abs": float(gate.detach().abs().max()),
                }
                for layer, gate in zip(self.write_sites[1:], self.extra_gates)
            ],
        }
        if self.thinking_gate is not None:
            snapshot["thinking_gate_mean_abs"] = float(self.thinking_gate.detach().abs().mean())
            snapshot["thinking_gate_max_abs"] = float(self.thinking_gate.detach().abs().max())
        if self.ablate_cross or self.ablate_grown or self.ablate_side is not None:
            snapshot["ablation"] = {
                "cross": bool(self.ablate_cross), "grown": bool(self.ablate_grown),
                "side": None if self.ablate_side is None else int(self.ablate_side),
            }
        return snapshot


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class MiMoMixOutput:
    """Structured forward result. ``telemetry`` is always JSON-safe."""

    logits: torch.Tensor
    hidden_states: torch.Tensor
    #: trunk output *before* the thinking core and final norm -- the state the
    #: MTP depths were trained to consume, and therefore the draft seed
    trunk_hidden: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    mtp_logits: List[torch.Tensor] = field(default_factory=list)
    past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    telemetry: Dict[str, object] = field(default_factory=dict)


class MultimodalProjectionHead(nn.Module):
    """MiMo-V2.5 style projection from raw multimodal features into model hidden_size.

    Maps continuous image or audio token embeddings into the shared sequence
    space, normalising with RMSNorm and an expansion-contraction MLP.
    """

    def __init__(self, input_dim: int, hidden_size: int, modality: str = "vision"):
        super().__init__()
        self.modality = modality
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            RMSNorm(hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Projects (batch, num_tokens, input_dim) -> (batch, num_tokens, hidden_size)."""
        return self.proj(features)


class MiMoMixModel(nn.Module):
    """Decoder-only transformer with the full MiMoMix stack."""

    def __init__(self, config: MiMoMixConfig):
        super().__init__()
        self.config = config
        self.layout = attention_layout(
            config.n_layers,
            config.hybrid_ratio,
            config.final_layer_global,
            getattr(config, "global_layers", None),
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Two rotary tables. Global layers carry the context-extension policy;
        # local layers keep their own (usually smaller, usually unscaled) base,
        # because a sliding window cannot express a dependency longer than
        # itself and extending its frequencies only distorts what it can see.
        self.rotary = RotaryEmbedding(config, kind="global")
        self.rotary_local = RotaryEmbedding(config, kind="swa")
        # v93 depth growth: the last `grow_layers` blocks are appended identity
        # blocks and are always dense (never MoE / MoD). 0 changes nothing.
        first_grown = int(config.n_layers) - int(getattr(config, "grow_layers", 0))
        self.layers = nn.ModuleList(
            [
                MiMoMixBlock(config, i, kind, force_dense=(i >= first_grown))
                for i, kind in enumerate(self.layout)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.mtp_modules = nn.ModuleList(
            [MultiTokenPredictionModule(config, depth) for depth in range(config.n_mtp_layers)]
        )
        self.thinking_core = RecursiveThinkingCore(config) if config.use_thinking_core else None
        # Built empty; the trainer installs the wiring with load_graph(), and a
        # checkpoint restores it from the buffers.
        self.cns_core = ConnectomeCore(config) if getattr(config, "use_cns_core", False) else None

        self.multimodal_projector = (
            MultimodalProjectionHead(
                input_dim=getattr(config, "multimodal_input_dim", 128),
                hidden_size=config.hidden_size,
            )
            if getattr(config, "use_multimodal", False)
            else None
        )

        # Propagate each layer's own RoPE attention-temperature policy. MTP
        # depths are global, so they take the global scaling.
        for module in self.modules():
            if isinstance(module, (HybridAttention, DifferentialHybridAttention, MultiLatentAttention)):
                source = self.rotary_local if getattr(module, "kind", "") == "swa" else self.rotary
                module._attention_scaling = source.attention_scaling

        self.apply(self._init_weights)
        # `self.apply` runs *after* every submodule's own __init__, so it
        # overwrites any deliberate initialisation a submodule set for itself.
        # Measured on the pre-v82 code: RecursiveThinkingCore.quality_head
        # intends an exact zero init but came out of construction with
        # weight.abs().sum() == 1.2718, and to_residual intends std 0.01 but
        # came out at 0.0200 -- i.e. the generic 0.02 normal, not the intended
        # one. Re-apply the deliberate inits here.
        #
        # This does NOT by itself wake the recursive core: `residual_scale`
        # still starts at `thinking_residual_init`, whose default is 0.0, and
        # that gate multiplies the core's own gradient. See
        # docs/V59_MECHANISM_CAUSALITY.md -- after 1,000 steps v58's gate had
        # reached 6.41e-04 and closing it entirely changed none of 12,192
        # held-out predictions. Restoring the intended init fixes *what the
        # weights are*, not *whether the mechanism is reachable*.
        self.restored_special_inits = self._restore_special_inits()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _restore_special_inits(self) -> int:
        """Re-run every submodule's ``reset_special_parameters`` hook.

        Any module whose initialisation is load-bearing (a deliberate zero, a
        non-default std, a bias chosen to make a gate near-identity) implements
        that hook, and this puts it back after the blanket ``_init_weights``
        sweep. Returns how many modules were restored.
        """

        restored = 0
        for module in self.modules():
            if module is self:
                continue
            hook = getattr(module, "reset_special_parameters", None)
            if callable(hook):
                hook()
                restored += 1
        return restored

    # -- parameter accounting ---------------------------------------------

    def parameter_report(self) -> Dict[str, int]:
        """Total vs. per-token-active parameters -- the headline MoE ratio.

        "Active" counts, for each MoE layer, the shared expert plus ``top_k``
        routed experts rather than all of them. It excludes the MTP depths,
        which are draft-only at inference.
        """

        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        for module in self.modules():
            if isinstance(module, SparseMoEFeedForward):
                per_expert = sum(p.numel() for p in module.experts[0].parameters())
                inactive += per_expert * (module.n_routed - module.top_k)
        mtp = sum(p.numel() for p in self.mtp_modules.parameters())
        return {
            "total": int(total),
            "active_per_token": int(total - inactive - mtp),
            "routed_but_idle": int(inactive),
            "mtp_draft": int(mtp),
        }

    # -- v93 growth helpers -------------------------------------------------

    @torch.no_grad()
    def zero_new_blocks(self, first_new_layer: int) -> Dict[str, int]:
        """Make every block from ``first_new_layer`` on an exact identity (D4).

        A pre-norm block is ``x + o_proj(attn(norm x))`` then
        ``x + down_proj(...)``, so zeroing ``o_proj`` and every ``down_proj``
        (dense MLP, MoE shared expert and every routed expert) makes both
        residual adds exactly zero -- the grown model's logits equal the
        source's bit for bit, while ``o_proj``/``down_proj`` receive non-zero
        gradient from the first step. Called by the trainer after the warm
        start has been loaded (the appended ``layers.{i}.*`` keys are the only
        ones the source checkpoint lacks). Returns the count of tensors
        zeroed per kind.
        """

        first = int(first_new_layer)
        if not (0 <= first < len(self.layers)):
            raise ValueError(f"first_new_layer {first} out of range for {len(self.layers)} layers")
        zeroed = {"layers": 0, "o_proj": 0, "down_proj": 0}
        for layer in list(self.layers)[first:]:
            zeroed["layers"] += 1
            layer.self_attn.o_proj.weight.zero_()
            zeroed["o_proj"] += 1
            for module in layer.mlp.modules():
                if isinstance(module, DenseFeedForward):
                    module.down_proj.weight.zero_()
                    zeroed["down_proj"] += 1
        return zeroed

    def _cns_cache_entry(
        self,
        past_key_values: Optional[Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
        past_len: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The temporal core's state entry ``past_key_values[n_layers]``.

        Returns ``None`` when nothing is cached. A cache that holds positions
        but no state entry is refused rather than silently restarted from
        rest: that would make cached decoding differ from the full forward.
        """

        entry = None
        if past_key_values is not None and len(past_key_values) > len(self.layers):
            entry = past_key_values[len(self.layers)]
        if entry is None or entry[0].numel() == 0:
            if past_len > 0:
                raise ValueError(
                    "a temporal connectome core needs its state entry at index n_layers of "
                    "past_key_values when positions are cached; pass the past_key_values the "
                    "model returned (trim_past keeps it)"
                )
            return None
        states = entry[0]
        if states.ndim != 3 or int(states.shape[-1]) != int(self.cns_core.n_nodes):
            raise ValueError(
                f"connectome state entry must be (B, T, {self.cns_core.n_nodes}), got {tuple(states.shape)}"
            )
        return (states, entry[1])

    # -- forward ------------------------------------------------------------

    def _collect_aux_loss(self, device, dtype) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=dtype)
        for module in self.modules():
            if isinstance(module, (SparseMoEFeedForward, MixtureOfDepthsRouter)):
                total = total + module.aux_loss().to(device=device, dtype=dtype)
        return total

    def step_router_bias(self) -> int:
        """Apply the aux-loss-free bias update to every MoE layer.

        Call once per optimizer step, **after** ``optimizer.step()``. Returns
        how many layers were updated. See
        :meth:`SparseMoEFeedForward.update_router_bias` for why this is not
        folded into ``forward``.
        """

        return sum(
            1
            for module in self.modules()
            if isinstance(module, SparseMoEFeedForward) and module.update_router_bias()
        )

    def telemetry(self) -> Dict[str, object]:
        """JSON-safe snapshot of the last forward pass."""

        sinks: List[float] = []
        loads: List[List[float]] = []
        router_entropy: List[float] = []
        diff_lambdas: List[List[float]] = []
        mod_skips: List[float] = []
        mod_modes: List[str] = []
        mod_predictor_agreement: List[float] = []
        sink_bearing: List[float] = []
        for module in self.modules():
            if isinstance(module, (HybridAttention, DifferentialHybridAttention, MultiLatentAttention)):
                mass = float(module.last_sink_mass.mean())
                sinks.append(mass)
                # A layer with no sink parameter reports a real-looking 0.0, so
                # averaging over every attention module dilutes the mean by
                # however many layers have no sink to use. Under
                # attention_sink_kinds="swa" that made the telemetry show a large
                # drop in sink usage when the per-layer usage had not moved at
                # all. `mean_sink_mass` is now the mean over sink-bearing layers
                # only; `mean_sink_mass_all_layers` keeps the old denominator so
                # a pre-v85 number is still reconstructable.
                if getattr(module, "sink", None) is not None:
                    sink_bearing.append(mass)
            if isinstance(module, DifferentialHybridAttention):
                diff_lambdas.append([float(v) for v in module.last_lambda])
            if isinstance(module, MixtureOfDepthsRouter):
                mod_skips.append(float(module.last_skip_ratio.item()))
                mod_modes.append(str(module.last_selection_mode))
                mod_predictor_agreement.append(float(module.last_predictor_agreement))
            if isinstance(module, SparseMoEFeedForward):
                loads.append([float(v) for v in module.last_expert_load])
                router_entropy.append(float(module.last_router_entropy))
        snapshot: Dict[str, object] = {
            "attention_layout": list(self.layout),
            "sliding_window": int(self.config.sliding_window),
            "rope_scaling": self.config.rope_scaling,
            "rope_attention_scaling": float(self.rotary.attention_scaling),
            "rotary_dim": int(self.rotary.rotary_dim),
            "head_dim": int(self.config.head_dim),
            "rope_global_base": float(self.rotary.effective_base),
            "rope_local_base": float(self.rotary_local.effective_base),
            "mean_sink_mass": (
                float(sum(sink_bearing) / len(sink_bearing)) if sink_bearing else 0.0
            ),
            "mean_sink_mass_all_layers": (
                float(sum(sinks) / len(sinks)) if sinks else 0.0
            ),
            "sink_bearing_layers": len(sink_bearing),
            "attention_modules": len(sinks),
            "per_layer_sink_mass": sinks,
            "expert_load": loads,
            "router_entropy": router_entropy,
            "parameters": self.parameter_report(),
        }
        if diff_lambdas:
            snapshot["differential_lambdas"] = diff_lambdas
            snapshot["differential_attention"] = True
        if mod_skips:
            snapshot["mod_skip_ratios"] = mod_skips
            snapshot["mod_mean_skip"] = float(sum(mod_skips) / len(mod_skips))
            snapshot["mod_selection_mode"] = mod_modes
            snapshot["mod_predictor_agreement"] = mod_predictor_agreement
            # Honesty flag. The block gates the residual contribution only;
            # attention and the MLP still run on every token, so "skipped"
            # tokens cost the same FLOPs as selected ones. Do not read
            # mod_mean_skip as a compute saving.
            snapshot["mod_compute_saved"] = False
            snapshot["mod_note"] = (
                "residual-gating only: no FLOPs are skipped, this is a routing "
                "study not a compute saving"
            )
        if getattr(self.config, "use_mla", False):
            snapshot["mla_active"] = True
            snapshot["mla_latent_dim"] = int(self.config.mla_latent_dim)
            snapshot["mla_pe_dim"] = int(self.config.mla_pe_dim)
            snapshot["mla_global_only"] = bool(getattr(self.config, "mla_global_only", True))
        if self.thinking_core is not None:
            snapshot["thinking"] = {
                "cycles_used": float(self.thinking_core.last_cycles_used),
                "ponder_cost": float(self.thinking_core.last_ponder_cost),
                "consistency_loss": float(self.thinking_core.last_consistency_loss),
                "quality_probability": float(self.thinking_core.last_quality_score),
                "continue_probability": float(self.thinking_core.last_continue_probability),
                "exit_reason": self.thinking_core.last_exit_reason,
                "temperature": float(self.thinking_core.temperature.detach()),
            }
        if self.cns_core is not None:
            snapshot["cns_core"] = self.cns_core.telemetry()
        return snapshot

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        thinking_cycles: Optional[int] = None,
        adaptive_thinking: bool = False,
        return_mtp: Optional[bool] = None,
        cache_slack: int = 0,
        past_length: Optional[int] = None,
    ) -> MiMoMixOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        if past_length is not None:
            # The caller knows the true absolute position. Trust it: inferring
            # it from cache lengths is only valid while some layer keeps an
            # untrimmed (global) cache.
            past_len = int(past_length)
        else:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0:
                # Only the layer entries describe positions; a v93 temporal
                # core's state entry (index n_layers) holds at most
                # 1 + cache_slack states and must not be read as a length.
                for entry in list(past_key_values)[: len(self.layers)]:
                    if entry is not None and entry[0].numel() > 0:
                        seq_dim = 1 if entry[0].ndim == 3 else 2
                        past_len = max(past_len, int(entry[0].shape[seq_dim]))
        if attention_mask is not None and past_len > 0:
            # Under a hybrid cache each layer retains a different number of past
            # keys, so one caller-supplied mask cannot describe them all. Rather
            # than silently misalign it, refuse the combination.
            raise ValueError(
                "attention_mask is only supported for a full-sequence forward; "
                "with a KV cache the per-layer key spans differ"
            )
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool, device=device)
            if attention_mask.shape != (bsz, seq_len):
                raise ValueError(
                    f"attention_mask must be {(bsz, seq_len)}, got {tuple(attention_mask.shape)}"
                )

        # Absolute positions: with a trimmed SWA cache the *cached* positions
        # differ per layer, so each layer receives its own key_positions below.
        query_positions = torch.arange(past_len, past_len + seq_len, device=device)
        cos, sin = self.rotary(query_positions)
        local_cos, local_sin = self.rotary_local(query_positions)

        hidden = self.embed_tokens(input_ids)
        presents: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        # v93 connectome core plumbing. `core_simple` is v91's exact path (one
        # site, per-token): the core is called as a whole module after its
        # block, which also keeps v91_analysis.CoreMode's forward patch alive.
        core = self.cns_core
        core_simple = core is not None and core.single_site
        core_reads: List[torch.Tensor] = []
        core_read: Optional[torch.Tensor] = None
        core_state_entry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if core is not None and not core_simple:
            read_sites, write_sites, run_after = core.read_sites, core.write_sites, core.read_sites[-1]
            if core.temporal:
                core_state_entry = self._cns_cache_entry(past_key_values, past_len)
        for index, layer in enumerate(self.layers):
            past_kv = None
            if past_key_values is not None and index < len(past_key_values):
                past_kv = past_key_values[index]
            if past_kv is None or past_kv[0].numel() == 0:
                cached = 0
            else:
                cached = int(past_kv[0].shape[1] if past_kv[0].ndim == 3 else past_kv[0].shape[2])
            key_positions = torch.arange(past_len - cached, past_len, device=device)
            layer_cos, layer_sin = (local_cos, local_sin) if layer.kind == "swa" else (cos, sin)
            hidden, present = layer(
                hidden,
                layer_cos,
                layer_sin,
                query_positions,
                key_positions,
                past_kv=past_kv,
                attention_mask=attention_mask,
                use_cache=use_cache,
                cache_slack=cache_slack,
            )
            presents.append(present)
            if core is None:
                continue
            if core_simple:
                if index == int(self.config.cns_after_layer):
                    hidden = core(hidden)
                continue
            if index in read_sites:
                core_reads.append(hidden)
            if index == run_after:
                previous_state = None
                if core_state_entry is not None and core_state_entry[0].numel() > 0:
                    previous_state = core_state_entry[0][:, -1]
                keep = (1 + max(0, int(cache_slack))) if use_cache else None
                core_read, history = core.compute(core_reads, state=previous_state, keep_states=keep)
                if use_cache and history is not None:
                    if core_state_entry is not None and core_state_entry[0].numel() > 0:
                        history = torch.cat([core_state_entry[0], history], dim=1)[:, -int(keep):]
                    companion = history.new_zeros((history.shape[0], history.shape[1], 0))
                    core_state_entry = (history, companion)
            if index in write_sites:
                hidden = core.write(write_sites.index(index), hidden, core_read)

        trunk_hidden = hidden
        thinking_info: Dict[str, torch.Tensor] = {}
        if self.thinking_core is not None:
            bond = None
            if core is not None and not core_simple and core_read is not None:
                bond = core.thinking_input(core_read)
            hidden, thinking_info = self.thinking_core(
                hidden, cycles=thinking_cycles, adaptive=adaptive_thinking, extra_input=bond
            )
        if use_cache and core is not None and not core_simple and core.temporal and core_state_entry is not None:
            # One extra rank-3 entry after the layer entries: (states (B, T', N),
            # companion (B, T', 0)). trim_past narrows both on axis 1, the layer
            # loop never reads index n_layers, and past_len inference skips it.
            presents.append(core_state_entry)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        want_mtp = self.config.n_mtp_layers > 0 if return_mtp is None else bool(return_mtp)
        mtp_logits: List[torch.Tensor] = []
        mtp_loss: Optional[torch.Tensor] = None
        if want_mtp and len(self.mtp_modules) > 0:
            prev_hidden = trunk_hidden
            embeddings = self.embed_tokens(input_ids)
            depth_losses: List[torch.Tensor] = []
            for depth, module in enumerate(self.mtp_modules, start=1):
                # depth k consumes the embedding of the token k positions ahead
                shifted = torch.zeros_like(embeddings)
                if seq_len > depth:
                    shifted[:, : seq_len - depth] = embeddings[:, depth:]
                prev_hidden = module(prev_hidden, shifted, cos, sin, query_positions, attention_mask)
                depth_logits = self.lm_head(self.norm(prev_hidden))
                mtp_logits.append(depth_logits)
                if labels is not None and seq_len > depth + 1:
                    # depth k at position i predicts label at i + k + 1
                    pred = depth_logits[:, : seq_len - depth - 1]
                    target = labels[:, depth + 1 :]
                    depth_losses.append(
                        F.cross_entropy(
                            pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="mean"
                        )
                    )
            if depth_losses:
                mtp_loss = torch.stack(depth_losses).mean()

        lm_loss: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None
        aux_loss = self._collect_aux_loss(logits.device, logits.dtype)
        if labels is not None:
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            lm_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1)
            )
            loss = lm_loss + aux_loss
            if mtp_loss is not None:
                loss = loss + self.config.mtp_loss_weight * mtp_loss
            if thinking_info:
                loss = (
                    loss
                    + self.config.ponder_loss_weight * thinking_info["ponder_cost"]
                    + self.config.consistency_loss_weight * thinking_info["consistency_loss"]
                )

        telemetry = self.telemetry()
        telemetry["sequence_length"] = int(seq_len)
        telemetry["past_length"] = int(past_len)
        if thinking_info:
            telemetry["thinking"]["quality_probability"] = float(
                thinking_info["quality_probability"].detach().mean()
            )
            telemetry["thinking"]["continue_probability"] = float(
                thinking_info["continue_probability"].detach().mean()
            )

        return MiMoMixOutput(
            logits=logits,
            hidden_states=hidden,
            trunk_hidden=trunk_hidden,
            loss=loss,
            lm_loss=lm_loss,
            mtp_loss=mtp_loss,
            aux_loss=aux_loss,
            mtp_logits=mtp_logits,
            past_key_values=presents if use_cache else None,
            telemetry=telemetry,
        )

    @torch.no_grad()
    def propose_draft(
        self, trunk_hidden_last: torch.Tensor, seed_token: torch.Tensor, position: int
    ) -> torch.Tensor:
        """Run the MTP chain to speculate ``n_mtp_layers`` tokens ahead.

        ``trunk_hidden_last`` is ``(B, 1, H)`` -- the trunk state at the last
        real position. ``seed_token`` is ``(B, 1)``: the token the trunk already
        committed for the next slot, which depth 1 conditions on.

        Draft quality only affects *speed*. Every speculative token is checked
        against the trunk in :mod:`mimomix_decoding`, so a bad draft costs
        throughput and never changes the emitted sequence.
        """

        if len(self.mtp_modules) == 0:
            return seed_token.new_zeros((seed_token.shape[0], 0))
        positions = torch.tensor([int(position)], device=seed_token.device)
        cos, sin = self.rotary(positions)
        prev = trunk_hidden_last
        current = seed_token
        drafted: List[torch.Tensor] = []
        for module in self.mtp_modules:
            embedded = self.embed_tokens(current)
            prev = module(prev, embedded, cos, sin, positions)
            current = self.lm_head(self.norm(prev)).argmax(dim=-1)
            drafted.append(current)
        return torch.cat(drafted, dim=1)

    def verifier_loss(self, correctness: torch.Tensor) -> torch.Tensor:
        """Supervised quality/continue objective for the thinking core.

        ``correctness`` is a float tensor in ``{0, 1}`` per flattened position.
        ``p(correct)`` is trained to match it and ``p(continue)`` to match its
        complement -- the v52 contract: keep thinking exactly when the current
        answer is wrong.
        """

        if self.thinking_core is None:
            raise RuntimeError("model was built without a thinking core")
        logits = getattr(self.thinking_core, "_last_quality_logits", None)
        if logits is None:
            raise RuntimeError("verifier_loss requires a forward pass that stored quality logits")
        target = correctness.reshape(-1).to(logits.dtype)
        return F.binary_cross_entropy_with_logits(
            logits[:, 0], target
        ) + F.binary_cross_entropy_with_logits(logits[:, 1], 1.0 - target)

    def encode_multimodal_tokens(
        self,
        features: torch.Tensor,
        modality: str = "vision",
    ) -> torch.Tensor:
        """Encode raw multimodal features into sequence-compatible token vectors (MiMo-V2.5).

        features: (batch_size, num_tokens, input_dim)
        returns: (batch_size, num_tokens, hidden_size)
        """
        if self.multimodal_projector is None:
            raise RuntimeError("multimodal_projector is not enabled in model config (set use_multimodal=True)")
        return self.multimodal_projector(features)


def build_mimomix(**overrides) -> MiMoMixModel:
    """Convenience constructor: ``build_mimomix(n_layers=4, use_moe=False)``."""

    return MiMoMixModel(MiMoMixConfig(**overrides))
