"""Supermix Expanse: Archimedes final widened by five more sources.

Trunk
    Archimedes final (``supermix-archimedes-v1``, 47.6M: the v93 trunk, v87
    experts, FlyCore and OmniCore v48/v38). Loaded verbatim; every new graft
    is born function-preserving (zero gate, dead or dormant slot), so the
    grafted model's logits equal Archimedes' until stage 2 opens a gate --
    with one deliberate exception, the FlyCore causality fix below.

Grafts (all written into the residual stream after ``cns_after_layer``)
    fly (causal)   ``FlyCore.sense`` in Archimedes mean-pools the hidden
                   state over *all* T positions (future tokens and PAD
                   included) and broadcasts one write to every position. That
                   leaks the future into every logit (~3e-3 at the trained
                   gate) and makes a KV-cached decode differ from the full
                   forward, because the decode step only sees one token.
                   :func:`fly_forward_causal` senses **per position**: position
                   t reads only h_t, so the fly core is causal and decode-exact.
    omni v48/v38   unchanged (prompt-constant ``OmniCore``).
    omni7          ``OmniV7Branch`` (src/omni_v7_branch.py): the frozen Omni
                   Collective v7 Frontier net read once per prompt; its 988-d
                   conclusion is bridged through a zero gate. Prompt-constant.
    cns_full       ``FullConnectomeCore`` (src/connectome_full.py): all 11,751
                   male-CNS cell types and all 3,830,931 type->type edges with
                   fixed measured weights, run per token. Per-position.
    donor experts  Qwen2.5-Coder-7B and BioMedLM layer-0/1 MLP neurons grafted
                   into new MoE slots (src/donor_graft.py), born dormant.

    Every graft is per-position or prompt-constant, so a KV-cached decode
    needs no extra cache state: the hook simply sees the new positions.

MoE slot growth
    ``SparseMoEFeedForward`` sizes itself from ``config.n_routed_experts +
    config.moe_spare_experts`` -- one config field shared by *every* MoE layer.
    Growing slots therefore grows all MoE layers (1-4) at once; donors are
    installed only in layers 1 and 2 and the new slots of layers 3 and 4 stay
    dead (spare capacity for later neurogenesis). A dead slot is masked to
    ``-inf`` before the router softmax, which makes growth preserve the
    function *mathematically* -- but not bitwise: exact top-k ties are common
    at the trained router bias and ``torch.topk`` breaks them differently on
    a longer row. :class:`ExpanseSparseMoE` therefore routes over the active
    width (up to the last alive slot), which with a dead suffix is the very
    computation the ungrown layer ran; growth then reproduces Archimedes
    bit for bit (tests/test_expanse_core.py).

Frozen encoders
    The v48/v38 encoders and the v7 net are frozen *and kept in eval mode*
    even while the model trains (their dropout would otherwise inject noise
    into features that are supposed to be fixed).

Checkpoint schema ``supermix-expanse-v1`` = the Archimedes schema with
``expanse`` in place of ``archimedes``: the build/train receipt plus the
structural spec (connectome shapes, v7 meta, fly config, donor receipts) that
:func:`load_expanse` needs to rebuild module shapes before the strict load.
The frozen v7 net and the connectome CSR buffers are in the state dict, so a
checkpoint is self-contained.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
EXPANSE_DIR = HERE.parent
EXP = EXPANSE_DIR.parent
REPO = EXP / "supermix-archimedes"
ARCH_SRC = REPO / "archimedes" / "src"
for _p in (str(HERE), str(ARCH_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import archimedes_core as arch  # noqa: E402  (adds champion/ to sys.path itself)
from archimedes_core import (  # noqa: E402
    GLOMERULI, ArchimedesModel, FlyCore, load_archimedes, text_utils, wake_grafted_experts,
)
from mimomix_core import MiMoMixConfig, MiMoMixModel, SparseMoEFeedForward  # noqa: E402

EXP_SCHEMA = "supermix-expanse-v1"
ARCH_SCHEMA = arch.SCHEMA  # "supermix-archimedes-v1"

#: Canonical locations (DESIGN.md "Paths"). CLIs take overrides; these are defaults.
PATHS: Dict[str, Path] = {
    "exp": EXP,
    "repo": REPO,
    "corpus": REPO / "corpus",
    "arch_final": EXP / "external" / "base" / "supermix_archimedes.pt",
    "omni_v7_dir": EXP / "external" / "omni_v7",
    "qwen_dir": EXP / "external" / "teachers" / "qwen2.5-coder-7b-instruct",
    "biomedlm_dir": EXP / "external" / "teachers" / "biomedlm",
    # fallback source for data/malecns_types.npz (shipped in the repo); rebuild it from the public
    # male-CNS v1.0 feathers with supermix-archimedes/archimedes/src/malecns_connectome.py build
    "npz_source": Path(os.environ.get("MALECNS_TYPES_NPZ", str(EXPANSE_DIR / "data" / "malecns_types.npz"))),
    "data": EXPANSE_DIR / "data",
    "npz": EXPANSE_DIR / "data" / "malecns_types.npz",
    "code_rows": EXPANSE_DIR / "data" / "code_rows.jsonl",
    "bio_rows": EXPANSE_DIR / "data" / "bio_rows.clean.jsonl",  # see clean_bio_rows.py
    "connectome_rows": EXPANSE_DIR / "data" / "connectome_rows.jsonl",
    "checkpoints": EXPANSE_DIR / "checkpoints",
    "grafted": EXPANSE_DIR / "checkpoints" / "supermix_expanse_grafted.pt",
    "final": EXPANSE_DIR / "checkpoints" / "supermix_expanse.pt",
}

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    return arch.sha256_file(Path(path))


def row_key(user: str, assistant: Optional[str] = "") -> str:
    """Stable key of one (user, assistant) row, for teacher caches and splits."""
    h = hashlib.sha1()
    h.update((user or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update((assistant or "").encode("utf-8"))
    return h.hexdigest()


def read_jsonl(path) -> List[Dict[str, Any]]:
    """Rows of a jsonl file ([] when it does not exist; blank lines skipped)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_replay(corpus_dir=None) -> List[Dict[str, Any]]:
    """Archimedes' replay corpus (omni/code/math, 3,743 rows) as training rows."""
    corpus_dir = Path(corpus_dir) if corpus_dir is not None else PATHS["corpus"]
    rows = []
    for name in ("omni", "code", "math"):
        for r in read_jsonl(corpus_dir / f"{name}.jsonl"):
            rows.append({"user": r["user"], "assistant": r["assistant"], "source": "replay",
                         "task": r.get("task", name), "domain": r.get("domain", name)})
    return rows


@torch.no_grad()
def append_embedding_rows(model: MiMoMixModel, rows: torch.Tensor) -> int:
    """Grow the tied embedding / lm_head by ``rows`` (existing ids never move)."""
    old = model.embed_tokens.weight
    weight = nn.Parameter(torch.cat([old.detach(), rows.to(old.dtype)], 0))
    model.embed_tokens.weight = weight
    model.embed_tokens.num_embeddings = int(weight.shape[0])
    model.lm_head.weight = weight  # tied
    model.lm_head.out_features = int(weight.shape[0])
    model.config.vocab_size = int(weight.shape[0])
    return int(rows.shape[0])


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars and arrays into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 4096 else f"<ndarray {obj.shape} {obj.dtype}>"
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.numel() <= 4096 else f"<tensor {tuple(obj.shape)} {obj.dtype}>"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# graft classes (resolved lazily so this module imports without them)
# ---------------------------------------------------------------------------

#: Classes used for the two new grafts. ``None`` means "import the real one"
#: (``connectome_full.FullConnectomeCore`` / ``omni_v7_branch.OmniV7Branch``).
#: Tests may point an entry at a stand-in with the same constructor/forward
#: contract; production code never touches this.
GRAFT_CLASSES: Dict[str, Optional[type]] = {"cns_full": None, "omni7": None}


def graft_class(name: str) -> type:
    cls = GRAFT_CLASSES.get(name)
    if cls is not None:
        return cls
    if name == "cns_full":
        from connectome_full import FullConnectomeCore  # noqa: WPS433
        return FullConnectomeCore
    if name == "omni7":
        from omni_v7_branch import OmniV7Branch  # noqa: WPS433
        return OmniV7Branch
    raise KeyError(name)


# ---------------------------------------------------------------------------
# 1. MoE slot growth
# ---------------------------------------------------------------------------


def moe_layers_in_state_dict(state_dict: Dict[str, torch.Tensor]) -> List[int]:
    """Indices of every trunk layer whose MLP is a ``SparseMoEFeedForward``."""
    out = set()
    for k in state_dict:
        if k.startswith("layers.") and k.endswith(".mlp.expert_alive"):
            out.add(int(k.split(".")[1]))
    return sorted(out)


class ExpanseSparseMoE(SparseMoEFeedForward):
    """``SparseMoEFeedForward`` that routes over its *active width* only.

    Why this exists: selection is ``softmax score + expert_bias`` with the
    bias near 22.47 (fp32 ulp ~2e-6), so scores below the ulp are absorbed
    and distinct experts regularly tie *exactly* (measured: 2 of 342 tokens
    in layer 1 on a 6-row batch). ``torch.topk`` breaks ties by an unstable
    ``nth_element`` whose outcome depends on the row length, so appending
    even fully masked (-inf) slots can flip a tied pick and move a logit by
    0.5 -- the dead mask alone is not bitwise function-preserving.

    The active width ``W`` is one past the last alive slot. Slots at or
    beyond ``W`` are dead, i.e. ``-inf`` in the base code: they can never be
    selected, score exactly 0 and drop out of the z-loss. Routing over the
    first ``W`` columns is therefore the same function, and when the grown
    slots are a dead suffix it is the *identical computation* the ungrown
    module ran (same matmul shape, same softmax length, same top-k row), so
    growth reproduces Archimedes bit for bit. Once a suffix slot is woken,
    ``W`` covers it and the base forward runs unchanged. Telemetry and
    load buffers stay ``n_routed`` long (zero-padded).

    Installed by ``ExpanseModel.__init__`` with a class swap: no new
    parameters, the state dict is unchanged.
    """

    def active_width(self) -> int:
        alive = (self.expert_alive != 0).nonzero()
        return int(alive.max()) + 1 if alive.numel() else self.n_routed

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        width = self.active_width()
        if width >= self.n_routed or width < self.top_k:
            return super().forward(x, token_mask)
        import torch.nn.functional as F  # local: keeps the module header identical to the base's imports

        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        n_tokens = flat.shape[0]
        flat_mask: Optional[torch.Tensor] = None
        if token_mask is not None:
            flat_mask = token_mask.reshape(-1).to(dtype=torch.bool, device=flat.device)
            if flat_mask.numel() != n_tokens:
                raise ValueError(f"token_mask has {flat_mask.numel()} entries for {n_tokens} tokens")

        logits = F.linear(flat, self.gate.weight[:width])
        alive = self.expert_alive[:width]
        n_balance = width
        dead_slots: Optional[torch.Tensor] = None
        if bool((alive == 0).any()):
            dead_slots = (alive == 0).unsqueeze(0)
            logits = logits.masked_fill(dead_slots, float("-inf"))
            n_balance = int(alive.sum())
        scores = self._scores(logits)
        selection_scores = scores + self.expert_bias[:width].to(scores.dtype).unsqueeze(0)
        if dead_slots is not None:
            selection_scores = selection_scores.masked_fill(dead_slots, float("-inf"))
        _, expert_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        gate_weights = scores.gather(-1, expert_indices)
        if self.norm_topk_prob and self.top_k > 1:
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        one_hot = F.one_hot(expert_indices, num_classes=width).sum(dim=1)  # (N, W)
        for expert_id in range(width):
            token_ids = torch.nonzero(one_hot[:, expert_id], as_tuple=False).flatten()
            if token_ids.numel() == 0:
                continue
            expert_out = self.experts[expert_id](flat.index_select(0, token_ids))
            weight = (gate_weights * (expert_indices == expert_id)).sum(dim=-1)
            weight = weight.index_select(0, token_ids).unsqueeze(-1)
            contribution = expert_out * weight.to(expert_out.dtype)
            output.index_add_(0, token_ids, contribution.to(output.dtype))
        if self.shared_expert is not None:
            output = output + self.shared_expert(flat)

        # routing telemetry and regularisers (as the base, W-wide, padded to n_routed)
        occupancy = one_hot.float()
        probabilities = scores.float()
        if flat_mask is not None:
            keep = flat_mask.float().unsqueeze(-1)
            denom = keep.sum().clamp_min(1.0)
            load = (occupancy * keep).sum(dim=0) / denom
            mean_prob = (probabilities * keep).sum(dim=0) / denom
        else:
            load = occupancy.mean(dim=0)
            mean_prob = probabilities.mean(dim=0)
        pad = self.n_routed - width
        self.last_expert_load = F.pad(load.detach(), (0, pad))
        self.last_router_entropy = -(mean_prob * mean_prob.clamp_min(1e-9).log()).sum().detach()
        z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        if self.balance_scope == "sequence" and len(original_shape) == 3:
            bsz, seq_len = int(original_shape[0]), int(original_shape[1])
            per_seq_occupancy = occupancy.view(bsz, seq_len, width)
            per_seq_prob = probabilities.view(bsz, seq_len, width)
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
            self._aux_loss = self.config.router_z_loss_coef * z_loss + self.config.router_balance_loss_coef * balance_loss
            with torch.no_grad():
                self.pending_load += F.pad(load.detach(), (0, pad))
                self.pending_batches += 1.0
            if self.auto_update_bias:
                self.update_router_bias()
        else:
            self._aux_loss = logits.new_zeros(())
            if self.collect_stats:
                with torch.no_grad():
                    self.stat_load_sum += F.pad(load.detach(), (0, pad))
                    self.stat_batches += 1.0
        return output.reshape(original_shape)


@torch.no_grad()
def grow_moe_slots(state_dict: Dict[str, torch.Tensor], config: MiMoMixConfig, n_new: int,
                   layers: Sequence[int] = (1, 2, 3, 4)) -> Tuple[Dict[str, torch.Tensor], MiMoMixConfig]:
    """Append ``n_new`` dead expert slots to every MoE layer.

    Returns a new (shallow-copied) state dict and a new config with
    ``moe_spare_experts += n_new``; a model built from that config loads the
    state dict strictly. Per MoE layer the padding is: router rows zero,
    ``expert_bias`` = min(alive bias) - 10, ``expert_alive`` = 0, expert
    weights zero. The dead mask sends the new router logits to ``-inf``
    before the softmax, so every alive score, every top-k choice and every
    logit is unchanged. ``layers`` must name every MoE layer present, because
    ``moe_spare_experts`` is one field shared by all of them (see module doc).
    """
    n_new = int(n_new)
    if n_new < 0:
        raise ValueError("n_new must be >= 0")
    found = moe_layers_in_state_dict(state_dict)
    want = sorted({int(l) for l in layers})
    if want != found:
        raise ValueError(
            f"grow_moe_slots: layers {want} != MoE layers in the state dict {found}; "
            "config.moe_spare_experts is shared by every SparseMoEFeedForward, so all MoE layers must grow together")
    cfg = MiMoMixConfig(**config.to_dict())
    old_slots = int(cfg.n_routed_experts) + int(getattr(cfg, "moe_spare_experts", 0))
    cfg.moe_spare_experts = int(getattr(cfg, "moe_spare_experts", 0)) + n_new
    sd = dict(state_dict)
    if n_new == 0:
        return sd, cfg
    for li in found:
        p = f"layers.{li}.mlp."
        gate = sd[p + "gate.weight"]
        if int(gate.shape[0]) != old_slots:
            raise ValueError(f"layer {li}: router has {gate.shape[0]} rows, config says {old_slots} slots")
        alive = sd[p + "expert_alive"]
        bias = sd[p + "expert_bias"]
        alive_bias = bias[alive.bool()]
        fill = (float(alive_bias.min()) if alive_bias.numel() else 0.0) - 10.0
        sd[p + "gate.weight"] = torch.cat([gate, gate.new_zeros(n_new, gate.shape[1])], 0)
        sd[p + "expert_bias"] = torch.cat([bias, bias.new_full((n_new,), fill)], 0)
        sd[p + "expert_alive"] = torch.cat([alive, alive.new_zeros(n_new)], 0)
        for name in ("gate_proj", "up_proj", "down_proj"):
            ref = sd[f"{p}experts.0.{name}.weight"]
            for slot in range(old_slots, old_slots + n_new):
                sd[f"{p}experts.{slot}.{name}.weight"] = torch.zeros_like(ref)
    return sd, cfg


# ---------------------------------------------------------------------------
# 2. causal fly core
# ---------------------------------------------------------------------------


def fly_forward_causal(fly_core: FlyCore, hidden: torch.Tensor,
                       fly_obs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """FlyCore read and written per position, so no position sees the future.

    ``obs_t = tanh(sensory_proj(norm(h_t)))`` (B, T, 14); the 22 brains run on
    the (B*T, 14) batch; each position gets its own write
    ``gate * to_trunk([descending.flatten(); consensus])``. With ``fly_obs``
    (B, 14) given, the brains run once per row and the write is broadcast --
    identical to ``FlyCore.forward(hidden, fly_obs)``, which never leaked.
    ``info`` tensors are reshaped to (B, T, ...); ``info['obs']`` is always
    (B, T, 14) (the trainer's sense loss reads the last prompt position).
    """
    B, T, H = hidden.shape
    if fly_obs is not None:
        obs_b = fly_obs.to(device=hidden.device, dtype=hidden.dtype)
        out = fly_core.brains_forward(obs_b)
        feat = torch.cat([out["descending"].flatten(1), out["consensus"]], dim=-1)
        write = (fly_core.gate * fly_core.to_trunk(feat)).unsqueeze(1)
        out["obs"] = obs_b.unsqueeze(1).expand(B, T, obs_b.shape[-1])
        return hidden + write, out
    obs = torch.tanh(fly_core.sensory_proj(fly_core.norm(hidden)))  # (B, T, 14)
    out = fly_core.brains_forward(obs.reshape(B * T, GLOMERULI))
    feat = torch.cat([out["descending"].flatten(1), out["consensus"]], dim=-1)  # (B*T, 92)
    write = fly_core.gate * fly_core.to_trunk(feat).view(B, T, H)
    info = {k: v.reshape(B, T, *v.shape[1:]) for k, v in out.items()}
    info["obs"] = obs
    return hidden + write, info


# ---------------------------------------------------------------------------
# 3. connectome graph metadata (what a checkpoint needs to rebuild shapes)
# ---------------------------------------------------------------------------


def graph_metadata(graph: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """Small, JSON-safe description of a ``build_full_graph`` dict."""
    return {
        "n": int(graph["n"]),
        "nnz": int(len(graph["post"])),
        "n_in": int(len(graph["in_idx"])),
        "n_out": int(len(graph["out_idx"])),
        "steps": int(steps),
        "rewired": bool(graph.get("rewired", False)),
        "receipt": jsonable(graph.get("receipt", {})),
    }


def skeleton_graph(meta: Dict[str, Any]) -> Dict[str, Any]:
    """A structurally valid graph dict with the checkpoint's shapes.

    ``FullConnectomeCore`` takes a graph dict; when loading a checkpoint the
    real graph is not needed because every buffer (CSR of W and W^T, io
    indices, sign) is persistent and comes from the state dict. This builds a
    placeholder with the same ``n``, ``nnz``, ``n_in``, ``n_out``: unique
    (post, pre) pairs sorted by post then pre, tiny values, distinct io
    indices. Its contents are overwritten by the strict load.
    """
    n, nnz = int(meta["n"]), int(meta["nnz"])
    n_in, n_out = int(meta["n_in"]), int(meta["n_out"])
    if nnz > n * n:
        raise ValueError(f"nnz {nnz} > n^2 for n={n}")
    k = np.arange(nnz, dtype=np.int64)
    post, pre = (k % n).astype(np.int32), (k // n).astype(np.int32)
    order = np.lexsort((pre, post))
    post, pre = post[order], pre[order]
    in_idx = np.arange(n_in, dtype=np.int64)
    out_idx = (np.arange(n_out, dtype=np.int64) + (n_in if n_in + n_out <= n else 0)) % max(n, 1)
    blank = np.array([""] * n, dtype=object)
    return {
        "n": n, "post": post, "pre": pre, "value": np.full(nnz, 1e-3, dtype=np.float32),
        "sign": np.ones(n, dtype=np.int8), "in_idx": in_idx, "out_idx": out_idx,
        "type_names": blank, "superclass": blank.copy(), "nt": blank.copy(),
        "receipt": dict(meta.get("receipt") or {}), "skeleton": True,
    }


# ---------------------------------------------------------------------------
# 4. the model
# ---------------------------------------------------------------------------


class ExpanseModel(ArchimedesModel):
    """Archimedes + causal fly + omni7 bridge + full connectome core + donor slots.

    ``expanse`` keys: ``graph`` (dict from ``build_full_graph``; ``None`` when
    loading), ``cns`` (:func:`graph_metadata`, used when ``graph`` is None),
    ``cns_steps`` (4), ``omni7_meta`` (``omni_v7_branch.load_v7_meta()``;
    ``None`` = no omni7 branch), ``fly_causal`` (True; False reproduces
    Archimedes' mean-pooled fly exactly), ``donor_experts`` (``{teacher:
    install receipt}``), ``vocab_base`` (Archimedes vocab size).
    """

    def __init__(self, config: MiMoMixConfig, fly_config: Optional[Dict[str, float]] = None,
                 with_omni: bool = True, expanse: Optional[Dict[str, Any]] = None):
        super().__init__(config, fly_config=fly_config, with_omni=with_omni)
        # active-width routing (class swap: same parameters, same state dict)
        for layer in self.layers:
            if type(layer.mlp) is SparseMoEFeedForward:
                layer.mlp.__class__ = ExpanseSparseMoE
        spec = dict(expanse or {})
        hidden = int(config.hidden_size)
        self.fly_config = copy.deepcopy(fly_config)
        self.with_omni = bool(with_omni)
        self.fly_causal = bool(spec.get("fly_causal", True))

        # full connectome core
        graph = spec.get("graph")
        cns_meta = spec.get("cns")
        steps = int(spec.get("cns_steps") or (cns_meta or {}).get("steps") or 4)
        self.cns_full: Optional[nn.Module] = None
        self.cns_meta: Optional[Dict[str, Any]] = None
        if graph is not None:
            self.cns_meta = graph_metadata(graph, steps)
        elif cns_meta:
            # loading: FullConnectomeCore accepts a shape-only dict, its buffers
            # come from the state dict
            self.cns_meta = dict(cns_meta)
            self.cns_meta["steps"] = steps
            graph = {k: int(self.cns_meta[k]) for k in ("n", "n_in", "n_out", "nnz")}
        if graph is not None:
            cls = graft_class("cns_full")
            try:
                self.cns_full = cls(hidden, graph, steps=steps)
            except (KeyError, TypeError, IndexError):
                if "post" in graph:
                    raise
                # a core that insists on arrays gets a placeholder of the same shapes
                self.cns_full = cls(hidden, skeleton_graph(self.cns_meta), steps=steps)
        del graph

        # omni collective v7 branch
        meta = spec.get("omni7_meta")
        self.omni7: Optional[nn.Module] = graft_class("omni7")(hidden, meta) if meta is not None else None
        self.omni7_meta = copy.deepcopy(getattr(self.omni7, "meta", meta)) if self.omni7 is not None else None

        self.donor_receipts: Dict[str, Any] = json.loads(json.dumps(jsonable(spec.get("donor_experts") or {})))
        self.vocab_base = spec.get("vocab_base")

    # -- the graft site ------------------------------------------------------
    def _graft_hook(self, module, inputs, output):
        hidden, present = output
        ctx = self._ctx
        if ctx.get("skip"):
            return output
        info: Dict[str, Any] = {}
        if self.fly_core is not None and ctx.get("fly", True):
            if self.fly_causal:
                hidden, fly_info = fly_forward_causal(self.fly_core, hidden, ctx.get("fly_obs"))
            else:  # Archimedes' original (mean-pooled, leaks the future)
                hidden, fly_info = self.fly_core(hidden, ctx.get("fly_obs"))
            info["fly"] = fly_info
        if self.omni_core is not None and ctx.get("omni_features") is not None:
            hidden, omni_info = self.omni_core(hidden, ctx["omni_features"])
            info["omni"] = omni_info
        if self.omni7 is not None and ctx.get("omni7", True) and ctx.get("omni7_state") is not None:
            hidden, o7_info = self.omni7(hidden, ctx["omni7_state"])
            info["omni7"] = o7_info
        if self.cns_full is not None and ctx.get("cns_full", True):
            hidden, cns_info = self.cns_full(hidden, ctx.get("token_mask"))
            info["cns_full"] = cns_info
        ctx["info"] = info
        return hidden, present

    def forward(self, input_ids, *args, fly_obs=None, omni_features=None, omni7_state=None, use_fly=True,
                use_omni7=True, use_cns_full=True, skip_grafts=False, token_mask=None, **kwargs):
        if token_mask is None:
            token_mask = input_ids != text_utils.PAD
        self._ctx = {"fly_obs": fly_obs, "omni_features": omni_features, "omni7_state": omni7_state,
                     "fly": use_fly, "omni7": use_omni7, "cns_full": use_cns_full, "skip": skip_grafts,
                     "token_mask": token_mask}
        try:
            out = MiMoMixModel.forward(self, input_ids, *args, **kwargs)
            info = self._ctx.get("info", {})
        finally:
            self._ctx = {}
        try:
            out.graft_info = info
        except Exception:
            pass
        self.last_graft_info = info
        return out

    def train(self, mode: bool = True) -> "ExpanseModel":
        super().train(mode)
        # frozen encoders stay deterministic (see module doc)
        if self.omni_core is not None:
            for m in (self.omni_core.collective, self.omni_core.collective_head,
                      self.omni_core.collective_norm, self.omni_core.vision):
                m.eval()
        if self.omni7 is not None and hasattr(self.omni7, "net"):
            self.omni7.net.eval()
        return self

    # -- helpers ---------------------------------------------------------------
    @torch.no_grad()
    def omni7_state_for(self, prompts: Sequence[str], batch_size: int = 32) -> Optional[torch.Tensor]:
        """(B, 988) v7 state for user prompts, or None without the branch."""
        if self.omni7 is None:
            return None
        rows = [self.omni7.encode(self.omni7.featurize(list(prompts[i:i + batch_size])))
                for i in range(0, len(prompts), batch_size)]
        return torch.cat(rows, 0) if rows else None

    def donor_slots(self) -> Dict[int, List[int]]:
        """{MoE layer: slots} over every donor receipt."""
        out: Dict[int, List[int]] = {}
        for rec in self.donor_receipts.values():
            for li, info in (rec.get("layers") or {}).items():
                out.setdefault(int(li), [])
                out[int(li)] += [int(s) for s in info.get("slots", []) if int(s) not in out[int(li)]]
        return out

    @torch.no_grad()
    def wake_donor_experts(self, progress: float) -> Dict[str, Dict[str, float]]:
        """Wake every donor slot (alive 1) with its bias annealed from the dormant bias
        to its own load-matched ``target_bias_by_slot`` (see donor_graft.build_donor_experts);
        receipts without per-slot targets fall back to ``wake_grafted_experts`` (alive median)."""
        progress = float(min(1.0, max(0.0, progress)))
        out: Dict[str, Dict[str, float]] = {}
        for name, rec in self.donor_receipts.items():
            if not rec.get("layers"):
                continue
            if not any(info.get("target_bias_by_slot") for info in rec["layers"].values()):
                out[name] = wake_grafted_experts(self, rec, progress)
                continue
            out[name] = {}
            for li, info in rec["layers"].items():
                mlp = self.layers[int(li)].mlp
                by_slot = {int(k): float(v) for k, v in (info.get("target_bias_by_slot") or {}).items()}
                for slot in info["slots"]:
                    target = by_slot.get(int(slot), float(info["target_bias"]))
                    mlp.expert_alive[slot] = 1
                    mlp.expert_bias[slot] = info["dormant_bias"] + (target - info["dormant_bias"]) * progress
                out[name][str(li)] = float(mlp.expert_bias[info["slots"]].mean()) if info["slots"] else 0.0
        return out

    @torch.no_grad()
    def set_donor_alive(self, alive: bool) -> None:
        """Ablation helper: mark every donor slot alive/dead (bias untouched)."""
        for li, slots in self.donor_slots().items():
            mlp = self.layers[li].mlp
            for s in slots:
                mlp.expert_alive[s] = 1 if alive else 0

    def graft_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().graft_parameters()
        if self.cns_full is not None:
            yield from self.cns_full.parameters()
        if self.omni7 is not None:
            for name, p in self.omni7.named_parameters():
                if not name.startswith("net."):
                    yield p

    def gate_report(self) -> Dict[str, Any]:
        # Archimedes' keys, computed on detached gates (no autograd warning)
        rep: Dict[str, Any] = {"fly_gate_abs_mean": float(self.fly_core.gate.detach().abs().mean())}
        if self.omni_core is not None:
            rep["omni_gate_abs_mean"] = float(self.omni_core.gate.detach().abs().mean())
        if self.cns_core is not None:
            rep["cns_gate_abs_mean"] = float(self.cns_core.gate.detach().abs().mean())
            rep["cns_alive"] = int(self.cns_core.alive.sum())
        if self.omni7 is not None:
            rep["omni7_gate_abs_mean"] = float(self.omni7.gate.detach().abs().mean())
        if self.cns_full is not None:
            rep["cns_full_gate_abs_mean"] = float(self.cns_full.gate.detach().abs().mean())
        for li, layer in enumerate(self.layers):
            if isinstance(layer.mlp, SparseMoEFeedForward):
                rep[f"moe{li}_alive"] = int(layer.mlp.expert_alive.sum())
        for name, rec in self.donor_receipts.items():
            alive = total = 0
            biases: List[float] = []
            w2 = 0.0
            nel = 0
            for li, info in (rec.get("layers") or {}).items():
                mlp = self.layers[int(li)].mlp
                for s in info.get("slots", []):
                    total += 1
                    alive += int(mlp.expert_alive[int(s)])
                    biases.append(float(mlp.expert_bias[int(s)]))
                    for p in mlp.experts[int(s)].parameters():
                        w2 += float(p.detach().pow(2).sum())
                        nel += p.numel()
            rep[f"donor_{name}_alive"] = alive
            rep[f"donor_{name}_slots"] = total
            if biases:
                rep[f"donor_{name}_bias_mean"] = float(sum(biases) / len(biases))
                rep[f"donor_{name}_weight_rms"] = math.sqrt(w2 / max(1, nel))
        return rep

    def expanse_spec(self) -> Dict[str, Any]:
        """Everything :func:`load_expanse` needs to rebuild module shapes."""
        return {
            "fly_causal": self.fly_causal,
            "fly_config": self.fly_config,
            "with_omni": self.with_omni,
            "cns": self.cns_meta,
            "omni7_meta": self.omni7_meta,
            "donor_experts": self.donor_receipts,
            "vocab_base": self.vocab_base,
        }


def expanse_from_archimedes(arch_payload: Dict[str, Any], *, n_new: int = 16, graph: Optional[Dict[str, Any]] = None,
                            cns_steps: int = 4, omni7_meta: Optional[Dict[str, Any]] = None,
                            fly_causal: bool = True, state_dict: Optional[Dict[str, torch.Tensor]] = None
                            ) -> Tuple[ExpanseModel, Dict[str, Any]]:
    """Build an ExpanseModel carrying the Archimedes weights (MoE grown by ``n_new``).

    ``arch_payload`` is the dict ``load_archimedes`` returns (its config,
    ``archimedes.fly_config`` / ``with_omni`` and, unless ``state_dict`` is
    given, its state dict). The only keys the Archimedes state lacks must be
    the new grafts' (``cns_full.*``, ``omni7.*``); anything else is an error.
    """
    config = MiMoMixConfig(**arch_payload["config"])
    meta = arch_payload.get("archimedes", {})
    sd = state_dict if state_dict is not None else arch_payload["state_dict"]
    sd, cfg = grow_moe_slots(sd, config, n_new, layers=moe_layers_in_state_dict(sd))
    model = ExpanseModel(cfg, fly_config=meta.get("fly_config"), with_omni=meta.get("with_omni", True),
                         expanse={"graph": graph, "cns_steps": cns_steps, "omni7_meta": omni7_meta,
                                  "fly_causal": fly_causal, "vocab_base": int(cfg.vocab_size)})
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(("cns_full.", "omni7."))]
    if bad or unexpected:
        raise RuntimeError(f"Archimedes -> Expanse load mismatch: missing {bad[:5]} unexpected {list(unexpected)[:5]}")
    info = {"moe_layers": moe_layers_in_state_dict(sd), "slots_before": int(config.n_routed_experts) + int(config.moe_spare_experts),
            "slots_after": int(cfg.n_routed_experts) + int(cfg.moe_spare_experts), "n_new": int(n_new),
            "new_graft_tensors": len(missing)}
    return model, info


# ---------------------------------------------------------------------------
# 5. decoding (greedy, non-speculative, KV-cached; grafts held constant)
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode(model: MiMoMixModel, prompt_ids: torch.Tensor, max_new_tokens: int = 64,
                  eos_id: int = text_utils.EOS, **graft_kwargs) -> List[int]:
    """One-token-at-a-time greedy decode of a (1, T) prompt.

    ``graft_kwargs`` (``omni_features``, ``omni7_state``, ``use_fly`` ...) are
    passed to every forward; they are prompt-constant, so the cached step
    sees exactly what the full forward would (see test_expanse_core).
    """
    model.eval()
    out = model(prompt_ids, use_cache=True, return_mtp=False, past_length=0, **graft_kwargs)
    past = out.past_key_values
    pos = int(prompt_ids.shape[1])
    token = out.logits[:, -1].argmax(-1, keepdim=True)
    emitted: List[int] = []
    for _ in range(int(max_new_tokens)):
        t = int(token[0, 0])
        emitted.append(t)
        if t == eos_id or len(emitted) >= max_new_tokens:
            break
        out = model(token, past_key_values=past, use_cache=True, return_mtp=False, past_length=pos, **graft_kwargs)
        past = out.past_key_values
        pos += 1
        token = out.logits[:, -1].argmax(-1, keepdim=True)
    return emitted


# ---------------------------------------------------------------------------
# 6. checkpoint I/O
# ---------------------------------------------------------------------------


def save_expanse(path, model: ExpanseModel, tok: text_utils.WordTokenizer, extra: Dict[str, Any],
                 receipt: Dict[str, Any]) -> None:
    """Atomic save (``.tmp`` + ``os.replace``). ``payload['expanse']`` is the
    receipt merged with :meth:`ExpanseModel.expanse_spec` (spec wins)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exp = {k: v for k, v in dict(receipt or {}).items() if k != "graph"}
    exp.update(model.expanse_spec())
    payload = {
        "schema": EXP_SCHEMA,
        "base_schema": ARCH_SCHEMA,
        "config": model.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "tokenizer": tok.to_dict(),
        "extra": jsonable(extra or {}),
        "expanse": jsonable(exp),
    }
    staging = path.with_name(path.name + ".tmp")
    torch.save(payload, staging)
    os.replace(staging, path)


def tokenizer_from_dict(d: Dict[str, Any]):
    """The checkpoint's tokenizer: ``kind == "bpe"`` -> ``bpe_tokenizer.BPETokenizer``
    (v3, byte-level BPE), anything else -> Archimedes' ``WordTokenizer`` (v1/v2
    dicts carry no ``kind``). Both expose the same duck-typed API; the BPE module
    is imported only when needed."""
    if isinstance(d, dict) and d.get("kind") == "bpe":
        from bpe_tokenizer import BPETokenizer  # noqa: WPS433
        return BPETokenizer.from_dict(d)
    return text_utils.WordTokenizer.from_dict(d)


def load_expanse(path, map_location: str = "cpu") -> Tuple[ExpanseModel, text_utils.WordTokenizer, Dict[str, Any]]:
    """Rebuild module shapes from ``payload['expanse']`` and strict-load.

    The returned ``payload['state_dict']`` is the *model's* state dict (it
    shares storage with the parameters) so the file's copy can be freed
    instead of doubling peak memory.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != EXP_SCHEMA:
        raise ValueError(f"not a {EXP_SCHEMA} checkpoint (got {payload.get('schema')})")
    config = MiMoMixConfig(**payload["config"])
    spec = dict(payload.get("expanse") or {})
    model = ExpanseModel(config, fly_config=spec.get("fly_config"), with_omni=spec.get("with_omni", True),
                         expanse={**spec, "graph": None})
    model.load_state_dict(payload["state_dict"], strict=True)
    payload["state_dict"] = model.state_dict()
    model.eval()
    tok = tokenizer_from_dict(payload["tokenizer"])
    return model, tok, payload


__all__ = [
    "EXP_SCHEMA", "ARCH_SCHEMA", "PATHS", "GRAFT_CLASSES", "ExpanseModel", "grow_moe_slots", "fly_forward_causal",
    "expanse_from_archimedes", "graph_metadata", "skeleton_graph", "greedy_decode", "save_expanse", "load_expanse",
    "load_archimedes", "row_key", "jsonable", "moe_layers_in_state_dict", "text_utils", "ExpanseSparseMoE",
    "read_jsonl", "load_replay", "append_embedding_rows", "tokenizer_from_dict",
]
