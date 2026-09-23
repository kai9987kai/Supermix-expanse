"""Supermix Expanse v2: cross-source latent consolidation.

This module turns Expanse's existing grafts into a deeper, teacher-free runtime
path without pretending heterogeneous checkpoints can be weight-averaged.

The design has two halves:

* ''NativeConsolidationStack'' is part of the final model.  It inserts causal,
  per-token latent-MoE residual blocks after selected transformer layers.  Every
  output gate is initialised to *exactly* zero, so attaching the stack does not
  change Expanse logits at birth.  These blocks are the place where knowledge
  from the different source systems is consolidated.
* ''TeacherFusionBank'' exists only while training.  It projects heterogeneous
  source features (Archimedes hidden states, Qwen/BioMed donor experts, Omni
  states, FlyCore and the full connectome) into a shared latent space and learns
  an attention-weighted consensus target.  The student latent blocks learn that
  target.  The bank is discarded when the v2 checkpoint is written.

This is deliberately different from parameter averaging.  Qwen2.5-Coder,
BioMedLM, Omni and Archimedes do not share a common parameter coordinate system;
representation consolidation is well-defined where raw weight arithmetic is not.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


V2_SCHEMA = "supermix-expanse-v2"
DEFAULT_SOURCE_DIMS: Dict[str, int] = {
    "arch": 320,
    "qwen": 320,
    "biomedlm": 320,
    "omni7": 988,
    "omni": 128,
    "fly": 320,
    "cns": 320,
}
DEFAULT_DOMAINS = ("replay", "fly", "code", "bio", "connectome")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist() if x.numel() <= 4096 else f"<tensor {tuple(x.shape)} {x.dtype}>"
    return x


def _rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True).to(x.dtype) + eps)


def masked_last(hidden: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return one vector per sequence from ''hidden'' (B,T,H).

    ''lengths'' are 1-based counts (prompt lengths in Expanse).  Without them,
    the final position is used.  A (B,H) input is passed through unchanged.
    """
    if hidden.dim() == 2:
        return hidden
    if hidden.dim() != 3:
        raise ValueError(f"expected (B,H) or (B,T,H), got {tuple(hidden.shape)}")
    if lengths is None:
        return hidden[:, -1]
    idx = lengths.to(hidden.device, dtype=torch.long).clamp(1, hidden.shape[1]) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def consolidation_phase_weights(progress: float, bake_start: float, teacher_free_start: float) -> Dict[str, float]:
    """Representation-loss schedule with a genuinely teacher-free final phase."""
    p = min(max(float(progress), 0.0), 1.0)
    if not (0.0 < bake_start < teacher_free_start < 1.0):
        raise ValueError("need 0 < bake_start < teacher_free_start < 1")
    warm_end = min(0.15, bake_start * 0.5)
    if p < warm_end:
        q = p / max(warm_end, 1e-8)
        return {"distill": 0.25 + 0.75 * q, "align": 1.0, "teacher": 1.0}
    if p < bake_start:
        return {"distill": 1.0, "align": 0.35, "teacher": 1.0}
    if p < teacher_free_start:
        q = (p - bake_start) / max(1e-8, teacher_free_start - bake_start)
        return {"distill": 1.0 - 0.65 * q, "align": 0.0, "teacher": 1.0}
    return {"distill": 0.0, "align": 0.0, "teacher": 0.0}


@dataclass
class V2Config:
    """Runtime architecture for the native consolidation stack."""

    hidden_size: int = 320
    shared_dim: int = 512
    layers: Tuple[int, ...] = (1, 2, 4)
    latent_experts: int = 8
    latent_rank: int = 128
    latent_top_k: int = 2
    memory_slots: int = 24
    dropout: float = 0.0
    source_dims: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SOURCE_DIMS))
    domains: Tuple[str, ...] = DEFAULT_DOMAINS

    def validate(self) -> "V2Config":
        if self.hidden_size <= 0 or self.shared_dim <= 0:
            raise ValueError("hidden_size/shared_dim must be positive")
        if not self.layers or min(self.layers) < 0:
            raise ValueError("layers must contain non-negative indices")
        if self.latent_experts <= 0:
            raise ValueError("latent_experts must be positive")
        if not 1 <= self.latent_top_k <= self.latent_experts:
            raise ValueError("latent_top_k must be in [1, latent_experts]")
        if self.latent_rank <= 0 or self.memory_slots < 0:
            raise ValueError("latent_rank must be positive and memory_slots non-negative")
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["layers"] = list(self.layers)
        d["domains"] = list(self.domains)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "V2Config":
        kw = dict(d)
        if "layers" in kw:
            kw["layers"] = tuple(int(v) for v in kw["layers"])
        if "domains" in kw:
            kw["domains"] = tuple(str(v) for v in kw["domains"])
        if "source_dims" in kw:
            kw["source_dims"] = {str(k): int(v) for k, v in dict(kw["source_dims"]).items()}
        return cls(**kw).validate()


class SwiGLULatentExpert(nn.Module):
    """Small shared-space expert; intentionally much cheaper than a trunk layer."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.up = nn.Linear(dim, rank * 2, bias=False)
        self.down = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class NativeLatentBlock(nn.Module):
    """Per-token shared-latent MoE + learned memory, projected back to the trunk.

    The block is an *exact identity* at initialisation because ''out_gate'' is
    zero and we branch around the residual write when the gate is all zero.
    Its internal latent path is still trainable before the gate is opened.
    """

    def __init__(self, cfg: V2Config):
        super().__init__()
        self.cfg = cfg
        h, d = cfg.hidden_size, cfg.shared_dim
        self.in_norm = nn.RMSNorm(h)
        self.to_shared = nn.Linear(h, d, bias=False)
        self.shared_norm = nn.RMSNorm(d)
        self.router = nn.Linear(d, cfg.latent_experts, bias=True)
        self.experts = nn.ModuleList([SwiGLULatentExpert(d, cfg.latent_rank) for _ in range(cfg.latent_experts)])
        self.expert_scale = nn.Parameter(torch.tensor(0.25))
        self.dropout = nn.Dropout(float(cfg.dropout))

        if cfg.memory_slots:
            self.memory = nn.Parameter(torch.empty(cfg.memory_slots, d))
            nn.init.normal_(self.memory, std=1.0 / math.sqrt(d))
            self.memory_q = nn.Linear(d, d, bias=False)
            self.memory_v = nn.Linear(d, d, bias=False)
            self.memory_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(d))))
            self.memory_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid ~= 0.119, internal only
        else:
            self.register_parameter("memory", None)
            self.memory_q = None
            self.memory_v = None
            self.register_parameter("memory_logit_scale", None)
            self.register_parameter("memory_gate", None)

        self.out_norm = nn.RMSNorm(d)
        self.to_hidden = nn.Linear(d, h, bias=False)
        self.out_gate = nn.Parameter(torch.zeros(h))

    def _route(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(z)
        k = int(self.cfg.latent_top_k)
        if k < logits.shape[-1]:
            topv, topi = logits.topk(k, dim=-1)
            masked = torch.full_like(logits, -torch.inf)
            masked.scatter_(-1, topi, topv)
            probs = torch.softmax(masked, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
        # Evaluating all tiny experts is cheaper and much easier to vectorise than
        # Python dispatch; top-k probabilities make unused experts contribute zero.
        outs = torch.stack([e(z) for e in self.experts], dim=-2)  # (...,E,D)
        mixed = torch.sum(probs.unsqueeze(-1) * outs, dim=-2)
        return mixed, probs

    def _memory(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.memory is None:
            return z, None
        q = self.memory_q(z)
        scale = self.memory_logit_scale.exp().clamp(max=1.0)
        attn = torch.softmax(torch.matmul(q, self.memory.t()) * scale, dim=-1)
        values = self.memory_v(self.memory)
        mem = torch.matmul(attn, values)
        return z + torch.sigmoid(self.memory_gate) * mem, attn

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z0 = self.to_shared(self.in_norm(hidden))
        z = self.shared_norm(z0)
        expert, route = self._route(z)
        z = z0 + self.expert_scale.tanh() * self.dropout(expert)
        z, memory_attn = self._memory(z)
        latent = self.out_norm(z)

        # Preserve the exact v1 function at birth.  At inference a closed gate
        # skips the projection entirely.  In training the write MUST be computed
        # even while the gate is exactly zero: otherwise ''out_gate'' is not in
        # the graph, receives no gradient and can never open, and the native
        # block never affects the model.  ''hidden + 0 * write'' equals ''hidden''
        # exactly for finite ''write'', so birth is still function-preserving.
        gate_closed = not torch.jit.is_scripting() and bool((self.out_gate.detach() == 0).all())
        needs_grad = self.training and torch.is_grad_enabled() and self.out_gate.requires_grad
        if gate_closed and not needs_grad:
            out = hidden
        else:
            write = self.to_hidden(latent)
            out = hidden + self.out_gate * write

        route_mean = route.float().reshape(-1, route.shape[-1]).mean(0)
        balance = route.shape[-1] * route_mean.pow(2).sum()
        entropy = -(route.clamp_min(1e-9).log() * route).sum(-1).float().mean()
        aux: Dict[str, torch.Tensor] = {
            "latent": latent,
            "route": route,
            "router_balance": balance,
            "router_entropy": entropy,
            "gate_abs_mean": self.out_gate.detach().abs().mean(),
        }
        if memory_attn is not None:
            aux["memory_attn"] = memory_attn
            aux["memory_entropy"] = -(memory_attn.clamp_min(1e-9).log() * memory_attn).sum(-1).float().mean()
        return out, aux


class NativeConsolidationStack(nn.Module):
    """Attach ''NativeLatentBlock'' instances to selected Expanse layers.

    Hooks operate only on each position's hidden vector, so they preserve the
    causal/KV-cache contract.  The stack keeps the latest pre-write hidden and
    latent tensors for the trainer; those tensors are not part of checkpoints.
    """

    def __init__(self, cfg: V2Config):
        super().__init__()
        self.cfg = cfg.validate()
        self.blocks = nn.ModuleDict({str(i): NativeLatentBlock(cfg) for i in cfg.layers})
        self.last: Dict[int, Dict[str, torch.Tensor]] = {}
        self._handles: List[Any] = []
        self._attached = False

    def _hook(self, layer_idx: int):
        def fn(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                new_hidden, aux = self.blocks[str(layer_idx)](hidden)
                self.last[layer_idx] = {"hidden": hidden, **aux}
                return (new_hidden, *output[1:])
            if isinstance(output, list):
                hidden = output[0]
                new_hidden, aux = self.blocks[str(layer_idx)](hidden)
                self.last[layer_idx] = {"hidden": hidden, **aux}
                return [new_hidden, *output[1:]]
            if torch.is_tensor(output):
                new_hidden, aux = self.blocks[str(layer_idx)](output)
                self.last[layer_idx] = {"hidden": output, **aux}
                return new_hidden
            raise TypeError(f"unsupported layer output type {type(output)!r}")
        return fn

    def attach(self, model: nn.Module) -> "NativeConsolidationStack":
        if self._attached:
            return self
        layers = getattr(model, "layers", None)
        if layers is None:
            raise AttributeError("model has no .layers for consolidation hooks")
        if max(self.cfg.layers) >= len(layers):
            raise ValueError(f"requested layer {max(self.cfg.layers)}, model has {len(layers)} layers")
        for i in self.cfg.layers:
            self._handles.append(layers[i].register_forward_hook(self._hook(i)))
        self._attached = True
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._attached = False
        self.last.clear()

    def aux_loss(self) -> torch.Tensor:
        vals = [v["router_balance"] for v in self.last.values() if "router_balance" in v]
        if not vals:
            p = next(self.parameters())
            return p.new_zeros(())
        # Minimum is 1.0 for perfectly balanced routing.
        return torch.stack(vals).mean() - 1.0

    def report(self) -> Dict[str, float]:
        rep: Dict[str, float] = {}
        for i in self.cfg.layers:
            b = self.blocks[str(i)]
            rep[f"v2_l{i}_gate_abs_mean"] = float(b.out_gate.detach().abs().mean())
            if i in self.last:
                rep[f"v2_l{i}_route_entropy"] = float(self.last[i]["router_entropy"].detach())
                if "memory_entropy" in self.last[i]:
                    rep[f"v2_l{i}_memory_entropy"] = float(self.last[i]["memory_entropy"].detach())
        return rep


class SourceProjector(nn.Module):
    """Low-rank heterogeneous feature -> common latent projector."""

    def __init__(self, in_dim: int, shared_dim: int, rank: int = 192):
        super().__init__()
        rank = min(int(rank), int(in_dim), int(shared_dim))
        self.norm = nn.RMSNorm(in_dim)
        self.a = nn.Linear(in_dim, rank, bias=False)
        self.b = nn.Linear(rank, shared_dim, bias=False)
        self.refine = nn.Sequential(
            nn.RMSNorm(shared_dim),
            nn.Linear(shared_dim, rank * 2, bias=False),
            nn.SiLU(),
            nn.Linear(rank * 2, shared_dim, bias=False),
        )
        self.refine_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.b(F.silu(self.a(self.norm(x))))
        return _rms(z + self.refine_scale.tanh() * self.refine(z))


class TeacherFusionBank(nn.Module):
    """Training-only heterogeneous source alignment and confidence router."""

    def __init__(self, cfg: V2Config, projector_rank: int = 192):
        super().__init__()
        self.cfg = cfg
        self.source_names = tuple(cfg.source_dims)
        self.source_index = {n: i for i, n in enumerate(self.source_names)}
        self.domain_index = {n: i for i, n in enumerate(cfg.domains)}
        d = cfg.shared_dim
        self.projectors = nn.ModuleDict({
            n: SourceProjector(int(dim), d, projector_rank) for n, dim in cfg.source_dims.items()
        })
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.source_bias = nn.Parameter(torch.zeros(len(self.source_names)))
        self.domain_bias = nn.Parameter(torch.zeros(len(cfg.domains), len(self.source_names)))
        self.temperature_log = nn.Parameter(torch.tensor(0.0))
        self._init_domain_priors()

    @torch.no_grad()
    def _init_domain_priors(self) -> None:
        # Weak priors only break symmetry; the router remains fully trainable.
        pairs = {
            ("replay", "arch"): 0.25,
            ("replay", "omni7"): 0.15,
            ("code", "qwen"): 0.50,
            ("bio", "biomedlm"): 0.50,
            ("fly", "fly"): 0.50,
            ("connectome", "cns"): 0.50,
        }
        for (dom, src), v in pairs.items():
            if dom in self.domain_index and src in self.source_index:
                self.domain_bias[self.domain_index[dom], self.source_index[src]] = v

    def project(self, sources: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, x in sources.items():
            if name not in self.projectors:
                continue
            if x.dim() == 3:
                x = x.mean(dim=1)
            if x.dim() != 2:
                raise ValueError(f"source {name} must be (B,D) or (B,T,D), got {tuple(x.shape)}")
            expected = self.cfg.source_dims[name]
            if x.shape[-1] != expected:
                raise ValueError(f"source {name}: expected {expected} dims, got {x.shape[-1]}")
            out[name] = self.projectors[name](x.float())
        if not out:
            raise ValueError("no recognised source features")
        return out

    def forward(
        self,
        sources: Mapping[str, torch.Tensor],
        student_latent: torch.Tensor,
        *,
        lengths: Optional[torch.Tensor] = None,
        domains: Optional[Sequence[str]] = None,
        confidence: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        projected = self.project(sources)
        names = list(projected)
        stack = torch.stack([projected[n] for n in names], dim=1)  # B,N,D
        student = _rms(masked_last(student_latent, lengths).float())
        q = self.query(student).unsqueeze(1)
        k = self.key(stack)
        temp = self.temperature_log.exp().clamp(0.25, 4.0)
        scores = (q * k).sum(-1) / math.sqrt(stack.shape[-1]) / temp
        idx = torch.tensor([self.source_index[n] for n in names], device=scores.device)
        scores = scores + self.source_bias.index_select(0, idx)

        if domains is not None:
            if len(domains) != scores.shape[0]:
                raise ValueError("domains length must equal batch size")
            did = torch.tensor([self.domain_index.get(str(d), 0) for d in domains], device=scores.device)
            db = self.domain_bias.index_select(0, did).index_select(1, idx)
            scores = scores + db
        if confidence:
            for j, n in enumerate(names):
                if n in confidence:
                    c = confidence[n].to(scores.device, scores.dtype).reshape(-1).clamp_min(1e-4)
                    scores[:, j] = scores[:, j] + c.log()

        weights = torch.softmax(scores, dim=-1)
        fused = torch.sum(weights.unsqueeze(-1) * stack, dim=1)
        return {"fused": _rms(fused), "projected": projected, "weights": weights, "names": names, "scores": scores}

    @staticmethod
    def alignment_loss(projected: Mapping[str, torch.Tensor], temperature: float = 0.12) -> torch.Tensor:
        """Multi-view InfoNCE + variance floor; prevents trivial projector collapse."""
        vals = list(projected.values())
        if len(vals) < 2:
            return vals[0].new_zeros(()) if vals else torch.tensor(0.0)
        losses: List[torch.Tensor] = []
        for i in range(len(vals)):
            a = F.normalize(vals[i], dim=-1)
            # A per-feature variance floor is useful even for B=1 (std unavailable
            # then, so it is simply omitted).
            if a.shape[0] > 1:
                std = vals[i].float().std(dim=0, unbiased=False)
                losses.append(0.02 * F.relu(0.5 - std).mean())
            for j in range(i + 1, len(vals)):
                b = F.normalize(vals[j], dim=-1)
                if a.shape[0] > 1:
                    logits = a @ b.t() / temperature
                    labels = torch.arange(a.shape[0], device=a.device)
                    losses.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)))
                else:
                    losses.append(1.0 - (a * b).sum(-1).mean())
        return torch.stack(losses).mean()

    @staticmethod
    def student_loss(student_latent: torch.Tensor, fused_target: torch.Tensor,
                     lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        student = _rms(masked_last(student_latent, lengths).float())
        target = _rms(fused_target.detach().float())
        # RMS-normalised vectors have norm sqrt(dim) (~22.6 at 512-d), so the
        # cosine and batch-geometry terms must use unit vectors: on RMS vectors
        # "1 - dot" ranges over +-dim and the Gram MSE over ~dim^2, which made the
        # distill term ~300x the language loss.
        s_unit, t_unit = F.normalize(student, dim=-1), F.normalize(target, dim=-1)
        cosine = 1.0 - (s_unit * t_unit).sum(-1).mean()
        smooth = F.smooth_l1_loss(student, target)
        if student.shape[0] > 1:
            rel_s = s_unit @ s_unit.t()
            rel_t = t_unit @ t_unit.t()
            relational = F.mse_loss(rel_s, rel_t)
        else:
            relational = student.new_zeros(())
        return {"cosine": cosine, "smooth": smooth, "relational": relational,
                "total": cosine + 0.5 * smooth + 0.1 * relational}


class InternalSourceExtractor:
    """Read source-specific signals already present inside a trained Expanse.

    Extraction is intentionally prompt-level and normally called under
    ''torch.no_grad''.  That keeps consolidation targets stable and avoids
    retaining a second autograd graph through the donor systems.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def _slots(self, donor: str, layer_idx: int) -> List[int]:
        recs = getattr(self.model, "donor_receipts", {}) or {}
        rec = recs.get(donor, {})
        layers = rec.get("layers", {}) if isinstance(rec, Mapping) else {}
        info = layers.get(str(layer_idx), layers.get(layer_idx, {}))
        return [int(s) for s in (info or {}).get("slots", [])]

    def _donor(self, hidden: torch.Tensor, donor: str, layer_idx: int) -> Optional[torch.Tensor]:
        slots = self._slots(donor, layer_idx)
        if not slots:
            return None
        mlp = self.model.layers[layer_idx].mlp
        outs = [mlp.experts[s](hidden) for s in slots]
        return torch.stack(outs, dim=0).mean(0) if outs else None

    @torch.no_grad()
    def collect(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        lengths: torch.Tensor,
        *,
        omni_features: Optional[torch.Tensor] = None,
        omni7_state: Optional[torch.Tensor] = None,
        include_slow: bool = True,
    ) -> Dict[str, torch.Tensor]:
        pooled = masked_last(hidden, lengths).detach()
        src: Dict[str, torch.Tensor] = {"arch": pooled}
        q = self._donor(pooled, "qwen", layer_idx)
        b = self._donor(pooled, "biomedlm", layer_idx)
        if q is not None:
            src["qwen"] = q
        if b is not None:
            src["biomedlm"] = b
        if omni7_state is not None:
            src["omni7"] = omni7_state.detach()
        if omni_features is not None:
            src["omni"] = omni_features.detach()

        # FlyCore and connectome signals are evaluated on only one prompt vector
        # per sample, so this is far cheaper than rerunning them for every token.
        if include_slow:
            try:
                import expanse_core as ec  # local import keeps this module standalone in tests
                if getattr(self.model, "fly_core", None) is not None:
                    fo, _ = ec.fly_forward_causal(self.model.fly_core, pooled.unsqueeze(1))
                    src["fly"] = (fo[:, 0] - pooled).detach()
            except Exception:
                pass
            cns = getattr(self.model, "cns_full", None)
            if cns is not None:
                try:
                    co, _ = cns(pooled.unsqueeze(1), token_mask=None)
                    src["cns"] = (co[:, 0] - pooled).detach()
                except Exception:
                    pass
        return src


def attach_consolidation_v2(model: nn.Module, cfg: V2Config) -> NativeConsolidationStack:
    """Register the stack as ''model.consolidation_v2'' and attach its hooks."""
    existing = getattr(model, "consolidation_v2", None)
    if existing is not None:
        if isinstance(existing, NativeConsolidationStack):
            existing.attach(model)
            return existing
        raise TypeError("model.consolidation_v2 already exists with a different type")
    stack = NativeConsolidationStack(cfg)
    # Assign first so its parameters become part of model.state_dict().  The
    # stack deliberately stores no reference back to the model (no module cycle).
    model.add_module("consolidation_v2", stack)
    stack.attach(model)
    return stack


def save_v2(path: str | Path, model: nn.Module, tok: Any, extra: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    """Write a self-contained v2 checkpoint.  Training-only fusion weights are omitted."""
    try:
        import expanse_core as ec
    except ImportError as e:  # pragma: no cover - repository runtime only
        raise RuntimeError("save_v2 must run from the Expanse repository") from e
    stack = getattr(model, "consolidation_v2", None)
    if not isinstance(stack, NativeConsolidationStack):
        raise ValueError("model has no NativeConsolidationStack attached")
    exp = {k: v for k, v in dict(receipt or {}).items() if k != "graph"}
    exp.update(model.expanse_spec())
    payload = {
        "schema": V2_SCHEMA,
        "base_schema": ec.EXP_SCHEMA,
        "config": model.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "tokenizer": tok.to_dict(),
        "extra": _jsonable(dict(extra or {})),
        "expanse": _jsonable(exp),
        "consolidation_v2": stack.cfg.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_v2(path: str | Path, map_location: str = "cpu") -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """Rebuild Expanse + native consolidation stack and strict-load v2 weights."""
    try:
        import expanse_core as ec
    except ImportError as e:  # pragma: no cover - repository runtime only
        raise RuntimeError("load_v2 must run from the Expanse repository") from e
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != V2_SCHEMA:
        raise ValueError(f"not a {V2_SCHEMA} checkpoint (got {payload.get('schema')})")
    cfg_base = ec.MiMoMixConfig(**payload["config"])
    spec = dict(payload.get("expanse") or {})
    model = ec.ExpanseModel(cfg_base, fly_config=spec.get("fly_config"), with_omni=spec.get("with_omni", True),
                            expanse={**spec, "graph": None})
    v2cfg = V2Config.from_dict(payload["consolidation_v2"])
    stack = attach_consolidation_v2(model, v2cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    payload["state_dict"] = model.state_dict()
    model.eval()
    tok = ec.text_utils.WordTokenizer.from_dict(payload["tokenizer"])
    return model, tok, payload


__all__ = [
    "V2_SCHEMA", "DEFAULT_SOURCE_DIMS", "DEFAULT_DOMAINS", "V2Config", "NativeLatentBlock",
    "NativeConsolidationStack", "TeacherFusionBank", "InternalSourceExtractor", "masked_last",
    "consolidation_phase_weights",
    "attach_consolidation_v2", "save_v2", "load_v2",
]
