"""Supermix Archimedes: one model grafted from four checkpoints and a fly lab.

Trunk
    ``Kai9987kai/supermix-v93`` (36.6M, 320-d, 6 layers, MoE 72 slots/layer,
    recursive thinking core, 832-node male-CNS connectome core). Loaded
    verbatim; every graft below is function-preserving at birth, so the
    ungrafted trunk's outputs are unchanged until fine-tuning opens a gate.

Grafts
    v87 experts    ``Kai9987kai/supermix-v87`` is 256-d; its vocabulary is a
                   strict subset of v93's. Two least-squares maps fitted on the
                   8,635 shared token embeddings lift its most-used experts
                   (lowest router bias) into v93's 12 dead expert slots per
                   MoE layer, router rows included.
    fly_core       An exact PyTorch port of the learned part of the Fly Lab's
                   22-brain syncytium (kai9987kai/FLY-DIAMOND-NEXUS): per-brain
                   AL->KC and KC->MBON synapses, 24 born Kenyon cells per brain
                   from neurogenesis, the APL loop, the 22x22x4 commissural
                   matrix and the gated descending consensus, initialised from
                   the v8 snapshot of a 10-minute headless run. Reads the trunk
                   through a sensory projection (320 -> 14 glomeruli) and writes
                   its 22x4 descending votes + 4 consensus logits back through
                   a zero gate.
    cns nodes      The same 22 brains also become 22 nodes in the connectome
                   core (22 of its 28 free slots), wired to each other by the
                   commissural matrix and to their biological analogues in the
                   real male-CNS graph (ORN/PAM for the forager, EPG for the
                   navigator, KC for the vault, ...) at synaptogenesis strength.
    omni_core      ``omni-collective-v48-frontier`` and
                   ``supermix-v38-native-image-xlite`` are ChampionNet models on
                   a 128-d hashed prompt feature. They share no ancestry (mean
                   weight cosine 0.16) so they are kept as two encoders, with
                   v48's hierarchical-MoE 10-way head and v38's 64x64 native
                   image decoder intact, and their pooled states bridged into
                   the trunk through a zero gate.

Checkpoint schema ``supermix-archimedes-v1`` = the v57 talk schema plus
``archimedes`` (graft receipts, fly snapshot digest, omni configs).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE / "champion") not in sys.path:
    sys.path.insert(0, str(HERE / "champion"))

import mimomix_text as text_utils  # noqa: E402
from mimomix_core import MiMoMixConfig, MiMoMixModel  # noqa: E402

SCHEMA = "supermix-archimedes-v1"
TALK_SCHEMA = "supermix-v57-talk-checkpoint-v1"

# ---------------------------------------------------------------------------
# Fly Lab constants (mirrors fly-brain-engine.js)
# ---------------------------------------------------------------------------
FLY_ROLES_16 = [
    "forager", "navigator", "sentinel", "vault", "pioneer", "optic", "executive", "metabolic",
    "lal", "vnc_cpg", "ammc", "pb", "eb", "no", "aotu", "smp",
]
FLY_ROLES_22 = FLY_ROLES_16 + ["plume", "climate", "threat_forecast", "route_memory", "uncertainty", "social"]
DESCENDING_GROUPS = {
    "sentinel": "reflex", "smp": "reflex", "forager": "goal", "navigator": "goal", "executive": "goal",
    "metabolic": "goal", "optic": "goal", "pioneer": "explore", "lal": "explore", "aotu": "explore",
    "eb": "explore", "pb": "explore", "vault": "support", "vnc_cpg": "support", "ammc": "support",
    "no": "support", "plume": "goal", "climate": "reflex", "threat_forecast": "reflex",
    "route_memory": "goal", "uncertainty": "explore", "social": "support",
}
GROUP_INDEX = {"reflex": 0, "goal": 1, "explore": 2, "support": 3}
ROLE_WEIGHTS = {
    "forager": 1.2, "navigator": 1.1, "sentinel": 1.4, "vault": 1.0, "pioneer": 0.9, "optic": 1.15,
    "executive": 1.35, "metabolic": 1.25, "lal": 1.1, "vnc_cpg": 1.05, "ammc": 1.15, "pb": 1.2,
    "eb": 1.15, "no": 1.05, "aotu": 1.1, "smp": 1.3,
}
ACTION_ANGLES = [-math.pi / 2, math.pi / 2, math.pi, 0.0]  # up, down, left, right
FLY_ACTIONS = ["up", "down", "left", "right"]
GLOMERULI = 14
KC_BASE = 32
MBON = 4

# Biological analogues of each fly-lab brain in the male-CNS module labels.
# Used to wire the grafted CNS nodes to real modules; first pattern that hits
# wins, with a role-level fallback.
CNS_ANALOGUES: Dict[str, List[str]] = {
    "forager": [r"ORN_", r"PAM\d"], "navigator": [r":EPG@", r"Delta7"], "sentinel": [r":LHPV", r":LHAD", r":LHAV"],
    "vault": [r":KC"], "pioneer": [r"PAM\d", r":PPL1"], "optic": [r"visual:T4", r"visual:T5", r"visual:LC\d"],
    "executive": [r":FB\d", r"vDelta", r"hDelta"], "metabolic": [r":GNG\d"], "lal": [r":LAL\d"],
    "vnc_cpg": [r"descending:DN", r"^vnc:"], "ammc": [r"sensory:JO", r":AMMC", r":WED"],
    "pb": [r":PFN", r"Delta7"], "eb": [r":ER\d"], "no": [r":PFNa", r":LNO"], "aotu": [r":AOTU"],
    "smp": [r":SMP\d"], "plume": [r"ORN_", r"visual:LC\d"], "climate": [r"^sensory:"],
    "threat_forecast": [r":LPLC", r"vnc:GF", r"descending:DNp"], "route_memory": [r":KC", r":MBON"],
    "uncertainty": [r":MBON", r":PPL1", r":APL", r"PAM\d"], "social": [r":P1", r":pC1", r":aSP", r":aIPg", r"sensory:JO"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-6)
    return x + torch.log(-torch.expm1(-x))


# ---------------------------------------------------------------------------
# 1. v87 -> v93 expert lifting
# ---------------------------------------------------------------------------
@dataclass
class LiftMaps:
    """Least-squares maps between two residual spaces, anchored on shared tokens.

    ``to_wide`` (d87 x d93): h93 ~ h87 @ to_wide.  ``to_narrow`` (d93 x d87):
    h87 ~ h93 @ to_narrow. Fitted on the embedding rows of every token the two
    vocabularies share, in the row-vector convention the models use.
    """

    to_wide: torch.Tensor
    to_narrow: torch.Tensor
    shared_tokens: int
    fit_r2_wide: float
    fit_r2_narrow: float

    @staticmethod
    def fit(e_narrow: torch.Tensor, e_wide: torch.Tensor, ridge: float = 1e-3) -> "LiftMaps":
        a, b = e_narrow.double(), e_wide.double()

        def solve(x, y):
            xtx = x.T @ x + ridge * torch.eye(x.shape[1], dtype=x.dtype)
            w = torch.linalg.solve(xtx, x.T @ y)
            resid = y - x @ w
            r2 = 1.0 - float((resid ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
            return w.float(), r2

        to_wide, r2w = solve(a, b)
        to_narrow, r2n = solve(b, a)
        return LiftMaps(to_wide, to_narrow, int(a.shape[0]), r2w, r2n)

    def lift_in(self, w: torch.Tensor) -> torch.Tensor:
        """Weight consuming a narrow hidden (out x d87) -> consuming a wide one (out x d93)."""
        return w @ self.to_narrow.T

    def lift_out(self, w: torch.Tensor) -> torch.Tensor:
        """Weight producing a narrow hidden (d87 x in) -> producing a wide one (d93 x in)."""
        return self.to_wide.T @ w


def token_index_map(tok_from: text_utils.WordTokenizer, tok_to: text_utils.WordTokenizer) -> Tuple[List[int], List[int]]:
    """Ids of every token in ``tok_from`` that also exists in ``tok_to``, paired."""
    lookup = {t: i for i, t in enumerate(tok_to.tokens)}
    src, dst = [], []
    for i, t in enumerate(tok_from.tokens):
        j = lookup.get(t)
        if j is not None:
            src.append(i)
            dst.append(j)
    return src, dst


@torch.no_grad()
def graft_v87_experts(
    model: MiMoMixModel,
    donor: MiMoMixModel,
    tok_model: text_utils.WordTokenizer,
    tok_donor: text_utils.WordTokenizer,
    per_layer: int = 12,
    layer_plan: Optional[Dict[int, Tuple[int, int]]] = None,
    dormant_margin: float = 1.5,
) -> Dict[str, Any]:
    """Fill the trunk's dead expert slots with the donor's most-used experts.

    ``layer_plan`` maps trunk MoE layer -> (donor MoE layer, rank offset). The
    donor's router bias ranks experts: DeepSeek-style loss-free balancing lowers
    the bias of an over-used expert, so the lowest biases are the most-used.

    Selection is ``softmax(router) + expert_bias`` and the bias (~22) dwarfs the
    score (in [0, 1]), so a grafted expert at the alive median would win top-k
    on its lifted router row alone and displace a trained expert: measured, that
    took v93 from 0.938 to 0.229. Even a bias no score can overcome leaves the
    extra logits in the router softmax, which shifts every other score and
    flips marginal top-k decisions (max logit diff 3.2). Each grafted expert
    is therefore born *dormant*: weights and router row in place, ``alive`` 0,
    so routing is bit-for-bit v93's. The trainer wakes them (``alive`` 1) at
    ``dormant_bias`` (``dormant_margin`` below the lowest alive bias) and
    anneals the bias up to ``target_bias`` (the alive median), so they enter
    routing gradually while the trunk is fine-tuned around them.
    """

    src_ids, dst_ids = token_index_map(tok_donor, tok_model)
    e_donor = donor.embed_tokens.weight[src_ids]
    e_model = model.embed_tokens.weight[dst_ids]
    maps = LiftMaps.fit(e_donor, e_model)

    donor_moe = [i for i, l in enumerate(donor.layers) if hasattr(l.mlp, "experts")]
    model_moe = [i for i, l in enumerate(model.layers) if hasattr(l.mlp, "experts")]
    if layer_plan is None:
        layer_plan = {}
        for k, li in enumerate(model_moe):
            dj = donor_moe[min(k, len(donor_moe) - 1)]
            offset = per_layer * max(0, k - (len(donor_moe) - 1))
            layer_plan[li] = (dj, offset)

    receipt: Dict[str, Any] = {
        "shared_tokens": maps.shared_tokens, "fit_r2_to_wide": maps.fit_r2_wide,
        "fit_r2_to_narrow": maps.fit_r2_narrow, "layers": {},
    }
    for li, (dj, offset) in layer_plan.items():
        mlp, dmlp = model.layers[li].mlp, donor.layers[dj].mlp
        dead = (mlp.expert_alive == 0).nonzero().flatten().tolist()[:per_layer]
        order = dmlp.expert_bias.argsort().tolist()
        chosen = order[offset: offset + len(dead)]
        alive_bias = mlp.expert_bias[mlp.expert_alive.bool()]
        target_bias = float(alive_bias.median()) if alive_bias.numel() else 0.0
        fill_bias = float(alive_bias.min()) - float(dormant_margin) if alive_bias.numel() else -float(dormant_margin)
        placed = []
        for slot, de in zip(dead, chosen):
            se, dexp = mlp.experts[slot], dmlp.experts[de]
            se.gate_proj.weight.copy_(maps.lift_in(dexp.gate_proj.weight))
            se.up_proj.weight.copy_(maps.lift_in(dexp.up_proj.weight))
            se.down_proj.weight.copy_(maps.lift_out(dexp.down_proj.weight))
            mlp.gate.weight[slot].copy_(maps.lift_in(dmlp.gate.weight[de: de + 1])[0])
            mlp.expert_bias[slot] = fill_bias
            mlp.expert_alive[slot] = 0  # dormant until the trainer wakes it
            placed.append({"slot": slot, "donor_expert": de, "donor_bias": float(dmlp.expert_bias[de])})
        receipt["layers"][li] = {"donor_layer": dj, "placed": placed, "alive_after": int(mlp.expert_alive.sum()),
                                 "dormant": len(placed),
                                 "dormant_bias": fill_bias, "target_bias": target_bias, "slots": [p["slot"] for p in placed]}
    return receipt


@torch.no_grad()
def wake_grafted_experts(model: MiMoMixModel, receipt: Dict[str, Any], progress: float) -> Dict[str, float]:
    """Trainer hook: at ``progress`` in [0, 1] set every grafted expert alive with
    its bias interpolated from ``dormant_bias`` to ``target_bias``."""
    progress = float(min(1.0, max(0.0, progress)))
    out = {}
    for li, info in receipt["layers"].items():
        mlp = model.layers[int(li)].mlp
        bias = info["dormant_bias"] + (info["target_bias"] - info["dormant_bias"]) * progress
        for slot in info["slots"]:
            mlp.expert_alive[slot] = 1
            mlp.expert_bias[slot] = bias
        out[str(li)] = bias
    return out


# ---------------------------------------------------------------------------
# 2. Fly core: the 22-brain syncytium as a PyTorch module
# ---------------------------------------------------------------------------
def load_fly_snapshot(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        snap = json.load(f)
    if snap.get("version") != "8.0.0":
        raise ValueError(f"expected a v8 (22-brain) Fly Lab snapshot, got version {snap.get('version')}")
    if snap.get("brainCount") != 22 or list(snap.get("roles", [])) != FLY_ROLES_22:
        raise ValueError("snapshot is not the 22-role syncytium")
    return snap


class FlyCore(nn.Module):
    """The learned, stateless part of ``TwentyTwoFlyBrainSyncytium.step``.

    Per brain: AL divisive normalisation -> AL->KC synapses (32 base + born
    Kenyon cells) -> APL feedback loop (8 damped iterations) -> KC->MBON ->
    innate obs-dependent drives (hazard avoidance, giant-fibre escape, bilateral
    tropotaxis, appetitive attraction, sugar drive) -> descending (4). Then the
    commissural cross-talk and the state-gated consensus, scaled by 16/22.

    The simulation's stateful registers (ring attractor, LAL flip-flop, CPG
    gait, SMP latch, odometry, circadian PDF) cannot live per token; their mean
    contribution is absorbed into ``role_bias`` (22 x 4) and ``vigor`` (22),
    both fitted on the run's experience log at graft time and trainable after.
    """

    def __init__(self, hidden_size: int, n_brains: int = 22, n_born: int = 24, config: Optional[Dict[str, float]] = None):
        super().__init__()
        cfg = config or {}
        self.n_brains, self.n_born = n_brains, n_born
        self.apl_feedback = float(cfg.get("aplFeedbackGain", 3.4))
        self.apl_divisive = float(cfg.get("aplDivisiveGain", 2.6))
        self.apl_subtractive = float(cfg.get("aplSubtractiveGain", 0.55))
        self.al_to_kc = nn.Parameter(torch.zeros(n_brains, KC_BASE, GLOMERULI))
        self.kc_to_mbon = nn.Parameter(torch.full((n_brains, KC_BASE, MBON), 0.25))
        self.born_al = nn.Parameter(torch.zeros(n_brains, n_born, GLOMERULI))
        self.born_mbon = nn.Parameter(torch.zeros(n_brains, n_born, MBON))
        self.register_buffer("born_alive", torch.zeros(n_brains, n_born))
        self.commissural = nn.Parameter(torch.zeros(n_brains, n_brains, MBON))  # [src, dst, action]
        self.role_bias = nn.Parameter(torch.zeros(n_brains, MBON))
        self.vigor = nn.Parameter(torch.ones(n_brains))
        self.register_buffer("role_weight", torch.tensor([ROLE_WEIGHTS.get(r, 1.0) for r in FLY_ROLES_22]))
        self.register_buffer("group", torch.tensor([GROUP_INDEX[DESCENDING_GROUPS[r]] for r in FLY_ROLES_22]))
        self.register_buffer("sensory_scale", self._sensory_scale())
        self.register_buffer("odor_roles", torch.tensor([r in ("forager", "navigator", "metabolic") for r in FLY_ROLES_22], dtype=torch.float32))
        self.register_buffer("metabolic_role", torch.tensor([r == "metabolic" for r in FLY_ROLES_22], dtype=torch.float32))
        self.register_buffer("action_angles", torch.tensor(ACTION_ANGLES))
        # bridge to the trunk
        self.norm = nn.LayerNorm(hidden_size)
        self.sensory_proj = nn.Linear(hidden_size, GLOMERULI)
        self.out_dim = n_brains * MBON + MBON
        self.to_trunk = nn.Linear(self.out_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.role_names = list(FLY_ROLES_22)

    @staticmethod
    def _sensory_scale() -> torch.Tensor:
        s = torch.ones(len(FLY_ROLES_22), GLOMERULI)
        s[FLY_ROLES_22.index("forager"), 4] = 1.3
        s[FLY_ROLES_22.index("sentinel"), 3] = 1.5
        return s

    @torch.no_grad()
    def load_snapshot(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        brains = snap["brains"]
        exact = snap["exactState"]
        n = self.n_brains
        for i in range(n):
            al = torch.tensor(exact["alToKcWeights"][i], dtype=torch.float32).view(KC_BASE, GLOMERULI)
            self.al_to_kc[i].copy_(al)
            self.kc_to_mbon[i].copy_(torch.tensor(brains[i]["kcToMbonWeights"], dtype=torch.float32).view(KC_BASE, MBON))
            born = brains[i]["bornNeurons"][: self.n_born]
            self.born_al[i].zero_(); self.born_mbon[i].zero_(); self.born_alive[i].zero_()
            for b, neuron in enumerate(born):
                self.born_al[i, b].copy_(torch.tensor(neuron["weights"], dtype=torch.float32))
                self.born_mbon[i, b].copy_(torch.tensor(neuron["mbonWeights"], dtype=torch.float32))
                self.born_alive[i, b] = 1.0
        comm = torch.tensor(snap["commissuralWeights"], dtype=torch.float32).view(n, n, MBON)
        self.commissural.copy_(comm)
        return {
            "tick": snap.get("tickCount"), "seed": snap.get("worldSeed"), "mitosis": snap.get("mitosisEvents"),
            "born_total": int(self.born_alive.sum()), "commissural_l1": float(comm.abs().sum()),
            "kc_to_mbon_mean": float(self.kc_to_mbon.mean()), "al_to_kc_nonzero": int((self.al_to_kc != 0).sum()),
        }

    def brains_forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """obs: (B, 14) -> descending (B, 22, 4), consensus (B, 4), probs (B, 4)."""
        B = obs.shape[0]
        x = obs.unsqueeze(1) * self.sensory_scale  # (B, 22, 14) role-modulated sensory
        al = F.relu(x)
        al = al / (0.2 + al.mean(-1, keepdim=True))
        kc = torch.einsum("bri,rki->brk", al, self.al_to_kc)
        born = torch.einsum("bri,rki->brk", al, self.born_al) * self.born_alive
        pool = torch.cat([kc, born], dim=-1)  # (B, 22, 56)
        n_pool = float(KC_BASE) + self.born_alive.sum(-1)  # (22,)
        apl = torch.zeros(B, self.n_brains, 1, device=obs.device, dtype=obs.dtype)
        for _ in range(8):
            v = pool / (1 + self.apl_divisive * apl) - self.apl_subtractive * apl
            s = F.relu(v).sum(-1, keepdim=True) / n_pool.view(1, -1, 1)
            apl = F.relu(apl + 0.6 * (self.apl_feedback * s - apl))
        act = (pool / (1 + self.apl_divisive * apl) - self.apl_subtractive * apl).clamp(0, 1)
        act_kc, act_born = act[..., :KC_BASE], act[..., KC_BASE:]
        mbon = torch.einsum("brk,rkm->brm", act_kc, self.kc_to_mbon) + torch.einsum("brk,rkm->brm", act_born, self.born_mbon)
        mbon = mbon.clamp(0, 2)

        # innate, obs-dependent drives (same geometry as fly-brain-engine.js)
        hazard = x[..., 3]  # (B, 22)  sentinel sees x1.5
        haz_angle = torch.atan2(obs[:, 8], obs[:, 9]).view(B, 1, 1)
        aa = self.action_angles.view(1, 1, 4)
        haz_align = F.relu(torch.cos(haz_angle - aa))
        innate_avoid = hazard.unsqueeze(-1) * 1.5 * haz_align
        gf = torch.sigmoid((hazard.unsqueeze(-1) - 0.65) * 20.0)
        escape = F.relu(torch.cos(haz_angle + math.pi - aa)) * 3.0 * gf
        odor_diff = (obs[:, 12] - obs[:, 13]).view(B, 1)
        left = (F.relu(odor_diff - 0.04) * 1.6).clamp(max=1.2)
        right = (F.relu(-odor_diff - 0.04) * 1.6).clamp(max=1.2)
        tropo = torch.zeros(B, 1, 4, device=obs.device, dtype=obs.dtype)
        tropo[:, :, 2] = left
        tropo[:, :, 3] = right
        dsense = obs[:, 2].view(B, 1, 1)
        d_angle = torch.atan2(obs[:, 6], obs[:, 7]).view(B, 1, 1)
        attract = F.relu(torch.cos(d_angle - aa)) * (dsense * 2.2).clamp(0, 2) * torch.sigmoid((dsense - 0.02) * 200.0)
        goal = (tropo + attract) * self.odor_roles.view(1, -1, 1)
        drive = mbon + escape + goal
        sugar = (1.2 - obs[:, 4]).clamp(0.2, 1.8).view(B, 1, 1)
        drive = drive * (1 + (sugar - 1) * self.metabolic_role.view(1, -1, 1))
        desc = (drive + self.role_bias.unsqueeze(0) - innate_avoid) * self.vigor.view(1, -1, 1)

        # commissural cross-talk, no self-projection
        eye = torch.eye(self.n_brains, device=obs.device, dtype=obs.dtype).unsqueeze(-1)
        cross = torch.einsum("bsa,sda->bda", desc, self.commissural * (1 - eye))

        # gated consensus
        threat = obs[:, 3].clamp(0, 1)
        appetitive = obs[:, 2].clamp(0, 1)
        hunger = (1 - obs[:, 4]).clamp(0, 1)
        pursue = torch.maximum(appetitive, appetitive * 0.5 + hunger * 0.6).clamp(0, 1)
        explore = (1 - torch.maximum(pursue, threat)).clamp(0, 1)
        boost = torch.stack([
            1 + threat * 2.6, 1 + pursue * 2.2 - threat * 0.5,
            1 + explore * 1.4 - pursue * 0.7 - threat * 0.6, torch.ones_like(threat),
        ], dim=-1)  # (B, 4 groups)
        gains = self.role_weight.view(1, -1) * boost[:, self.group].clamp_min(0.15)
        gains = gains * (self.role_weight.sum() / gains.sum(-1, keepdim=True))
        consensus = ((desc + 0.2 * cross) * gains.unsqueeze(-1)).sum(1) * (16.0 / self.n_brains)
        return {"descending": desc, "cross": cross, "consensus": consensus, "probs": F.softmax(consensus, dim=-1),
                "gains": gains, "kc_sparsity": (act > 0).float().mean(-1)}

    def sense(self, hidden: torch.Tensor) -> torch.Tensor:
        """Trunk hidden (B, T, H) -> glomerular observation (B, 14) via the last position."""
        return torch.tanh(self.sensory_proj(self.norm(hidden.mean(dim=1))))

    def forward(self, hidden: torch.Tensor, obs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if obs is None:
            obs = self.sense(hidden)
        out = self.brains_forward(obs)
        feat = torch.cat([out["descending"].flatten(1), out["consensus"]], dim=-1)
        write = (self.gate * self.to_trunk(feat)).unsqueeze(1)
        out["obs"] = obs
        return hidden + write, out


# ---------------------------------------------------------------------------
# 3. Omni core: the two ChampionNet models
# ---------------------------------------------------------------------------
class OmniCore(nn.Module):
    """v48 collective encoder + H-MoE head, v38 vision encoder + image decoder, bridged."""

    def __init__(self, hidden_size: int):
        super().__init__()
        from champion.run import ChampionNet  # noqa: WPS433
        from champion.model_variants import HierarchicalMoEClassifierHead
        from champion.model_native_image_xlite_v38 import ChampionNetUltraExpertNativeImageExtraLite

        base = ChampionNet()
        self.collective = nn.ModuleList([base.layers[i] for i in range(10)])
        self.collective_head = HierarchicalMoEClassifierHead(256, 10)
        self.collective_norm = base.layers[11]
        self.vision = ChampionNetUltraExpertNativeImageExtraLite()
        self.bridge_norm = nn.LayerNorm(512)
        self.to_trunk = nn.Linear(512, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.feature_dim = 128

    @torch.no_grad()
    def load_sources(self, v48_path: Path, v38_path: Path) -> Dict[str, Any]:
        sd48 = torch.load(v48_path, map_location="cpu", weights_only=False)
        body = {k[len("layers."):]: v for k, v in sd48.items() if int(k.split(".")[1]) < 10}
        head = {k[len("layers.10."):]: v for k, v in sd48.items() if k.startswith("layers.10.")}
        norm = {k[len("layers.11."):]: v for k, v in sd48.items() if k.startswith("layers.11.")}
        self.collective.load_state_dict(body, strict=True)
        self.collective_head.load_state_dict(head, strict=True)
        self.collective_norm.load_state_dict(norm, strict=True)
        sd38 = {k: v.float() for k, v in torch.load(v38_path, map_location="cpu", weights_only=False).items()}
        self.vision.load_state_dict(sd38, strict=True)
        return {"v48_tensors": len(sd48), "v38_tensors": len(sd38),
                "v48_params": sum(v.numel() for v in sd48.values()), "v38_params": sum(v.numel() for v in sd38.values())}

    @staticmethod
    def featurize(prompts: Sequence[str]) -> torch.Tensor:
        from champion.chat_pipeline import featurize_context_mix_v4
        return torch.stack([featurize_context_mix_v4(p) for p in prompts])  # (B, 128)

    def encode(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = features.view(features.shape[0], 1, self.feature_dim)
        h = x
        for layer in self.collective:
            h = layer(h)
        h48 = h[:, 0]
        h38 = self.vision.encode_image_condition(x)
        return h48, h38

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        x = features.view(features.shape[0], 1, self.feature_dim)
        h = x
        for layer in self.collective:
            h = layer(h)
        return self.collective_norm(self.collective_head(h))[:, 0]

    def render(self, features: torch.Tensor) -> torch.Tensor:
        return self.vision.forward_image(features.view(features.shape[0], 1, self.feature_dim))

    def forward(self, hidden: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h48, h38 = self.encode(features)
        pooled = self.bridge_norm(torch.cat([h48, h38], dim=-1))
        write = (self.gate * self.to_trunk(pooled)).unsqueeze(1)
        return hidden + write, {"h48": h48, "h38": h38}


# ---------------------------------------------------------------------------
# 4. CNS node graft
# ---------------------------------------------------------------------------
@torch.no_grad()
def graft_fly_into_cns(model: MiMoMixModel, fly: FlyCore, npz_path: Optional[Path], step: int) -> Dict[str, Any]:
    """Make the 22 fly brains 22 nodes of the connectome core."""

    core = model.cns_core
    free = core.free_slots()
    if len(free) < fly.n_brains:
        raise RuntimeError(f"connectome core has {len(free)} free slots, need {fly.n_brains}")
    slots = [int(s) for s in free[: fly.n_brains]]
    labels: List[str] = []
    if npz_path is not None and npz_path.exists():
        z = np.load(npz_path, allow_pickle=True)
        labels = [str(x) for x in z["module_label"]]
    alive_idx = core.alive.nonzero().flatten()
    bias_fill = float(core.node_bias[alive_idx].median())
    leak_fill = float(core.leak_logit[alive_idx].median())
    # commissural strength -> edge logit: mean over the four action channels
    comm = fly.commissural.detach().mean(-1)  # [src, dst]
    events = []
    for i, slot in enumerate(slots):
        role = fly.role_names[i]
        core.alive[slot] = 1
        core.born_step[slot] = int(step)
        core.sign[slot] = 1  # commissural budgets are non-negative: excitatory
        core.hemisphere[slot] = 0 if DESCENDING_GROUPS[role] in ("reflex", "goal") else 1
        core.node_bias[slot] = bias_fill
        core.leak_logit[slot] = leak_fill
        core.in_mask[slot] = 1.0
        core.out_mask[slot] = 1.0
        core.tap_grown[slot, 0] = 1
        core.tap_grown[slot, 1] = 1
        core.read_in.weight[slot] = 0.0
        for tap in core.extra_read_in:
            tap.weight[slot] = 0.0
        for proj in [core.read_out, *core.extra_read_out] + ([core.to_thinking] if core.to_thinking is not None else []):
            proj.weight[:, slot] = 0.0
        events.append({"slot": slot, "role": role})
    # fly <-> fly edges from the learned commissural matrix
    n_comm = 0
    for s in range(fly.n_brains):
        for d in range(fly.n_brains):
            if s == d:
                continue
            w = float(comm[s, d])
            if w <= 0:
                continue
            post, pre = slots[d], slots[s]
            core.mask[post, pre] = 1.0
            core.edge_logit[post, pre] = float(inv_softplus(torch.tensor(w)))
            core.edge_grown[post, pre] = 1
            core.init_fraction[post, pre] = 0.0
            n_comm += 1
    # fly <-> real modules: bidirectional at synaptogenesis strength (-7 -> 9.1e-4)
    analogues: Dict[str, List[str]] = {}
    n_bio = 0
    if labels:
        for i, slot in enumerate(slots):
            role = fly.role_names[i]
            hits: List[int] = []
            for pat in CNS_ANALOGUES.get(role, []):
                hits += [j for j, l in enumerate(labels) if re.search(pat, l) and j not in hits]
                if len(hits) >= 6:
                    break
            hits = [j for j in hits if int(core.alive[j]) == 1][:6]
            analogues[role] = [labels[j] for j in hits]
            for j in hits:
                if core.grow_edge(slot, j, step, logit=-7.0):
                    n_bio += 1
                if core.grow_edge(j, slot, step, logit=-7.0):
                    n_bio += 1
    return {"slots": slots, "alive_after": int(core.alive.sum()), "fly_fly_edges": n_comm,
            "fly_bio_edges": n_bio, "analogues": analogues, "events": events}


# ---------------------------------------------------------------------------
# 5. The model
# ---------------------------------------------------------------------------
class ArchimedesModel(MiMoMixModel):
    """v93 trunk + fly_core + omni_core, written into the residual stream after
    ``cns_after_layer`` through zero-initialised gates (function-preserving)."""

    def __init__(self, config: MiMoMixConfig, fly_config: Optional[Dict[str, float]] = None, with_omni: bool = True):
        super().__init__(config)
        hidden = int(config.hidden_size)
        self.fly_core = FlyCore(hidden, config=fly_config)
        self.omni_core = OmniCore(hidden) if with_omni else None
        self.graft_layer = int(getattr(config, "cns_after_layer", 2))
        self._ctx: Dict[str, Any] = {}
        self.layers[self.graft_layer].register_forward_hook(self._graft_hook)

    def _graft_hook(self, module, inputs, output):
        hidden, present = output
        ctx = self._ctx
        if ctx.get("skip"):
            return output
        info: Dict[str, Any] = {}
        if self.fly_core is not None and ctx.get("fly", True):
            hidden, fly_info = self.fly_core(hidden, ctx.get("fly_obs"))
            info["fly"] = fly_info
        if self.omni_core is not None and ctx.get("omni_features") is not None:
            hidden, omni_info = self.omni_core(hidden, ctx["omni_features"])
            info["omni"] = omni_info
        ctx["info"] = info
        return hidden, present

    def forward(self, input_ids, *args, fly_obs=None, omni_features=None, use_fly=True, skip_grafts=False, **kwargs):
        self._ctx = {"fly_obs": fly_obs, "omni_features": omni_features, "fly": use_fly, "skip": skip_grafts}
        out = super().forward(input_ids, *args, **kwargs)
        info = self._ctx.get("info", {})
        self._ctx = {}
        try:
            out.graft_info = info  # MiMoMixOutput is a dataclass; attribute attach is tolerated
        except Exception:
            pass
        self.last_graft_info = info
        return out

    def graft_parameters(self) -> Iterable[nn.Parameter]:
        for m in (self.fly_core, self.omni_core):
            if m is not None:
                yield from m.parameters()

    def gate_report(self) -> Dict[str, float]:
        rep = {"fly_gate_abs_mean": float(self.fly_core.gate.abs().mean())}
        if self.omni_core is not None:
            rep["omni_gate_abs_mean"] = float(self.omni_core.gate.abs().mean())
        core = self.cns_core
        if core is not None:
            rep["cns_gate_abs_mean"] = float(core.gate.abs().mean())
            rep["cns_alive"] = int(core.alive.sum())
        return rep


# ---------------------------------------------------------------------------
# 6. Checkpoint I/O
# ---------------------------------------------------------------------------
def save_archimedes(path: Path, model: ArchimedesModel, tokenizer: text_utils.WordTokenizer, extra: Dict[str, Any], archimedes: Dict[str, Any]) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "base_schema": TALK_SCHEMA,
        "config": model.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "tokenizer": tokenizer.to_dict(),
        "extra": extra,
        "archimedes": archimedes,
    }
    staging = path.with_name(path.name + ".tmp")
    torch.save(payload, staging)
    os.replace(staging, path)


def load_archimedes(path, map_location: str = "cpu") -> Tuple[ArchimedesModel, text_utils.WordTokenizer, Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} checkpoint (got {payload.get('schema')})")
    config = MiMoMixConfig(**payload["config"])
    arch = payload.get("archimedes", {})
    model = ArchimedesModel(config, fly_config=arch.get("fly_config"), with_omni=arch.get("with_omni", True))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    tokenizer = text_utils.WordTokenizer.from_dict(payload["tokenizer"])
    return model, tokenizer, payload


def build_archimedes_from_v93(v93_payload: Dict[str, Any], fly_config: Dict[str, float]) -> ArchimedesModel:
    """Instantiate the grafted architecture and load the v93 trunk weights into it."""
    config = MiMoMixConfig(**v93_payload["config"])
    model = ArchimedesModel(config, fly_config=fly_config, with_omni=True)
    missing, unexpected = model.load_state_dict(v93_payload["state_dict"], strict=False)
    bad = [k for k in missing if not (k.startswith("fly_core.") or k.startswith("omni_core."))]
    if bad or unexpected:
        raise RuntimeError(f"trunk load mismatch: missing {bad[:5]} unexpected {unexpected[:5]}")
    return model
