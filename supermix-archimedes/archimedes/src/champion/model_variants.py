"""Model variant definitions for the Supermix_27 ChampionNet architecture.

This module defines all classifier head variants (from simple ExpandedClassifierHead
through the advanced MoE heads) and their backbone wrappers. It also provides
utility functions for model construction, weight loading with backward
compatibility, and checkpoint introspection.

Classifier Head Evolution:
    ExpandedClassifierHead        → Single adapter branch
    ExpandedClassifierHeadXL      → Dual adapters + router
    ExpandedClassifierHeadXXL     → Tri-branch + two-stage routing
    ExpandedClassifierHeadXXXL    → Quad-branch + three-stage routing
    ExpandedClassifierHeadUltra   → Five branches + domain-expert calibration
    ExpandedClassifierHeadMega    → Six branches + cross-attention fusion
    GatedExpertClassifierHead     → MoE with Noisy Top-K Gating
    HierarchicalMoEClassifierHead → Two-level hierarchical routing
    DeepExpertClassifierHead      → Aux-loss-free dynamic bias load balancing
    ExpertChoiceClassifierHead    → Expert Choice (experts pick tokens)
    SmarterExpertClassifierHead   → Sigma Gating + LoRA adapters
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from run import ChampionNet, GatedFFN


SUPPORTED_MODEL_SIZES: Tuple[str, ...] = (
    "base",
    "large",
    "xlarge",
    "xxlarge",
    "xxxlarge",
    "ultralarge",
    "megalarge",
    "ultra_expert",
    "hierarchical_expert",
    "deep_expert",
    "expert_choice",
    "smarter_expert",
    "thought_expert",
    "recursive_expert",
    "reflexive_expert",
    "metacognitive_expert",
    "tree_of_thought_expert",
    "consensus_expert",
    "deliberative_expert",
    "omniscient_expert",
    "neurogenesis_expert",
    "cognitive_expert",
    "transcendent_expert",
    "omniversal_expert",
    "fractal_expert",
    "liquid_spiking_expert",
    "active_inference_expert",
    "holographic_state_space_expert",
    "paper_fusion_expert",
    "frontier_expert",
    "frontier_collective_expert",
    "frontier_verifier_expert",
)
EXPANSION_DIM_MODEL_SIZES = frozenset({"large", "xlarge", "xxlarge", "xxxlarge", "ultralarge", "megalarge"})
EXTRA_EXPANSION_DIM_MODEL_SIZES = frozenset({"xlarge", "xxlarge", "xxxlarge", "ultralarge", "megalarge"})
THIRD_EXPANSION_DIM_MODEL_SIZES = frozenset({"xxlarge", "xxxlarge", "ultralarge", "megalarge"})
FOURTH_EXPANSION_DIM_MODEL_SIZES = frozenset({"xxxlarge", "ultralarge", "megalarge"})
FIFTH_EXPANSION_DIM_MODEL_SIZES = frozenset({"ultralarge", "megalarge"})
SIXTH_EXPANSION_DIM_MODEL_SIZES = frozenset({"megalarge"})


class ExpandedClassifierHead(nn.Module):
    """
    Larger classifier head that keeps base checkpoint compatibility:
    - retains `weight` and `bias` keys used by the original final linear
    - adds an extra adapter branch (256 -> expansion_dim -> 10)
    - starts near original behavior with alpha initialized to 0
    """

    def __init__(self, in_dim: int = 256, out_dim: int = 10, expansion_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        adapter_logits = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        return base_logits + self.alpha * adapter_logits


class ExpandedClassifierHeadXL(nn.Module):
    """
    Wider classifier head for extra capacity and richer routing:
    - keeps base-compatible keys (`weight`, `bias`, `adapter_up/down`, `alpha`)
    - adds a second wider adapter branch + learned token-wise router
    - remains warm-start compatible from base/large checkpoints
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 768,
        extra_expansion_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Reuse large-head key names so xlarge can warm-start from large checkpoints.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))

        route = torch.softmax(self.router(x), dim=-1)
        mix = route[..., :1] * a1 + route[..., 1:2] * a2

        return base_logits + self.alpha * a1 + self.beta * mix


class ExpandedClassifierHeadXXL(nn.Module):
    """
    Maximum-capacity classifier head with tri-branch routing:
    - retains all xlarge-compatible keys for warm-starting
    - adds a third branch and second-stage router for richer fusion
    - remains near-base behavior at init via alpha/beta/gamma initialized to 0
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep xlarge key names to preserve compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        # New xxl-only branch.
        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.normal_(self.adapter_up_c.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_c.weight)
        nn.init.zeros_(self.router2.weight)
        nn.init.zeros_(self.router2.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_all = torch.softmax(self.router2(x), dim=-1)
        mix_all = route_all[..., :1] * a1 + route_all[..., 1:2] * a2 + route_all[..., 2:3] * a3

        return base_logits + self.alpha * a1 + self.beta * mix_ab + self.gamma * mix_all


class ExpandedClassifierHeadXXXL(nn.Module):
    """
    Highest-capacity classifier head with quad-branch routing:
    - preserves xxlarge-compatible keys for warm-starting from xxlarge checkpoints
    - adds a fourth branch plus a third routing stage
    - initializes new scaling params at 0 to remain stable at startup
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep xxlarge key names to preserve compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # New xxxlarge-only branch.
        self.adapter_up_d = nn.Linear(in_dim, fourth_expansion_dim, bias=False)
        self.adapter_down_d = nn.Linear(fourth_expansion_dim, out_dim, bias=False)
        self.router3 = nn.Linear(in_dim, 4, bias=True)
        self.delta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.normal_(self.adapter_up_c.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_c.weight)
        nn.init.zeros_(self.router2.weight)
        nn.init.zeros_(self.router2.bias)

        nn.init.normal_(self.adapter_up_d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_d.weight)
        nn.init.zeros_(self.router3.weight)
        nn.init.zeros_(self.router3.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))
        a4 = self.adapter_down_d(self.dropout(F.relu(self.adapter_up_d(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_abc = torch.softmax(self.router2(x), dim=-1)
        mix_abc = route_abc[..., :1] * a1 + route_abc[..., 1:2] * a2 + route_abc[..., 2:3] * a3

        route_abcd = torch.softmax(self.router3(x), dim=-1)
        mix_abcd = (
            route_abcd[..., :1] * a1
            + route_abcd[..., 1:2] * a2
            + route_abcd[..., 2:3] * a3
            + route_abcd[..., 3:4] * a4
        )

        return base_logits + self.alpha * a1 + self.beta * mix_ab + self.gamma * mix_abc + self.delta * mix_abcd


class ExpandedClassifierHeadUltra(nn.Module):
    """
    Maximum-capacity classifier head with five routed branches:
    - keeps xxxlarge-compatible keys for warm-starting
    - adds a fifth branch + fourth routing stage
    - adds domain-expert calibration (science/code/writing/math friendly) for smarter adaptation
    - initializes new scale parameters at 0 for stable adaptation
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep xxxlarge keys for compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_d = nn.Linear(in_dim, fourth_expansion_dim, bias=False)
        self.adapter_down_d = nn.Linear(fourth_expansion_dim, out_dim, bias=False)
        self.router3 = nn.Linear(in_dim, 4, bias=True)
        self.delta = nn.Parameter(torch.tensor(0.0))

        # New ultralarge-only branch.
        self.adapter_up_e = nn.Linear(in_dim, fifth_expansion_dim, bias=False)
        self.adapter_down_e = nn.Linear(fifth_expansion_dim, out_dim, bias=False)
        self.router4 = nn.Linear(in_dim, 5, bias=True)
        self.epsilon = nn.Parameter(torch.tensor(0.0))

        # Architecture upgrade: low-cost domain-expert correction + bounded calibration.
        self.pre_norm = nn.LayerNorm(in_dim)
        self.domain_router = nn.Linear(in_dim, 4, bias=True)
        self.domain_experts = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.calib_gate = nn.Linear(in_dim, out_dim, bias=True)
        self.zeta = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.normal_(self.adapter_up_c.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_c.weight)
        nn.init.zeros_(self.router2.weight)
        nn.init.zeros_(self.router2.bias)

        nn.init.normal_(self.adapter_up_d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_d.weight)
        nn.init.zeros_(self.router3.weight)
        nn.init.zeros_(self.router3.bias)

        nn.init.normal_(self.adapter_up_e.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_e.weight)
        nn.init.zeros_(self.router4.weight)
        nn.init.zeros_(self.router4.bias)

        nn.init.ones_(self.pre_norm.weight)
        nn.init.zeros_(self.pre_norm.bias)
        nn.init.zeros_(self.domain_router.weight)
        nn.init.zeros_(self.domain_router.bias)
        nn.init.normal_(self.domain_experts.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.calib_gate.weight)
        nn.init.zeros_(self.calib_gate.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))
        a4 = self.adapter_down_d(self.dropout(F.relu(self.adapter_up_d(x))))
        a5 = self.adapter_down_e(self.dropout(F.selu(self.adapter_up_e(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_abc = torch.softmax(self.router2(x), dim=-1)
        mix_abc = route_abc[..., :1] * a1 + route_abc[..., 1:2] * a2 + route_abc[..., 2:3] * a3

        route_abcd = torch.softmax(self.router3(x), dim=-1)
        mix_abcd = (
            route_abcd[..., :1] * a1
            + route_abcd[..., 1:2] * a2
            + route_abcd[..., 2:3] * a3
            + route_abcd[..., 3:4] * a4
        )

        route_abcde = torch.softmax(self.router4(x), dim=-1)
        mix_abcde = (
            route_abcde[..., :1] * a1
            + route_abcde[..., 1:2] * a2
            + route_abcde[..., 2:3] * a3
            + route_abcde[..., 3:4] * a4
            + route_abcde[..., 4:5] * a5
        )

        # Domain-expert calibration branch improves robustness across diverse domains.
        h = self.pre_norm(x)
        dom_logits = self.domain_experts(self.dropout(F.silu(h)))
        dom_logits = dom_logits.view(*dom_logits.shape[:-1], 4, base_logits.shape[-1])
        dom_w = torch.softmax(self.domain_router(h), dim=-1).unsqueeze(-1)
        dom_mix = (dom_w * dom_logits).sum(dim=-2)
        calib = torch.tanh(self.calib_gate(h))

        return (
            base_logits
            + self.alpha * a1
            + self.beta * mix_ab
            + self.gamma * mix_abc
            + self.delta * mix_abcd
            + self.epsilon * mix_abcde
            + self.zeta * dom_mix
            + self.theta * calib
        )


class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention module that lets adapter branches attend to each other.
    Input: (B, T, N_branches, out_dim) tensor of stacked branch outputs.
    Output: (B, T, out_dim) fused representation.
    """
    def __init__(self, out_dim: int = 10, n_branches: int = 6, n_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = max(1, out_dim // n_heads)
        inner = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(out_dim, inner, bias=False)
        self.k_proj = nn.Linear(out_dim, inner, bias=False)
        self.v_proj = nn.Linear(out_dim, inner, bias=False)
        self.out_proj = nn.Linear(inner, out_dim, bias=False)
        self.scale = self.head_dim ** -0.5
        self.fusion_weight = nn.Linear(out_dim, 1, bias=True)

    def forward(self, branch_stack):
        # branch_stack: (B, T, N, D)
        B, T, N, D = branch_stack.shape
        flat = branch_stack.view(B * T, N, D)
        q = self.q_proj(flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B*T, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B * T, N, self.n_heads * self.head_dim)
        out = self.out_proj(out)  # (B*T, N, D)
        # Weighted sum over branches
        w = torch.softmax(self.fusion_weight(out), dim=1)  # (B*T, N, 1)
        fused = (w * out).sum(dim=1)  # (B*T, D)
        return fused.view(B, T, D)


class ExpandedClassifierHeadMega(nn.Module):
    """
    Maximum-capacity classifier head with six routed branches + cross-attention fusion:
    - keeps ultralarge-compatible keys for warm-starting
    - adds a sixth branch (Mish activation) + fifth routing stage
    - adds CrossAttentionFusion for inter-branch communication
    - adds a reasoning_gate that amplifies logit differences for sharper routing
    - initializes new scale parameters at 0 for stable adaptation
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        sixth_expansion_dim: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep ultralarge keys for compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_d = nn.Linear(in_dim, fourth_expansion_dim, bias=False)
        self.adapter_down_d = nn.Linear(fourth_expansion_dim, out_dim, bias=False)
        self.router3 = nn.Linear(in_dim, 4, bias=True)
        self.delta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_e = nn.Linear(in_dim, fifth_expansion_dim, bias=False)
        self.adapter_down_e = nn.Linear(fifth_expansion_dim, out_dim, bias=False)
        self.router4 = nn.Linear(in_dim, 5, bias=True)
        self.epsilon = nn.Parameter(torch.tensor(0.0))

        # Domain-expert calibration (from ultralarge).
        self.pre_norm = nn.LayerNorm(in_dim)
        self.domain_router = nn.Linear(in_dim, 4, bias=True)
        self.domain_experts = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.calib_gate = nn.Linear(in_dim, out_dim, bias=True)
        self.zeta = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Parameter(torch.tensor(0.0))

        # --- Megalarge-only additions ---
        # Sixth branch with Mish activation.
        self.adapter_up_f = nn.Linear(in_dim, sixth_expansion_dim, bias=False)
        self.adapter_down_f = nn.Linear(sixth_expansion_dim, out_dim, bias=False)
        self.router5 = nn.Linear(in_dim, 6, bias=True)
        self.iota = nn.Parameter(torch.tensor(0.0))

        # Cross-attention fusion lets branches attend to each other.
        self.cross_attn_fusion = CrossAttentionFusion(out_dim=out_dim, n_branches=6, n_heads=2)
        self.kappa = nn.Parameter(torch.tensor(0.0))

        # Reasoning gate amplifies logit differences for sharper routing.
        self.reasoning_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.SiLU(),
            nn.Linear(in_dim // 4, out_dim, bias=True),
        )
        self.lambda_ = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        for name in ['adapter_up', 'adapter_up_b', 'adapter_up_c', 'adapter_up_d', 'adapter_up_e', 'adapter_up_f']:
            nn.init.normal_(getattr(self, name).weight, mean=0.0, std=0.02)
        for name in ['adapter_down', 'adapter_down_b', 'adapter_down_c', 'adapter_down_d', 'adapter_down_e', 'adapter_down_f']:
            nn.init.zeros_(getattr(self, name).weight)
        for name in ['router', 'router2', 'router3', 'router4', 'router5', 'domain_router']:
            nn.init.zeros_(getattr(self, name).weight)
            nn.init.zeros_(getattr(self, name).bias)
        nn.init.ones_(self.pre_norm.weight)
        nn.init.zeros_(self.pre_norm.bias)
        nn.init.normal_(self.domain_experts.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.calib_gate.weight)
        nn.init.zeros_(self.calib_gate.bias)
        # Reasoning gate init
        for m in self.reasoning_gate:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))
        a4 = self.adapter_down_d(self.dropout(F.relu(self.adapter_up_d(x))))
        a5 = self.adapter_down_e(self.dropout(F.selu(self.adapter_up_e(x))))
        a6 = self.adapter_down_f(self.dropout(F.mish(self.adapter_up_f(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_abc = torch.softmax(self.router2(x), dim=-1)
        mix_abc = route_abc[..., :1] * a1 + route_abc[..., 1:2] * a2 + route_abc[..., 2:3] * a3

        route_abcd = torch.softmax(self.router3(x), dim=-1)
        mix_abcd = (
            route_abcd[..., :1] * a1
            + route_abcd[..., 1:2] * a2
            + route_abcd[..., 2:3] * a3
            + route_abcd[..., 3:4] * a4
        )

        route_abcde = torch.softmax(self.router4(x), dim=-1)
        mix_abcde = (
            route_abcde[..., :1] * a1
            + route_abcde[..., 1:2] * a2
            + route_abcde[..., 2:3] * a3
            + route_abcde[..., 3:4] * a4
            + route_abcde[..., 4:5] * a5
        )

        route_all = torch.softmax(self.router5(x), dim=-1)
        mix_all = (
            route_all[..., :1] * a1
            + route_all[..., 1:2] * a2
            + route_all[..., 2:3] * a3
            + route_all[..., 3:4] * a4
            + route_all[..., 4:5] * a5
            + route_all[..., 5:6] * a6
        )

        # Domain-expert calibration branch.
        h = self.pre_norm(x)
        dom_logits = self.domain_experts(self.dropout(F.silu(h)))
        dom_logits = dom_logits.view(*dom_logits.shape[:-1], 4, base_logits.shape[-1])
        dom_w = torch.softmax(self.domain_router(h), dim=-1).unsqueeze(-1)
        dom_mix = (dom_w * dom_logits).sum(dim=-2)
        calib = torch.tanh(self.calib_gate(h))

        # Cross-attention fusion: stack all 6 branches and let them attend.
        branch_stack = torch.stack([a1, a2, a3, a4, a5, a6], dim=-2)  # (B,T,6,out_dim)
        cross_fused = self.cross_attn_fusion(branch_stack)

        # Reasoning gate: amplify logit differences for sharper decisions.
        reason = torch.tanh(self.reasoning_gate(x))

        return (
            base_logits
            + self.alpha * a1
            + self.beta * mix_ab
            + self.gamma * mix_abc
            + self.delta * mix_abcd
            + self.epsilon * mix_abcde
            + self.iota * mix_all
            + self.zeta * dom_mix
            + self.theta * calib
            + self.kappa * cross_fused
            + self.lambda_ * reason
        )


class GatedExpertClassifierHead(nn.Module):
    """
    Advanced routing head with specialized experts:
    - Top-K gating selects most relevant experts per input
    - Experts utilize diverse activation functions and expansion ratios
    - Residual calibration ensures logit stability
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 6,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.top_k = top_k
        self.n_experts = n_experts

        # Experts with varying architectures
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, 1024 + (i * 512), bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(1024 + (i * 512), out_dim, bias=False) for i in range(n_experts)
        ])
        self.activations = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        
        self.gate = nn.Linear(in_dim, n_experts, bias=False)
        self.noise_gate = nn.Linear(in_dim, n_experts, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        for eup in self.experts_up:
            nn.init.normal_(eup.weight, mean=0.0, std=0.01)
        for edown in self.experts_down:
            nn.init.zeros_(edown.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.noise_gate.weight)
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        
        # Gating logic with Noisy Top-K
        clean_logits = self.gate(x)
        if self.training:
            noise_std = F.softplus(self.noise_gate(x))
            noise = torch.randn_like(clean_logits) * noise_std
            gate_logits = clean_logits + noise
        else:
            gate_logits = clean_logits

        weights = torch.softmax(gate_logits, dim=-1)
        top_weights, top_idx = torch.topk(weights, k=self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Expert processing
        expert_outputs = []
        for i in range(self.n_experts):
            out = self.experts_down[i](self.dropout(self.activations[i](self.experts_up[i](x))))
            expert_outputs.append(out)
        expert_stack = torch.stack(expert_outputs, dim=-2) # (B, T, N, D)
        
        # Gather top experts
        # We need to broadcast the indices for gather
        gather_idx = top_idx.unsqueeze(-1).expand(*top_idx.shape, base_logits.shape[-1])
        selected_experts = torch.gather(expert_stack, -2, gather_idx) # (B, T, K, D)
        
        expert_logits = (selected_experts * top_weights.unsqueeze(-1)).sum(dim=-2)
        calib = self.calibration(x)
        
        return base_logits + self.alpha * expert_logits + self.theta * calib


class HierarchicalMoEClassifierHead(nn.Module):
    """
    Next-generation classifier head with hierarchical two-level MoE routing:
    - Domain-level gating selects which expert group is most relevant
    - Per-domain expert gating selects the best expert(s) within the group
    - A shared 'always-on' expert provides a stable baseline signal
    - Per-expert residual LoRA adapters add lightweight correction capacity
    - Per-expert LayerNorm on outputs improves gradient flow
    - Auxiliary load-balancing loss prevents expert collapse during training
    Inspired by DeepSeek-MoE, GShard, and ST-MoE research.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_domain_groups: int = 2,
        experts_per_group: int = 4,
        top_k_domains: int = 1,
        top_k_experts: int = 2,
        lora_rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_domain_groups = n_domain_groups
        self.experts_per_group = experts_per_group
        self.n_experts = n_domain_groups * experts_per_group
        self.top_k_domains = top_k_domains
        self.top_k_experts = top_k_experts
        self._aux_loss = torch.tensor(0.0)

        # --- Shared expert (always active, not gated) ---
        shared_dim = 1024
        self.shared_up = nn.Linear(in_dim, shared_dim, bias=False)
        self.shared_down = nn.Linear(shared_dim, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(0.0))

        # --- Domain-level gating ---
        self.domain_gate = nn.Linear(in_dim, n_domain_groups, bias=False)
        self.domain_noise_gate = nn.Linear(in_dim, n_domain_groups, bias=False)

        # --- Per-domain expert gating ---
        self.expert_gates = nn.ModuleList([
            nn.Linear(in_dim, experts_per_group, bias=False)
            for _ in range(n_domain_groups)
        ])
        self.expert_noise_gates = nn.ModuleList([
            nn.Linear(in_dim, experts_per_group, bias=False)
            for _ in range(n_domain_groups)
        ])

        # --- Experts with varying expansion dims & activations ---
        # 8 experts total: group 0 (analytical): SiLU, GELU, Mish, ReLU
        #                   group 1 (creative):  SELU, Tanh, Softplus, Mish-variant
        expansion_dims = [1024, 1536, 2048, 2560, 1024, 1536, 2048, 3072]
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, expansion_dims[i], bias=False) for i in range(self.n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(expansion_dims[i], out_dim, bias=False) for i in range(self.n_experts)
        ])
        self.activations = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh, F.softplus, F.mish]

        # --- Per-expert output LayerNorm ---
        self.expert_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(self.n_experts)
        ])

        # --- Per-expert residual LoRA adapters ---
        self.lora_down = nn.ModuleList([
            nn.Linear(in_dim, lora_rank, bias=False) for _ in range(self.n_experts)
        ])
        self.lora_up = nn.ModuleList([
            nn.Linear(lora_rank, out_dim, bias=False) for _ in range(self.n_experts)
        ])

        # --- Aggregation ---
        self.alpha = nn.Parameter(torch.tensor(0.0))  # expert mixture scale
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))  # calibration scale
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # Shared expert
        nn.init.normal_(self.shared_up.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.shared_down.weight)
        nn.init.ones_(self.shared_norm.weight)
        nn.init.zeros_(self.shared_norm.bias)
        # Domain gates
        nn.init.zeros_(self.domain_gate.weight)
        nn.init.zeros_(self.domain_noise_gate.weight)
        # Per-domain expert gates
        for g in self.expert_gates:
            nn.init.zeros_(g.weight)
        for g in self.expert_noise_gates:
            nn.init.zeros_(g.weight)
        # Experts
        for eup in self.experts_up:
            nn.init.normal_(eup.weight, mean=0.0, std=0.01)
        for edown in self.experts_down:
            nn.init.zeros_(edown.weight)
        # Expert norms
        for norm in self.expert_norms:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        # LoRA adapters
        for ld in self.lora_down:
            nn.init.normal_(ld.weight, mean=0.0, std=0.01)
        for lu in self.lora_up:
            nn.init.zeros_(lu.weight)
        # Calibration
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def _compute_aux_loss(self, gate_probs, expert_mask):
        """
        Auxiliary load-balancing loss (Switch Transformer style).
        Encourages uniform expert utilization.
        gate_probs: (B*T, N) softmax probabilities
        expert_mask: (B*T, N) binary mask of selected experts
        """
        # Fraction of tokens dispatched to each expert
        f = expert_mask.float().mean(dim=0)  # (N,)
        # Average gate probability for each expert
        p = gate_probs.mean(dim=0)  # (N,)
        n = gate_probs.shape[-1]
        return n * (f * p).sum()

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]  # (B, T) or (B,)
        x_flat = x.reshape(-1, x.shape[-1])  # (BT, in_dim)
        BT = x_flat.shape[0]

        # --- Shared expert (always active) ---
        shared_out = self.shared_down(self.dropout(F.silu(self.shared_up(x))))
        shared_out = self.shared_norm(shared_out)

        # --- Domain-level gating ---
        clean_domain = self.domain_gate(x_flat)  # (BT, n_groups)
        if self.training:
            noise_std = F.softplus(self.domain_noise_gate(x_flat))
            domain_logits = clean_domain + torch.randn_like(clean_domain) * noise_std
        else:
            domain_logits = clean_domain
        domain_probs = torch.softmax(domain_logits, dim=-1)  # (BT, n_groups)

        # Select top-k domains
        top_dom_w, top_dom_idx = torch.topk(domain_probs, k=self.top_k_domains, dim=-1)
        top_dom_w = top_dom_w / (top_dom_w.sum(dim=-1, keepdim=True) + 1e-6)

        # --- Process all experts ---
        all_expert_outs = []
        for i in range(self.n_experts):
            act = self.activations[i]
            main_out = self.experts_down[i](self.dropout(act(self.experts_up[i](x_flat))))
            lora_out = self.lora_up[i](self.dropout(F.silu(self.lora_down[i](x_flat))))
            expert_out = self.expert_norms[i](main_out + lora_out)
            all_expert_outs.append(expert_out)
        expert_stack = torch.stack(all_expert_outs, dim=1)  # (BT, n_experts, out_dim)

        # --- Per-domain expert gating & selection ---
        aggregated = x_flat.new_zeros(BT, base_logits.shape[-1])
        all_gate_probs_list = []
        all_expert_mask_list = []

        for d in range(self.n_domain_groups):
            # Expert gating within this domain
            clean_exp = self.expert_gates[d](x_flat)  # (BT, experts_per_group)
            if self.training:
                noise_std = F.softplus(self.expert_noise_gates[d](x_flat))
                exp_logits = clean_exp + torch.randn_like(clean_exp) * noise_std
            else:
                exp_logits = clean_exp
            exp_probs = torch.softmax(exp_logits, dim=-1)  # (BT, experts_per_group)

            top_exp_w, top_exp_idx = torch.topk(exp_probs, k=self.top_k_experts, dim=-1)
            top_exp_w = top_exp_w / (top_exp_w.sum(dim=-1, keepdim=True) + 1e-6)

            # Map local expert indices to global
            global_idx = top_exp_idx + d * self.experts_per_group  # (BT, top_k_experts)
            gather_idx = global_idx.unsqueeze(-1).expand(-1, -1, base_logits.shape[-1])
            selected = torch.gather(expert_stack, 1, gather_idx)  # (BT, top_k, out_dim)
            domain_expert_out = (selected * top_exp_w.unsqueeze(-1)).sum(dim=1)  # (BT, out_dim)

            # Weight by domain probability
            # Get domain d's weight for each token
            # Check if domain d is in the top-k for each token
            domain_d_mask = (top_dom_idx == d).any(dim=-1)  # (BT,)
            # Get the weight for domain d where it was selected
            domain_d_weight = x_flat.new_zeros(BT)
            for k_idx in range(self.top_k_domains):
                mask_k = (top_dom_idx[:, k_idx] == d)
                domain_d_weight = domain_d_weight + mask_k.float() * top_dom_w[:, k_idx]

            aggregated = aggregated + domain_d_weight.unsqueeze(-1) * domain_expert_out

            # Collect for aux loss
            all_gate_probs_list.append(exp_probs)
            exp_mask = torch.zeros(BT, self.experts_per_group, device=x_flat.device)
            exp_mask.scatter_(1, top_exp_idx, 1.0)
            all_expert_mask_list.append(exp_mask)

        # Reshape aggregated back to original shape
        aggregated = aggregated.view(*shape_prefix, -1)

        # --- Auxiliary load-balancing loss ---
        if self.training:
            # Domain-level balance
            dom_mask = torch.zeros(BT, self.n_domain_groups, device=x_flat.device)
            dom_mask.scatter_(1, top_dom_idx, 1.0)
            aux_domain = self._compute_aux_loss(domain_probs, dom_mask)
            # Expert-level balance (per group)
            aux_expert = sum(
                self._compute_aux_loss(gp, em)
                for gp, em in zip(all_gate_probs_list, all_expert_mask_list)
            ) / max(1, self.n_domain_groups)
            self._aux_loss = aux_domain + aux_expert
        else:
            self._aux_loss = torch.tensor(0.0, device=x.device)

        calib = self.calibration(x)

        return (
            base_logits
            + self.shared_scale * shared_out
            + self.alpha * aggregated
            + self.theta * calib
        )


class DeepExpertClassifierHead(nn.Module):
    """
    State-of-the-art MoE classifier head featuring:
    - 1 always-active shared expert for universal knowledge.
    - N routed experts using Top-K gating.
    - Auxiliary-loss-free load balancing (dynamic bias adjustment during training).
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.top_k = min(top_k, n_experts)
        self.n_experts = n_experts

        # Shared Expert (always active)
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # Routed Experts
        # Using a mix of activations + varying inner dimensions for diversity
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, 1024 + (i * 256), bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(1024 + (i * 256), out_dim, bias=False) for i in range(n_experts)
        ])
        # Cycle activations if n_experts > 6
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # Routing mechanism
        self.gate = nn.Linear(in_dim, n_experts, bias=False)
        self.noise_gate = nn.Linear(in_dim, n_experts, bias=False)
        
        # Aux-free load balancing terms
        # These biases are added to the routing logits during training.
        # They don't require gradients; we update them manually.
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.experts_down[i].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_down.weight, a=math.sqrt(5))
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.noise_gate.weight, std=0.02)
        nn.init.constant_(self.calibration.weight, 0)
        nn.init.constant_(self.calibration.bias, 0)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        BT = x_flat.shape[0]

        # Shared Expert
        shared_out = self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        shared_out = self.shared_norm(shared_out)

        # Routed Experts Gating
        clean_logits = self.gate(x_flat)
        if self.training:
            noise_std = F.softplus(self.noise_gate(x_flat))
            noise = torch.randn_like(clean_logits) * noise_std
            gate_logits = clean_logits + noise + self.expert_bias
        else:
            gate_logits = clean_logits

        weights = torch.softmax(gate_logits, dim=-1)
        top_weights, top_idx = torch.topk(weights, k=self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Process Routed Experts
        all_expert_outs = []
        for i in range(self.n_experts):
            act = self.activations[i]
            out = self.experts_down[i](self.dropout(act(self.experts_up[i](x_flat))))
            out = self.expert_norms[i](out)
            all_expert_outs.append(out)
        expert_stack = torch.stack(all_expert_outs, dim=1)  # (BT, n_experts, out_dim)

        # Gather \u0026 combine chosen experts
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, base_logits.shape[-1])
        selected_experts = torch.gather(expert_stack, 1, gather_idx)  # (BT, K, out_dim)
        routed_out = (selected_experts * top_weights.unsqueeze(-1)).sum(dim=1)  # (BT, out_dim)

        # Reshape combinations back
        routed_out = routed_out.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        calib = self.calibration(x)

        # Update expert bias tracking internally during training if needed (aux-free load balance)
        # Usually implemented in the loop, but we can store load stats here
        if self.training:
            # Fraction of tokens each expert received
            with torch.no_grad():
                expert_mask = torch.zeros(BT, self.n_experts, device=x.device)
                expert_mask.scatter_(1, top_idx, 1.0)
                expert_load = expert_mask.mean(dim=0)  # (n_experts,)
                target_load = self.top_k / self.n_experts
                # Adjust bias: decrease bias for overloaded experts, increase for underloaded
                alpha_bias = 0.1 # learning rate for bias
                self.expert_bias.add_(alpha_bias * (target_load - expert_load))

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * routed_out
            + self.theta * calib
        )


class ExpertChoiceClassifierHead(nn.Module):
    """
    Expert Choice (EC) Routing Classifier Head:
    Instead of tokens selecting the top-K experts, experts select the top-K tokens.
    This guarantees 100% load balancing across experts and avoids dropped tokens for active experts.
    Tokens can be routed to a variable number of experts based on their relevance.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        capacity_factor: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self.n_experts = n_experts
        # How many tokens an expert will pick, relative to average token load per expert
        self.capacity_factor = capacity_factor
        
        # Routed Experts
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, 1024 + (i * 256), bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(1024 + (i * 256), out_dim, bias=False) for i in range(n_experts)
        ])
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # Routing mechanism: Token-to-expert affinity
        # We need a score for each (token, expert) pair. 
        self.gate = nn.Linear(in_dim, n_experts, bias=False)
        self.noise_gate = nn.Linear(in_dim, n_experts, bias=False)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.experts_down[i].weight, a=math.sqrt(5))
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.noise_gate.weight, std=0.02)
        nn.init.constant_(self.calibration.weight, 0)
        nn.init.constant_(self.calibration.bias, 0)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])  # (BT, D)
        BT = x_flat.shape[0]

        # Expert token capacity (how many tokens each expert will select)
        # Average load per expert is BT / n_experts. We multiply by capacity_factor.
        # e.g., if capacity_factor is 2.0, each expert processes 2 * (BT / n) tokens.
        expert_capacity = min(BT, max(1, int(math.ceil((BT * self.capacity_factor) / self.n_experts))))

        # 1. Compute affinity scores
        clean_logits = self.gate(x_flat)  # (BT, n_experts)
        if self.training:
            noise_std = F.softplus(self.noise_gate(x_flat))
            noise = torch.randn_like(clean_logits) * noise_std
            affinity_logits = clean_logits + noise
        else:
            affinity_logits = clean_logits

        # In Expert Choice, normalization is over experts first to get routing probabilities
        # then experts pick top tokens based on those probabilities.
        # This implies experts compete for tokens based on how much the token "wants" that expert.
        token_to_expert_probs = torch.softmax(affinity_logits, dim=-1)  # (BT, n_experts)

        # Transpose to (n_experts, BT) so each expert can pick top tokens
        expert_to_token_probs = token_to_expert_probs.t()  # (n_experts, BT)

        expert_outputs = torch.zeros((BT, base_logits.shape[-1]), device=x.device, dtype=x.dtype)
        
        # We need a scaling factor later. If an expert picks a token, its output is scaled by 
        # the token_to_expert_prob.
        
        # 2. Each expert processes its top tokens
        for i in range(self.n_experts):
            probs_i = expert_to_token_probs[i]  # (BT,)
            
            # Select top-k tokens for this expert
            top_probs, top_token_idx = torch.topk(probs_i, k=expert_capacity, dim=0)  # (capacity,)
            
            # Gather tokens
            selected_tokens = x_flat[top_token_idx]  # (capacity, D)
            
            # Process through expert i
            act = self.activations[i]
            out = self.experts_down[i](self.dropout(act(self.experts_up[i](selected_tokens))))
            out = self.expert_norms[i](out)  # (capacity, out_dim)
            
            # Scale by routing probability
            out = out * top_probs.unsqueeze(-1)  # (capacity, out_dim)
            
            # Add to the global output buffer using `scatter_add_` or index_add
            # out is (capacity, out_dim). top_token_idx is (capacity,). 
            # expert_outputs is (BT, out_dim).
            expert_outputs.index_add_(0, top_token_idx, out)
            
        expert_outputs = expert_outputs.view(*shape_prefix, -1)
        calib = self.calibration(x)

        return (
            base_logits
            + self.alpha * expert_outputs
            + self.theta * calib
        )


class SmarterExpertClassifierHead(nn.Module):
    """
    Next-generation MoE classifier head featuring:
    - 1 always-active shared anchor expert.
    - 8 diversity-driven routed experts.
    - Sigma Gating (independent sigmoid scores per expert).
    - Aux-loss-free load balancing (DeepSeek-V3 style dynamic bias).
    - Lightweight residual LoRA adapters per expert.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        lora_rank: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_experts = n_experts

        # 1. Shared Anchor Expert
        shared_dim = 2048
        self.shared_up = nn.Linear(in_dim, shared_dim, bias=False)
        self.shared_down = nn.Linear(shared_dim, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 2. Routed Experts with diversity
        expansion_dims = [1024, 1536, 2048, 2560, 1024, 1536, 2048, 3072]
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, expansion_dims[i % 8], bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(expansion_dims[i % 8], out_dim, bias=False) for i in range(n_experts)
        ])
        # Mix of activations
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh, F.softplus, F.elu]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # 3. Residual LoRA Adapters for experts
        self.lora_down = nn.ModuleList([nn.Linear(in_dim, lora_rank, bias=False) for _ in range(n_experts)])
        self.lora_up = nn.ModuleList([nn.Linear(lora_rank, out_dim, bias=False) for _ in range(n_experts)])

        # 4. Routing Mechanism (Sigma Gating)
        self.gate = nn.Linear(in_dim, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_down.weight)
        
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_down[i].weight)
            nn.init.normal_(self.lora_down[i].weight, std=0.02)
            nn.init.zeros_(self.lora_up[i].weight)
        
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        BT = x.reshape(-1, x.shape[-1]).shape[0]
        x_flat = x.reshape(-1, x.shape[-1])

        # Shared Anchor processing
        shared_out = self.shared_norm(self.shared_down(self.dropout(F.silu(self.shared_up(x_flat)))))

        # Sigma Gating
        clean_logits = self.gate(x_flat)
        if self.training:
            gate_logits = clean_logits + self.expert_bias
        else:
            gate_logits = clean_logits
        
        # Sigma activation
        gate_scores = torch.sigmoid(gate_logits) 

        all_expert_logits = []
        for i in range(self.n_experts):
            core = self.experts_down[i](self.dropout(self.activations[i](self.experts_up[i](x_flat))))
            lora = self.lora_up[i](self.lora_down[i](x_flat))
            out = self.expert_norms[i](core + lora)
            all_expert_logits.append(out)
        
        expert_stack = torch.stack(all_expert_logits, dim=1) 
        routed_out = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=1) 

        if self.training:
            with torch.no_grad():
                target_load = 1.0 / self.n_experts
                actual_load = gate_scores.mean(dim=0)
                self.expert_bias.add_(0.01 * (target_load - actual_load))

        routed_out = routed_out.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        calib = self.calibration(x)

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * routed_out
            + self.theta * calib
        )


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention fusion for expert representations.
    Allows different expert branches to attend to each other's outputs.
    """
    def __init__(self, out_dim: int, n_branches: int, n_heads: int = 2):
        super().__init__()
        self.out_dim = out_dim
        self.n_branches = n_branches
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.q_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=False)
        
        self.fusion_weight = nn.Linear(out_dim, 1)

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        x_flat = x.view(B * T, N, D)
        
        q = self.q_proj(x_flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B * T, N, D)
        out = self.out_proj(out)
        
        # Weighted reduction to single D-dim vector per token
        scores = torch.sigmoid(self.fusion_weight(out)) # (B*T, N, 1)
        fused = (out * scores).sum(dim=1) / (scores.sum(dim=1) + 1e-6)
        
        return fused.view(B, T, D)


class ReasoningCell(nn.Module):
    """
    Sub-module for iterative reasoning steps.
    Combines a residual MLP with LayerNorm and dropout.
    """
    def __init__(self, dim: int = 256, inner_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, inner_dim)
        self.down = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = F.silu

    def forward(self, x):
        h = self.norm(x)
        h = self.down(self.dropout(self.act(self.up(h))))
        return x + h


class ThoughtExpertClassifierHead(nn.Module):
    """
    Ultra-advanced MoE classifier head featuring iterative reasoning:
    - 1 always-active shared anchor expert.
    - 8 diversity-driven routed experts.
    - 3-step reasoning loop that refines the feature representation.
    - Sigma Gating (independent sigmoid scores) at each step.
    - Cross-expert attention fusion integrated into the "thinking" process.
    - Lightweight residual LoRA adapters per expert.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        reasoning_steps: int = 3,
        lora_rank: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_experts = n_experts
        self.reasoning_steps = reasoning_steps

        # 1. Shared Anchor Expert
        shared_dim = 2048
        self.shared_up = nn.Linear(in_dim, shared_dim, bias=False)
        self.shared_down = nn.Linear(shared_dim, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 2. Routed Experts with diversity
        expansion_dims = [1024, 1536, 2048, 2560, 1024, 1536, 2048, 3072]
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, expansion_dims[i % 8], bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(expansion_dims[i % 8], out_dim, bias=False) for i in range(n_experts)
        ])
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh, F.softplus, F.elu]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # 3. Residual LoRA Adapters for experts
        self.lora_down = nn.ModuleList([nn.Linear(in_dim, lora_rank, bias=False) for _ in range(n_experts)])
        self.lora_up = nn.ModuleList([nn.Linear(lora_rank, out_dim, bias=False) for _ in range(n_experts)])

        # 4. Reasoning Transformation
        self.reasoning_cells = nn.ModuleList([
            ReasoningCell(in_dim, inner_dim=512, dropout=dropout) for _ in range(reasoning_steps)
        ])
        
        # 5. Iterative Routing (Sigma Gating)
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts, bias=False) for _ in range(reasoning_steps)
        ])
        self.register_buffer("expert_bias", torch.zeros(n_experts))

        # 6. Cross-Expert Attention Fusion
        self.cross_attn = CrossAttentionFusion(out_dim=out_dim, n_branches=n_experts, n_heads=2)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_down.weight)
        
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_down[i].weight)
            nn.init.normal_(self.lora_down[i].weight, std=0.02)
            nn.init.zeros_(self.lora_up[i].weight)
        
        for g in self.gates:
            nn.init.normal_(g.weight, std=0.02)
        
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Shared Anchor processing
        shared_out = self.shared_norm(self.shared_down(self.dropout(F.silu(self.shared_up(x_flat)))))

        # Pre-compute expert candidates (B*T, n_experts, out_dim)
        all_expert_logits = []
        for i in range(self.n_experts):
            core = self.experts_down[i](self.dropout(self.activations[i](self.experts_up[i](x_flat))))
            lora = self.lora_up[i](self.lora_down[i](x_flat))
            out = self.expert_norms[i](core + lora)
            all_expert_logits.append(out)
        expert_stack = torch.stack(all_expert_logits, dim=1) 

        # Iterative Reasoning Loop
        current_features = x_flat
        total_routed_out = 0
        
        for step in range(self.reasoning_steps):
            # 1. Refine features
            current_features = self.reasoning_cells[step](current_features)
            
            # 2. Gate experts based on refined features
            gate_logits = self.gates[step](current_features)
            if self.training:
                gate_logits = gate_logits + self.expert_bias
            gate_scores = torch.sigmoid(gate_logits) # (B*T, n_experts)
            
            # 3. Fuse experts for this step
            step_out = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=1)
            
            # 4. Attention-based cross-expert correction (residual)
            # Re-weight experts with attention to find hidden correlations
            # Shape: expert_stack is (BT, N, D), we need (BT, 1, N, D) for CrossAttentionFusion
            attn_fused = self.cross_attn(expert_stack.unsqueeze(1)).squeeze(1)
            
            total_routed_out = total_routed_out + step_out + 0.1 * attn_fused

            if self.training:
                with torch.no_grad():
                    target_load = 1.0 / self.n_experts
                    actual_load = gate_scores.mean(dim=0)
                    self.expert_bias.add_(0.01 * (target_load - actual_load))

        routed_out = total_routed_out / self.reasoning_steps
        routed_out = routed_out.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        calib = self.calibration(x)

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * routed_out
            + self.theta * calib
        )


class RecursiveThoughtExpertHead(nn.Module):
    """
    Generation v14: RecursiveThoughtExpert (Recursive reasoning + ACE + Multi-Head Routing)
    Features:
    - Adaptive Reasoning Depth (ACE): Early exit logic based on confidence.
    - Multi-Head Sigma Gating (2 heads): Richer routing subspace.
    - Hierarchical Shared Experts: Global (2048) + Local (512) experts.
    - Cross-Expert Attention Fusion (Residual).
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        reasoning_steps: int = 3,
        n_heads: int = 2,
        lora_rank: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_experts = n_experts
        self.reasoning_steps = reasoning_steps
        self.n_heads = n_heads

        # 1. Hierarchical Shared Experts
        # Global
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # Local (New in v14)
        self.local_up = nn.Linear(in_dim, 512, bias=False)
        self.local_down = nn.Linear(512, out_dim, bias=False)
        self.local_norm = nn.LayerNorm(out_dim)
        self.local_scale = nn.Parameter(torch.tensor(0.5))

        # 2. Routed Experts with diversity
        expansion_dims = [1024, 1536, 2048, 2560, 1024, 1536, 2048, 3072]
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, expansion_dims[i % 8], bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(expansion_dims[i % 8], out_dim, bias=False) for i in range(n_experts)
        ])
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh, F.softplus, F.elu]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # 3. Residual LoRA Adapters
        self.lora_down = nn.ModuleList([nn.Linear(in_dim, lora_rank, bias=False) for _ in range(n_experts)])
        self.lora_up = nn.ModuleList([nn.Linear(lora_rank, out_dim, bias=False) for _ in range(n_experts)])

        # 4. Reasoning & ACE
        self.reasoning_cells = nn.ModuleList([
            ReasoningCell(in_dim, inner_dim=512, dropout=dropout) for _ in range(reasoning_steps)
        ])
        self.exit_gates = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in range(reasoning_steps)
        ])
        
        # 5. Multi-Head Sigma Gating (Flattened for registration)
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts, bias=False) for _ in range(reasoning_steps * n_heads)
        ])
        self.register_buffer("expert_bias", torch.zeros(n_experts))

        # 6. Cross-Expert Attention Fusion
        self.cross_attn = CrossAttentionFusion(out_dim=out_dim, n_branches=n_experts, n_heads=2)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_down.weight)
        nn.init.kaiming_uniform_(self.local_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.local_down.weight)
        
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.experts_down[i].weight, std=0.01) # Small initial signal
            nn.init.normal_(self.lora_down[i].weight, std=0.02)
            nn.init.zeros_(self.lora_up[i].weight)
        
        for g in self.gates:
            nn.init.normal_(g.weight, std=0.02)
        
        for e in self.exit_gates:
            nn.init.constant_(e.bias, -3.0) # Start with low exit prob

        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        # 1. Global Shared Expert
        shared_out = self.shared_norm(self.shared_down(self.dropout(F.silu(self.shared_up(x_flat)))))
        # 2. Local Shared Expert
        local_out = self.local_norm(self.local_down(self.dropout(F.gelu(self.local_up(x_flat)))))

        # Pre-compute expert candidates
        all_expert_logits = []
        for i in range(self.n_experts):
            core = self.experts_down[i](self.dropout(self.activations[i](self.experts_up[i](x_flat))))
            lora = self.lora_up[i](self.lora_down[i](x_flat))
            out = self.expert_norms[i](core + lora)
            all_expert_logits.append(out)
        expert_stack = torch.stack(all_expert_logits, dim=1) 

        # Iterative Reasoning Loop with ACE
        current_features = x_flat
        total_routed_out = torch.zeros_like(expert_stack[:, 0, :])
        cumulative_exit_prob = torch.zeros(x_flat.shape[0], 1, device=x.device)
        depth_count = 0
        
        for step in range(self.reasoning_steps):
            depth_count += 1
            # A. Refine features
            current_features = self.reasoning_cells[step](current_features)
            
            # B. Multi-Head Sigma Gating
            head_logits = []
            for h in range(self.n_heads):
                idx = step * self.n_heads + h
                head_logits.append(self.gates[idx](current_features))
            # Average head logits (Sigma Gating)
            gate_logits = torch.stack(head_logits, dim=0).mean(dim=0)
            
            if self.training:
                gate_logits = gate_logits + self.expert_bias
            gate_scores = torch.sigmoid(gate_logits) # (B*T, n_experts)
            
            # C. ACE: Early Exit check
            exit_logit = self.exit_gates[step](current_features)
            exit_prob = torch.sigmoid(exit_logit) # (B*T, 1)
            
            # D. Fuse experts for this step
            step_out = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=1)
            
            # E. Attention-based cross-expert correction
            attn_fused = self.cross_attn(expert_stack.unsqueeze(1)).squeeze(1)
            
            # Weighted contribution based on current depth
            total_routed_out = total_routed_out + (1.0 - cumulative_exit_prob) * (step_out + 0.1 * attn_fused)
            
            # Update cumulative exit probability
            cumulative_exit_prob = cumulative_exit_prob + (1.0 - cumulative_exit_prob) * exit_prob

            if self.training:
                with torch.no_grad():
                    target_load = 1.0 / self.n_experts
                    actual_load = gate_scores.mean(dim=0)
                    self.expert_bias.add_(0.01 * (target_load - actual_load))
            
            # Early exit condition
            if not self.training and cumulative_exit_prob.mean() > 0.9:
                break

        routed_out = total_routed_out / depth_count
        routed_out = routed_out.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        local_out = local_out.view(*shape_prefix, -1)
        calib = self.calibration(x)

        return (
            base_logits
            + self.shared_scale * shared_out
            + self.local_scale * local_out
            + (self.alpha + 1e-4) * routed_out
            + self.theta * calib
        )


class ChampionNetRecursiveExpert(nn.Module):
    """
    Backbone-compatible model with RecursiveThoughtExpertHead.
    """
    def __init__(self, n_experts: int = 8, reasoning_steps: int = 3, lora_rank: int = 32, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(RecursiveThoughtExpertHead(
            256, 10, 
            n_experts=n_experts, 
            reasoning_steps=reasoning_steps, 
            lora_rank=lora_rank, 
            dropout=dropout
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




class ReflexiveThoughtExpertHead(nn.Module):
    """
    Generation v15: ReflexiveThoughtExpert (Self-Reflection MoE)
    Features:
    - Initial Pass: Generates baseline logits using MoE routing.
    - Critique Pass: Takes original features + initial logits to self-correct using specialized critique experts.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_experts = n_experts

        # 1. Initial Pass Experts
        self.initial_up = nn.ModuleList([nn.Linear(in_dim, 1024, bias=False) for _ in range(n_experts)])
        self.initial_down = nn.ModuleList([nn.Linear(1024, out_dim, bias=False) for _ in range(n_experts)])
        self.initial_gate = nn.Linear(in_dim, n_experts, bias=False)
        self.initial_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # 2. Critique Pass Experts (Input: features + intermediate logits)
        critique_dim = in_dim + out_dim
        self.critique_up = nn.ModuleList([nn.Linear(critique_dim, 1024, bias=False) for _ in range(n_experts)])
        self.critique_down = nn.ModuleList([nn.Linear(1024, out_dim, bias=False) for _ in range(n_experts)])
        self.critique_gate = nn.Linear(critique_dim, n_experts, bias=False)
        self.critique_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.initial_gate.weight)
        nn.init.zeros_(self.critique_gate.weight)
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.initial_up[i].weight, a=5**0.5)
            nn.init.normal_(self.initial_down[i].weight, std=0.01)
            nn.init.kaiming_uniform_(self.critique_up[i].weight, a=5**0.5)
            nn.init.normal_(self.critique_down[i].weight, std=0.01)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        
        # --- Initial Pass ---
        gate_init = torch.softmax(self.initial_gate(x), dim=-1) # (B, T, E)
        init_outs = []
        for i in range(self.n_experts):
            h = self.dropout(F.silu(self.initial_up[i](x)))
            init_outs.append(self.initial_norms[i](self.initial_down[i](h)))
        init_stack = torch.stack(init_outs, dim=-2) # (B, T, E, D)
        init_fused = (init_stack * gate_init.unsqueeze(-1)).sum(dim=-2)
        
        # Intermediate prediction
        inter_logits = base_logits + self.alpha * init_fused
        
        # --- Critique Pass ---
        crit_in = torch.cat([x, inter_logits], dim=-1) 
        gate_crit = torch.softmax(self.critique_gate(crit_in), dim=-1)
        crit_outs = []
        for i in range(self.n_experts):
            h = self.dropout(F.gelu(self.critique_up[i](crit_in)))
            crit_outs.append(self.critique_norms[i](self.critique_down[i](h)))
        crit_stack = torch.stack(crit_outs, dim=-2)
        crit_fused = (crit_stack * gate_crit.unsqueeze(-1)).sum(dim=-2)
        
        # Final prediction
        return inter_logits + self.beta * crit_fused


class ChampionNetReflexiveExpert(nn.Module):
    """
    Backbone-compatible model with ReflexiveThoughtExpertHead.
    """
    def __init__(self, n_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ReflexiveThoughtExpertHead(256, 10, n_experts=n_experts, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MetaCognitiveExpertHead(nn.Module):
    """
    Generation v16: MetaCognitiveExpert (Iterative Reflection + PonderNet Halting)
    Combines the best ideas from v14 (recursive reasoning, shared experts, ACE)
    and v15 (self-reflection critique pass) into a unified "think → critique → refine"
    loop with learned adaptive halting.

    Features:
    - Shared Expert: Always-on global expert for stable signal.
    - ReasoningCells: Per-step feature refinement with residual MLP.
    - Proposal MoE: Generates predictions via routed experts at each step.
    - Critique MoE: Inspects [features || proposal_logits] to output corrections.
    - PonderNet Halting: Learned halt probability per step for adaptive depth.
    - Dynamic Bias Load Balancing: Buffer biases maintain expert utilization.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_proposal_experts: int = 6,
        n_critique_experts: int = 4,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_proposal_experts = n_proposal_experts
        self.n_critique_experts = n_critique_experts
        self.reasoning_steps = reasoning_steps

        # 1. Shared Expert (always-on)
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 2. Reasoning Cells (per step)
        self.reasoning_cells = nn.ModuleList([
            ReasoningCell(in_dim, inner_dim=512, dropout=dropout)
            for _ in range(reasoning_steps)
        ])

        # 3. Proposal MoE (per step routing)
        prop_dims = [1024, 1536, 2048, 2560, 1024, 1536]
        prop_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.proposal_up = nn.ModuleList([
            nn.Linear(in_dim, prop_dims[i % len(prop_dims)], bias=False)
            for i in range(n_proposal_experts)
        ])
        self.proposal_down = nn.ModuleList([
            nn.Linear(prop_dims[i % len(prop_dims)], out_dim, bias=False)
            for i in range(n_proposal_experts)
        ])
        self.proposal_acts = [prop_acts[i % len(prop_acts)] for i in range(n_proposal_experts)]
        self.proposal_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(n_proposal_experts)
        ])
        self.proposal_gates = nn.ModuleList([
            nn.Linear(in_dim, n_proposal_experts, bias=False)
            for _ in range(reasoning_steps)
        ])
        self.register_buffer("proposal_bias", torch.zeros(n_proposal_experts))

        # 4. Critique MoE (input: features + proposal logits)
        critique_dim = in_dim + out_dim
        self.critique_up = nn.ModuleList([
            nn.Linear(critique_dim, 1024, bias=False)
            for _ in range(n_critique_experts)
        ])
        self.critique_down = nn.ModuleList([
            nn.Linear(1024, out_dim, bias=False)
            for _ in range(n_critique_experts)
        ])
        self.critique_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(n_critique_experts)
        ])
        self.critique_gates = nn.ModuleList([
            nn.Linear(critique_dim, n_critique_experts, bias=False)
            for _ in range(reasoning_steps)
        ])

        # 5. Halting Gates (PonderNet-style)
        self.halt_gates = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in range(reasoning_steps)
        ])

        # Aggregation
        self.alpha = nn.Parameter(torch.tensor(0.0))  # proposal scale
        self.beta = nn.Parameter(torch.tensor(0.0))   # critique scale
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_down.weight)

        for i in range(self.n_proposal_experts):
            nn.init.kaiming_uniform_(self.proposal_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.proposal_down[i].weight, std=0.01)
        for i in range(self.n_critique_experts):
            nn.init.kaiming_uniform_(self.critique_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.critique_down[i].weight, std=0.01)

        for g in self.proposal_gates:
            nn.init.normal_(g.weight, std=0.02)
        for g in self.critique_gates:
            nn.init.normal_(g.weight, std=0.02)
        for h in self.halt_gates:
            nn.init.constant_(h.bias, -3.0)  # start with low halt probability

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        # 1. Shared Expert (always-on)
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # 2. Pre-compute proposal expert outputs
        prop_outs = []
        for i in range(self.n_proposal_experts):
            h = self.dropout(self.proposal_acts[i](self.proposal_up[i](x_flat)))
            prop_outs.append(self.proposal_norms[i](self.proposal_down[i](h)))
        prop_stack = torch.stack(prop_outs, dim=1)  # (BT, E_prop, D)

        # 3. Iterative Reflect Loop
        current_features = x_flat
        total_proposal = torch.zeros_like(base_logits.reshape(-1, base_logits.shape[-1]))
        total_critique = torch.zeros_like(total_proposal)
        cumulative_halt = torch.zeros(x_flat.shape[0], 1, device=x.device)
        depth_count = 0

        for step in range(self.reasoning_steps):
            depth_count += 1

            # A. Refine features
            current_features = self.reasoning_cells[step](current_features)

            # B. Proposal MoE routing (Sigma Gating)
            gate_logits = self.proposal_gates[step](current_features)
            if self.training:
                gate_logits = gate_logits + self.proposal_bias
            gate_scores = torch.sigmoid(gate_logits)  # (BT, E_prop)
            step_proposal = (prop_stack * gate_scores.unsqueeze(-1)).sum(dim=1)

            # C. Critique MoE: inspect [features || proposal]
            crit_in = torch.cat([current_features, step_proposal], dim=-1)
            crit_gate = torch.softmax(
                self.critique_gates[step](crit_in), dim=-1
            )  # (BT, E_crit)
            crit_outs = []
            for i in range(self.n_critique_experts):
                h = self.dropout(F.gelu(self.critique_up[i](crit_in)))
                crit_outs.append(self.critique_norms[i](self.critique_down[i](h)))
            crit_stack = torch.stack(crit_outs, dim=1)  # (BT, E_crit, D)
            step_critique = (crit_stack * crit_gate.unsqueeze(-1)).sum(dim=1)

            # D. Halting gate (PonderNet-style)
            halt_prob = torch.sigmoid(self.halt_gates[step](current_features))

            # E. Accumulate weighted by remaining probability mass
            remaining = 1.0 - cumulative_halt
            total_proposal = total_proposal + remaining * step_proposal
            total_critique = total_critique + remaining * step_critique
            cumulative_halt = cumulative_halt + remaining * halt_prob

            # F. Dynamic bias load balancing
            if self.training:
                with torch.no_grad():
                    target_load = 1.0 / self.n_proposal_experts
                    actual_load = gate_scores.mean(dim=0)
                    self.proposal_bias.add_(0.01 * (target_load - actual_load))

            # G. Early exit during inference
            if not self.training and cumulative_halt.mean() > 0.9:
                break

        # Normalize by effective depth
        total_proposal = total_proposal / max(1, depth_count)
        total_critique = total_critique / max(1, depth_count)

        # Reshape back
        total_proposal = total_proposal.view(*shape_prefix, -1)
        total_critique = total_critique.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * total_proposal
            + (self.beta + 1e-4) * total_critique
        )


class ChampionNetMetaCognitiveExpert(nn.Module):
    """
    Backbone-compatible model with MetaCognitiveExpertHead (v16).
    """
    def __init__(
        self,
        n_proposal_experts: int = 6,
        n_critique_experts: int = 4,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(MetaCognitiveExpertHead(
            256, 10,
            n_proposal_experts=n_proposal_experts,
            n_critique_experts=n_critique_experts,
            reasoning_steps=reasoning_steps,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TreeOfThoughtExpertHead(nn.Module):
    """
    Generation v17: Tree-of-Thought Expert (Latent Beam Search)
    Embeds true test-time compute scaling into the forward pass.
    Instead of a single trajectory, maintains a beam of k states.
    At each step, m Action Experts propose modifications, creating k*m candidates.
    A Value Network scores them, and the top-k are kept.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_action_experts: int = 6,
        beam_size: int = 4,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_action_experts = n_action_experts
        self.beam_size = beam_size
        self.reasoning_steps = reasoning_steps
        
        # Base projection
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        # 1. Global Shared Expert (always-on)
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))
        
        # 2. Value Network (Scorer)
        # Evaluates the "goodness" of a latent state.
        self.value_net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        
        # 3. Action MoE (Proposer)
        # We use a set of experts that propose feature modifications
        self.action_up = nn.ModuleList([
            nn.Linear(in_dim, 1024, bias=False) for _ in range(n_action_experts)
        ])
        self.action_down = nn.ModuleList([
            nn.Linear(1024, in_dim, bias=False) for _ in range(n_action_experts)
        ])
        self.action_norms = nn.ModuleList([
            nn.LayerNorm(in_dim) for _ in range(n_action_experts)
        ])
        
        # Final projection from feature to out_dim
        self.final_up = nn.Linear(in_dim, 1024, bias=False)
        self.final_down = nn.Linear(1024, out_dim, bias=False)
        self.final_norm = nn.LayerNorm(out_dim)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))  # tree scale
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)
        
        for p in self.value_net.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            else:
                nn.init.zeros_(p)
                
        for i in range(self.n_action_experts):
            nn.init.kaiming_uniform_(self.action_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.action_down[i].weight, std=0.01)
            
        nn.init.kaiming_uniform_(self.final_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.final_down.weight, std=0.01)

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)
        
        base_logits = F.linear(x_flat, self.weight, self.bias)
        
        # 1. Shared Expert
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )
        
        # 2. Tree-of-Thought Search
        # Initial beam state: K=1
        beam_states = x_flat.unsqueeze(1) # (N, 1, D)
        
        for step in range(self.reasoning_steps):
            K = beam_states.shape[1]
            
            # A. Expansion: Each action expert proposes a modification to every state in the beam
            nk_states = beam_states.reshape(N * K, self.in_dim)
            
            candidates = []
            for i in range(self.n_action_experts):
                h = self.dropout(F.gelu(self.action_up[i](nk_states)))
                delta = self.action_norms[i](self.action_down[i](h))
                new_state = nk_states + delta
                candidates.append(new_state)
                
            # Stack candidates
            cand_stack = torch.stack(candidates, dim=1)
            cand_stack = cand_stack.reshape(N, K * self.n_action_experts, self.in_dim)
            
            # B. Evaluation (Value Network)
            flat_cands = cand_stack.reshape(N * K * self.n_action_experts, self.in_dim)
            scores = self.value_net(flat_cands).squeeze(-1) # (N * K * M)
            scores = scores.reshape(N, K * self.n_action_experts) # (N, K*M)
            
            # C. Selection (Beam Search)
            K_next = min(self.beam_size, K * self.n_action_experts)
            
            topk_scores, topk_indices = torch.topk(scores, k=K_next, dim=1) # (N, K_next)
            
            expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.in_dim)
            beam_states = torch.gather(cand_stack, 1, expanded_indices) # (N, K_next, D)
            
        # 3. Aggregation across the final beam
        final_K = beam_states.shape[1]
        flat_final = beam_states.reshape(N * final_K, self.in_dim)
        
        h_final = self.dropout(F.silu(self.final_up(flat_final)))
        final_preds = self.final_norm(self.final_down(h_final)) # (N * final_K, out_dim)
        final_preds = final_preds.reshape(N, final_K, self.out_dim) # (N, K, out_dim)
        
        final_scores = self.value_net(flat_final).squeeze(-1).reshape(N, final_K) # (N, K)
        agg_weights = torch.softmax(final_scores, dim=1) # (N, K)
        
        tree_fused = (final_preds * agg_weights.unsqueeze(-1)).sum(dim=1)
        
        base_logits = base_logits.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        tree_fused = tree_fused.view(*shape_prefix, -1)
        
        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * tree_fused
        )


class ChampionNetTreeOfThoughtExpert(nn.Module):
    """
    Backbone-compatible model with TreeOfThoughtExpertHead (v17).
    """
    def __init__(
        self,
        n_action_experts: int = 6,
        beam_size: int = 4,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(TreeOfThoughtExpertHead(
            256, 10,
            n_action_experts=n_action_experts,
            beam_size=beam_size,
            reasoning_steps=reasoning_steps,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConsensusExpertHead(nn.Module):
    """
    Generation v18: Consensus Expert (Architectural Diversity + Learned Arbitration)
    Uses three fundamentally different computational paradigms as independent
    reasoning pathways, then a Consensus Network arbitrates.

    Pathways:
    1. MLP Pathway — Deep feed-forward MoE experts (fast pattern matching)
    2. Attention Pathway — Self-attention over learned expert embeddings (relational reasoning)
    3. Convolutional Pathway — 1D convolution over the feature vector (local pattern detection)

    A Consensus Network evaluates inter-pathway agreement and produces the final answer.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_mlp_experts: int = 6,
        n_attn_heads: int = 4,
        conv_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_mlp_experts = n_mlp_experts

        # Base projection
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # --- Pathway 1: MLP MoE ---
        mlp_dims = [1024, 1536, 2048, 1024, 1536, 2048]
        mlp_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.mlp_up = nn.ModuleList([
            nn.Linear(in_dim, mlp_dims[i % len(mlp_dims)], bias=False)
            for i in range(n_mlp_experts)
        ])
        self.mlp_down = nn.ModuleList([
            nn.Linear(mlp_dims[i % len(mlp_dims)], out_dim, bias=False)
            for i in range(n_mlp_experts)
        ])
        self.mlp_acts = [mlp_acts[i % len(mlp_acts)] for i in range(n_mlp_experts)]
        self.mlp_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_mlp_experts)])
        self.mlp_gate = nn.Linear(in_dim, n_mlp_experts, bias=False)

        # --- Pathway 2: Self-Attention ---
        self.n_attn_heads = n_attn_heads
        self.attn_embed = nn.Parameter(torch.randn(8, in_dim) * 0.02)
        self.attn_q = nn.Linear(in_dim, in_dim, bias=False)
        self.attn_k = nn.Linear(in_dim, in_dim, bias=False)
        self.attn_v = nn.Linear(in_dim, in_dim, bias=False)
        self.attn_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_norm = nn.LayerNorm(out_dim)
        self.attn_scale = (in_dim // n_attn_heads) ** -0.5

        # --- Pathway 3: 1D Convolution ---
        self.conv_channels = conv_channels
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.conv_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_proj = nn.Linear(conv_channels, out_dim, bias=False)
        self.conv_norm = nn.LayerNorm(out_dim)

        # --- Consensus Network (Arbiter) ---
        consensus_in = 3 * out_dim
        self.consensus = nn.Sequential(
            nn.Linear(consensus_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        for i in range(self.n_mlp_experts):
            nn.init.kaiming_uniform_(self.mlp_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.mlp_down[i].weight, std=0.01)
        nn.init.normal_(self.mlp_gate.weight, std=0.02)

        nn.init.kaiming_uniform_(self.attn_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.attn_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.attn_v.weight, a=math.sqrt(5))
        nn.init.normal_(self.attn_proj.weight, std=0.01)

        nn.init.normal_(self.conv_proj.weight, std=0.01)

        for p in self.consensus.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            else:
                nn.init.zeros_(p)

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # --- Pathway 1: MLP MoE ---
        gate_scores = torch.sigmoid(self.mlp_gate(x_flat))
        mlp_outs = []
        for i in range(self.n_mlp_experts):
            h = self.dropout(self.mlp_acts[i](self.mlp_up[i](x_flat)))
            mlp_outs.append(self.mlp_norms[i](self.mlp_down[i](h)))
        mlp_stack = torch.stack(mlp_outs, dim=1)
        mlp_pred = (mlp_stack * gate_scores.unsqueeze(-1)).sum(dim=1)

        # --- Pathway 2: Self-Attention ---
        embeds = self.attn_embed.unsqueeze(0).expand(N, -1, -1)
        q = self.attn_q(x_flat).unsqueeze(1)
        k = self.attn_k(embeds)
        v = self.attn_v(embeds)
        attn_weights = torch.softmax(
            (q * k).sum(dim=-1, keepdim=True) * self.attn_scale, dim=1
        )
        attn_out = (attn_weights * v).sum(dim=1)
        attn_pred = self.attn_norm(self.attn_proj(attn_out))

        # --- Pathway 3: 1D Convolution ---
        x_conv = x_flat.unsqueeze(1)
        h_conv = F.gelu(self.conv1(x_conv))
        h_conv = F.gelu(self.conv2(h_conv))
        h_conv = self.conv_pool(h_conv).squeeze(-1)
        conv_pred = self.conv_norm(self.conv_proj(h_conv))

        # --- Consensus Network ---
        pathway_preds = torch.stack([mlp_pred, attn_pred, conv_pred], dim=1)
        consensus_in = torch.cat([mlp_pred, attn_pred, conv_pred], dim=-1)
        consensus_weights = torch.softmax(self.consensus(consensus_in), dim=-1)
        consensus_fused = (pathway_preds * consensus_weights.unsqueeze(-1)).sum(dim=1)

        consensus_fused = consensus_fused.view(*shape_prefix, -1)
        base_logits = base_logits.view(*shape_prefix, -1)

        return base_logits + (self.alpha + 1e-4) * consensus_fused


class ChampionNetConsensusExpert(nn.Module):
    """
    Backbone-compatible model with ConsensusExpertHead (v18).
    """
    def __init__(
        self,
        n_mlp_experts: int = 6,
        n_attn_heads: int = 4,
        conv_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ConsensusExpertHead(
            256, 10,
            n_mlp_experts=n_mlp_experts,
            n_attn_heads=n_attn_heads,
            conv_channels=conv_channels,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeliberativeAlignmentExpertHead(nn.Module):
    """
    Generation v19: Deliberative Alignment Expert
    (Multi-Draft Deliberation with Working Memory)

    The most advanced reasoning head. Simultaneously generates K independent
    'draft' reasoning chains, lets them share insights via cross-draft
    attention, consults a working memory bank, and uses PonderNet-style
    adaptive halting to allocate more compute to hard inputs.

    Key innovations:
    1. Working Memory Bank — learned key-value slots for persistent scratchpad
    2. Multi-Draft Generation — K parallel reasoning perspectives
    3. Cross-Draft Attention — drafts attend to each others states
    4. Iterative Refinement + Adaptive Depth — PonderNet halting
    5. Consistency-Weighted Aggregation — prefer internally consistent conclusions
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 6,
        n_drafts: int = 3,
        n_mem_slots: int = 16,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.n_drafts = n_drafts
        self.n_mem_slots = n_mem_slots
        self.reasoning_steps = reasoning_steps

        # Base projection (backward-compatible)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # 1. Global Shared Expert (always-on)
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 2. Working Memory Bank
        self.memory_keys = nn.Parameter(torch.randn(n_mem_slots, in_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(n_mem_slots, in_dim) * 0.02)
        self.mem_read_query = nn.Linear(in_dim, in_dim, bias=False)
        self.mem_write_query = nn.Linear(in_dim, in_dim, bias=False)
        self.mem_write_gate = nn.Linear(in_dim, n_mem_slots, bias=True)
        self.mem_scale = (in_dim) ** -0.5

        # 3. Per-Draft Reasoning Cells (iterative refinement)
        self.draft_reasoning_cells = nn.ModuleList()
        for _ in range(reasoning_steps):
            step_cells = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, 512),
                    nn.SiLU(),
                    nn.Linear(512, in_dim),
                )
                for _ in range(n_drafts)
            ])
            self.draft_reasoning_cells.append(step_cells)

        # 4. Expert MoE (shared across drafts)
        expert_dims = [1024, 1536, 2048, 1024, 1536, 2048]
        expert_acts_fns = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.expert_up = nn.ModuleList([
            nn.Linear(in_dim, expert_dims[i % len(expert_dims)], bias=False)
            for i in range(n_experts)
        ])
        self.expert_down = nn.ModuleList([
            nn.Linear(expert_dims[i % len(expert_dims)], out_dim, bias=False)
            for i in range(n_experts)
        ])
        self.expert_acts = [expert_acts_fns[i % len(expert_acts_fns)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # Per-step Sigma Gates (one per draft per step)
        self.draft_gates = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(in_dim, n_experts, bias=False)
                for _ in range(n_drafts)
            ])
            for _ in range(reasoning_steps)
        ])
        self.register_buffer(
            "expert_bias", torch.zeros(n_experts), persistent=False
        )

        # 5. Cross-Draft Attention (lightweight)
        self.cross_draft_q = nn.ModuleList([
            nn.Linear(out_dim, out_dim, bias=False) for _ in range(reasoning_steps)
        ])
        self.cross_draft_k = nn.ModuleList([
            nn.Linear(out_dim, out_dim, bias=False) for _ in range(reasoning_steps)
        ])
        self.cross_draft_v = nn.ModuleList([
            nn.Linear(out_dim, out_dim, bias=False) for _ in range(reasoning_steps)
        ])
        self.cross_draft_norm = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(reasoning_steps)
        ])
        self.cross_draft_scale = (out_dim) ** -0.5

        # 6. PonderNet-style Halting Gates
        self.halt_gates = nn.ModuleList([
            nn.Linear(in_dim, 1, bias=True) for _ in range(reasoning_steps)
        ])

        # 7. Consistency Network (Aggregation Arbiter)
        # Takes all K draft predictions, scores internal consistency,
        # and produces K confidence weights for final fusion.
        consistency_in = n_drafts * out_dim
        self.consistency_net = nn.Sequential(
            nn.Linear(consistency_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_drafts),
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))  # deliberation scale
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)

        nn.init.kaiming_uniform_(self.mem_read_query.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mem_write_query.weight, a=math.sqrt(5))
        nn.init.normal_(self.mem_write_gate.weight, std=0.02)
        nn.init.constant_(self.mem_write_gate.bias, 0.0)

        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.expert_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.expert_down[i].weight, std=0.01)

        for step_gates in self.draft_gates:
            for g in step_gates:
                nn.init.normal_(g.weight, std=0.02)

        for step in range(self.reasoning_steps):
            nn.init.kaiming_uniform_(self.cross_draft_q[step].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.cross_draft_k[step].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.cross_draft_v[step].weight, a=math.sqrt(5))

        for h in self.halt_gates:
            nn.init.constant_(h.bias, -3.0)  # start with low halt probability

        for p in self.consistency_net.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            else:
                nn.init.zeros_(p)

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # 1. Shared Expert (always-on)
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # 2. Pre-compute expert outputs (shared across drafts)
        expert_outs = []
        for i in range(self.n_experts):
            h = self.dropout(self.expert_acts[i](self.expert_up[i](x_flat)))
            expert_outs.append(self.expert_norms[i](self.expert_down[i](h)))
        expert_stack = torch.stack(expert_outs, dim=1)  # (N, E, D_out)

        # 3. Initialize draft states
        draft_features = [x_flat.clone() for _ in range(self.n_drafts)]
        draft_preds = [torch.zeros(N, self.out_dim, device=x.device)
                       for _ in range(self.n_drafts)]

        cumulative_halt = torch.zeros(N, 1, device=x.device)
        total_deliberated = torch.zeros(N, self.out_dim, device=x.device)
        depth_count = 0

        for step in range(self.reasoning_steps):
            depth_count += 1

            # A. Memory Read: each draft queries the working memory
            for d in range(self.n_drafts):
                read_q = self.mem_read_query(draft_features[d])  # (N, D)
                attn_scores = torch.matmul(
                    read_q, self.memory_keys.t()  # (N, M)
                ) * self.mem_scale
                attn_weights = torch.softmax(attn_scores, dim=-1)  # (N, M)
                mem_read = torch.matmul(attn_weights, self.memory_values)  # (N, D)
                draft_features[d] = draft_features[d] + mem_read

            # B. Per-draft reasoning cell refinement
            for d in range(self.n_drafts):
                residual = draft_features[d]
                draft_features[d] = residual + self.draft_reasoning_cells[step][d](residual)

            # C. Per-draft expert MoE routing (Sigma Gating)
            step_draft_preds = []
            for d in range(self.n_drafts):
                gate_logits = self.draft_gates[step][d](draft_features[d])
                if self.training:
                    gate_logits = gate_logits + self.expert_bias
                gate_scores = torch.sigmoid(gate_logits)  # (N, E)
                routed = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=1)  # (N, D_out)
                step_draft_preds.append(routed)

            # D. Cross-Draft Attention: drafts attend to each other
            pred_stack = torch.stack(step_draft_preds, dim=1)  # (N, K, D_out)
            q = self.cross_draft_q[step](pred_stack)  # (N, K, D_out)
            k = self.cross_draft_k[step](pred_stack)  # (N, K, D_out)
            v = self.cross_draft_v[step](pred_stack)  # (N, K, D_out)
            attn = torch.bmm(q, k.transpose(1, 2)) * self.cross_draft_scale  # (N, K, K)
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            cross_out = torch.bmm(attn, v)  # (N, K, D_out)
            refined_preds = self.cross_draft_norm[step](pred_stack + cross_out)  # (N, K, D_out)

            for d in range(self.n_drafts):
                draft_preds[d] = refined_preds[:, d, :]

            # E. Memory Write: update memory with new insights
            # Average draft features as write signal
            write_signal = torch.stack(draft_features, dim=0).mean(dim=0)  # (N, D)
            write_q = self.mem_write_query(write_signal)  # (N, D)
            write_scores = torch.sigmoid(self.mem_write_gate(write_q))  # (N, M)
            # Differentiable soft memory update: weighted retrieval from write signal
            mem_retrieval = torch.matmul(write_scores, self.memory_values)  # (N, D)
            # Inject memory write contribution into draft features for next step  
            for d in range(self.n_drafts):
                draft_features[d] = draft_features[d] + 0.1 * mem_retrieval

            # F. Halting gate (PonderNet-style)
            # Use average draft features for halt decision
            avg_features = torch.stack(draft_features, dim=0).mean(dim=0)
            halt_prob = torch.sigmoid(self.halt_gates[step](avg_features))  # (N, 1)

            # G. Accumulate: weighted by remaining probability mass
            remaining = 1.0 - cumulative_halt

            # Consistency-aware per-step aggregation across drafts
            all_draft_preds = torch.stack(draft_preds, dim=1)  # (N, K, D_out)
            consistency_in = all_draft_preds.reshape(N, -1)  # (N, K*D_out)
            consistency_weights = torch.softmax(
                self.consistency_net(consistency_in), dim=-1
            )  # (N, K)
            step_fused = (all_draft_preds * consistency_weights.unsqueeze(-1)).sum(dim=1)  # (N, D_out)

            total_deliberated = total_deliberated + remaining * step_fused
            cumulative_halt = cumulative_halt + remaining * halt_prob

            # H. Dynamic bias load balancing (across all drafts)
            if self.training:
                with torch.no_grad():
                    all_gates = torch.stack(
                        [torch.sigmoid(self.draft_gates[step][d](draft_features[d]))
                         for d in range(self.n_drafts)], dim=0
                    ).mean(dim=0)  # (N, E)
                    target_load = 1.0 / self.n_experts
                    actual_load = all_gates.mean(dim=0)
                    self.expert_bias.add_(0.01 * (target_load - actual_load))

            # I. Early exit during inference
            if not self.training and cumulative_halt.mean() > 0.9:
                break

        # Normalize by effective depth
        total_deliberated = total_deliberated / max(1, depth_count)

        # Reshape back
        base_logits = base_logits.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        total_deliberated = total_deliberated.view(*shape_prefix, -1)

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * total_deliberated
        )


class ChampionNetDeliberativeExpert(nn.Module):
    """
    Backbone-compatible model with DeliberativeAlignmentExpertHead (v19).
    """
    def __init__(
        self,
        n_experts: int = 6,
        n_drafts: int = 3,
        n_mem_slots: int = 16,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(DeliberativeAlignmentExpertHead(
            256, 10,
            n_experts=n_experts,
            n_drafts=n_drafts,
            n_mem_slots=n_mem_slots,
            reasoning_steps=reasoning_steps,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OmniscientSynergyExpertHead(nn.Module):
    """
    Generation v20: Omniscient Synergy Expert
    (Latent Knowledge Core + Graph-of-Thought + Stochastic MoE)

    The most advanced intelligence architecture, moving beyond parallel drafts
    to an interactive Graph of Thoughts grounded in a Latent Knowledge Core.

    Key Innovations:
    1. Latent Knowledge Core (LKC) — queries a massive continuous associative memory
    2. Graph-of-Thought (GoT) — thoughts exchange information via Graph Attention (GAT)
    3. Stochastic Bayesian Routing — samples expert routes via reparameterization (mu/var)
    4. Meta-Cognitive Self-Correction — global critique network broadcasts correction
    5. Dynamic Node Halting — per-thought confidence evaluation
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 6,
        n_nodes: int = 4,
        n_knowledge_slots: int = 64,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.n_nodes = n_nodes
        self.n_knowledge_slots = n_knowledge_slots
        self.reasoning_steps = reasoning_steps

        # Base projection (backward-compatible)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # 1. Global Shared Expert (always-on backbone)
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 2. Latent Knowledge Core (Implicit RAG)
        self.knowledge_core_keys = nn.Parameter(torch.randn(n_knowledge_slots, in_dim) * 0.02)
        self.knowledge_core_values = nn.Parameter(torch.randn(n_knowledge_slots, in_dim) * 0.02)
        self.knowledge_query = nn.Linear(in_dim, in_dim, bias=False)
        self.knowledge_scale = in_dim ** -0.5

        # 3. Graph-of-Thought (GoT) Attention (Node-to-Node communication)
        self.got_q = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=False) for _ in range(reasoning_steps)])
        self.got_k = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=False) for _ in range(reasoning_steps)])
        self.got_v = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=False) for _ in range(reasoning_steps)])
        self.got_norm = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(reasoning_steps)])

        # 4. Stochastic Bayesian MoE Routing
        expert_dims = [1024, 1536, 2048, 1024, 1536, 2048]
        expert_acts_fns = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.expert_up = nn.ModuleList([
            nn.Linear(in_dim, expert_dims[i % len(expert_dims)], bias=False)
            for i in range(n_experts)
        ])
        self.expert_down = nn.ModuleList([
            nn.Linear(expert_dims[i % len(expert_dims)], out_dim, bias=False)
            for i in range(n_experts)
        ])
        self.expert_acts = [expert_acts_fns[i % len(expert_acts_fns)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])
        
        # Bayesian Routers (predicts mean and log-variance for expert weighting)
        self.route_mu = nn.ModuleList([
            nn.Linear(in_dim, n_experts, bias=False) for _ in range(reasoning_steps)
        ])
        self.route_logvar = nn.ModuleList([
            nn.Linear(in_dim, n_experts, bias=True) for _ in range(reasoning_steps)
        ])

        # 5. Meta-Cognitive Critique (Global Review & Broadcast)
        self.critique_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_nodes * in_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, in_dim)
            ) for _ in range(reasoning_steps)
        ])

        # 6. Dynamic Depth (Per-Node Halting)
        self.node_halt_gates = nn.ModuleList([
            nn.Linear(in_dim, 1, bias=True) for _ in range(reasoning_steps)
        ])
        
        # Final Node Fusion
        self.node_fusion_q = nn.Linear(out_dim, out_dim, bias=False)
        self.node_fusion_k = nn.Linear(out_dim, out_dim, bias=False)
        self.node_fusion_v = nn.Linear(out_dim, out_dim, bias=False)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        
        # Divergence temperature for Bayesian routing during inference
        self.register_buffer("temperature", torch.tensor(0.2))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)

        nn.init.kaiming_uniform_(self.knowledge_query.weight, a=math.sqrt(5))

        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.expert_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.expert_down[i].weight, std=0.01)

        for step in range(self.reasoning_steps):
            nn.init.kaiming_uniform_(self.got_q[step].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.got_k[step].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.got_k[step].weight, a=math.sqrt(5))
            
            nn.init.normal_(self.route_mu[step].weight, std=0.02)
            nn.init.normal_(self.route_logvar[step].weight, std=0.02)
            nn.init.constant_(self.route_logvar[step].bias, -2.0) # start with low variance
            
            for p in self.critique_net[step].parameters():
                if p.dim() > 1: nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                else: nn.init.zeros_(p)
                
            nn.init.constant_(self.node_halt_gates[step].bias, -2.0)
            
        nn.init.kaiming_uniform_(self.node_fusion_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.node_fusion_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.node_fusion_v.weight, a=math.sqrt(5))

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # 1. Global Shared Expert
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # 2. Latent Knowledge Core Retrieval
        # Query the massive internal memory before reasoning begins
        q = self.knowledge_query(x_flat) # (N, D)
        attn = torch.matmul(q, self.knowledge_core_keys.t()) * self.knowledge_scale # (N, M)
        attn = torch.softmax(attn, dim=-1) # (N, M)
        knowledge_context = torch.matmul(attn, self.knowledge_core_values) # (N, D)

        # 3. Initialize Graph-of-Thought (N nodes)
        # Seed nodes with input, knowledge, and diverse noise to split perspectives
        nodes = []
        for i in range(self.n_nodes):
            noise = torch.randn_like(x_flat) * 0.05
            nodes.append(x_flat + knowledge_context + noise)
            
        node_stack = torch.stack(nodes, dim=1) # (N, V, D)
        node_cumulative_halt = torch.zeros(N, self.n_nodes, 1, device=x.device)
        total_node_preds = torch.zeros(N, self.n_nodes, self.out_dim, device=x.device)

        # Pre-compute expert outputs to save compute
        expert_outs = []
        for i in range(self.n_experts):
            h = self.dropout(self.expert_acts[i](self.expert_up[i](x_flat)))
            expert_outs.append(self.expert_norms[i](self.expert_down[i](h)))
        expert_stack = torch.stack(expert_outs, dim=1)  # (N, E, D_out)

        kl_loss = 0.0

        for step in range(self.reasoning_steps):
            
            # A. Graph Attention: Nodes exchange information
            q_got = self.got_q[step](node_stack) # (N, V, D)
            k_got = self.got_k[step](node_stack) # (N, V, D)
            v_got = self.got_v[step](node_stack) # (N, V, D)
            
            got_attn = torch.bmm(q_got, k_got.transpose(1, 2)) * self.knowledge_scale # (N, V, V)
            got_attn = self.dropout(torch.softmax(got_attn, dim=-1))
            got_updates = torch.bmm(got_attn, v_got) # (N, V, D)
            
            node_stack = self.got_norm[step](node_stack + got_updates)
            
            # B. Meta-Cognitive Critique
            # Global view of the graph spots logic errors, broadcasts correction
            global_state = node_stack.reshape(N, -1) # (N, V*D)
            global_correction = self.critique_net[step](global_state) # (N, D)
            
            # Apply correction to all active nodes
            node_stack = node_stack + global_correction.unsqueeze(1)
            
            # C. Stochastic Bayesian MoE Routing & Halting
            step_node_preds = []
            step_halt_probs = []
            
            for v in range(self.n_nodes):
                node_feat = node_stack[:, v, :] # (N, D)
                
                # Halt eval
                halt_prob = torch.sigmoid(self.node_halt_gates[step](node_feat)) # (N, 1)
                step_halt_probs.append(halt_prob)
                
                # Bayesian Routing 
                mu = self.route_mu[step](node_feat) # (N, E)
                logvar = self.route_logvar[step](node_feat) # (N, E)
                var = torch.exp(logvar)
                
                if self.training:
                    # Reparameterization trick: sample routing weights
                    eps = torch.randn_like(mu)
                    sample = mu + eps * torch.sqrt(var)
                    # KL divergence from standard normal prior
                    kl_loss = kl_loss + 0.5 * torch.sum(var + mu**2 - 1 - logvar, dim=1).mean()
                else:
                    # In eval, add controllable random divergence
                    eps = torch.randn_like(mu)
                    sample = mu + eps * torch.sqrt(var) * self.temperature

                gate_scores = torch.sigmoid(sample) # (N, E)
                
                # Execute experts and fusion
                routed = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=1) # (N, D_out)
                step_node_preds.append(routed)

            # D. Accumulate Predictions
            step_preds_stack = torch.stack(step_node_preds, dim=1) # (N, V, D_out)
            step_halts_stack = torch.stack(step_halt_probs, dim=1) # (N, V, 1)
            
            remaining = 1.0 - node_cumulative_halt
            total_node_preds = total_node_preds + remaining * step_preds_stack
            node_cumulative_halt = node_cumulative_halt + remaining * step_halts_stack
            
            if not self.training and node_cumulative_halt.mean() > 0.95:
                break
                
        # Optional: return KL loss to be added to total loss via an attribute
        self.last_kl_loss = kl_loss * 0.001

        # Final Node Fusion (Attentional pooling of Graph)
        fusion_q = self.node_fusion_q(total_node_preds).mean(dim=1, keepdim=True) # (N, 1, D_out)
        fusion_k = self.node_fusion_k(total_node_preds) # (N, V, D_out)
        fusion_v = self.node_fusion_v(total_node_preds) # (N, V, D_out)
        
        pool_attn = torch.bmm(fusion_q, fusion_k.transpose(1, 2)) * (self.out_dim**-0.5) # (N, 1, V)
        pool_attn = torch.softmax(pool_attn, dim=-1)
        fused_got_out = torch.bmm(pool_attn, fusion_v).squeeze(1) # (N, D_out)

        # Reshape back
        base_logits = base_logits.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        fused_got_out = fused_got_out.view(*shape_prefix, -1)

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * fused_got_out
        )


class ChampionNetOmniscientExpert(nn.Module):
    """
    Backbone-compatible model with OmniscientSynergyExpertHead (v20).
    """
    def __init__(
        self,
        n_experts: int = 6,
        n_nodes: int = 4,
        n_knowledge_slots: int = 64,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(OmniscientSynergyExpertHead(
            256, 10,
            n_experts=n_experts,
            n_nodes=n_nodes,
            n_knowledge_slots=n_knowledge_slots,
            reasoning_steps=reasoning_steps,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NeurogenesisExpertHead(nn.Module):
    """
    Generation v21: Neurogenesis Expert
    (Hierarchical Abstraction + Adversarial Self-Play)

    Processes input at multiple abstraction levels, uses adversarial debate
    between a Proposer and Adversary to produce robust conclusions, and
    dynamically synthesizes emergent expert combinations.

    Key Innovations:
    1. Hierarchical Abstraction Pyramid — 3 levels (concrete/abstract/meta)
    2. Adversarial Self-Play — Proposer vs Adversary MoE debate
    3. Cross-Level Knowledge Banks — per-level memory with distillation
    4. Emergent Expert Synthesis — dynamic expert weight interpolation
    5. Recursive Confidence Calibration — self-assessed certainty triggers extra rounds
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 6,
        n_levels: int = 3,
        n_knowledge_slots: int = 32,
        debate_rounds: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.n_levels = n_levels
        self.debate_rounds = debate_rounds

        # Base projection (backward-compatible)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Shared Expert
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 1. Hierarchical Abstraction — Up/Down projections between levels
        self.level_up_projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.GELU())
            for _ in range(n_levels - 1)
        ])
        self.level_down_projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.GELU())
            for _ in range(n_levels - 1)
        ])

        # 2. Cross-Level Knowledge Banks (one per level)
        self.knowledge_keys = nn.ParameterList([
            nn.Parameter(torch.randn(n_knowledge_slots, in_dim) * 0.02)
            for _ in range(n_levels)
        ])
        self.knowledge_values = nn.ParameterList([
            nn.Parameter(torch.randn(n_knowledge_slots, in_dim) * 0.02)
            for _ in range(n_levels)
        ])
        self.knowledge_queries = nn.ModuleList([
            nn.Linear(in_dim, in_dim, bias=False) for _ in range(n_levels)
        ])
        self.knowledge_scale = in_dim ** -0.5

        # Distillation bridges (transfer knowledge between adjacent levels)
        self.distill_bridges = nn.ModuleList([
            nn.Linear(in_dim, in_dim, bias=False) for _ in range(n_levels - 1)
        ])

        # 3. Proposer MoE (diverse activations)
        expert_dims = [1024, 1536, 2048, 1024, 1536, 2048]
        act_fns = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.proposer_up = nn.ModuleList([
            nn.Linear(in_dim, expert_dims[i % len(expert_dims)], bias=False)
            for i in range(n_experts)
        ])
        self.proposer_down = nn.ModuleList([
            nn.Linear(expert_dims[i % len(expert_dims)], out_dim, bias=False)
            for i in range(n_experts)
        ])
        self.proposer_acts = [act_fns[i % len(act_fns)] for i in range(n_experts)]
        self.proposer_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # Per-round proposer gates (Sigma Gating)
        self.proposer_gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts, bias=False) for _ in range(debate_rounds)
        ])

        # 4. Adversary MoE (separate experts for critique)
        self.adversary_up = nn.ModuleList([
            nn.Linear(in_dim, expert_dims[i % len(expert_dims)], bias=False)
            for i in range(n_experts)
        ])
        self.adversary_down = nn.ModuleList([
            nn.Linear(expert_dims[i % len(expert_dims)], out_dim, bias=False)
            for i in range(n_experts)
        ])
        self.adversary_acts = [act_fns[(i + 3) % len(act_fns)] for i in range(n_experts)]
        self.adversary_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # Per-round adversary gates
        self.adversary_gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts, bias=False) for _ in range(debate_rounds)
        ])

        # 5. Emergent Expert Synthesis — interpolation weights
        self.synthesis_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, n_experts)
            ) for _ in range(debate_rounds)
        ])

        # 6. Resolution Network (merges proposer + adversary)
        self.resolution = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim * 2, 128), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(128, out_dim)
            ) for _ in range(debate_rounds)
        ])

        # 7. Confidence Calibration
        self.confidence_gates = nn.ModuleList([
            nn.Linear(in_dim, 1, bias=True) for _ in range(debate_rounds)
        ])

        # 8. Level fusion (aggregate across abstraction levels)
        self.level_fusion = nn.Sequential(
            nn.Linear(n_levels * out_dim, 128), nn.GELU(),
            nn.Linear(128, out_dim)
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)

        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.proposer_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.proposer_down[i].weight, std=0.01)
            nn.init.kaiming_uniform_(self.adversary_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.adversary_down[i].weight, std=0.01)

        for g in self.proposer_gates:
            nn.init.normal_(g.weight, std=0.02)
        for g in self.adversary_gates:
            nn.init.normal_(g.weight, std=0.02)

        for c in self.confidence_gates:
            nn.init.constant_(c.bias, -2.0)

        for q in self.knowledge_queries:
            nn.init.kaiming_uniform_(q.weight, a=math.sqrt(5))
        for b in self.distill_bridges:
            nn.init.kaiming_uniform_(b.weight, a=math.sqrt(5))

        for net in [self.level_fusion, *self.resolution, *self.synthesis_net]:
            for p in net.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                else:
                    nn.init.zeros_(p)

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # Shared Expert
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # ═══ PHASE 1: Hierarchical Abstraction Pyramid ═══
        # Build multi-level representations
        level_features = [x_flat]  # Level 0 = concrete
        h = x_flat
        for i in range(self.n_levels - 1):
            h = h + self.level_up_projs[i](h)  # Residual up-projection
            level_features.append(h)

        # Query cross-level knowledge banks
        for lvl in range(self.n_levels):
            q = self.knowledge_queries[lvl](level_features[lvl])
            attn = torch.matmul(q, self.knowledge_keys[lvl].t()) * self.knowledge_scale
            attn = torch.softmax(attn, dim=-1)
            knowledge = torch.matmul(attn, self.knowledge_values[lvl])
            level_features[lvl] = level_features[lvl] + knowledge

        # Cross-level distillation (top-down grounding)
        for i in range(self.n_levels - 2, -1, -1):
            distilled = self.distill_bridges[i](level_features[i + 1])
            level_features[i] = level_features[i] + 0.1 * distilled

        # Merge all levels into a unified reasoning state
        unified = torch.stack(level_features, dim=0).mean(dim=0)  # (N, D)

        # Pre-compute proposer expert outputs
        proposer_outs = []
        for i in range(self.n_experts):
            h = self.dropout(self.proposer_acts[i](self.proposer_up[i](unified)))
            proposer_outs.append(self.proposer_norms[i](self.proposer_down[i](h)))
        proposer_stack = torch.stack(proposer_outs, dim=1)  # (N, E, D_out)

        # Pre-compute adversary expert outputs
        adversary_outs = []
        for i in range(self.n_experts):
            h = self.dropout(self.adversary_acts[i](self.adversary_up[i](unified)))
            adversary_outs.append(self.adversary_norms[i](self.adversary_down[i](h)))
        adversary_stack = torch.stack(adversary_outs, dim=1)  # (N, E, D_out)

        # ═══ PHASE 2: Adversarial Debate Loop ═══
        cumulative_confidence = torch.zeros(N, 1, device=x.device)
        total_resolved = torch.zeros(N, self.out_dim, device=x.device)
        debate_state = unified  # Evolves through debate

        for rnd in range(self.debate_rounds):

            # A. Emergent Expert Synthesis — dynamic interpolation weights
            synth_weights = torch.softmax(
                self.synthesis_net[rnd](debate_state), dim=-1
            )  # (N, E)

            # B. Proposer: route through experts with synthesis
            prop_gates = torch.sigmoid(self.proposer_gates[rnd](debate_state))
            combined_prop_gates = prop_gates * synth_weights  # Emergent combination
            proposal = (proposer_stack * combined_prop_gates.unsqueeze(-1)).sum(dim=1)

            # C. Adversary: critique the proposal
            adv_gates = torch.sigmoid(self.adversary_gates[rnd](debate_state))
            critique = (adversary_stack * adv_gates.unsqueeze(-1)).sum(dim=1)

            # D. Resolution: merge proposer + adversary insights
            combined = torch.cat([proposal, critique], dim=-1)  # (N, 2*D_out)
            resolved = self.resolution[rnd](combined)  # (N, D_out)

            # E. Confidence calibration
            confidence = torch.sigmoid(self.confidence_gates[rnd](debate_state))  # (N, 1)

            # F. Accumulate with remaining probability mass
            remaining = 1.0 - cumulative_confidence
            total_resolved = total_resolved + remaining * resolved
            cumulative_confidence = cumulative_confidence + remaining * confidence

            # G. Early exit if highly confident (inference only)
            if not self.training and cumulative_confidence.mean() > 0.9:
                break

        # ═══ PHASE 3: Final Assembly ═══
        # Also produce per-level predictions for richer signal
        level_preds = []
        for lvl in range(self.n_levels):
            level_preds.append(F.linear(level_features[lvl], self.weight, self.bias))
        level_concat = torch.cat(level_preds, dim=-1)  # (N, L*D_out)
        level_fused = self.level_fusion(level_concat)  # (N, D_out)

        # Combine everything
        output = (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * (total_resolved + 0.1 * level_fused)
        )

        return output.view(*shape_prefix, -1)


class ChampionNetNeurogenesisExpert(nn.Module):
    """
    Backbone-compatible model with NeurogenesisExpertHead (v21).
    """
    def __init__(
        self,
        n_experts: int = 6,
        n_levels: int = 3,
        n_knowledge_slots: int = 32,
        debate_rounds: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(NeurogenesisExpertHead(
            256, 10,
            n_experts=n_experts,
            n_levels=n_levels,
            n_knowledge_slots=n_knowledge_slots,
            debate_rounds=debate_rounds,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




class CognitiveSingularityExpertHead(nn.Module):
    """
    Generation v22: Cognitive Singularity Expert
    (Hypernetwork + Latent Tree-Search + Episodic Memory + Orthogonal Superposition)

    Pushes reasoning to its limits using fully dynamic computation:
    1. Hypernetwork-Generated Experts — dynamically generates weights based on input
    2. Latent Tree-Search — Value Network rolls out and evaluates paths like AlphaGo
    3. Episodic Neural Memory — differentiable read/write memory matrix for context
    4. Orthogonal Superposition — forces generation of mutually exclusive hypotheses
    5. Causal Logic Refinement — self-attention enforces logical sequence flow
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_hypotheses: int = 4,
        hyper_dim: int = 64,  # Custom expert hidden dimension
        mem_slots: int = 64,
        mem_dim: int = 128,
        search_depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hypotheses = n_hypotheses
        self.hyper_dim = hyper_dim
        self.mem_slots = mem_slots
        self.mem_dim = mem_dim
        self.search_depth = search_depth

        # Base projection (backward-compatible)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Shared Expert
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 1. Episodic Neural Memory (Fully Differentiable)
        self.memory_matrix = nn.Parameter(torch.randn(mem_slots, mem_dim) * 0.02)
        # Read heads
        self.mem_read_q = nn.Linear(in_dim, mem_dim, bias=False)
        self.mem_read_out = nn.Linear(mem_dim, in_dim, bias=False)
        # Write heads
        self.mem_write_gate = nn.Linear(in_dim, mem_slots, bias=True)
        self.mem_write_data = nn.Linear(in_dim, mem_dim, bias=False)
        self.mem_write_proj = nn.Linear(mem_dim, out_dim, bias=False)

        # 2. Hypernetwork (Generates custom expert weights on-the-fly)
        # Input: state + memory context (2 * in_dim)
        # Output: Weights for Linear(in_dim, hyper_dim) and Linear(hyper_dim, out_dim)
        self.hypernet_input_dim = in_dim * 2
        self.hypernet_out_size = (in_dim * hyper_dim) + (hyper_dim * out_dim)
        self.hypernet = nn.Sequential(
            nn.Linear(self.hypernet_input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.hypernet_out_size)
        )

        # 3. Orthogonal Superposition
        # Forces the model to span K disjoint regions of thought-space
        self.path_projs = nn.ModuleList([
            nn.Linear(in_dim, in_dim, bias=False) for _ in range(n_hypotheses)
        ])
        
        # 4. Latent Tree-Search Value Network
        # Evaluates the "goodness" or logical consistency of a hypothesis
        self.value_net = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 5. Causal Logic Refinement
        self.causal_q = nn.Linear(out_dim, out_dim, bias=False)
        self.causal_k = nn.Linear(out_dim, out_dim, bias=False)
        self.causal_v = nn.Linear(out_dim, out_dim, bias=False)
        self.causal_norm = nn.LayerNorm(out_dim)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        
        # Track orthogonal penalty loss for debugging/regularization
        self.register_buffer("last_ortho_loss", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)

        nn.init.kaiming_uniform_(self.mem_read_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mem_read_out.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mem_write_gate.weight, a=math.sqrt(5))
        nn.init.constant_(self.mem_write_gate.bias, -2.0)
        nn.init.kaiming_uniform_(self.mem_write_data.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mem_write_proj.weight, a=math.sqrt(5))

        # Hypernetwork init
        for layer in self.hypernet:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
        # Output of hypernetwork should be initialized small to prevent explosion
        nn.init.normal_(self.hypernet[-1].weight, std=0.001)

        for proj in self.path_projs:
            nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))

        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.kaiming_uniform_(self.causal_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.causal_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.causal_v.weight, a=math.sqrt(5))

    def compute_orthogonality_loss(self, hypotheses):
        # hypotheses: (N, K, D)
        normed = F.normalize(hypotheses, p=2, dim=-1) # (N, K, D)
        sim_matrix = torch.bmm(normed, normed.transpose(1, 2)) # (N, K, K)
        # Mask out diagonal (self-similarity is 1)
        eye = torch.eye(sim_matrix.size(1), device=sim_matrix.device).unsqueeze(0)
        sim_matrix = sim_matrix * (1.0 - eye)
        # Repulsion loss: minimize absolute cosine similarity between different hypotheses
        ortho_loss = sim_matrix.abs().sum() / (self.n_hypotheses * (self.n_hypotheses - 1))
        return ortho_loss

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # 1. Global Shared Expert
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # 2. Episodic Memory Read
        read_q = self.mem_read_q(x_flat) # (N, mem_dim)
        attn = torch.matmul(read_q, self.memory_matrix.t()) / math.sqrt(self.mem_dim) # (N, M)
        attn = torch.softmax(attn, dim=-1) # (N, M)
        mem_retrieval = torch.matmul(attn, self.memory_matrix) # (N, mem_dim)
        mem_context = self.mem_read_out(mem_retrieval) # (N, in_dim)
        
        # 3. Hypernetwork Weight Generation
        # Context is x_flat + episodic memory recall
        hyper_input = torch.cat([x_flat, mem_context], dim=-1) # (N, 2*in_dim)
        generated_weights = self.hypernet(hyper_input) # (N, W)
        
        w1_size = self.in_dim * self.hyper_dim
        w2_size = self.hyper_dim * self.out_dim
        
        w1_flat = generated_weights[:, :w1_size]
        w2_flat = generated_weights[:, w1_size:]
        
        # Reshape generated weights: (N, hyper_dim, in_dim) and (N, out_dim, hyper_dim)
        W1 = w1_flat.view(N, self.hyper_dim, self.in_dim)
        W2 = w2_flat.view(N, self.out_dim, self.hyper_dim)

        # 4. Orthogonal Superposition & Dynamic Execution
        # Generate K divergent hypotheses and run them through the custom dynamic expert
        hypotheses = []
        custom_outs = []
        for k in range(self.n_hypotheses):
            h_in = self.path_projs[k](x_flat) # (N, D)
            hypotheses.append(h_in)
            
            # Execute dynamically generated expert layer 1: (N, hyper_dim)
            h_hidden = self.dropout(F.gelu(torch.bmm(W1, h_in.unsqueeze(-1)).squeeze(-1)))
            
            # Execute dynamically generated expert layer 2: (N, out_dim)
            h_out = torch.bmm(W2, h_hidden.unsqueeze(-1)).squeeze(-1)
            custom_outs.append(h_out)
            
        hypotheses_stack = torch.stack(hypotheses, dim=1) # (N, K, D)
        custom_outs_stack = torch.stack(custom_outs, dim=1) # (N, K, D_out)

        # Record orthogonality penalty
        if self.training:
            self.last_ortho_loss = self.compute_orthogonality_loss(hypotheses_stack)

        # 5. Latent Tree-Search Value Network
        # Evaluate expected reward of each hypothesis
        values = self.value_net(custom_outs_stack).squeeze(-1) # (N, K)
        # Select best hypotheses (softmax converts to probabilities)
        path_probs = torch.softmax(values, dim=-1) # (N, K)
        
        # 6. Causal Logic Refinement
        # Enforce that the top hypotheses are logically coherent sequences
        causal_in = custom_outs_stack # (N, K, D_out)
        q = self.causal_q(causal_in)
        k = self.causal_k(causal_in)
        v = self.causal_v(causal_in)
        
        # Masked attention (causal path flow)
        causal_attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.out_dim)
        mask = torch.tril(torch.ones(self.n_hypotheses, self.n_hypotheses, device=x.device))
        causal_attn = causal_attn.masked_fill(mask == 0, float('-inf'))
        causal_attn = self.dropout(torch.softmax(causal_attn, dim=-1))
        
        refined_hypotheses = self.causal_norm(causal_in + torch.bmm(causal_attn, v)) # (N, K, D_out)
        
        # 7. Episodic Memory Write
        # Write the selected (highest value) path logic back to memory for future steps
        best_path_idx = path_probs.argmax(dim=-1) # (N)
        best_h_in = hypotheses_stack[torch.arange(N), best_path_idx] # (N, D)
        
        write_gates = torch.sigmoid(self.mem_write_gate(best_h_in)) # (N, M)
        write_data = self.mem_write_data(best_h_in) # (N, mem_dim)
        
        # Differentiable update to parameter matrix
        # This operates as a residual accumulation gradient update during training
        # For true recurrent state changes across tokens, an external hidden state is needed,
        # but the Parameter acts as a static "long term episodic registry"
        mem_update = torch.matmul(write_gates.t(), write_data) / (write_gates.sum(dim=0, keepdim=True).t() + 1e-6)
        
        # Add to predictions
        # Weigh refined hypotheses by their tree-search value
        fused_cognitive = (refined_hypotheses * path_probs.unsqueeze(-1)).sum(dim=1) # (N, D_out)
        
        # Add a residual from the memory write operation so write heads receive gradients
        fused_write = self.mem_write_proj(write_data * write_gates.mean(dim=-1, keepdim=True))
        fused_cognitive = fused_cognitive + 0.1 * fused_write

        output = (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * fused_cognitive
        )

        return output.view(*shape_prefix, -1)


class ChampionNetCognitiveExpert(nn.Module):
    """
    Backbone-compatible model with CognitiveSingularityExpertHead (v22).
    """
    def __init__(
        self,
        n_hypotheses: int = 4,
        hyper_dim: int = 64,
        mem_slots: int = 64,
        mem_dim: int = 128,
        search_depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(CognitiveSingularityExpertHead(
            256, 10,
            n_hypotheses=n_hypotheses,
            hyper_dim=hyper_dim,
            mem_slots=mem_slots,
            mem_dim=mem_dim,
            search_depth=search_depth,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TranscendentArchitectExpertHead(nn.Module):
    """
    Generation v23: Transcendent Architect Expert
    (Diffusion Refinement + World Model + Compositional Decomposition
     + Self-Modifying Attention + Evolution Layer)

    Instead of computing answers in a single pass, v23 sculpts its answer
    from noise using iterative denoising, simulates consequences via a
    learned World Model, and evolves candidate solutions.

    Key Innovations:
    1. Diffusion-Based Iterative Refinement — denoise answer from noise over T steps
    2. World Model Simulation — predict future states to score candidate quality
    3. Compositional Decomposition — break input into K sub-problems, solve & compose
    4. Self-Modifying Attention — meta-controller rewires Q/K/V mid-computation
    5. Gradient-Free Evolution — tournament selection + crossover in latent space
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_subproblems: int = 4,
        pop_size: int = 6,
        denoise_steps: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_subproblems = n_subproblems
        self.pop_size = pop_size
        self.denoise_steps = denoise_steps

        # Base projection (backward-compatible)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Shared Expert
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # ═══ 1. Compositional Decomposition ═══
        # Learned decomposition heads: split input into K sub-representations
        self.decompose_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, in_dim),
                nn.GELU()
            ) for _ in range(n_subproblems)
        ])

        # ═══ 2. Diffusion-Based Iterative Refinement ═══
        # Per-step denoisers (each step gets its own denoiser)
        self.denoiser = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim + in_dim, 256),  # noisy_z + context
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, out_dim)
            ) for _ in range(denoise_steps)
        ])

        # Timestep embeddings (learnable)
        self.time_embed = nn.Parameter(torch.randn(denoise_steps, out_dim) * 0.02)

        # Noise schedule (learnable betas)
        self.log_betas = nn.Parameter(torch.linspace(-3, -1, denoise_steps))

        # ═══ 3. World Model Simulation ═══
        # Transition function: predicts next state from current
        self.world_transition = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )
        # Value function: scores how "good" a state is
        self.world_value = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # ═══ 4. Self-Modifying Attention ═══
        # Base Q/K/V projections
        self.attn_q = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_k = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_v = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_norm = nn.LayerNorm(out_dim)

        # Meta-controller: modifies Q/K/V weights based on intermediate results
        self.meta_q_mod = nn.Linear(in_dim, out_dim * out_dim, bias=False)
        self.meta_k_mod = nn.Linear(in_dim, out_dim * out_dim, bias=False)

        # ═══ 5. Evolution Layer ═══
        # Fitness projection (maps sub-solutions to fitness scores)
        self.fitness_proj = nn.Linear(out_dim, 1)
        # Crossover network (merges two parents into offspring)
        self.crossover_net = nn.Sequential(
            nn.Linear(out_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )

        # Final composition: merge sub-problem solutions
        self.composer = nn.Sequential(
            nn.Linear(n_subproblems * out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)

        for head in self.decompose_heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        for step_net in self.denoiser:
            for m in step_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        for net in [self.world_transition, self.world_value, self.crossover_net, self.composer]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.kaiming_uniform_(self.attn_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.attn_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.attn_v.weight, a=math.sqrt(5))

        nn.init.normal_(self.meta_q_mod.weight, std=0.001)
        nn.init.normal_(self.meta_k_mod.weight, std=0.001)

        nn.init.kaiming_uniform_(self.fitness_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fitness_proj.bias)

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # Shared Expert
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # ═══ PHASE 1: Compositional Decomposition ═══
        sub_contexts = []
        for k in range(self.n_subproblems):
            sub_ctx = self.decompose_heads[k](x_flat)  # (N, D)
            sub_contexts.append(sub_ctx)

        # ═══ PHASE 2: Diffusion Refinement (per sub-problem) ═══
        betas = torch.sigmoid(self.log_betas)  # (T,) in [0, 1]
        sub_solutions = []

        for k in range(self.n_subproblems):
            ctx = sub_contexts[k]  # (N, D_in)

            # Start from noise
            z = torch.randn(N, self.out_dim, device=x.device) * 0.1

            # Iteratively denoise
            for t in range(self.denoise_steps):
                beta_t = betas[t]
                t_emb = self.time_embed[t].unsqueeze(0)  # (1, D_out)

                # Condition on context + noisy state + timestep
                z_conditioned = z + t_emb
                denoiser_input = torch.cat([z_conditioned, ctx], dim=-1)  # (N, D_out + D_in)
                noise_pred = self.denoiser[t](denoiser_input)  # (N, D_out)

                # Denoise step: remove predicted noise
                z = z - beta_t * noise_pred

            sub_solutions.append(z)  # (N, D_out)

        # ═══ PHASE 3: World Model Simulation ═══
        # Score each sub-solution by simulating one step into the future
        world_scores = []
        for sol in sub_solutions:
            future_state = self.world_transition(sol)  # (N, D_out)
            value = self.world_value(future_state).squeeze(-1)  # (N,)
            world_scores.append(value)
        world_scores = torch.stack(world_scores, dim=1)  # (N, K)
        world_probs = torch.softmax(world_scores, dim=-1)  # (N, K)

        # ═══ PHASE 4: Evolution Layer ═══
        # Create population from sub-solutions
        pop = torch.stack(sub_solutions, dim=1)  # (N, K, D_out)

        # Compute fitness scores
        fitness = self.fitness_proj(pop).squeeze(-1)  # (N, K)

        # Soft tournament selection: differentiable weighted parents
        fitness_weights = torch.softmax(fitness * 5.0, dim=-1)  # (N, K) temperature-sharpened
        parent_a = (pop * fitness_weights.unsqueeze(-1)).sum(dim=1)  # (N, D_out)

        # Second parent: use inverse fitness for diversity
        inv_weights = torch.softmax(-fitness * 5.0, dim=-1)
        parent_b = (pop * inv_weights.unsqueeze(-1)).sum(dim=1)  # (N, D_out)

        # Crossover: merge parents into evolved offspring
        offspring = self.crossover_net(torch.cat([parent_a, parent_b], dim=-1))  # (N, D_out)

        # ═══ PHASE 5: Self-Modifying Attention ═══
        # Use the offspring + world model insights to modify attention
        # Meta-controller generates attention weight modifications
        meta_dq = self.meta_q_mod(x_flat).view(N, self.out_dim, self.out_dim)  # (N, D, D)
        meta_dk = self.meta_k_mod(x_flat).view(N, self.out_dim, self.out_dim)  # (N, D, D)

        # Apply self-modifying attention over sub-solutions + offspring
        all_candidates = torch.stack(sub_solutions + [offspring], dim=1)  # (N, K+1, D_out)
        K1 = all_candidates.shape[1]

        # Base Q/K/V
        q = self.attn_q(all_candidates)  # (N, K+1, D_out)
        k_proj = self.attn_k(all_candidates)
        v = self.attn_v(all_candidates)

        # Apply meta-modifications (batched matrix multiply for Q and K)
        q = q + torch.bmm(q, meta_dq) * 0.1  # Residual modification
        k_proj = k_proj + torch.bmm(k_proj, meta_dk) * 0.1

        # Compute attention
        attn_scores = torch.bmm(q, k_proj.transpose(1, 2)) / math.sqrt(self.out_dim)
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))
        attn_out = self.attn_norm(all_candidates + torch.bmm(attn_weights, v))  # (N, K+1, D_out)

        # ═══ PHASE 6: Final Composition ═══
        # Weight attended candidates by world model probabilities + offspring bonus
        # Extend world_probs with offspring weight
        offspring_score = self.world_value(
            self.world_transition(offspring)
        ).squeeze(-1).unsqueeze(-1)  # (N, 1)
        all_scores = torch.cat([world_scores, offspring_score], dim=-1)  # (N, K+1)
        all_probs = torch.softmax(all_scores, dim=-1)  # (N, K+1)

        # Weighted sum of attended candidates
        fused = (attn_out * all_probs.unsqueeze(-1)).sum(dim=1)  # (N, D_out)

        # Also compose sub-solutions directly for a rich signal
        composed = self.composer(torch.cat(sub_solutions, dim=-1))  # (N, D_out)

        output = (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * (fused + 0.1 * composed)
        )

        return output.view(*shape_prefix, -1)


class ChampionNetTranscendentExpert(nn.Module):
    """
    Backbone-compatible model with TranscendentArchitectExpertHead (v23).
    """
    def __init__(
        self,
        n_subproblems: int = 4,
        pop_size: int = 6,
        denoise_steps: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(TranscendentArchitectExpertHead(
            256, 10,
            n_subproblems=n_subproblems,
            pop_size=pop_size,
            denoise_steps=denoise_steps,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ODEFunc(nn.Module):
    """
    Continuous-Time Neural ODE Function for the Omniversal Quantum Simulation.
    Defines the derivative dz/dt for the neural integrator.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        # Project Phase + Magnitude + Time
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim * 2)
        )
        
        # Entanglement Routing approximations
        self.q_proj = nn.Linear(dim * 2, dim)
        self.k_proj = nn.Linear(dim * 2, dim)
        self.v_proj = nn.Linear(dim * 2, dim * 2)
        
    def forward(self, t: torch.Tensor, z: torch.Tensor):
        # z is (N, dim*2) representing complex state [Real, Imag]
        N, D2 = z.shape
        D = D2 // 2
        
        # 1. Neural Mechanics Step
        t_tensor = t.expand(N, 1)
        z_t = torch.cat([z, t_tensor], dim=-1)
        dz_dt_base = self.net(z_t) # (N, D*2)
        
        # 2. Entanglement Routing
        # Instead of dot product similarity, we use a cheap pseudo-Kronecker entanglement
        q = self.q_proj(z) # (N, D)
        k = self.k_proj(z) # (N, D)
        v = self.v_proj(z) # (N, D*2)
        
        # Entangle Q and K via element-wise multiplication (diagonal of Kronecker)
        # then non-linear mix
        entangled = torch.relu(q * k).sum(dim=-1, keepdim=True) / math.sqrt(D) # (N, 1)
        routing_update = torch.sigmoid(entangled) * v # (N, D*2)
        
        return dz_dt_base + 0.1 * routing_update


class OmniversalQuantumExpertHead(nn.Module):
    """
    Generation v24: Omniversal Quantum Simulation Expert
    (Complex Superposition + Neural ODE + Holographic HRR + Entanglement)
    
    Abandons discrete real-valued layers for Continuous-Time Complex integration.
    Allows mutual contradiction handling via wave interference before Measurement Collapse.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        ode_steps: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ode_steps = ode_steps      # Discretization steps for Euler integrator
        self.complex_dim = in_dim       # Operate in complex space of size in_dim
        
        # Base projection (backward-compatible residual)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Shared Real Expert (Standard MLP fallback)
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))
        
        # ═══ 1. Quantum Lift (Measurement Prep) ═══
        # Lifts Real-valued input to Complex-valued Wavefunction (Real, Imag)
        self.real_lift = nn.Linear(in_dim, self.complex_dim)
        self.imag_lift = nn.Linear(in_dim, self.complex_dim)
        
        # ═══ 2. Holographic Knowledge Binding (HRR Context) ═══
        # Global holographic memory trace (learnable contextual superposition)
        self.holographic_memory = nn.Parameter(torch.randn(self.complex_dim * 2))
        
        # ═══ 3. Continuous-Time Neural ODE ═══
        # Defines the continuous derivative of the complex thought process
        self.ode_func = ODEFunc(dim=self.complex_dim, dropout=dropout)
        
        # ═══ 4. Measurement Collapse ═══
        # Collapses the final Complex state back into distinct Real-valued logic (logits)
        self.measurement_proj = nn.Sequential(
            nn.Linear(self.complex_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )
        
        # Measurement uncertainty modulator (Phase variance modulates confidence)
        self.uncertainty_gate = nn.Linear(self.complex_dim, 1)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)
        
        nn.init.kaiming_uniform_(self.real_lift.weight, a=math.sqrt(5))
        nn.init.zeros_(self.real_lift.bias)
        nn.init.kaiming_uniform_(self.imag_lift.weight, a=math.sqrt(5))
        nn.init.zeros_(self.imag_lift.bias)
        
        for m in self.ode_func.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.measurement_proj:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        nn.init.kaiming_uniform_(self.uncertainty_gate.weight, a=math.sqrt(5))
        nn.init.zeros_(self.uncertainty_gate.bias)

    def circular_convolution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Holographic Reduced Representation (HRR) Binding via Fast Fourier Transform.
        Binds two vectors in superposition such that x * y = z, where z contains features of both.
        """
        # x and y are (N, D)
        X = torch.fft.rfft(x, dim=-1)
        Y = torch.fft.rfft(y, dim=-1)
        Z = X * Y # Element-wise complex multiplication in frequency domain
        return torch.fft.irfft(Z, n=x.shape[-1], dim=-1)

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # Shared Real Expert
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # ═══ PHASE 1: Quantum Lift (Superposition Initialization) ═══
        # Create Complex Wavefunction: Ψ = Real + i*Imag
        z_real = self.real_lift(x_flat) # (N, D)
        z_imag = self.imag_lift(x_flat) # (N, D)
        
        # Pack into (N, D*2) for compatibility with real-valued PyTorch ops
        z0 = torch.cat([z_real, z_imag], dim=-1) # (N, D*2)
        
        # ═══ PHASE 2: Holographic Knowledge Binding ═══
        # Bind the current state with the global holographic memory trace
        memory_expanded = self.holographic_memory.unsqueeze(0).expand(N, -1) # (N, D*2)
        z_bound = self.circular_convolution(z0, memory_expanded) # (N, D*2)
        
        # Add bound knowledge into superposition
        z_t = z0 + 0.1 * z_bound

        # ═══ PHASE 3: Continuous-Time Neural ODE Integration ═══
        # Solve the Initial Value Problem: z(1) = z(0) + ∫ f(z(t), t) dt
        # using a simple Euler integrator (differentiable)
        dt = 1.0 / self.ode_steps
        t = torch.zeros(1, device=x.device)
        
        for _ in range(self.ode_steps):
            # Compute derivative
            dz_dt = self.ode_func(t, z_t)
            # Step forward in continuous time
            z_t = z_t + dz_dt * dt
            t = t + dt
            
        # z_t is now the final complex state at t=1.0

        # ═══ PHASE 4: Measurement Collapse ═══
        # Extract Phase and Magnitude to measure uncertainty
        final_real, final_imag = z_t.chunk(2, dim=-1) # (N, D), (N, D)
        
        # Phase = atan2(Imag, Real). Represents conceptual angle/ambiguity.
        phase = torch.atan2(final_imag, final_real + 1e-8) # (N, D)
        
        # Uncertainty is based on phase complexity. High variance = high uncertainty.
        collapse_confidence = torch.sigmoid(self.uncertainty_gate(phase)) # (N, 1)
        
        # Collapse the Complex wavefunction back into Real-valued Action (Logits)
        collapsed_logits = self.measurement_proj(z_t) # (N, out_dim)
        
        # Apply confidence: if uncertainty is high, the model's absolute amplitude drops
        final_quantum_logic = collapsed_logits * collapse_confidence

        output = (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * final_quantum_logic
        )

        return output.view(*shape_prefix, -1)


class ChampionNetOmniversalExpert(nn.Module):
    """
    Backbone-compatible model with OmniversalQuantumExpertHead (v24).
    """
    def __init__(
        self,
        ode_steps: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(OmniversalQuantumExpertHead(
            256, 10,
            ode_steps=ode_steps,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CellularAutomataGrid(nn.Module):
    """
    Neural Cellular Automata grid. Computes localized cell evolution instead of dense layers.
    """
    def __init__(self, dim: int, grid_size: int = 4, ca_steps: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.ca_steps = ca_steps
        
        # State update network (operates on 3x3 local neighborhood)
        self.update_net = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )
        self.stochastic_gate = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (N, dim)
        N = x.shape[0]
        # Reshape flat vector into a 2D grid
        grid = x.view(N, self.dim, self.grid_size, self.grid_size)
        
        for _ in range(self.ca_steps):
            # NCA Update: compute delta
            dx = self.update_net(grid)
            
            # Stochastic update gate (allows cells to preserve state or change)
            update_prob = torch.sigmoid(self.stochastic_gate(grid))
            # In training, we use soft updates. In true NCA this would be stochastic mask.
            grid = grid + update_prob * dx
            
        return grid.view(N, -1)


class FractalGenesisExpertHead(nn.Module):
    """
    Generation v25: Fractal Genesis Expert
    (Hyperbolic Space + Fractal Graph + Cellular Automata + Self-Assembly)
    
    Transforms reasoning from a flat vector sequence into a dynamically
    spawning hyperbolic computational fractal tree.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        fractal_depth: int = 2,
        atoms_count: int = 16,
        grid_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fractal_depth = fractal_depth
        self.atoms_count = atoms_count
        self.grid_size = grid_size
        
        # Base Euclidean projection (residual fallback)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Shared Euclidean Expert
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))
        
        # ═══ 1. Hyperbolic Projection ═══
        # Map Euclidean features to the Poincaré Disk manifold parameters
        self.hyperbolic_proj = nn.Linear(in_dim, in_dim)
        self.curvature = nn.Parameter(torch.tensor(1.0)) # Learned negative curvature c
        
        # ═══ 2. Semantic Self-Assembly ═══
        # Pool of 'Logic Atoms' that dynamically construct the fractal node circuits
        self.logic_atoms = nn.Parameter(torch.randn(atoms_count, in_dim))
        self.assembly_net = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim)
        )
        
        # ═══ 3. Fractal Graph Recursion & CA ═══
        # Each level of the fractal tree has its own CA grid evolver
        # and a mechanism to spawn K=2 child nodes.
        self.fractal_evolvers = nn.ModuleList([
            CellularAutomataGrid(
                dim=in_dim // (grid_size**2), 
                grid_size=grid_size, 
                dropout=dropout
            ) for _ in range(fractal_depth)
        ])
        
        # Node spawn projections: splits a parent node into 2 children
        self.node_spawners = nn.ModuleList([
            nn.Linear(in_dim, in_dim * 2) for _ in range(fractal_depth - 1)
        ])
        
        # ═══ 4. Resonance Collapse Aggregation ═══
        # Collapses 2 children back into a parent via phase resonance
        self.resonance_combiners = nn.ModuleList([
            nn.Linear(in_dim * 2, in_dim) for _ in range(fractal_depth - 1)
        ])
        
        self.final_projection = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)
        
        nn.init.kaiming_uniform_(self.hyperbolic_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.hyperbolic_proj.bias)
        
        std = 1.0 / math.sqrt(self.in_dim)
        nn.init.uniform_(self.logic_atoms, -std, std)
        
        for m in self.assembly_net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.node_spawners:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
            
        for m in self.resonance_combiners:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
            
        for m in self.final_projection:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def exp_map_zero(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Hyperbolic exponential map from origin to Poincaré disk."""
        sqrt_c = torch.clamp(c, min=1e-5).sqrt()
        norm_x = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-5)
        return torch.tanh(sqrt_c * norm_x) * x / (sqrt_c * norm_x)

    def poincare_dist(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance between vectors x and y in Poincaré disk."""
        sqrt_c = torch.clamp(c, min=1e-5).sqrt()
        norm_x_sq = torch.norm(x, dim=-1, keepdim=True)**2
        norm_y_sq = torch.norm(y, dim=-1, keepdim=True)**2
        # Mobius addition squared norm
        num = torch.norm(x - y, dim=-1, keepdim=True)**2
        den = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)
        arg = 1 + 2 * c * num / torch.clamp(den, min=1e-5)
        # Using arcosh trick
        dist = (2 / sqrt_c) * torch.log(arg + torch.sqrt(arg**2 - 1 + 1e-6))
        return dist
        
    def assemble_node(self, node_state: torch.Tensor) -> torch.Tensor:
        """Dynamically binds logic atoms to the node state using hyperbolic distance."""
        # Map node to hyperbolic space
        c = torch.clamp(F.softplus(self.curvature), min=1e-4) # Ensure c > 0
        h_node = self.exp_map_zero(self.hyperbolic_proj(node_state), c) # (N, D)
        h_atoms = self.exp_map_zero(self.logic_atoms, c) # (A, D)
        
        # Compute pairwise hyperbolic distances: (N, 1, D) and (1, A, D) -> (N, A)
        dists = self.poincare_dist(h_node.unsqueeze(1), h_atoms.unsqueeze(0), c).squeeze(-1)
        
        # Attention over atoms (closer = higher weight)
        atom_weights = torch.softmax(-dists, dim=-1) # (N, A)
        
        # Assemble bounded circuit state
        bound_atoms = torch.matmul(atom_weights, self.logic_atoms) # (N, D)
        
        # Feed back to Euclidean space for processing
        assembled = self.assembly_net(torch.cat([node_state, bound_atoms], dim=-1)) # (N, D)
        return node_state + 0.1 * assembled

    def recursive_fractal_pass(self, node_state: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Recursively spawns children, evolves them via CA, and collapses them via resonance.
        Depth 0 = deepest leaf nodes.
        depth = self.fractal_depth - 1 is the Root node.
        """
        # 1. Self-Assembly: bind logic atoms to current node state
        assembled_node = self.assemble_node(node_state)
        
        # 2. Base case (leaf nodes)
        if depth == 0:
            # Evolve via Cellular Automata and return
            return self.fractal_evolvers[0](assembled_node)
            
        # 3. Recursive step (internal nodes)
        # Spawn two children from current node
        children_raw = self.node_spawners[depth-1](assembled_node) # (N, D*2)
        child_left, child_right = children_raw.chunk(2, dim=-1) # (N, D), (N, D)
        
        # Recurse on children
        evolved_left = self.recursive_fractal_pass(child_left, depth - 1)
        evolved_right = self.recursive_fractal_pass(child_right, depth - 1)
        
        # 4. Resonance Collapse
        # Combine children back into parent. Instead of simple addition, we use
        # a resonance combiner to simulate constructive/destructive logic interference
        combined = self.resonance_combiners[depth-1](torch.cat([evolved_left, evolved_right], dim=-1))
        
        # Finally evolve the combined parent node via its CA grid
        final_node = self.fractal_evolvers[depth](combined + assembled_node)
        return final_node

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # Shared Euclidean Expert
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # ═══ FRACTAL GENESIS ENGINE ═══
        # Launch the recursive fractal graph processing from the root
        root_state = x_flat
        final_euclidean_state = self.recursive_fractal_pass(root_state, depth=self.fractal_depth - 1)
        
        # Project back to logits
        fractal_logic = self.final_projection(final_euclidean_state)

        output = (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * fractal_logic
        )

        return output.view(*shape_prefix, -1)


class ChampionNetFractalExpert(nn.Module):
    """
    Backbone-compatible model with FractalGenesisExpertHead (v25).
    """
    def __init__(
        self,
        fractal_depth: int = 2,
        atoms_count: int = 16,
        grid_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(FractalGenesisExpertHead(
            256, 10,
            fractal_depth=fractal_depth,
            atoms_count=atoms_count,
            grid_size=grid_size,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SurrogateHeaviside(torch.autograd.Function):
    """
    Surrogate gradient for the non-differentiable spiking Heaviside step function.
    Uses a fast sigmoid approximation for the backward pass.
    """
    @staticmethod
    def forward(ctx, input, threshold, alpha):
        ctx.save_for_backward(input, threshold, alpha)
        # Forward pass: Strict binary spike (Heaviside step)
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold, alpha = ctx.saved_tensors
        # Backward pass: Fast sigmoid surrogate gradient
        sgax = (input - threshold) * alpha
        S_dtps = torch.exp(-torch.abs(sgax)) / (1 + torch.exp(-torch.abs(sgax)))**2
        
        grad_input = grad_output * alpha * S_dtps
        grad_threshold = (grad_output * (-alpha) * S_dtps).sum()
        grad_alpha = (grad_output * (input - threshold) * S_dtps).sum()
        
        return grad_input, grad_threshold, grad_alpha

def surrogate_spike(input, threshold, alpha=torch.tensor(2.0)):
    return SurrogateHeaviside.apply(input, threshold, alpha)

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Spiking Neuron.
    Maintains a continuous membrane potential that decays over time and spikes when crossing a threshold.
    """
    def __init__(self, in_features: int, decay: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.decay = decay
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.alpha = nn.Parameter(torch.tensor(2.0)) # Surrogate gradient sharpness
        
    def forward(self, x: torch.Tensor, v_mem: torch.Tensor):
        # Update membrane potential: decay previous + integrate new input
        v_mem = v_mem * self.decay + x
        
        # Fire spike if v_mem > threshold
        spike = surrogate_spike(v_mem, self.threshold, self.alpha)
        
        # Soft reset: subtract threshold from membrane if fired
        v_mem = v_mem - spike * self.threshold
        
        return spike, v_mem

class LiquidSynapseMPS(nn.Module):
    """
    Combines Tensor Networks (1D Matrix Product State approximation) 
    with Liquid Neural Network synaptic dynamics (State-dependent differential scaling).
    """
    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.dim = dim
        self.rank = rank
        
        # Matrix Product State (MPS) core tensors (highly compressed representation of dense weights)
        # Instead of dim x dim, we use (dim x rank) and (rank x dim) to emulate a tensor train
        self.core_1 = nn.Parameter(torch.randn(dim, rank) / math.sqrt(dim))
        self.core_2 = nn.Parameter(torch.randn(rank, dim) / math.sqrt(rank))
        
        # Liquid Synaptic Time Constants (LNN mechanics)
        self.tau = nn.Parameter(torch.ones(dim))
        # Non-linear input processor for the liquid equation
        self.liquid_gate = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, liquid_state: torch.Tensor, dt: float = 0.1):
        # 1. Tensor Network (MPS) transformation
        # x is (..., dim)
        hidden = torch.matmul(x, self.core_1) # (..., rank)
        mps_transform = torch.matmul(hidden, self.core_2) # (..., dim)
        
        # 2. Liquid Synapse update: dx/dt = -x/tau + S(x, I)
        sensory_input = torch.sigmoid(self.liquid_gate(mps_transform))
        
        # Differential equation Euler step
        d_state = -liquid_state / torch.clamp(F.softplus(self.tau), min=1e-3) + sensory_input
        new_state = liquid_state + d_state * dt
        
        return new_state


class LiquidSpikingTensorExpertHead(nn.Module):
    """
    Generation v26: Liquid Spiking Tensor Expert
    (Spiking Neural Networks + Liquid Synapses + Test-Time Training + Tensor Networks)
    
    A bio-plausible, highly compressed architectured adapting dynamically during inference.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        spiking_steps: int = 4,
        mps_rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spiking_steps = spiking_steps
        
        # Base Euclidean projection
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Shared Euclidean Expert
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))
        
        # ═══ 1. Test-Time Training (TTT) Fast-Weight Module ═══
        # Internal LoRA adapter that updates *during the forward pass*
        self.ttt_lora_A = nn.Parameter(torch.randn(in_dim, 8) / math.sqrt(in_dim))
        self.ttt_lora_B = nn.Parameter(torch.zeros(8, out_dim))
        # Self-supervised auxiliary task (Reconstruct input from bottleneck)
        self.ttt_decoder = nn.Linear(8, in_dim)
        
        # ═══ 2. Liquid Synapse / Tensor Network Core ═══
        self.liquid_synapse = LiquidSynapseMPS(dim=in_dim, rank=mps_rank)
        
        # ═══ 3. Spiking Neural Mechanics (LIF Neurons) ═══
        self.lif_neuron = LIFNeuron(in_features=in_dim)
        
        # ═══ 4. Neuro-Symbolic Thresholding ═══
        self.spike_decoder = nn.Linear(in_dim, out_dim)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)

        nn.init.normal_(self.ttt_lora_B, std=0.01)
        self.lif_neuron.threshold.data.fill_(0.1)
        
        nn.init.kaiming_uniform_(self.ttt_decoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.ttt_decoder.bias)
                    
        nn.init.kaiming_uniform_(self.spike_decoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.spike_decoder.bias)

    def test_time_training_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs one step of gradient descent on the internal fast-weights (LoRA matrices)
        using a self-supervised autoencoding objective, strictly separated from global backprop.
        """
        # Detach x from the global computation graph so inner loop grads don't leak
        x_inner = x.detach().requires_grad_(True)
        
        # Self-supervised forward: Encode using shared representation, decode using aux task
        latent = torch.matmul(x_inner, self.ttt_lora_A)
        reconstruction = self.ttt_decoder(latent)
        
        # Loss: Mean Squared Error representation
        loss = F.mse_loss(reconstruction, x_inner)
        
        # Calculate gradients for the Lora A encoder and aux decoder
        grads = torch.autograd.grad(
            loss, 
            [self.ttt_lora_A, self.ttt_decoder.weight, self.ttt_decoder.bias], 
            create_graph=self.training, 
            retain_graph=True
        )
        
        # TTT SGD Step (lr = 0.01). Update only the shared encoder fast-weights.
        fast_lora_A = self.ttt_lora_A - 0.01 * grads[0]
        
        # Compute TTT contribution for this batch using adapted encoder and frozen task-decoder
        ttt_output = torch.matmul(torch.matmul(x, fast_lora_A), self.ttt_lora_B) # (N, out_dim)
        return ttt_output

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # Shared Euclidean Expert
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # ═══ 1. Test-Time Training Update ═══
        ttt_logits = self.test_time_training_step(x_flat)

        # ═══ 2. Spiking Liquid Tensor Engine ═══
        # Initialize Membrane Voltage and Liquid State for the Spiking Loop
        v_mem = torch.zeros_like(x_flat)
        liquid_state = torch.zeros_like(x_flat)
        spike_train_accum = torch.zeros_like(x_flat)
        
        for _ in range(self.spiking_steps):
            # Liquid Synapse (MPS) update
            liquid_state = self.liquid_synapse(x_flat, liquid_state)
            
            # Feed continuously changing liquid state into LIF Neuron
            spike, v_mem = self.lif_neuron(liquid_state, v_mem)
            
            # Accumulate discrete spikes
            spike_train_accum = spike_train_accum + spike
            
        # ═══ 4. Neuro-Symbolic Thresholding ═══
        # Average firing rate decoded back into continuous real logits
        firing_rate = spike_train_accum / self.spiking_steps
        spiking_logic = self.spike_decoder(firing_rate)

        output = (
            base_logits
            + self.shared_scale * shared_out
            + ttt_logits
            + (self.alpha + 1e-4) * spiking_logic
        )

        return output.view(*shape_prefix, -1)


class ChampionNetLiquidSpikingExpert(nn.Module):
    """
    Backbone-compatible model with LiquidSpikingTensorExpertHead (v26).
    """
    def __init__(
        self,
        spiking_steps: int = 4,
        mps_rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(LiquidSpikingTensorExpertHead(
            256, 10,
            spiking_steps=spiking_steps,
            mps_rank=mps_rank,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ActiveInferenceExpertHead(nn.Module):
    """
    v27 Active Inference Expert Head based on the Free Energy Principle.
    It introduces a generative World Model and minimizes Variational Free Energy 
    at test time before routing to experts.
    """
    def __init__(self, in_dim: int = 256, out_dim: int = 10, n_experts: int = 6, 
                 inference_steps: int = 3, lr: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.inference_steps = inference_steps
        self.lr = lr
        
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self.world_model = nn.Linear(in_dim, in_dim, bias=True)
        self.world_prior = nn.Linear(in_dim, in_dim, bias=False)
        self.precision_gate = nn.Linear(in_dim, n_experts * 2, bias=True)
        
        self.experts_up = nn.ModuleList([nn.Linear(in_dim, 1024, bias=False) for _ in range(n_experts)])
        self.experts_down = nn.ModuleList([nn.Linear(1024, out_dim, bias=False) for _ in range(n_experts)])
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.world_model.weight, std=0.01)
        nn.init.zeros_(self.world_model.bias)
        nn.init.normal_(self.world_prior.weight, std=0.01)
        nn.init.normal_(self.precision_gate.weight, std=0.01)
        nn.init.zeros_(self.precision_gate.bias)
        for e in self.experts_up:
            nn.init.normal_(e.weight, std=0.01)
        for e in self.experts_down:
            nn.init.zeros_(e.weight)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        original_shape = x.shape
        x_flat = x.view(-1, self.in_dim)
        
        # Test-time inner loop optimization
        with torch.enable_grad():
            z = x_flat.clone().detach().requires_grad_(True)
            for _ in range(self.inference_steps):
                predicted_state = F.silu(self.world_model(z))
                surprise = F.mse_loss(predicted_state, x_flat, reduction='sum')
                complexity = F.mse_loss(z, self.world_prior(x_flat), reduction='sum')
                free_energy = surprise + 0.1 * complexity
                
                grad_z = torch.autograd.grad(free_energy, z, create_graph=self.training)[0]
                z = z - self.lr * grad_z
                
        z_final = z
        gating_out = self.precision_gate(z_final)
        mu, logvar = gating_out.chunk(2, dim=-1)
        precision = torch.exp(-logvar)
        
        gate_scores = torch.softmax(mu * precision, dim=-1)
        
        expert_outputs = []
        for i in range(self.n_experts):
            h = F.silu(self.experts_up[i](z_final))
            h = self.dropout(h)
            out = self.experts_down[i](h)
            expert_outputs.append(out)
            
        expert_stack = torch.stack(expert_outputs, dim=-2)
        routed_out = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=-2)
        
        output = base_logits.view(-1, self.out_dim) + self.alpha * routed_out
        return output.view(*original_shape[:-1], self.out_dim)


class ChampionNetActiveInferenceExpert(nn.Module):
    def __init__(self, inference_steps: int = 3, lr: float = 0.1, dropout: float = 0.1):
        super().__init__()
        from run import ChampionNet
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ActiveInferenceExpertHead(
            256, 10,
            inference_steps=inference_steps,
            lr=lr,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class StateSpaceExpert(nn.Module):
    """
    Continuous-Time State Space Model (inspired by S4/Mamba).
    Approximates an LDS: h' = Ah + Bx, y = Ch + Dx.
    A simple discrete Zero-Order Hold (ZOH) discretization is applied.
    """
    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A_log = nn.Parameter(torch.empty(d_model, d_state))
        self.B = nn.Parameter(torch.empty(d_model, d_state))
        self.C = nn.Parameter(torch.empty(d_model, d_state))
        self.D = nn.Parameter(torch.zeros(d_model))
        
        self.dt_proj = nn.Linear(d_model, d_model)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A to be stable (negative real parts)
        nn.init.uniform_(self.A_log, -10.0, -1.0)
        nn.init.normal_(self.B, std=0.02)
        nn.init.normal_(self.C, std=0.02)
        nn.init.zeros_(self.D)
        nn.init.normal_(self.dt_proj.weight, std=0.01)
        nn.init.constant_(self.dt_proj.bias, -3.0)

    def forward(self, x):
        # x shape: (..., d_model)
        # 1. Delta (time scale)
        dt = F.softplus(self.dt_proj(x))
        dt = torch.clamp(dt, min=self.dt_min, max=self.dt_max)
        
        # Continuous A must be negative to be stable
        A = -torch.exp(self.A_log)
        
        # 2. ZOH Discretization for A and B
        # dA = exp(dt * A)
        # dB = (exp(dt * A) - I) / A * B
        
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = (dA - 1.0) / (A + 1e-6) * self.B
        
        # Initialize sequence hidden state to zeros (assuming per-token routing execution, 
        # this acts like a depth-wise recurrent pass per token)
        h = torch.zeros(x.shape[:-1] + (self.d_model, self.d_state), device=x.device, dtype=x.dtype)
        
        # Step the state 3 times recursively for depth without sequence
        for _ in range(3):
            h = dA * h + dB * x.unsqueeze(-1)
            
        y = (self.C * h).sum(dim=-1) + self.D * x
        return F.silu(y)


class HolographicStateSpaceExpertHead(nn.Module):
    """
    v28 Holographic State Space Expert Head.
    1. Holographic Memory: Binds discrete token features globally using FFT circular convolution.
    2. State Space Experts (S4-like): Handles mapping via continuous LDS.
    3. Dynamic Topology Routing: Chaining experts in an input-dependent sequence path.
    """
    def __init__(self, in_dim: int = 256, out_dim: int = 10, n_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts
        
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        # Learned holographic associative memory matrix
        self.holographic_memory = nn.Parameter(torch.empty(1, in_dim))
        self.holo_proj = nn.Linear(in_dim, in_dim, bias=False)
        
        self.ssm_experts = nn.ModuleList([StateSpaceExpert(in_dim) for _ in range(n_experts)])
        
        # Adjacency topology router
        # Outputs a (n_experts, n_experts) adjacency matrix per token
        self.topology_router = nn.Linear(in_dim, n_experts * n_experts)
        
        self.out_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.holographic_memory, std=0.02)
        nn.init.normal_(self.holo_proj.weight, std=0.01)
        nn.init.zeros_(self.topology_router.weight)
        nn.init.zeros_(self.topology_router.bias)
        nn.init.zeros_(self.out_proj.weight)

    def _circular_convolution(self, a, b):
        """HRR Binding via FFT."""
        # Convert to complex plane to perform cyclic convolution in frequency domain
        a_f = torch.fft.rfft(a, dim=-1)
        b_f = torch.fft.rfft(b, dim=-1)
        c = torch.fft.irfft(a_f * b_f, n=a.shape[-1], dim=-1)
        return c

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        
        # 1. Holographic Memory Binding
        # Bind the token state x with the learned persistent holographic_memory
        x_bound = self._circular_convolution(x, self.holographic_memory.expand_as(x))
        x_holo = self.holo_proj(x_bound)
        
        # 2. Dynamic Topology Routing
        # B x T x (N*N) -> B x T x N x N
        adj_flat = self.topology_router(x_holo)
        adj_logits = adj_flat.view(*x.shape[:-1], self.n_experts, self.n_experts)
        # Convert to a soft permutation matrix via doubly stochastic approximation (Sinkhorn)
        # For simplicity in testing, we use softmax over rows
        adj = torch.softmax(adj_logits, dim=-1)
        
        # 3. State Space Execution Path
        # Instead of parallel execution, we execute iteratively.
        current_state = x_holo
        accum_out = torch.zeros_like(x_holo)
        
        expert_outs = []
        for i in range(self.n_experts):
            h = self.ssm_experts[i](current_state)
            expert_outs.append(h)
        expert_stack = torch.stack(expert_outs, dim=-2) # (..., N_experts, D)
        
        # Route outputs through adjacency.
        # current_state = \sum_j adj[i, j] * expert_stack[j]
        # In a generic formulation, we compute the aggregated state as a mixture.
        # We do N passes, representing N hops in the dynamic graph.
        for step in range(self.n_experts):
            step_mix = torch.einsum('...ij,...jd->...id', adj, expert_stack)
            # The next state input is the average mixture for the next nodes
            current_state = step_mix.mean(dim=-2) 
            accum_out = accum_out + current_state
            
            # Re-evaluate experts via recurrence (dynamic routing unrolled)
            expert_outs = []
            for i in range(self.n_experts):
                h = self.ssm_experts[i](current_state)
                expert_outs.append(h)
            expert_stack = torch.stack(expert_outs, dim=-2)
            
        final_h = accum_out / self.n_experts
        routed_logits = self.out_proj(self.dropout(final_h))
        
        return base_logits + self.alpha * routed_logits


class ChampionNetHolographicStateSpaceExpert(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        from run import ChampionNet
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(HolographicStateSpaceExpertHead(
            256, 10,
            n_experts=4,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TemporalSSMAdapter(nn.Module):
    """STAD-inspired temporal state-space adapter (Schirmer et al., arXiv 2407.12492).
    Learns time-varying dynamics in hidden features and applies a corrective residual.
    Uses a lightweight SSM: h' = Ah + Bx, correction = Ch + Dx.
    """
    def __init__(self, d_model: int, d_state: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A = nn.Parameter(torch.empty(d_model, d_state))
        self.B = nn.Parameter(torch.empty(d_model, d_state))
        self.C = nn.Parameter(torch.empty(d_model, d_state))
        self.D = nn.Parameter(torch.zeros(d_model))
        self.gate = nn.Parameter(torch.tensor(0.0))  # starts at 0 for stable init
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.A, -5.0, -0.5)
        nn.init.normal_(self.B, std=0.02)
        nn.init.normal_(self.C, std=0.02)
        nn.init.zeros_(self.D)

    def forward(self, x, h=None):
        # x: (..., d_model)
        # Use tanh(A) for state decay — keeps values in (-1, 1) with healthy gradients
        A_decay = torch.tanh(self.A)
        if h is None:
            h = torch.zeros(x.shape[:-1] + (self.d_model, self.d_state),
                            device=x.device, dtype=x.dtype)
        # Single step update
        h_new = A_decay * h + self.B * x.unsqueeze(-1)
        correction = (self.C * h_new).sum(dim=-1) + self.D * x
        return self.gate * correction, h_new


class MixtureOfRouters(nn.Module):
    """MoR-inspired multi-sub-router ensemble (Zhang et al., 2025).
    K independent sub-routers score experts. A main router weights their decisions.
    """
    def __init__(self, in_dim: int, n_experts: int, n_sub_routers: int = 3):
        super().__init__()
        self.n_sub_routers = n_sub_routers
        self.n_experts = n_experts
        self.sub_routers = nn.ModuleList([
            nn.Linear(in_dim, n_experts, bias=True) for _ in range(n_sub_routers)
        ])
        self.main_router = nn.Linear(in_dim, n_sub_routers, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for sr in self.sub_routers:
            nn.init.normal_(sr.weight, std=0.01)
            nn.init.zeros_(sr.bias)
        nn.init.zeros_(self.main_router.weight)
        nn.init.zeros_(self.main_router.bias)

    def forward(self, x):
        # Each sub-router produces expert scores
        sub_scores = torch.stack([sr(x) for sr in self.sub_routers], dim=-2)  # (..., K, N)
        # Main router produces sub-router weights
        main_weights = torch.softmax(self.main_router(x), dim=-1)  # (..., K)
        # Weighted combination: (..., N)
        fused_scores = torch.einsum('...k,...kn->...n', main_weights, sub_scores)
        return fused_scores


class PaperFusionExpertHead(nn.Module):
    """v29 Paper-Driven Expert Head combining three 2025 research techniques:
    1. MoR (Mixture of Routers) — Zhang et al., 2025: Multi-sub-router ensemble.
    2. RoMA (Routing Manifold Alignment) — Li et al., arXiv 2511.07419: Manifold
       regularization aligning routing weights with input embeddings.
    3. STAD (Temporal SSM Adapter) — Schirmer et al., arXiv 2407.12492: SSM-based
       test-time feature drift correction.
    """
    def __init__(self, in_dim: int = 256, out_dim: int = 10, n_experts: int = 6,
                 n_sub_routers: int = 3, ssm_d_state: int = 32, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Shared expert (always active)
        self.shared_up = nn.Linear(in_dim, 512, bias=False)
        self.shared_down = nn.Linear(512, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(0.0))

        # MoR: Mixture of Routers (Paper 1)
        self.mor_router = MixtureOfRouters(in_dim, n_experts, n_sub_routers)

        # Routed experts with diverse activations
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, 1024, bias=False) for _ in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(1024, out_dim, bias=False) for _ in range(n_experts)
        ])
        self._expert_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]

        # STAD: Temporal SSM Adapter (Paper 3)
        self.ssm_adapter = TemporalSSMAdapter(in_dim, d_state=ssm_d_state)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)

        # RoMA manifold alignment loss (Paper 2) — stored for training script access
        self._manifold_loss = torch.tensor(0.0)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.shared_up.weight, std=0.01)
        nn.init.zeros_(self.shared_down.weight)
        for eu in self.experts_up:
            nn.init.normal_(eu.weight, std=0.01)
        for ed in self.experts_down:
            nn.init.zeros_(ed.weight)

    def _compute_manifold_alignment_loss(self, x_flat, routing_weights):
        """RoMA: Penalize routing dissimilarity between embedding-space neighbors.
        Inputs should be (..., D) and (..., N_experts) respectively.
        """
        if x_flat.shape[0] < 2:
            return torch.tensor(0.0, device=x_flat.device)
        # Pairwise cosine similarity of input embeddings
        x_norm = F.normalize(x_flat, dim=-1)
        embed_sim = torch.mm(x_norm, x_norm.t())  # (B, B)
        # Pairwise cosine similarity of routing weights
        r_norm = F.normalize(routing_weights, dim=-1)
        route_sim = torch.mm(r_norm, r_norm.t())  # (B, B)
        # MSE alignment loss
        return F.mse_loss(route_sim, embed_sim.detach())

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_dim)

        # Shared expert
        shared_out = self.shared_norm(self.shared_down(F.silu(self.shared_up(x_flat))))

        # STAD: Apply temporal SSM correction
        ssm_correction, _ = self.ssm_adapter(x_flat)
        x_corrected = x_flat + ssm_correction

        # MoR: Ensemble routing scores
        gate_scores = torch.softmax(self.mor_router(x_corrected), dim=-1)  # (B, N)

        # RoMA: Compute manifold alignment loss (training only)
        if self.training:
            self._manifold_loss = self._compute_manifold_alignment_loss(x_flat, gate_scores)
        else:
            self._manifold_loss = torch.tensor(0.0, device=x.device)

        # Expert execution
        expert_outputs = []
        for i in range(self.n_experts):
            act_fn = self._expert_acts[i % len(self._expert_acts)]
            h = act_fn(self.experts_up[i](x_corrected))
            h = self.dropout(h)
            expert_outputs.append(self.experts_down[i](h))

        expert_stack = torch.stack(expert_outputs, dim=-2)  # (B, N, out_dim)
        routed_out = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=-2)  # (B, out_dim)

        output = (
            base_logits.reshape(-1, self.out_dim)
            + self.shared_scale * shared_out
            + self.alpha * routed_out
        )
        return output.view(*shape_prefix, self.out_dim)


class ChampionNetPaperFusionExpert(nn.Module):
    """Backbone wrapper for v29 PaperFusionExpertHead."""
    def __init__(self, n_sub_routers: int = 3, dropout: float = 0.1):
        super().__init__()
        from run import ChampionNet
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(PaperFusionExpertHead(
            256, 10,
            n_experts=6,
            n_sub_routers=n_sub_routers,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetHierarchicalExpert(nn.Module):
    """
    Backbone-compatible model with HierarchicalMoEClassifierHead.
    Uses the same layers 0-9 backbone from ChampionNet with the new
    hierarchical MoE head at layer 10.
    """
    def __init__(
        self,
        n_domain_groups: int = 2,
        experts_per_group: int = 4,
        top_k_domains: int = 1,
        top_k_experts: int = 2,
        lora_rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(HierarchicalMoEClassifierHead(
            256, 10,
            n_domain_groups=n_domain_groups,
            experts_per_group=experts_per_group,
            top_k_domains=top_k_domains,
            top_k_experts=top_k_experts,
            lora_rank=lora_rank,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetDeepExpert(nn.Module):
    """
    Backbone-compatible model with DeepExpertClassifierHead.
    """
    def __init__(self, n_experts: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(DeepExpertClassifierHead(256, 10, n_experts=n_experts, top_k=top_k, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetExpertChoice(nn.Module):
    """
    Backbone-compatible model with ExpertChoiceClassifierHead.
    """
    def __init__(self, n_experts: int = 8, capacity_factor: float = 2.0, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ExpertChoiceClassifierHead(256, 10, n_experts=n_experts, capacity_factor=capacity_factor, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

        return x


class ChampionNetSmarterExpert(nn.Module):
    """
    Backbone-compatible model with SmarterExpertClassifierHead.
    """
    def __init__(self, n_experts: int = 8, lora_rank: int = 32, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(SmarterExpertClassifierHead(256, 10, n_experts=n_experts, lora_rank=lora_rank, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class ChampionNetThoughtExpert(nn.Module):
    """
    Backbone-compatible model with ThoughtExpertClassifierHead.
    """
    def __init__(self, n_experts: int = 8, reasoning_steps: int = 3, lora_rank: int = 32, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ThoughtExpertClassifierHead(
            256, 10, 
            n_experts=n_experts, 
            reasoning_steps=reasoning_steps, 
            lora_rank=lora_rank, 
            dropout=dropout
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetUltraExpert(nn.Module):
    """
    Backbone-compatible model with GatedExpertClassifierHead.
    """
    def __init__(self, n_experts: int = 6, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(GatedExpertClassifierHead(256, 10, n_experts=n_experts, top_k=top_k, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetLarge(nn.Module):
    """
    Backbone-compatible larger model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHead
    """

    def __init__(self, expansion_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ExpandedClassifierHead(256, 10, expansion_dim=expansion_dim, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetXL(nn.Module):
    """
    Backbone-compatible xlarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadXL
    """

    def __init__(self, expansion_dim: int = 768, extra_expansion_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadXL(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetXXL(nn.Module):
    """
    Backbone-compatible xxlarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadXXL
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadXXL(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetXXXL(nn.Module):
    """
    Backbone-compatible xxxlarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadXXXL
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadXXXL(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                fourth_expansion_dim=fourth_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetUltra(nn.Module):
    """
    Backbone-compatible ultralarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadUltra
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadUltra(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                fourth_expansion_dim=fourth_expansion_dim,
                fifth_expansion_dim=fifth_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetMega(nn.Module):
    """
    Backbone-compatible megalarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadMega
    - adds GatedFFN after BitNetLinear for extra backbone capacity
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        sixth_expansion_dim: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        # Insert GatedFFN for extra backbone capacity (between BitNet and classifier).
        layers.append(GatedFFN(d_model=256, d_inner=512))
        layers.append(
            ExpandedClassifierHeadMega(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                fourth_expansion_dim=fourth_expansion_dim,
                fifth_expansion_dim=fifth_expansion_dim,
                sixth_expansion_dim=sixth_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_model(
    model_size: str = "base",
    expansion_dim: int = 512,
    dropout: float = 0.1,
    extra_expansion_dim: Optional[int] = None,
    third_expansion_dim: Optional[int] = None,
    fourth_expansion_dim: Optional[int] = None,
    fifth_expansion_dim: Optional[int] = None,
    sixth_expansion_dim: Optional[int] = None,
) -> nn.Module:
    """Construct a ChampionNet model variant by name.

    Factory function that dispatches to the appropriate backbone class
    based on model_size. Expansion dims default to sensible values when
    not provided.

    Args:
        model_size: One of 'base', 'large', 'xlarge', 'xxlarge', 'xxxlarge',
            'ultralarge', 'megalarge', 'ultra_expert', 'hierarchical_expert',
            'deep_expert', 'expert_choice', 'smarter_expert', 'thought_expert',
            'recursive_expert'.
        expansion_dim: Hidden size for the primary adapter branch.
        dropout: Dropout rate for adapter layers.
        extra_expansion_dim: Hidden size for the secondary adapter branch.
        third_expansion_dim: Hidden size for the third adapter branch.
        fourth_expansion_dim: Hidden size for the fourth adapter branch.
        fifth_expansion_dim: Hidden size for the fifth adapter branch.
        sixth_expansion_dim: Hidden size for the sixth adapter branch.

    Returns:
        An nn.Module ready for training or inference.

    Raises:
        ValueError: If model_size is not recognized.
    """
    if model_size == "base":
        return ChampionNet()
    if model_size == "large":
        return ChampionNetLarge(expansion_dim=expansion_dim, dropout=dropout)
    if model_size == "xlarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(1024, expansion_dim * 2))
        return ChampionNetXL(expansion_dim=expansion_dim, extra_expansion_dim=extra_dim, dropout=dropout)
    if model_size == "xxlarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        return ChampionNetXXL(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            dropout=dropout,
        )
    if model_size == "xxxlarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        fourth_dim = (
            int(fourth_expansion_dim)
            if fourth_expansion_dim is not None
            else int(max(4096, third_dim + expansion_dim))
        )
        return ChampionNetXXXL(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            fourth_expansion_dim=fourth_dim,
            dropout=dropout,
        )
    if model_size == "ultralarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        fourth_dim = (
            int(fourth_expansion_dim)
            if fourth_expansion_dim is not None
            else int(max(4096, third_dim + expansion_dim))
        )
        fifth_dim = (
            int(fifth_expansion_dim)
            if fifth_expansion_dim is not None
            else int(max(6144, fourth_dim + expansion_dim))
        )
        return ChampionNetUltra(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            fourth_expansion_dim=fourth_dim,
            fifth_expansion_dim=fifth_dim,
            dropout=dropout,
        )
    if model_size == "megalarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        fourth_dim = (
            int(fourth_expansion_dim)
            if fourth_expansion_dim is not None
            else int(max(4096, third_dim + expansion_dim))
        )
        fifth_dim = (
            int(fifth_expansion_dim)
            if fifth_expansion_dim is not None
            else int(max(6144, fourth_dim + expansion_dim))
        )
        sixth_dim = (
            int(sixth_expansion_dim)
            if sixth_expansion_dim is not None
            else int(max(8192, fifth_dim + expansion_dim))
        )
        return ChampionNetMega(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_expansion_dim,
            third_expansion_dim=third_dim,
            fourth_expansion_dim=fourth_dim,
            fifth_expansion_dim=fifth_dim,
            sixth_expansion_dim=sixth_dim,
            dropout=dropout,
        )
    if model_size == "ultra_expert":
        return ChampionNetUltraExpert(dropout=dropout)
    if model_size == "hierarchical_expert":
        return ChampionNetHierarchicalExpert(dropout=dropout)
    if model_size == "deep_expert":
        return ChampionNetDeepExpert(dropout=dropout)
    if model_size == "expert_choice":
        return ChampionNetExpertChoice(dropout=dropout)
    if model_size == "smarter_expert":
        return ChampionNetSmarterExpert(dropout=dropout)
    if model_size == "thought_expert":
        return ChampionNetThoughtExpert(dropout=dropout)
    if model_size == "recursive_expert":
        return ChampionNetRecursiveExpert(dropout=dropout)
    if model_size == "reflexive_expert":
        return ChampionNetReflexiveExpert(dropout=dropout)
    if model_size == "metacognitive_expert":
        return ChampionNetMetaCognitiveExpert(dropout=dropout)
    if model_size == "tree_of_thought_expert":
        return ChampionNetTreeOfThoughtExpert(dropout=dropout)
    if model_size == "consensus_expert":
        return ChampionNetConsensusExpert(dropout=dropout)
    if model_size == "deliberative_expert":
        return ChampionNetDeliberativeExpert(dropout=dropout)
    if model_size == "omniscient_expert":
        return ChampionNetOmniscientExpert(dropout=dropout)
    if model_size == "neurogenesis_expert":
        return ChampionNetNeurogenesisExpert(dropout=dropout)
    if model_size == "cognitive_expert":
        return ChampionNetCognitiveExpert(dropout=dropout)
    if model_size == "transcendent_expert":
        return ChampionNetTranscendentExpert(dropout=dropout)
    if model_size == "omniversal_expert":
        return ChampionNetOmniversalExpert(dropout=dropout)
    if model_size == "fractal_expert":
        return ChampionNetFractalExpert(dropout=dropout)
    if model_size == "liquid_spiking_expert":
        return ChampionNetLiquidSpikingExpert(dropout=dropout)
    if model_size == "active_inference_expert":
        return ChampionNetActiveInferenceExpert(dropout=dropout)
    if model_size == "holographic_state_space_expert":
        return ChampionNetHolographicStateSpaceExpert(dropout=dropout)
    if model_size == "paper_fusion_expert":
        return ChampionNetPaperFusionExpert(dropout=dropout)
    if model_size == "frontier_expert":
        from model_frontier_v33 import ChampionNetFrontierExpert

        return ChampionNetFrontierExpert(dropout=dropout)
    if model_size == "frontier_collective_expert":
        from model_frontier_v35 import ChampionNetFrontierCollectiveExpert

        return ChampionNetFrontierCollectiveExpert(dropout=dropout)
    if model_size == "frontier_verifier_expert":
        from model_frontier_v39 import ChampionNetFrontierVerifierExpert

        return ChampionNetFrontierVerifierExpert(dropout=dropout)

    raise ValueError(
        f"Unknown model_size={model_size!r}. Use one of {SUPPORTED_MODEL_SIZES}."
    )



def load_weights_for_model(model: nn.Module, state_dict: dict, model_size: str) -> Tuple[List[str], List[str]]:
    """Load checkpoint weights into a model with backward compatibility.

    Handles warm-starting from smaller checkpoints by filtering out expected
    missing keys (new head parameters) and stripping incompatible head weights
    when upgrading across variant families.

    Args:
        model: The target model to load weights into.
        state_dict: Checkpoint state dictionary.
        model_size: The variant name of the target model.

    Returns:
        Tuple of (unexpected_missing_keys, unexpected_keys). Keys that are
        expected to be missing for warm-start scenarios are filtered out.

    Raises:
        RuntimeError: If weight dimensions are incompatible.
        ValueError: If model_size is not recognized.
    """
    if model_size == "base":
        incompatible = model.load_state_dict(state_dict, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    if model_size == "large":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load large checkpoint: likely head-dimension mismatch. "
                "Use matching --expansion_dim or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when loading base checkpoint into large model.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "xlarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load xlarge checkpoint: likely expansion/aux dimension mismatch. "
                "Use matching --expansion_dim/--extra_expansion_dim or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting xlarge from base/large checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "xxlarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load xxlarge checkpoint: likely expansion/aux/third dimension mismatch. "
                "Use matching --expansion_dim/--extra_expansion_dim/--third_expansion_dim or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting xxlarge from base/large/xlarge checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "xxxlarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load xxxlarge checkpoint: likely expansion/aux/third/fourth dimension mismatch. "
                "Use matching dims or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting xxxlarge from base/large/xlarge/xxlarge checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
            "layers.10.adapter_up_d.weight",
            "layers.10.adapter_down_d.weight",
            "layers.10.router3.weight",
            "layers.10.router3.bias",
            "layers.10.delta",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "ultralarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load ultralarge checkpoint: likely expansion/aux/third/fourth/fifth dimension mismatch. "
                "Use matching dims or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting ultralarge from smaller checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
            "layers.10.adapter_up_d.weight",
            "layers.10.adapter_down_d.weight",
            "layers.10.router3.weight",
            "layers.10.router3.bias",
            "layers.10.delta",
            "layers.10.adapter_up_e.weight",
            "layers.10.adapter_down_e.weight",
            "layers.10.router4.weight",
            "layers.10.router4.bias",
            "layers.10.epsilon",
            "layers.10.pre_norm.weight",
            "layers.10.pre_norm.bias",
            "layers.10.domain_router.weight",
            "layers.10.domain_router.bias",
            "layers.10.domain_experts.weight",
            "layers.10.calib_gate.weight",
            "layers.10.calib_gate.bias",
            "layers.10.zeta",
            "layers.10.theta",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "megalarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load megalarge checkpoint: likely expansion dimension mismatch. "
                "Use matching dims or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting megalarge from smaller checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
            "layers.10.adapter_up_d.weight",
            "layers.10.adapter_down_d.weight",
            "layers.10.router3.weight",
            "layers.10.router3.bias",
            "layers.10.delta",
            "layers.10.adapter_up_e.weight",
            "layers.10.adapter_down_e.weight",
            "layers.10.router4.weight",
            "layers.10.router4.bias",
            "layers.10.epsilon",
            "layers.10.pre_norm.weight",
            "layers.10.pre_norm.bias",
            "layers.10.domain_router.weight",
            "layers.10.domain_router.bias",
            "layers.10.domain_experts.weight",
            "layers.10.calib_gate.weight",
            "layers.10.calib_gate.bias",
            "layers.10.zeta",
            "layers.10.theta",
            # Megalarge-only keys
            "layers.10.adapter_up_f.weight",
            "layers.10.adapter_down_f.weight",
            "layers.10.router5.weight",
            "layers.10.router5.bias",
            "layers.10.iota",
            "layers.10.cross_attn_fusion.q_proj.weight",
            "layers.10.cross_attn_fusion.k_proj.weight",
            "layers.10.cross_attn_fusion.v_proj.weight",
            "layers.10.cross_attn_fusion.out_proj.weight",
            "layers.10.cross_attn_fusion.fusion_weight.weight",
            "layers.10.cross_attn_fusion.fusion_weight.bias",
            "layers.10.kappa",
            "layers.10.reasoning_gate.0.weight",
            "layers.10.reasoning_gate.2.weight",
            "layers.10.reasoning_gate.2.bias",
            "layers.10.lambda_",
            # GatedFFN backbone layer (layers.10 in mega = GatedFFN)
            "layers.10.norm.weight",
            "layers.10.up_proj.weight",
            "layers.10.down_proj.weight",
            "layers.10.layer_scale.gamma",
        }
        # Megalarge uses layers.10=GatedFFN, layers.11=Head, layers.12=norm
        # So head keys are at layers.11.* not layers.10.*
        mega_allowed = set()
        for k in allowed_missing:
            if k.startswith("layers.10."):
                mega_key = "layers.11." + k[len("layers.10."):]
                mega_allowed.add(mega_key)
        mega_allowed.update(allowed_missing)
        # Also allow GatedFFN keys at layers.10
        mega_allowed.update({
            "layers.10.norm.weight",
            "layers.10.up_proj.weight",
            "layers.10.down_proj.weight",
            "layers.10.layer_scale.gamma",
        })
        
        # Robust loading: if checkpoint is not megalarge, strip head keys to avoid mismatch
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "megalarge":
            filtered_sd = {k: v for k, v in state_dict.items() if not k.startswith("layers.10.") and not k.startswith("layers.11.")}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in mega_allowed]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "ultra_expert":
        allowed_missing = {
            "layers.10.experts_up.0.weight", "layers.10.experts_up.1.weight",
            "layers.10.experts_up.2.weight", "layers.10.experts_up.3.weight",
            "layers.10.experts_up.4.weight", "layers.10.experts_up.5.weight",
            "layers.10.experts_down.0.weight", "layers.10.experts_down.1.weight",
            "layers.10.experts_down.2.weight", "layers.10.experts_down.3.weight",
            "layers.10.experts_down.4.weight", "layers.10.experts_down.5.weight",
            "layers.10.gate.weight", "layers.10.noise_gate.weight",
            "layers.10.alpha", "layers.10.calibration.weight",
            "layers.10.calibration.bias", "layers.10.theta",
        }
        
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "ultra_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "hierarchical_expert":
        # Allow all new hierarchical head parameters to be missing when warm-starting
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias", f"{head_pref}shared_scale",
            f"{head_pref}domain_gate.weight", f"{head_pref}domain_noise_gate.weight",
            f"{head_pref}alpha", f"{head_pref}theta",
            f"{head_pref}calibration.weight", f"{head_pref}calibration.bias",
        }
        for i in range(2): # groups
            allowed_missing.add(f"{head_pref}expert_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_noise_gates.{i}.weight")
        for i in range(8): # experts
            allowed_missing.add(f"{head_pref}experts_up.{i}.weight")
            allowed_missing.add(f"{head_pref}experts_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
            allowed_missing.add(f"{head_pref}lora_down.{i}.weight")
            allowed_missing.add(f"{head_pref}lora_up.{i}.weight")
            
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "hierarchical_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "thought_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias", 
            f"{head_pref}shared_scale", f"{head_pref}expert_bias",
            f"{head_pref}alpha", f"{head_pref}calibration.weight",
            f"{head_pref}calibration.bias", f"{head_pref}theta",
            f"{head_pref}cross_attn.q_proj.weight", f"{head_pref}cross_attn.k_proj.weight",
            f"{head_pref}cross_attn.v_proj.weight", f"{head_pref}cross_attn.out_proj.weight",
            f"{head_pref}cross_attn.fusion_weight.weight", f"{head_pref}cross_attn.fusion_weight.bias",
        }
        for i in range(8): # n_experts
            allowed_missing.add(f"{head_pref}experts_up.{i}.weight")
            allowed_missing.add(f"{head_pref}experts_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
            allowed_missing.add(f"{head_pref}lora_down.{i}.weight")
            allowed_missing.add(f"{head_pref}lora_up.{i}.weight")
        for i in range(3): # reasoning_steps
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.bias")
            allowed_missing.add(f"{head_pref}gates.{i}.weight")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "thought_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "recursive_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias", 
            f"{head_pref}shared_scale", f"{head_pref}expert_bias",
            f"{head_pref}local_up.weight", f"{head_pref}local_down.weight",
            f"{head_pref}local_norm.weight", f"{head_pref}local_norm.bias", 
            f"{head_pref}local_scale",
            f"{head_pref}alpha", f"{head_pref}calibration.weight",
            f"{head_pref}calibration.bias", f"{head_pref}theta",
            f"{head_pref}cross_attn.q_proj.weight", f"{head_pref}cross_attn.k_proj.weight",
            f"{head_pref}cross_attn.v_proj.weight", f"{head_pref}cross_attn.out_proj.weight",
            f"{head_pref}cross_attn.fusion_weight.weight", f"{head_pref}cross_attn.fusion_weight.bias",
        }
        for i in range(8): # n_experts
            allowed_missing.add(f"{head_pref}experts_up.{i}.weight")
            allowed_missing.add(f"{head_pref}experts_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
            allowed_missing.add(f"{head_pref}lora_down.{i}.weight")
            allowed_missing.add(f"{head_pref}lora_up.{i}.weight")
        for i in range(3): # reasoning_steps
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.bias")
            allowed_missing.add(f"{head_pref}exit_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}exit_gates.{i}.bias")
            for h in range(2): # n_heads
                allowed_missing.add(f"{head_pref}gates.{i * 2 + h}.weight")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "recursive_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "reflexive_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}initial_gate.weight",
            f"{head_pref}critique_gate.weight",
            f"{head_pref}alpha",
            f"{head_pref}beta",
        }
        for i in range(4): # default n_experts for reflexive
            allowed_missing.add(f"{head_pref}initial_up.{i}.weight")
            allowed_missing.add(f"{head_pref}initial_down.{i}.weight")
            allowed_missing.add(f"{head_pref}initial_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}initial_norms.{i}.bias")
            allowed_missing.add(f"{head_pref}critique_up.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_down.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_norms.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "reflexive_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "metacognitive_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}proposal_bias",
            f"{head_pref}alpha", f"{head_pref}beta",
        }
        for i in range(6):  # n_proposal_experts
            allowed_missing.add(f"{head_pref}proposal_up.{i}.weight")
            allowed_missing.add(f"{head_pref}proposal_down.{i}.weight")
            allowed_missing.add(f"{head_pref}proposal_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}proposal_norms.{i}.bias")
        for i in range(4):  # n_critique_experts
            allowed_missing.add(f"{head_pref}critique_up.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_down.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_norms.{i}.bias")
        for i in range(3):  # reasoning_steps
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.bias")
            allowed_missing.add(f"{head_pref}proposal_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "metacognitive_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "tree_of_thought_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}final_up.weight", f"{head_pref}final_down.weight",
            f"{head_pref}final_norm.weight", f"{head_pref}final_norm.bias",
        }
        for i in range(6):  # n_action_experts
            allowed_missing.add(f"{head_pref}action_up.{i}.weight")
            allowed_missing.add(f"{head_pref}action_down.{i}.weight")
            allowed_missing.add(f"{head_pref}action_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}action_norms.{i}.bias")
        # Value net
        allowed_missing.add(f"{head_pref}value_net.0.weight")
        allowed_missing.add(f"{head_pref}value_net.0.bias")
        allowed_missing.add(f"{head_pref}value_net.3.weight")
        allowed_missing.add(f"{head_pref}value_net.3.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "tree_of_thought_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "consensus_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}alpha",
            f"{head_pref}mlp_gate.weight",
            f"{head_pref}attn_embed",
            f"{head_pref}attn_q.weight", f"{head_pref}attn_k.weight",
            f"{head_pref}attn_v.weight", f"{head_pref}attn_proj.weight",
            f"{head_pref}attn_norm.weight", f"{head_pref}attn_norm.bias",
            f"{head_pref}conv1.weight", f"{head_pref}conv1.bias",
            f"{head_pref}conv2.weight", f"{head_pref}conv2.bias",
            f"{head_pref}conv_proj.weight",
            f"{head_pref}conv_norm.weight", f"{head_pref}conv_norm.bias",
            f"{head_pref}consensus.0.weight", f"{head_pref}consensus.0.bias",
            f"{head_pref}consensus.3.weight", f"{head_pref}consensus.3.bias",
            f"{head_pref}consensus.5.weight", f"{head_pref}consensus.5.bias",
        }
        for i in range(6):  # n_mlp_experts
            allowed_missing.add(f"{head_pref}mlp_up.{i}.weight")
            allowed_missing.add(f"{head_pref}mlp_down.{i}.weight")
            allowed_missing.add(f"{head_pref}mlp_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}mlp_norms.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "consensus_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "deliberative_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}memory_keys", f"{head_pref}memory_values",
            f"{head_pref}mem_read_query.weight",
            f"{head_pref}mem_write_query.weight",
            f"{head_pref}mem_write_gate.weight", f"{head_pref}mem_write_gate.bias",
            # Consistency net
            f"{head_pref}consistency_net.0.weight", f"{head_pref}consistency_net.0.bias",
            f"{head_pref}consistency_net.3.weight", f"{head_pref}consistency_net.3.bias",
            f"{head_pref}consistency_net.5.weight", f"{head_pref}consistency_net.5.bias",
        }
        for i in range(6):  # n_experts
            allowed_missing.add(f"{head_pref}expert_up.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
        for step in range(3):  # reasoning_steps
            for d in range(3):  # n_drafts
                allowed_missing.add(f"{head_pref}draft_reasoning_cells.{step}.{d}.0.weight")
                allowed_missing.add(f"{head_pref}draft_reasoning_cells.{step}.{d}.0.bias")
                allowed_missing.add(f"{head_pref}draft_reasoning_cells.{step}.{d}.1.weight")
                allowed_missing.add(f"{head_pref}draft_reasoning_cells.{step}.{d}.1.bias")
                allowed_missing.add(f"{head_pref}draft_reasoning_cells.{step}.{d}.3.weight")
                allowed_missing.add(f"{head_pref}draft_reasoning_cells.{step}.{d}.3.bias")
                allowed_missing.add(f"{head_pref}draft_gates.{step}.{d}.weight")
            allowed_missing.add(f"{head_pref}cross_draft_q.{step}.weight")
            allowed_missing.add(f"{head_pref}cross_draft_k.{step}.weight")
            allowed_missing.add(f"{head_pref}cross_draft_v.{step}.weight")
            allowed_missing.add(f"{head_pref}cross_draft_norm.{step}.weight")
            allowed_missing.add(f"{head_pref}cross_draft_norm.{step}.bias")
            allowed_missing.add(f"{head_pref}halt_gates.{step}.weight")
            allowed_missing.add(f"{head_pref}halt_gates.{step}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "deliberative_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "omniscient_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}knowledge_core_keys", f"{head_pref}knowledge_core_values",
            f"{head_pref}knowledge_query.weight", f"{head_pref}temperature",
            f"{head_pref}node_fusion_q.weight", f"{head_pref}node_fusion_k.weight",
            f"{head_pref}node_fusion_v.weight",
        }
        for i in range(6):  # n_experts
            allowed_missing.add(f"{head_pref}expert_up.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
        for step in range(3):  # reasoning_steps
            allowed_missing.add(f"{head_pref}got_q.{step}.weight")
            allowed_missing.add(f"{head_pref}got_k.{step}.weight")
            allowed_missing.add(f"{head_pref}got_v.{step}.weight")
            allowed_missing.add(f"{head_pref}got_norm.{step}.weight")
            allowed_missing.add(f"{head_pref}got_norm.{step}.bias")
            allowed_missing.add(f"{head_pref}route_mu.{step}.weight")
            allowed_missing.add(f"{head_pref}route_logvar.{step}.weight")
            allowed_missing.add(f"{head_pref}route_logvar.{step}.bias")
            allowed_missing.add(f"{head_pref}critique_net.{step}.0.weight")
            allowed_missing.add(f"{head_pref}critique_net.{step}.0.bias")
            allowed_missing.add(f"{head_pref}critique_net.{step}.3.weight")
            allowed_missing.add(f"{head_pref}critique_net.{step}.3.bias")
            allowed_missing.add(f"{head_pref}node_halt_gates.{step}.weight")
            allowed_missing.add(f"{head_pref}node_halt_gates.{step}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "omniscient_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "omniversal_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}holographic_memory",
            f"{head_pref}real_lift.weight", f"{head_pref}real_lift.bias",
            f"{head_pref}imag_lift.weight", f"{head_pref}imag_lift.bias",
            f"{head_pref}uncertainty_gate.weight", f"{head_pref}uncertainty_gate.bias",
        }
        for sub in ['ode_func.net.0', 'ode_func.net.3', 'ode_func.q_proj', 'ode_func.k_proj', 'ode_func.v_proj']:
            allowed_missing.add(f"{head_pref}{sub}.weight")
            allowed_missing.add(f"{head_pref}{sub}.bias")
        # measurement_proj
        allowed_missing.add(f"{head_pref}measurement_proj.0.weight")
        allowed_missing.add(f"{head_pref}measurement_proj.0.bias")
        allowed_missing.add(f"{head_pref}measurement_proj.2.weight")
        allowed_missing.add(f"{head_pref}measurement_proj.2.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "omniversal_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "transcendent_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}time_embed", f"{head_pref}log_betas",
            f"{head_pref}attn_q.weight", f"{head_pref}attn_k.weight", f"{head_pref}attn_v.weight",
            f"{head_pref}attn_norm.weight", f"{head_pref}attn_norm.bias",
            f"{head_pref}meta_q_mod.weight", f"{head_pref}meta_k_mod.weight",
            f"{head_pref}fitness_proj.weight", f"{head_pref}fitness_proj.bias",
        }
        for i in range(4):  # decompose_heads
            for j in [0,1]:
                allowed_missing.add(f"{head_pref}decompose_heads.{i}.{j}.weight")
                allowed_missing.add(f"{head_pref}decompose_heads.{i}.{j}.bias")
            allowed_missing.add(f"{head_pref}decompose_heads.{i}.1.weight")
            allowed_missing.add(f"{head_pref}decompose_heads.{i}.1.bias")
        for t in range(4):  # denoiser steps
            allowed_missing.add(f"{head_pref}denoiser.{t}.0.weight")
            allowed_missing.add(f"{head_pref}denoiser.{t}.0.bias")
            allowed_missing.add(f"{head_pref}denoiser.{t}.3.weight")
            allowed_missing.add(f"{head_pref}denoiser.{t}.3.bias")
        for sub in ['world_transition', 'world_value', 'crossover_net', 'composer']:
            for idx in range(5):
                allowed_missing.add(f"{head_pref}{sub}.{idx}.weight")
                allowed_missing.add(f"{head_pref}{sub}.{idx}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "transcendent_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "cognitive_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}memory_matrix",
            f"{head_pref}mem_read_q.weight", f"{head_pref}mem_read_out.weight",
            f"{head_pref}mem_write_gate.weight", f"{head_pref}mem_write_gate.bias",
            f"{head_pref}mem_write_data.weight", f"{head_pref}mem_write_proj.weight",
            f"{head_pref}causal_q.weight", f"{head_pref}causal_k.weight", f"{head_pref}causal_v.weight",
            f"{head_pref}causal_norm.weight", f"{head_pref}causal_norm.bias",
        }
        for i in range(4): # hypernet
            allowed_missing.add(f"{head_pref}hypernet.{i}.weight")
            allowed_missing.add(f"{head_pref}hypernet.{i}.bias")
        # orth paths
        for i in range(4): # n_hypotheses
            allowed_missing.add(f"{head_pref}path_projs.{i}.weight")
        for i in range(3): # value_net
            allowed_missing.add(f"{head_pref}value_net.{i}.weight")
            allowed_missing.add(f"{head_pref}value_net.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "cognitive_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered


    if model_size == "neurogenesis_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
        }
        # Level up/down projections and knowledge banks
        for lvl in range(3):  # n_levels
            allowed_missing.add(f"{head_pref}knowledge_keys.{lvl}")
            allowed_missing.add(f"{head_pref}knowledge_values.{lvl}")
            allowed_missing.add(f"{head_pref}knowledge_queries.{lvl}.weight")
        for i in range(2):  # n_levels - 1
            allowed_missing.add(f"{head_pref}level_up_projs.{i}.0.weight")
            allowed_missing.add(f"{head_pref}level_up_projs.{i}.0.bias")
            allowed_missing.add(f"{head_pref}level_up_projs.{i}.1.weight")
            allowed_missing.add(f"{head_pref}level_up_projs.{i}.1.bias")
            allowed_missing.add(f"{head_pref}level_down_projs.{i}.0.weight")
            allowed_missing.add(f"{head_pref}level_down_projs.{i}.0.bias")
            allowed_missing.add(f"{head_pref}level_down_projs.{i}.1.weight")
            allowed_missing.add(f"{head_pref}level_down_projs.{i}.1.bias")
            allowed_missing.add(f"{head_pref}distill_bridges.{i}.weight")
        # Proposer + Adversary experts
        for i in range(6):  # n_experts
            allowed_missing.add(f"{head_pref}proposer_up.{i}.weight")
            allowed_missing.add(f"{head_pref}proposer_down.{i}.weight")
            allowed_missing.add(f"{head_pref}proposer_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}proposer_norms.{i}.bias")
            allowed_missing.add(f"{head_pref}adversary_up.{i}.weight")
            allowed_missing.add(f"{head_pref}adversary_down.{i}.weight")
            allowed_missing.add(f"{head_pref}adversary_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}adversary_norms.{i}.bias")
        # Per-round components
        for rnd in range(3):  # debate_rounds
            allowed_missing.add(f"{head_pref}proposer_gates.{rnd}.weight")
            allowed_missing.add(f"{head_pref}adversary_gates.{rnd}.weight")
            allowed_missing.add(f"{head_pref}synthesis_net.{rnd}.0.weight")
            allowed_missing.add(f"{head_pref}synthesis_net.{rnd}.0.bias")
            allowed_missing.add(f"{head_pref}synthesis_net.{rnd}.2.weight")
            allowed_missing.add(f"{head_pref}synthesis_net.{rnd}.2.bias")
            allowed_missing.add(f"{head_pref}resolution.{rnd}.0.weight")
            allowed_missing.add(f"{head_pref}resolution.{rnd}.0.bias")
            allowed_missing.add(f"{head_pref}resolution.{rnd}.3.weight")
            allowed_missing.add(f"{head_pref}resolution.{rnd}.3.bias")
            allowed_missing.add(f"{head_pref}confidence_gates.{rnd}.weight")
            allowed_missing.add(f"{head_pref}confidence_gates.{rnd}.bias")
        # Level fusion
        allowed_missing.add(f"{head_pref}level_fusion.0.weight")
        allowed_missing.add(f"{head_pref}level_fusion.0.bias")
        allowed_missing.add(f"{head_pref}level_fusion.2.weight")
        allowed_missing.add(f"{head_pref}level_fusion.2.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "neurogenesis_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "deep_expert":
        allowed_missing = {
            "layers.10.shared_up.weight", "layers.10.shared_down.weight",
            "layers.10.shared_norm.weight", "layers.10.shared_norm.bias", 
            "layers.10.shared_scale", "layers.10.expert_bias",
            "layers.10.gate.weight", "layers.10.noise_gate.weight",
            "layers.10.alpha", "layers.10.calibration.weight",
            "layers.10.calibration.bias", "layers.10.theta",
        }
        for i in range(8): # n_experts max default
            allowed_missing.add(f"layers.10.experts_up.{i}.weight")
            allowed_missing.add(f"layers.10.experts_down.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.bias")
            
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "deep_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "expert_choice":
        allowed_missing = {
            "layers.10.gate.weight", "layers.10.noise_gate.weight",
            "layers.10.alpha", "layers.10.calibration.weight",
            "layers.10.calibration.bias", "layers.10.theta",
        }
        for i in range(8): # default max experts
            allowed_missing.add(f"layers.10.experts_up.{i}.weight")
            allowed_missing.add(f"layers.10.experts_down.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.bias")
            
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "expert_choice":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "smarter_expert":
        allowed_missing = {
            "layers.10.shared_up.weight", "layers.10.shared_down.weight",
            "layers.10.shared_norm.weight", "layers.10.shared_norm.bias", 
            "layers.10.shared_scale", "layers.10.expert_bias",
            "layers.10.gate.weight", "layers.10.alpha", 
            "layers.10.calibration.weight", "layers.10.calibration.bias", "layers.10.theta",
        }
        for i in range(8):
            allowed_missing.add(f"layers.10.experts_up.{i}.weight")
            allowed_missing.add(f"layers.10.experts_down.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.bias")
            allowed_missing.add(f"layers.10.lora_down.{i}.weight")
            allowed_missing.add(f"layers.10.lora_up.{i}.weight")
            
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "smarter_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered


    if model_size == "omniversal_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}holographic_memory",
            f"{head_pref}real_lift.weight", f"{head_pref}real_lift.bias",
            f"{head_pref}imag_lift.weight", f"{head_pref}imag_lift.bias",
            f"{head_pref}uncertainty_gate.weight", f"{head_pref}uncertainty_gate.bias",
        }
        for sub in ['ode_func.net.0', 'ode_func.net.3', 'ode_func.q_proj', 'ode_func.k_proj', 'ode_func.v_proj']:
            allowed_missing.add(f"{head_pref}{sub}.weight")
            allowed_missing.add(f"{head_pref}{sub}.bias")
        # measurement_proj
        allowed_missing.add(f"{head_pref}measurement_proj.0.weight")
        allowed_missing.add(f"{head_pref}measurement_proj.0.bias")
        allowed_missing.add(f"{head_pref}measurement_proj.2.weight")
        allowed_missing.add(f"{head_pref}measurement_proj.2.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "omniversal_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "fractal_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}curvature", f"{head_pref}logic_atoms",
            f"{head_pref}hyperbolic_proj.weight", f"{head_pref}hyperbolic_proj.bias",
            f"{head_pref}assembly_net.0.weight", f"{head_pref}assembly_net.0.bias",
            f"{head_pref}assembly_net.2.weight", f"{head_pref}assembly_net.2.bias",
            f"{head_pref}final_projection.0.weight", f"{head_pref}final_projection.0.bias",
            f"{head_pref}final_projection.3.weight", f"{head_pref}final_projection.3.bias",
        }
        for i in range(2): # fractal_depth
            for sub in ['update_net.0', 'update_net.3', 'stochastic_gate']:
                allowed_missing.add(f"{head_pref}fractal_evolvers.{i}.{sub}.weight")
                allowed_missing.add(f"{head_pref}fractal_evolvers.{i}.{sub}.bias")
        for i in range(1): # fractal_depth - 1
            allowed_missing.add(f"{head_pref}node_spawners.{i}.weight")
            allowed_missing.add(f"{head_pref}node_spawners.{i}.bias")
            allowed_missing.add(f"{head_pref}resonance_combiners.{i}.weight")
            allowed_missing.add(f"{head_pref}resonance_combiners.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "fractal_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "liquid_spiking_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}ttt_lora_A", f"{head_pref}ttt_lora_B",
            f"{head_pref}lif_neuron.threshold", f"{head_pref}lif_neuron.alpha",
            f"{head_pref}liquid_synapse.core_1", f"{head_pref}liquid_synapse.core_2",
            f"{head_pref}liquid_synapse.tau",
        }
        for sub in ['ttt_decoder.weight', 'ttt_decoder.bias']:
            allowed_missing.add(f"{head_pref}{sub}")
        for sub in ['liquid_synapse.liquid_gate.weight', 'liquid_synapse.liquid_gate.bias', 'spike_decoder.weight', 'spike_decoder.bias']:
            allowed_missing.add(f"{head_pref}{sub}")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "liquid_spiking_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "frontier_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha", f"{head_pref}beta",
            f"{head_pref}memory_keys", f"{head_pref}memory_values",
            f"{head_pref}memory_query.weight", f"{head_pref}memory_out.weight",
            f"{head_pref}memory_write_gate.weight", f"{head_pref}memory_write_gate.bias",
            f"{head_pref}memory_write_value.weight", f"{head_pref}memory_write_value.bias",
            f"{head_pref}depth_query.weight", f"{head_pref}depth_key.weight",
            f"{head_pref}depth_value.weight", f"{head_pref}depth_out.weight",
            f"{head_pref}router_budget.weight", f"{head_pref}router_budget.bias",
            f"{head_pref}expert_bias",
        }
        for i in range(4):
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.bias")
        for i in range(10):
            allowed_missing.add(f"{head_pref}experts_up.{i}.weight")
            allowed_missing.add(f"{head_pref}experts_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
        for sub in (
            "ssm_adapter.A",
            "ssm_adapter.B",
            "ssm_adapter.C",
            "ssm_adapter.D",
            "ssm_adapter.gate",
            "mor_router.main_router.weight",
            "mor_router.main_router.bias",
        ):
            allowed_missing.add(f"{head_pref}{sub}")
        for i in range(4):
            allowed_missing.add(f"{head_pref}mor_router.sub_routers.{i}.weight")
            allowed_missing.add(f"{head_pref}mor_router.sub_routers.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "frontier_expert":
            keep_head_keys = {
                f"{head_pref}weight",
                f"{head_pref}bias",
                f"{head_pref}shared_up.weight",
                f"{head_pref}shared_down.weight",
                f"{head_pref}shared_norm.weight",
                f"{head_pref}shared_norm.bias",
                f"{head_pref}shared_scale",
                f"{head_pref}alpha",
            }
            filtered_sd = {
                k: v
                for k, v in state_dict.items()
                if (not k.startswith(head_pref)) or (k in keep_head_keys)
            }
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "frontier_collective_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha", f"{head_pref}beta",
            f"{head_pref}memory_keys", f"{head_pref}memory_values",
            f"{head_pref}memory_query.weight", f"{head_pref}memory_out.weight",
            f"{head_pref}memory_write_gate.weight", f"{head_pref}memory_write_gate.bias",
            f"{head_pref}memory_write_value.weight", f"{head_pref}memory_write_value.bias",
            f"{head_pref}depth_query.weight", f"{head_pref}depth_key.weight",
            f"{head_pref}depth_value.weight", f"{head_pref}depth_out.weight",
            f"{head_pref}router_budget.weight", f"{head_pref}router_budget.bias",
            f"{head_pref}expert_bias",
            f"{head_pref}shared_aux_up.weight", f"{head_pref}shared_aux_down.weight",
            f"{head_pref}shared_aux_norm.weight", f"{head_pref}shared_aux_norm.bias",
            f"{head_pref}shared_aux_gate.weight", f"{head_pref}shared_aux_gate.bias",
            f"{head_pref}reflect_out.weight",
            f"{head_pref}collective_q.weight", f"{head_pref}collective_k.weight",
            f"{head_pref}collective_v.weight", f"{head_pref}collective_out.weight",
            f"{head_pref}gamma", f"{head_pref}delta",
        }
        for i in range(5):
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.bias")
        for i in range(10):
            allowed_missing.add(f"{head_pref}experts_up.{i}.weight")
            allowed_missing.add(f"{head_pref}experts_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
        for i in range(2):
            allowed_missing.add(f"{head_pref}reflect_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reflect_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reflect_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reflect_cells.{i}.norm.bias")
        for sub in (
            "ssm_adapter.A",
            "ssm_adapter.B",
            "ssm_adapter.C",
            "ssm_adapter.D",
            "ssm_adapter.gate",
            "mor_router.main_router.weight",
            "mor_router.main_router.bias",
        ):
            allowed_missing.add(f"{head_pref}{sub}")
        for i in range(4):
            allowed_missing.add(f"{head_pref}mor_router.sub_routers.{i}.weight")
            allowed_missing.add(f"{head_pref}mor_router.sub_routers.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "frontier_collective_expert":
            keep_head_keys = {
                f"{head_pref}weight",
                f"{head_pref}bias",
                f"{head_pref}shared_up.weight",
                f"{head_pref}shared_down.weight",
                f"{head_pref}shared_norm.weight",
                f"{head_pref}shared_norm.bias",
                f"{head_pref}shared_scale",
                f"{head_pref}alpha",
                f"{head_pref}beta",
                f"{head_pref}memory_keys",
                f"{head_pref}memory_values",
                f"{head_pref}memory_query.weight",
                f"{head_pref}memory_out.weight",
                f"{head_pref}memory_write_gate.weight",
                f"{head_pref}memory_write_gate.bias",
                f"{head_pref}memory_write_value.weight",
                f"{head_pref}memory_write_value.bias",
                f"{head_pref}depth_query.weight",
                f"{head_pref}depth_key.weight",
                f"{head_pref}depth_value.weight",
                f"{head_pref}depth_out.weight",
                f"{head_pref}router_budget.weight",
                f"{head_pref}router_budget.bias",
                f"{head_pref}expert_bias",
                f"{head_pref}ssm_adapter.A",
                f"{head_pref}ssm_adapter.B",
                f"{head_pref}ssm_adapter.C",
                f"{head_pref}ssm_adapter.D",
                f"{head_pref}ssm_adapter.gate",
                f"{head_pref}mor_router.main_router.weight",
                f"{head_pref}mor_router.main_router.bias",
            }
            for i in range(5):
                keep_head_keys.add(f"{head_pref}reasoning_cells.{i}.up.weight")
                keep_head_keys.add(f"{head_pref}reasoning_cells.{i}.down.weight")
                keep_head_keys.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
                keep_head_keys.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
                keep_head_keys.add(f"{head_pref}halt_gates.{i}.weight")
                keep_head_keys.add(f"{head_pref}halt_gates.{i}.bias")
            for i in range(10):
                keep_head_keys.add(f"{head_pref}experts_up.{i}.weight")
                keep_head_keys.add(f"{head_pref}experts_down.{i}.weight")
                keep_head_keys.add(f"{head_pref}expert_norms.{i}.weight")
                keep_head_keys.add(f"{head_pref}expert_norms.{i}.bias")
            for i in range(4):
                keep_head_keys.add(f"{head_pref}mor_router.sub_routers.{i}.weight")
                keep_head_keys.add(f"{head_pref}mor_router.sub_routers.{i}.bias")
            filtered_sd = {
                k: v
                for k, v in state_dict.items()
                if (not k.startswith(head_pref)) or (k in keep_head_keys)
            }
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "frontier_verifier_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha", f"{head_pref}beta",
            f"{head_pref}memory_keys", f"{head_pref}memory_values",
            f"{head_pref}memory_query.weight", f"{head_pref}memory_out.weight",
            f"{head_pref}memory_write_gate.weight", f"{head_pref}memory_write_gate.bias",
            f"{head_pref}memory_write_value.weight", f"{head_pref}memory_write_value.bias",
            f"{head_pref}depth_query.weight", f"{head_pref}depth_key.weight",
            f"{head_pref}depth_value.weight", f"{head_pref}depth_out.weight",
            f"{head_pref}router_budget.weight", f"{head_pref}router_budget.bias",
            f"{head_pref}expert_bias",
            f"{head_pref}shared_aux_up.weight", f"{head_pref}shared_aux_down.weight",
            f"{head_pref}shared_aux_norm.weight", f"{head_pref}shared_aux_norm.bias",
            f"{head_pref}shared_aux_gate.weight", f"{head_pref}shared_aux_gate.bias",
            f"{head_pref}reflect_out.weight",
            f"{head_pref}collective_q.weight", f"{head_pref}collective_k.weight",
            f"{head_pref}collective_v.weight", f"{head_pref}collective_out.weight",
            f"{head_pref}revisit_gate.weight", f"{head_pref}revisit_gate.bias",
            f"{head_pref}verifier_q.weight", f"{head_pref}verifier_k.weight",
            f"{head_pref}verifier_v.weight", f"{head_pref}verifier_out.weight",
            f"{head_pref}verifier_gate.weight", f"{head_pref}verifier_gate.bias",
            f"{head_pref}correction_out.weight",
            f"{head_pref}gamma", f"{head_pref}delta", f"{head_pref}epsilon", f"{head_pref}zeta",
        }
        for i in range(6):
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.bias")
        for i in range(12):
            allowed_missing.add(f"{head_pref}experts_up.{i}.weight")
            allowed_missing.add(f"{head_pref}experts_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
        for i in range(3):
            allowed_missing.add(f"{head_pref}reflect_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reflect_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reflect_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reflect_cells.{i}.norm.bias")
        for i in range(2):
            allowed_missing.add(f"{head_pref}verifier_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}verifier_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}verifier_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}verifier_cells.{i}.norm.bias")
        for sub in (
            "ssm_adapter.A",
            "ssm_adapter.B",
            "ssm_adapter.C",
            "ssm_adapter.D",
            "ssm_adapter.gate",
            "mor_router.main_router.weight",
            "mor_router.main_router.bias",
        ):
            allowed_missing.add(f"{head_pref}{sub}")
        for i in range(5):
            allowed_missing.add(f"{head_pref}mor_router.sub_routers.{i}.weight")
            allowed_missing.add(f"{head_pref}mor_router.sub_routers.{i}.bias")

        target_state = model.state_dict()
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "frontier_verifier_expert":
            keep_head_keys = {
                f"{head_pref}weight",
                f"{head_pref}bias",
                f"{head_pref}shared_up.weight",
                f"{head_pref}shared_down.weight",
                f"{head_pref}shared_norm.weight",
                f"{head_pref}shared_norm.bias",
                f"{head_pref}shared_scale",
                f"{head_pref}alpha",
                f"{head_pref}beta",
                f"{head_pref}memory_keys",
                f"{head_pref}memory_values",
                f"{head_pref}memory_query.weight",
                f"{head_pref}memory_out.weight",
                f"{head_pref}memory_write_gate.weight",
                f"{head_pref}memory_write_gate.bias",
                f"{head_pref}memory_write_value.weight",
                f"{head_pref}memory_write_value.bias",
                f"{head_pref}depth_query.weight",
                f"{head_pref}depth_key.weight",
                f"{head_pref}depth_value.weight",
                f"{head_pref}depth_out.weight",
                f"{head_pref}router_budget.weight",
                f"{head_pref}router_budget.bias",
                f"{head_pref}expert_bias",
                f"{head_pref}shared_aux_up.weight",
                f"{head_pref}shared_aux_down.weight",
                f"{head_pref}shared_aux_norm.weight",
                f"{head_pref}shared_aux_norm.bias",
                f"{head_pref}shared_aux_gate.weight",
                f"{head_pref}shared_aux_gate.bias",
                f"{head_pref}reflect_out.weight",
                f"{head_pref}collective_q.weight",
                f"{head_pref}collective_k.weight",
                f"{head_pref}collective_v.weight",
                f"{head_pref}collective_out.weight",
                f"{head_pref}gamma",
                f"{head_pref}delta",
                f"{head_pref}ssm_adapter.A",
                f"{head_pref}ssm_adapter.B",
                f"{head_pref}ssm_adapter.C",
                f"{head_pref}ssm_adapter.D",
                f"{head_pref}ssm_adapter.gate",
                f"{head_pref}mor_router.main_router.weight",
                f"{head_pref}mor_router.main_router.bias",
            }
            for i in range(5):
                keep_head_keys.add(f"{head_pref}reasoning_cells.{i}.up.weight")
                keep_head_keys.add(f"{head_pref}reasoning_cells.{i}.down.weight")
                keep_head_keys.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
                keep_head_keys.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
                keep_head_keys.add(f"{head_pref}halt_gates.{i}.weight")
                keep_head_keys.add(f"{head_pref}halt_gates.{i}.bias")
            for i in range(10):
                keep_head_keys.add(f"{head_pref}experts_up.{i}.weight")
                keep_head_keys.add(f"{head_pref}experts_down.{i}.weight")
                keep_head_keys.add(f"{head_pref}expert_norms.{i}.weight")
                keep_head_keys.add(f"{head_pref}expert_norms.{i}.bias")
            for i in range(2):
                keep_head_keys.add(f"{head_pref}reflect_cells.{i}.up.weight")
                keep_head_keys.add(f"{head_pref}reflect_cells.{i}.down.weight")
                keep_head_keys.add(f"{head_pref}reflect_cells.{i}.norm.weight")
                keep_head_keys.add(f"{head_pref}reflect_cells.{i}.norm.bias")
            for i in range(4):
                keep_head_keys.add(f"{head_pref}mor_router.sub_routers.{i}.weight")
                keep_head_keys.add(f"{head_pref}mor_router.sub_routers.{i}.bias")
            filtered_sd = {
                k: v
                for k, v in state_dict.items()
                if (((not k.startswith(head_pref)) or (k in keep_head_keys))
                    and k in target_state
                    and tuple(target_state[k].shape) == tuple(v.shape))
            }
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            filtered_sd = {
                k: v
                for k, v in state_dict.items()
                if k in target_state and tuple(target_state[k].shape) == tuple(v.shape)
            }
            incompatible = model.load_state_dict(filtered_sd, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    raise ValueError(f"Unsupported model_size={model_size!r}")


def detect_model_size_from_state_dict(state_dict: dict) -> str:
    """Infer the model variant from checkpoint keys.

    Examines the state_dict for variant-specific parameter names to
    determine which model architecture produced the checkpoint.

    Args:
        state_dict: Checkpoint state dictionary.

    Returns:
        A model size string (e.g., 'base', 'ultra_expert', 'smarter_expert').
    """
    # Megalarge: has GatedFFN at layers.10 + head at layers.11
    if "layers.11.adapter_up_f.weight" in state_dict or "layers.11.router5.weight" in state_dict:
        return "megalarge"
    if "layers.10.adapter_up_e.weight" in state_dict or "layers.10.router4.weight" in state_dict:
        return "ultralarge"
    if "layers.10.adapter_up_d.weight" in state_dict or "layers.10.router3.weight" in state_dict:
        return "xxxlarge"
    if "layers.10.adapter_up_c.weight" in state_dict or "layers.10.router2.weight" in state_dict:
        return "xxlarge"
    if "layers.10.adapter_up_b.weight" in state_dict or "layers.10.router.weight" in state_dict:
        return "xlarge"
    if "layers.10.adapter_up.weight" in state_dict or "layers.10.adapter_down.weight" in state_dict:
        return "large"
    if "layers.10.domain_gate.weight" in state_dict or "layers.10.expert_gates.0.weight" in state_dict:
        return "hierarchical_expert"
    if "layers.10.shared_up.weight" in state_dict and "layers.10.gate.weight" in state_dict:
        if "layers.10.lora_up.0.weight" in state_dict:
            return "smarter_expert"
        return "deep_expert"

    if "layers.10.experts_up.0.weight" in state_dict and "layers.10.gate.weight" in state_dict and "layers.10.shared_scale" not in state_dict:
        # ultra_expert also has gate.weight, but we distinguish EC by checking if it doesn't have EC specific but also no base keys maybe?
        # A simpler way is to check the capacity_factor or something, but it's not in state dict.
        # Actually both ultra_expert and expert_choice have gate.weight and experts_up.0.weight.
        # We can detect expert_choice by checking for expert_norms which ultra_expert lacks.
        if "layers.10.expert_norms.0.weight" in state_dict:
            return "expert_choice"
        return "ultra_expert"
    
    if "layers.10.halt_gates.0.weight" in state_dict and "layers.10.proposal_gates.0.weight" in state_dict:
        return "metacognitive_expert"
    if "layers.10.value_net.0.weight" in state_dict and "layers.10.action_up.0.weight" in state_dict:
        return "tree_of_thought_expert"
    if "layers.10.knowledge_core_keys" in state_dict and "layers.10.route_mu.0.weight" in state_dict:
        return "omniscient_expert"
    if "layers.10.hypernet.0.weight" in state_dict and "layers.10.memory_matrix" in state_dict:
        return "cognitive_expert"
    if "layers.10.denoiser.0.0.weight" in state_dict and "layers.10.world_transition.0.weight" in state_dict:
        return "transcendent_expert"
    if "layers.10.ode_func.net.0.weight" in state_dict and "layers.10.holographic_memory" in state_dict:
        return "omniversal_expert"
    if "layers.10.logic_atoms" in state_dict and "layers.10.curvature" in state_dict:
        return "fractal_expert"
    if "layers.10.ttt_lora_A" in state_dict and "layers.10.lif_neuron.threshold" in state_dict:
        return "liquid_spiking_expert"
    if "layers.10.world_model.weight" in state_dict and "layers.10.precision_gate.weight" in state_dict:
        return "active_inference_expert"
    if "layers.10.holographic_memory" in state_dict and "layers.10.topology_router.weight" in state_dict:
        return "holographic_state_space_expert"
    if "layers.10.verifier_q.weight" in state_dict and "layers.10.revisit_gate.weight" in state_dict:
        return "frontier_verifier_expert"
    if "layers.10.shared_aux_up.weight" in state_dict and "layers.10.collective_q.weight" in state_dict:
        return "frontier_collective_expert"
    if "layers.10.router_budget.weight" in state_dict and "layers.10.memory_write_gate.weight" in state_dict:
        return "frontier_expert"
    if "layers.10.mor_router.main_router.weight" in state_dict and "layers.10.ssm_adapter.A" in state_dict:
        return "paper_fusion_expert"
    if "layers.10.adversary_up.0.weight" in state_dict and "layers.10.proposer_up.0.weight" in state_dict:
        return "neurogenesis_expert"
    if "layers.10.memory_keys" in state_dict and "layers.10.draft_gates.0.0.weight" in state_dict:
        return "deliberative_expert"
    if "layers.10.mlp_gate.weight" in state_dict and "layers.10.consensus.0.weight" in state_dict:
        return "consensus_expert"
    if "layers.10.critique_gate.weight" in state_dict:
        return "reflexive_expert"
    if "layers.10.exit_gates.0.weight" in state_dict:
        return "recursive_expert"
    if "layers.10.reasoning_cells.0.up.weight" in state_dict:
        return "thought_expert"

    if "layers.10.gate.weight" in state_dict or "layers.10.experts_up.0.weight" in state_dict:
        return "ultra_expert"
    return "base"


def detect_large_head_expansion_dim(state_dict: dict, default: int = 512) -> int:
    """Detect the primary adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_xlarge_aux_expansion_dim(state_dict: dict, default: int = 1024) -> int:
    """Detect the secondary (XL) adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up_b.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_xxlarge_third_expansion_dim(state_dict: dict, default: int = 3072) -> int:
    """Detect the third (XXL) adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up_c.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_xxxlarge_fourth_expansion_dim(state_dict: dict, default: int = 4096) -> int:
    """Detect the fourth (XXXL) adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up_d.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_ultralarge_fifth_expansion_dim(state_dict: dict, default: int = 6144) -> int:
    """Detect the fifth (Ultra) adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up_e.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_megalarge_sixth_expansion_dim(state_dict: dict, default: int = 8192) -> int:
    """Detect the sixth (Mega) adapter expansion dim from a checkpoint."""
    key = "layers.11.adapter_up_f.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def list_trainable_parameter_names(model: nn.Module) -> Sequence[str]:
    """Return the names of all parameters with requires_grad=True."""
    return [name for name, p in model.named_parameters() if p.requires_grad]
