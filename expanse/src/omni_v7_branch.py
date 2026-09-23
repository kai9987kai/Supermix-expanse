"""Omni Collective v7 Frontier as a frozen, prompt-conditioned branch of Expanse.

Why a branch and not a merge
    ``omni_collective_v7_frontier`` (77.5M parameters) is not a transformer: it
    reads a prompt four ways at once -- a character-level bidirectional GRU, a
    character CNN, a 16,384-bucket hashed word bag and a 12-number hand-made
    prompt descriptor (plus two image CNNs that see a blank image for text) --
    and fuses them in a 960-d state that nine rounds of routed experts and a
    verifier refine. None of that shares a coordinate system with the 320-d
    Archimedes trunk, so nothing can be weight-averaged across. What *can* be
    shared is what v7 concludes about a prompt, so Expanse keeps the whole net
    frozen and reads its conclusion:

        state = [fused 960 | softmax(intent) 15 | softmax(domain) 13]   (988)

    ``fused`` is the output of ``net.response_refiner`` -- the last state every
    v7 head (intent, response, vision, domain) reads -- captured with a forward
    hook so the v7 code runs unmodified.

Bridge
    ``bridge_norm`` (LayerNorm 988) -> ``to_trunk`` (988 -> H, no bias) -> ``gate``
    (H, zeros), added to every position of the trunk's residual stream after the
    graft layer. The gate starts at exactly zero, so the branch is
    function-preserving at birth (``hidden + 0 * w`` returns ``hidden``'s values).
    The state depends on the *user prompt only*, so writing the same vector at
    every position leaks nothing from the future: it is causal-safe, and a
    KV-cached decode sees the same write as the full forward.

Distillation heads
    ``aux_logits`` (H -> 15 intents, H -> 13 domains) let the trainer distil v7's
    intent/domain judgement into the trunk's own hidden state (KL to cached v7
    distributions at the last prompt position), so the knowledge survives even
    where the branch gate stays small.

Exactness
    ``featurize`` + ``encode`` reproduce ``OmniCollectiveEngineV7._run_prompt``
    (the single forward the engine's multi-pass deliberation is built from):
    same char vocabulary, ``max_len`` 384, 84 hashed words, 12 prompt features, a
    zero 128x128 image with ``has_image`` 0. The two image CNNs cost ~6.5 GFLOP
    per call and see the same blank image for every text prompt, so ``encode``
    runs them once on one blank image (exactly the engine's B=1 computation) and
    reuses that output for the whole batch; real images fall back to the normal
    path. ``tests/test_omni_v7_branch.py`` checks the logits against the engine.

The v7 python sources live in ``EXP/external/omni_v7`` (need sympy + Pillow);
``EXPANSE_OMNI_V7_DIR`` overrides the location.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
OMNI_V7_DIR = Path(os.environ.get("EXPANSE_OMNI_V7_DIR", str(EXP / "external" / "omni_v7")))
if str(OMNI_V7_DIR) not in sys.path:
    sys.path.insert(0, str(OMNI_V7_DIR))

from image_recognition_model import SCIENCE_IMAGE_CLASSES  # noqa: E402
from omni_collective_model import (  # noqa: E402
    OMNI_DOMAIN_LABELS_V2,
    OMNI_INTENTS_V2,
    encode_text,
    encode_word_hashes,
    prompt_feature_vector,
)
from omni_collective_v4_model import OmniCollectiveNetV4  # noqa: E402  (OmniCollectiveNetV7 is this class)

#: Size of v7's response bank (``response_head`` rows). The bank itself (5.2 MB
#: of text) is not part of the small meta Expanse carries, so the count travels
#: as ``meta["num_responses"]``; this is the fallback for the one v7 checkpoint.
V7_NUM_RESPONSES = 22456
META_FILE = "omni_collective_v7_frontier_meta.json"
WEIGHTS_FILE = "omni_collective_v7_frontier.pth"


def load_v7_meta(meta_path=None) -> Dict[str, Any]:
    """The small meta ``OmniV7Branch`` wants: every key except ``response_bank``,
    plus ``num_responses`` (its length) so the net can be rebuilt without it."""

    path = Path(meta_path) if meta_path is not None else OMNI_V7_DIR / META_FILE
    full = json.loads(path.read_text(encoding="utf-8"))
    small = {k: v for k, v in full.items() if k != "response_bank"}
    small["num_responses"] = len(full.get("response_bank") or []) or V7_NUM_RESPONSES
    return small


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ConstantImageEncoder(nn.Module):
    """Stand-in for one of v7's image CNNs while ``encode`` runs: returns the
    precomputed output for the blank image, one row per prompt."""

    def __init__(self, row: torch.Tensor):
        super().__init__()
        self.row = row  # (1, C), plain attribute: never part of a state dict

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.row.expand(image_tensor.shape[0], -1)


class OmniV7Branch(nn.Module):
    """Frozen Omni Collective v7 net + trainable bridge into the trunk."""

    def __init__(self, hidden_size: int, meta: Dict[str, Any]):
        super().__init__()
        meta = {k: v for k, v in dict(meta).items() if k != "response_bank"}
        self.hidden_size = int(hidden_size)
        # Everything featurize() needs, resolved exactly as OmniCollectiveEngineV7.__init__ does.
        self.vocab = {str(k): int(v) for k, v in dict(meta.get("vocab") or {}).items()}
        self.intent_labels = tuple(str(x) for x in list(meta.get("intent_labels") or OMNI_INTENTS_V2))
        self.domain_labels = tuple(str(x) for x in list(meta.get("domain_labels") or OMNI_DOMAIN_LABELS_V2))
        self.image_size = int(meta.get("image_size") or 128)
        self.max_len = int(meta.get("max_len") or 360)
        self.word_buckets = int(meta.get("word_buckets") or 16384)
        self.max_words = int(meta.get("max_words") or 84)
        self.num_responses = int(meta.get("num_responses") or V7_NUM_RESPONSES)
        meta["num_responses"] = self.num_responses
        self.meta = meta

        self.net = OmniCollectiveNetV4(
            vocab_size=max(len(self.vocab), 2),
            num_intents=len(self.intent_labels),
            num_responses=max(self.num_responses, 1),
            num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
            num_domains=max(len(self.domain_labels), 1),
            base_embed_dim=int(meta.get("embed_dim") or 116),
            text_hidden=int(meta.get("text_hidden") or 236),
            image_channels=int(meta.get("image_channels") or 48),
            word_buckets=int(meta.get("word_buckets") or 16384),
            word_embed_dim=int(meta.get("word_embed_dim") or 108),
            deep_text_channels=int(meta.get("deep_text_channels") or 336),
            deep_image_channels=int(meta.get("deep_image_channels") or 112),
            fusion_hidden=int(meta.get("fusion_hidden") or 960),
            memory_slots=int(meta.get("memory_slots") or 22),
            depth_steps=int(meta.get("depth_steps") or 9),
            expert_count=int(meta.get("expert_count") or 8),
            expert_hidden=int(meta.get("expert_hidden") or 1536),
            context_top_k=int(meta.get("context_top_k") or 4),
            expert_top_k=int(meta.get("expert_top_k") or 2),
        )
        self.fusion_hidden = int(meta.get("fusion_hidden") or 960)
        self.n_intents, self.n_domains = len(self.intent_labels), len(self.domain_labels)
        self.state_dim = self.fusion_hidden + self.n_intents + self.n_domains  # 988 for v7
        # The v7 net is frozen for good: no gradients, always eval (dropout off,
        # BatchNorm on running stats). The trainer never sees its parameters.
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self._fused: Optional[torch.Tensor] = None
        self.net.response_refiner.register_forward_hook(self._capture_fused)

        # bridge (trainable)
        self.bridge_norm = nn.LayerNorm(self.state_dim)
        self.to_trunk = nn.Linear(self.state_dim, self.hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(self.hidden_size))
        # v7 intent/domain distillation heads read the trunk's hidden state
        self.intent_head = nn.Linear(self.hidden_size, self.n_intents)
        self.domain_head = nn.Linear(self.hidden_size, self.n_domains)

    # -- loading -------------------------------------------------------------
    def _capture_fused(self, module, inputs, output):
        self._fused = output

    @torch.no_grad()
    def load_v7(self, pth_path=None) -> Dict[str, Any]:
        """Strict-load the v7 frontier weights into ``self.net`` and freeze it."""

        path = Path(pth_path) if pth_path is not None else OMNI_V7_DIR / WEIGHTS_FILE
        state = torch.load(path, map_location="cpu", weights_only=False)
        n_resp = int(state["response_head.weight"].shape[0])
        if n_resp != self.num_responses:
            raise ValueError(f"v7 checkpoint has {n_resp} responses, branch was built for {self.num_responses}; "
                             "pass meta from load_v7_meta() so num_responses is right")
        device = next(self.net.parameters()).device
        self.net.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        return {
            "path": str(path), "sha256": _sha256(path), "tensors": len(state),
            "params": int(sum(v.numel() for v in state.values() if v.is_floating_point())),
            "num_responses": n_resp, "state_dim": self.state_dim,
            "intent_labels": list(self.intent_labels), "domain_labels": list(self.domain_labels),
        }

    def train(self, mode: bool = True) -> "OmniV7Branch":
        super().train(mode)
        self.net.eval()  # the frozen net never leaves eval
        return self

    # -- prompt -> v7 state ----------------------------------------------------
    def featurize(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Engine-identical inputs for a batch of prompts (text only: blank image)."""

        device = self.gate.device
        prompts = [str(p or "") for p in prompts]
        size = int(self.image_size)
        return {
            "token_ids": torch.tensor([encode_text(p, self.vocab, self.max_len) for p in prompts], dtype=torch.long, device=device),
            "word_ids": torch.tensor(
                [encode_word_hashes(p, buckets=self.word_buckets, max_words=self.max_words) for p in prompts],
                dtype=torch.long, device=device),
            "prompt_features": torch.tensor([prompt_feature_vector(p) for p in prompts], dtype=torch.float32, device=device),
            "image_tensor": torch.zeros((len(prompts), 3, size, size), dtype=torch.float32, device=device),
            "has_image": torch.zeros((len(prompts),), dtype=torch.float32, device=device),
        }

    def _run_net(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        net = self.net
        image, has_image = feats["image_tensor"], feats["has_image"]
        blank = not bool(has_image.any()) and not bool(image.any())
        if not blank:
            return net(feats["token_ids"], image, has_image, feats["word_ids"], feats["prompt_features"])
        # Text-only batch: every row sees the same zero image, so run each image
        # CNN once on one blank image (the engine's own B=1 computation) and
        # hand that row to every prompt.
        one = image[:1]
        enc, deep = net.image_encoder, net.deep_image_encoder
        stub_enc, stub_deep = _ConstantImageEncoder(enc(one)), _ConstantImageEncoder(deep(one))
        try:
            net._modules["image_encoder"] = stub_enc
            net._modules["deep_image_encoder"] = stub_deep
            return net(feats["token_ids"], image, has_image, feats["word_ids"], feats["prompt_features"])
        finally:
            net._modules["image_encoder"] = enc
            net._modules["deep_image_encoder"] = deep

    @torch.no_grad()
    def encode(self, feats: Dict[str, torch.Tensor], return_logits: bool = False):
        """(B, 988) = [fused 960 | softmax(intent) 15 | softmax(domain) 13].

        ``return_logits=True`` also returns the raw v7 head outputs (intent,
        domain, vision, response) for distillation caches and tests."""

        self.net.eval()
        self._fused = None
        out = self._run_net(feats)
        fused = self._fused
        self._fused = None
        if fused is None:
            raise RuntimeError("response_refiner hook did not fire")
        state = torch.cat([fused, torch.softmax(out["intent"], dim=-1), torch.softmax(out["domain"], dim=-1)], dim=-1)
        if return_logits:
            return state, {k: out[k] for k in ("intent", "domain", "vision", "response")}
        return state

    def encode_prompts(self, prompts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        """Convenience: featurize + encode in batches."""

        rows = []
        for i in range(0, len(prompts), batch_size):
            rows.append(self.encode(self.featurize(prompts[i:i + batch_size])))
        return torch.cat(rows, dim=0) if rows else torch.zeros(0, self.state_dim)

    # -- trunk side ------------------------------------------------------------
    def forward(self, hidden: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """hidden (B, T, H), state (B, 988) -> hidden + gate * to_trunk(bridge_norm(state)) at every position."""

        write = self.gate * self.to_trunk(self.bridge_norm(state.to(hidden.dtype)))
        out = hidden + write.unsqueeze(1)
        return out, {"write_rms": write.detach().pow(2).mean().sqrt(),
                     "intent_probs": state[:, self.fusion_hidden:self.fusion_hidden + self.n_intents].detach(),
                     "domain_probs": state[:, self.fusion_hidden + self.n_intents:].detach()}

    def aux_logits(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trunk hidden (B, H) -> (intent logits (B, 15), domain logits (B, 13)) for v7 distillation."""

        return self.intent_head(h), self.domain_head(h)

    @staticmethod
    def split_state(state: torch.Tensor, fusion_hidden: int = 960, n_intents: int = 15) -> Dict[str, torch.Tensor]:
        """Views of a (B, 988) state: fused, intent probs, domain probs."""

        return {"fused": state[:, :fusion_hidden], "intent": state[:, fusion_hidden:fusion_hidden + n_intents],
                "domain": state[:, fusion_hidden + n_intents:]}


__all__ = ["OmniV7Branch", "load_v7_meta", "OMNI_V7_DIR", "V7_NUM_RESPONSES"]
