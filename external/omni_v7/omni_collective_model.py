from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

try:
    from image_feature_utils import describe_image_for_text
    from image_recognition_model import CLASS_LABELS, SCIENCE_IMAGE_CLASSES, load_image_tensor
    from math_equation_model import heuristic_intent, solve_intent
except ImportError:  # pragma: no cover
    from .image_feature_utils import describe_image_for_text
    from .image_recognition_model import CLASS_LABELS, SCIENCE_IMAGE_CLASSES, load_image_tensor
    from .math_equation_model import heuristic_intent, solve_intent


OMNI_INTENTS_V1: tuple[str, ...] = (
    "general",
    "coding",
    "creative",
    "comparison",
    "planning",
    "current_info",
    "model_selection",
    "math",
    "vision",
    "command",
    "image_prompt",
)

OMNI_INTENTS_V2: tuple[str, ...] = OMNI_INTENTS_V1 + (
    "language",
    "spatial_3d",
    "video",
    "knowledge",
)

# Keep these aliases for backward compatibility with the original v1 trainer.
OMNI_INTENTS = OMNI_INTENTS_V1
INTENT_TO_INDEX = {name: idx for idx, name in enumerate(OMNI_INTENTS)}
INTENT_TO_INDEX_V2 = {name: idx for idx, name in enumerate(OMNI_INTENTS_V2)}

OMNI_DOMAIN_LABELS_V2: tuple[str, ...] = (
    "general",
    "coding",
    "creative",
    "planning",
    "knowledge",
    "language",
    "math",
    "vision",
    "image_prompt",
    "spatial_3d",
    "video",
    "tool",
    "model_selection",
)
DOMAIN_TO_INDEX_V2 = {name: idx for idx, name in enumerate(OMNI_DOMAIN_LABELS_V2)}

WORD_RE = re.compile(r"[A-Za-z0-9_:+#./-]{2,}")
PROMPT_FEATURE_DIM = 12


def build_char_vocab(texts: Sequence[str], *, min_frequency: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for ch in str(text or "").lower():
            counts[ch] = counts.get(ch, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, count in sorted(counts.items()):
        if count >= min_frequency and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    cooked = str(text or "").strip().lower()
    ids = [vocab.get(ch, 1) for ch in cooked[: int(max_len)]]
    if len(ids) < int(max_len):
        ids.extend([0] * (int(max_len) - len(ids)))
    return ids


def encode_word_hashes(text: str, *, buckets: int, max_words: int) -> List[int]:
    words = WORD_RE.findall(str(text or "").lower())
    encoded: List[int] = []
    bucket_count = max(int(buckets), 8)
    for word in words[: int(max_words)]:
        digest = hashlib.sha1(word.encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], "big", signed=False)
        encoded.append(1 + (value % bucket_count))
    if len(encoded) < int(max_words):
        encoded.extend([0] * (int(max_words) - len(encoded)))
    return encoded


def prompt_feature_vector(text: str) -> List[float]:
    cooked = str(text or "").strip()
    lowered = cooked.lower()
    length = max(len(cooked), 1)
    digits = sum(ch.isdigit() for ch in cooked) / length
    uppercase = sum(ch.isupper() for ch in cooked) / length
    punctuation = sum(ch in ".,:;!?()[]{}" for ch in cooked) / length
    newlines = cooked.count("\n") / max(length, 1)
    non_ascii = sum(ord(ch) > 127 for ch in cooked) / length
    code_markers = float(any(token in lowered for token in ("python", "javascript", "traceback", "stack trace", "regex", "sql", "api", "function", "bug", "debug", "class ", "def ")))
    math_markers = float(any(token in lowered for token in ("solve", "integrate", "derivative", "equation", "factor", "simplify")) or bool(re.search(r"[\d\)\]][\+\-\*/\^]", cooked)))
    vision_markers = float(any(token in lowered for token in ("image", "photo", "diagram", "uploaded", "recognize", "recognise", "identify", "visual", "picture")))
    image_prompt_markers = float(any(token in lowered for token in ("generate", "create", "render", "draw", "illustration", "poster", "wallpaper", "thumbnail", "prompt")))
    spatial_markers = float(any(token in lowered for token in ("3d", "mesh", "model", "cube", "pyramid", "spatial", "distance", "orbit", "video")))
    planning_markers = float(any(token in lowered for token in ("plan", "steps", "roadmap", "tradeoff", "compare", "strategy")))
    current_markers = float(any(token in lowered for token in ("latest", "current", "today", "recent", "news", "web")))
    return [
        math.log1p(length) / 8.0,
        digits,
        uppercase,
        punctuation,
        newlines,
        non_ascii,
        code_markers,
        math_markers,
        vision_markers,
        image_prompt_markers,
        spatial_markers + planning_markers,
        current_markers,
    ]


class OmniCollectiveNet(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_intents: int,
        num_responses: int,
        num_vision_classes: int,
        embed_dim: int = 56,
        text_hidden: int = 96,
        image_channels: int = 24,
        fusion_hidden: int = 176,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_encoder = nn.GRU(embed_dim, text_hidden, batch_first=True, bidirectional=True)
        self.text_norm = nn.LayerNorm(text_hidden * 2)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, image_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(image_channels, image_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(image_channels * 2, image_channels * 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_channels * 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.image_norm = nn.LayerNorm(image_channels * 3)
        self.fusion = nn.Sequential(
            nn.Linear(text_hidden * 2 + image_channels * 3 + 1, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.18),
        )
        self.intent_head = nn.Linear(fusion_hidden, num_intents)
        self.response_head = nn.Linear(fusion_hidden, num_responses)
        self.vision_head = nn.Linear(fusion_hidden, num_vision_classes)

    def forward(self, token_ids: torch.Tensor, image_tensor: torch.Tensor, has_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(token_ids)
        _, hidden = self.text_encoder(embedded)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if hidden.dim() == 3:
            text_state = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            text_state = hidden
        text_state = self.text_norm(text_state)
        image_state = self.image_norm(self.image_encoder(image_tensor))
        fused = self.fusion(torch.cat([text_state, image_state, has_image.view(-1, 1)], dim=1))
        return {
            "intent": self.intent_head(fused),
            "response": self.response_head(fused),
            "vision": self.vision_head(fused),
        }


class SqueezeExcite2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, int(channels) // max(1, int(reduction)))
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.layers(x).unsqueeze(-1).unsqueeze(-1)
        return x * gate


class ResidualFusionBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        delta = self.layers(h)
        return x + delta * torch.sigmoid(self.gate(h))


class OmniCollectiveNetV2(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_intents: int,
        num_responses: int,
        num_vision_classes: int,
        num_domains: int,
        base_embed_dim: int = 56,
        text_hidden: int = 96,
        image_channels: int = 24,
        word_buckets: int = 4096,
        word_embed_dim: int = 48,
        deep_text_channels: int = 96,
        deep_image_channels: int = 40,
        fusion_hidden: int = 320,
    ) -> None:
        super().__init__()
        self.word_buckets = int(word_buckets)
        self.embedding = nn.Embedding(vocab_size, base_embed_dim, padding_idx=0)
        self.text_encoder = nn.GRU(base_embed_dim, text_hidden, batch_first=True, bidirectional=True)
        self.text_norm = nn.LayerNorm(text_hidden * 2)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, image_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(image_channels, image_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(image_channels * 2, image_channels * 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_channels * 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.image_norm = nn.LayerNorm(image_channels * 3)

        self.word_embedding = nn.Embedding(self.word_buckets + 1, word_embed_dim, padding_idx=0)
        self.word_norm = nn.LayerNorm(word_embed_dim)
        self.char_conv = nn.Sequential(
            nn.Conv1d(base_embed_dim, deep_text_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(deep_text_channels, deep_text_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.char_conv_norm = nn.LayerNorm(deep_text_channels)
        self.prompt_feature_proj = nn.Sequential(
            nn.Linear(PROMPT_FEATURE_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, 24),
        )
        self.prompt_feature_norm = nn.LayerNorm(24)

        deep_c = int(deep_image_channels)
        self.deep_image_encoder = nn.Sequential(
            nn.Conv2d(3, deep_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(deep_c),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(deep_c, deep_c * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(deep_c * 2),
            nn.SiLU(),
            SqueezeExcite2d(deep_c * 2),
            nn.MaxPool2d(2),
            nn.Conv2d(deep_c * 2, deep_c * 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(deep_c * 3),
            nn.SiLU(),
            SqueezeExcite2d(deep_c * 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.deep_image_norm = nn.LayerNorm(deep_c * 3)

        fusion_input_dim = (text_hidden * 2) + (image_channels * 3) + deep_text_channels + word_embed_dim + (deep_c * 3) + 24 + 1
        self.fusion_in = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.SiLU(),
            nn.Dropout(p=0.12),
        )
        self.depth_router = nn.Linear(fusion_hidden, 3)
        self.depth_blocks = nn.ModuleList(
            [ResidualFusionBlock(fusion_hidden, fusion_hidden * 2, dropout=0.1) for _ in range(3)]
        )
        self.intent_head = nn.Linear(fusion_hidden, num_intents)
        self.response_head = nn.Linear(fusion_hidden, num_responses)
        self.vision_head = nn.Linear(fusion_hidden, num_vision_classes)
        self.domain_head = nn.Linear(fusion_hidden, num_domains)

    @staticmethod
    def _sequence_average(embedded: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        weights = padding_mask.unsqueeze(-1).float()
        summed = (embedded * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(
        self,
        token_ids: torch.Tensor,
        image_tensor: torch.Tensor,
        has_image: torch.Tensor,
        word_ids: torch.Tensor,
        prompt_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(token_ids)
        _, hidden = self.text_encoder(embedded)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if hidden.dim() == 3:
            text_state = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            text_state = hidden
        text_state = self.text_norm(text_state)

        image_state = self.image_norm(self.image_encoder(image_tensor))
        deep_image_state = self.deep_image_norm(self.deep_image_encoder(image_tensor))

        char_state = self.char_conv_norm(self.char_conv(embedded.transpose(1, 2)))
        word_embedded = self.word_embedding(word_ids)
        word_mask = word_ids.ne(0)
        word_state = self.word_norm(self._sequence_average(word_embedded, word_mask))

        prompt_state = self.prompt_feature_norm(self.prompt_feature_proj(prompt_features))
        fused = self.fusion_in(
            torch.cat(
                [
                    text_state,
                    image_state,
                    char_state,
                    word_state,
                    deep_image_state,
                    prompt_state,
                    has_image.view(-1, 1),
                ],
                dim=1,
            )
        )
        route = torch.softmax(self.depth_router(fused), dim=1)
        for idx, block in enumerate(self.depth_blocks):
            candidate = block(fused)
            fused = fused + route[:, idx : idx + 1] * (candidate - fused)
        return {
            "intent": self.intent_head(fused),
            "response": self.response_head(fused),
            "vision": self.vision_head(fused),
            "domain": self.domain_head(fused),
        }


@dataclass
class OmniPrediction:
    intent: str
    response_text: str
    vision_concept: str
    vision_confidence: float
    top_vision: List[Dict[str, float]]
    domain: str = "general"
    domain_confidence: float = 0.0


class OmniCollectiveEngine:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.vocab = {str(key): int(value) for key, value in dict(self.meta.get("vocab") or {}).items()}
        self.responses = tuple(str(item) for item in list(self.meta.get("response_bank") or []))
        self.class_info = {str(key): dict(value) for key, value in dict(self.meta.get("class_info") or {}).items()}
        self.image_size = int(self.meta.get("image_size") or 96)
        self.max_len = int(self.meta.get("max_len") or 220)
        self.architecture_version = int(self.meta.get("architecture_version") or 1)
        self.intent_labels = tuple(
            str(item) for item in list(self.meta.get("intent_labels") or (OMNI_INTENTS_V2 if self.architecture_version >= 2 else OMNI_INTENTS_V1))
        )
        self.domain_labels = tuple(
            str(item) for item in list(self.meta.get("domain_labels") or (OMNI_DOMAIN_LABELS_V2 if self.architecture_version >= 2 else ("general",)))
        )
        self.word_buckets = int(self.meta.get("word_buckets") or 4096)
        self.max_words = int(self.meta.get("max_words") or 48)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.architecture_version >= 2:
            self.model = OmniCollectiveNetV2(
                vocab_size=max(len(self.vocab), 2),
                num_intents=len(self.intent_labels),
                num_responses=max(len(self.responses), 1),
                num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
                num_domains=max(len(self.domain_labels), 1),
                base_embed_dim=int(self.meta.get("embed_dim") or 56),
                text_hidden=int(self.meta.get("text_hidden") or 96),
                image_channels=int(self.meta.get("image_channels") or 24),
                word_buckets=int(self.meta.get("word_buckets") or 4096),
                word_embed_dim=int(self.meta.get("word_embed_dim") or 48),
                deep_text_channels=int(self.meta.get("deep_text_channels") or 96),
                deep_image_channels=int(self.meta.get("deep_image_channels") or 40),
                fusion_hidden=int(self.meta.get("fusion_hidden") or 320),
            ).to(self.device)
        else:
            self.model = OmniCollectiveNet(
                vocab_size=max(len(self.vocab), 2),
                num_intents=len(self.intent_labels),
                num_responses=max(len(self.responses), 1),
                num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
                embed_dim=int(self.meta.get("embed_dim") or 56),
                text_hidden=int(self.meta.get("text_hidden") or 96),
                image_channels=int(self.meta.get("image_channels") or 24),
                fusion_hidden=int(self.meta.get("fusion_hidden") or 176),
            ).to(self.device)

        state = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def status(self) -> Dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "responses": len(self.responses),
            "image_size": self.image_size,
            "device": str(self.device),
            "architecture_version": self.architecture_version,
            "intent_labels": list(self.intent_labels),
            "domain_labels": list(self.domain_labels),
        }

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        payload = {
            "token_ids": torch.tensor([encode_text(prompt, self.vocab, self.max_len)], dtype=torch.long, device=self.device),
        }
        if self.architecture_version >= 2:
            payload["word_ids"] = torch.tensor(
                [encode_word_hashes(prompt, buckets=self.word_buckets, max_words=self.max_words)],
                dtype=torch.long,
                device=self.device,
            )
            payload["prompt_features"] = torch.tensor(
                [prompt_feature_vector(prompt)],
                dtype=torch.float32,
                device=self.device,
            )
        return payload

    def _encode_image(self, image_path: Optional[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not image_path:
            size = int(self.image_size)
            return (
                torch.zeros((1, 3, size, size), dtype=torch.float32, device=self.device),
                torch.zeros((1,), dtype=torch.float32, device=self.device),
            )
        tensor = load_image_tensor(str(image_path), self.image_size).unsqueeze(0).to(self.device)
        return tensor, torch.ones((1,), dtype=torch.float32, device=self.device)

    def predict(self, prompt: str, image_path: Optional[str] = None) -> OmniPrediction:
        prompt_payload = self._encode_prompt(prompt)
        image_tensor, has_image = self._encode_image(image_path)
        with torch.inference_mode():
            if self.architecture_version >= 2:
                outputs = self.model(
                    prompt_payload["token_ids"],
                    image_tensor,
                    has_image,
                    prompt_payload["word_ids"],
                    prompt_payload["prompt_features"],
                )
            else:
                outputs = self.model(prompt_payload["token_ids"], image_tensor, has_image)
            intent_idx = int(outputs["intent"][0].argmax(dim=0).item())
            response_idx = int(outputs["response"][0].argmax(dim=0).item()) if self.responses else 0
            vision_probs = torch.softmax(outputs["vision"][0], dim=0).detach().cpu()
            domain_probs = (
                torch.softmax(outputs["domain"][0], dim=0).detach().cpu()
                if "domain" in outputs
                else torch.ones((1,), dtype=torch.float32)
            )
        values, indices = torch.topk(vision_probs, k=min(3, len(SCIENCE_IMAGE_CLASSES)))
        vision_key = SCIENCE_IMAGE_CLASSES[int(indices[0].item())]
        top_vision = [
            {
                "concept": SCIENCE_IMAGE_CLASSES[int(index.item())],
                "confidence": round(float(value.item()), 4),
            }
            for value, index in zip(values, indices)
        ]
        domain_idx = int(domain_probs.argmax(dim=0).item()) if domain_probs.numel() else 0
        domain_name = self.domain_labels[min(domain_idx, len(self.domain_labels) - 1)] if self.domain_labels else "general"
        return OmniPrediction(
            intent=self.intent_labels[min(intent_idx, len(self.intent_labels) - 1)] if self.intent_labels else "general",
            response_text=self.responses[response_idx] if self.responses else "",
            vision_concept=vision_key,
            vision_confidence=float(values[0].item()),
            top_vision=top_vision,
            domain=domain_name,
            domain_confidence=float(domain_probs[domain_idx].item()) if domain_probs.numel() else 0.0,
        )

    def answer(self, prompt: str, image_path: Optional[str] = None) -> str:
        prompt_text = str(prompt or "").strip()
        math_intent = heuristic_intent(prompt_text)
        prediction = self.predict(prompt_text, image_path=image_path)
        if math_intent:
            try:
                solved = solve_intent(prompt_text, math_intent)
                return str(solved.get("response") or prediction.response_text or "Math intent detected.")
            except Exception:
                return prediction.response_text or "This looks like a math problem, but I could not solve it cleanly."
        if image_path:
            info = self.class_info.get(prediction.vision_concept, {})
            descriptor = describe_image_for_text(str(image_path))
            lines = [
                f"Image read: {CLASS_LABELS.get(prediction.vision_concept, prediction.vision_concept.replace('_', ' '))} ({prediction.vision_confidence * 100:.1f}% confidence).",
            ]
            if str(info.get("caption") or "").strip():
                lines.append(f"Likely concept: {str(info.get('caption')).strip()}.")
            lines.append(f"Visual descriptor: {descriptor}.")
            alt = [
                f"{CLASS_LABELS.get(item['concept'], item['concept'].replace('_', ' '))} {item['confidence'] * 100:.1f}%"
                for item in prediction.top_vision[1:]
            ]
            if alt:
                lines.append("Other matches: " + ", ".join(alt) + ".")
            if prediction.response_text:
                lines.append(prediction.response_text)
            return "\n".join(lines)
        if prediction.intent == "current_info":
            return prediction.response_text or "Turn on Web Search when you need current external information."
        if prediction.intent == "command":
            return prediction.response_text or "I can ask the runtime to open Command Prompt when you explicitly request it."
        return prediction.response_text or "I did not have a strong fused answer for that prompt."
