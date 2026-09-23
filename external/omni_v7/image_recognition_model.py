from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image

try:
    from image_feature_utils import describe_image_for_text
except ImportError:  # pragma: no cover
    from .image_feature_utils import describe_image_for_text


SCIENCE_IMAGE_CLASSES: tuple[str, ...] = (
    "cell",
    "magnet",
    "circuit",
    "water_cycle",
    "moon_phases",
    "photosynthesis",
    "solar_system",
    "dna",
    "temperature",
    "states_of_matter",
    "light_dispersion",
)

CLASS_LABELS: Dict[str, str] = {
    "cell": "cell diagram",
    "magnet": "magnet poles",
    "circuit": "simple circuit",
    "water_cycle": "water cycle",
    "moon_phases": "moon phases",
    "photosynthesis": "photosynthesis",
    "solar_system": "solar system",
    "dna": "DNA",
    "temperature": "temperature / thermometer",
    "states_of_matter": "states of matter",
    "light_dispersion": "light dispersion",
}

VISION_PROMPT_RE = re.compile(
    r"\b(what is in this image|what does this image show|analyze this image|recognize|identify|"
    r"uploaded image|look at the image|describe the image|visual clues|what concept)\b",
    re.IGNORECASE,
)


@dataclass
class VisionPrediction:
    concept_key: str
    label: str
    confidence: float
    top_predictions: List[Dict[str, float]]
    descriptor: str
    caption: str
    tags: str


class ScienceImageRecognitionNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 24, hidden_dim: int = 96) -> None:
        super().__init__()
        channels = int(base_channels)
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 2, channels * 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 3, channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.18),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image_tensor))


def load_image_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR)
        buffer = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = buffer.view(int(image_size), int(image_size), 3).permute(2, 0, 1).float() / 255.0
    return tensor


def looks_like_vision_prompt(prompt: str) -> bool:
    return bool(VISION_PROMPT_RE.search(str(prompt or "")))


class ScienceImageRecognitionEngine:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.image_size = int(self.meta.get("image_size") or 96)
        self.classes = tuple(str(item) for item in (self.meta.get("classes") or SCIENCE_IMAGE_CLASSES))
        self.class_info = {
            str(key): dict(value)
            for key, value in dict(self.meta.get("class_info") or {}).items()
        }
        base_channels = int(self.meta.get("base_channels") or 24)
        hidden_dim = int(self.meta.get("hidden_dim") or 96)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScienceImageRecognitionNet(
            num_classes=len(self.classes),
            base_channels=base_channels,
            hidden_dim=hidden_dim,
        ).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def status(self) -> Dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "image_size": self.image_size,
            "classes": list(self.classes),
            "device": str(self.device),
        }

    def predict(self, image_path: str | Path, top_k: int = 3) -> VisionPrediction:
        tensor = load_image_tensor(image_path, self.image_size).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
        top_count = max(1, min(int(top_k), len(self.classes)))
        values, indices = torch.topk(probs, k=top_count)
        concept_key = self.classes[int(indices[0].item())]
        info = self.class_info.get(concept_key, {})
        descriptor = describe_image_for_text(str(image_path))
        top_predictions = [
            {
                "concept": self.classes[int(index.item())],
                "confidence": round(float(value.item()), 4),
            }
            for value, index in zip(values, indices)
        ]
        return VisionPrediction(
            concept_key=concept_key,
            label=CLASS_LABELS.get(concept_key, concept_key.replace("_", " ")),
            confidence=float(values[0].item()),
            top_predictions=top_predictions,
            descriptor=descriptor,
            caption=str(info.get("caption") or "").strip(),
            tags=str(info.get("tags") or "").strip(),
        )

    def answer(self, prompt: str, image_path: str | Path) -> str:
        prediction = self.predict(image_path)
        prompt_text = str(prompt or "").strip().lower()
        lines: List[str] = []
        lines.append(
            f"Best match: {prediction.label} ({prediction.confidence * 100:.1f}% confidence)."
        )
        if "clue" in prompt_text or "why" in prompt_text or "evidence" in prompt_text:
            if prediction.caption:
                lines.append(f"Visual clues: {prediction.caption}.")
            lines.append(f"Image descriptor: {prediction.descriptor}.")
        elif "tag" in prompt_text:
            lines.append(f"Tags: {prediction.tags or prediction.concept_key}.")
        elif "brief" in prompt_text or "short" in prompt_text:
            return lines[0]
        else:
            if prediction.caption:
                lines.append(f"It most likely shows {prediction.caption}.")
            if prediction.tags:
                lines.append(f"Topic tags: {prediction.tags}.")
        alternatives = [
            f"{CLASS_LABELS.get(item['concept'], item['concept'].replace('_', ' '))} {item['confidence'] * 100:.1f}%"
            for item in prediction.top_predictions[1:]
        ]
        if alternatives:
            lines.append("Other plausible matches: " + ", ".join(alternatives) + ".")
        return "\n".join(line for line in lines if line.strip())
