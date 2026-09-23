from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    from image_feature_utils import describe_image_for_text
    from image_recognition_model import CLASS_LABELS, SCIENCE_IMAGE_CLASSES, load_image_tensor
    from math_equation_model import heuristic_intent, solve_intent
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, encode_text, encode_word_hashes, prompt_feature_vector
    from omni_collective_v4_model import OmniCollectiveNetV4
except ImportError:  # pragma: no cover
    from .image_feature_utils import describe_image_for_text
    from .image_recognition_model import CLASS_LABELS, SCIENCE_IMAGE_CLASSES, load_image_tensor
    from .math_equation_model import heuristic_intent, solve_intent
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, encode_text, encode_word_hashes, prompt_feature_vector
    from .omni_collective_v4_model import OmniCollectiveNetV4


OmniCollectiveNetV5 = OmniCollectiveNetV4


def _task_focus(prompt: str) -> str:
    lowered = str(prompt or "").lower()
    if "openscad" in lowered or ".scad" in lowered or "module(" in lowered or "difference()" in lowered:
        return "openscad"
    coding_markers = (
        "python",
        "javascript",
        "typescript",
        "traceback",
        "stack trace",
        "exception",
        "function",
        "class ",
        "sql",
        "regex",
        "bash",
        "powershell",
        "git ",
        "bug",
        "refactor",
        "code",
        "algorithm",
        "openscad",
    )
    if any(marker in lowered for marker in coding_markers):
        return "coding"
    if any(marker in lowered for marker in ("why", "compare", "tradeoff", "reason", "plan", "steps")):
        return "reasoning"
    return "general"


def _prompt_variants(prompt: str, draft_answer: str = "") -> List[str]:
    prompt_text = " ".join(str(prompt or "").split())
    focus = _task_focus(prompt_text)
    variants = [
        prompt_text,
        (
            "Understand the user's request before answering.\n"
            f"Request: {prompt_text}\n"
            "Reinstate any explicit constraints, keep the answer on-topic, and avoid unrelated stock replies."
        ),
        (
            "Determine the main task, output style, and domain before answering.\n"
            f"User request: {prompt_text}\n"
            "Choose the answer that best matches the request and its constraints."
        ),
    ]
    if draft_answer:
        variants.append(
            (
                "Review the candidate answer for correctness and relevance.\n"
                f"Request: {prompt_text}\n"
                f"Candidate answer: {draft_answer}\n"
                "If the candidate is off-topic, prefer a more relevant final answer."
            )
        )
    if focus == "coding":
        variants.append(
            (
                "This is a coding request.\n"
                f"Request: {prompt_text}\n"
                "Prefer exact syntax, concrete debugging steps, and technically correct explanations."
            )
        )
    elif focus == "openscad":
        variants.append(
            (
                "This is an OpenSCAD request.\n"
                f"Request: {prompt_text}\n"
                "Prefer valid OpenSCAD primitives, transforms, boolean operations, modules, and parameterized examples."
            )
        )
    elif focus == "reasoning":
        variants.append(
            (
                "This request benefits from careful reasoning.\n"
                f"Request: {prompt_text}\n"
                "Focus on the user's real goal, the hidden constraints, and the most direct correct answer."
            )
        )
    deduped: List[str] = []
    seen = set()
    for item in variants:
        cooked = str(item).strip()
        if cooked and cooked not in seen:
            deduped.append(cooked)
            seen.add(cooked)
    return deduped


@dataclass
class OmniPredictionV5:
    intent: str
    response_text: str
    vision_concept: str
    vision_confidence: float
    top_vision: List[Dict[str, float]]
    domain: str = "general"
    domain_confidence: float = 0.0
    response_confidence: float = 0.0
    reasoning_passes: int = 0
    reasoning_mode: str = "deliberate_consensus"


class OmniCollectiveEngineV5:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.vocab = {str(key): int(value) for key, value in dict(self.meta.get("vocab") or {}).items()}
        self.responses = tuple(str(item) for item in list(self.meta.get("response_bank") or []))
        self.class_info = {str(key): dict(value) for key, value in dict(self.meta.get("class_info") or {}).items()}
        self.image_size = int(self.meta.get("image_size") or 128)
        self.max_len = int(self.meta.get("max_len") or 320)
        self.intent_labels = tuple(str(item) for item in list(self.meta.get("intent_labels") or OMNI_INTENTS_V2))
        self.domain_labels = tuple(str(item) for item in list(self.meta.get("domain_labels") or OMNI_DOMAIN_LABELS_V2))
        self.word_buckets = int(self.meta.get("word_buckets") or 16384)
        self.max_words = int(self.meta.get("max_words") or 72)
        self.deliberation_passes = int(self.meta.get("deliberation_passes") or 4)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = OmniCollectiveNetV5(
            vocab_size=max(len(self.vocab), 2),
            num_intents=len(self.intent_labels),
            num_responses=max(len(self.responses), 1),
            num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
            num_domains=max(len(self.domain_labels), 1),
            base_embed_dim=int(self.meta.get("embed_dim") or 96),
            text_hidden=int(self.meta.get("text_hidden") or 184),
            image_channels=int(self.meta.get("image_channels") or 40),
            word_buckets=int(self.meta.get("word_buckets") or 16384),
            word_embed_dim=int(self.meta.get("word_embed_dim") or 84),
            deep_text_channels=int(self.meta.get("deep_text_channels") or 224),
            deep_image_channels=int(self.meta.get("deep_image_channels") or 76),
            fusion_hidden=int(self.meta.get("fusion_hidden") or 640),
            memory_slots=int(self.meta.get("memory_slots") or 14),
            depth_steps=int(self.meta.get("depth_steps") or 6),
            expert_count=int(self.meta.get("expert_count") or 5),
            expert_hidden=int(self.meta.get("expert_hidden") or 1024),
            context_top_k=int(self.meta.get("context_top_k") or 3),
            expert_top_k=int(self.meta.get("expert_top_k") or 2),
        ).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        return {
            "token_ids": torch.tensor([encode_text(prompt, self.vocab, self.max_len)], dtype=torch.long, device=self.device),
            "word_ids": torch.tensor(
                [encode_word_hashes(prompt, buckets=self.word_buckets, max_words=self.max_words)],
                dtype=torch.long,
                device=self.device,
            ),
            "prompt_features": torch.tensor([prompt_feature_vector(prompt)], dtype=torch.float32, device=self.device),
        }

    def _encode_image(self, image_path: Optional[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not image_path:
            size = int(self.image_size)
            return (
                torch.zeros((1, 3, size, size), dtype=torch.float32, device=self.device),
                torch.zeros((1,), dtype=torch.float32, device=self.device),
            )
        tensor = load_image_tensor(str(image_path), self.image_size).unsqueeze(0).to(self.device)
        return tensor, torch.ones((1,), dtype=torch.float32, device=self.device)

    def _run_prompt(self, prompt: str, image_tensor: torch.Tensor, has_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        payload = self._encode_prompt(prompt)
        with torch.inference_mode():
            outputs = self.model(
                payload["token_ids"],
                image_tensor,
                has_image,
                payload["word_ids"],
                payload["prompt_features"],
            )
        return {name: value[0].detach().cpu() for name, value in outputs.items() if name in {"intent", "response", "vision", "domain"}}

    @staticmethod
    def _combine_outputs(weighted_outputs: List[tuple[float, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        merged: Dict[str, torch.Tensor] = {}
        total_weight = 0.0
        for weight, outputs in weighted_outputs:
            total_weight += float(weight)
            for name, tensor in outputs.items():
                if name not in merged:
                    merged[name] = tensor.float() * float(weight)
                else:
                    merged[name] = merged[name] + tensor.float() * float(weight)
        if total_weight <= 0:
            return {name: tensor.float() for name, tensor in weighted_outputs[0][1].items()}
        return {name: tensor / total_weight for name, tensor in merged.items()}

    def predict(self, prompt: str, image_path: Optional[str] = None) -> OmniPredictionV5:
        prompt_text = str(prompt or "").strip()
        image_tensor, has_image = self._encode_image(image_path)
        raw_outputs = self._run_prompt(prompt_text, image_tensor, has_image)
        draft_idx = int(raw_outputs["response"].argmax(dim=0).item()) if self.responses else 0
        draft_answer = self.responses[draft_idx] if self.responses else ""
        variants = _prompt_variants(prompt_text, draft_answer)
        weighted_outputs = []
        for index, variant in enumerate(variants, start=1):
            weight = 1.15 if index == 1 else 1.0
            if "OpenSCAD" in variant:
                weight = 1.10
            weighted_outputs.append((weight, self._run_prompt(variant, image_tensor, has_image)))
            if index >= self.deliberation_passes:
                break
        outputs = self._combine_outputs(weighted_outputs)

        intent_probs = torch.softmax(outputs["intent"], dim=0)
        response_probs = torch.softmax(outputs["response"], dim=0) if self.responses else torch.ones(1)
        vision_probs = torch.softmax(outputs["vision"], dim=0)
        domain_probs = torch.softmax(outputs["domain"], dim=0)

        intent_idx = int(intent_probs.argmax(dim=0).item()) if intent_probs.numel() else 0
        response_idx = int(response_probs.argmax(dim=0).item()) if response_probs.numel() else 0
        domain_idx = int(domain_probs.argmax(dim=0).item()) if domain_probs.numel() else 0
        values, indices = torch.topk(vision_probs, k=min(3, len(SCIENCE_IMAGE_CLASSES)))
        vision_key = SCIENCE_IMAGE_CLASSES[int(indices[0].item())]
        top_vision = [
            {"concept": SCIENCE_IMAGE_CLASSES[int(index.item())], "confidence": round(float(value.item()), 4)}
            for value, index in zip(values, indices)
        ]
        return OmniPredictionV5(
            intent=self.intent_labels[min(intent_idx, len(self.intent_labels) - 1)] if self.intent_labels else "general",
            response_text=self.responses[response_idx] if self.responses else "",
            vision_concept=vision_key,
            vision_confidence=float(values[0].item()),
            top_vision=top_vision,
            domain=self.domain_labels[min(domain_idx, len(self.domain_labels) - 1)] if self.domain_labels else "general",
            domain_confidence=float(domain_probs[domain_idx].item()) if domain_probs.numel() else 0.0,
            response_confidence=float(response_probs[response_idx].item()) if response_probs.numel() else 0.0,
            reasoning_passes=min(len(variants), self.deliberation_passes),
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
            return prediction.response_text or "Use web search for current external information."
        if prediction.intent == "command":
            return prediction.response_text or "I can ask the runtime to open Command Prompt when you explicitly request it."
        return prediction.response_text or "I did not have a strong fused answer for that prompt."


__all__ = [
    "OmniCollectiveEngineV5",
    "OmniCollectiveNetV5",
    "OmniPredictionV5",
    "_prompt_variants",
    "_task_focus",
]
