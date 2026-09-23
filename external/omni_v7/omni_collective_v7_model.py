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
    from omni_collective_v5_model import _task_focus
except ImportError:  # pragma: no cover
    from .image_feature_utils import describe_image_for_text
    from .image_recognition_model import CLASS_LABELS, SCIENCE_IMAGE_CLASSES, load_image_tensor
    from .math_equation_model import heuristic_intent, solve_intent
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, encode_text, encode_word_hashes, prompt_feature_vector
    from .omni_collective_v4_model import OmniCollectiveNetV4
    from .omni_collective_v5_model import _task_focus


OmniCollectiveNetV7 = OmniCollectiveNetV4


def _prompt_variants_v7(
    prompt: str,
    draft_answer: str = "",
    *,
    response_confidence: float = 0.0,
    domain_confidence: float = 0.0,
) -> List[str]:
    prompt_text = " ".join(str(prompt or "").split())
    lowered = prompt_text.lower()
    focus = _task_focus(prompt_text)
    variants = [
        prompt_text,
        (
            "Understand the request before answering.\n"
            f"Request: {prompt_text}\n"
            "Identify the real task, output format, safety boundary, and any hidden constraints before drafting the answer."
        ),
        (
            "Ground the answer in what is actually supported by the request and local knowledge.\n"
            f"User request: {prompt_text}\n"
            "Prefer bounded claims, explicit uncertainty, and verifiable local knowledge over confident invention."
        ),
        (
            "Extract the user's constraints and preserve them exactly.\n"
            f"Request: {prompt_text}\n"
            "Keep the response on-topic, respect formatting instructions, and do not drift into canned filler."
        ),
        (
            "Run a hallucination guard pass.\n"
            f"Request: {prompt_text}\n"
            "Remove unsupported specifics, current-event guesses, fake citations, and details that are not grounded in the prompt or local knowledge."
        ),
    ]
    if draft_answer:
        variants.append(
            (
                "Review this candidate answer for correctness, grounding, and missing constraints.\n"
                f"Request: {prompt_text}\n"
                f"Candidate answer: {draft_answer}\n"
                "Keep the strongest parts, remove speculative claims, and repair the weak sections."
            )
        )
    if response_confidence < 0.50 or domain_confidence < 0.42:
        variants.append(
            (
                "Confidence is low. Produce the most grounded answer you can.\n"
                f"Request: {prompt_text}\n"
                "Prefer concise uncertainty and narrow claims over broad speculation."
            )
        )
    if focus == "coding":
        variants.append(
            (
                "This is a coding request.\n"
                f"Request: {prompt_text}\n"
                "Prefer exact syntax, reproducible debugging logic, concrete fixes, and technically correct terminology."
            )
        )
    elif focus == "openscad":
        variants.append(
            (
                "This is an OpenSCAD request.\n"
                f"Request: {prompt_text}\n"
                "Prefer valid primitives, transforms, boolean ops, modules, and parameterized examples."
            )
        )
    elif focus == "reasoning":
        variants.append(
            (
                "This request benefits from careful reasoning.\n"
                f"Request: {prompt_text}\n"
                "Work through the true goal, hidden constraints, and the safest direct answer before responding."
            )
        )
    else:
        variants.append(
            (
                "This is a conversation-focused request.\n"
                f"Request: {prompt_text}\n"
                "Answer naturally, clearly, and helpfully while staying concise and grounded."
            )
        )
    if any(token in lowered for token in ("math", "equation", "integrate", "differentiate", "solve", "factor", "derivative", "integral")):
        variants.append(
            (
                "This request may involve math.\n"
                f"Request: {prompt_text}\n"
                "Prefer exact symbolic reasoning or a clean final statement over loose intuition."
            )
        )
    if any(token in lowered for token in ("protein", "folding", "amino acid", "residue", "alpha helix", "beta sheet", "hydrophobic", "chaperone", "contact map")):
        variants.append(
            (
                "This request may involve protein folding or structural biology.\n"
                f"Request: {prompt_text}\n"
                "Prefer grounded protein-folding concepts, structural terminology, and uncertainty over invented biochemical detail."
            )
        )
    if any(token in lowered for token in ("image", "photo", "diagram", "vision", "recognize", "uploaded")):
        variants.append(
            (
                "This request may involve vision or an uploaded image.\n"
                f"Request: {prompt_text}\n"
                "Use only the provided visual signal or explicit local model knowledge. Do not pretend to see an image that is not present."
            )
        )
    if any(token in lowered for token in ("best model", "which model", "selector", "collective", "agent mode", "auto mode")):
        variants.append(
            (
                "This is a model-selection request.\n"
                f"Request: {prompt_text}\n"
                "Choose the best local specialist by matching the task to the model's real strengths and limitations."
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
class OmniPredictionV7:
    intent: str
    response_text: str
    vision_concept: str
    vision_confidence: float
    top_vision: List[Dict[str, float]]
    domain: str = "general"
    domain_confidence: float = 0.0
    response_confidence: float = 0.0
    reasoning_passes: int = 0
    reasoning_mode: str = "all_model_grounded_consensus_deliberation_v7"


class OmniCollectiveEngineV7:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.vocab = {str(key): int(value) for key, value in dict(self.meta.get("vocab") or {}).items()}
        self.responses = tuple(str(item) for item in list(self.meta.get("response_bank") or []))
        self.class_info = {str(key): dict(value) for key, value in dict(self.meta.get("class_info") or {}).items()}
        self.image_size = int(self.meta.get("image_size") or 128)
        self.max_len = int(self.meta.get("max_len") or 360)
        self.intent_labels = tuple(str(item) for item in list(self.meta.get("intent_labels") or OMNI_INTENTS_V2))
        self.domain_labels = tuple(str(item) for item in list(self.meta.get("domain_labels") or OMNI_DOMAIN_LABELS_V2))
        self.word_buckets = int(self.meta.get("word_buckets") or 16384)
        self.max_words = int(self.meta.get("max_words") or 84)
        self.deliberation_passes = int(self.meta.get("deliberation_passes") or 8)
        self.minimum_passes = int(self.meta.get("minimum_passes") or 4)
        self.grounding_threshold = float(self.meta.get("grounding_threshold") or 0.50)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = OmniCollectiveNetV7(
            vocab_size=max(len(self.vocab), 2),
            num_intents=len(self.intent_labels),
            num_responses=max(len(self.responses), 1),
            num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
            num_domains=max(len(self.domain_labels), 1),
            base_embed_dim=int(self.meta.get("embed_dim") or 116),
            text_hidden=int(self.meta.get("text_hidden") or 236),
            image_channels=int(self.meta.get("image_channels") or 48),
            word_buckets=int(self.meta.get("word_buckets") or 16384),
            word_embed_dim=int(self.meta.get("word_embed_dim") or 108),
            deep_text_channels=int(self.meta.get("deep_text_channels") or 336),
            deep_image_channels=int(self.meta.get("deep_image_channels") or 112),
            fusion_hidden=int(self.meta.get("fusion_hidden") or 960),
            memory_slots=int(self.meta.get("memory_slots") or 22),
            depth_steps=int(self.meta.get("depth_steps") or 9),
            expert_count=int(self.meta.get("expert_count") or 8),
            expert_hidden=int(self.meta.get("expert_hidden") or 1536),
            context_top_k=int(self.meta.get("context_top_k") or 4),
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

    def _confidence_from_outputs(self, outputs: Dict[str, torch.Tensor]) -> tuple[float, float]:
        response_probs = torch.softmax(outputs["response"], dim=0) if self.responses else torch.ones(1)
        domain_probs = torch.softmax(outputs["domain"], dim=0)
        response_conf = float(response_probs.max().item()) if response_probs.numel() else 0.0
        domain_conf = float(domain_probs.max().item()) if domain_probs.numel() else 0.0
        return response_conf, domain_conf

    def predict(self, prompt: str, image_path: Optional[str] = None) -> OmniPredictionV7:
        prompt_text = str(prompt or "").strip()
        image_tensor, has_image = self._encode_image(image_path)
        raw_outputs = self._run_prompt(prompt_text, image_tensor, has_image)
        response_conf, domain_conf = self._confidence_from_outputs(raw_outputs)
        draft_idx = int(raw_outputs["response"].argmax(dim=0).item()) if self.responses else 0
        draft_answer = self.responses[draft_idx] if self.responses else ""
        variants = _prompt_variants_v7(
            prompt_text,
            draft_answer,
            response_confidence=response_conf,
            domain_confidence=domain_conf,
        )

        weighted_outputs: List[tuple[float, Dict[str, torch.Tensor]]] = []
        combined = raw_outputs
        for index, variant in enumerate(variants, start=1):
            weight = 1.22 if index == 1 else 1.0
            lowered = variant.lower()
            if "hallucination guard" in lowered:
                weight = 1.10
            elif "review this candidate answer" in lowered:
                weight = 1.08
            elif "confidence is low" in lowered:
                weight = 1.06
            weighted_outputs.append((weight, self._run_prompt(variant, image_tensor, has_image)))
            combined = self._combine_outputs(weighted_outputs)
            response_conf, domain_conf = self._confidence_from_outputs(combined)
            if index >= self.minimum_passes and response_conf >= self.grounding_threshold and domain_conf >= 0.44:
                break
            if index >= self.deliberation_passes:
                break

        outputs = combined
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
        return OmniPredictionV7(
            intent=self.intent_labels[min(intent_idx, len(self.intent_labels) - 1)] if self.intent_labels else "general",
            response_text=self.responses[response_idx] if self.responses else "",
            vision_concept=vision_key,
            vision_confidence=float(values[0].item()),
            top_vision=top_vision,
            domain=self.domain_labels[min(domain_idx, len(self.domain_labels) - 1)] if self.domain_labels else "general",
            domain_confidence=float(domain_probs[domain_idx].item()) if domain_probs.numel() else 0.0,
            response_confidence=float(response_probs[response_idx].item()) if response_probs.numel() else 0.0,
            reasoning_passes=len(weighted_outputs),
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
        if prediction.response_confidence < self.grounding_threshold and prediction.domain in {"knowledge", "general", "planning", "language", "model_selection"}:
            if prediction.response_text:
                return f"Best grounded answer from my local training: {prediction.response_text}"
            return "I do not have enough grounded evidence in my local training for a confident answer."
        return prediction.response_text or "I did not have a strong fused answer for that prompt."


__all__ = [
    "OmniCollectiveEngineV7",
    "OmniCollectiveNetV7",
    "OmniPredictionV7",
    "_prompt_variants_v7",
]
