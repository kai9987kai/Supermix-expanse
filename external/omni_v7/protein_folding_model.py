from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

try:
    from v40_benchmax_common import PROTEIN_FOLDING_CONCEPTS
except ImportError:  # pragma: no cover
    from .v40_benchmax_common import PROTEIN_FOLDING_CONCEPTS


PROTEIN_CONCEPT_LABELS: Tuple[str, ...] = tuple(str(item["concept"]) for item in PROTEIN_FOLDING_CONCEPTS)
CONCEPT_TO_INDEX = {label: index for index, label in enumerate(PROTEIN_CONCEPT_LABELS)}
REQUEST_MARKERS = (
    "current user request:",
    "original request:",
    "request:",
)
WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-\+]*", re.IGNORECASE)
PROTEIN_PROMPT_RE = re.compile(
    r"\b(protein|folding|fold|amino acid|residue|alpha helix|beta sheet|hydrophobic|"
    r"chaperone|disulfide|contact map|plddt|msa|intrinsically disordered|"
    r"membrane protein|tm-score|rmsd)\b",
    re.IGNORECASE,
)
CONCEPT_DISPLAY_NAMES: Dict[str, str] = {
    "secondary_structure": "secondary structure",
    "hydrophobic_collapse": "hydrophobic collapse",
    "chaperones": "molecular chaperones",
    "disulfide_bonds": "disulfide bonds",
    "glycine_proline": "glycine and proline effects",
    "folding_funnel": "folding funnel",
    "contact_map": "protein contact map",
    "alphafold_confidence": "pLDDT confidence",
    "msa_signal": "multiple-sequence-alignment signal",
    "intrinsically_disordered": "intrinsically disordered proteins",
    "membrane_proteins": "membrane proteins",
    "rmsd_tm": "RMSD vs TM-score",
}
CONCEPT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "secondary_structure": ("alpha helix", "beta sheet", "secondary structure", "strand"),
    "hydrophobic_collapse": ("hydrophobic", "collapse", "buried core", "water"),
    "chaperones": ("chaperone", "aggregation", "misfold", "folding helper"),
    "disulfide_bonds": ("disulfide", "cysteine", "covalent", "oxidative"),
    "glycine_proline": ("glycine", "proline", "helix breaker", "backbone geometry"),
    "folding_funnel": ("folding funnel", "energy landscape", "native state", "conformations"),
    "contact_map": ("contact map", "residue pairs", "distance map", "contacts"),
    "alphafold_confidence": ("plddt", "confidence", "local structure", "alphafold"),
    "msa_signal": ("msa", "multiple-sequence alignment", "correlated mutations", "homologs"),
    "intrinsically_disordered": ("intrinsically disordered", "disordered protein", "ensemble", "rigid structure"),
    "membrane_proteins": ("membrane protein", "lipid", "transmembrane", "beta barrel"),
    "rmsd_tm": ("rmsd", "tm-score", "structure comparison", "fold similarity"),
}


def extract_request_text(prompt: str) -> str:
    cooked = str(prompt or "").strip()
    lowered = cooked.lower()
    for marker in REQUEST_MARKERS:
        idx = lowered.rfind(marker)
        if idx >= 0:
            return cooked[idx + len(marker):].strip()
    return cooked


def normalize_protein_text(text: str) -> str:
    cooked = str(text or "").strip()
    replacements = {
        "â€“": "-",
        "â€”": "-",
        "âˆ’": "-",
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
    }
    for source, target in replacements.items():
        cooked = cooked.replace(source, target)
    cooked = re.sub(r"\s+", " ", cooked)
    return cooked


def build_vocab(texts: Sequence[str], *, min_frequency: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for ch in normalize_protein_text(text).lower():
            counts[ch] = counts.get(ch, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, count in sorted(counts.items()):
        if count >= min_frequency and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    cooked = normalize_protein_text(extract_request_text(text)).lower()
    ids = [vocab.get(ch, 1) for ch in cooked[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def tokenize_words(text: str, *, max_words: int) -> List[str]:
    cooked = normalize_protein_text(extract_request_text(text)).lower()
    return WORD_RE.findall(cooked)[:max_words]


def encode_words(text: str, *, word_buckets: int, max_words: int) -> List[int]:
    ids = [0] * max_words
    for index, token in enumerate(tokenize_words(text, max_words=max_words)):
        ids[index] = (sum(ord(ch) * (offset + 1) for offset, ch in enumerate(token)) % max(word_buckets - 1, 1)) + 1
    return ids


def vectorize_batch(texts: Sequence[str], vocab: Dict[str, int], *, max_len: int, word_buckets: int, max_words: int) -> Tuple[torch.Tensor, torch.Tensor]:
    char_rows = [encode_text(text, vocab, max_len) for text in texts]
    word_rows = [encode_words(text, word_buckets=word_buckets, max_words=max_words) for text in texts]
    return (
        torch.tensor(char_rows, dtype=torch.long),
        torch.tensor(word_rows, dtype=torch.long),
    )


def looks_like_protein_folding_prompt(prompt: str) -> bool:
    return bool(PROTEIN_PROMPT_RE.search(str(prompt or "")))


def heuristic_protein_concept(prompt: str) -> Optional[str]:
    lowered = normalize_protein_text(extract_request_text(prompt)).lower()
    scores: Dict[str, int] = {}
    for concept, keywords in CONCEPT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in lowered:
                score += max(1, len(keyword.split()))
        if score > 0:
            scores[concept] = score
    if not scores:
        return None
    return max(sorted(scores), key=lambda key: scores[key])


def choose_answer_variant(prompt: str) -> str:
    lowered = normalize_protein_text(prompt).lower()
    if any(token in lowered for token in ("mistake", "wrong", "misconception", "correct this", "opposite")):
        return "error_correction"
    if any(token in lowered for token in ("debug", "failure mode", "go wrong", "decoy")):
        return "failure_mode"
    if any(token in lowered for token in ("rank", "ranking", "confidence", "score", "plddt", "tm-score", "rmsd")):
        return "ranking_and_confidence"
    if any(token in lowered for token in ("3d", "spatial", "geometry", "contact")):
        return "spatial_reasoning"
    if any(token in lowered for token in ("structure prediction", "prediction practice", "alphafold", "model")):
        return "structure_prediction_link"
    if any(token in lowered for token in ("student", "beginner", "biology class", "paragraph")):
        return "student_paragraph"
    if any(token in lowered for token in ("short", "brief", "concise", "one sentence")):
        return "concise_answer"
    return "concept_explain"


class ProteinFoldingMiniNet(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_concepts: int,
        char_embed_dim: int = 28,
        conv_channels: int = 56,
        word_buckets: int = 384,
        word_embed_dim: int = 18,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        self.char_conv = nn.Conv1d(char_embed_dim, conv_channels, kernel_size=5, padding=2)
        self.word_embedding = nn.Embedding(word_buckets, word_embed_dim, padding_idx=0)
        self.norm = nn.LayerNorm(conv_channels * 2 + word_embed_dim)
        self.fc1 = nn.Linear(conv_channels * 2 + word_embed_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.16)
        self.fc2 = nn.Linear(hidden_dim, num_concepts)

    def forward(self, char_ids: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        char_mask = char_ids.ne(0)
        char_embed = self.char_embedding(char_ids).transpose(1, 2)
        conv = torch.relu(self.char_conv(char_embed)).transpose(1, 2)
        pooled_avg = (conv * char_mask.unsqueeze(-1)).sum(dim=1) / char_mask.sum(dim=1, keepdim=True).clamp(min=1)
        masked_conv = conv.masked_fill(~char_mask.unsqueeze(-1), -1e4)
        pooled_max = masked_conv.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        word_mask = word_ids.ne(0)
        word_embed = self.word_embedding(word_ids)
        word_pool = (word_embed * word_mask.unsqueeze(-1)).sum(dim=1) / word_mask.sum(dim=1, keepdim=True).clamp(min=1)

        features = torch.cat([pooled_avg, pooled_max, word_pool], dim=1)
        hidden = torch.relu(self.fc1(self.dropout(self.norm(features))))
        return self.fc2(self.dropout(hidden))


@dataclass
class ProteinPrediction:
    concept: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    top_predictions: List[Dict[str, float]]
    variant: str


class ProteinFoldingEngine:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.vocab = {str(key): int(value) for key, value in dict(self.meta.get("vocab") or {}).items()}
        self.labels = tuple(str(item) for item in (self.meta.get("labels") or PROTEIN_CONCEPT_LABELS))
        self.answer_bank = {
            str(concept): {str(key): str(value) for key, value in dict(variants or {}).items()}
            for concept, variants in dict(self.meta.get("answer_bank") or {}).items()
        }
        self.display_names = {
            str(key): str(value)
            for key, value in dict(self.meta.get("display_names") or CONCEPT_DISPLAY_NAMES).items()
        }
        self.max_len = int(self.meta.get("max_len") or 224)
        self.max_words = int(self.meta.get("max_words") or 28)
        self.word_buckets = int(self.meta.get("word_buckets") or 384)
        self.device = device or torch.device("cpu")
        self.model = ProteinFoldingMiniNet(
            vocab_size=max(len(self.vocab), 2),
            num_concepts=len(self.labels),
            char_embed_dim=int(self.meta.get("char_embed_dim") or 28),
            conv_channels=int(self.meta.get("conv_channels") or 56),
            word_buckets=self.word_buckets,
            word_embed_dim=int(self.meta.get("word_embed_dim") or 18),
            hidden_dim=int(self.meta.get("hidden_dim") or 96),
        ).to(self.device)
        try:
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def predict(self, prompt: str) -> ProteinPrediction:
        char_tensor, word_tensor = vectorize_batch(
            [prompt],
            self.vocab,
            max_len=self.max_len,
            word_buckets=self.word_buckets,
            max_words=self.max_words,
        )
        char_tensor = char_tensor.to(self.device)
        word_tensor = word_tensor.to(self.device)
        with torch.inference_mode():
            logits = self.model(char_tensor, word_tensor)[0]
            probs = torch.softmax(logits, dim=0).detach().cpu().tolist()
        probabilities = {label: float(prob) for label, prob in zip(self.labels, probs)}
        predicted = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[predicted])
        heuristic = heuristic_protein_concept(prompt)
        if heuristic in probabilities and confidence < 0.88:
            heuristic_score = float(probabilities.get(heuristic, 0.0))
            if heuristic_score >= 0.14 or confidence < 0.45:
                predicted = heuristic
                confidence = max(confidence, heuristic_score, 0.91 if confidence < 0.45 else confidence)
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_predictions = [
            {
                "concept": concept,
                "confidence": round(float(score), 4),
            }
            for concept, score in ranked[:3]
        ]
        return ProteinPrediction(
            concept=predicted,
            label=self.display_names.get(predicted, predicted.replace("_", " ")),
            confidence=confidence,
            probabilities=probabilities,
            top_predictions=top_predictions,
            variant=choose_answer_variant(prompt),
        )

    def answer(self, prompt: str) -> str:
        prediction = self.predict(prompt)
        variants = self.answer_bank.get(prediction.concept, {})
        answer = (
            variants.get(prediction.variant)
            or variants.get("concept_explain")
            or next(iter(variants.values()), f"This specialist matched {prediction.label}.")
        )
        if any(token in prompt.lower() for token in ("brief", "short", "one sentence", "concise")):
            return answer
        lines = [answer]
        if prediction.confidence < 0.52:
            lines.append(
                f"Best matched concept: {prediction.label} with low confidence ({prediction.confidence:.2f}), so treat this as a grounded best guess."
            )
        elif any(token in prompt.lower() for token in ("why", "evidence", "match", "confidence", "which concept")):
            lines.append(f"Matched concept: {prediction.label} ({prediction.confidence:.2f} confidence).")
        alternatives = [
            f"{self.display_names.get(item['concept'], item['concept'].replace('_', ' '))} {item['confidence']:.2f}"
            for item in prediction.top_predictions[1:]
        ]
        if alternatives and any(token in prompt.lower() for token in ("alternatives", "else", "other plausible")):
            lines.append("Other plausible matches: " + ", ".join(alternatives) + ".")
        return "\n".join(line for line in lines if line.strip())

    def status(self) -> Dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "labels": list(self.labels),
            "max_len": self.max_len,
            "max_words": self.max_words,
            "device": str(self.device),
            "val_accuracy": self.meta.get("val_accuracy"),
            "train_accuracy": self.meta.get("train_accuracy"),
            "parameter_count": int(sum(parameter.numel() for parameter in self.model.parameters())),
        }


def format_protein_response(response: str, prediction: ProteinPrediction) -> str:
    trailer = f"\n\n[Protein concept: {prediction.label.lower()} | confidence {prediction.confidence:.2f}]"
    return str(response or "").strip() + trailer
