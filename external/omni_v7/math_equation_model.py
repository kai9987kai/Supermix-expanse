from __future__ import annotations

import json
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy as sp
import torch
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from torch import nn


INTENT_LABELS: Tuple[str, ...] = (
    "arithmetic",
    "solve_equation",
    "solve_system",
    "simplify",
    "factor",
    "expand",
    "differentiate",
    "integrate",
)

LABEL_TO_INDEX = {label: index for index, label in enumerate(INTENT_LABELS)}
TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)
SAFE_SYMBOLS = {name: sp.symbols(name) for name in string.ascii_lowercase}

REQUEST_MARKERS = (
    "current user request:",
    "original request:",
    "request:",
)

MATH_PROMPT_RE = re.compile(
    r"(\bsolve\b|\bsimplify\b|\bfactor\b|\bexpand\b|\bdifferentiate\b|\bderivative\b|"
    r"\bintegrate\b|\bcalculate\b|\bevaluate\b|\bequation\b|\bsystem of equations\b|"
    r"\bpolynomial\b|\bquadratic\b|=|[\d\)\]][\+\-\*/\^])",
    re.IGNORECASE,
)
VARIABLE_HINT_RE = re.compile(r"(?:with respect to|w\.r\.t\.|for)\s+([a-z])\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9_\.\+\-\*/\^\=\(\),\[\]\{\}:; ]+")


class TinyMathIntentNet(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, embed_dim: int = 24, hidden_dim: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = token_ids.ne(0).unsqueeze(-1)
        embedded = self.embedding(token_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        hidden = torch.relu(self.fc1(self.norm(pooled)))
        return self.fc2(hidden)


def extract_request_text(prompt: str) -> str:
    cooked = str(prompt or "").strip()
    lowered = cooked.lower()
    for marker in REQUEST_MARKERS:
        idx = lowered.rfind(marker)
        if idx >= 0:
            return cooked[idx + len(marker):].strip()
    return cooked


def normalize_math_text(text: str) -> str:
    cooked = str(text or "").strip()
    replacements = {
        "×": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "π": "pi",
        "√": "sqrt",
    }
    for source, target in replacements.items():
        cooked = cooked.replace(source, target)
    cooked = re.sub(r"\s+", " ", cooked)
    return cooked


def build_vocab(texts: Sequence[str], *, min_frequency: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for ch in normalize_math_text(text).lower():
            counts[ch] = counts.get(ch, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, count in sorted(counts.items()):
        if count >= min_frequency and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    cooked = normalize_math_text(text).lower()
    ids = [vocab.get(ch, 1) for ch in cooked[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def vectorize_batch(texts: Sequence[str], vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    rows = [encode_text(text, vocab, max_len) for text in texts]
    return torch.tensor(rows, dtype=torch.long)


def looks_like_math_prompt(prompt: str) -> bool:
    return bool(MATH_PROMPT_RE.search(str(prompt or "")))


def heuristic_intent(prompt: str) -> Optional[str]:
    text = extract_request_text(prompt).lower()
    if "system of equations" in text or text.count("=") >= 2:
        return "solve_system"
    if any(token in text for token in ("differentiate", "derivative", "derive ")):
        return "differentiate"
    if any(token in text for token in ("integrate", "integral", "antiderivative")):
        return "integrate"
    if "factor" in text:
        return "factor"
    if "expand" in text:
        return "expand"
    if "simplify" in text:
        return "simplify"
    if "=" in text or any(token in text for token in ("solve", "find x", "find y", "root")):
        return "solve_equation"
    if any(ch.isdigit() for ch in text) and any(op in text for op in ("+", "-", "*", "/", "^")):
        return "arithmetic"
    return None


def choose_symbol(expr: sp.Expr, prompt: str) -> sp.Symbol:
    hint = VARIABLE_HINT_RE.search(prompt)
    if hint:
        return sp.symbols(hint.group(1).lower())
    symbols = sorted(expr.free_symbols, key=lambda item: str(item))
    if symbols:
        return symbols[0]
    return sp.symbols("x")


def _strip_prompt_to_body(prompt: str) -> str:
    text = extract_request_text(prompt)
    text = normalize_math_text(text)
    text = re.sub(
        r"^(please\s+)?(help me\s+)?(solve|calculate|evaluate|simplify|factor|expand|differentiate|integrate)\b",
        lambda match: match.group(2) or match.group(0),
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(" .!?")


def _clean_expression_text(text: str) -> str:
    cooked = normalize_math_text(text)
    cooked = re.sub(r"(?i)\b(what is|calculate|compute|evaluate|find|show|please|solve|simplify|factor|expand|differentiate|integrate|the|expression|equation|system|of|for|me)\b", " ", cooked)
    cooked = re.sub(r"\s+", " ", cooked).strip(" :,.?")
    match = TOKEN_RE.search(cooked)
    return match.group(0).strip() if match else cooked


def parse_expression(text: str) -> sp.Expr:
    cooked = _clean_expression_text(text)
    if not cooked:
        raise ValueError("No expression found.")
    return parse_expr(cooked, local_dict=SAFE_SYMBOLS, transformations=TRANSFORMATIONS, evaluate=True)


def parse_equation(text: str) -> Tuple[sp.Expr, sp.Symbol]:
    body = _clean_expression_text(text)
    if "=" in body:
        lhs_text, rhs_text = body.split("=", 1)
        lhs = parse_expression(lhs_text)
        rhs = parse_expression(rhs_text)
        expr = sp.expand(lhs - rhs)
    else:
        expr = parse_expression(body)
    variable = choose_symbol(expr, text)
    return expr, variable


def parse_system(text: str) -> Tuple[List[sp.Expr], List[sp.Symbol]]:
    body = _strip_prompt_to_body(text)
    body = re.sub(r"(?i)\b(system of equations|solve the system|solve system)\b", " ", body)
    chunks = [chunk.strip() for chunk in re.split(r"[;\n]|,(?=[^0-9])|\band\b", body) if chunk.strip()]
    equations: List[sp.Expr] = []
    symbols: List[sp.Symbol] = []
    for chunk in chunks:
        if "=" not in chunk:
            continue
        lhs_text, rhs_text = chunk.split("=", 1)
        lhs = parse_expression(lhs_text)
        rhs = parse_expression(rhs_text)
        expr = sp.expand(lhs - rhs)
        equations.append(expr)
        symbols.extend(list(expr.free_symbols))
    unique_symbols = sorted({symbol for symbol in symbols}, key=lambda item: str(item))
    if not equations or not unique_symbols:
        raise ValueError("Could not parse a system of equations.")
    return equations, unique_symbols


def _format_solution_values(values: Sequence[Any]) -> str:
    rendered = []
    for value in values:
        simplified = sp.simplify(value)
        rendered.append(str(simplified))
    return ", ".join(rendered)


def solve_intent(prompt: str, intent: str) -> Dict[str, Any]:
    request = extract_request_text(prompt)
    if intent == "arithmetic":
        expr = parse_expression(request)
        exact = sp.simplify(expr)
        approx = sp.N(exact)
        response = f"Result: {exact}"
        if exact.is_number and not sp.Integer(approx) == exact:
            response += f"\nApproximation: {approx}"
        return {"response": response, "expression": str(expr)}

    if intent == "solve_equation":
        expr, variable = parse_equation(request)
        roots = sp.solve(sp.Eq(expr, 0), variable)
        if not roots:
            response = f"No symbolic solution found for {variable}."
        else:
            response = f"Solve for {variable}: {variable} = {_format_solution_values(roots)}"
        return {"response": response, "expression": str(expr), "variable": str(variable)}

    if intent == "solve_system":
        equations, variables = parse_system(request)
        solutions = sp.solve([sp.Eq(expr, 0) for expr in equations], variables, dict=True)
        if not solutions:
            response = "No symbolic solution found for the system."
        else:
            rows = []
            for idx, solution in enumerate(solutions, start=1):
                bits = [f"{symbol} = {sp.simplify(solution[symbol])}" for symbol in variables if symbol in solution]
                rows.append(f"Solution {idx}: " + ", ".join(bits))
            response = "\n".join(rows)
        return {
            "response": response,
            "equations": [str(expr) for expr in equations],
            "variables": [str(symbol) for symbol in variables],
        }

    if intent == "simplify":
        expr = parse_expression(request)
        simplified = sp.simplify(expr)
        return {"response": f"Simplified: {simplified}", "expression": str(expr)}

    if intent == "factor":
        expr = parse_expression(request)
        factored = sp.factor(expr)
        return {"response": f"Factored: {factored}", "expression": str(expr)}

    if intent == "expand":
        expr = parse_expression(request)
        expanded = sp.expand(expr)
        return {"response": f"Expanded: {expanded}", "expression": str(expr)}

    if intent == "differentiate":
        expr = parse_expression(request)
        variable = choose_symbol(expr, request)
        derivative = sp.diff(expr, variable)
        return {
            "response": f"Derivative with respect to {variable}: {derivative}",
            "expression": str(expr),
            "variable": str(variable),
        }

    if intent == "integrate":
        expr = parse_expression(request)
        variable = choose_symbol(expr, request)
        integral = sp.integrate(expr, variable)
        return {
            "response": f"Indefinite integral with respect to {variable}: {integral} + C",
            "expression": str(expr),
            "variable": str(variable),
        }

    raise ValueError(f"Unsupported math intent: {intent}")


@dataclass
class MathPrediction:
    intent: str
    confidence: float
    probabilities: Dict[str, float]


class MathEquationEngine:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.vocab = {str(key): int(value) for key, value in dict(self.meta.get("vocab") or {}).items()}
        self.labels = tuple(str(item) for item in (self.meta.get("labels") or INTENT_LABELS))
        self.max_len = int(self.meta.get("max_len") or 192)
        embed_dim = int(self.meta.get("embed_dim") or 24)
        hidden_dim = int(self.meta.get("hidden_dim") or 48)
        self.device = device or torch.device("cpu")
        self.model = TinyMathIntentNet(
            vocab_size=max(len(self.vocab), 2),
            num_labels=len(self.labels),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        try:
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def predict(self, prompt: str) -> MathPrediction:
        request = extract_request_text(prompt)
        encoded = vectorize_batch([request], self.vocab, self.max_len).to(self.device)
        with torch.inference_mode():
            logits = self.model(encoded)[0]
            probs = torch.softmax(logits, dim=0).detach().cpu().tolist()
        probabilities = {label: float(prob) for label, prob in zip(self.labels, probs)}
        predicted = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[predicted])
        heuristic = heuristic_intent(request)
        if heuristic in probabilities and confidence < 0.9:
            predicted = heuristic
            confidence = max(confidence, 0.93)
        elif heuristic in probabilities and probabilities.get(heuristic, 0.0) >= 0.18:
            predicted = heuristic
            confidence = max(confidence, float(probabilities[heuristic]))
        return MathPrediction(intent=predicted, confidence=confidence, probabilities=probabilities)

    def solve(self, prompt: str) -> Dict[str, Any]:
        prediction = self.predict(prompt)
        try:
            solved = solve_intent(prompt, prediction.intent)
            solved["intent"] = prediction.intent
            solved["confidence"] = round(prediction.confidence, 4)
            solved["probabilities"] = prediction.probabilities
            return solved
        except Exception as exc:
            return {
                "intent": prediction.intent,
                "confidence": round(prediction.confidence, 4),
                "probabilities": prediction.probabilities,
                "response": (
                    "I could not solve that prompt directly.\n"
                    f"Detected math intent: {prediction.intent}.\n"
                    f"Parsing error: {exc}"
                ),
                "error": str(exc),
            }

    def status(self) -> Dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "labels": list(self.labels),
            "max_len": self.max_len,
            "device": str(self.device),
            "val_accuracy": self.meta.get("val_accuracy"),
            "train_accuracy": self.meta.get("train_accuracy"),
            "parameter_count": int(sum(parameter.numel() for parameter in self.model.parameters())),
        }


def format_math_response(result: Dict[str, Any]) -> str:
    intent = str(result.get("intent") or "math")
    confidence = float(result.get("confidence") or 0.0)
    headline = {
        "arithmetic": "Arithmetic",
        "solve_equation": "Equation",
        "solve_system": "System",
        "simplify": "Simplify",
        "factor": "Factor",
        "expand": "Expand",
        "differentiate": "Derivative",
        "integrate": "Integral",
    }.get(intent, "Math")
    response = str(result.get("response") or "").strip()
    trailer = f"\n\n[Math intent: {headline.lower()} | confidence {confidence:.2f}]"
    return response + trailer
