from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    from dcgan_image_model import DCGAN_SPECS
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
    from multimodel_catalog import ModelRecord, discover_model_records
    from multimodel_runtime import UnifiedModelManager
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from omni_collective_v7_model import OmniCollectiveEngineV7, OmniCollectiveNetV7
    from train_image_recognition_model import ensure_base_images
    from train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, _rows_from_jsonl, split_rows
    from train_omni_collective_v3 import _score_candidate
    from train_omni_collective_v4 import _load_expanded_state_from_zip, _resolve_local_qwen_base_model, _train_stage
    from train_omni_collective_v6 import (
        _build_rows_v6,
        _conversation_alignment_rows_v6,
        _grounding_rows_v6,
        _math_exact_rows_v6,
        _protein_rows_v6,
        _sample_teacher_rows,
    )
    from v40_benchmax_common import build_protein_folding_rows, build_v33_style_rows, build_v39_style_rows, prompt_row_to_omni
except ImportError:  # pragma: no cover
    from .dcgan_image_model import DCGAN_SPECS
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES
    from .multimodel_catalog import ModelRecord, discover_model_records
    from .multimodel_runtime import UnifiedModelManager
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from .omni_collective_v7_model import OmniCollectiveEngineV7, OmniCollectiveNetV7
    from .train_image_recognition_model import ensure_base_images
    from .train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, _rows_from_jsonl, split_rows
    from .train_omni_collective_v3 import _score_candidate
    from .train_omni_collective_v4 import _load_expanded_state_from_zip, _resolve_local_qwen_base_model, _train_stage
    from .train_omni_collective_v6 import (
        _build_rows_v6,
        _conversation_alignment_rows_v6,
        _grounding_rows_v6,
        _math_exact_rows_v6,
        _protein_rows_v6,
        _sample_teacher_rows,
    )
    from .v40_benchmax_common import build_protein_folding_rows, build_v33_style_rows, build_v39_style_rows, prompt_row_to_omni


DEFAULT_BASE_ZIP = Path("output/supermix_omni_collective_v6_frontier_20260331.zip")


def _safe_domain(domain: str) -> str:
    cooked = str(domain or "general")
    return cooked if cooked in OMNI_DOMAIN_LABELS_V2 else "general"


def _dedupe_rows(rows: Sequence[OmniRow]) -> List[OmniRow]:
    deduped: List[OmniRow] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.prompt).strip().lower(),
            str(row.response_text).strip().lower(),
            str(row.domain).strip().lower(),
            str(row.source).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _omni_row_to_payload(row: OmniRow) -> Dict[str, Any]:
    return {
        "prompt": row.prompt,
        "intent": row.intent,
        "response_text": row.response_text,
        "domain": row.domain,
        "source": row.source,
        "image_path": row.image_path,
        "vision_label": row.vision_label,
    }


def _omni_row_from_payload(payload: Dict[str, Any]) -> OmniRow:
    return OmniRow(
        prompt=str(payload.get("prompt") or ""),
        intent=str(payload.get("intent") or "general"),
        response_text=str(payload.get("response_text") or ""),
        domain=_safe_domain(str(payload.get("domain") or "general")),
        source=str(payload.get("source") or "teacher_resume_v7"),
        image_path=str(payload.get("image_path")) if payload.get("image_path") else None,
        vision_label=str(payload.get("vision_label")) if payload.get("vision_label") else None,
    )


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    if text:
        text += "\n"
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _teacher_resume_dir_v7(repo_root: Path, seed: int, limit: int, teacher_keys: Sequence[str]) -> Path:
    teacher_tag = f"teachers_{len(teacher_keys)}"
    return (repo_root / "output" / "omni_v7_teacher_resume" / f"seed_{int(seed)}_limit_{int(limit)}_{teacher_tag}").resolve()


def _teacher_sample_path_v7(resume_dir: Path) -> Path:
    return resume_dir / "teacher_sample.jsonl"


def _teacher_manifest_path_v7(resume_dir: Path) -> Path:
    return resume_dir / "teacher_manifest.json"


def _teacher_state_path_v7(resume_dir: Path, teacher_key: str) -> Path:
    return resume_dir / f"{teacher_key}.json"


def _save_teacher_sample_v7(sample_path: Path, rows: Sequence[OmniRow]) -> None:
    _write_jsonl_atomic(sample_path, [_omni_row_to_payload(row) for row in rows])


def _load_teacher_sample_v7(sample_path: Path) -> List[OmniRow]:
    rows: List[OmniRow] = []
    if not sample_path.exists():
        return rows
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        cooked = line.strip()
        if not cooked:
            continue
        try:
            payload = json.loads(cooked)
        except Exception:
            continue
        rows.append(_omni_row_from_payload(payload))
    return rows


def _load_teacher_state_v7(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {
            "status": "missing",
            "rows": {},
            "empty_row_indices": set(),
            "processed_rows": 0,
        }
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "status": "corrupt",
            "rows": {},
            "empty_row_indices": set(),
            "processed_rows": 0,
        }
    rows_payload = payload.get("rows") or []
    rows: Dict[int, Tuple[float, str]] = {}
    for item in rows_payload:
        try:
            row_index = int(item.get("row_index"))
            rows[row_index] = (float(item.get("score") or 0.0), str(item.get("candidate") or ""))
        except Exception:
            continue
    empty_indices = {int(value) for value in (payload.get("empty_row_indices") or []) if str(value).isdigit()}
    return {
        "status": str(payload.get("status") or "partial"),
        "rows": rows,
        "empty_row_indices": empty_indices,
        "processed_rows": int(payload.get("processed_rows") or (len(rows) + len(empty_indices))),
        "teacher": str(payload.get("teacher") or ""),
        "sample_total": int(payload.get("sample_total") or 0),
    }


def _conversation_alignment_rows_v7() -> List[OmniRow]:
    base = list(_conversation_alignment_rows_v6())
    items = [
        (
            "The user wants a direct answer with no filler. What should the assistant optimize for?",
            "Optimize for directness, preserve the requested format, and remove stock framing that does not advance the task.",
            "general",
        ),
        (
            "Why is it useful for a local model to mention uncertainty instead of inventing details?",
            "Because an explicit uncertainty statement preserves trust and still moves the task forward, while invented details create confident but fragile failures.",
            "knowledge",
        ),
        (
            "What should a model do if a user asks for a benchmark comparison between two local models?",
            "Compare the actual recorded strengths, benchmark scores, and capabilities instead of guessing from names alone.",
            "model_selection",
        ),
        (
            "Answer in one short paragraph: why should a coding assistant stay close to the repo context?",
            "Staying close to the repo context keeps fixes compatible with the real code, avoids generic advice, and reduces the chance of proposing changes that do not fit the current implementation.",
            "coding",
        ),
    ]
    base.extend(
        [
            OmniRow(
                prompt=prompt,
                intent="model_selection" if domain == "model_selection" else ("coding" if domain == "coding" else "general"),
                response_text=_normalize(response, 420),
                domain=_safe_domain(domain),
                source="conversation_alignment_v7",
            )
            for prompt, response, domain in items
        ]
    )
    return base


def _grounding_rows_v7() -> List[OmniRow]:
    base = list(_grounding_rows_v6())
    items = [
        (
            "How should a grounded assistant answer a question about a local model it has never loaded?",
            "It should describe only the catalog facts it can verify and avoid claiming benchmark or runtime behavior it has not actually observed.",
            "knowledge",
        ),
        (
            "What is the safer answer when a request depends on a file that might not exist?",
            "Inspect or verify the file path first; if that cannot be done yet, say so instead of pretending the file is present.",
            "coding",
        ),
        (
            "Why is narrowing the answer often better than improvising extra details?",
            "A narrower answer stays closer to the evidence you have, which makes it more reliable than a broader answer padded with unsupported specifics.",
            "general",
        ),
        (
            "How should a local model behave when it is only weakly confident about a domain classification?",
            "It should default to a grounded general answer, keep claims bounded, and avoid overcommitting to a specialist tone it cannot support.",
            "planning",
        ),
    ]
    base.extend(
        [
            OmniRow(
                prompt=prompt,
                intent="coding" if domain == "coding" else "knowledge",
                response_text=_normalize(response, 420),
                domain=_safe_domain(domain),
                source="grounding_v7",
            )
            for prompt, response, domain in items
        ]
    )
    return base


def _conversation_expansion_rows_v7(repo_root: Path, seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    datasets_dir = repo_root / "datasets"
    rows: List[OmniRow] = []
    counts: Dict[str, int] = defaultdict(int)
    dataset_specs = [
        ("conversation_data.supermix_plus_v27_500k.jsonl", 2600, "knowledge", "conversation_supermix_plus_v7"),
        ("conversation_data.mega_reasoning_creative_v25_75582.jsonl", 1320, "planning", "conversation_reasoning_v7"),
        ("conversation_data.mega_creative_250k_v2.jsonl", 1400, "creative", "conversation_creative_v7"),
        ("conversation_data.book_extracts_public_domain_v2_120k.jsonl", 920, "language", "conversation_books_v7"),
        ("conversation_data.coding_knowledge_2026_02_19.jsonl", 420, "coding", "conversation_coding_v7"),
        ("conversation_data.dictionary_wordnet_meanings_smoke.jsonl", 260, "language", "conversation_dictionary_v7"),
        ("conversation_data.science_essentials_smoke.jsonl", 240, "knowledge", "conversation_science_v7"),
        ("conversation_data.science_novel_examples_smoke.jsonl", 220, "knowledge", "conversation_science_novel_v7"),
        ("conversation_data.world_events_2026_02_19.jsonl", 220, "knowledge", "conversation_world_events_v7"),
        ("conversation_data.english_math_smoke_v3.jsonl", 240, "math", "conversation_math_v7"),
    ]
    for rel_name, limit, domain, source_tag in dataset_specs:
        path = datasets_dir / rel_name
        if not path.exists():
            continue
        sampled = _rows_from_jsonl(path, limit=limit, seed=seed + len(rows) + limit, domain=domain, source_tag=source_tag)
        rows.extend(sampled)
        counts[source_tag] += len(sampled)
    return rows, dict(sorted(counts.items()))


def _benchmax_rows_v7(repo_root: Path, seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    counts: Dict[str, int] = {}
    v33_rows, _ = build_v33_style_rows(repo_root, seed=seed + 31, sample_size=1792)
    v39_rows, _ = build_v39_style_rows(repo_root, seed=seed + 57, sample_size=2048)
    for source_tag, payload_rows in (("benchmax_v33_v7", v33_rows), ("benchmax_v39_v7", v39_rows)):
        local_count = 0
        for item in payload_rows:
            cooked = prompt_row_to_omni(item, source=source_tag)
            prompt = _normalize(cooked.get("prompt") or "", 260)
            response = _normalize(cooked.get("response_text") or "", 420)
            if len(prompt) < 8 or len(response) < 4:
                continue
            rows.append(
                OmniRow(
                    prompt=prompt,
                    intent=str(cooked.get("intent") or "general"),
                    response_text=response,
                    domain=_safe_domain(str(cooked.get("domain") or "general")),
                    source=source_tag,
                )
            )
            local_count += 1
        counts[source_tag] = local_count
    return rows, counts


def _record_score_fragment(repo_root: Path, record: ModelRecord) -> str:
    if record.common_overall_exact is not None:
        return f"Its common benchmark score is {record.common_overall_exact:.4f}."
    if record.recipe_eval_accuracy is not None:
        return f"Its internal validation score is {record.recipe_eval_accuracy:.4f}."
    if record.kind == "dcgan_image":
        for path in sorted((repo_root / "output").rglob(f"{record.key}_benchmark_summary.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            score = payload.get("specialist_score")
            if score is not None:
                return f"Its specialist GAN score is {float(score):.4f}."
    return ""


def _specialist_profile_rows_v7(repo_root: Path, models_dir: Path, seed: int) -> Tuple[List[OmniRow], Dict[str, object]]:
    del seed
    records = sorted(discover_model_records(models_dir=models_dir), key=lambda item: item.key)
    rows: List[OmniRow] = []
    family_map: Dict[str, List[str]] = defaultdict(list)
    non_chat_keys: List[str] = []
    for record in records:
        family_map[record.family].append(record.label)
        if not record.supports_chat:
            non_chat_keys.append(record.key)
        family_text = record.family.replace("_", " ")
        kind_text = record.kind.replace("_", " ")
        capability_text = ", ".join(sorted(record.capabilities)) or "specialist"
        score_text = _record_score_fragment(repo_root, record)
        use_text = str(record.benchmark_hint or record.note or f"{record.label} is a local specialist model.").strip()
        base_answer = _normalize(
            f"{record.label} is a {family_text} {kind_text} model with capabilities {capability_text}. "
            f"{record.note or ''} Best use: {use_text} {score_text}".strip(),
            420,
        )
        rows.append(
            OmniRow(
                prompt=f"What is {record.label} best used for in the local Supermix catalog?",
                intent="model_selection",
                response_text=base_answer,
                domain="model_selection",
                source=f"{record.key}_profile_v7",
            )
        )
        if record.supports_chat:
            response = _normalize(
                f"Choose {record.label} when the task matches its strongest local use case: {use_text} {score_text}".strip(),
                420,
            )
        else:
            response = _normalize(
                f"{record.label} is not a normal chat consultant. Use it through its specialist pipeline for {use_text.lower()} rather than expecting ordinary back-and-forth chat.",
                420,
            )
        rows.append(
            OmniRow(
                prompt=f"When should I choose {record.label} instead of a generic local chat model?",
                intent="model_selection",
                response_text=response,
                domain="model_selection",
                source=f"{record.key}_selection_v7",
            )
        )
        if record.kind == "dcgan_image":
            spec = DCGAN_SPECS.get(record.key)
            if spec is not None:
                rows.append(
                    OmniRow(
                        prompt=f"How does {record.label} behave when prompted with text?",
                        intent="model_selection",
                        response_text=_normalize(
                            f"{record.label} is an unconditional GAN. The text prompt only seeds latent sampling; it does not understand the prompt semantically like a chat model.",
                            420,
                        ),
                        domain="image_prompt",
                        source=f"{record.key}_gan_v7",
                    )
                )
        elif record.kind == "image_recognition":
            rows.append(
                OmniRow(
                    prompt=f"What kind of uploaded images is {record.label} strongest at?",
                    intent="vision",
                    response_text=_normalize(
                        f"{record.label} is strongest on the bundled science-diagram style categories such as {', '.join(name.replace('_', ' ') for name in SCIENCE_IMAGE_CLASSES[:5])}.",
                        420,
                    ),
                    domain="vision",
                    source=f"{record.key}_vision_v7",
                )
            )
        elif record.kind == "protein_folding":
            rows.append(
                OmniRow(
                    prompt="Which local model is the protein-folding specialist?",
                    intent="model_selection",
                    response_text=_normalize(f"{record.label} is the protein-folding specialist in the current local model catalog.", 240),
                    domain="model_selection",
                    source=f"{record.key}_protein_v7",
                )
            )
        elif record.kind == "math_equation":
            rows.append(
                OmniRow(
                    prompt="Which local model is best for exact symbolic math and equation solving?",
                    intent="model_selection",
                    response_text=_normalize(f"{record.label} is the exact symbolic math specialist in the local model catalog.", 240),
                    domain="model_selection",
                    source=f"{record.key}_math_v7",
                )
            )
    for family, labels in sorted(family_map.items()):
        label_text = ", ".join(sorted(labels))
        rows.append(
            OmniRow(
                prompt=f"Which local models belong to the {family.replace('_', ' ')} family?",
                intent="model_selection",
                response_text=_normalize(f"The {family.replace('_', ' ')} family currently includes: {label_text}.", 420),
                domain="model_selection",
                source=f"{family}_family_map_v7",
            )
        )
    if non_chat_keys:
        rows.append(
            OmniRow(
                prompt="Which local models are specialist generators or non-chat systems rather than normal chat assistants?",
                intent="model_selection",
                response_text=_normalize(
                    "The non-chat specialist models include: "
                    + ", ".join(sorted(non_chat_keys))
                    + ". Use them through image generation or specialist pipelines instead of normal text chat.",
                    420,
                ),
                domain="model_selection",
                source="non_chat_specialists_v7",
            )
        )
    return rows, {
        "record_keys": [record.key for record in records],
        "record_count": len(records),
        "non_chat_record_keys": sorted(non_chat_keys),
        "family_counts": {family: len(labels) for family, labels in sorted(family_map.items())},
        "row_count": len(rows),
    }


def _teacher_order_key_v7(record_key: str) -> Tuple[int, str]:
    priority = {
        "v40_benchmax": 0,
        "qwen_v28": 1,
        "qwen_v30": 2,
        "omni_collective_v6": 3,
        "omni_collective_v5": 4,
        "v33_final": 5,
        "omni_collective_v4": 6,
        "omni_collective_v3": 7,
    }
    return (priority.get(record_key, 20), record_key)


def _model_style_for_row_v7(row: OmniRow) -> str:
    if row.domain in {"coding", "model_selection"}:
        return "coding"
    if row.domain in {"creative", "image_prompt", "language"}:
        return "creative"
    return "analyst"


def _teacher_worker_python() -> str:
    venv_python = Path(str(os.environ.get("VIRTUAL_ENV") or "")).resolve() / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run_teacher_worker_v7(
    *,
    repo_root: Path,
    models_dir: Path,
    teacher_key: str,
    sample_path: Path,
    state_path: Path,
    stall_seconds: int = 1200,
    poll_seconds: int = 10,
) -> int:
    worker_script = SOURCE_DIR / "run_omni_collective_v7_teacher.py"
    cmd = [
        _teacher_worker_python(),
        "-u",
        str(worker_script),
        "--repo_root",
        str(repo_root),
        "--models_dir",
        str(models_dir),
        "--teacher_key",
        str(teacher_key),
        "--sample_jsonl",
        str(sample_path),
        "--state_json",
        str(state_path),
    ]
    process = subprocess.Popen(cmd, cwd=str(repo_root))
    last_state_mtime: Optional[float] = state_path.stat().st_mtime if state_path.exists() else None
    last_progress_ts = time.time()
    while True:
        return_code = process.poll()
        current_mtime = state_path.stat().st_mtime if state_path.exists() else None
        if current_mtime is not None and current_mtime != last_state_mtime:
            last_state_mtime = current_mtime
            last_progress_ts = time.time()
        if return_code is not None:
            return int(return_code)
        if time.time() - last_progress_ts > max(int(stall_seconds), 60):
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=30)
            except Exception:
                pass
            return 124
        time.sleep(max(int(poll_seconds), 1))


def _aggregate_teacher_states_v7(
    teacher_states: Dict[str, Dict[str, Any]],
    sample_total: int,
) -> Tuple[Dict[int, Tuple[float, str, str]], Dict[int, List[Tuple[float, str, str]]], Dict[str, int], List[str], List[str]]:
    best_by_index: Dict[int, Tuple[float, str, str]] = {}
    candidates_by_index: Dict[int, List[Tuple[float, str, str]]] = defaultdict(list)
    empty_counts: Dict[str, int] = {}
    complete_teachers: List[str] = []
    partial_teachers: List[str] = []
    for teacher_key, state in teacher_states.items():
        rows = state.get("rows") or {}
        empty_indices = state.get("empty_row_indices") or set()
        empty_counts[teacher_key] = len(empty_indices)
        if str(state.get("status") or "") == "complete":
            complete_teachers.append(teacher_key)
        elif rows or empty_indices:
            partial_teachers.append(teacher_key)
        for row_index, payload in rows.items():
            score, candidate = payload
            candidates_by_index[int(row_index)].append((float(score), teacher_key, str(candidate)))
            best = best_by_index.get(int(row_index))
            if best is None or float(score) > best[0]:
                best_by_index[int(row_index)] = (float(score), teacher_key, str(candidate))
    for teacher_key in empty_counts:
        empty_counts[teacher_key] = min(sample_total, max(0, int(empty_counts[teacher_key])))
    return best_by_index, candidates_by_index, dict(sorted(empty_counts.items())), sorted(complete_teachers), sorted(partial_teachers)


def _all_model_distill_rows_v7(
    rows: Sequence[OmniRow],
    *,
    repo_root: Path,
    models_dir: Path,
    seed: int,
    limit: int,
    teacher_model_limit: int,
) -> Tuple[List[OmniRow], Dict[str, object]]:
    sample = _sample_teacher_rows(rows, seed=seed, limit=limit)
    if not str(os.environ.get("SUPERMIX_QWEN_BASE_MODEL_DIR") or "").strip():
        try:
            os.environ["SUPERMIX_QWEN_BASE_MODEL_DIR"] = str(_resolve_local_qwen_base_model())
        except Exception:
            pass
    all_records = sorted(discover_model_records(models_dir=models_dir), key=lambda item: item.key)
    teacher_records = [record for record in all_records if record.supports_chat and record.key != "omni_collective_v7"]
    teacher_records.sort(key=lambda item: _teacher_order_key_v7(item.key))
    if teacher_model_limit > 0:
        teacher_records = teacher_records[: int(teacher_model_limit)]

    resume_dir = _teacher_resume_dir_v7(repo_root=repo_root, seed=seed, limit=limit, teacher_keys=[record.key for record in teacher_records])
    resume_dir.mkdir(parents=True, exist_ok=True)
    sample_path = _teacher_sample_path_v7(resume_dir)
    if sample_path.exists():
        cached_sample = _load_teacher_sample_v7(sample_path)
        if cached_sample:
            sample = cached_sample
    else:
        _save_teacher_sample_v7(sample_path, sample)
    _write_json_atomic(
        _teacher_manifest_path_v7(resume_dir),
        {
            "seed": int(seed),
            "limit": int(limit),
            "sample_total": len(sample),
            "teacher_keys": [record.key for record in teacher_records],
        },
    )

    direct_counts: Dict[str, int] = defaultdict(int)
    repair_counts: Dict[str, int] = defaultdict(int)
    unavailable_teachers: Dict[str, str] = {}
    timed_out_teachers: List[str] = []
    teacher_states: Dict[str, Dict[str, Any]] = {}

    for teacher_idx, record in enumerate(teacher_records, start=1):
        state_path = _teacher_state_path_v7(resume_dir, record.key)
        teacher_states[record.key] = _load_teacher_state_v7(state_path)
        if teacher_states[record.key].get("status") == "complete":
            print(
                json.dumps(
                    {
                        "event": "teacher_model_resume_hit",
                        "teacher_index": teacher_idx,
                        "teacher_total": len(teacher_records),
                        "teacher": record.key,
                        "processed_rows": int(teacher_states[record.key].get("processed_rows") or 0),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {
                    "event": "teacher_model_start",
                    "teacher_index": teacher_idx,
                    "teacher_total": len(teacher_records),
                    "teacher": record.key,
                    "resume_processed_rows": int(teacher_states[record.key].get("processed_rows") or 0),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        return_code = _run_teacher_worker_v7(
            repo_root=repo_root,
            models_dir=models_dir,
            teacher_key=record.key,
            sample_path=sample_path,
            state_path=state_path,
        )
        teacher_states[record.key] = _load_teacher_state_v7(state_path)
        if return_code == 124:
            timed_out_teachers.append(record.key)
            print(
                json.dumps(
                    {
                        "event": "teacher_model_timeout",
                        "teacher_index": teacher_idx,
                        "teacher_total": len(teacher_records),
                        "teacher": record.key,
                        "processed_rows": int(teacher_states[record.key].get("processed_rows") or 0),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        if return_code != 0 and not teacher_states[record.key].get("rows") and not teacher_states[record.key].get("empty_row_indices"):
            unavailable_teachers[record.key] = f"worker_exit_{return_code}"
            print(
                json.dumps(
                    {
                        "event": "teacher_model_unavailable",
                        "teacher_index": teacher_idx,
                        "teacher_total": len(teacher_records),
                        "teacher": record.key,
                        "reason": unavailable_teachers[record.key],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    best_by_index, candidates_by_index, empty_counts, complete_teachers, partial_teachers = _aggregate_teacher_states_v7(
        teacher_states,
        sample_total=len(sample),
    )

    accepted: List[OmniRow] = []
    consensus_rows: List[OmniRow] = []
    discarded = 0
    for row_index, row in enumerate(sample, start=1):
        candidates = sorted(candidates_by_index.get(row_index, []), key=lambda item: item[0], reverse=True)
        best = best_by_index.get(row_index)
        if best is None:
            discarded += 1
            continue
        score, teacher_key, candidate = best
        if score >= 0.26:
            accepted.append(
                OmniRow(
                    prompt=row.prompt,
                    intent=row.intent,
                    response_text=_normalize(candidate, 420),
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source=f"{teacher_key}_distill_v7",
                )
            )
            direct_counts[teacher_key] += 1
        elif score >= 0.12:
            accepted.append(
                OmniRow(
                    prompt=_normalize(
                        "Repair and ground this draft answer so it becomes concise, correct, and less speculative.\n"
                        f"Request: {row.prompt}\n"
                        f"Draft: {candidate}",
                        360,
                    ),
                    intent=row.intent,
                    response_text=row.response_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source=f"{teacher_key}_repair_v7",
                )
            )
            repair_counts[teacher_key] += 1
        else:
            discarded += 1

        if len(candidates) >= 2 and len(consensus_rows) < max(18, int(limit) // 3):
            first, second = candidates[0], candidates[1]
            if first[0] >= 0.18 and second[0] >= 0.18 and first[2].strip().lower() != second[2].strip().lower():
                consensus_rows.append(
                    OmniRow(
                        prompt=_normalize(
                            "Synthesize the strongest grounded answer from these teacher drafts.\n"
                            f"Request: {row.prompt}\n"
                            f"Draft A ({first[1]}): {first[2]}\n"
                            f"Draft B ({second[1]}): {second[2]}",
                            360,
                        ),
                        intent=row.intent,
                        response_text=row.response_text,
                        domain=row.domain,
                        image_path=row.image_path,
                        vision_label=row.vision_label,
                        source="teacher_consensus_v7",
                    )
                )

        if row_index % 8 == 0 or row_index == len(sample):
            print(
                json.dumps(
                    {
                        "event": "teacher_league_progress",
                        "completed": row_index,
                        "total": len(sample),
                        "accepted": len(accepted),
                        "consensus": len(consensus_rows),
                        "discarded": discarded,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    gc.collect()
    accepted.extend(consensus_rows)
    return accepted, {
        "requested": int(limit),
        "sampled": len(sample),
        "accepted_total": len(accepted),
        "accepted_direct": dict(sorted(direct_counts.items())),
        "accepted_repair": dict(sorted(repair_counts.items())),
        "accepted_consensus": len(consensus_rows),
        "empty_counts": dict(sorted(empty_counts.items())),
        "discarded": discarded,
        "teacher_keys": [record.key for record in teacher_records],
        "all_record_keys": [record.key for record in all_records],
        "non_chat_record_keys": [record.key for record in all_records if not record.supports_chat],
        "unavailable_teachers": dict(sorted(unavailable_teachers.items())),
        "timed_out_teachers": sorted(timed_out_teachers),
        "complete_teachers": complete_teachers,
        "partial_teachers": partial_teachers,
        "resume_dir": str(resume_dir),
        "forced_priority_keys": ["v40_benchmax", "qwen_v28", "qwen_v30", "omni_collective_v6"],
    }


def build_training_rows(
    *,
    repo_root: Path,
    models_dir: Path,
    images_dir: Path,
    seed: int,
    distill_limit: int,
    teacher_model_limit: int,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, object]]:
    full_rows, source_counts = _build_rows_v6(repo_root=repo_root, models_dir=models_dir, images_dir=images_dir, seed=seed)
    conversation_rows, conversation_counts = _conversation_expansion_rows_v7(repo_root=repo_root, seed=seed + 601)
    benchmax_rows, benchmax_counts = _benchmax_rows_v7(repo_root=repo_root, seed=seed + 641)
    specialist_rows, specialist_summary = _specialist_profile_rows_v7(repo_root=repo_root, models_dir=models_dir, seed=seed + 659)
    grounding_rows = _grounding_rows_v7()
    alignment_rows = _conversation_alignment_rows_v7()
    math_rows = _math_exact_rows_v6(repo_root=repo_root, seed=seed + 677, limit=96)
    protein_rows = _protein_rows_v6(seed=seed + 701, limit=240)
    protein_pack_rows, protein_pack_summary = build_protein_folding_rows(seed=seed + 727, max_rows=220)
    protein_pack_cooked = [
        OmniRow(
            prompt=_normalize(str(item.get("prompt") or ""), 260),
            intent=str(item.get("intent") or "knowledge"),
            response_text=_normalize(str(item.get("response_text") or item.get("assistant") or ""), 420),
            domain=_safe_domain(str(item.get("domain") or "knowledge")),
            source="protein_pack_v7",
        )
        for item in protein_pack_rows
        if len(_normalize(str(item.get("prompt") or ""), 260)) >= 8 and len(_normalize(str(item.get("response_text") or item.get("assistant") or ""), 420)) >= 8
    ]

    full_rows.extend(conversation_rows)
    full_rows.extend(benchmax_rows)
    full_rows.extend(specialist_rows)
    full_rows.extend(grounding_rows)
    full_rows.extend(alignment_rows)
    full_rows.extend(math_rows)
    full_rows.extend(protein_rows)
    full_rows.extend(protein_pack_cooked)

    for key, value in conversation_counts.items():
        source_counts[key] = source_counts.get(key, 0) + int(value)
    for key, value in benchmax_counts.items():
        source_counts[key] = source_counts.get(key, 0) + int(value)
    source_counts["specialist_profiles_v7"] = len(specialist_rows)
    source_counts["grounding_v7"] = len([row for row in grounding_rows if row.source == "grounding_v7"])
    source_counts["conversation_alignment_v7"] = len([row for row in alignment_rows if row.source == "conversation_alignment_v7"])
    source_counts["math_exact_v7_added"] = len(math_rows)
    source_counts["protein_folding_v7_added"] = len(protein_rows)
    source_counts["protein_pack_v7"] = len(protein_pack_cooked)

    distill_rows, distill_summary = _all_model_distill_rows_v7(
        full_rows,
        repo_root=repo_root,
        models_dir=models_dir,
        seed=seed + 787,
        limit=distill_limit,
        teacher_model_limit=teacher_model_limit,
    )
    full_rows.extend(distill_rows)
    source_counts["all_model_distill_total_v7"] = len(distill_rows)

    pre_dedupe_rows = len(full_rows)
    full_rows = _dedupe_rows(full_rows)
    rng = random.Random(int(seed))
    rng.shuffle(full_rows)
    stage1_rows = [row for row in full_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    return stage1_rows, list(full_rows), {
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(full_rows),
        "pre_dedupe_rows": pre_dedupe_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "teacher_league": distill_summary,
        "specialist_profiles": specialist_summary,
        "protein_pack_summary": protein_pack_summary,
    }


def train_model(
    *,
    repo_root: Path,
    output_dir: Path,
    models_dir: Path,
    base_zip: Path,
    images_dir: Path,
    image_size: int,
    batch_size: int,
    stage1_epochs: int,
    stage2_epochs: int,
    stage1_lr: float,
    stage2_lr: float,
    seed: int,
    distill_limit: int,
    teacher_model_limit: int,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    _stage1_rows, full_rows, dataset_summary = build_training_rows(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        seed=seed,
        distill_limit=distill_limit,
        teacher_model_limit=teacher_model_limit,
    )
    print(json.dumps({"event": "dataset_built", "summary": dataset_summary}, ensure_ascii=True), flush=True)
    train_rows, val_rows = split_rows(full_rows, seed=seed + 17)
    train_stage1 = [row for row in train_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    vocab = build_char_vocab([row.prompt for row in train_rows], min_frequency=1)
    response_bank = sorted({row.response_text for row in full_rows if row.response_text})
    print(
        json.dumps(
            {
                "event": "label_space",
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "response_bank": len(response_bank),
                "vocab_size": len(vocab),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    max_len = 384
    max_words = 84
    word_buckets = 16384
    device = torch.device("cpu")
    model = OmniCollectiveNetV7(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
        base_embed_dim=116,
        text_hidden=236,
        image_channels=48,
        word_buckets=word_buckets,
        word_embed_dim=108,
        deep_text_channels=336,
        deep_image_channels=112,
        fusion_hidden=960,
        memory_slots=22,
        depth_steps=9,
        expert_count=8,
        expert_hidden=1536,
        context_top_k=4,
        expert_top_k=2,
    ).to(device)
    warm_start = _load_expanded_state_from_zip(model, base_zip)
    print(json.dumps({"event": "warm_start", "info": warm_start}, ensure_ascii=True), flush=True)

    stage1 = _train_stage(
        model=model,
        train_rows=train_stage1,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=batch_size,
        learning_rate=stage1_lr,
        epochs=stage1_epochs,
        seed=seed + 101,
        device=device,
        loss_weights={"intent": 0.58, "response": 1.00, "domain": 0.80, "vision": 0.76},
        balance_weight=0.032,
    )
    stage2 = _train_stage(
        model=model,
        train_rows=train_rows,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=max(12, int(batch_size) // 2),
        learning_rate=stage2_lr,
        epochs=stage2_epochs,
        seed=seed + 151,
        device=device,
        loss_weights={"intent": 0.54, "response": 1.00, "domain": 0.78, "vision": 1.04},
        balance_weight=0.045,
    )

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_omni_collective_v7_frontier_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "omni_collective_v7_frontier.pth"
    meta_path = artifact_dir / "omni_collective_v7_frontier_meta.json"
    summary_path = artifact_dir / "omni_collective_v7_frontier_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v7_frontier_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = ensure_base_images(images_dir)
    meta = {
        "architecture_version": 7,
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": class_info,
        "intent_labels": list(OMNI_INTENTS_V2),
        "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
        "max_len": max_len,
        "image_size": int(image_size),
        "embed_dim": 116,
        "text_hidden": 236,
        "image_channels": 48,
        "word_buckets": word_buckets,
        "max_words": max_words,
        "word_embed_dim": 108,
        "deep_text_channels": 336,
        "deep_image_channels": 112,
        "fusion_hidden": 960,
        "memory_slots": 22,
        "depth_steps": 9,
        "expert_count": 8,
        "expert_hidden": 1536,
        "context_top_k": 4,
        "expert_top_k": 2,
        "parameter_count": parameter_count,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "seed": int(seed),
        "deliberation_passes": 8,
        "minimum_passes": 4,
        "grounding_threshold": 0.50,
        "prompt_understanding_mode": "all_model_grounded_consensus_math_protein_vision_conversation",
        "notes": [
            "v7 distills every discovered local model family by combining a full chat-teacher league with explicit specialist-profile supervision for non-chat models.",
            "The continuation grows v6 again, adds more conversation and benchmark-style rows, and increases internal deliberation depth for more grounded answers.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    engine = OmniCollectiveEngineV7(weights_path=weights_path, meta_path=meta_path, device=device)
    sample_prompts = [
        "Reply in exactly two sentences explaining why regression tests matter.",
        "Solve 3*x + 7 = 19.",
        "What should a grounded assistant do when it cannot verify a current fact locally?",
        "Write a tiny OpenSCAD snippet for a centered cylinder with a hole.",
        "Why do multiple-sequence alignments help protein structure prediction?",
        "Which local model is best for benchmark-focused reasoning prompts?",
    ]
    summary = {
        "artifact": zip_path.name,
        "parameter_count": parameter_count,
        "dataset_summary": dataset_summary,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "sample_outputs": [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts],
        "notes": [
            "v7 grows v6 again and blends full-catalog distillation, specialist-profile supervision, more benchmark rows, and heavier conversation/math/protein grounding.",
            "Inference uses a longer all-model grounded deliberation loop to improve prompt understanding and reduce hallucinated specifics.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(weights_path, arcname=weights_path.name)
        archive.write(meta_path, arcname=meta_path.name)
        archive.write(summary_path, arcname=summary_path.name)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, desktop_zip_path)
    return {
        "zip_path": str(zip_path),
        "desktop_zip_path": str(desktop_zip_path),
        "artifact_dir": str(artifact_dir),
        "parameter_count": parameter_count,
        "stage1_val": stage1["val_metrics"],
        "stage2_val": stage2["val_metrics"],
        "warm_start": warm_start,
        "dataset_summary": dataset_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the omni_collective_v7 continuation model.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--base_zip", default=str(DEFAULT_BASE_ZIP))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--stage1_epochs", type=int, default=1)
    parser.add_argument("--stage2_epochs", type=int, default=1)
    parser.add_argument("--stage1_lr", type=float, default=0.00034)
    parser.add_argument("--stage2_lr", type=float, default=0.00015)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--distill_limit", type=int, default=104)
    parser.add_argument("--teacher_model_limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    if args.dry_run:
        stage1_rows, full_rows, summary = build_training_rows(
            repo_root=repo_root,
            models_dir=models_dir,
            images_dir=images_dir,
            seed=int(args.seed),
            distill_limit=int(args.distill_limit),
            teacher_model_limit=int(args.teacher_model_limit),
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "stage1_rows": len(stage1_rows),
                    "stage2_rows": len(full_rows),
                    "dataset_summary": summary,
                },
                indent=2,
            )
        )
        return
    result = train_model(
        repo_root=repo_root,
        output_dir=output_dir,
        models_dir=models_dir,
        base_zip=Path(args.base_zip).resolve(),
        images_dir=images_dir,
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=int(args.stage2_epochs),
        stage1_lr=float(args.stage1_lr),
        stage2_lr=float(args.stage2_lr),
        seed=int(args.seed),
        distill_limit=int(args.distill_limit),
        teacher_model_limit=int(args.teacher_model_limit),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
