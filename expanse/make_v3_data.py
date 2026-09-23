"""Expanse v3 data: more verified rows from the generators the replay corpus came from (V3_DESIGN.md B.2).

    python expanse/make_v3_data.py                          # full targets -> data/v3/
    python expanse/make_v3_data.py --omni 300 --code 200 --math 200 --out_dir <scratch>   # quick look

Why fresh rows, and why from these generators
    Archimedes' replay corpus (``supermix-archimedes/corpus``, 3,743 rows) was drawn
    from three generators whose every answer is checked by an independent code
    path: ``build_omni_corpus`` (12 science tasks, each worked answer re-derived by
    the NexusMind solver and dropped on disagreement), ``build_code_corpus`` (9
    code-reading tasks, each answer taken from *executing* the snippet the prompt
    shows) and ``build_scratchpad_math`` (place-value working). Rerunning them with
    new seeds gives more rows of exactly the kind the trunk already learned from,
    correct by construction -- the cheapest honest way to "train on much more
    data" without a teacher. They are vendored (MIT, provenance in their README)
    under ``supermix-archimedes/models/supermix-v93/src`` and run as subprocesses
    with the CLI their docs describe; ``per_task = ceil(target / n_tasks)``.
Seeds
    79 / 87 / 66 are the builders' defaults and 2026 is the seed the replay corpus
    was built with; the fresh seeds (3079 / 3087 / 3066) must differ from all four,
    so a fresh row is a new draw rather than a replay row under a new name.
Filters (a dropped row is counted, per reason, in ``fresh.report.json``)
    replay_prompt        ``user`` equals a replay-corpus prompt: the same question
                         would count twice (and might carry the other's answer
                         format)
    eval_heldout_prompt  ``user`` equals a prompt ``eval_expanse.heldout_problems``
                         can ask (every prompt it yields for each ``--heldout_seeds``
                         seed; it is prefix-stable in ``n``, so this covers any
                         ``--gen_limit``): the exact-answer metric must stay unseen
    duplicate            ``user`` already kept (from this family or an earlier one;
                         order omni, code, math), so no prompt carries two replies
Split
    3% of the kept rows are ``split = "heldout"`` by a stable hash of ``user``
    alone (same prompt -> same side, whatever else changes); the trainer never
    trains on them. The rest are ``train``; the trainer carves its dev rows out of
    those with its usual hash split. Rows keep the builder's fields and gain
    ``source = "fresh"``, ``family`` and ``split``.
Connectome
    ``connectome_text`` again, with v1's seed and held-out *type* fraction (read
    from ``data/connectome_rows.report.json``) but ``--n_rows 12000``. The held-out
    type pool depends only on (type names, seed, fraction), so it must come out
    identical to v1's -- asserted (pool equal, no v1 held-out type in a v3 train
    row, every v1 held-out row's type still held out) *before* anything is written:
    a v3 model's "unseen types" score is only comparable with v1's if the types
    are the same ones.

Outputs (``--out_dir``, default ``data/v3``): ``fresh_omni.jsonl``, ``fresh_code.jsonl``,
``fresh_math.jsonl``, ``fresh.report.json``, ``connectome_rows_v3.jsonl`` and its
``.report.json``. Everything is deterministic in the arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import expanse_core as ec  # noqa: E402
from expanse_core import PATHS  # noqa: E402

BUILDERS = PATHS["repo"] / "models" / "supermix-v93" / "src"
V3_DIR = PATHS["data"] / "v3"
REPORT_SCHEMA = "supermix-expanse-v3-fresh-v1"
FAMILIES = ("omni", "code", "math")
#: family -> (builder script, importable module (task count), fresh seed, builder default seed)
BUILDER: Dict[str, Tuple[str, Optional[str], int, int]] = {
    "omni": ("build_omni_corpus.py", "build_omni_corpus", 3079, 79),
    "code": ("build_code_corpus.py", "build_code_corpus", 3087, 87),
    "math": ("build_scratchpad_math.py", None, 3066, 66),
}
#: the builders' default seeds and the replay corpus's build seed: a fresh seed may be none of them
USED_SEEDS = frozenset({79, 87, 66, 2026})
HELDOUT_FRAC = 0.03
DROP_REASONS = ("replay_prompt", "eval_heldout_prompt", "duplicate")


def log(*a) -> None:
    print("[v3data]", *a, flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ec.jsonable(obj), indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# split + filters (pure; tested in tests/test_v3_data.py)
# ---------------------------------------------------------------------------
def heldout_split(user: str, frac: float = HELDOUT_FRAC) -> str:
    """``"heldout"`` for a stable ``frac`` of prompts (hash of ``user`` only), else ``"train"``."""
    h = int(hashlib.sha1(f"expanse-v3-heldout|{user}".encode("utf-8")).hexdigest()[:12], 16)
    return "heldout" if h < frac * float(1 << 48) else "train"


def filter_fresh(built: Mapping[str, Sequence[Dict[str, Any]]], blocked: Mapping[str, Set[str]],
                 frac: float = HELDOUT_FRAC) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, int]]]:
    """Drop blocked / duplicate prompts, tag the survivors.

    ``built``: family -> builder rows, visited in :data:`FAMILIES` order (then any
    other family, sorted). ``blocked``: reason -> prompts, checked in
    :data:`DROP_REASONS` order, so a row is counted under its first reason.
    Returns (family -> kept rows with ``source``/``family``/``split``, family ->
    {reason: dropped}).
    """
    seen: Set[str] = set()
    kept: Dict[str, List[Dict[str, Any]]] = {}
    dropped: Dict[str, Dict[str, int]] = {}
    order = [f for f in FAMILIES if f in built] + sorted(f for f in built if f not in FAMILIES)
    for fam in order:
        out: List[Dict[str, Any]] = []
        drop = {k: 0 for k in DROP_REASONS}
        for r in built[fam]:
            user = r["user"]
            reason = next((k for k in DROP_REASONS[:-1] if user in blocked.get(k, ())), None)
            if reason is None and user in seen:
                reason = "duplicate"
            if reason is not None:
                drop[reason] += 1
                continue
            seen.add(user)
            out.append(dict(r, source="fresh", family=fam, split=heldout_split(user, frac)))
        kept[fam], dropped[fam] = out, drop
    return kept, dropped


def eval_heldout_prompts(seeds: Sequence[int], n: int) -> Tuple[Set[str], Dict[str, int]]:
    """Every prompt ``eval_expanse.heldout_problems(n, seed)`` yields, over ``seeds``."""
    import eval_expanse as ee  # heavy (torch model code); only needed here

    prompts: Set[str] = set()
    per: Dict[str, int] = {}
    for s in seeds:
        ps = ee.heldout_problems(n, int(s))
        per[str(s)] = len(ps)
        prompts |= {p["user"] for p in ps}
    return prompts, per


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def builder_task_count(family: str) -> int:
    """``len(TASKS)`` of the vendored builder (its default task set); 1 for math (``--target`` is a row count)."""
    module = BUILDER[family][1]
    if module is None:
        return 1
    if str(BUILDERS) not in sys.path:
        sys.path.insert(0, str(BUILDERS))
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep the vendored folder free of __pycache__
    try:
        return len(importlib.import_module(module).TASKS)
    finally:
        sys.dont_write_bytecode = old


def builder_command(family: str, target: int, seed: int, output: Path, natural_phrasings: bool) -> Tuple[List[str], Optional[int], int]:
    """(argv, per_task or None, n_tasks) for one builder run."""
    script = BUILDERS / BUILDER[family][0]
    if family == "math":
        return [sys.executable, str(script), "--target", str(target), "--seed", str(seed), "--output", str(output)], None, 1
    n_tasks = builder_task_count(family)
    per_task = int(math.ceil(target / n_tasks))
    cmd = [sys.executable, str(script), "--per_task", str(per_task), "--seed", str(seed), "--unique", "--output", str(output)]
    if family == "omni" and natural_phrasings:
        cmd.append("--natural_phrasings")
    return cmd, per_task, n_tasks


def run_builder(family: str, target: int, seed: int, work: Path, natural_phrasings: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out = work / f"{family}.jsonl"
    cmd, per_task, n_tasks = builder_command(family, target, seed, out, natural_phrasings)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(BUILDERS), env=env, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{BUILDER[family][0]} exited {proc.returncode}:\n{proc.stderr[-3000:]}")
    rows = ec.read_jsonl(out)
    info: Dict[str, Any] = {
        "builder": BUILDER[family][0], "builder_sha256": sha256_file(BUILDERS / BUILDER[family][0]),
        "argv": _argv_for_report(cmd), "seed": seed, "target": target, "n_tasks": n_tasks, "per_task": per_task,
        "built": len(rows), "seconds": round(time.time() - t0, 1),
    }
    rep_path = out.with_suffix(".report.json")
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        info["builder_report"] = {k: rep[k] for k in ("schema", "rows", "per_task", "dropped_failing_verification",
                                                      "dropped_failing_execution", "drop_rate", "short_of_requested",
                                                      "verified_by", "options") if k in rep}
    return rows, info


def _argv_for_report(cmd: Sequence[str]) -> List[str]:
    """Builder argv with the interpreter dropped and the temp output path masked."""
    out = [Path(cmd[1]).name]
    skip = False
    for a in cmd[2:]:
        if skip:
            out.append("<tmp>")
            skip = False
            continue
        out.append(a)
        skip = a == "--output"
    return out


# ---------------------------------------------------------------------------
# connectome
# ---------------------------------------------------------------------------
def heldout_type_check(v3_report: Mapping[str, Any], v3_rows: Sequence[Mapping[str, Any]],
                       v1_report: Mapping[str, Any], v1_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Is v3's held-out type set v1's? (pool equal, no leak into train, v1's held-out rows still held out)."""
    pool_v1 = set(v1_report["heldout_type_names"])
    pool_v3 = set(v3_report["heldout_type_names"])
    train_types = {r["cell_type"] for r in v3_rows if r.get("split") == "train"}
    train_users = {r["user"] for r in v3_rows if r.get("split") == "train"}
    v1_held = [r for r in v1_rows if r.get("split") == "heldout"]
    res = {
        "pool_equal": pool_v1 == pool_v3,
        "pool_size": {"v1": len(pool_v1), "v3": len(pool_v3)},
        "v1_heldout_types_in_v3_train_rows": len(train_types & pool_v1),
        "v1_heldout_rows": len(v1_held),
        "v1_heldout_row_types_not_heldout_in_v3": len({r["cell_type"] for r in v1_held} - pool_v3),
        "v1_heldout_prompts_in_v3_train": sum(1 for r in v1_held if r["user"] in train_users),
    }
    res["equal"] = bool(res["pool_equal"] and not res["v1_heldout_types_in_v3_train_rows"]
                        and not res["v1_heldout_row_types_not_heldout_in_v3"] and not res["v1_heldout_prompts_in_v3_train"])
    return res


def make_connectome(out_path: Path, *, n_rows: int, npz: Path, v1_report_path: Path, v1_rows_path: Path,
                    seed: Optional[int] = None, heldout_fraction: Optional[float] = None) -> Dict[str, Any]:
    import connectome_full as cf
    import connectome_text as ct

    v1 = json.loads(Path(v1_report_path).read_text(encoding="utf-8"))
    seed = int(v1["seed"] if seed is None else seed)
    frac = float(v1["heldout_fraction"] if heldout_fraction is None else heldout_fraction)
    t0 = time.time()
    graph = cf.build_full_graph(npz)
    rows = ct.build_connectome_rows(graph, seed, n_rows, heldout_fraction=frac)
    n_heldout = int(round(n_rows * frac / (1.0 - frac)))
    rows, dropped = ct.verify_rows(rows, npz)
    out_str = str(Path(out_path).resolve()).replace("\\", "/")
    report = ct.rows_report(rows, graph, seed, n_rows, n_heldout, dropped, out_str, frac)
    del graph
    check = heldout_type_check(report, rows, v1, ec.read_jsonl(v1_rows_path))
    check.update({"v1_report": Path(v1_report_path).as_posix(), "v1_seed": v1["seed"], "v1_heldout_fraction": v1["heldout_fraction"],
                  "seed": seed, "heldout_fraction": frac,
                  "npz_sha256_matches_v1": report.get("npz_sha256") == v1.get("npz_sha256")})
    log(f"connectome: {len(rows)} rows ({report['train_rows']} train / {report['heldout_rows']} held out), "
        f"held-out types {check['pool_size']} equal={check['equal']} ({time.time() - t0:.0f}s)")
    if not check["equal"]:
        raise AssertionError(f"v3 connectome held-out types differ from v1's -- nothing written: {json.dumps(check)}")
    report["v3"] = {**check, "schema": "supermix-expanse-v3-connectome-v1",
                    "note": "same seed + held-out fraction as v1 (so the same held-out types), more rows"}
    write_jsonl(Path(out_path), rows)
    write_json(Path(out_path).with_name(Path(out_path).stem + ".report.json"), report)
    return {k: report[k] for k in ("rows", "train_rows", "heldout_rows", "distinct_types", "dropped_failing_verification",
                                   "output")} | {"heldout_type_check": check, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def make_fresh(out_dir: Path, targets: Mapping[str, int], seeds: Mapping[str, int], *, natural_phrasings: bool,
               heldout_seeds: Sequence[int], heldout_n: int, frac: float, corpus_dir: Path) -> Dict[str, Any]:
    t0 = time.time()
    for fam, s in seeds.items():
        if int(s) in USED_SEEDS:
            raise ValueError(f"--seed_{fam} {s} is a seed the replay corpus / builder defaults already used ({sorted(USED_SEEDS)})")
    replay = {r["user"] for r in ec.load_replay(corpus_dir)}
    eval_prompts, eval_per_seed = eval_heldout_prompts(heldout_seeds, heldout_n)
    log(f"blocked prompts: replay {len(replay)}, eval heldout_problems {len(eval_prompts)} {eval_per_seed}")
    built: Dict[str, List[Dict[str, Any]]] = {}
    info: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="expanse_v3_build_") as work:
        for fam in FAMILIES:
            if int(targets.get(fam, 0)) <= 0:
                continue
            built[fam], info[fam] = run_builder(fam, int(targets[fam]), int(seeds[fam]), Path(work), natural_phrasings)
            log(f"{fam}: built {len(built[fam])} rows ({info[fam]['seconds']}s)")
    kept, dropped = filter_fresh(built, {"replay_prompt": replay, "eval_heldout_prompt": eval_prompts}, frac)
    families: Dict[str, Any] = {}
    for fam, rows in kept.items():
        path = out_dir / f"fresh_{fam}.jsonl"
        write_jsonl(path, rows)
        split = Counter(r["split"] for r in rows)
        families[fam] = {**info[fam], "dropped": dropped[fam], "kept": len(rows),
                         "split": {"train": split.get("train", 0), "heldout": split.get("heldout", 0)},
                         "kept_per_task": dict(sorted(Counter(r.get("task", "?") for r in rows).items())),
                         "output": str(path).replace("\\", "/"), "output_sha256": sha256_file(path)}
        log(f"{fam}: kept {len(rows)} (dropped {dropped[fam]}), heldout {families[fam]['split']['heldout']} -> {path}")
    report = {
        "schema": REPORT_SCHEMA, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "builders_dir": str(BUILDERS).replace("\\", "/"),
        "builders_provenance": "kai9987kai/supermix-archimedes@1952cf1c92cb506fff60f9ee0c055c2365988e5e models/supermix-v93/src (MIT)",
        "row_fields_added": {"source": "fresh", "family": "omni|code|math", "split": "train|heldout"},
        "heldout_split": {"fraction": frac, "rule": "sha1('expanse-v3-heldout|' + user)[:12] < fraction * 2^48"},
        "filters": {"order": list(DROP_REASONS), "replay_prompts": len(replay), "replay_corpus": str(corpus_dir).replace("\\", "/"),
                    "eval_heldout_prompts": len(eval_prompts), "eval_heldout_seeds": [int(s) for s in heldout_seeds],
                    "eval_heldout_n": int(heldout_n), "eval_heldout_per_seed": eval_per_seed},
        "seeds_not_reused": sorted(USED_SEEDS),
        "totals": {"built": sum(len(v) for v in built.values()), "kept": sum(len(v) for v in kept.values()),
                   "dropped": {k: sum(d[k] for d in dropped.values()) for k in DROP_REASONS},
                   "heldout": sum(f["split"]["heldout"] for f in families.values())},
        "families": families,
        "seconds": round(time.time() - t0, 1),
    }
    write_json(out_dir / "fresh.report.json", report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Expanse v3 data: fresh verified rows + v3 connectome rows")
    ap.add_argument("--omni", type=int, default=10000, help="target rows (per_task = ceil(target / 12))")
    ap.add_argument("--code", type=int, default=8000, help="target rows (per_task = ceil(target / 9))")
    ap.add_argument("--math", type=int, default=6000, help="target rows (build_scratchpad_math --target)")
    ap.add_argument("--seed_omni", type=int, default=BUILDER["omni"][2])
    ap.add_argument("--seed_code", type=int, default=BUILDER["code"][2])
    ap.add_argument("--seed_math", type=int, default=BUILDER["math"][2])
    ap.add_argument("--no_natural_phrasings", action="store_true",
                    help="omni prompts from the 4-5 textbook templates only (the replay omni rows used the wide bank)")
    ap.add_argument("--heldout_frac", type=float, default=HELDOUT_FRAC)
    ap.add_argument("--heldout_seeds", default="2026,260923",
                    help="eval_expanse.heldout_problems seeds whose prompts are blocked (v1 / v2 training seeds)")
    ap.add_argument("--heldout_n", type=int, default=100000, help="heldout_problems n (prefix-stable; large = all)")
    ap.add_argument("--n_rows", type=int, default=12000, help="connectome train rows")
    ap.add_argument("--connectome_seed", type=int, default=None, help="default: v1's (from its report)")
    ap.add_argument("--heldout_fraction", type=float, default=None, help="connectome held-out type fraction; default v1's")
    ap.add_argument("--v1_connectome_report", default=str(PATHS["data"] / "connectome_rows.report.json"))
    ap.add_argument("--v1_connectome_rows", default=str(PATHS["connectome_rows"]))
    ap.add_argument("--npz", default=str(PATHS["npz"]))
    ap.add_argument("--corpus", default=str(PATHS["corpus"]), help="replay corpus dir (its prompts are blocked)")
    ap.add_argument("--out_dir", default=str(V3_DIR))
    ap.add_argument("--skip_fresh", action="store_true")
    ap.add_argument("--skip_connectome", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: Dict[str, Any] = {}
    if not args.skip_fresh:
        rep = make_fresh(out_dir, {"omni": args.omni, "code": args.code, "math": args.math},
                         {"omni": args.seed_omni, "code": args.seed_code, "math": args.seed_math},
                         natural_phrasings=not args.no_natural_phrasings,
                         heldout_seeds=[int(s) for s in args.heldout_seeds.split(",") if s.strip()],
                         heldout_n=args.heldout_n, frac=args.heldout_frac, corpus_dir=Path(args.corpus))
        summary["fresh"] = rep["totals"]
    if not args.skip_connectome:
        summary["connectome"] = make_connectome(out_dir / "connectome_rows_v3.jsonl", n_rows=args.n_rows, npz=Path(args.npz),
                                                v1_report_path=Path(args.v1_connectome_report),
                                                v1_rows_path=Path(args.v1_connectome_rows), seed=args.connectome_seed,
                                                heldout_fraction=args.heldout_fraction)
    log(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    print(json.dumps(ec.jsonable(summary), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
