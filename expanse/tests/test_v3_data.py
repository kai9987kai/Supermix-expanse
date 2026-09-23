"""Tests for make_v3_data.py and the trainer's v3 corpus (V3_DESIGN.md B.1-B.3).

What the v3 data must guarantee, checked without a model:

1. the held-out split is a stable function of ``user`` alone and takes ~3%;
2. the filters drop replay prompts, eval ``heldout_problems`` prompts and
   duplicates (each row counted under its first reason) and tag survivors;
3. the builders are called as the spec says (``per_task = ceil(target /
   n_tasks)`` over 12 omni / 9 code tasks, ``--unique``, seeds never reused),
   and each vendored builder really runs and emits verified-format rows whose
   files match the provenance README;
4. the connectome check catches a held-out type leaking into train, and the
   real regeneration keeps v1's held-out types (needs the npz, ~20 s);
5. ``train_expanse.load_corpus`` reads a data/v3 folder: ``fresh`` is one more
   source, held-out rows are never trained or used as dev, the row cap is per
   builder family, ``connectome_rows_v3`` replaces the v1 rows, and
   ``v3=False`` is the v1 corpus.

Run: ``python -m pytest -q expanse/tests/test_v3_data.py`` (~30 s, < 1 GB).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))

import make_v3_data as mv  # noqa: E402
import train_expanse as te  # noqa: E402
from expanse_core import PATHS  # noqa: E402

torch.set_num_threads(2)


def _row(user, assistant="a, total 1", **kw):
    return {"user": user, "assistant": assistant, **kw}


# ---------------------------------------------------------------------------
# split + filters
# ---------------------------------------------------------------------------
def test_heldout_split_is_stable_and_about_three_percent():
    users = [f"What is {i} + {3 * i + 1}?" for i in range(20000)]
    sides = [mv.heldout_split(u) for u in users]
    assert sides == [mv.heldout_split(u) for u in users]
    rate = sides.count("heldout") / len(users)
    assert 0.025 < rate < 0.035, rate
    assert {mv.heldout_split(u, 0.0) for u in users[:200]} == {"train"}
    assert {mv.heldout_split(u, 1.0) for u in users[:200]} == {"heldout"}
    # a hash of the prompt only: the reply does not move a row across the split
    assert mv.filter_fresh({"omni": [_row(users[0], "x")]}, {})[0]["omni"][0]["split"] == \
        mv.filter_fresh({"omni": [_row(users[0], "y")]}, {})[0]["omni"][0]["split"]


def test_filter_fresh_reasons_order_and_tags():
    built = {
        "code": [_row("c1"), _row("shared"), _row("c1")],
        "omni": [_row("r1", domain="physics", task="force"), _row("e1"), _row("o1"), _row("shared")],
        "math": [_row("m1", topic="basic_math"), _row("o1")],
    }
    before = json.dumps(built, sort_keys=True)
    kept, dropped = mv.filter_fresh(built, {"replay_prompt": {"r1"}, "eval_heldout_prompt": {"e1", "r1"}})
    assert json.dumps(built, sort_keys=True) == before  # inputs untouched
    # omni is visited first, so its "shared" wins and code's copy is the duplicate
    assert [r["user"] for r in kept["omni"]] == ["o1", "shared"]
    assert [r["user"] for r in kept["code"]] == ["c1"]
    assert [r["user"] for r in kept["math"]] == ["m1"]
    assert dropped["omni"] == {"replay_prompt": 1, "eval_heldout_prompt": 1, "duplicate": 0}  # r1 counted once
    assert dropped["code"] == {"replay_prompt": 0, "eval_heldout_prompt": 0, "duplicate": 2}
    assert dropped["math"] == {"replay_prompt": 0, "eval_heldout_prompt": 0, "duplicate": 1}
    for fam, rows in kept.items():
        for r in rows:
            assert r["source"] == "fresh" and r["family"] == fam and r["split"] in ("train", "heldout")
    assert kept["math"][0]["topic"] == "basic_math"  # builder fields kept


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def test_builder_commands_follow_the_spec(tmp_path):
    assert mv.builder_task_count("omni") == 12 and mv.builder_task_count("code") == 9
    cmd, per_task, n = mv.builder_command("omni", 10000, 3079, tmp_path / "o.jsonl", natural_phrasings=True)
    assert (per_task, n) == (834, 12) and cmd[cmd.index("--per_task") + 1] == "834"
    assert "--unique" in cmd and "--natural_phrasings" in cmd and cmd[cmd.index("--seed") + 1] == "3079"
    cmd, per_task, n = mv.builder_command("code", 8000, 3087, tmp_path / "c.jsonl", natural_phrasings=True)
    assert (per_task, n) == (889, 9) and "--unique" in cmd and "--natural_phrasings" not in cmd
    cmd, per_task, _ = mv.builder_command("omni", 10000, 3079, tmp_path / "o.jsonl", natural_phrasings=False)
    assert "--natural_phrasings" not in cmd
    cmd, per_task, _ = mv.builder_command("math", 6000, 3066, tmp_path / "m.jsonl", natural_phrasings=True)
    assert per_task is None and cmd[cmd.index("--target") + 1] == "6000" and "--unique" not in cmd
    assert Path(cmd[1]).parent == mv.BUILDERS and Path(cmd[1]).name == "build_scratchpad_math.py"
    # the fresh seeds are new draws, not the replay corpus's (79/87/66 defaults, 2026 build)
    assert not {spec[2] for spec in mv.BUILDER.values()} & mv.USED_SEEDS
    assert {spec[3] for spec in mv.BUILDER.values()} <= mv.USED_SEEDS


def test_reused_seed_is_refused_before_anything_runs(tmp_path):
    with pytest.raises(ValueError, match="already used"):
        mv.make_fresh(tmp_path, {"omni": 10, "code": 10, "math": 10}, {"omni": 79, "code": 3087, "math": 3066},
                      natural_phrasings=True, heldout_seeds=[2026], heldout_n=10, frac=0.03, corpus_dir=PATHS["corpus"])
    assert not list(tmp_path.iterdir())


def test_vendored_builders_run_and_match_the_readme(tmp_path):
    readme = (mv.BUILDERS / "README.md").read_text(encoding="utf-8")
    listed = dict((name, sha) for sha, name in re.findall(r"^\s+([0-9a-f]{64})\s+(\S+\.py)$", readme, re.M))
    assert set(listed) == {"build_omni_corpus.py", "build_code_corpus.py", "build_scratchpad_math.py", "nexus_solver.py",
                           "science_plan.py", "natural_phrasings.py", "mimomix_text.py"}
    for name, sha in listed.items():
        assert hashlib.sha256((mv.BUILDERS / name).read_bytes()).hexdigest() == sha, name
    total = re.compile(r"total -?\d+(\.\d+)?$")
    for fam, target in (("omni", 24), ("code", 18), ("math", 20)):
        rows, info = mv.run_builder(fam, target, mv.BUILDER[fam][2], tmp_path, natural_phrasings=True)
        assert len(rows) >= target * 0.9, (fam, len(rows))
        assert info["builder_sha256"] == listed[mv.BUILDER[fam][0]]
        assert "<tmp>" in info["argv"] and str(tmp_path) not in json.dumps(info["argv"])
        for r in rows:
            assert r["user"] and r.get("task") and total.search(r["assistant"]), (fam, r)
        if fam != "math":
            assert len({r["user"] for r in rows}) == len(rows)  # --unique
            assert not any(info["builder_report"].get(k) for k in ("dropped_failing_verification",
                                                                    "dropped_failing_execution"))


# ---------------------------------------------------------------------------
# connectome
# ---------------------------------------------------------------------------
def test_heldout_type_check_catches_leaks():
    v1_report = {"heldout_type_names": ["A", "B"]}
    v1_rows = [_row("q A", split="heldout", cell_type="A"), _row("q C", split="train", cell_type="C")]
    v3_rows = [_row("q2 C", split="train", cell_type="C"), _row("q2 D", split="train", cell_type="D"),
               _row("q2 A", split="heldout", cell_type="A")]
    ok = mv.heldout_type_check({"heldout_type_names": ["B", "A"]}, v3_rows, v1_report, v1_rows)
    assert ok["equal"] and ok["pool_equal"] and ok["v1_heldout_rows"] == 1
    leak = mv.heldout_type_check({"heldout_type_names": ["A", "B"]}, v3_rows + [_row("q A", split="train", cell_type="A")],
                                 v1_report, v1_rows)
    assert not leak["equal"] and leak["v1_heldout_types_in_v3_train_rows"] == 1 and leak["v1_heldout_prompts_in_v3_train"] == 1
    other = mv.heldout_type_check({"heldout_type_names": ["A", "E"]}, v3_rows, v1_report, v1_rows)
    assert not other["equal"] and not other["pool_equal"]


V1_CONN_REPORT = PATHS["data"] / "connectome_rows.report.json"


@pytest.mark.skipif(not (PATHS["npz"].exists() and V1_CONN_REPORT.exists()), reason="male-CNS npz / v1 report absent")
def test_connectome_v3_keeps_v1_heldout_types(tmp_path):
    out = tmp_path / "connectome_rows_v3.jsonl"
    res = mv.make_connectome(out, n_rows=700, npz=PATHS["npz"], v1_report_path=V1_CONN_REPORT,
                             v1_rows_path=PATHS["connectome_rows"])
    chk = res["heldout_type_check"]
    assert chk["equal"] and chk["pool_size"]["v3"] == chk["pool_size"]["v1"] > 0
    assert res["train_rows"] == 700 and res["heldout_rows"] == round(700 * 0.1 / 0.9)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    held = set(json.loads(V1_CONN_REPORT.read_text(encoding="utf-8"))["heldout_type_names"])
    assert not {r["cell_type"] for r in rows if r["split"] == "train"} & held
    assert {r["cell_type"] for r in rows if r["split"] == "heldout"} <= held
    rep = json.loads(out.with_name("connectome_rows_v3.report.json").read_text(encoding="utf-8"))
    assert rep["v3"]["equal"] and rep["seed"] == json.loads(V1_CONN_REPORT.read_text(encoding="utf-8"))["seed"]


# ---------------------------------------------------------------------------
# the trainer's corpus
# ---------------------------------------------------------------------------
def _write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def v3_dir(tmp_path):
    fams = {"omni": 40, "code": 30, "math": 20}
    for fam, n in fams.items():
        rows = [_row(f"{fam} q{i}", f"{fam} a{i}, total {i}", task=f"{fam}_t", source="fresh", family=fam,
                     split="train") for i in range(n)]
        rows += [_row(f"{fam} held{i}", f"h{i}, total {i}", task=f"{fam}_t", source="fresh", family=fam, split="heldout")
                 for i in range(3)]
        _write(tmp_path / f"fresh_{fam}.jsonl", rows)
    conn = [_row(f"type T{i} q", f"type T{i} r", split="train", cell_type=f"T{i}", answer="x") for i in range(25)]
    conn += [_row(f"type H{i} q", f"type H{i} r", split="heldout", cell_type=f"H{i}", answer="x") for i in range(6)]
    _write(tmp_path / "connectome_rows_v3.jsonl", conn)
    return tmp_path


def _keys(rows):
    return {(r["user"], r["assistant"]) for r in rows}


def test_load_corpus_reads_v3(v3_dir):
    kw = dict(fly_rows=0, seed=7, dev_frac=0.2, dev_cap=5, max_rows_per_source=12, data={"v3": v3_dir})
    corpus = te.load_corpus(None, **kw)
    assert set(corpus) == set(te.SOURCES) and te.SOURCES[-1] == "fresh" and te.V1_SOURCES == te.SOURCES[:5]
    fresh = corpus["fresh"]
    fams = {r["family"] for r in fresh["train"]}
    assert fams == {"omni", "code", "math"}
    for fam in fams:  # the cap is per builder family
        assert len([r for r in fresh["train"] if r["family"] == fam]) <= 12
    assert 0 < len(fresh["dev"]) <= 5 and not _keys(fresh["dev"]) & _keys(fresh["train"])
    assert len(fresh["heldout"]) == 9 and all(r["split"] == "heldout" for r in fresh["heldout"])
    assert not _keys(fresh["heldout"]) & (_keys(fresh["train"]) | _keys(fresh["dev"]))
    assert all(r["source"] == "fresh" for part in fresh.values() for r in part)
    conn = corpus["connectome"]
    assert {r["user"] for r in conn["train"]} <= {f"type T{i} q" for i in range(25)} and len(conn["train"]) == 12
    assert {r["cell_type"] for r in conn["dev"]} <= {f"H{i}" for i in range(6)} and len(conn["heldout"]) == 6
    again = te.load_corpus(None, **kw)
    assert all(_keys(again[s][p]) == _keys(corpus[s][p]) for s in te.SOURCES for p in ("train", "dev", "heldout"))
    files = te.corpus_files({"v3": v3_dir})
    assert files["v3"] and files["connectome"].name == "connectome_rows_v3.jsonl" and len(files["fresh"]) == 3


def test_load_corpus_v3_off_is_the_v1_corpus(v3_dir, tmp_path_factory):
    kw = dict(fly_rows=0, seed=7, dev_frac=0.05, dev_cap=5, max_rows_per_source=12)
    off = te.load_corpus(None, **kw, data={"v3": v3_dir}, v3=False)
    empty = te.load_corpus(None, **kw, data={"v3": tmp_path_factory.mktemp("no_v3")})  # auto, nothing there
    assert off["fresh"] == {"train": [], "dev": [], "heldout": []}
    for s in te.V1_SOURCES:
        for p in ("train", "dev", "heldout"):
            assert _keys(off[s][p]) == _keys(empty[s][p])
    v1_users = {r["user"] for r in te.ec.read_jsonl(PATHS["connectome_rows"])}
    assert {r["user"] for r in off["connectome"]["train"]} <= v1_users
    files = te.corpus_files({"v3": v3_dir}, v3=False)
    assert not files["v3"] and files["fresh"] == [] and files["connectome"] == Path(PATHS["connectome_rows"])
    with pytest.raises(FileNotFoundError):
        te.corpus_files({"v3": tmp_path_factory.mktemp("still_no_v3")}, v3=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
