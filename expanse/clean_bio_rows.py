"""Clean the BioMedLM corpus before build/train, and build an unbiased PubMedQA eval set.

    python expanse/clean_bio_rows.py

Why (measured on the raw corpus, 2026-09-23):

* **Yes-bias.** A PubMedQA row is kept only when BioMedLM's decision matches the
  expert ``final_decision``. BioMedLM is a base LM and answers almost everything
  "yes", so the kept rows were 482 yes / 7 no, against a gold split of
  552 yes / 338 no / 110 maybe. Trained as-is, the student would learn "always
  yes". Train rows with decision "yes" are capped at ``--yes_cap`` (all no/maybe
  rows kept).
* **Biased eval.** eval_expanse scored PubMedQA on held-out *rows*, which exist
  only where BioMedLM agreed with gold -- nearly all "yes" -- so an always-yes
  model would score ~100%. ``data/pubmedqa_eval.jsonl`` instead holds EVERY
  held-out PubMedQA question the teacher saw, with its gold label, whatever
  BioMedLM answered; its majority-class baseline is written to the report.
* **Fragments.** Definitions from the early smoke run predate the whole-word
  filter ("anaphyl", "controver"); a term must occur as a whole word in the
  PubMedQA abstracts (teacher_data.whole_word_counts) or its row is dropped.

Inputs are never modified; outputs: ``data/bio_rows.clean.jsonl`` (what build,
train and eval read via ``PATHS["bio_rows"]``), ``data/pubmedqa_eval.jsonl``,
``data/bio_rows.clean.report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import teacher_data as td  # noqa: E402

DATA = ROOT / "data"


def _h(s: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}|{s}".encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes_cap", type=int, default=40, help="max PubMedQA train rows answering yes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = td.read_jsonl(DATA / "bio_rows.jsonl")
    cands = td.read_jsonl(DATA / "bio_candidates.jsonl")
    pqa = td.load_pubmedqa(DATA)
    corpus = td.whole_word_counts(" ".join(q["contexts"]) + " " + q.get("long_answer", "") for q in pqa)

    kept, dropped = [], Counter()
    yes_train = []
    for r in rows:
        if r.get("task") == "bio_definition":
            term = str((r.get("check") or {}).get("term", "")).lower()
            if not term or corpus.get(term, 0) == 0:
                dropped["definition:fragment_or_absent"] += 1
                continue
            kept.append(r)
        elif r.get("task") == "bio_pubmedqa":
            decision = r["assistant"].split(",")[0].strip().lower()
            if r.get("split") == "train" and decision == "yes":
                yes_train.append(r)
            else:
                kept.append(r)
        else:
            kept.append(r)
    yes_train.sort(key=lambda r: _h(r["user"], args.seed))
    kept += yes_train[: args.yes_cap]
    dropped["pubmedqa:yes_over_cap"] = max(0, len(yes_train) - args.yes_cap)
    with open(DATA / "bio_rows.clean.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # unbiased PubMedQA eval: every held-out question the teacher answered, gold label attached
    by_id = {int(q["pubid"]): q for q in pqa}
    ev = []
    for c in cands:
        if c.get("kind") != "pubmedqa" or c.get("split") != "heldout" or not c.get("user"):
            continue
        pubid = int(str(c["key"]).split(":")[1])
        gold = c.get("final_decision") or by_id.get(pubid, {}).get("final_decision")
        if gold:
            ev.append({"user": c["user"], "final_decision": gold, "pubid": pubid,
                       "teacher_decision": c.get("decision"), "key": c["key"]})
    with open(DATA / "pubmedqa_eval.jsonl", "w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gold_counts = Counter(r["final_decision"] for r in ev)
    teacher_acc = (sum(r["teacher_decision"] == r["final_decision"] for r in ev) / len(ev)) if ev else None

    def split_task(rs):
        return {f"{s}:{t}": n for (s, t), n in sorted(Counter((r["split"], r["task"]) for r in rs).items())}

    report = {
        "input_rows": len(rows), "kept_rows": len(kept), "dropped": dict(dropped), "yes_cap": args.yes_cap,
        "raw_by_split_task": split_task(rows), "clean_by_split_task": split_task(kept),
        "clean_pubmedqa_train_decisions": dict(Counter(r["assistant"].split(",")[0].strip().lower() for r in kept
                                                       if r["task"] == "bio_pubmedqa" and r["split"] == "train")),
        "pubmedqa_eval": {"items": len(ev), "gold": dict(gold_counts),
                          "majority_baseline": (max(gold_counts.values()) / len(ev)) if ev else None,
                          "teacher_biomedlm_accuracy": teacher_acc},
    }
    (DATA / "bio_rows.clean.report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
