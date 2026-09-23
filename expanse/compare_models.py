"""Compare Expanse checkpoints (v1, v2, v3, ...) on the same held-out evidence.

    python expanse/compare_models.py --model v1=expanse/checkpoints/supermix_expanse.pt \
        --model v2=expanse/checkpoints/supermix_expanse_v2.pt --gen_limit 50

Tokenizer-agnostic: every model is scored on the *same rows* (the v1 training
run's dev split, rebuilt from the v1 receipt) and the same generation items, and
dev loss is reported both per token (only comparable between models that share
a tokenizer) and **per reply character** (comparable across tokenizers, e.g. the
word-level v1/v2 against a BPE v3). Rows a word-level tokenizer cannot encode
without <unk> are reported separately; "all-covered" numbers use only the rows
every compared model encodes without <unk>.

A v2 checkpoint (schema ``supermix-expanse-v2``) is also scored with its native
consolidation gates zeroed (``<name>_gates0``): the gain that disappears there is
the part attributable to the new blocks rather than to the extra training.
Writes ``--out`` (json) and the same path with ``.md``.

v3: tokenizers come from ``tokenizer_from_dict`` (word or BPE) and each model is
scored at ``seq = max(--seq, its max_position_embeddings)``, so a longer-context
v3 is not cut to 128 tokens (the rows are the same for everyone; a row too long
for one model is left out of that model's numbers only). The corpus is rebuilt
the way the ``--ref`` run built it: the v1 files unless its receipt says it
trained on ``data/v3`` (``corpus_args.v3_data``). Checkpoint headers (schema,
tokenizer, receipt) are read with ``mmap``, not by loading every tensor.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import expanse_core as ec  # noqa: E402
import eval_expanse as ee  # noqa: E402
import train_expanse as te  # noqa: E402
from expanse_core import PATHS  # noqa: E402


def log(*a) -> None:
    print("[compare]", *a, flush=True)


def load_any(path: str):
    """(model, tokenizer, payload, schema) for a v1 or v2 Expanse checkpoint (word or BPE tokenizer)."""
    model, tok, payload, schema = te.load_checkpoint(path)
    payload["state_dict"] = None
    model.eval()
    return model, tok, payload, schema


def model_seq(model, floor: int) -> int:
    """Scoring length: at least ``floor`` (128), and the model's whole context when it is longer."""
    return max(int(floor), int(model.config.max_position_embeddings))


@contextlib.contextmanager
def v2_gates_closed(model):
    stack = getattr(model, "consolidation_v2", None)
    saved = []
    with torch.no_grad():
        for b in (stack.blocks.values() if stack is not None else []):
            saved.append((b, b.out_gate.detach().clone()))
            b.out_gate.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for b, g in saved:
                b.out_gate.copy_(g)


@torch.no_grad()
def dev_block(runner: ee.Runner, dev: Dict[str, List[Dict[str, Any]]], common: Dict[str, List[bool]],
              seq: int, batch: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s, rows in dev.items():
        if not rows:
            continue
        losses = runner.row_losses(rows, seq, batch)
        cov = ee.covered(runner.tok, rows)
        chars = [len(r["assistant"]) + 1 for r in rows]          # +1: the end-of-reply token
        def agg(mask):
            sel = [(l, c) for l, c, m in zip(losses, chars, mask) if m and l is not None]
            if not sel:
                return None, None
            nats = sum(l[0] for l, _ in sel)
            return nats / max(1, sum(l[1] for l, _ in sel)), nats / max(1, sum(c for _, c in sel))
        tok_all, chr_all = agg([True] * len(rows))
        tok_cov, chr_cov = agg(common[s])
        out[s] = {"rows": len(rows), "coverage": sum(cov) / len(rows),
                  "nats_per_token_all": tok_all, "nats_per_char_all": chr_all,
                  "nats_per_char_all_covered": chr_cov, "nats_per_token_all_covered": tok_cov,
                  "rows_all_covered": int(sum(common[s]))}
    return out


def markdown(rep: Dict[str, Any]) -> str:
    names = list(rep["models"].keys())
    lines = ["# Expanse model comparison", "",
             "Same rows for every model (v1 training dev split) and the same generation items. "
             "**nats/char** is comparable across tokenizers; nats/token only within one tokenizer.", "",
             "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    srcs = [s for s in te.SOURCES if any(s in m["dev"] for m in rep["models"].values())]
    for key, label in (("nats_per_char_all_covered", "nats/char (rows all models cover)"),
                       ("nats_per_char_all", "nats/char (all rows)"),
                       ("nats_per_token_all", "nats/token (all rows)")):
        for s in srcs:
            vals = []
            for n in names:
                v = rep["models"][n]["dev"].get(s, {}).get(key)
                vals.append("-" if v is None else f"{v:.4f}")
            lines.append(f"| {label} {s} | " + " | ".join(vals) + " |")
    gen_rows = [("exact-answer accuracy", "exact", "accuracy"), ("code pass rate", "code", "pass_rate"),
                ("bio token-F1", "bio", "token_f1"), ("PubMedQA accuracy", "bio", "pubmedqa_accuracy"),
                ("connectome exact match", "connectome", "exact_match"),
                ("connectome token-F1", "connectome", "token_f1")]
    for label, blk, key in gen_rows:
        vals = []
        for n in names:
            v = ((rep["models"][n].get("gen") or {}).get(blk) or {}).get(key)
            vals.append("-" if v is None else f"{v:.4f}")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", f"Generation items per metric: {json.dumps(rep['items'])}; greedy, max {rep['args']['max_new_tokens']} new tokens.",
              "Scoring length / tokenizer per model: " + ", ".join(f"{n} {m.get('seq', '?')} ({m.get('tokenizer', '?')})"
                                                             for n, m in rep["models"].items()) + "."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--ref", default=str(PATHS["final"]), help="v1 checkpoint whose receipt defines the dev split")
    ap.add_argument("--arch", default=str(PATHS["arch_final"]), help="Archimedes (its original fly core re-derives fly rows)")
    ap.add_argument("--gen_limit", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--seq", type=int, default=128, help="minimum scoring length (each model: max(this, its context))")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no_gen", action="store_true")
    ap.add_argument("--out", default=str(PATHS["checkpoints"] / "compare_models.json"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    training = (te.peek_checkpoint(args.ref, "expanse")["expanse"] or {}).get("training") or {}
    cargs = dict(training.get("corpus_args") or {})
    seed = int(cargs.get("seed", training.get("seed", 2026)))
    # the ref run's corpus: v1 files unless it trained on data/v3 (then the same v3 dir)
    v3 = bool(cargs.get("v3_data", False))
    A, _, _ = ec.load_archimedes(args.arch)
    corpus = te.load_corpus(A.fly_core, fly_rows=int(cargs.get("fly_rows", 4000)), seed=seed,
                            dev_frac=float(cargs.get("dev_frac", 0.05)), dev_cap=int(cargs.get("dev_cap", 200)),
                            max_rows_per_source=int(cargs.get("max_rows_per_source", 6000)),
                            data={"v3": Path(cargs["v3_dir"])} if cargs.get("v3_dir") else None, v3=v3)
    del A
    dev_keys = training.get("dev_keys") or {}
    dev: Dict[str, List[Dict[str, Any]]] = {}
    for s in te.SOURCES:
        rows = corpus[s]["dev"]
        if dev_keys.get(s):
            want = set(dev_keys[s])
            rows = [r for r in rows if ec.row_key(r["user"], r["assistant"]) in want]
        dev[s] = rows
    n = args.gen_limit
    items = {"exact": ee.heldout_problems(n, seed), "code": corpus["code"]["heldout"][:n],
             "bio": [r for r in corpus["bio"]["heldout"] if r.get("task") == "bio_definition"][:n],
             "pubmedqa": (ec.read_jsonl(PATHS["data"] / "pubmedqa_eval.jsonl")[:n]
                          if (PATHS["data"] / "pubmedqa_eval.jsonl").exists() else []),
             "connectome": corpus["connectome"]["heldout"][:n]}
    try:
        from teacher_data import verify_code_reply as code_verify
    except Exception:
        code_verify = None
    log("dev rows " + json.dumps({s: len(v) for s, v in dev.items()}) + "; items " + json.dumps({k: len(v) for k, v in items.items()}))

    specs: List[Tuple[str, str]] = []
    for m in args.model:
        name, path = m.split("=", 1)
        specs.append((name, path))
    # rows every model covers without <unk> (needs each tokenizer once)
    toks = {name: te.tokenizer_from_dict(te.peek_checkpoint(path, "tokenizer")["tokenizer"]) for name, path in specs}
    common = {s: [all(c) for c in zip(*[ee.covered(t, rows) for t in toks.values()])] for s, rows in dev.items() if rows}

    rep: Dict[str, Any] = {"args": vars(args), "seed": seed, "items": {k: len(v) for k, v in items.items()},
                           "dev_rows": {s: len(v) for s, v in dev.items()}, "corpus_v3": v3, "models": {}}
    for name, path in specs:
        model, tok, payload, schema = load_any(path)
        runner = ee.Runner(name, model, tok, True)
        seq = model_seq(model, args.seq)
        variants = [(name, contextlib.nullcontext())]
        if schema == te.V2_SCHEMA:
            variants.append((name + "_gates0", v2_gates_closed(model)))
        for vname, ctx in variants:
            t1 = time.time()
            with ctx:
                entry: Dict[str, Any] = {"path": path, "schema": schema, "vocab": tok.vocab_size, "seq": seq,
                                         "tokenizer": te.tokenizer_kind(tok),
                                         "dev": dev_block(runner, dev, common, seq, args.batch)}
                log(f"{vname} dev nats/char " + json.dumps({s: round(v['nats_per_char_all_covered'] or -1, 4) for s, v in entry['dev'].items()}))
                if not args.no_gen:
                    entry["gen"] = ee.generation_block([runner], items, args.max_new_tokens, code_verify)[name]
                    log(f"{vname} gen " + json.dumps({k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))} for k, v in entry['gen'].items() if isinstance(v, dict)})[:600])
            entry["seconds"] = round(time.time() - t1, 1)
            rep["models"][vname] = entry
        del model, runner
    rep["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.write_text(json.dumps(ec.jsonable(rep), indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(rep), encoding="utf-8")
    log(f"wrote {out} and {out.with_suffix('.md')} in {rep['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
