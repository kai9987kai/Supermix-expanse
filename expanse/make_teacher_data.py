"""Build the Expanse teacher corpora: verified code rows and filtered biomedical rows.

    python make_teacher_data.py convert-biomedlm            # HF -> Q8_0 GGUF, then perplexity + samples
    python make_teacher_data.py code  --target 1500 --heldout 150 [--limit N]
    python make_teacher_data.py bio   --target 1500 --heldout 150 [--limit N]
    python make_teacher_data.py judge [--limit N]
    python make_teacher_data.py all   --target 1600          # everything, end to end

Every subcommand is resumable and append-only (see `src/teacher_data.py`):
rerunning continues where the last run stopped, and ``--limit`` caps the teacher
calls of one invocation, so a long job can be done in slices. Only one
``llama-server`` runs at a time -- the 7B coder is 4.7 GB and RAM is shared.

Outputs (in ``expanse/data`` unless ``--data-dir``):
  code_rows.jsonl, code_attempts.jsonl, code_rows.receipt.json
  bio_rows.jsonl, bio_candidates.jsonl, bio_judgements.jsonl, bio_rows.receipt.json
  pubmedqa_pqa_labeled.jsonl (cache), biomed_ppl_text.txt
and ``external/teachers/gguf/biomedlm-q8_0.gguf`` + ``.receipt.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import teacher_data as td  # noqa: E402


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def coder_server(args) -> td.LlamaServer:
    return td.LlamaServer(td.CODER_GGUF, args.coder_port, threads=args.threads, ctx=2048, parallel=args.parallel)


def bio_server(args) -> td.LlamaServer:
    # BioMedLM is GPT-2: learned positions, n_ctx 1024 -- the context cannot be larger.
    return td.LlamaServer(td.BIOMEDLM_GGUF, args.bio_port, threads=args.threads, ctx=1024, parallel=args.parallel)


# ---------------------------------------------------------------------------
# convert-biomedlm
# ---------------------------------------------------------------------------

def verify_biomedlm(args) -> dict:
    """Perplexity on our own biomedical prose + 3 greedy sample completions + tokens/s."""

    receipt_path = td.BIOMEDLM_GGUF.with_suffix(".receipt.json")
    receipt = json.load(open(receipt_path, encoding="utf-8"))
    ppl = td.run_perplexity(td.BIOMEDLM_GGUF, threads=args.threads, data_dir=args.data_dir, log=log)
    samples = []
    with bio_server(args) as server:
        for prompt in td.BIO_SAMPLE_PROMPTS:
            text = server.complete(prompt, n_predict=40, temperature=0.0, seed=0)
            samples.append({"prompt": prompt, "completion": text})
            log(f"[sample] {prompt!r} -> {text!r}")
        speed = server.speed()
    receipt.update(perplexity=ppl, samples=samples, speed=speed,
                   ppl_ok=bool(ppl["ppl"] is not None and ppl["ppl"] < 30.0))
    td.write_json_atomic(receipt_path, receipt)
    log(f"[verify] PPL {ppl['ppl']} (+/- {ppl['ppl_err']}) gen {speed['gen_tokens_per_s']} tok/s "
        f"prompt {speed['prompt_tokens_per_s']} tok/s")
    return receipt


def cmd_convert(args) -> dict:
    receipt = td.convert_biomedlm(force=args.force, wait_minutes=args.wait_minutes, log=log)
    if not args.skip_verify and (args.force or "perplexity" not in receipt):
        receipt = verify_biomedlm(args)
    return receipt


# ---------------------------------------------------------------------------
# code / bio / judge
# ---------------------------------------------------------------------------

def cmd_code(args) -> dict:
    with coder_server(args) as server:
        log(f"[code] coder up in {server.load_seconds:.0f}s ({server.threads} threads)")
        return td.generate_code_rows(server, data_dir=args.data_dir, target=args.target, heldout=args.heldout,
                                     limit=args.limit, seed=args.seed, temperature=args.temperature, log=log)


def cmd_bio(args) -> dict:
    if not td.BIOMEDLM_GGUF.exists():
        raise SystemExit(f"{td.BIOMEDLM_GGUF} missing: run 'convert-biomedlm' first")
    with bio_server(args) as server:
        log(f"[bio] BioMedLM up in {server.load_seconds:.0f}s ({server.threads} threads)")
        return td.generate_bio_candidates(server, data_dir=args.data_dir, target=args.target, heldout=args.heldout,
                                          limit=args.limit, seed=args.seed, log=log)


def cmd_judge(args) -> dict:
    pending = td._bio_state(args.data_dir)["pending"]
    if not pending:
        log("[judge] nothing pending")
        return td.write_bio_receipt(args.data_dir)
    with coder_server(args) as server:
        log(f"[judge] coder up in {server.load_seconds:.0f}s; {len(pending)} pending")
        return td.judge_bio_candidates(server, data_dir=args.data_dir, limit=args.limit, log=log)


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------

def cmd_all(args) -> dict:
    """convert -> [bio generate -> coder (code rows + judge)] x rounds, until both corpora meet targets.

    The coder session does code generation and judging back to back so the
    4.7 GB model is loaded once per round. A later round only happens when
    the judge rejected more definitions than the running accept rate predicted.
    """

    t0 = time.time()
    if not td.BIOMEDLM_GGUF.exists() or "perplexity" not in _gguf_receipt():
        cmd_convert(args)
    history = []
    for rnd in range(1, args.max_rounds + 1):
        st = td._bio_state(args.data_dir)
        bio_done = st["counts"]["train"] >= args.target and st["counts"]["heldout"] >= args.heldout
        exhausted = False
        if not bio_done:
            bio = cmd_bio(args)
            exhausted = bio.get("queue_exhausted", False)
        code_rows = td.write_code_receipt(args.data_dir)["rows"]
        code_done = code_rows.get("train", 0) >= args.target and code_rows.get("heldout", 0) >= args.heldout
        pending = td._bio_state(args.data_dir)["pending"]
        if not code_done or pending:
            with coder_server(args) as server:
                log(f"[all] round {rnd}: coder up in {server.load_seconds:.0f}s")
                if not code_done:
                    td.generate_code_rows(server, data_dir=args.data_dir, target=args.target, heldout=args.heldout,
                                          limit=args.limit, seed=args.seed, temperature=args.temperature, log=log)
                if pending:
                    td.judge_bio_candidates(server, data_dir=args.data_dir, limit=None, log=log)
        st = td._bio_state(args.data_dir)
        code_rows = td.write_code_receipt(args.data_dir)["rows"]
        history.append({"round": rnd, "bio_rows": dict(st["counts"]), "code_rows": code_rows,
                        "accept_rate": round(st["accept_rate"], 3)})
        log(f"[all] after round {rnd}: {history[-1]}")
        bio_done = st["counts"]["train"] >= args.target and st["counts"]["heldout"] >= args.heldout
        code_done = code_rows.get("train", 0) >= args.target and code_rows.get("heldout", 0) >= args.heldout
        if (bio_done or exhausted) and code_done:
            break
        if args.limit is not None:
            break               # a --limit run is a slice, not a loop
    summary = {"rounds": history, "minutes": round((time.time() - t0) / 60, 1), "target": args.target,
               "heldout": args.heldout}
    td.write_json_atomic(Path(args.data_dir) / "teacher_data.all.receipt.json", summary)
    log(f"[all] done in {summary['minutes']} min")
    return summary


def _gguf_receipt() -> dict:
    p = td.BIOMEDLM_GGUF.with_suffix(".receipt.json")
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=td.DATA)
    common.add_argument("--limit", type=int, default=None, help="max teacher calls in this invocation")
    common.add_argument("--target", type=int, default=1500, help="train rows wanted (per corpus)")
    common.add_argument("--heldout", type=int, default=150, help="held-out rows wanted (per corpus)")
    common.add_argument("--threads", type=int, default=6)
    common.add_argument("--parallel", type=int, default=4, help="llama-server slots / concurrent requests")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--temperature", type=float, default=0.3, help="coder sampling temperature")
    common.add_argument("--coder-port", type=int, default=td.CODER_PORT)
    common.add_argument("--bio-port", type=int, default=td.BIO_PORT)
    common.add_argument("--wait-minutes", type=float, default=40.0, help="wait this long for the BioMedLM fetch")
    common.add_argument("--force", action="store_true", help="convert-biomedlm: redo the conversion")
    common.add_argument("--skip-verify", action="store_true", help="convert-biomedlm: skip perplexity/samples")
    common.add_argument("--max-rounds", type=int, default=4, help="all: bio generate/judge rounds")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("convert-biomedlm", cmd_convert), ("code", cmd_code), ("bio", cmd_bio),
                     ("judge", cmd_judge), ("all", cmd_all)):
        sp = sub.add_parser(name, parents=[common])
        sp.set_defaults(fn=fn)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.data_dir = Path(args.data_dir)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    out = args.fn(args)
    if isinstance(out, dict):
        brief = {k: out[k] for k in ("rows", "attempts", "candidates", "perplexity", "speed", "rounds") if k in out}
        print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
