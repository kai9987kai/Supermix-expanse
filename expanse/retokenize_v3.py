"""v3 step 1: give a trained Expanse checkpoint a byte-level BPE tokenizer.

    python expanse/retokenize_v3.py --inp expanse/checkpoints/supermix_expanse.pt \\
        --out expanse/checkpoints/supermix_expanse_v3_init.pt --vocab 24000 --context auto

Why (V3_DESIGN.md "A")
    The word tokenizer is Expanse's ceiling: every bio dev row has an
    ``<unk>`` and code writing is 0%, because a word the table never saw can
    be neither read nor written. v3 keeps every trained weight and swaps the
    tokenizer for a byte-level BPE (src/bpe_tokenizer.py). Only the tied
    embedding / LM-head table depends on the vocabulary, so only its rows are
    re-initialised -- as close to what the trunk already reads as the old
    table allows -- and training (train_expanse.py) teaches the rest.

Steps
    1. load the base: schema ``supermix-expanse-v1`` (``ec.load_expanse``) or
       ``supermix-expanse-v2`` (``consolidation_v2.load_v2``);
    2. tokenizer text = user + assistant of every **train-split** row the v3
       trainer uses: ``train_expanse.load_corpus(...)[source]["train"]`` for
       every source (replay, fly -- self-distilled from this checkpoint's own
       FlyCore, exactly as the trainer does -- code, bio, connectome, and
       ``fresh`` once data/v3 exists), with the trainer's seed / dev fraction,
       so a dev row (5% stable hash split) or a held-out row (``split ==
       "heldout"``, held-out connectome types) is never seen. If the trainer
       does not read data/v3 yet, ``fresh_*.jsonl`` / ``connectome_rows_v3``
       are split here with the trainer's own ``split_source`` / ``_cap``, and a
       final guard drops any train row whose text is also a dev / held-out row;
    3. train the BPE (``--vocab``, ``--min_frequency``);
    4. new embedding row per non-special token, first rule that applies:
         copy          the decoded string is an old token: its old row;
         old_tokenise  mean of the old rows of its old-tokenizer encoding
                       (``<unk>`` dropped) -- " mitochondrial" -> nothing,
                       ".\\n" -> mean(".", "\\n");
         fragment      stripped, lowercased, >= 3 chars: mean of (at most 64,
                       most frequent first) old rows whose token contains it --
                       "cardio" -> " cardiovascular", "cardiomyocyte", ...;
         mean_noise    old mean + 0.1 * std * N(0, 1) (per dim, seeded);
       rows from the last three rules are rescaled to the median old-row norm
       (a mean of k rows is shorter than a row, and the LM head reads norms as
       logit scale). The 6 specials copy old rows 0-5;
    5. context: ``auto`` = smallest multiple of 32 >= the 99th percentile of
       encoded training-row length (``encode_turn``), capped at 256; sets
       ``vocab_size``, ``max_position_embeddings`` and ``native_context``
       (``rope_scaling`` is "none", so no rotary table changes);
    6. save with the base's own saver (v1 -> ``ec.save_expanse``, v2 ->
       ``consolidation_v2.save_v2``); ``payload["tokenizer"]`` is the BPE dict
       (``{"kind": "bpe", ...}``), the receipt gains a ``retokenize_v3`` block
       (vocab, rule counts, context, length percentiles old vs new,
       tokens-per-word, sha256 of the base);
    7. sanity: reload the written file (tokenizer via
       ``ec.tokenizer_from_dict``), one forward (logits must equal the
       pre-save model's), greedy-decode 2 prompts. The sanity block is in the
       sidecar ``<out>.receipt.json`` (the checkpoint is written before it).

``vocab_base`` is set to the new vocabulary size: the whole table belongs to
the new tokenizer, so there is no appended-rows range for the trainer's
``RowBoost`` to single out (the v1 value 9,451 would boost an arbitrary id
range of the new table).

Memory: one model (~0.6 GB for v1) plus the loader's transient copy; the
reloaded model is built only after the first is freed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expanse_core as ec  # noqa: E402
from expanse_core import PATHS, text_utils  # noqa: E402
import bpe_tokenizer as bt  # noqa: E402

N_SPECIAL = len(text_utils.SPECIAL_TOKENS)
RULES = ("special", "copy", "old_tokenise", "fragment", "mean_noise")
V3_DATA = PATHS["data"] / "v3"
CONTEXT_MULTIPLE = 32
CONTEXT_CAP = 256
V2_SCHEMA = "supermix-expanse-v2"
DEFAULT_OUT = PATHS["checkpoints"] / "supermix_expanse_v3_init.pt"
FALLBACK_PROMPTS = ("A car travels 150 meters in 30 seconds. What is its average speed?",
                    "Write a Python function that returns the sum of the squares of a list of integers.")


def log(*a) -> None:
    print("[retok]", *a, flush=True)


def _key(r: Dict[str, Any]) -> str:
    return ec.row_key(r["user"], r["assistant"])


def _rel(path) -> str:
    """Repo-relative posix path for receipts (absolute when outside the repo)."""
    path = Path(path)
    try:
        return path.resolve().relative_to(PATHS["exp"]).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# 1. base checkpoint I/O (v1 or v2, by schema)
# ---------------------------------------------------------------------------
def load_base(path) -> Tuple[Any, Any, Dict[str, Any], str]:
    """(model, tokenizer, payload, schema) for a v1 or v2 Expanse checkpoint.

    The schema is read from an mmap'd load (no tensor data is paged in) so the
    file is only materialised once, by the matching loader."""
    head = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    schema = str(head.get("schema"))
    del head
    gc.collect()
    if schema == V2_SCHEMA:
        import consolidation_v2 as cv2
        model, tok, payload = cv2.load_v2(path)
    elif schema == ec.EXP_SCHEMA:
        model, tok, payload = ec.load_expanse(path)
    else:
        raise ValueError(f"{path}: schema {schema!r} is neither {ec.EXP_SCHEMA} nor {V2_SCHEMA}")
    payload["state_dict"] = None  # the model owns the weights
    model.eval()
    return model, tok, payload, schema


def save_like_base(schema: str, path, model, tok, extra: Dict[str, Any], receipt: Dict[str, Any]) -> None:
    if schema == V2_SCHEMA:
        import consolidation_v2 as cv2
        cv2.save_v2(path, model, tok, extra, receipt)
    else:
        ec.save_expanse(path, model, tok, extra, receipt)


# ---------------------------------------------------------------------------
# 2. tokenizer training text: the trainer's train split, never dev / held-out
# ---------------------------------------------------------------------------
def training_rows(fly_core, *, seed: int, fly_rows: int, dev_frac: float, dev_cap: int,
                  max_rows_per_source: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train-split rows of every source the v3 trainer reads (see module doc, step 2)."""
    import train_expanse as te  # lazy: only the CLI needs the trainer's corpus code

    corpus = te.load_corpus(fly_core, fly_rows=fly_rows, seed=seed, dev_frac=dev_frac, dev_cap=dev_cap,
                            max_rows_per_source=max_rows_per_source)
    by_src: Dict[str, List[Dict[str, Any]]] = {s: list(v.get("train", [])) for s, v in corpus.items()}
    excluded = {_key(r) for v in corpus.values() for part in ("dev", "heldout") for r in v.get(part, [])}
    info: Dict[str, Any] = {"via": "train_expanse.load_corpus", "fallbacks": [],
                            "args": {"seed": seed, "fly_rows": fly_rows, "dev_frac": dev_frac, "dev_cap": dev_cap,
                                     "max_rows_per_source": max_rows_per_source}}

    fresh_files = sorted(V3_DATA.glob("fresh_*.jsonl"))
    info["fresh_files"] = [f.name for f in fresh_files]
    if fresh_files and "fresh" not in corpus:
        rows = [dict(r, source="fresh") for f in fresh_files for r in ec.read_jsonl(f)]
        tr, _ = te.split_source([r for r in rows if r.get("split", "train") == "train"], dev_frac, seed, dev_cap,
                                max_rows_per_source)
        by_src["fresh"] = tr
        excluded |= {_key(r) for r in rows if r.get("split", "train") != "train"}
        # every row hashed to dev (also those past dev_cap, which the trainer drops) stays out
        thr = dev_frac * float(1 << 48)
        excluded |= {_key(r) for r in rows if te._h(_key(r), seed) < thr}
        info["fallbacks"].append("fresh_*.jsonl split here (trainer's load_corpus has no 'fresh' source)")

    conn_v3 = V3_DATA / "connectome_rows_v3.jsonl"
    info["connectome_file"] = _rel(PATHS["connectome_rows"])
    if conn_v3.exists():
        rows = [dict(r, source="connectome") for r in ec.read_jsonl(conn_v3)]
        v3_keys = {_key(r) for r in rows}
        if any(_key(r) not in v3_keys for r in by_src.get("connectome", [])):
            by_src["connectome"] = te._cap([r for r in rows if r.get("split", "train") == "train"],
                                           max_rows_per_source, seed + 1)
            excluded |= {_key(r) for r in rows if r.get("split", "train") != "train"}
            info["fallbacks"].append("connectome_rows_v3.jsonl read here (trainer's load_corpus read the v1 rows)")
        info["connectome_file"] = _rel(conn_v3)

    order = [s for s in te.SOURCES if s in by_src] + sorted(s for s in by_src if s not in te.SOURCES)
    rows_out: List[Dict[str, Any]] = []
    dropped = 0
    for s in order:
        for r in by_src[s]:
            if r.get("split", "train") != "train" or _key(r) in excluded:
                dropped += 1
                continue
            rows_out.append(r)
    info["per_source"] = {s: sum(1 for r in rows_out if r.get("source") == s) for s in order}
    info["rows"] = len(rows_out)
    info["leak_guard_dropped"] = dropped
    return rows_out, info


# ---------------------------------------------------------------------------
# 3-4. embedding rows for the new vocabulary
# ---------------------------------------------------------------------------
@torch.no_grad()
def init_embedding(old_weight: torch.Tensor, old_tok, new_tok, *, seed: int = 0, noise: float = 0.1,
                   frag_min_chars: int = 3, frag_max_rows: int = 64, n_examples: int = 8
                   ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """(V_new, H) rows for ``new_tok`` from the old table (rules in the module doc, step 4).

    Returns the table (dtype of ``old_weight``) and ``info``: ``counts`` per
    rule, ``rule_ids`` (index into ``RULES`` per new id), ``target_norm``,
    and a few ``examples`` per rule."""
    old = old_weight.detach().float().cpu()
    hidden = int(old.shape[1])
    for i, s in enumerate(text_utils.SPECIAL_TOKENS):
        if old_tok.tokens[i] != s or new_tok.tokens[i] != s:
            raise ValueError(f"special id {i} is not {s!r} in both tokenizers")
    body = old[N_SPECIAL:]
    target_norm = float(body.norm(dim=1).median())
    mean, std = body.mean(0), body.std(0)
    g = torch.Generator().manual_seed(int(seed))
    lowered = [(j, t.lower()) for j, t in enumerate(old_tok.tokens) if j >= N_SPECIAL]

    n_new = int(new_tok.vocab_size)
    new = torch.empty(n_new, hidden)
    new[:N_SPECIAL] = old[:N_SPECIAL]
    rule_ids = torch.zeros(n_new, dtype=torch.long)  # 0 = special
    counts = {r: 0 for r in RULES}
    counts["special"] = N_SPECIAL
    examples: Dict[str, List[Any]] = {r: [] for r in RULES[1:]}
    for i in range(N_SPECIAL, n_new):
        s = new_tok.tokens[i]
        j = old_tok.index.get(s)
        if j is not None and j >= N_SPECIAL:
            new[i] = old[j]
            rule, detail = "copy", None
        else:
            ids = [k for k in old_tok.encode(s) if k >= N_SPECIAL]  # drops <unk> (and any special)
            if ids:
                row, rule = old[ids].mean(0), "old_tokenise"
                detail = [old_tok.tokens[k] for k in ids[:4]]
            else:
                frag = s.strip().lower()
                hits: List[int] = []
                if len(frag) >= frag_min_chars:
                    hits = list(itertools.islice((j for j, t in lowered if frag in t), frag_max_rows))
                if hits:
                    row, rule = old[hits].mean(0), "fragment"
                    detail = [len(hits)] + [old_tok.tokens[k] for k in hits[:3]]
                else:
                    row, rule = mean + noise * std * torch.randn(hidden, generator=g), "mean_noise"
                    detail = None
            norm = float(row.norm())
            new[i] = row * (target_norm / norm) if norm > 0 else row
        counts[rule] += 1
        rule_ids[i] = RULES.index(rule)
        if len(examples[rule]) < n_examples:
            examples[rule].append([s] if detail is None else [s, detail])
    return new.to(old_weight.dtype), {"counts": counts, "rule_ids": rule_ids, "target_norm": target_norm,
                                      "examples": examples, "noise": noise, "frag_min_chars": frag_min_chars,
                                      "frag_max_rows": frag_max_rows, "seed": int(seed)}


@torch.no_grad()
def set_embedding(model: nn.Module, weight: torch.Tensor) -> int:
    """Replace the tied embedding / LM-head table (any size) and the config's ``vocab_size``."""
    old = model.embed_tokens.weight
    if model.lm_head.weight is not old:
        raise ValueError("expected a tied embedding / lm_head (tie_word_embeddings=True)")
    p = nn.Parameter(weight.detach().to(dtype=old.dtype, device=old.device).contiguous(),
                     requires_grad=old.requires_grad)
    model.embed_tokens.weight = p
    model.embed_tokens.num_embeddings = int(p.shape[0])
    model.lm_head.weight = p  # tied
    model.lm_head.out_features = int(p.shape[0])
    model.config.vocab_size = int(p.shape[0])
    return int(p.shape[0])


# ---------------------------------------------------------------------------
# 5. context and length statistics
# ---------------------------------------------------------------------------
def choose_context(lengths: Sequence[int], context: str = "auto", cap: int = CONTEXT_CAP,
                   multiple: int = CONTEXT_MULTIPLE) -> Tuple[int, float]:
    """(context, p99). ``auto``: smallest multiple of ``multiple`` >= p99, capped at ``cap``."""
    p99 = float(np.percentile(np.asarray(lengths, dtype=np.float64), 99)) if len(lengths) else 0.0
    if str(context) == "auto":
        ctx = min(int(cap), max(int(multiple), int(multiple) * math.ceil(p99 / int(multiple))))
    else:
        ctx = int(context)
        if ctx <= 0:
            raise ValueError(f"context must be positive, got {ctx}")
    return ctx, p99


def length_stats(lengths: Sequence[int], ctx: int) -> Dict[str, Any]:
    a = np.asarray(lengths, dtype=np.float64)
    if a.size == 0:
        return {"rows": 0}
    return {"rows": int(a.size), "mean": round(float(a.mean()), 2),
            **{f"p{q}": round(float(np.percentile(a, q)), 1) for q in (50, 90, 95, 99)},
            "max": int(a.max()), "context": int(ctx), "over_context": int((a > ctx).sum())}


def encode_stats(tok, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-row ``encode_turn`` lengths plus tokens-per-word and ``<unk>`` rate over user + assistant."""
    users = [r["user"] for r in rows]
    replies = [r["assistant"] for r in rows]
    if hasattr(tok, "encode_batch"):
        eu, ea = tok.encode_batch(users), tok.encode_batch(replies)
    else:
        eu, ea = [tok.encode(t) for t in users], [tok.encode(t) for t in replies]
    lengths = [len(u) + len(a) + 4 for u, a in zip(eu, ea)]  # <bos><user> u <assistant> a <eos>
    n_tok = sum(len(u) + len(a) for u, a in zip(eu, ea))
    n_unk = sum(x.count(text_utils.UNK) for x in itertools.chain(eu, ea))
    n_words = sum(len(u.split()) + len(a.split()) for u, a in zip(users, replies))
    return {"lengths": lengths, "tokens": n_tok, "words": n_words,
            "tokens_per_word": round(n_tok / max(1, n_words), 4), "unk_rate": round(n_unk / max(1, n_tok), 6),
            "rows_with_unk": sum(1 for u, a in zip(eu, ea) if text_utils.UNK in u or text_utils.UNK in a)}


# ---------------------------------------------------------------------------
# the retokenisation itself (model in memory; no file I/O)
# ---------------------------------------------------------------------------
def retokenize(model, old_tok, rows: Sequence[Dict[str, Any]], *, vocab: int = 24000, min_frequency: int = 2,
               context: str = "auto", context_cap: int = CONTEXT_CAP, seed: int = 0) -> Tuple[Any, Dict[str, Any]]:
    """Train the BPE on ``rows``, re-initialise the tied table, set the context.

    Mutates ``model`` (embedding, ``config.vocab_size`` /
    ``max_position_embeddings`` / ``native_context``, ``vocab_base``) and
    returns ``(new_tok, block)``; ``block`` is the receipt's
    ``retokenize_v3`` content without the base / sanity / data parts."""
    t0 = time.time()
    texts = [t for r in rows for t in (r["user"], r["assistant"])]
    new_tok = bt.train_bpe(texts, vocab_size=vocab, min_frequency=min_frequency)
    t_train = time.time() - t0
    log(f"BPE trained on {len(texts):,} texts ({len(rows):,} rows): vocab {new_tok.vocab_size:,} "
        f"(asked {vocab:,}) in {t_train:.1f}s")

    old_vocab = int(model.config.vocab_size)
    old_ctx = int(model.config.max_position_embeddings)
    weight, init = init_embedding(model.embed_tokens.weight, old_tok, new_tok, seed=seed)
    set_embedding(model, weight)
    log("embedding rules " + " ".join(f"{k} {v}" for k, v in init["counts"].items())
        + f" (rescaled to median old norm {init['target_norm']:.4f})")

    st_old, st_new = encode_stats(old_tok, rows), encode_stats(new_tok, rows)
    ctx, p99 = choose_context(st_new["lengths"], context, context_cap)
    model.config.max_position_embeddings = int(ctx)
    model.config.native_context = int(ctx)
    model.vocab_base = int(new_tok.vocab_size)
    if str(getattr(model.config, "rope_scaling", "none")) != "none":
        log(f"WARNING rope_scaling={model.config.rope_scaling!r}: the rotary tables were built for the old context; "
            "the reload check compares logits")
    log(f"context {old_ctx} -> {ctx} (p99 new {p99:.1f}, cap {context_cap}); tokens/word old "
        f"{st_old['tokens_per_word']} new {st_new['tokens_per_word']}; old unk rate {st_old['unk_rate']}")

    tok_json = new_tok.to_dict()["json"]
    block = {
        "tokenizer": {"kind": bt.KIND, "pre_tokenizer": bt.PRE_TOKENIZER_SPEC, "decoder": "ByteLevel",
                      "specials": list(text_utils.SPECIAL_TOKENS), "json_sha256": hashlib.sha256(tok_json.encode("utf-8")).hexdigest(),
                      "merges": int(new_tok.vocab_size) - N_SPECIAL - 256,
                      "sample_merged_tokens": new_tok.tokens[N_SPECIAL + 256:N_SPECIAL + 256 + 40],
                      "train_seconds": round(t_train, 1)},
        "vocab": {"requested": int(vocab), "size": int(new_tok.vocab_size), "min_frequency": int(min_frequency),
                  "old_size": old_vocab, "vocab_base": int(new_tok.vocab_size),
                  "vocab_base_note": "whole table belongs to the new tokenizer: no appended-row range for RowBoost"},
        "rules": dict(init["counts"]),
        "rule_examples": init["examples"],
        "init": {"target_norm_median_old": init["target_norm"], "noise": init["noise"], "seed": init["seed"],
                 "fragment_min_chars": init["frag_min_chars"], "fragment_max_rows": init["frag_max_rows"],
                 "rescaled_rows": sum(init["counts"][r] for r in ("old_tokenise", "fragment", "mean_noise"))},
        "context": {"arg": str(context), "chosen": int(ctx), "old": old_ctx, "p99_new": round(p99, 1),
                    "cap": int(context_cap), "multiple": CONTEXT_MULTIPLE,
                    "set": ["vocab_size", "max_position_embeddings", "native_context"],
                    "rope_scaling": str(getattr(model.config, "rope_scaling", "?"))},
        "lengths": {"old": length_stats(st_old["lengths"], old_ctx), "new": length_stats(st_new["lengths"], ctx)},
        "tokens_per_word": {"old": st_old["tokens_per_word"], "new": st_new["tokens_per_word"]},
        "old_unk": {"rate": st_old["unk_rate"], "rows_with_unk": st_old["rows_with_unk"], "rows": len(rows)},
        "counted_text": {"tokens_old": st_old["tokens"], "tokens_new": st_new["tokens"], "words": st_new["words"]},
    }
    return new_tok, block


# ---------------------------------------------------------------------------
# 7. sanity
# ---------------------------------------------------------------------------
def graft_kwargs(model, prompts: Sequence[str]) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if getattr(model, "omni_core", None) is not None:
        kw["omni_features"] = ec.arch.OmniCore.featurize(list(prompts))
    if getattr(model, "omni7", None) is not None:
        kw["omni7_state"] = model.omni7_state_for(list(prompts))
    return kw


@torch.no_grad()
def prompt_logits(model, tok, prompt: str) -> torch.Tensor:
    ids, _ = tok.encode_turn(prompt, None)
    model.eval()
    return model(torch.tensor([ids]), return_mtp=False, **graft_kwargs(model, [prompt])).logits[0].float()


@torch.no_grad()
def decode_prompts(model, tok, prompts: Sequence[str], max_new: int) -> List[Dict[str, Any]]:
    out = []
    for p in prompts:
        ids, _ = tok.encode_turn(p, None)
        t0 = time.time()
        new = ec.greedy_decode(model, torch.tensor([ids]), max_new_tokens=max_new, **graft_kwargs(model, [p]))
        out.append({"prompt": p, "prompt_tokens": len(ids), "ids": [int(i) for i in new], "text": tok.decode(new),
                    "seconds": round(time.time() - t0, 1)})
    return out


def pick_prompts(rows: Sequence[Dict[str, Any]], given: Optional[Sequence[str]]) -> List[str]:
    if given:
        return list(given)
    out: List[str] = []
    for src in ("replay", "code", "bio"):
        hit = next((r["user"] for r in rows if r.get("source") == src), None)
        if hit and len(out) < 2:
            out.append(hit)
    return (out + list(FALLBACK_PROMPTS))[:2]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Expanse v3: byte-level BPE retokenisation of a v1/v2 checkpoint")
    ap.add_argument("--inp", default=str(PATHS["final"]), help="v1 or v2 Expanse checkpoint")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--vocab", type=int, default=24000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--context", default="auto", help="auto | N")
    ap.add_argument("--context_cap", type=int, default=CONTEXT_CAP)
    ap.add_argument("--seed", type=int, default=2026, help="the trainer's --seed (dev split, fly rows); also the init noise seed")
    ap.add_argument("--fly_rows", type=int, default=4000, help="the trainer's --fly_rows")
    ap.add_argument("--dev_frac", type=float, default=0.05, help="the trainer's --dev_frac")
    ap.add_argument("--dev_cap", type=int, default=200, help="the trainer's --dev_cap")
    ap.add_argument("--max_rows_per_source", type=int, default=0,
                    help="0 = every train-split row (rows past the trainer's cap are still train-split, never dev)")
    ap.add_argument("--prompt", action="append", default=None, help="sanity prompt (repeat; default: 2 corpus prompts)")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2, help="torch threads and tokenizers (rayon) threads")
    args = ap.parse_args()
    inp, out = Path(args.inp), Path(args.out)
    if out.resolve() == inp.resolve():
        raise SystemExit("--out must differ from --inp (the base checkpoint is never overwritten)")
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    t0 = time.time()

    base_sha = ec.sha256_file(inp)
    model, old_tok, payload, schema = load_base(inp)
    log(f"loaded {inp} ({schema}): vocab {old_tok.vocab_size:,}, context {model.config.max_position_embeddings}, "
        f"{sum(p.numel() for p in model.parameters()):,} params in {time.time() - t0:.1f}s")

    rows, data_info = training_rows(model.fly_core, seed=args.seed, fly_rows=args.fly_rows, dev_frac=args.dev_frac,
                                    dev_cap=args.dev_cap, max_rows_per_source=args.max_rows_per_source)
    log(f"tokenizer text: {data_info['rows']:,} train-split rows " + str(data_info["per_source"])
        + (f"; fallbacks {data_info['fallbacks']}" if data_info["fallbacks"] else "")
        + f"; leak guard dropped {data_info['leak_guard_dropped']}")
    prompts = pick_prompts(rows, args.prompt)
    before = decode_prompts(model, old_tok, prompts, args.max_new_tokens)
    for d in before:
        log(f"before: {d['prompt'][:70]!r} -> {d['text'][:120]!r}")

    new_tok, block = retokenize(model, old_tok, rows, vocab=args.vocab, min_frequency=args.min_frequency,
                                context=args.context, context_cap=args.context_cap, seed=args.seed)
    block["base"] = {"path": _rel(inp), "sha256": base_sha, "schema": schema, "vocab": old_tok.vocab_size,
                     "tokenizer_kind": getattr(old_tok, "kind", "word")}
    block["data"] = data_info
    block["seconds_to_save"] = round(time.time() - t0, 1)
    ref_logits = prompt_logits(model, new_tok, prompts[0])

    receipt = dict(payload.get("expanse") or {})
    receipt["stage"] = "v3-init (retokenized, untrained)"
    receipt["retokenize_v3"] = block
    extra = dict(payload.get("extra") or {})
    extra.update({"note": f"v3 init: byte-level BPE retokenisation of {inp.name} ({schema}), untrained",
                  "tokenizer": "bpe", "retokenized_from_sha256": base_sha})
    save_like_base(schema, out, model, new_tok, extra, receipt)
    log(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    tok_json = new_tok.to_dict()["json"]
    del model, payload
    gc.collect()

    # sanity on the file as written -----------------------------------------------------------
    m2, tok2, pay2, schema2 = load_base(out)
    loaded_logits = prompt_logits(m2, tok2, prompts[0])
    sanity = {
        "schema": schema2, "tokenizer_class": type(tok2).__name__, "tokenizer_json_identical": tok2.to_dict()["json"] == tok_json,
        "config": {k: getattr(m2.config, k) for k in ("vocab_size", "max_position_embeddings", "native_context")},
        "embedding_shape": list(m2.embed_tokens.weight.shape), "tied": m2.lm_head.weight is m2.embed_tokens.weight,
        "forward": {"prompt": prompts[0], "logits_shape": list(loaded_logits.shape),
                    "finite": bool(torch.isfinite(loaded_logits).all()),
                    "max_abs_diff_vs_presave": float((loaded_logits - ref_logits).abs().max()),
                    "top5_next": [tok2.tokens[int(i)] for i in loaded_logits[-1].topk(5).indices]},
        "decodes": decode_prompts(m2, tok2, prompts, args.max_new_tokens),
        "decodes_before_word_tokenizer": before,
    }
    ok = (sanity["tokenizer_class"] == "BPETokenizer" and sanity["tokenizer_json_identical"] and sanity["tied"]
          and sanity["forward"]["finite"] and sanity["forward"]["max_abs_diff_vs_presave"] < 1e-5
          and sanity["config"]["vocab_size"] == block["vocab"]["size"])
    sanity["passed"] = bool(ok)
    for d in sanity["decodes"]:
        log(f"after: {d['prompt'][:70]!r} -> {d['text'][:120]!r}")
    block["sanity"] = sanity
    block["seconds"] = round(time.time() - t0, 1)
    receipt_out = dict(pay2.get("expanse") or receipt)
    receipt_out["retokenize_v3"] = block
    rpath = out.with_suffix(".receipt.json")
    rpath.write_text(json.dumps(ec.jsonable({k: v for k, v in receipt_out.items() if k != "omni7_meta"}), indent=1),
                     encoding="utf-8")
    log(f"sanity {'passed' if ok else 'FAILED'} (reload max|dlogit| {sanity['forward']['max_abs_diff_vs_presave']:.3e}); "
        f"receipt {rpath}; {block['seconds']:.0f}s total")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
