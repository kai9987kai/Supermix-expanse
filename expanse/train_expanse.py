"""Stage 2: distil and fine-tune the grafted Supermix Expanse checkpoint.

    python expanse/train_expanse.py --steps 600 --batch 8 --max_minutes 600
    python expanse/train_expanse.py --smoke --threads 2      # 2 steps on 16 rows

Based on ``supermix-archimedes/archimedes/train_archimedes.py``.

Corpus (dev = 5% per source by a stable hash of the row, capped)
    replay      Archimedes' own corpus (omni/code/math, 3,743 rows): the trunk's
                task families, so nothing it could do is forgotten
    fly         self-distilled from the *frozen original* FlyCore (the Fly Lab
                experience log is not available): observations are sampled
                from U-shaped (arcsine) distributions over each sense's range
                -- bearings as (sin, cos) of a uniform angle, which is the same
                law -- rounded to the prompt's integer-percent precision so the
                text fully determines them; the reply is the frozen core's
                consensus votes and argmax action, in ``fly_row``'s format
    code        Qwen2.5-Coder-7B rows that passed our own tests (data/code_rows.jsonl)
    bio         BioMedLM rows that passed the cross-teacher / PubMedQA filters
    connectome  exact facts of the full male-CNS graph; dev = held-out *types*
Losses
    model       ``out.loss`` (LM + MTP + router + thinking terms, as Archimedes)
    kd_arch     0.5 x KL(T=2) to frozen Archimedes on replay rows, from cached
                top-32 teacher logits (``data/kd_arch_top32.pt``, computed once)
    omni7_kd    0.2 x KL of ``omni7.aux_logits(h at the last prompt position)``
                to v7's cached intent/domain distributions (every row); ``h``
                is the trunk's final (normed) hidden state, so the trunk itself
                learns to carry v7's judgement
    fly_aux     1.0 x [KL(teacher probs || softmax(brains_forward(obs)))
                + MSE(sense at the last prompt position, obs)] on fly rows
Parameter groups
    trunk 3e-5 (wd 0.01); grafts 2e-4 (wd 0): fly_core, omni_core bridge,
    omni7 bridge + aux heads, cns_full, donor expert slots. Frozen: omni
    v48/v38 encoders, omni7 net. New embedding rows and donor router rows live
    inside trunk tensors; they get the graft learning rate by **scaling their
    realised update** by lr_graft/lr_trunk (:class:`RowBoost`). A gradient
    hook would not do it: Adam divides each element's gradient by its own
    running RMS, so a constant gradient scale cancels out of the step (and
    would inflate the global clip norm on top).
Schedule
    donor experts woken over progress 0.1 -> 0.5 (``wake_grafted_experts``:
    alive, bias annealed dormant -> alive median); cosine lr with 5% warmup and
    a 5% floor; grad clip 1.0.
Checkpoints
    a rolling partial (``<out>.partial.pt``: trainable state + optimizer + step
    + data order + rng + history) is overwritten in place every
    ``--save_every`` steps and when the wall budget or ``--stop_after`` ends
    the run; rerunning the same command resumes from it. The final model is
    ``checkpoints/supermix_expanse.pt`` + receipt json.

v3 (V3_DESIGN.md B.3) -- the same trainer continues a *trained* model
    input      a v1 (``supermix-expanse-v1``) or v2 (``supermix-expanse-v2``,
               ``consolidation_v2.load_v2``) checkpoint, saved back with the
               matching saver; the v2 native latent blocks train in the graft
               group (plus their router-balance term, ``--v2_balance``, as in
               train_consolidation_v2). The schema is peeked with ``mmap`` so
               the file is read once.
    tokenizer  whatever the checkpoint carries (``tokenizer_from_dict``: word or
               byte-level BPE after ``retokenize_v3.py``); ``--seq`` defaults to
               the checkpoint's ``max_position_embeddings``.
    fresh      a sixth source: ``data/v3/fresh_{omni,code,math}.jsonl``
               (make_v3_data.py) when present -- ``split == "heldout"`` rows are
               never trained, the train rows get the usual hash dev split, and
               ``--max_rows_per_source`` caps each builder family separately (one
               cap over all three would throw most of them away).
               ``data/v3/connectome_rows_v3.jsonl`` replaces the v1 connectome
               rows when present (same held-out types, so dev is unchanged in
               kind). ``--v3_data off`` reproduces the v1 corpus exactly.
    kd_arch    needs Archimedes' vocabulary (its cached logits are indexed by
               word ids): it is switched off, and logged, when the tokenizer is
               not a word tokenizer extending Archimedes' vocabulary.
    embeddings after retokenisation every row of the tied embedding / LM head is
               new, so all of them get the row boost (not just ids >= vocab_base);
               ``--freeze_trunk_steps N`` lets only those rows and the graft/v2
               parameters move for the first N steps (the other trunk grads are
               dropped -- set to None, so AdamW neither steps nor decays them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import expanse_core as ec  # noqa: E402
from expanse_core import PATHS, text_utils  # noqa: E402

FLY_AGENTS = ("agent1", "agent2", "agent3")
FLY_REGIMES = ("FORAGE", "EVADE", "SCOUT", "PIONEER", "CONSOLIDATE", "MAP_BEACON")
SOURCES = ("replay", "fly", "code", "bio", "connectome", "fresh")
V1_SOURCES = SOURCES[:5]  # the v1 corpus (train_consolidation_v2's domains)
FRESH_FAMILIES = ("omni", "code", "math")
V2_SCHEMA = "supermix-expanse-v2"
PARTIAL_SCHEMA = "supermix-expanse-partial-v1"
KD_CACHE_VERSION = 1
O7_CACHE_VERSION = 1


def log(*a) -> None:
    print("[train]", *a, flush=True)


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
def _h(text: str, seed: int) -> int:
    return int(hashlib.sha1(f"{seed}|{text}".encode("utf-8")).hexdigest()[:12], 16)


def fly_row(r: Dict[str, Any]) -> Tuple[str, str]:
    """``train_archimedes.fly_row``: one fly row as (prompt, reply) (same format, same vocabulary)."""
    def pct(v: float) -> str:
        return str(int(round(float(v) * 100)))
    o, e, p, a = r["obs"], r["env"], r["probs"], int(r["action"])
    user = (f"Fly {r['agent']} {r['regime']}: x {pct(o[0])} y {pct(o[1])} food {pct(o[2])} threat {pct(o[3])} energy {pct(o[4])} "
            f"dist {pct(o[5])} food bearing {pct(o[6])} {pct(o[7])} threat bearing {pct(o[8])} {pct(o[9])} peer {pct(o[10])} {pct(o[11])} "
            f"antennae {pct(o[12])} {pct(o[13])} temp {e['t']:.0f} humidity {pct(e['h'])} wind {pct(e['wx'])} {pct(e['wy'])} "
            f"rain {pct(e['rain'])} predator {pct(e['pred'])}. Which way?")
    reply = (f"threat {pct(o[3])} food {pct(o[2])} energy {pct(o[4])}, votes up {pct(p[0])} down {pct(p[1])} left {pct(p[2])} right {pct(p[3])}, "
             f"the fly moves {ec.arch.FLY_ACTIONS[a]}, action {a}")
    return user, reply


@torch.no_grad()
def make_fly_rows(fly_core, n: int, seed: int) -> List[Dict[str, Any]]:
    """Self-distilled fly rows from a (frozen) FlyCore; see module doc."""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)

    def u(lo: float, hi: float) -> np.ndarray:  # arcsine (Beta(0.5, 0.5)) on [lo, hi]
        return lo + (hi - lo) * rng.beta(0.5, 0.5, size=n)

    fa, ha = rng.uniform(-math.pi, math.pi, n), rng.uniform(-math.pi, math.pi, n)
    obs = np.stack([u(0, 1), u(0, 1), u(0, 1), u(0, 1), u(0, 1), u(0, 1), np.sin(fa), np.cos(fa), np.sin(ha), np.cos(ha),
                    u(-1, 1), u(-1, 1), u(0, 1), u(0, 1)], axis=1)
    obs = np.round(obs * 100) / 100  # what the prompt can express
    env = {"t": u(8, 38), "h": u(0, 1), "wx": u(-1, 1), "wy": u(-1, 1), "rain": u(0, 1), "pred": u(0, 1)}
    agents = rng.integers(0, len(FLY_AGENTS), n)
    regimes = rng.integers(0, len(FLY_REGIMES), n)
    was = fly_core.training
    fly_core.eval()
    probs = fly_core.brains_forward(torch.tensor(obs, dtype=torch.float32))["probs"].double().numpy()
    fly_core.train(was)
    rows = []
    for i in range(n):
        r = {"obs": [float(v) for v in obs[i]], "probs": [float(v) for v in probs[i]], "action": int(probs[i].argmax()),
             "env": {k: float(v[i]) for k, v in env.items()}, "agent": FLY_AGENTS[agents[i]], "regime": FLY_REGIMES[regimes[i]]}
        user, reply = fly_row(r)
        rows.append({"user": user, "assistant": reply, "source": "fly", "task": "fly_" + r["regime"].lower(),
                     "obs": r["obs"], "probs": r["probs"], "action": r["action"]})
    return rows


def _cap(rows: List[Dict[str, Any]], cap: int, seed: int) -> List[Dict[str, Any]]:
    if cap and len(rows) > cap:
        rows = sorted(rows, key=lambda r: _h(ec.row_key(r["user"], r["assistant"]), seed))[:cap]
    return rows


def split_source(rows: List[Dict[str, Any]], frac: float, seed: int, dev_cap: int,
                 train_cap: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stable hash split: a row's side depends only on its text (and the seed)."""
    thr = frac * float(1 << 48)
    train, dev = [], []
    for r in rows:
        (dev if _h(ec.row_key(r["user"], r["assistant"]), seed) < thr else train).append(r)
    return _cap(train, train_cap, seed + 1), _cap(dev, dev_cap, seed + 2)


def corpus_files(data: Optional[Dict[str, Path]] = None, v3: Optional[bool] = None) -> Dict[str, Any]:
    """Which files :func:`load_corpus` reads.

    ``v3``: ``None`` = the v3 files (``<v3>/fresh_*.jsonl``,
    ``<v3>/connectome_rows_v3.jsonl``; ``<v3>`` = ``data["v3"]`` or
    ``data/v3``) wherever present; ``False`` = the v1 corpus only (what a v1/v2
    run's split is rebuilt from); ``True`` = the v3 files, which must exist.
    """
    paths = dict(PATHS)
    paths.update(data or {})
    v3_dir = Path(paths.get("v3") or PATHS["data"] / "v3")
    fresh = [v3_dir / f"fresh_{f}.jsonl" for f in FRESH_FAMILIES]
    conn_v3 = v3_dir / "connectome_rows_v3.jsonl"
    if v3 is True:
        absent = [str(p) for p in fresh + [conn_v3] if not p.exists()]
        if absent:
            raise FileNotFoundError(f"v3 data requested but missing: {absent} (run make_v3_data.py)")
    use = v3 is not False
    fresh = [p for p in fresh if use and p.exists()]
    conn = conn_v3 if use and conn_v3.exists() else Path(paths["connectome_rows"])
    return {"v3_dir": v3_dir, "fresh": fresh, "connectome": conn, "v3": bool(fresh) or conn == conn_v3}


def load_corpus(fly_core, *, fly_rows: int, seed: int, dev_frac: float = 0.05, dev_cap: int = 200,
                max_rows_per_source: int = 0, data: Optional[Dict[str, Path]] = None,
                v3: Optional[bool] = None) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """{source: {"train": rows, "dev": rows, "heldout": rows}}. Deterministic in ``seed``.

    ``heldout`` (code/bio/fresh: ``split == "heldout"``) is never trained on nor
    used as dev; eval_expanse generates on it. Connectome dev = its held-out
    types. ``v3`` / ``data["v3"]``: see :func:`corpus_files`; every source key
    is present (``fresh`` empty without v3 data).
    """
    paths = dict(PATHS)
    paths.update(data or {})
    files = corpus_files(data, v3)
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    tr, dv = split_source(ec.load_replay(paths["corpus"]), dev_frac, seed, dev_cap, max_rows_per_source)
    out["replay"] = {"train": tr, "dev": dv, "heldout": []}
    tr, dv = split_source(make_fly_rows(fly_core, fly_rows, seed), dev_frac, seed, dev_cap, max_rows_per_source)
    out["fly"] = {"train": tr, "dev": dv, "heldout": []}
    for src, key in (("code", "code_rows"), ("bio", "bio_rows")):
        rows = [dict(r, source=src) for r in ec.read_jsonl(paths[key])]
        tr, dv = split_source([r for r in rows if r.get("split", "train") == "train"], dev_frac, seed, dev_cap, max_rows_per_source)
        out[src] = {"train": tr, "dev": dv, "heldout": [r for r in rows if r.get("split") == "heldout"]}
    rows = [dict(r, source="connectome") for r in ec.read_jsonl(files["connectome"])]
    held = [r for r in rows if r.get("split") == "heldout"]
    train = [r for r in rows if r.get("split", "train") == "train"]
    out["connectome"] = {"train": _cap(train, max_rows_per_source, seed + 1), "dev": _cap(held, dev_cap, seed + 2),
                         "heldout": held}
    # fresh (v3): one source; the row cap applies per builder family
    rows = []
    for p in files["fresh"]:
        fam = p.stem[len("fresh_"):]
        rows += [dict(r, source="fresh", family=r.get("family", fam)) for r in ec.read_jsonl(p)]
    tr, dv = split_source([r for r in rows if r.get("split", "train") == "train"], dev_frac, seed, dev_cap)
    fams = list(dict.fromkeys(r["family"] for r in tr))
    tr = [r for fam in fams for r in _cap([r for r in tr if r["family"] == fam], max_rows_per_source, seed + 1)]
    out["fresh"] = {"train": tr, "dev": dv, "heldout": [r for r in rows if r.get("split") == "heldout"]}
    return out


def smoke_subset(corpus, n_train: int, n_dev_per_source: int, seed: int):
    """Round-robin over the non-empty sources: ``n_train`` train rows, a few dev rows each."""
    rng = random.Random(seed)
    pools = {s: list(v["train"]) for s, v in corpus.items() if v["train"]}
    for p in pools.values():
        rng.shuffle(p)
    train: List[Dict[str, Any]] = []
    while len(train) < n_train and any(pools.values()):
        for s in SOURCES:
            if pools.get(s) and len(train) < n_train:
                train.append(pools[s].pop())
    dev = [r for s in SOURCES for r in corpus.get(s, {}).get("dev", [])[:n_dev_per_source]]
    return train, dev


def encode_rows(rows, tok, seq_len: int):
    xs, ys, plens, meta = [], [], [], []
    dropped = 0
    for r in rows:
        ids, plen = tok.encode_turn(r["user"], r["assistant"])
        if len(ids) > seq_len:
            dropped += 1
            continue
        pad = seq_len - len(ids)
        xs.append(ids + [text_utils.PAD] * pad)
        ys.append([-100] * plen + ids[plen:] + [-100] * pad)
        plens.append(plen)
        meta.append(r)
    x = torch.tensor(xs, dtype=torch.long).view(-1, seq_len)
    y = torch.tensor(ys, dtype=torch.long).view(-1, seq_len)
    return x, y, torch.tensor(plens, dtype=torch.long), meta, dropped


# ---------------------------------------------------------------------------
# teacher caches
# ---------------------------------------------------------------------------
@torch.no_grad()
def arch_topk(teacher, tok_vocab_teacher: int, rows, x, plens, feats, k: int, batch: int, log_every: int = 20):
    """Top-``k`` Archimedes logits at every reply-predicting position of each row.

    Positions ``plen-1 .. n-2`` (the ones whose next token is a reply token).
    Student ids past the teacher's vocabulary are mapped to UNK first (the
    replay rows have none). Returns {row_key: (idx int16 (L, k), val fp16 (L, k))}.
    """
    teacher.eval()
    out = {}
    t0 = time.time()
    for bi, i in enumerate(range(0, x.shape[0], batch)):
        xb = x[i:i + batch].clone()
        xb[xb >= tok_vocab_teacher] = text_utils.UNK
        lg = teacher(xb, omni_features=feats[i:i + batch], return_mtp=False).logits
        for j in range(xb.shape[0]):
            r = rows[i + j]
            n = int((x[i + j] != text_utils.PAD).sum())
            p = int(plens[i + j])
            v, ix = lg[j, p - 1: n - 1].float().topk(k, dim=-1)
            out[ec.row_key(r["user"], r["assistant"])] = (ix.to(torch.int16), v.to(torch.float16))
        if log_every and (bi + 1) % log_every == 0:
            log(f"   kd_arch cache {i + xb.shape[0]}/{x.shape[0]} rows ({time.time() - t0:.0f}s)")
    return out


def kd_topk_loss(student: torch.Tensor, t_idx: torch.Tensor, t_val: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """KL(p_T || q_S) over the teacher's top-k ids at temperature T (x T^2).

    ``p_T`` is the teacher's softmax renormalised over its top-k logits; ``q_S``
    is the student's full-vocabulary softmax read at those ids."""
    s_logp = F.log_softmax(student.float() / T, dim=-1).gather(-1, t_idx)
    t_logp = F.log_softmax(t_val.float() / T, dim=-1)
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean() * (T * T)


class RowBoost:
    """Give selected rows of trunk tensors the graft learning rate.

    ``snapshot()`` before ``opt.step()``, ``apply()`` after: the rows' realised
    update (Adam step and decoupled decay alike) is multiplied by ``factor``.
    Tied tensors (embedding / lm_head) are one Parameter, handled once.
    """

    def __init__(self, targets: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]], factor: float):
        merged: Dict[int, Tuple[torch.nn.Parameter, torch.Tensor]] = {}
        for p, rows in targets:
            rows = torch.as_tensor(rows, dtype=torch.long).flatten()
            if rows.numel() == 0:
                continue
            if id(p) in merged:
                rows = torch.unique(torch.cat([merged[id(p)][1], rows]))
            merged[id(p)] = (p, rows)
        self.targets = list(merged.values())
        self.factor = float(factor)
        self._old: List[torch.Tensor] = []

    @property
    def n_rows(self) -> int:
        return sum(int(r.numel()) for _, r in self.targets)

    @torch.no_grad()
    def snapshot(self) -> None:
        self._old = [p.detach()[rows].clone() for p, rows in self.targets]

    @torch.no_grad()
    def apply(self) -> None:
        if self.factor == 1.0:
            return
        for (p, rows), old in zip(self.targets, self._old):
            p[rows] = old + self.factor * (p[rows] - old)
        self._old = []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def constant_state_keys(model) -> List[str]:
    """State entries training never changes (frozen encoders, v7 net, connectome W): left out of partials."""
    keys = []
    for k in model.state_dict().keys():
        if k.startswith("omni7.net."):
            keys.append(k)
        elif k.startswith("omni_core.") and not k.startswith(("omni_core.to_trunk", "omni_core.gate", "omni_core.bridge_norm")):
            keys.append(k)
        elif k.startswith("cns_full.") and k.split(".")[1] in ("crow", "col", "val", "crow_t", "col_t", "val_t", "in_idx", "out_idx", "sign"):
            keys.append(k)
    return keys


def atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def data_fingerprint(meta_tr, meta_dv, seq: int, vocab: int, tok_tag: str = "") -> str:
    """Identity of the packed data. ``tok_tag`` (:func:`tokenizer_tag`) is empty for a
    word tokenizer, so pre-v3 partials keep their fingerprint."""
    h = hashlib.sha1(f"{seq}|{vocab}".encode())
    if tok_tag:
        h.update(f"|{tok_tag}".encode())
    for tag, rows in (("t", meta_tr), ("d", meta_dv)):
        for r in rows:
            h.update(tag.encode())
            h.update(ec.row_key(r["user"], r["assistant"]).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# checkpoints + tokenizers (v1 / v2 schema, word / BPE)
# ---------------------------------------------------------------------------
def tokenizer_from_dict(d: Dict[str, Any]):
    """``expanse_core.tokenizer_from_dict`` (word or BPE); a word-only fallback until it exists."""
    fn = getattr(ec, "tokenizer_from_dict", None)
    if fn is not None:
        return fn(d)
    if (d or {}).get("kind", "word") != "word":
        raise RuntimeError(f"tokenizer kind {d.get('kind')!r} needs expanse_core.tokenizer_from_dict")
    return text_utils.WordTokenizer.from_dict(d)


def is_word_tokenizer(tok) -> bool:
    return isinstance(tok, text_utils.WordTokenizer)


def tokenizer_kind(tok) -> str:
    return "word" if is_word_tokenizer(tok) else str(tok.to_dict().get("kind", type(tok).__name__))


def tokenizer_tag(tok) -> str:
    """'' for a word tokenizer, else ``kind:<sha1 of its dict>`` (two BPEs of one size differ)."""
    if is_word_tokenizer(tok):
        return ""
    d = tok.to_dict()
    blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return f"{d.get('kind', type(tok).__name__)}:{hashlib.sha1(blob).hexdigest()[:16]}"


def peek_checkpoint(path, *keys: str) -> Dict[str, Any]:
    """Top-level entries of a torch checkpoint without materialising its tensors (``mmap``)."""
    try:
        head = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialisation cannot be mapped
        head = torch.load(path, map_location="cpu", weights_only=False)
    out = {k: head.get(k) for k in keys}
    del head
    return out


def checkpoint_schema(path) -> Optional[str]:
    return peek_checkpoint(path, "schema")["schema"]


def load_checkpoint(path):
    """(model, tokenizer, payload, schema) for a v1 or v2 Expanse checkpoint."""
    schema = checkpoint_schema(path)
    if schema == V2_SCHEMA:
        import consolidation_v2 as cv2
        model, _, payload = cv2.load_v2(path)
    else:
        model, _, payload = ec.load_expanse(path)
    return model, tokenizer_from_dict(payload["tokenizer"]), payload, schema


def save_checkpoint(path, schema: Optional[str], model, tok, extra: Dict[str, Any], receipt: Dict[str, Any]) -> None:
    """Save with the saver matching the input schema (v2 keeps its native stack)."""
    if schema == V2_SCHEMA:
        import consolidation_v2 as cv2
        cv2.save_v2(path, model, tok, extra, receipt)
    else:
        ec.save_expanse(path, model, tok, extra, receipt)


def arch_vocab_compatible(tok, arch_path) -> Tuple[bool, str]:
    """Can cached Archimedes logits supervise this tokenizer? (a word vocabulary extending Archimedes')."""
    if not is_word_tokenizer(tok):
        return False, f"tokenizer is {type(tok).__name__}, not Archimedes' word tokenizer"
    if not Path(arch_path).exists():
        return True, "word tokenizer (Archimedes checkpoint absent: vocabulary prefix not checked)"
    t_tok = text_utils.WordTokenizer.from_dict(peek_checkpoint(arch_path, "tokenizer")["tokenizer"])
    if list(tok.tokens[:t_tok.vocab_size]) != list(t_tok.tokens):
        return False, f"word vocabulary does not extend Archimedes' ({t_tok.vocab_size} tokens)"
    return True, f"word vocabulary extends Archimedes' ({t_tok.vocab_size} tokens)"


# ---------------------------------------------------------------------------
# parameter groups
# ---------------------------------------------------------------------------
def split_params(model) -> Tuple[List[str], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """(frozen names, graft params, trunk params); see the module doc's parameter groups.

    Frozen (``requires_grad`` off): the omni7 net and the omni v48/v38 encoders.
    Graft: fly_core, the omni_core bridge, the omni7 bridge + heads, cns_full,
    the v2 native latent blocks (``consolidation_v2.*``) and donor expert slots.
    Everything else is trunk (incl. the tied embedding / LM head).
    """
    donor = model.donor_slots()
    frozen: List[str] = []
    graft: List[torch.nn.Parameter] = []
    trunk: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if name.startswith("omni7.net.") or (name.startswith("omni_core.") and not name.startswith(
                ("omni_core.to_trunk", "omni_core.gate", "omni_core.bridge_norm"))):
            p.requires_grad_(False)
            frozen.append(name)
            continue
        is_graft = name.startswith(("fly_core.", "omni_core.", "omni7.", "cns_full.", "consolidation_v2."))
        if not is_graft and ".mlp.experts." in name:
            parts = name.split(".")
            is_graft = int(parts[4]) in donor.get(int(parts[1]), [])
        (graft if is_graft else trunk).append(p)
    return frozen, graft, trunk


def held_while_frozen(model, trunk_params: Sequence[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    """Trunk tensors ``--freeze_trunk_steps`` holds: all but the tied embedding / LM head."""
    tied = {id(model.embed_tokens.weight), id(model.lm_head.weight)}
    return [p for p in trunk_params if id(p) not in tied]


def drop_grads(params: Sequence[torch.nn.Parameter]) -> None:
    """``grad = None``: AdamW skips the tensor entirely (no step, no decoupled decay, no moment update)."""
    for p in params:
        p.grad = None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2: distil + fine-tune Supermix Expanse")
    ap.add_argument("--inp", default=None, help="grafted / v1 / v2 / v3-init checkpoint "
                                                  "(default checkpoints/supermix_expanse_grafted.pt)")
    ap.add_argument("--out", default=None, help="final checkpoint (default checkpoints/supermix_expanse.pt)")
    ap.add_argument("--arch", default=str(PATHS["arch_final"]), help="frozen Archimedes teacher for kd_arch")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=None, help="default: the checkpoint's max_position_embeddings")
    ap.add_argument("--v3_data", choices=("auto", "on", "off"), default="auto",
                    help="auto: data/v3 files where present; off: the v1 corpus exactly; on: require them")
    ap.add_argument("--v3_dir", default=str(PATHS["data"] / "v3"), help="where make_v3_data.py wrote its files")
    ap.add_argument("--boost_all_embeddings", action="store_true",
                    help="give every embedding row the graft lr (automatic right after retokenize_v3)")
    ap.add_argument("--freeze_trunk_steps", type=int, default=0,
                    help="first N steps: only the tied embedding/LM-head rows and graft/v2 params update")
    ap.add_argument("--v2_balance", type=float, default=0.01, help="v2 native router balance weight (v2 inputs only)")
    ap.add_argument("--fly_rows", type=int, default=4000)
    ap.add_argument("--dev_frac", type=float, default=0.05)
    ap.add_argument("--dev_cap", type=int, default=200)
    ap.add_argument("--max_rows_per_source", type=int, default=6000)
    ap.add_argument("--lr_trunk", type=float, default=3e-5)
    ap.add_argument("--lr_graft", type=float, default=2e-4)
    ap.add_argument("--kd_arch", type=float, default=0.5)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--kd_topk", type=int, default=32)
    ap.add_argument("--omni7_kd", type=float, default=0.2)
    ap.add_argument("--fly_aux", type=float, default=1.0)
    ap.add_argument("--wake", type=str, default="0.1,0.5")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--kd_cache", default=str(PATHS["data"] / "kd_arch_top32.pt"))
    ap.add_argument("--omni7_cache", default=str(PATHS["data"] / "omni7_states.pt"))
    ap.add_argument("--resume", default="auto", help="auto | none | <partial path>")
    ap.add_argument("--stop_after", type=int, default=0, help="stop (and save the partial) after N steps in this run")
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--finalize_on_budget", action="store_true", help="also write the final checkpoint when the budget ends a run")
    ap.add_argument("--smoke", action="store_true", help="2 steps on 16 rows; caches in memory only")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    smoke_dir = PATHS["checkpoints"] / "smoke"
    if args.smoke:
        args.steps = 2 if args.steps == 600 else args.steps
        args.eval_every = max(args.eval_every, args.steps + 1) if args.eval_every == 50 else args.eval_every
        args.save_every = 1
        args.fly_rows = min(args.fly_rows, 64)
        if args.inp is None and (smoke_dir / "supermix_expanse_grafted_smoke.pt").exists():
            args.inp = str(smoke_dir / "supermix_expanse_grafted_smoke.pt")
        if args.out is None:
            args.out = str(smoke_dir / "supermix_expanse_smoke.pt")
    args.inp = args.inp or str(PATHS["grafted"])
    args.out = args.out or str(PATHS["final"])
    out_path = Path(args.out)
    partial_path = out_path.with_name(out_path.stem + ".partial.pt")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    t0 = time.time()

    # model --------------------------------------------------------------------
    model, tok, payload, schema = load_checkpoint(args.inp)
    receipt_in = dict(payload.get("expanse") or {})
    payload["state_dict"] = None
    stack = getattr(model, "consolidation_v2", None)  # v2 native latent blocks (None for v1)
    vocab_base = int(model.vocab_base or (receipt_in.get("vocab") or {}).get("base") or tok.vocab_size)
    args.seq = int(args.seq or model.config.max_position_embeddings)
    tok_kind = tokenizer_kind(tok)
    # Boost every embedding row only on the first run after retokenize_v3 (all rows are then freshly
    # re-initialised); a continuation of a trained v3 keeps the normal vocab_base rule.
    retokenized = (str(receipt_in.get("stage", "")).startswith("v3-init") or args.boost_all_embeddings)
    log(f"loaded {args.inp} ({schema}): {sum(p.numel() for p in model.parameters()):,} params, {tok_kind} vocab "
        f"{tok.vocab_size} (base {vocab_base}), seq {args.seq}" + (", v2 native stack" if stack is not None else ""))
    fly_teacher = ec.FlyCore(int(model.config.hidden_size), config=model.fly_config)
    fly_teacher.load_state_dict(model.fly_core.state_dict())  # frozen copy of the grafted (= original) fly core
    fly_teacher.requires_grad_(False).eval()
    kd_arch_note = None
    if args.kd_arch > 0:
        ok, why = arch_vocab_compatible(tok, args.arch)
        if not ok:
            log(f"kd_arch disabled: {why} -- Archimedes' logits index a different vocabulary")
            kd_arch_note = {"requested": args.kd_arch, "disabled": why}
            args.kd_arch = 0.0

    # data -----------------------------------------------------------------------
    v3_mode = {"auto": None, "on": True, "off": False}[args.v3_data]
    v3_paths = {"v3": Path(args.v3_dir)}
    files = corpus_files(v3_paths, v3_mode)
    corpus = load_corpus(fly_teacher, fly_rows=args.fly_rows, seed=args.seed, dev_frac=args.dev_frac,
                         dev_cap=args.dev_cap, max_rows_per_source=args.max_rows_per_source, data=v3_paths, v3=v3_mode)
    log(f"corpus files: connectome {files['connectome'].name}, fresh {[p.name for p in files['fresh']] or 'none'}"
        f" (v3 data {'on' if files['v3'] else 'off'})")
    counts = {s: {k: len(v) for k, v in d.items()} for s, d in corpus.items()}
    log("rows " + " ".join(f"{s}: train {c['train']} dev {c['dev']}" for s, c in counts.items()))
    missing = [s for s in ("code", "bio", "connectome") if counts[s]["train"] == 0]
    if missing:
        log(f"WARNING no rows for {missing} (data files absent?) -- training without them")
    if args.smoke:
        train_rows, dev_rows = smoke_subset(corpus, 16, 2, args.seed)
    else:
        train_rows = [r for s in SOURCES for r in corpus[s]["train"]]
        dev_rows = [r for s in SOURCES for r in corpus[s]["dev"]]
    x_tr, y_tr, pl_tr, m_tr, d1 = encode_rows(train_rows, tok, args.seq)
    x_dv, y_dv, pl_dv, m_dv, d2 = encode_rows(dev_rows, tok, args.seq)
    unk = float((x_tr == text_utils.UNK).float().sum() / (x_tr != text_utils.PAD).float().sum().clamp_min(1))
    log(f"packed train {tuple(x_tr.shape)} dev {tuple(x_dv.shape)} dropped {d1}+{d2} over-length; unk rate {unk:.4f}")
    src_tr = [r["source"] for r in m_tr]
    src_dv = [r["source"] for r in m_dv]
    fingerprint = data_fingerprint(m_tr, m_dv, args.seq, tok.vocab_size, tokenizer_tag(tok))

    # prompt-constant features: omni v48/v38 and omni7 -----------------------------
    feats_tr = ec.arch.OmniCore.featurize([r["user"] for r in m_tr])
    feats_dv = ec.arch.OmniCore.featurize([r["user"] for r in m_dv])
    o7_tr = o7_dv = None
    if model.omni7 is not None:
        o7_sha = str(((receipt_in.get("grafts") or {}).get("omni7") or {}).get("sha256", "unknown"))
        cache: Dict[str, torch.Tensor] = {}
        cpath = Path(args.omni7_cache)
        if not args.smoke and cpath.exists():
            c = torch.load(cpath, map_location="cpu", weights_only=False)
            if c.get("version") == O7_CACHE_VERSION and c.get("v7_sha256") == o7_sha:
                cache = dict(zip(c["keys"], c["state"]))
        users = sorted({r["user"] for r in m_tr + m_dv})
        need = [u for u in users if ec.row_key(u, "") not in cache]
        if need:
            log(f"omni7 states: encoding {len(need)} prompts ({len(users) - len(need)} cached)")
            t1 = time.time()
            for i in range(0, len(need), 64):
                st = model.omni7_state_for(need[i:i + 64])
                for u, s in zip(need[i:i + 64], st):
                    cache[ec.row_key(u, "")] = s.clone()
                if (i // 64 + 1) % 20 == 0:
                    log(f"   {i + 64}/{len(need)} ({time.time() - t1:.0f}s)")
            if not args.smoke:
                keys = list(cache.keys())
                atomic_save({"version": O7_CACHE_VERSION, "v7_sha256": o7_sha, "keys": keys,
                             "state": torch.stack([cache[k] for k in keys])}, cpath)
        o7_tr = torch.stack([cache[ec.row_key(r["user"], "")] for r in m_tr])
        o7_dv = torch.stack([cache[ec.row_key(r["user"], "")] for r in m_dv])
        del cache
        fh, ni = int(model.omni7.fusion_hidden), int(model.omni7.n_intents)

    # kd_arch cache (replay rows) ---------------------------------------------------------
    kd: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    replay_idx = [i for i, s in enumerate(src_tr) if s == "replay"]
    if args.kd_arch > 0 and replay_idx:
        arch_sha = ec.sha256_file(args.arch)
        kpath = Path(args.kd_cache)
        if not args.smoke and kpath.exists():
            c = torch.load(kpath, map_location="cpu", weights_only=False)
            if (c.get("version"), c.get("arch_sha256"), c.get("k"), c.get("seq")) == (KD_CACHE_VERSION, arch_sha, args.kd_topk, args.seq):
                kd = c["entries"]
        # smoke: only the rows this run trains on; full: every replay row (train + dev) once
        want = [(x_tr, pl_tr, m_tr, feats_tr, i) for i in replay_idx]
        if not args.smoke:
            want += [(x_dv, pl_dv, m_dv, feats_dv, i) for i, s in enumerate(src_dv) if s == "replay"]
        want = [w for w in want if ec.row_key(w[2][w[4]]["user"], w[2][w[4]]["assistant"]) not in kd]
        if want:
            log(f"kd_arch: caching top-{args.kd_topk} Archimedes logits for {len(want)} replay rows")
            teacher, t_tok, _ = ec.load_archimedes(args.arch)
            teacher.requires_grad_(False)
            xs = torch.stack([w[0][w[4]] for w in want])
            pls = torch.stack([w[1][w[4]] for w in want])
            fs = torch.stack([w[3][w[4]] for w in want])
            rs = [w[2][w[4]] for w in want]
            kd.update(arch_topk(teacher, t_tok.vocab_size, rs, xs, pls, fs, args.kd_topk, batch=16))
            del teacher, xs
            if not args.smoke:
                atomic_save({"version": KD_CACHE_VERSION, "arch_sha256": arch_sha, "k": args.kd_topk, "seq": args.seq,
                             "entries": kd}, kpath)
        log(f"kd_arch cache: {len(kd)} rows")
    kd_tr = [kd.get(ec.row_key(r["user"], r["assistant"])) if s == "replay" else None for r, s in zip(m_tr, src_tr)]
    del kd

    is_fly_tr = torch.tensor([s == "fly" for s in src_tr])
    obs_tr = torch.tensor([r.get("obs", [0.0] * 14) for r in m_tr], dtype=torch.float32)
    probs_tr = torch.tensor([r.get("probs", [0.25] * 4) for r in m_tr], dtype=torch.float32).clamp_min(1e-6)
    probs_tr = probs_tr / probs_tr.sum(-1, keepdim=True)

    # parameter groups ---------------------------------------------------------------
    donor = model.donor_slots()
    frozen, graft_params, trunk_params = split_params(model)
    n_trunk, n_graft = sum(p.numel() for p in trunk_params), sum(p.numel() for p in graft_params)
    log(f"trainable: trunk {n_trunk:,} graft {n_graft:,}; frozen {len(frozen)} tensors "
        f"({sum(p.numel() for n, p in model.named_parameters() if n in set(frozen)):,} params)")
    opt = torch.optim.AdamW([
        {"params": trunk_params, "lr": args.lr_trunk, "weight_decay": 0.01},
        {"params": graft_params, "lr": args.lr_graft, "weight_decay": 0.0},
    ], betas=(0.9, 0.95))
    boost_targets = []
    if retokenized:  # every row was re-initialised by retokenize_v3
        boost_targets.append((model.embed_tokens.weight, torch.arange(tok.vocab_size)))
    elif tok.vocab_size > vocab_base:
        boost_targets.append((model.embed_tokens.weight, torch.arange(vocab_base, tok.vocab_size)))
    for li, slots in donor.items():
        boost_targets.append((model.layers[li].mlp.gate.weight, torch.tensor(slots)))
    boost = RowBoost(boost_targets, args.lr_graft / args.lr_trunk)
    log(f"row boost x{boost.factor:.2f} on {boost.n_rows} rows ("
        + ("all embedding rows (retokenised)" if retokenized else "new embeddings") + " + donor router rows)")
    # --freeze_trunk_steps: trunk tensors other than the tied embedding / LM head sit still
    held_early = held_while_frozen(model, trunk_params)
    if args.freeze_trunk_steps > 0:
        log(f"freeze_trunk_steps {args.freeze_trunk_steps}: until then only the tied embedding/LM head "
            f"({model.embed_tokens.weight.shape[0]} rows) + graft/v2 params move; "
            f"{len(held_early)} trunk tensors ({sum(p.numel() for p in held_early):,} params) held")
    warm = max(1, int(0.05 * args.steps))

    def lr_scale(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, args.steps - warm)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

    wake0, wake1 = [float(v) for v in args.wake.split(",")]

    def wake_progress(step: int) -> Optional[float]:
        prog = step / max(1, args.steps)
        return None if prog < wake0 else min(1.0, (prog - wake0) / max(1e-6, wake1 - wake0))

    # evaluation ------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(tag: str) -> Dict[str, Any]:
        model.eval()
        tot: Dict[str, float] = {}
        cnt: Dict[str, int] = {}
        fly_hit = fly_n = 0
        sense_err = 0.0
        o7_hit = [0, 0]
        o7_n = 0
        for i in range(0, x_dv.shape[0], args.batch):
            xb, yb = x_dv[i:i + args.batch], y_dv[i:i + args.batch]
            kw = {"omni_features": feats_dv[i:i + args.batch]}
            if o7_dv is not None:
                kw["omni7_state"] = o7_dv[i:i + args.batch]
            out = model(xb, labels=yb, return_mtp=False, **kw)
            ce = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.shape[-1]), yb[:, 1:].reshape(-1),
                                 reduction="none").view(xb.shape[0], -1)
            m = (yb[:, 1:] != -100)
            for j in range(xb.shape[0]):
                s = src_dv[i + j]
                tot[s] = tot.get(s, 0.0) + float(ce[j][m[j]].sum())
                cnt[s] = cnt.get(s, 0) + int(m[j].sum())
            info = model.last_graft_info.get("fly", {})
            plb = pl_dv[i:i + args.batch]
            if model.omni7 is not None:
                h = out.hidden_states[torch.arange(xb.shape[0]), plb - 1]
                il, dl = model.omni7.aux_logits(h)
                st = kw["omni7_state"]
                o7_hit[0] += int((il.argmax(-1) == st[:, fh:fh + ni].argmax(-1)).sum())
                o7_hit[1] += int((dl.argmax(-1) == st[:, fh + ni:].argmax(-1)).sum())
                o7_n += xb.shape[0]
            for j, r in enumerate(m_dv[i:i + args.batch]):
                if r["source"] != "fly":
                    continue
                fly_n += 1
                pr = model.fly_core.brains_forward(torch.tensor([r["obs"]]))["probs"][0]
                fly_hit += int(pr.argmax() == torch.tensor(r["probs"]).argmax())
                if "obs" in info and info["obs"].dim() == 3:
                    sense_err += float((info["obs"][j, int(plb[j]) - 1] - torch.tensor(r["obs"])).abs().mean())
        model.train()
        rep: Dict[str, Any] = {"dev_loss": sum(tot.values()) / max(1, sum(cnt.values()))}
        for s in SOURCES:
            if cnt.get(s):
                rep[f"dev_{s}"] = tot[s] / cnt[s]
        if fly_n:
            rep.update({"fly_port_agree": fly_hit / fly_n, "sense_mae": sense_err / fly_n})
        if o7_n:
            rep.update({"omni7_intent_agree": o7_hit[0] / o7_n, "omni7_domain_agree": o7_hit[1] / o7_n})
        rep.update(model.gate_report())
        if stack is not None:
            rep.update(stack.report())
        log(f"[eval {tag}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items()))
        return rep

    # resume --------------------------------------------------------------------------------
    step, ptr = 0, 0
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(x_tr.shape[0], generator=g)
    history: List[Dict[str, Any]] = []
    train_seconds = 0.0
    resumed_from = None
    rp = None if args.resume == "none" else (partial_path if args.resume == "auto" else Path(args.resume))
    const_keys = set(constant_state_keys(model))
    if rp is not None and rp.exists():
        part = torch.load(rp, map_location="cpu", weights_only=False)
        if part.get("schema") != PARTIAL_SCHEMA:
            raise ValueError(f"{rp} is not a {PARTIAL_SCHEMA} file")
        if part.get("fingerprint") != fingerprint:
            raise ValueError(f"{rp} was written for different data/tokenizer (fingerprint mismatch); "
                             "pass --resume none to start over")
        for k in ("steps", "batch", "seq", "lr_trunk", "lr_graft"):
            if part["args"].get(k) != getattr(args, k):
                log(f"WARNING resuming with {k}={getattr(args, k)} (partial had {part['args'].get(k)})")
        missing_k, unexpected_k = model.load_state_dict(part["model"], strict=False)
        bad = [k for k in missing_k if k not in const_keys]
        if bad or unexpected_k:
            raise RuntimeError(f"partial state mismatch: missing {bad[:5]} unexpected {list(unexpected_k)[:5]}")
        opt.load_state_dict(part["optimizer"])
        step, idx, ptr = int(part["step"]), part["order"]["idx"], int(part["order"]["ptr"])
        g.set_state(part["order"]["gen"])
        history = list(part.get("history", []))
        train_seconds = float(part.get("train_seconds", 0.0))
        torch.set_rng_state(part["rng"]["torch"])
        random.setstate(part["rng"]["python"])
        np.random.set_state(part["rng"]["numpy"])
        resumed_from = {"path": str(rp), "step": step}
        log(f"resumed from {rp} at step {step}/{args.steps}")
        del part

    saved_at = {"step": -1}

    def save_partial(reason: str) -> None:
        if saved_at["step"] == step:
            return  # this exact state is already on disk
        saved_at["step"] = step
        t1 = time.time()
        state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k not in const_keys}
        atomic_save({"schema": PARTIAL_SCHEMA, "step": step, "model": state, "optimizer": opt.state_dict(),
                     "order": {"idx": idx, "ptr": ptr, "gen": g.get_state()}, "history": history, "fingerprint": fingerprint,
                     "args": vars(args), "inp": args.inp, "reason": reason,
                     "train_seconds": train_seconds + (time.time() - t_run),
                     "rng": {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}},
                    partial_path)
        log(f"partial checkpoint ({reason}) step {step} -> {partial_path} in {time.time() - t1:.1f}s")

    # train ---------------------------------------------------------------------------------
    if not history:
        history.append({"step": step, **evaluate("start")})
    model.train()
    t_run = time.time()
    steps_this_run = 0
    ended_by = "done"
    last_losses: Dict[str, float] = {}
    while step < args.steps:
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            ended_by = "budget"
            break
        if args.stop_after and steps_this_run >= args.stop_after:
            ended_by = "stop_after"
            break
        if ptr + args.batch > idx.numel():
            idx = torch.randperm(x_tr.shape[0], generator=g)
            ptr = 0
        bi = idx[ptr:ptr + args.batch]
        ptr += args.batch
        xb, yb, plb = x_tr[bi], y_tr[bi], pl_tr[bi]
        wp = wake_progress(step)
        if wp is not None:
            model.wake_donor_experts(wp)
        for grp, base in zip(opt.param_groups, (args.lr_trunk, args.lr_graft)):
            grp["lr"] = base * lr_scale(step)
        kw = {"omni_features": feats_tr[bi]}
        if o7_tr is not None:
            kw["omni7_state"] = o7_tr[bi]
        out = model(xb, labels=yb, **kw)
        loss = out.loss
        losses = {"model": float(out.loss.detach()), "lm": float(out.lm_loss.detach())}
        # kd_arch on replay rows (cached top-k Archimedes logits)
        if args.kd_arch > 0:
            s_rows, t_idx, t_val = [], [], []
            for j, b in enumerate(bi.tolist()):
                ent = kd_tr[b]
                if ent is None:
                    continue
                n = int((xb[j] != text_utils.PAD).sum())
                p = int(plb[j])
                if ent[0].shape[0] != n - p:
                    continue  # tokenisation changed since caching: skip rather than misalign
                s_rows.append(out.logits[j, p - 1: n - 1])
                t_idx.append(ent[0].long())
                t_val.append(ent[1].float())
            if s_rows:
                l_kd = kd_topk_loss(torch.cat(s_rows), torch.cat(t_idx), torch.cat(t_val), args.kd_T)
                loss = loss + args.kd_arch * l_kd
                losses["kd_arch"] = float(l_kd.detach())
        # omni7 intent/domain distillation (every row)
        if args.omni7_kd > 0 and model.omni7 is not None:
            h = out.hidden_states[torch.arange(xb.shape[0]), plb - 1]
            il, dl = model.omni7.aux_logits(h)
            st = kw["omni7_state"]
            l_o7 = (F.kl_div(F.log_softmax(il, -1), st[:, fh:fh + ni], reduction="batchmean")
                    + F.kl_div(F.log_softmax(dl, -1), st[:, fh + ni:], reduction="batchmean"))
            loss = loss + args.omni7_kd * l_o7
            losses["omni7_kd"] = float(l_o7.detach())
        # fly: brains match the frozen original; senses readable from the text
        fly_mask = is_fly_tr[bi]
        if args.fly_aux > 0 and bool(fly_mask.any()):
            fo = model.fly_core.brains_forward(obs_tr[bi][fly_mask])
            l_port = F.kl_div(F.log_softmax(fo["consensus"], -1), probs_tr[bi][fly_mask], reduction="batchmean")
            sensed = model.last_graft_info["fly"]["obs"][torch.arange(xb.shape[0]), plb - 1][fly_mask]
            l_sense = F.mse_loss(sensed, obs_tr[bi][fly_mask])
            loss = loss + args.fly_aux * (l_port + l_sense)
            losses["fly_port"] = float(l_port.detach())
            losses["fly_sense"] = float(l_sense.detach())
        # v2: keep the native latent routers balanced (train_consolidation_v2's term; min 0 when balanced)
        if stack is not None and args.v2_balance > 0:
            l_bal = stack.aux_loss()
            loss = loss + args.v2_balance * l_bal
            losses["v2_balance"] = float(l_bal.detach())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step < args.freeze_trunk_steps:
            drop_grads(held_early)
        elif step == args.freeze_trunk_steps and args.freeze_trunk_steps > 0:
            log(f"trunk unfrozen at step {step}")
        torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]], 1.0)
        boost.snapshot()
        opt.step()
        boost.apply()
        step += 1
        steps_this_run += 1
        last_losses = losses
        if step % 10 == 0 or step == 1 or args.smoke:
            el = time.time() - t_run
            log(f"[step {step}/{args.steps}] loss {float(loss.detach()):.4f} " + " ".join(f"{k} {v:.4f}" for k, v in losses.items())
                + f" lr {opt.param_groups[0]['lr']:.2e} wake {wp if wp is not None else 0.0:.2f} {el / steps_this_run:.2f}s/step "
                  f"eta {(args.steps - step) * el / steps_this_run / 60:.1f}m")
        if step % args.eval_every == 0 and step < args.steps:
            history.append({"step": step, **evaluate(str(step))})
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            save_partial("rolling")

    if step < args.steps:
        save_partial(ended_by)
        if not (ended_by == "budget" and args.finalize_on_budget):
            log(f"stopped at step {step}/{args.steps} ({ended_by}); rerun the same command to resume")
            return 0

    model.wake_donor_experts(1.0)
    final = evaluate("final")
    history.append({"step": step, **final})
    model.eval()
    receipt = dict(receipt_in)
    prior = receipt_in.get("training") or {}
    receipt.update({
        "stage": "trained" if step >= args.steps else "partially-trained",
        "training": {
            "steps": step, "target_steps": args.steps, "batch": args.batch, "seq": args.seq,
            "lr_trunk": args.lr_trunk, "lr_graft": args.lr_graft, "kd_arch": args.kd_arch, "kd_T": args.kd_T,
            "kd_topk": args.kd_topk, "omni7_kd": args.omni7_kd, "fly_aux": args.fly_aux, "wake": args.wake,
            "seed": args.seed, "smoke": bool(args.smoke),
            "init": {"path": str(args.inp), "schema": schema, "stage": receipt_in.get("stage"),
                     "prior_training": {k: prior.get(k) for k in ("steps", "seq", "seed", "corpus_args")} if prior else None},
            "tokenizer": {"kind": tok_kind, "vocab": tok.vocab_size, "retokenized": retokenized,
                          "kd_arch": kd_arch_note or ("enabled" if args.kd_arch > 0 else "off (--kd_arch 0)")},
            "freeze_trunk_steps": args.freeze_trunk_steps,
            "v2_balance": args.v2_balance if stack is not None else None,
            "corpus_args": {"fly_rows": args.fly_rows, "dev_frac": args.dev_frac, "dev_cap": args.dev_cap,
                            "max_rows_per_source": args.max_rows_per_source, "seed": args.seed,
                            "v3_data": bool(files["v3"]), "v3_dir": files["v3_dir"].as_posix()},
            "corpus_files": {"connectome": files["connectome"].as_posix(), "fresh": [p.as_posix() for p in files["fresh"]]},
            "rows": counts, "packed": {"train": int(x_tr.shape[0]), "dev": int(x_dv.shape[0]), "dropped": d1 + d2},
            "dev_keys": {s: sorted(ec.row_key(r["user"], r["assistant"]) for r, ss in zip(m_dv, src_dv) if ss == s)
                         for s in SOURCES},
            "kd_arch_rows": sum(1 for e in kd_tr if e is not None), "unk_rate": unk,
            "params": {"trunk": n_trunk, "graft": n_graft, "frozen_tensors": len(frozen), "row_boost_rows": boost.n_rows,
                       "row_boost_factor": boost.factor},
            "fly_distillation": "self-distilled from the frozen original FlyCore (experience log unavailable)",
            "resumed_from": resumed_from, "ended_by": ended_by, "last_losses": last_losses,
            "train_seconds": round(train_seconds + time.time() - t_run, 1),
            "wall_minutes_this_run": round((time.time() - t0) / 60, 2), "history": history, "final": final,
        },
    })
    extra = dict(payload.get("extra") or {})
    extra.update({"note": "stage 2: distilled + fine-tuned" + (" (smoke)" if args.smoke else ""), "steps": step,
                  "best_dev_loss": final["dev_loss"]})
    save_checkpoint(out_path, schema, model, tok, extra, receipt)
    out_path.with_suffix(".receipt.json").write_text(
        json.dumps(ec.jsonable({k: v for k, v in receipt.items() if k != "omni7_meta"}), indent=1), encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()  # this run's own rolling file; the final checkpoint supersedes it
    log(f"saved {out_path} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
