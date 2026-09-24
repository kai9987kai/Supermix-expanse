"""Tests for the v3 tokenizer swap: src/bpe_tokenizer.py, expanse_core.tokenizer_from_dict
and retokenize_v3.py (V3_DESIGN.md "A").

What the v3 trainer, eval and chat code rely on:

1. ``BPETokenizer`` is a drop-in for ``WordTokenizer``: the six specials at ids
   0-5 with the same strings, ``encode_turn`` with the identical structure and
   prompt length, ``decode`` dropping specials, ``unknown_rate`` 0.
2. ``decode(encode(t)) == t`` for ASCII and UTF-8 text -- including text that
   *contains* a special-token string, which must stay text (never an id 0-5).
3. Digits are always single tokens (no merge can fuse two digits).
4. ``to_dict``/``from_dict`` round-trip (JSON-safe, special handling restored)
   and ``tokenizer_from_dict`` dispatches on ``kind``.
5. The embedding re-initialisation follows its four rules exactly, and a
   retokenised model -- v1 schema and v2 schema -- saves, reloads with a
   ``BPETokenizer`` and gives the same logits.

Everything is tiny: BPE vocabularies of a few hundred on synthetic text and a
3-layer 32-d ``ExpanseModel`` without omni/omni7/connectome grafts, so no
checkpoint is needed and the file runs in seconds.

Run: ``python -m pytest -q expanse/tests/test_bpe_tokenizer.py``
"""

from __future__ import annotations

import json
import re
import math
import random
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "src"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bpe_tokenizer as bt  # noqa: E402
import expanse_core as ec  # noqa: E402
import retokenize_v3 as rt  # noqa: E402
from expanse_core import text_utils  # noqa: E402

torch.set_num_threads(2)

WORDS = ("the synapse neuron releases acetylcholine glutamate cardiovascular cardiomyocyte mitochondria "
         "function returns value list index loop while sum total energy momentum velocity average speed "
         "presynaptic postsynaptic inhibitory excitatory descending ascending protein binding receptor").split()


def synthetic_rows(n: int = 240, seed: int = 0):
    """House-style (user, assistant) rows with words, numbers, code and punctuation runs."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randint(0, 999), rng.randint(0, 999)
        w = rng.sample(WORDS, 4)
        kind = i % 4
        if kind == 0:
            user, reply = f"What is {a} + {b}?", f"{a} + {b} = {a + b}, total {a + b}"
        elif kind == 1:
            user = f"def f(xs):\n    return sum(xs) + {a}\nWhat does f([{b}]) return?"
            reply = f"sum([{b}]) is {b}, plus {a}, total {a + b}"
        elif kind == 2:
            user, reply = f"Which {w[0]} does the {w[1]} use?", f"The {w[1]} uses {w[2]} ({w[3]}), total {a}"
        else:
            user, reply = f"A {w[0]} moves {a} m in {b % 50 + 1} s. Speed?", f"speed = {a} / {b % 50 + 1}, the {w[1]}."
        rows.append({"user": user, "assistant": reply, "source": "replay"})
    return rows


ROWS = synthetic_rows()
TEXTS = [t for r in ROWS for t in (r["user"], r["assistant"])]

SAMPLES = [
    "What is 12 + 7?",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\n\n  indented\r\nwindows",
    "héllo wörld -- naïve café, Größe, ½ + ¾",
    "日本語のテキスト 中文 한국어",
    "emoji 🧠🪰 and ZWJ 👩‍🔬",
    "a literal <eos> and <pad><user>hi<assistant> inside text",
    "def f(x):\n    return x ** 2  # 3.14159e-10\n",
    "",
    " ",
    "\n",
    "x" * 300,
]


@pytest.fixture(scope="module")
def tok():
    return bt.train_bpe(TEXTS, vocab_size=400, min_frequency=2)


# ---------------------------------------------------------------------------
# 1-3. the tokenizer
# ---------------------------------------------------------------------------
def test_specials_at_0_5(tok):
    assert tuple(tok.tokens[:6]) == text_utils.SPECIAL_TOKENS
    for i, s in enumerate(text_utils.SPECIAL_TOKENS):
        assert tok.index[s] == i and tok.backend.token_to_id(s) == i
    assert (bt.PAD, bt.BOS, bt.EOS, bt.UNK, bt.USER, bt.ASSISTANT) == (0, 1, 2, 3, 4, 5)
    assert (text_utils.PAD, text_utils.BOS, text_utils.EOS, text_utils.UNK, text_utils.USER,
            text_utils.ASSISTANT) == (0, 1, 2, 3, 4, 5)
    assert 6 + 256 < tok.vocab_size <= 400 and tok.vocab_size == len(tok.tokens)
    assert tok.digit_tokens is True and tok.reverse_digits is False
    # tokens are decoded strings (" the"), not byte-level symbols ("Ġthe")
    assert " the" in tok.index and not any(t.startswith("Ġ") and len(t) > 1 for t in tok.tokens)


@pytest.mark.parametrize("text", SAMPLES)
def test_round_trip(tok, text):
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert all(i >= len(text_utils.SPECIAL_TOKENS) for i in ids), "special string in text became a special id"
    assert tok.unknown_rate(text) == 0.0


def test_round_trip_corpus(tok):
    assert all(tok.decode(tok.encode(t)) == t for t in TEXTS)
    assert tok.encode_batch(TEXTS[:20]) == [tok.encode(t) for t in TEXTS[:20]]


def test_digits_split(tok):
    ids = tok.encode("12345")
    assert [tok.tokens[i] for i in ids] == list("12345")
    for text in ("x = 2024 + 17", "total 1000000", "3.14159", "a1b22c333"):
        for i in tok.encode(text):
            t = tok.tokens[i]
            if any(c.isdigit() for c in t):
                assert re.fullmatch(r" ?\d", t), f"{t!r} fuses a digit with something else"
    # a number's first digit keeps its leading space (" 2024" -> " 2", "0", "2", "4"); nothing else merges
    assert [tok.tokens[i] for i in tok.encode("total 2024")][-4:] == [" 2", "0", "2", "4"]
    assert all(re.fullmatch(r" ?\d", t) for t in tok.tokens if any(c in "0123456789" for c in t))


def test_encode_turn_structure(tok):
    user, reply = "What is 2 + 3?", "2 + 3 = 5, total 5"
    ids, plen = tok.encode_turn(user, reply)
    u, a = tok.encode(user), tok.encode(reply)
    assert ids == [bt.BOS, bt.USER] + u + [bt.ASSISTANT] + a + [bt.EOS]
    assert plen == 3 + len(u) and ids[plen - 1] == bt.ASSISTANT
    p_ids, p_len = tok.encode_turn(user)
    assert p_ids == ids[:plen] and p_len == plen == len(p_ids)
    # identical layout to the word tokenizer's (only the text ids differ)
    word = text_utils.WordTokenizer.build(TEXTS, digit_tokens=True)
    w_ids, w_plen = word.encode_turn(user, reply)
    wu = word.encode(user)
    assert w_ids[:2] == ids[:2] and w_ids[w_plen - 1] == ids[plen - 1] and w_ids[-1] == ids[-1]
    assert w_plen == 3 + len(wu)


def test_decode_drops_specials_and_out_of_range(tok):
    text = "The synapse, total 42"
    ids = [bt.BOS, bt.USER] + tok.encode(text) + [bt.ASSISTANT, bt.EOS, bt.PAD, bt.UNK, 10 ** 6, -1]
    assert tok.decode(ids) == text
    assert tok.decode([]) == "" and tok.decode([bt.EOS]) == ""


def test_to_from_dict(tok, tmp_path):
    d = tok.to_dict()
    assert d["kind"] == "bpe" and d["digit_tokens"] is True and isinstance(d["json"], str)
    spec = json.loads(d["json"])
    assert spec["model"]["type"] == "BPE" and spec["decoder"]["type"] == "ByteLevel"
    assert [p["type"] for p in spec["pre_tokenizer"]["pretokenizers"]] == ["Split", "ByteLevel"]
    assert spec["pre_tokenizer"]["pretokenizers"][0]["behavior"] == "Isolated"
    assert spec["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] is False
    back = bt.BPETokenizer.from_dict(json.loads(json.dumps(d)))  # JSON-safe as stored in a checkpoint
    assert back.tokens == tok.tokens and back.index == tok.index and back.vocab_size == tok.vocab_size
    for t in SAMPLES:
        assert back.encode(t) == tok.encode(t)
    assert all(i >= 6 for i in back.encode("<eos>")), "special-as-text handling must survive a reload"
    tok.save(tmp_path / "tok.json")
    assert bt.BPETokenizer.load(tmp_path / "tok.json").tokens == tok.tokens
    with pytest.raises(ValueError):
        bt.BPETokenizer.from_dict({"tokens": ["<pad>"], "digit_tokens": True})


def test_tokenizer_from_dict_dispatch(tok):
    got = ec.tokenizer_from_dict(tok.to_dict())
    assert isinstance(got, bt.BPETokenizer) and got.tokens == tok.tokens
    word = text_utils.WordTokenizer.build(TEXTS, digit_tokens=True)
    got_w = ec.tokenizer_from_dict(word.to_dict())
    assert isinstance(got_w, text_utils.WordTokenizer) and got_w.tokens == word.tokens and got_w.digit_tokens


def test_training_is_deterministic(tok):
    again = bt.train_bpe(TEXTS, vocab_size=400, min_frequency=2)
    assert again.to_dict()["json"] == tok.to_dict()["json"]


# ---------------------------------------------------------------------------
# 5a. embedding rules
# ---------------------------------------------------------------------------
def test_init_embedding_rules(tok):
    old_tok = text_utils.WordTokenizer.build(TEXTS, max_vocab=70, digit_tokens=True)
    g = torch.Generator().manual_seed(0)
    old = torch.randn(old_tok.vocab_size, 16, generator=g) * torch.linspace(0.5, 2.0, old_tok.vocab_size).unsqueeze(1)
    new, info = rt.init_embedding(old, old_tok, tok, seed=3)
    assert new.shape == (tok.vocab_size, 16) and new.dtype == old.dtype
    counts = info["counts"]
    assert sum(counts.values()) == tok.vocab_size and counts["special"] == 6
    assert all(counts[r] > 0 for r in rt.RULES), counts
    target = float(old[6:].norm(dim=1).median())
    assert math.isclose(info["target_norm"], target, rel_tol=1e-6)
    assert torch.equal(new[:6], old[:6])
    lowered = [(j, t.lower()) for j, t in enumerate(old_tok.tokens) if j >= 6]
    for i in range(6, tok.vocab_size):
        s, rule = tok.tokens[i], rt.RULES[int(info["rule_ids"][i])]
        j = old_tok.index.get(s)
        if rule == "copy":
            assert j is not None and j >= 6 and torch.equal(new[i], old[j])
            continue
        assert j is None or j < 6
        assert math.isclose(float(new[i].norm()), target, rel_tol=1e-4)
        ids = [k for k in old_tok.encode(s) if k >= 6]
        frag = s.strip().lower()
        hits = [j for j, t in lowered if len(frag) >= 3 and frag in t][:64]
        if rule == "old_tokenise":
            assert ids
            ref = old[ids].mean(0)
        elif rule == "fragment":
            assert not ids and hits
            ref = old[hits].mean(0)
        else:
            assert rule == "mean_noise" and not ids and not hits
            continue
        assert torch.allclose(new[i], ref * (target / float(ref.norm())), atol=1e-5)
    # the noise rows are seeded
    again, _ = rt.init_embedding(old, old_tok, tok, seed=3)
    assert torch.equal(new, again)


def test_choose_context():
    ctx, p99 = rt.choose_context([10] * 99 + [70], "auto")
    assert ctx == 32 and p99 == pytest.approx(10.6)  # one long row does not set the context
    ctx, p99 = rt.choose_context(list(range(1, 101)), "auto")
    assert ctx == 128 and 99 <= p99 <= 100  # smallest multiple of 32 >= p99
    assert rt.choose_context([64] * 50, "auto")[0] == 64  # exact multiple stays
    assert rt.choose_context([5] * 10, "auto")[0] == 32
    assert rt.choose_context([1000] * 10, "auto")[0] == 256  # capped
    assert rt.choose_context([1000] * 10, "auto", cap=192)[0] == 192
    assert rt.choose_context([1000] * 10, "160")[0] == 160


# ---------------------------------------------------------------------------
# 5b. tiny synthetic model: retokenise, save, reload (v1 and v2 schema)
# ---------------------------------------------------------------------------
def tiny_model(vocab: int):
    torch.manual_seed(0)
    cfg = ec.MiMoMixConfig(vocab_size=vocab, hidden_size=32, n_layers=3, n_heads=2, n_kv_heads=1, intermediate_size=64,
                           n_routed_experts=4, moe_intermediate_size=16, n_mtp_layers=1, sliding_window=16,
                           native_context=128, max_position_embeddings=128, rope_scaling="none")
    model = ec.ExpanseModel(cfg, fly_config=None, with_omni=False, expanse={"vocab_base": vocab})
    with torch.no_grad():
        model.fly_core.gate.fill_(0.01)
    return model.eval()


def _retokenized(schema: str):
    old_tok = text_utils.WordTokenizer.build(TEXTS, max_vocab=40, digit_tokens=True)
    model = tiny_model(old_tok.vocab_size)
    if schema == rt.V2_SCHEMA:
        import consolidation_v2 as cv2
        cv2.attach_consolidation_v2(model, cv2.V2Config(hidden_size=32, shared_dim=24, layers=(0, 2), latent_experts=4,
                                                        latent_rank=8, latent_top_k=2, memory_slots=4))
        with torch.no_grad():
            for b in model.consolidation_v2.blocks.values():
                b.out_gate.fill_(0.02)
    old_emb = model.embed_tokens.weight.detach().clone()
    new_tok, block = rt.retokenize(model, old_tok, ROWS, vocab=420, min_frequency=2, context="auto", seed=5)
    return model, old_tok, old_emb, new_tok, block


@pytest.mark.parametrize("schema", [ec.EXP_SCHEMA, rt.V2_SCHEMA])
def test_retokenize_tiny_model_save_reload(schema, tmp_path):
    model, old_tok, old_emb, new_tok, block = _retokenized(schema)
    V = new_tok.vocab_size
    # the model now speaks the new vocabulary
    assert isinstance(new_tok, bt.BPETokenizer) and 262 < V <= 420
    assert model.config.vocab_size == V and tuple(model.embed_tokens.weight.shape) == (V, 32)
    assert model.lm_head.weight is model.embed_tokens.weight and model.lm_head.out_features == V
    assert model.embed_tokens.num_embeddings == V and model.vocab_base == V
    assert torch.equal(model.embed_tokens.weight[:6], old_emb[:6])
    # context: multiple of 32 covering the 99th percentile of encoded row lengths
    lengths = [len(new_tok.encode_turn(r["user"], r["assistant"])[0]) for r in ROWS]
    ctx = block["context"]["chosen"]
    assert ctx % 32 == 0 and ctx - 32 < block["context"]["p99_new"] <= ctx <= 256
    assert model.config.max_position_embeddings == ctx == model.config.native_context
    assert block["lengths"]["new"]["max"] == max(lengths) and block["lengths"]["new"]["rows"] == len(ROWS)
    assert sum(block["rules"].values()) == V and block["vocab"]["size"] == V
    assert block["tokens_per_word"]["new"] > 0 and block["old_unk"]["rate"] > 0  # the old table (40 tokens) misses words
    json.dumps(ec.jsonable(block))  # receipt-safe

    # save with the base schema's saver, reload through the dispatching loader
    prompt = ROWS[1]["user"]
    ref = rt.prompt_logits(model, new_tok, prompt)
    path = tmp_path / "v3_init.pt"
    rt.save_like_base(schema, path, model, new_tok, {"note": "test"}, {"retokenize_v3": block})
    m2, tok2, pay2, schema2 = rt.load_base(path)
    assert schema2 == schema and pay2["schema"] == schema
    assert pay2["tokenizer"]["kind"] == "bpe" and isinstance(tok2, bt.BPETokenizer) and tok2.tokens == new_tok.tokens
    assert m2.config.vocab_size == V and m2.config.max_position_embeddings == ctx and m2.vocab_base == V
    assert pay2["expanse"]["retokenize_v3"]["rules"] == block["rules"]
    if schema == rt.V2_SCHEMA:
        assert getattr(m2, "consolidation_v2", None) is not None
    got = rt.prompt_logits(m2, tok2, prompt)
    assert got.shape[-1] == V and torch.equal(got, ref)
    dec = rt.decode_prompts(m2, tok2, [prompt], max_new=6)[0]
    assert 1 <= len(dec["ids"]) <= 6 and all(0 <= i < V for i in dec["ids"]) and isinstance(dec["text"], str)


@pytest.mark.parametrize("schema", [ec.EXP_SCHEMA, rt.V2_SCHEMA])
def test_loaders_still_return_word_tokenizer(schema, tmp_path):
    """v1/v2 checkpoints (tokenizer dict without ``kind``) load exactly as before the change."""
    import consolidation_v2 as cv2
    word = text_utils.WordTokenizer.build(TEXTS, max_vocab=60, digit_tokens=True)
    model = tiny_model(word.vocab_size)
    path = tmp_path / "word.pt"
    if schema == rt.V2_SCHEMA:
        cv2.attach_consolidation_v2(model, cv2.V2Config(hidden_size=32, shared_dim=24, layers=(1,), latent_experts=4,
                                                        latent_rank=8, latent_top_k=2, memory_slots=0))
        cv2.save_v2(path, model, word, {}, {})
        _, tok2, _ = cv2.load_v2(path)
    else:
        ec.save_expanse(path, model, word, {}, {})
        _, tok2, _ = ec.load_expanse(path)
    assert type(tok2) is text_utils.WordTokenizer and tok2.to_dict() == word.to_dict()


def test_training_rows_keeps_only_train_split(tmp_path, monkeypatch):
    """Source selection and the leak guard, against a stand-in trainer corpus (no real data)."""
    def row(u, a, **kw):
        return {"user": u, "assistant": a, **kw}

    heldout_dup = row("held q", "held a", source="code", split="heldout")
    corpus = {
        "replay": {"train": [row("r1", "a1", source="replay")], "dev": [row("r2", "a2", source="replay")], "heldout": []},
        "code": {"train": [row("c1", "a1", source="code", split="train"), dict(heldout_dup, split="train")],
                 "dev": [], "heldout": [heldout_dup]},
        "connectome": {"train": [row("n1", "a1", source="connectome", split="train")], "dev": [], "heldout": []},
    }

    def split_source(rows, frac, seed, dev_cap, train_cap=0):
        return [r for r in rows if not r["user"].startswith("dev")], [r for r in rows if r["user"].startswith("dev")]

    fake = types.SimpleNamespace(
        SOURCES=("replay", "fly", "code", "bio", "connectome"),
        load_corpus=lambda fly_core, **kw: corpus,
        split_source=split_source,
        _cap=lambda rows, cap, seed: rows,
        _h=lambda text, seed: (0 if text == ec.row_key("dev fresh", "x") else 1 << 60),
    )
    monkeypatch.setitem(sys.modules, "train_expanse", fake)
    v3 = tmp_path / "v3"
    v3.mkdir()
    fresh = [row("f1", "x", split="train"), row("f2", "x", split="heldout"), row("dev fresh", "x", split="train")]
    (v3 / "fresh_omni.jsonl").write_text("\n".join(json.dumps(r) for r in fresh), encoding="utf-8")
    monkeypatch.setattr(rt, "V3_DATA", v3)
    rows, info = rt.training_rows(None, seed=1, fly_rows=0, dev_frac=0.05, dev_cap=10, max_rows_per_source=0)
    users = [r["user"] for r in rows]
    assert users == ["r1", "c1", "n1", "f1"], users  # trainer's source order, then fresh
    assert info["leak_guard_dropped"] == 1  # the train copy of a held-out row
    assert info["per_source"] == {"replay": 1, "code": 1, "connectome": 1, "fresh": 1}
    assert info["fallbacks"] and info["fresh_files"] == ["fresh_omni.jsonl"]
