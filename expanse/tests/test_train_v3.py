"""Tests for the v3 additions to train_expanse.py and compare_models.py (V3_DESIGN.md B.3-B.4).

No checkpoint is loaded (the CLI ``--smoke`` runs cover the real v1 / v2 /
v3-init models); these pin the pieces the smoke runs cannot isolate:

1. checkpoint headers are read without loading tensors (``mmap``) and the
   schema picks the loader / saver;
2. tokenizers: a word dict round-trips through ``tokenizer_from_dict``; a BPE
   (when ``bpe_tokenizer`` is present) is tagged, fingerprinted differently and
   switches ``kd_arch`` off; a word vocabulary is ``kd_arch``-compatible only if
   it extends Archimedes'; word fingerprints are unchanged from pre-v3;
3. parameter groups: v2 native blocks and donor slots are graft, the tied
   embedding is trunk, the omni encoders are frozen;
4. ``--freeze_trunk_steps``: dropped grads leave the held trunk tensors
   bit-identical under AdamW with weight decay while the embedding and graft
   parameters move;
5. compare_models scores each model at ``max(128, its context)``.

Run: ``python -m pytest -q expanse/tests/test_train_v3.py`` (seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))

import train_expanse as te  # noqa: E402
from expanse_core import PATHS, text_utils  # noqa: E402

torch.set_num_threads(2)

try:
    import bpe_tokenizer  # noqa: F401  (agent A's module; absent before v3)
    HAVE_BPE = True
except ImportError:  # pragma: no cover
    HAVE_BPE = False


# ---------------------------------------------------------------------------
# checkpoints + tokenizers
# ---------------------------------------------------------------------------
def test_peek_reads_the_header_only(tmp_path):
    path = tmp_path / "ck.pt"
    tok = text_utils.WordTokenizer(["alpha", " beta"], digit_tokens=True)
    torch.save({"schema": te.V2_SCHEMA, "tokenizer": tok.to_dict(), "state_dict": {"w": torch.arange(6.0)},
                "expanse": {"training": {"corpus_args": {"seed": 5}}}}, path)
    assert te.checkpoint_schema(path) == te.V2_SCHEMA
    head = te.peek_checkpoint(path, "tokenizer", "expanse", "missing")
    assert head["missing"] is None and head["expanse"]["training"]["corpus_args"]["seed"] == 5
    back = te.tokenizer_from_dict(head["tokenizer"])
    assert te.is_word_tokenizer(back) and back.tokens == tok.tokens and te.tokenizer_kind(back) == "word"
    assert te.tokenizer_tag(back) == ""


def test_word_fingerprint_is_unchanged_and_tokenizer_tag_splits_it():
    rows = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(5)]
    import hashlib
    h = hashlib.sha1(b"128|100")  # the pre-v3 recipe, byte for byte
    for tag, rs in (("t", rows[:3]), ("d", rows[3:])):
        for r in rs:
            h.update(tag.encode())
            h.update(te.ec.row_key(r["user"], r["assistant"]).encode())
    assert te.data_fingerprint(rows[:3], rows[3:], 128, 100) == h.hexdigest()
    assert te.data_fingerprint(rows[:3], rows[3:], 128, 100, "") == h.hexdigest()
    assert te.data_fingerprint(rows[:3], rows[3:], 128, 100, "bpe:abc") != h.hexdigest()


class FakeBPE:
    """Duck-typed non-word tokenizer (stands in when bpe_tokenizer is absent)."""

    vocab_size = 300

    def to_dict(self):
        return {"kind": "bpe", "json": "{}", "digit_tokens": True}


def test_non_word_tokenizer_turns_kd_arch_off():
    ok, why = te.arch_vocab_compatible(FakeBPE(), PATHS["arch_final"])
    assert not ok and "not Archimedes' word tokenizer" in why
    assert te.tokenizer_kind(FakeBPE()) == "bpe" and te.tokenizer_tag(FakeBPE()).startswith("bpe:")


@pytest.mark.skipif(not HAVE_BPE, reason="bpe_tokenizer (agent A) not present")
def test_real_bpe_dispatch_tag_and_kd_gate():
    import bpe_tokenizer as bt

    texts = [f"The force on a {i} kg mass is {i * 3} newtons, total {i * 3}" for i in range(200)]
    tok = bt.train_bpe(texts, vocab_size=400, min_frequency=2)
    back = te.tokenizer_from_dict(tok.to_dict())
    assert not te.is_word_tokenizer(back) and te.tokenizer_kind(back) == "bpe"
    assert back.encode(texts[3]) == tok.encode(texts[3])
    assert te.tokenizer_tag(back) == te.tokenizer_tag(tok) and te.tokenizer_tag(back).startswith("bpe:")
    other = bt.train_bpe(texts[::-1][:120] + ["zebra zebra quokka quokka"] * 5, vocab_size=400, min_frequency=2)
    assert te.tokenizer_tag(other) != te.tokenizer_tag(tok)
    assert not te.arch_vocab_compatible(back, PATHS["arch_final"])[0]


@pytest.mark.skipif(not PATHS["arch_final"].exists(), reason="Archimedes final checkpoint not present")
def test_word_vocab_must_extend_archimedes():
    arch_tok = text_utils.WordTokenizer.from_dict(te.peek_checkpoint(PATHS["arch_final"], "tokenizer")["tokenizer"])
    extended = text_utils.WordTokenizer.from_dict({"tokens": list(arch_tok.tokens) + [" zebra", "quokka"],
                                                   "digit_tokens": arch_tok.digit_tokens})
    assert te.arch_vocab_compatible(extended, PATHS["arch_final"])[0]
    swapped = list(arch_tok.tokens)
    swapped[10], swapped[11] = swapped[11], swapped[10]
    retrained = text_utils.WordTokenizer.from_dict({"tokens": swapped, "digit_tokens": True})
    ok, why = te.arch_vocab_compatible(retrained, PATHS["arch_final"])
    assert not ok and "does not extend" in why


# ---------------------------------------------------------------------------
# parameter groups + freeze
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.gate = nn.Linear(d, n, bias=False)
        self.experts = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n)])


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d, bias=False)
        self.mlp = _MLP(d, 3)


class TinyExpanse(nn.Module):
    """Parameter names laid out like ExpanseModel (+ an attached v2 stack)."""

    def __init__(self, v=12, d=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(v, d)
        self.lm_head = nn.Linear(d, v, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # tied, as in MiMoMix
        self.layers = nn.ModuleList([_Layer(d), _Layer(d)])
        self.norm = nn.LayerNorm(d)
        self.fly_core = nn.Linear(d, d)
        self.omni_core = nn.ModuleDict({"to_trunk": nn.Linear(d, d), "collective": nn.Linear(d, d)})
        self.omni7 = nn.ModuleDict({"net": nn.Linear(d, d), "to_trunk": nn.Linear(d, d)})
        self.cns_full = nn.Linear(d, d)
        self.consolidation_v2 = nn.ModuleDict({"blocks": nn.ModuleDict({"1": nn.Linear(d, d)})})

    def donor_slots(self):
        return {1: [2]}

    def forward(self, x):
        h = self.embed_tokens(x)
        for li, layer in enumerate(self.layers):
            h = h + layer.attn(h) + sum(e(h) for e in layer.mlp.experts) + layer.mlp.gate(h).sum(-1, keepdim=True)
            if li == 1:
                h = h + self.consolidation_v2["blocks"]["1"](h)
        h = self.norm(h + self.fly_core(h) + self.omni_core["to_trunk"](h) + self.omni7["to_trunk"](h) + self.cns_full(h))
        return self.lm_head(h)


def test_split_params_groups():
    m = TinyExpanse()
    frozen, graft, trunk = te.split_params(m)
    names = {id(p): n for n, p in m.named_parameters()}
    g = {names[id(p)] for p in graft}
    t = {names[id(p)] for p in trunk}
    assert set(frozen) == {"omni_core.collective.weight", "omni_core.collective.bias", "omni7.net.weight", "omni7.net.bias"}
    assert not any(p.requires_grad for n, p in m.named_parameters() if n in frozen)
    assert {"consolidation_v2.blocks.1.weight", "consolidation_v2.blocks.1.bias", "layers.1.mlp.experts.2.weight",
            "fly_core.weight", "omni_core.to_trunk.weight", "omni7.to_trunk.weight", "cns_full.weight"} <= g
    assert "embed_tokens.weight" in t and "layers.0.mlp.experts.2.weight" in t and "layers.1.mlp.experts.1.weight" in t
    assert not g & t and len(g) + len(t) + len(frozen) == len(list(m.named_parameters()))
    held = te.held_while_frozen(m, trunk)
    assert all(p is not m.embed_tokens.weight for p in held) and m.lm_head.weight is m.embed_tokens.weight
    assert {names[id(p)] for p in held} == t - {"embed_tokens.weight"}


def test_freeze_trunk_steps_hold_the_trunk_exactly():
    torch.manual_seed(0)
    m = TinyExpanse()
    _, graft, trunk = te.split_params(m)
    held = te.held_while_frozen(m, trunk)
    opt = torch.optim.AdamW([{"params": trunk, "lr": 1e-2, "weight_decay": 0.01},
                             {"params": graft, "lr": 1e-2, "weight_decay": 0.0}], betas=(0.9, 0.95))
    before = {id(p): p.detach().clone() for p in trunk + graft}
    x = torch.randint(0, 12, (3, 5))
    for step in range(3):
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, 12), x.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        te.drop_grads(held)
        torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]], 1.0)
        opt.step()
    for p in held:  # no step, no decoupled weight decay, no optimizer state
        assert torch.equal(p.detach(), before[id(p)]) and p not in opt.state
    assert not torch.equal(m.embed_tokens.weight.detach(), before[id(m.embed_tokens.weight)])
    moved = [p for p in graft if not torch.equal(p.detach(), before[id(p)])]
    assert len(moved) >= len(graft) - 1  # every graft tensor on the loss path moves


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------
def test_compare_scores_at_the_model_context():
    import compare_models as cm

    def fake(ctx):
        return SimpleNamespace(config=SimpleNamespace(max_position_embeddings=ctx))

    assert cm.model_seq(fake(128), 128) == 128
    assert cm.model_seq(fake(96), 128) == 128
    assert cm.model_seq(fake(224), 128) == 224


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
