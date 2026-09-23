"""Tests for src/omni_v7_branch.py against the real Omni Collective v7 Frontier.

* the branch reproduces ``OmniCollectiveEngineV7._run_prompt`` intent/domain
  (and vision/response) logits for three prompts, one at a time (bit-level) and
  batched (float tolerance: a batched GRU may block its GEMMs differently);
* the captured ``fused`` state is the tensor every v7 head reads
  (``intent_head(fused)`` == the engine's intent logits);
* zero gate => ``forward`` returns the trunk hidden unchanged (torch.equal);
* the v7 net stays frozen and in eval under ``train()``; the bridge trains;
* the branch round-trips through a strict ``state_dict`` load (self-contained
  Expanse checkpoints), and a meta without ``num_responses`` still builds.

Run: ``python -m pytest -q tests/test_omni_v7_branch.py`` (or as a script).
Needs ~1.5 GB RAM (engine + branch each hold the 310 MB net).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import omni_v7_branch as ob  # noqa: E402

PROMPTS = [
    "Write a Python function that returns the second largest number in a list.",
    "Give a grounded summary of what hydrophobic collapse does during protein folding.",
    "Plan a three day trip to Kyoto on a small budget and compare train versus bus.",
]
WEIGHTS = ob.OMNI_V7_DIR / ob.WEIGHTS_FILE
META = ob.OMNI_V7_DIR / ob.META_FILE

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="omni v7 weights not present")


@pytest.fixture(scope="module")
def pair():
    torch.set_num_threads(2)
    from omni_collective_v7_model import OmniCollectiveEngineV7

    engine = OmniCollectiveEngineV7(weights_path=WEIGHTS, meta_path=META, device=torch.device("cpu"))
    branch = ob.OmniV7Branch(320, ob.load_v7_meta(META))
    receipt = branch.load_v7(WEIGHTS)
    branch.eval()
    return engine, branch, receipt


def _engine_outputs(engine, prompt):
    size = engine.image_size
    img = torch.zeros((1, 3, size, size))
    has = torch.zeros((1,))
    return engine._run_prompt(prompt, img, has)


def test_receipt_and_dims(pair):
    _, branch, receipt = pair
    assert receipt["tensors"] == 253
    assert receipt["num_responses"] == ob.V7_NUM_RESPONSES
    assert branch.state_dim == 988 and branch.n_intents == 15 and branch.n_domains == 13
    print("load_v7 receipt:", {k: receipt[k] for k in ("tensors", "params", "num_responses", "state_dim")})


def test_reproduces_engine_one_by_one(pair):
    engine, branch, _ = pair
    worst = {}
    for p in PROMPTS:
        ref = _engine_outputs(engine, p)
        state, logits = branch.encode(branch.featurize([p]), return_logits=True)
        for name in ("intent", "domain", "vision", "response"):
            d = float((logits[name][0] - ref[name]).abs().max())
            worst[name] = max(worst.get(name, 0.0), d)
        # the state's probability blocks are the softmaxed engine logits
        assert torch.allclose(state[0, 960:975], torch.softmax(ref["intent"], 0), atol=1e-6)
        assert torch.allclose(state[0, 975:], torch.softmax(ref["domain"], 0), atol=1e-6)
        # fused is what the heads read
        assert torch.allclose(branch.net.intent_head(state[:, :960])[0], ref["intent"], atol=1e-5)
        print(f"  {p[:50]!r}: intent={engine.intent_labels[int(ref['intent'].argmax())]}, "
              f"domain={engine.domain_labels[int(ref['domain'].argmax())]}")
    print("B=1 max |logit diff| vs engine._run_prompt:", worst)
    assert worst["intent"] <= 1e-6 and worst["domain"] <= 1e-6, worst


def test_reproduces_engine_batched(pair):
    engine, branch, _ = pair
    t0 = time.time()
    state, logits = branch.encode(branch.featurize(PROMPTS), return_logits=True)
    dt = time.time() - t0
    worst = 0.0
    for i, p in enumerate(PROMPTS):
        ref = _engine_outputs(engine, p)
        for name in ("intent", "domain"):
            worst = max(worst, float((logits[name][i] - ref[name]).abs().max()))
    print(f"batched (B=3) max |intent/domain logit diff| = {worst:.3g}; encode {dt:.2f}s")
    assert state.shape == (3, 988)
    assert worst < 1e-4


def test_real_image_path_matches_engine(pair):
    """A non-blank image bypasses the blank-image shortcut and still matches."""
    engine, branch, _ = pair
    torch.manual_seed(0)
    img = torch.rand(1, 3, 128, 128)
    feats = branch.featurize([PROMPTS[0]])
    feats["image_tensor"], feats["has_image"] = img, torch.ones(1)
    _, logits = branch.encode(feats, return_logits=True)
    ref = engine._run_prompt(PROMPTS[0], img, torch.ones(1))
    assert float((logits["intent"][0] - ref["intent"]).abs().max()) <= 1e-6


def test_zero_gate_exact_and_causal_constant(pair):
    _, branch, _ = pair
    state = branch.encode(branch.featurize(PROMPTS[:2]))
    hidden = torch.randn(2, 7, 320)
    out, info = branch(hidden, state)
    assert torch.equal(out, hidden), "zero gate must return hidden exactly"
    assert float(info["write_rms"]) == 0.0
    with torch.no_grad():
        branch.gate.normal_(0, 0.1)
    out, _ = branch(hidden, state)
    delta = out - hidden
    # prompt-constant write: identical at every position
    assert torch.allclose(delta, delta[:, :1].expand_as(delta), atol=1e-6)
    with torch.no_grad():
        branch.gate.zero_()


def test_frozen_and_trainable(pair):
    _, branch, _ = pair
    branch.train()
    assert not branch.net.training and all(not m.training for m in branch.net.modules())
    assert branch.bridge_norm.training
    assert all(not p.requires_grad for p in branch.net.parameters())
    state = branch.encode(branch.featurize(PROMPTS[:1]))
    assert not state.requires_grad
    with torch.no_grad():
        branch.gate.fill_(0.01)
    hidden = torch.randn(1, 3, 320, requires_grad=True)
    out, _ = branch(hidden, state)
    il, dl = branch.aux_logits(out[:, -1])
    assert il.shape == (1, 15) and dl.shape == (1, 13)
    (out.sum() + il.sum() + dl.sum()).backward()
    assert branch.to_trunk.weight.grad is not None and branch.gate.grad is not None
    assert branch.intent_head.weight.grad is not None
    assert all(p.grad is None for p in branch.net.parameters())
    branch.zero_grad(set_to_none=True)
    with torch.no_grad():
        branch.gate.zero_()
    branch.eval()


def test_state_dict_roundtrip_and_meta_fallback(pair):
    _, branch, _ = pair
    full = json.loads(META.read_text(encoding="utf-8"))
    small = {k: v for k, v in full.items() if k != "response_bank"}  # no num_responses key
    clone = ob.OmniV7Branch(320, small)
    clone.load_state_dict(branch.state_dict(), strict=True)
    clone.eval()
    a = branch.encode(branch.featurize(PROMPTS[1:2]))
    b = clone.encode(clone.featurize(PROMPTS[1:2]))
    assert torch.equal(a, b)
    assert json.dumps(clone.meta)  # meta stays JSON-serialisable for the checkpoint receipt


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
