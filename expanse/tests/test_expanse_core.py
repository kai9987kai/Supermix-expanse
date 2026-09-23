"""Tests for src/expanse_core.py on the real Archimedes final checkpoint.

The four properties the Expanse build and trainer rely on:

1. ``grow_moe_slots`` + strict load reproduces Archimedes' logits **bitwise**
   (16 dead slots appended to every MoE layer; fly at its trained gate).
   ``ExpanseSparseMoE`` routes over the active width, which is what makes
   this exact -- plain ``SparseMoEFeedForward`` on the grown config flips
   exact top-k ties (reported, not asserted).
2. ``fly_forward_causal`` is causal with the fly gate at its trained value:
   perturbing the last real token leaves every earlier logit unchanged,
   while Archimedes' mean-pooled fly leaks.
3. KV-cached decode equals the full forward (max |dlogit| < 1e-4) with every
   Expanse graft live: fly/omni/omni7/cns_full gates small random, donor
   slots woken with random experts.
4. ``save_expanse`` / ``load_expanse`` round-trip is exact (state, spec,
   tokenizer, logits).

The two new grafts use the real ``FullConnectomeCore`` (on a tiny random
graph) and the real ``OmniV7Branch`` class (random v7 net, no weights load)
when importable; otherwise small local stand-ins with the same contract.

Run: ``python -m pytest -q -s tests/test_expanse_core.py`` (or as a script).
Needs ~2 GB RAM, ~1-2 min with 2 threads.
"""

from __future__ import annotations

import copy
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import expanse_core as ec  # noqa: E402

ARCH = ec.PATHS["arch_final"]
pytestmark = pytest.mark.skipif(not ARCH.exists(), reason="Archimedes final checkpoint not present")


# ---------------------------------------------------------------------------
# stand-ins (only used when the real modules are missing)
# ---------------------------------------------------------------------------
class StandInCNS(nn.Module):
    """Per-token stand-in for FullConnectomeCore: same constructor/forward contract."""

    def __init__(self, hidden_size, graph, steps=4):
        super().__init__()
        n = int(graph["n"])
        n_in = len(graph["in_idx"]) if "in_idx" in graph else int(graph["n_in"])
        n_out = len(graph["out_idx"]) if "out_idx" in graph else int(graph["n_out"])
        self.register_buffer("w", torch.zeros(n, n))
        if "post" in graph:
            self.w[torch.as_tensor(graph["post"]).long(), torch.as_tensor(graph["pre"]).long()] = torch.as_tensor(graph["value"])
        self.in_proj = nn.Linear(hidden_size, n_in, bias=False)
        self.out_proj = nn.Linear(n_out, hidden_size, bias=False)
        self.register_buffer("in_idx", torch.as_tensor(np.asarray(graph.get("in_idx", np.arange(n_in)))).long())
        self.register_buffer("out_idx", torch.as_tensor(np.asarray(graph.get("out_idx", np.arange(n_out)))).long())
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.steps = steps

    def forward(self, hidden, token_mask=None):
        b, t, h = hidden.shape
        flat = hidden.reshape(b * t, h)
        sel = torch.arange(b * t) if token_mask is None else token_mask.reshape(-1).nonzero().squeeze(1)
        if sel.numel() == 0:
            return hidden.clone(), {"M": 0}
        hs = flat.index_select(0, sel)
        u = hs.new_zeros(self.w.shape[0], hs.shape[0]).index_copy(0, self.in_idx, self.in_proj(hs).t())
        r = torch.zeros_like(u)
        for _ in range(self.steps):
            r = 0.5 * r + 0.5 * torch.relu(self.w @ r + u)
        e = r.index_select(0, self.out_idx).t()
        e = e / torch.sqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)
        return flat.index_copy(0, sel, hs + self.gate * self.out_proj(e)).view(b, t, h), {"M": int(sel.numel())}


class StandInOmni7(nn.Module):
    def __init__(self, hidden_size, meta):
        super().__init__()
        self.meta = dict(meta)
        self.state_dim = int(meta.get("state_dim", 988))
        self.net = nn.Linear(4, 4)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.bridge_norm = nn.LayerNorm(self.state_dim)
        self.to_trunk = nn.Linear(self.state_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.intent_head = nn.Linear(hidden_size, 15)
        self.domain_head = nn.Linear(hidden_size, 13)

    def featurize(self, prompts):
        return {"n": len(prompts)}

    @torch.no_grad()
    def encode(self, feats):
        g = torch.Generator().manual_seed(feats["n"])
        return torch.rand(feats["n"], self.state_dim, generator=g)

    def forward(self, hidden, state):
        w = self.gate * self.to_trunk(self.bridge_norm(state))
        return hidden + w.unsqueeze(1), {}

    def aux_logits(self, h):
        return self.intent_head(h), self.domain_head(h)


def _resolve_graft_classes():
    used = {}
    try:
        ec.graft_class("cns_full")
        used["cns_full"] = "real FullConnectomeCore"
    except ImportError:
        ec.GRAFT_CLASSES["cns_full"] = StandInCNS
        used["cns_full"] = "stand-in"
    try:
        ec.graft_class("omni7")
        used["omni7"] = "real OmniV7Branch"
    except ImportError:
        ec.GRAFT_CLASSES["omni7"] = StandInOmni7
        used["omni7"] = "stand-in"
    return used


def _omni7_meta():
    if ec.GRAFT_CLASSES.get("omni7") is StandInOmni7:
        return {"state_dim": 988}
    import omni_v7_branch as ob
    return ob.load_v7_meta()


def tiny_graph(n=64, n_in=12, n_out=10, nnz=420, seed=0):
    """A random graph dict with the documented build_full_graph keys."""
    rng = np.random.default_rng(seed)
    keys = rng.choice(n * n, size=nnz, replace=False)
    post, pre = (keys // n).astype(np.int64), (keys % n).astype(np.int64)
    order = np.lexsort((pre, post))
    post, pre = post[order], pre[order]
    sign = np.where(rng.random(n) < 0.7, 1, -1).astype(np.int8)
    w = rng.integers(1, 50, size=nnz).astype(np.float64)
    tot = np.bincount(post, weights=w, minlength=n)
    value = (w / tot[post] * sign[pre] * 0.9).astype(np.float32)
    perm = rng.permutation(n)
    names = np.array([f"T{i}" for i in range(n)], dtype=object)
    return {"n": n, "post": post.astype(np.int32), "pre": pre.astype(np.int32), "value": value, "sign": sign,
            "in_idx": np.sort(perm[:n_in]).astype(np.int64), "out_idx": np.sort(perm[n_in:n_in + n_out]).astype(np.int64),
            "type_names": names, "superclass": names.copy(), "nt": names.copy(),
            "receipt": {"n_types": n, "n_edges": nnz, "note": "tiny random test graph"}}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def arch():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    used = _resolve_graft_classes()
    print("\n[graft classes]", used)
    t0 = time.time()
    A, tok, payload = ec.load_archimedes(ARCH)
    A.eval()
    rows = []
    for name in ("omni", "code", "math"):
        rows += [json.loads(l) for l in open(ec.PATHS["corpus"] / f"{name}.jsonl", encoding="utf-8")]
    random.Random(1234).shuffle(rows)
    rows = rows[:6]
    enc = [tok.encode_turn(r["user"], r["assistant"])[0] for r in rows]
    L = max(map(len, enc))
    x = torch.tensor([ids + [ec.text_utils.PAD] * (L - len(ids)) for ids in enc])
    feats = ec.arch.OmniCore.featurize([r["user"] for r in rows])
    print(f"[arch] loaded in {time.time() - t0:.1f}s, batch {tuple(x.shape)}")
    return {"A": A, "tok": tok, "payload": payload, "rows": rows, "x": x, "lens": [len(e) for e in enc], "feats": feats}


@pytest.fixture(scope="module")
def grown(arch):
    """ExpanseModel with only the Archimedes grafts, strict-loaded from the grown state."""
    pay = arch["payload"]
    sd, cfg = ec.grow_moe_slots(pay["state_dict"], ec.MiMoMixConfig(**pay["config"]), 16)
    E = ec.ExpanseModel(cfg, fly_config=pay["archimedes"]["fly_config"], with_omni=True,
                        expanse={"graph": None, "omni7_meta": None, "fly_causal": True})
    E.load_state_dict(sd, strict=True)
    E.eval()
    return {"E": E, "sd": sd, "cfg": cfg}


def _logits(model, x, feats, **kw):
    with torch.no_grad():
        return model(x, omni_features=feats, return_mtp=False, **kw).logits


# ---------------------------------------------------------------------------
# 1. slot growth
# ---------------------------------------------------------------------------
def test_grow_moe_slots_padding(arch):
    pay = arch["payload"]
    cfg0 = ec.MiMoMixConfig(**pay["config"])
    sd0 = pay["state_dict"]
    sd, cfg = ec.grow_moe_slots(sd0, cfg0, 16)
    assert cfg.moe_spare_experts == cfg0.moe_spare_experts + 16
    assert ec.moe_layers_in_state_dict(sd) == [1, 2, 3, 4]
    for li in (1, 2, 3, 4):
        p = f"layers.{li}.mlp."
        assert sd[p + "gate.weight"].shape == (88, 320)
        assert torch.equal(sd[p + "gate.weight"][:72], sd0[p + "gate.weight"])
        assert sd[p + "gate.weight"][72:].abs().sum() == 0
        alive0 = sd0[p + "expert_alive"].bool()
        fill = float(sd0[p + "expert_bias"][alive0].min()) - 10.0
        assert torch.allclose(sd[p + "expert_bias"][72:], torch.full((16,), fill))
        assert int(sd[p + "expert_alive"][72:].sum()) == 0 and sd[p + "expert_alive"].dtype == torch.uint8
        for s in (72, 87):
            for name in ("gate_proj", "up_proj", "down_proj"):
                assert sd[f"{p}experts.{s}.{name}.weight"].abs().sum() == 0
    assert "layers.1.mlp.experts.72.gate_proj.weight" not in sd0  # input untouched
    with pytest.raises(ValueError):
        ec.grow_moe_slots(sd0, cfg0, 16, layers=(1, 2))


def test_grow_reproduces_archimedes_exactly(arch, grown):
    A, E, x, feats = arch["A"], grown["E"], arch["x"], arch["feats"]
    assert float(A.fly_core.gate.detach().abs().mean()) > 0  # the trained fly gate is live
    ref = _logits(A, x, feats)
    ref_nofly = _logits(A, x, feats, use_fly=False)
    E.fly_causal = False  # Archimedes' own (mean-pooled) fly
    try:
        got = _logits(E, x, feats)
    finally:
        E.fly_causal = True
    got_nofly = _logits(E, x, feats, use_fly=False)
    print(f"\n[grow] max|dlogit| fly legacy {float((got - ref).abs().max()):.3e}  fly off {float((got_nofly - ref_nofly).abs().max()):.3e}")
    assert torch.equal(got, ref), "grown Expanse (legacy fly) != Archimedes"
    assert torch.equal(got_nofly, ref_nofly), "grown Expanse (fly off) != Archimedes (fly off)"
    # what the causal fix changes (expected non-zero, small)
    causal = _logits(E, x, feats)
    print(f"[grow] causal fly vs Archimedes: max|dlogit| {float((causal - ref).abs().max()):.3e}")
    # for the record: the base MoE class on the grown config is NOT bitwise (top-k ties)
    B = ec.ArchimedesModel(grown["cfg"], fly_config=arch["payload"]["archimedes"]["fly_config"])
    B.load_state_dict(grown["sd"], strict=True)
    B.eval()
    base = _logits(B, x, feats, use_fly=False)
    print(f"[grow] plain SparseMoEFeedForward on the grown config: max|dlogit| {float((base - ref_nofly).abs().max()):.3e} (tie flips)")
    del B


# ---------------------------------------------------------------------------
# 2. causal fly
# ---------------------------------------------------------------------------
def test_fly_obs_broadcast_matches_flycore(arch):
    fly = arch["A"].fly_core
    torch.manual_seed(3)
    h = torch.randn(2, 5, 320)
    obs = torch.rand(2, 14)
    with torch.no_grad():
        ref, _ = fly(h, obs)
        got, info = ec.fly_forward_causal(fly, h, obs)
    assert torch.equal(ref, got)
    assert info["obs"].shape == (2, 5, 14)


def test_fly_forward_causal_module_is_bitwise_causal(arch):
    """At the module: changing h at the last position leaves earlier writes bit-identical."""
    fly = arch["A"].fly_core
    torch.manual_seed(4)
    h = torch.randn(3, 9, 320)
    h2 = h.clone()
    h2[:, -1] += torch.randn(3, 320)
    with torch.no_grad():
        a, info = ec.fly_forward_causal(fly, h)
        b, _ = ec.fly_forward_causal(fly, h2)
        la, _ = fly(h)   # Archimedes: mean-pooled sense
        lb, _ = fly(h2)
    assert torch.equal(a[:, :-1], b[:, :-1])
    assert not torch.equal(a[:, -1], b[:, -1])
    assert float((la[:, :-1] - lb[:, :-1]).abs().max()) > 0  # the leak being fixed
    assert info["obs"].shape == (3, 9, 14) and info["probs"].shape == (3, 9, 4)


def test_fly_forward_causal_is_causal(arch, grown):
    """In the model, fly gate at its trained value, one row at a time.

    The trunk itself is causal only up to float noise: changing the last
    token changes which tokens each routed expert batches together, and GEMM
    rounding depends on the batch, so earlier logits move by a few 1e-6 even
    with the fly off (measured per row below as the noise floor). The causal
    fly must stay at that floor; Archimedes' mean-pooled fly sits 10-100x
    above it.
    """
    A, E, tok = arch["A"], grown["E"], arch["tok"]
    assert E.fly_causal and float(E.fly_core.gate.detach().abs().mean()) > 0
    worst = {"expanse": 0.0, "floor": 0.0, "archimedes": 0.0}
    for r in arch["rows"]:
        ids, _ = tok.encode_turn(r["user"], r["assistant"])
        x = torch.tensor([ids])
        x2 = x.clone()
        x2[0, -1] = 100 if int(x[0, -1]) != 100 else 101
        f = ec.arch.OmniCore.featurize([r["user"]])
        n = len(ids)
        for name, model, kw in (("expanse", E, {}), ("floor", E, {"use_fly": False}), ("archimedes", A, {})):
            d = float((_logits(model, x, f, **kw)[0, : n - 1] - _logits(model, x2, f, **kw)[0, : n - 1]).abs().max())
            worst[name] = max(worst[name], d)
    print(f"\n[causal] earlier-position max|dlogit| after changing the last token: {worst}")
    assert worst["expanse"] < 1e-5 and worst["expanse"] <= 4 * max(worst["floor"], 1e-6), "causal fly leaks the future"
    assert worst["archimedes"] > 2 * worst["expanse"], "expected Archimedes' mean-pooled fly to leak"
    # the per-position sense exposes (B, T, 14) observations for the aux loss
    with torch.no_grad():
        E(arch["x"], omni_features=arch["feats"], return_mtp=False)
    assert E.last_graft_info["fly"]["obs"].shape == (arch["x"].shape[0], arch["x"].shape[1], 14)


# ---------------------------------------------------------------------------
# 3 + 4. full Expanse: KV decode and checkpoint round trip
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def full(arch):
    t0 = time.time()
    E, info = ec.expanse_from_archimedes(arch["payload"], n_new=16, graph=tiny_graph(), cns_steps=4,
                                         omni7_meta=_omni7_meta(), fly_causal=True)
    E.eval()
    print(f"\n[full] built in {time.time() - t0:.1f}s: {info}")
    return E


def _install_random_donors(E, seed=7):
    """Fill slots 72..87 of layers 1-2 like donor_graft.install_experts would (dormant)."""
    g = torch.Generator().manual_seed(seed)
    receipts = {"qwen": {"layers": {}}, "biomedlm": {"layers": {}}}
    with torch.no_grad():
        for li in (1, 2):
            mlp = E.layers[li].mlp
            alive = mlp.expert_alive.bool()
            dormant = float(mlp.expert_bias[alive].min()) - 1.5
            target = float(mlp.expert_bias[alive].median())
            row_norm = float(mlp.gate.weight[alive].norm(dim=1).median())
            for name, slots in (("qwen", list(range(72, 80))), ("biomedlm", list(range(80, 88)))):
                for s in slots:
                    for p in mlp.experts[s].parameters():
                        p.copy_(torch.randn(p.shape, generator=g) * 0.05)
                    r = torch.randn(320, generator=g)
                    mlp.gate.weight[s].copy_(r / r.norm() * row_norm * 3.0)
                    mlp.expert_bias[s] = dormant
                    mlp.expert_alive[s] = 0
                receipts[name]["layers"][str(li)] = {"slots": slots, "dormant_bias": dormant, "target_bias": target,
                                                     "placed": [{"slot": s} for s in slots]}
    E.donor_receipts = receipts
    return receipts


def test_zero_new_gates_are_identity(arch, full):
    """cns_full + omni7 at zero gates, donors dormant: exactly Archimedes (legacy fly)."""
    E = full
    _install_random_donors(E)
    E.fly_causal = False
    try:
        state = E.omni7_state_for([r["user"] for r in arch["rows"]])
        got = _logits(E, arch["x"], arch["feats"], omni7_state=state)
    finally:
        E.fly_causal = True
    ref = _logits(arch["A"], arch["x"], arch["feats"])
    print(f"\n[identity] max|dlogit| {float((got - ref).abs().max()):.3e}")
    assert torch.equal(got, ref)


def _set_all_gates(E, seed=11, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in (E.fly_core, E.omni_core, E.omni7, E.cns_full):
            mod.gate.add_(torch.randn(mod.gate.shape, generator=g) * scale)


def test_kv_decode_matches_full_forward(arch, full):
    E = full
    _install_random_donors(E)
    E.wake_donor_experts(1.0)
    _set_all_gates(E)
    rep = E.gate_report()
    for k in ("fly_gate_abs_mean", "omni_gate_abs_mean", "omni7_gate_abs_mean", "cns_full_gate_abs_mean"):
        assert rep[k] > 0, k
    assert rep["donor_qwen_alive"] == 16 and rep["donor_biomedlm_alive"] == 16
    tok = arch["tok"]
    worst = 0.0
    for r in arch["rows"][:3]:
        ids, plen = tok.encode_turn(r["user"], r["assistant"])
        ids = ids[: plen + 24]
        x = torch.tensor([ids])
        feats = ec.arch.OmniCore.featurize([r["user"]])
        state = E.omni7_state_for([r["user"]])
        kw = {"omni_features": feats, "omni7_state": state}
        with torch.no_grad():
            full_logits = E(x, return_mtp=False, **kw).logits[0]
            out = E(x[:, :plen], use_cache=True, return_mtp=False, past_length=0, **kw)
            steps = [out.logits[0]]
            past, pos = out.past_key_values, plen
            for t in range(plen, len(ids)):
                out = E(x[:, t:t + 1], past_key_values=past, use_cache=True, return_mtp=False, past_length=pos, **kw)
                past, pos = out.past_key_values, pos + 1
                steps.append(out.logits[0])
            inc = torch.cat(steps, 0)
            # the donor experts and the connectome core really ran
            assert E.last_graft_info["cns_full"]["M"] == 1
        d = float((inc - full_logits).abs().max())
        worst = max(worst, d)
        # greedy decode from the prompt reproduces the full forward's argmax chain
        gen = ec.greedy_decode(E, x[:, :plen], max_new_tokens=6, **kw)
        chain = x[:, :plen]
        with torch.no_grad():
            for t in gen[:-1]:
                chain = torch.cat([chain, torch.tensor([[t]])], 1)
            ref_tokens = E(chain, return_mtp=False, **kw).logits[0, plen - 1:].argmax(-1).tolist()
        assert gen == ref_tokens[: len(gen)]
    loads = [float(E.layers[li].mlp.last_expert_load[72:].sum()) for li in (1, 2)]
    print(f"\n[kv] max|dlogit| incremental vs full = {worst:.3e}; donor load (last step) {loads}")
    assert worst < 1e-4


def test_save_load_roundtrip_exact(arch, full, tmp_path_factory):
    E = full
    tok = arch["tok"]
    _install_random_donors(E, seed=5)
    E.wake_donor_experts(0.5)
    _set_all_gates(E, seed=12)
    path = Path(tmp_path_factory.mktemp("expanse")) / "roundtrip.pt"
    receipt = {"stage": "test", "note": "round trip", "graph": {"must": "be dropped"}}
    t0 = time.time()
    ec.save_expanse(path, E, tok, {"run_name": "test"}, receipt)
    size_mb = path.stat().st_size / 1e6
    E2, tok2, payload = ec.load_expanse(path)
    print(f"\n[io] saved+loaded {size_mb:.1f} MB in {time.time() - t0:.1f}s")
    assert payload["schema"] == ec.EXP_SCHEMA and payload["base_schema"] == ec.ARCH_SCHEMA
    assert "graph" not in payload["expanse"] and payload["expanse"]["stage"] == "test"
    assert payload["expanse"]["cns"]["nnz"] == 420 and payload["expanse"]["cns"]["steps"] == 4
    assert payload["expanse"]["donor_experts"] == json.loads(json.dumps(E.donor_receipts))
    assert tok2.tokens == tok.tokens
    s1, s2 = E.state_dict(), E2.state_dict()
    assert s1.keys() == s2.keys()
    bad = [k for k in s1 if s1[k].dtype != s2[k].dtype or not torch.equal(s1[k], s2[k])]
    assert not bad, bad[:5]
    x, feats = arch["x"], arch["feats"]
    state = E.omni7_state_for([r["user"] for r in arch["rows"]])
    a = _logits(E, x, feats, omni7_state=state)
    b = _logits(E2, x, feats, omni7_state=E2.omni7_state_for([r["user"] for r in arch["rows"]]))
    assert torch.equal(a, b)
    assert E2.gate_report() == E.gate_report()
    del E2, payload
    path.unlink()


# ---------------------------------------------------------------------------
# trainer helpers (train_expanse.py)
# ---------------------------------------------------------------------------
def _te():
    sys.path.insert(0, str(HERE.parent))
    import train_expanse as te
    return te


def test_rowboost_scales_the_realised_update():
    """RowBoost multiplies the optimizer step of chosen rows (a gradient scale would cancel in Adam)."""
    te = _te()
    torch.manual_seed(0)
    init = torch.randn(5, 3)
    grad = torch.randn(5, 3)

    def one_step(boost_factor):
        p = torch.nn.Parameter(init.clone())
        opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.01)
        rb = te.RowBoost([(p, torch.tensor([1, 3])), (p, torch.tensor([3]))], boost_factor)
        p.grad = grad.clone()
        rb.snapshot()
        opt.step()
        rb.apply()
        return p.detach() - init

    base, boosted = one_step(1.0), one_step(4.0)
    assert torch.allclose(boosted[[1, 3]], 4.0 * base[[1, 3]], atol=1e-6)
    assert torch.equal(boosted[[0, 2, 4]], base[[0, 2, 4]])
    # and why not a gradient hook: Adam's first step is sign-like, invariant to a gradient scale
    p1, p2 = torch.nn.Parameter(init.clone()), torch.nn.Parameter(init.clone())
    for p, g in ((p1, grad), (p2, 4.0 * grad)):
        opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.0)
        p.grad = g.clone()
        opt.step()
    assert torch.allclose(p1, p2, atol=1e-6)


def test_kd_topk_loss():
    te = _te()
    torch.manual_seed(1)
    logits = torch.randn(6, 32)
    idx = torch.arange(32).expand(6, 32)
    assert float(te.kd_topk_loss(logits, idx, logits.clone())) < 1e-6  # V == k, same logits: KL 0
    s = torch.randn(6, 100, requires_grad=True)
    t_val, t_idx = torch.randn(6, 100).topk(8, dim=-1)
    loss = te.kd_topk_loss(s, t_idx, t_val)
    loss.backward()
    assert float(loss) > 0 and s.grad is not None and bool(torch.isfinite(s.grad).all())


def test_fly_self_distilled_rows(arch):
    """Rows are in fly_row format, fully covered by the Archimedes tokenizer, labelled by the frozen core."""
    te = _te()
    fly = arch["A"].fly_core
    rows = te.make_fly_rows(fly, 64, seed=3)
    assert rows == te.make_fly_rows(fly, 64, seed=3)  # deterministic
    tok = arch["tok"]
    for r in rows:
        ids, plen = tok.encode_turn(r["user"], r["assistant"])
        assert ec.text_utils.UNK not in ids and len(ids) <= 128
        assert r["action"] == int(torch.tensor(r["probs"]).argmax())
        assert f"action {r['action']}" in r["assistant"]
    obs = torch.tensor([r["obs"] for r in rows])
    with torch.no_grad():
        probs = fly.brains_forward(obs)["probs"]
    assert torch.allclose(probs.double(), torch.tensor([r["probs"] for r in rows], dtype=torch.float64), atol=1e-6)
    # senses are U-shaped: more mass near the ends of [0, 1] than in the middle
    food = obs[:, 2]
    assert float(((food < 0.2) | (food > 0.8)).float().mean()) > 0.45


def test_dev_split_is_stable():
    te = _te()
    rows = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(400)]
    tr, dv = te.split_source(rows, 0.05, 7, dev_cap=0)
    tr2, dv2 = te.split_source(rows[::-1][:300], 0.05, 7, dev_cap=0)
    keys = {r["user"] for r in dv}
    assert 5 <= len(dv) <= 40 and len(tr) + len(dv) == 400
    assert {r["user"] for r in dv2} == {r["user"] for r in rows[::-1][:300]} & keys


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
