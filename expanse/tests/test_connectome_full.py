"""Tests for the full-connectome core (``src/connectome_full.py``).

What must hold, and why:

* zero gate  -> the core is *bitwise* the identity (function preservation at
  graft time; Expanse's build step compares logits against Archimedes);
* causality  -> a change at token t never changes any position < t (the core
  is per-column, so KV decode equals the full forward);
* CSRMatmul  -> its hand-written backward (``W^T @ grad``) matches finite
  differences, and the whole core passes gradcheck in double;
* receipts   -> the numbers the design promises for the real npz
  (11,751 types, 3,830,931 edges, 1,277 in, 713 out, |W| radius <= 0.9);
* the null   -> keeps degrees, row sums, inhibitory-input counts and signs;
* text rows  -> (``connectome_text.py``) every stated fact is exact, held-out
  types never appear in train rows, replies fit the house style.

The timing measurement (full graph, M=512, 8 threads) is heavy and is run
once through the CLI (``python src/connectome_full.py time``); here it only
runs with ``EXPANSE_RUN_TIMING=1``.

Run: ``python -m pytest tests/test_connectome_full.py -q``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import connectome_full as cf  # noqa: E402
import connectome_text as ct  # noqa: E402

REAL_NPZ = ROOT / "data" / "malecns_types.npz"
REAL_SHA256 = "a582ac7da0c86221c654c0793aef78f3049d5fc4fa4f8c264e9a1f3eda2b0271"

torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def write_tiny_npz(path: Path, n: int = 48, density: float = 0.15, seed: int = 0) -> Path:
    """A synthetic type table in the exact ``malecns_types.npz`` schema."""

    rng = np.random.default_rng(seed)
    names = np.array([f"T{i:03d}" for i in range(n)], dtype=object)
    pool = (["cb_sensory"] * 3 + ["visual_projection"] * 2 + ["ascending_neuron"] * 2 + ["descending_neuron"] * 3
            + ["vnc_motor"] * 2 + ["cb_intrinsic"] * 10 + ["vnc_intrinsic"] * 4)
    superclass = np.array([pool[i % len(pool)] for i in range(n)], dtype=object)
    nts = ["acetylcholine", "gaba", "glutamate", "dopamine", "unclear"]
    nt = np.array([nts[int(rng.integers(0, len(nts)))] for _ in range(n)], dtype=object)
    sign = np.array([-1 if x in ("gaba", "glutamate") else 1 for x in nt], dtype=np.int8)
    mask = rng.random((n, n)) < density
    mask[0, :] = False  # type 0 receives no input at all
    post, pre = np.nonzero(mask)
    perm = rng.permutation(len(post))  # unsorted on purpose
    weight = rng.integers(1, 400, size=len(post)).astype(np.int64)
    np.savez_compressed(
        path,
        type_names=names, superclass=superclass, cell_class=np.array(["unknown"] * n, dtype=object), nt=nt,
        sign=sign, sign_is_modulatory=np.array([x == "dopamine" for x in nt]),
        n_neurons=rng.integers(1, 30, size=n).astype(np.int32),
        has_flywire=np.ones(n, dtype=bool), has_manc=np.zeros(n, dtype=bool),
        pre=pre[perm].astype(np.int32), post=post[perm].astype(np.int32), weight=weight[perm],
    )
    return path


@pytest.fixture(scope="module")
def tiny_graph(tmp_path_factory):
    path = write_tiny_npz(tmp_path_factory.mktemp("cns") / "tiny.npz")
    return cf.build_full_graph(path)


@pytest.fixture(scope="module")
def real_graph():
    if not REAL_NPZ.is_file():
        pytest.skip(f"{REAL_NPZ} not present")
    return cf.build_full_graph(REAL_NPZ)


def make_core(graph, hidden=16, steps=4, gate=0.0, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    core = cf.FullConnectomeCore(hidden, graph, steps=steps).to(dtype)
    with torch.no_grad():
        # Move every parameter off its init so the test exercises real dynamics.
        core.log_gain_in.normal_(0.0, 0.3)
        core.log_gain_out.normal_(0.0, 0.3)
        core.bias.normal_(0.0, 0.1)
        core.leak_logit.normal_(0.0, 0.5)
        core.in_proj.weight.mul_(3.0)
        if gate == 0.0:
            core.gate.zero_()
        else:
            core.gate.normal_(0.0, gate)
    return core


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------
def test_graph_weights_and_order(tiny_graph):
    g = tiny_graph
    n = g["n"]
    key = g["post"].astype(np.int64) * n + g["pre"]
    assert (np.diff(key) > 0).all(), "edges must be sorted by post then pre, without duplicates"
    assert g["post"].dtype == np.int32 and g["pre"].dtype == np.int32 and g["value"].dtype == np.float32
    # value = fraction * sign[pre] * radius  => |W| rows sum to radius where there is input
    row = np.bincount(g["post"], weights=np.abs(g["value"]).astype(np.float64), minlength=n)
    has_in = np.bincount(g["post"], minlength=n) > 0
    assert np.allclose(row[has_in], 0.9, atol=1e-6)
    assert row[0] == 0.0 and g["receipt"]["types_without_input"] == 1
    assert (np.sign(g["value"]) == g["sign"][g["pre"]]).all()
    sc = np.array([str(s) for s in g["superclass"]])
    assert set(sc[g["in_idx"]]) <= cf.SUPERCLASS_IN and set(sc[g["out_idx"]]) <= cf.SUPERCLASS_OUT
    assert len(g["in_idx"]) == int(np.isin(sc, list(cf.SUPERCLASS_IN)).sum())
    r = g["receipt"]
    assert r["abs_spectral_radius"] <= 0.9 + 1e-6
    assert r["abs_spectral_radius_upper_bound"] >= r["abs_spectral_radius"] - 1e-9
    assert r["synapse_mass_kept"] == 1.0 and r["n_edges"] == len(g["post"])


def test_threshold_uses_fraction_of_all_input(tmp_path):
    path = write_tiny_npz(tmp_path / "t.npz", seed=3)
    full = cf.build_full_graph(path)
    cut = cf.build_full_graph(path, min_input_fraction=0.05)
    assert 0 < cut["receipt"]["n_edges"] < full["receipt"]["n_edges"]
    assert cut["receipt"]["synapse_mass_kept"] < 1.0
    # Kept edges keep exactly the value they had in the full graph.
    kf = dict(zip(zip(full["post"].tolist(), full["pre"].tolist()), full["value"].tolist()))
    for q, p, v in zip(cut["post"].tolist(), cut["pre"].tolist(), cut["value"].tolist()):
        assert kf[(q, p)] == v
        assert abs(v) >= 0.05 * 0.9 - 1e-7


# ---------------------------------------------------------------------------
# the sparse operator
# ---------------------------------------------------------------------------
def _operators(graph, dtype):
    crow, col, val = cf.csr_arrays(graph["post"], graph["pre"], graph["value"], graph["n"])
    crow_t, col_t, val_t = cf.csr_arrays(graph["pre"], graph["post"], graph["value"], graph["n"])
    w = cf.make_csr(torch.from_numpy(crow), torch.from_numpy(col), torch.from_numpy(val).to(dtype), graph["n"])
    w_t = cf.make_csr(torch.from_numpy(crow_t), torch.from_numpy(col_t), torch.from_numpy(val_t).to(dtype),
                      graph["n"])
    return w, w_t


def test_csr_matmul_matches_dense(tiny_graph):
    g = tiny_graph
    w, w_t = _operators(g, torch.float64)
    dense = torch.zeros(g["n"], g["n"], dtype=torch.float64)
    dense[torch.from_numpy(g["post"].astype(np.int64)), torch.from_numpy(g["pre"].astype(np.int64))] = \
        torch.from_numpy(g["value"]).double()
    x = torch.randn(g["n"], 5, dtype=torch.float64)
    assert torch.allclose(cf.CSRMatmul.apply(x, w, w_t), dense @ x, atol=1e-12)
    assert torch.allclose(torch.sparse.mm(w_t, x), dense.t() @ x, atol=1e-12)


def test_spmm_column_slabs_change_nothing(tiny_graph):
    w, w_t = _operators(tiny_graph, torch.float32)
    x = torch.randn(tiny_graph["n"], 11)
    whole = torch.sparse.mm(w, x)
    for chunk in (1, 3, 4, 11, 128):
        assert torch.equal(cf.spmm(w, x, chunk=chunk), whole), chunk
    xg = torch.randn(tiny_graph["n"], 300, dtype=torch.float64, requires_grad=True)  # > SPMM_CHUNK columns
    w64, w64_t = _operators(tiny_graph, torch.float64)
    y = cf.CSRMatmul.apply(xg, w64, w64_t)
    g = torch.randn_like(y)
    (y * g).sum().backward()
    assert torch.allclose(xg.grad, torch.sparse.mm(w64_t, g), atol=1e-12)


def test_csr_matmul_gradcheck(tiny_graph):
    w, w_t = _operators(tiny_graph, torch.float64)
    x = torch.randn(tiny_graph["n"], 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: cf.CSRMatmul.apply(inp, w, w_t), (x,), eps=1e-6, atol=1e-8)


def test_core_gradcheck_double(tiny_graph):
    core = make_core(tiny_graph, hidden=6, steps=3, gate=0.5, dtype=torch.float64)
    hidden = torch.randn(1, 3, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda h: core(h)[0], (hidden,), eps=1e-6, atol=1e-6)
    # Gradients reach every trainable piece of the core.
    out, _ = core(hidden)
    out.sum().backward()
    for name in ("in_proj.weight", "out_proj.weight", "log_gain_in", "log_gain_out", "bias", "leak_logit", "gate"):
        grad = dict(core.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


# ---------------------------------------------------------------------------
# function preservation and causality
# ---------------------------------------------------------------------------
def test_zero_gate_is_bitwise_identity(tiny_graph):
    core = make_core(tiny_graph, hidden=16, gate=0.0)
    hidden = torch.randn(3, 9, 16) * 4.0
    mask = torch.rand(3, 9) > 0.3
    for token_mask in (None, mask):
        # training path (grad enabled -> the core really runs)
        out, info = core(hidden.clone().requires_grad_(True), token_mask)
        assert torch.equal(out.detach(), hidden)
        assert "skipped" not in info and info["M"] == (27 if token_mask is None else int(mask.sum()))
        # inference path (closed gate short-circuits)
        with torch.no_grad():
            out2, info2 = core(hidden, token_mask)
        assert torch.equal(out2, hidden) and info2.get("skipped")
        assert out2.data_ptr() != hidden.data_ptr()  # a copy, never an alias


def test_open_gate_writes_only_selected_positions(tiny_graph):
    core = make_core(tiny_graph, hidden=16, gate=0.5)
    hidden = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0]], dtype=torch.bool)
    with torch.no_grad():
        out, info = core(hidden, mask)
    assert info["M"] == 6 and 0.0 < info["fraction_active"] <= 1.0
    assert torch.equal(out[~mask], hidden[~mask])
    assert ((out[mask] - hidden[mask]).abs().amax(dim=-1) > 0).all()


def test_causality_tiny(tiny_graph):
    core = make_core(tiny_graph, hidden=16, gate=0.7)
    hidden = torch.randn(2, 8, 16)
    t = 5
    changed = hidden.clone()
    changed[:, t:] += torch.randn(2, 8 - t, 16)
    with torch.no_grad():
        a, _ = core(hidden)
        b, _ = core(changed)
    assert torch.equal(a[:, :t], b[:, :t]), "a later token changed an earlier position"
    assert not torch.equal(a[:, t], b[:, t])


def test_incremental_decode_matches_full(tiny_graph):
    """KV decode: the hook sees prefix then one new position at a time."""

    core = make_core(tiny_graph, hidden=16, gate=0.7)
    hidden = torch.randn(2, 7, 16)
    with torch.no_grad():
        full, _ = core(hidden)
        parts = [core(hidden[:, :4])[0]] + [core(hidden[:, i:i + 1])[0] for i in range(4, 7)]
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-6, rtol=0)


def test_causality_and_zero_gate_full_graph(real_graph):
    core = make_core(real_graph, hidden=32, gate=0.0, seed=1)
    hidden = torch.randn(1, 6, 32)
    out, info = core(hidden.clone().requires_grad_(True))
    assert torch.equal(out.detach(), hidden) and info["M"] == 6
    with torch.no_grad():
        core.gate.normal_(0.0, 0.5)
        changed = hidden.clone()
        changed[:, 4] += 1.0
        a, info_a = core(hidden)
        b, _ = core(changed)
    assert torch.equal(a[:, :4], b[:, :4]) and not torch.equal(a[:, 4], b[:, 4])
    assert torch.isfinite(a).all() and info_a["fraction_active"] > 0
    # The torch CSR operators over the real 3.8M-edge graph agree with scipy (fp64).
    w, w_t = core._operators()
    x = torch.randn(real_graph["n"], 3)
    ref = cf._scipy_csr(real_graph["post"], real_graph["pre"], real_graph["value"], real_graph["n"])
    assert np.abs(cf.spmm(w, x).double().numpy() - ref @ x.double().numpy()).max() < 1e-4
    assert np.abs(cf.spmm(w_t, x).double().numpy() - ref.T @ x.double().numpy()).max() < 1e-4


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip_from_shapes_only(tiny_graph):
    src = make_core(tiny_graph, hidden=16, gate=0.3)
    meta = src.graph_meta()
    dst = cf.FullConnectomeCore(16, {k: meta[k] for k in ("n", "n_in", "n_out", "nnz")}, steps=meta["steps"])
    hidden = torch.randn(2, 5, 16)
    with torch.no_grad():
        dst.gate.fill_(0.3)
        dst(hidden)  # builds (and caches) CSR views over the all-zero placeholder buffers
        dst.load_state_dict(src.state_dict())
        assert torch.equal(dst(hidden)[0], src(hidden)[0]), "stale CSR cache after load_state_dict"
    names = set(src.state_dict())
    assert {"crow", "col", "val", "crow_t", "col_t", "val_t", "in_idx", "out_idx", "sign"} <= names
    assert src.crow.dtype == torch.int32 and src.col.dtype == torch.int32 and src.val.dtype == torch.float32


def test_set_graph_swaps_the_wiring(tiny_graph):
    core = make_core(tiny_graph, hidden=16, gate=0.5)
    hidden = torch.randn(1, 4, 16)
    with torch.no_grad():
        before, _ = core(hidden)
        core.set_graph(cf.rewired_graph(tiny_graph, seed=3, rounds=10))
        after, _ = core(hidden)
        core.set_graph(tiny_graph)
        again, _ = core(hidden)
    assert not torch.equal(before, after)
    assert torch.equal(before, again)


# ---------------------------------------------------------------------------
# the null
# ---------------------------------------------------------------------------
def test_rewired_graph_invariants(tiny_graph):
    g = tiny_graph
    r = cf.rewired_graph(g, seed=0, rounds=30)
    n = g["n"]
    rec = r["receipt"]
    assert rec["rewire"]["swaps_accepted"] > 0 and rec["edge_overlap_with_real"] < 1.0
    for field in ("in_degree_changed_types", "out_degree_changed_types", "inhibitory_input_count_changed_types",
                  "duplicate_edges"):
        assert rec[field] == 0, field
    assert rec["abs_row_sum_max_abs_diff"] < 1e-9 and rec["value_sign_matches_source"]
    key = r["post"].astype(np.int64) * n + r["pre"]
    assert (np.diff(key) > 0).all()
    assert ((r["post"] == r["pre"]).sum()) == ((g["post"] == g["pre"]).sum())
    # Each type keeps its exact multiset of input values.
    for q in range(n):
        assert sorted(g["value"][g["post"] == q].tolist()) == sorted(r["value"][r["post"] == q].tolist())
    # A synapse count belongs to a real (pre, post) pair; the null must not carry one.
    assert "weight" not in r


# ---------------------------------------------------------------------------
# the real npz
# ---------------------------------------------------------------------------
def test_receipt_real_npz(real_graph):
    r = real_graph["receipt"]
    assert r["npz_sha256"] == REAL_SHA256
    assert r["n_types"] == 11751 and real_graph["n"] == 11751
    assert r["n_edges"] == 3_830_931 and r["n_edges_source"] == 3_830_931
    assert r["synapses_total"] == 122_318_556 and r["synapse_mass_kept"] == 1.0
    assert r["n_in"] == 1277 and r["n_out"] == 713
    assert r["threshold"] == 0.0 and r["radius"] == 0.9
    assert 0.89 < r["abs_spectral_radius"] <= 0.9 + 1e-6
    assert r["abs_row_sum_max"] <= 0.9 + 1e-6
    assert 0.0 < r["signed_spectral_radius_est"] <= r["abs_spectral_radius"] + 1e-6
    sc = r["sign_counts"]
    assert sc["excitatory_types"] + sc["inhibitory_types"] + sc["modulatory_types"] == 11751
    assert sc["inhibitory_types"] == 2778 + 2429 + 16  # gaba + glutamate + histamine (v91 receipt)
    assert sc["modulatory_types"] == 86 + 44 + 52      # serotonin + dopamine + octopamine
    assert real_graph["post"].dtype == np.int32 and real_graph["value"].dtype == np.float32
    assert (np.sign(real_graph["value"]) == real_graph["sign"][real_graph["pre"]]).all()


@pytest.mark.skipif(os.environ.get("EXPANSE_RUN_TIMING") != "1", reason="set EXPANSE_RUN_TIMING=1 (heavy)")
def test_timing_full_graph_m512(real_graph):
    result = cf.time_core(real_graph, columns=(512,), steps=4, threads=8, repeats=1)
    torch.set_num_threads(2)
    row = result["columns"]["512"]
    print(f"\nfull graph, M=512, steps=4, 8 threads: {row['forward_s']:.3f} s/forward, "
          f"{row['backward_s']:.3f} s/backward")
    assert row["forward_s"] > 0 and row["backward_s"] > 0


# ---------------------------------------------------------------------------
# connectome_text: language distillation rows
# ---------------------------------------------------------------------------
def _brute_force_facts(npz_path, name):
    """Facts for one type straight from the raw arrays, with plain Python loops."""

    d = cf.load_npz(npz_path)
    names = [str(x) for x in d["type_names"]]
    i = names.index(name)
    edges = list(zip(d["pre"].tolist(), d["post"].tolist(), d["weight"].tolist()))
    ins = [(w, names[p]) for p, q, w in edges if q == i]
    outs = [(w, names[q]) for p, q, w in edges if p == i]

    def top(pairs):
        if not pairs:
            return "none 0"
        w = max(p[0] for p in pairs)
        return f"{min(n for ww, n in pairs if ww == w)} {w}"

    return {"cns_top_input": top(ins), "cns_top_output": top(outs), "cns_in_degree": str(len(ins)),
            "cns_type_count": str(int(d["n_neurons"][i])), "cns_nt": str(d["nt"][i]),
            "cns_superclass": str(d["superclass"][i])}


def test_text_rows_exact_and_split(tmp_path):
    path = write_tiny_npz(tmp_path / "t.npz", n=60, seed=5)
    graph = cf.build_full_graph(path)
    rows = ct.build_connectome_rows(graph, seed=7, n_rows=140, heldout_fraction=0.2)
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "heldout"]
    assert len(train) == 140 and len(held) == 35
    assert {r["task"] for r in train} == set(ct.TASKS)
    assert not ({r["cell_type"] for r in train} & {r["cell_type"] for r in held}), "held-out type leaked"
    assert len({(r["task"], r["cell_type"]) for r in rows}) == len(rows)
    assert len({r["user"] for r in rows}) == len(rows)
    for r in rows:
        assert r["domain"] == "connectome" and r["user"].isascii() and r["assistant"].isascii()
        assert "\n" not in r["assistant"] and len(r["assistant"].split()) <= 60
        if r["task"] in ct.NUMERIC_TASKS:
            assert r["assistant"].endswith(f"total {r['answer'].split()[-1]}")
        if r["task"] != "cns_path_role":
            assert r["answer"] == _brute_force_facts(path, r["cell_type"])[r["task"]], r
    kept, dropped = ct.verify_rows(rows, path)
    assert len(kept) == len(rows) and not dropped
    # A tampered answer is caught by the independent verifier.
    bad = dict(train[0], answer="wrong")
    assert ct.verify_rows([bad], path)[1] == {bad["task"]: 1}


def test_text_rows_deterministic(tmp_path):
    path = write_tiny_npz(tmp_path / "t.npz", n=60, seed=5)
    graph = cf.build_full_graph(path)
    a = ct.build_connectome_rows(graph, seed=3, n_rows=70)
    b = ct.build_connectome_rows(graph, seed=3, n_rows=70)
    assert json.dumps(a) == json.dumps(b)
    assert ct.split_types(graph["type_names"], 3) == ct.split_types(graph["type_names"], 3)


def test_text_needs_full_graph(tmp_path):
    path = write_tiny_npz(tmp_path / "t.npz", seed=5)
    with pytest.raises(ValueError):
        ct.build_connectome_rows(cf.build_full_graph(path, min_input_fraction=0.05), seed=0, n_rows=10)
