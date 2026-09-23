"""The full male-CNS connectome as a fixed recurrent core for Supermix Expanse.

## Why this exists

Archimedes (v93) carried the Janelia male-CNS connectome as a 2 x 416-module
graph: 11,751 cell types clustered into modules and thresholded at a 1%
input fraction, so the core that ran inside the trunk was a caricature of the
wiring -- a few thousand module-to-module edges. Expanse runs the **real type
graph**: every one of the 11,751 cell types is a unit and every one of the
3,830,931 type-to-type edges is a synapse with its measured weight. Nothing is
clustered, nothing is thresholded by default, and the weights are not learned.
What the trunk learns is only how to *drive* the afferent types (a projection
into the 1,277 sensory / visual-projection / ascending types) and how to *read*
the efferent types (a projection out of the 713 descending / motor / efferent /
endocrine types), plus a per-type gain, bias and leak.

## The maths

``W[post, pre] = input_fraction(post, pre) * sign[pre] * radius`` where
``input_fraction`` is the share of all synapses onto ``post`` that come from
``pre`` (over ALL edges, so each row of ``|W|`` sums to ``radius`` for every
type that receives input). Dale's sign belongs to the presynaptic type
(acetylcholine +1; GABA, glutamate, histamine -1; monoamines +1 and flagged
modulatory -- the v91 convention in ``malecns_connectome.NT_SIGN``). Because
``|W|`` is ``radius`` times a (sub)stochastic matrix, its spectral radius is at
most ``radius`` (0.9), so the rate dynamics below are contractive on the
linear part; the receipt measures it by power iteration rather than trusting
the bound.

Per token (one column), with ``a = sigmoid(leak_logit)``::

    U[in_idx] = in_proj(in_norm(h))
    r_0 = 0
    r_{k+1} = (1 - a) r_k + a relu( g_in * W (g_out * r_k) + U + bias )
    e = r_K[out_idx];   write = out_proj(e / rms(e))
    h <- h + gate * write

Columns never mix: the core is applied to each selected token position on its
own, so it is exactly causal and KV-cache decode equals the full forward.

## Performance

``W`` has 3.8M non-zeros; one spmm over ``M`` token columns is ``2 * nnz * M``
flops. It runs as a torch sparse CSR matmul on int32 indices (fp32 only: bf16
is ~1 GFLOP/s on the Snapdragon CPU). ``CSRMatmul`` keeps ``W`` and ``W^T`` both
prebuilt so the backward pass is one more CSR spmm and never transposes on the
fly, and runs the spmm in 128-column slabs (cache-friendly, ~1.8x faster at
M=1024). The first step skips the spmm entirely because ``r_0 = 0``.
Measured on the full graph, steps=4, 8 threads, while other jobs shared the
CPU (``data/connectome_full_timing.json``): M=512 columns 0.16 s forward /
0.17 s backward; M=1024 0.33 s / 0.35 s.

## Nulls

:func:`rewired_graph` is the degree-preserving, stratified null the eval uses
as an ablation. It follows ``malecns_connectome.stratified_rewire`` (targets
swap only between edges whose *source* types share a stratum; each edge's
value travels with its *target*, so every type keeps its in-degree,
out-degree, self-loop and exact multiset of input fractions) but is
vectorised: the Python loop there would need ~40M iterations on this graph.

Source data: Janelia FlyEM male-cns v1.0 (CC BY 4.0), via the v91
``malecns_types.npz`` table (see ``malecns_connectome.py`` in Archimedes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXPANSE = HERE.parent
DEFAULT_NPZ = EXPANSE / "data" / "malecns_types.npz"

GRAPH_SCHEMA = "supermix-expanse-full-cns-graph-v1"

#: Superclasses whose types the trunk drives (afferent side of the core).
SUPERCLASS_IN = {"cb_sensory", "vnc_sensory", "ol_sensory", "sensory_ascending", "sensory_descending",
                 "visual_projection", "ascending_neuron"}   # 1,277 types
#: Superclasses whose types the trunk reads (efferent side of the core).
SUPERCLASS_OUT = {"descending_neuron", "cb_motor", "vnc_motor", "vnc_efferent", "cb_efferent",
                  "efferent_ascending", "efferent_descending", "cb_endocrine", "vnc_endocrine"}  # 713 types

#: Rewiring strata use a 3-way role of the source type.
ROLE_OTHER, ROLE_IN, ROLE_OUT = 0, 1, 2


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: os.PathLike) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _scipy_csr(post: np.ndarray, pre: np.ndarray, value: np.ndarray, n: int):
    import scipy.sparse as sp

    return sp.csr_matrix((value.astype(np.float64), (post.astype(np.int64), pre.astype(np.int64))), shape=(n, n))


def abs_spectral_radius(post: np.ndarray, pre: np.ndarray, value: np.ndarray, n: int,
                        iters: int = 1000, tol: float = 1e-10) -> Dict[str, float]:
    """Perron root of ``|W|`` by shifted power iteration, with Collatz-Wielandt bounds.

    Iterating ``|W| + c I`` (c = mean |row sum|) instead of ``|W|`` removes any
    periodicity: the Perron root stays the eigenvalue of largest modulus after
    the shift, so the iteration converges for reducible graphs too. For the
    final positive vector ``x``, ``max_i (|W| x)_i / x_i`` is a rigorous upper
    bound on the spectral radius.
    """

    a = _scipy_csr(post, pre, np.abs(value), n)
    row_sum = np.asarray(a.sum(axis=1)).ravel()
    shift = float(row_sum.mean()) if row_sum.size else 0.0
    x = np.full(n, 1.0 / math.sqrt(max(n, 1)))
    lam = prev = 0.0
    it = 0
    for it in range(1, iters + 1):
        y = a @ x + shift * x
        norm = float(np.linalg.norm(y))
        if norm == 0.0:
            return {"estimate": 0.0, "upper_bound": 0.0, "iterations": it, "row_sum_max": 0.0}
        lam = norm - shift  # ||x|| == 1
        x = y / norm
        if it > 10 and abs(lam - prev) <= tol * max(1.0, abs(lam)):
            break
        prev = lam
    ax = a @ x
    ratio = np.divide(ax, x, out=np.zeros_like(ax), where=x > 0)
    return {
        "estimate": float(x @ ax),  # Rayleigh quotient at the converged unit vector
        "upper_bound": float(ratio.max()),
        "iterations": int(it),
        "row_sum_max": float(row_sum.max()) if row_sum.size else 0.0,
    }


def signed_spectral_radius(post: np.ndarray, pre: np.ndarray, value: np.ndarray, n: int,
                           seed: int = 0) -> Dict[str, Any]:
    """Largest-modulus eigenvalue of the signed ``W`` (ARPACK), with a Gelfand fallback.

    The signed matrix is not symmetric and its dominant eigenvalue may be a
    complex pair, so plain power iteration does not converge. ARPACK (``eigs``)
    handles that; if it fails to converge the growth rate
    ``||W^k x||^(1/k)`` over 200 steps is reported instead (an estimate that
    approaches the spectral radius from Gelfand's formula).
    """

    from scipy.sparse.linalg import ArpackNoConvergence, eigs

    w = _scipy_csr(post, pre, value, n)
    rng = np.random.default_rng(seed)
    try:
        vals = eigs(w, k=1, which="LM", tol=1e-6, maxiter=5000, v0=rng.standard_normal(n),
                    return_eigenvectors=False)
        lam = complex(vals[0])
        return {"estimate": float(abs(lam)), "eigenvalue": [float(lam.real), float(lam.imag)],
                "method": "arpack_eigs_LM"}
    except (ArpackNoConvergence, Exception) as exc:  # pragma: no cover - fallback path
        x = rng.standard_normal(n)
        x /= np.linalg.norm(x)
        log_growth = 0.0
        steps = 200
        for _ in range(steps):
            x = w @ x
            norm = float(np.linalg.norm(x))
            if norm == 0.0:
                return {"estimate": 0.0, "method": "gelfand_growth", "note": "W is nilpotent on x"}
            log_growth += math.log(norm)
            x /= norm
        return {"estimate": float(math.exp(log_growth / steps)), "method": "gelfand_growth_200",
                "arpack_error": type(exc).__name__}


def build_full_graph(npz_path, min_input_fraction: float = 0.0, radius: float = 0.9) -> dict:
    """The type graph with fixed signed input-fraction weights, CSR-ready.

    Returns numpy arrays ``post``/``pre`` (int32, sorted by post then pre),
    ``value`` (float32, ``input_fraction * sign[pre] * radius``), ``sign``,
    ``in_idx``/``out_idx`` (int64, sorted), ``type_names``/``superclass``/``nt``
    (object), ``n`` and a ``receipt``. The default keeps all 3,830,931 edges.
    Extra keys for the text distillation and eval (not used by the core):
    ``weight`` (int64 synapse counts aligned with post/pre), ``n_neurons``,
    ``cell_class`` and ``sign_is_modulatory``.
    """

    started = time.time()
    npz_path = Path(npz_path)
    data = load_npz(npz_path)
    names = data["type_names"]
    n = int(len(names))
    pre = data["pre"].astype(np.int64)
    post = data["post"].astype(np.int64)
    weight = data["weight"].astype(np.int64)
    sign = data["sign"].astype(np.int8)
    if not (len(pre) == len(post) == len(weight)):
        raise ValueError("pre/post/weight lengths differ")
    if len(pre) and (pre.min() < 0 or post.min() < 0 or pre.max() >= n or post.max() >= n):
        raise ValueError("edge endpoint out of range")
    if (weight <= 0).any():
        raise ValueError("non-positive synapse count in the edge list")
    if not np.isin(sign, (-1, 1)).all():
        raise ValueError("sign must be +-1 for every type")
    key = post * n + pre
    if len(np.unique(key)) != len(key):
        raise ValueError("duplicate (post, pre) edges; the type table should be reduced already")

    # Input fraction over ALL edges, before any threshold.
    total_in = np.bincount(post, weights=weight.astype(np.float64), minlength=n)
    fraction = weight / total_in[post]
    keep = fraction >= float(min_input_fraction) if min_input_fraction > 0 else np.ones(len(pre), dtype=bool)
    order = np.flatnonzero(keep)
    order = order[np.argsort(key[order], kind="stable")]  # post major, pre minor (key = post*n + pre)
    post_k, pre_k, w_k, f_k = post[order], pre[order], weight[order], fraction[order]
    value = (f_k * sign[pre_k].astype(np.float64) * float(radius)).astype(np.float32)

    superclass = data["superclass"]
    sc = np.array([str(s) for s in superclass], dtype=object)
    in_idx = np.flatnonzero(np.isin(sc, sorted(SUPERCLASS_IN))).astype(np.int64)
    out_idx = np.flatnonzero(np.isin(sc, sorted(SUPERCLASS_OUT))).astype(np.int64)
    modulatory = data["sign_is_modulatory"].astype(bool)
    nt = data["nt"]

    rho_abs = abs_spectral_radius(post_k, pre_k, value, n)
    rho_signed = signed_spectral_radius(post_k, pre_k, value, n)
    synapses_total = int(weight.sum())
    receipt = {
        "schema": GRAPH_SCHEMA,
        "npz": str(npz_path.resolve()).replace("\\", "/"),
        "npz_sha256": sha256_file(npz_path),
        "n_types": n,
        "n_edges": int(len(order)),
        "n_edges_source": int(len(pre)),
        "self_loops": int((post_k == pre_k).sum()),
        "synapses_total": synapses_total,
        "synapses_kept": int(w_k.sum()),
        "synapse_mass_kept": float(w_k.sum() / max(1, synapses_total)),
        "threshold": float(min_input_fraction),
        "radius": float(radius),
        "n_in": int(len(in_idx)),
        "n_out": int(len(out_idx)),
        "superclass_in": sorted(SUPERCLASS_IN),
        "superclass_out": sorted(SUPERCLASS_OUT),
        "types_without_input": int((total_in == 0).sum()),
        "sign_counts": {
            "excitatory_types": int(((sign > 0) & ~modulatory).sum()),
            "inhibitory_types": int((sign < 0).sum()),
            "modulatory_types": int(modulatory.sum()),
            "unclear_nt_types_signed_plus": int(sum(1 for x in nt if str(x) == "unclear")),
        },
        "edge_sign_counts": {"excitatory": int((value > 0).sum()), "inhibitory": int((value < 0).sum())},
        "abs_row_sum_max": float(rho_abs["row_sum_max"]),
        "abs_spectral_radius": float(rho_abs["estimate"]),
        "abs_spectral_radius_upper_bound": float(rho_abs["upper_bound"]),
        "abs_power_iterations": int(rho_abs["iterations"]),
        "signed_spectral_radius_est": float(rho_signed["estimate"]),
        "signed_spectral_method": rho_signed.get("method"),
        "signed_dominant_eigenvalue": rho_signed.get("eigenvalue"),
        "weights": "value = input_fraction(post, pre) * sign[pre] * radius; input fraction over all edges",
        "seconds": round(time.time() - started, 2),
    }
    return {
        "n": n,
        "post": post_k.astype(np.int32),
        "pre": pre_k.astype(np.int32),
        "value": value,
        "sign": sign,
        "in_idx": in_idx,
        "out_idx": out_idx,
        "type_names": names,
        "superclass": superclass,
        "nt": nt,
        "receipt": receipt,
        # extras (text distillation / eval)
        "weight": w_k,
        "n_neurons": data["n_neurons"].astype(np.int32),
        "cell_class": data["cell_class"],
        "sign_is_modulatory": modulatory,
    }


def graph_shapes(graph: dict) -> Dict[str, int]:
    """``{"n", "n_in", "n_out", "nnz"}`` of a graph dict (full or shape-only)."""

    if "post" in graph:
        return {"n": int(graph["n"]), "n_in": int(len(graph["in_idx"])), "n_out": int(len(graph["out_idx"])),
                "nnz": int(len(graph["post"]))}
    return {k: int(graph[k]) for k in ("n", "n_in", "n_out", "nnz")}


# ---------------------------------------------------------------------------
# The stratified null
# ---------------------------------------------------------------------------
def source_roles(graph: dict) -> np.ndarray:
    """Per type: 1 afferent (in_idx), 2 efferent (out_idx), 0 other."""

    role = np.full(int(graph["n"]), ROLE_OTHER, dtype=np.int64)
    role[np.asarray(graph["in_idx"], dtype=np.int64)] = ROLE_IN
    role[np.asarray(graph["out_idx"], dtype=np.int64)] = ROLE_OUT
    return role


def _vectorised_stratified_swaps(post: np.ndarray, pre: np.ndarray, value: np.ndarray, edge_stratum: np.ndarray,
                                 n: int, rounds: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Batched Maslov-Sneppen target swaps within edge strata; values move with targets.

    Each round shuffles the movable (non-self-loop) edges inside their stratum
    and pairs neighbours ``i=(a->b)``, ``j=(c->d)``. A pair swaps to
    ``a->d``, ``c->b`` unless that would create a self-loop, repeat an
    endpoint, or produce an edge already present *before the round* (a
    conservative check, so two swaps in one round can never collide with each
    other's removals), or a new edge proposed twice in the round. Accepted
    pairs are disjoint, so all of them apply at once.
    """

    rng = np.random.default_rng(seed)
    post = post.astype(np.int64).copy()
    pre = pre.astype(np.int64)
    value = value.copy()
    n64 = np.int64(n)
    movable = np.flatnonzero(post != pre)
    stratum_m = edge_stratum[movable]
    accepted = attempts = 0
    per_round = []
    for _ in range(rounds):
        present = np.sort(post * n64 + pre)
        # Group by stratum, random order inside it: one float argsort (a lexsort
        # on 3.8M rows is ~6x slower here).
        order = movable[np.argsort(stratum_m + rng.random(len(movable)))]
        s = edge_stratum[order]
        m = len(order) // 2
        i, j = order[0:2 * m:2], order[1:2 * m:2]
        same = s[0:2 * m:2] == s[1:2 * m:2]
        i, j = i[same], j[same]
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        ok = (a != c) & (b != d) & (a != d) & (c != b)
        k1 = d * n64 + a  # new edge a -> d
        k2 = b * n64 + c  # new edge c -> b

        def exists(keys: np.ndarray) -> np.ndarray:
            # Sort the queries first: searchsorted with random-order queries
            # is cache-bound and ~10x slower on 3.8M keys.
            q_order = np.argsort(keys)
            q_sorted = keys[q_order]
            pos = np.minimum(np.searchsorted(present, q_sorted), len(present) - 1)
            hit = np.empty(len(keys), dtype=bool)
            hit[q_order] = present[pos] == q_sorted
            return hit

        ok &= ~exists(k1) & ~exists(k2)
        cand = np.flatnonzero(ok)
        new_keys = np.concatenate([k1[cand], k2[cand]])
        uniq, inverse, counts = np.unique(new_keys, return_inverse=True, return_counts=True)
        dup = counts[inverse] > 1
        dup_pair = dup[: len(cand)] | dup[len(cand):]
        cand = cand[~dup_pair]
        ii, jj = i[cand], j[cand]
        post[ii], post[jj] = d[cand], b[cand]
        vi = value[ii].copy()
        value[ii] = value[jj]
        value[jj] = vi
        attempts += int(len(i))
        accepted += int(len(cand))
        per_round.append(int(len(cand)))
    return post, value, {
        "swaps_accepted": accepted,
        "swap_attempts": attempts,
        "rounds": int(rounds),
        "accepted_per_round": per_round,
        "swaps_per_movable_edge": round(accepted / max(1, len(movable)), 3),
        "edges_rewired": int(len(movable)),
        "self_loops_held": int(len(post) - len(movable)),
    }


def rewired_graph(graph: dict, seed: int = 0, rounds: int = 20) -> dict:
    """Degree-preserving null: swap targets only between edges whose sources share (sign, role).

    Role is afferent (``in_idx``) / efferent (``out_idx``) / other. Every
    edge's value travels with its target, so each type keeps its in-degree,
    its out-degree, its self-loop and the exact multiset of its input values
    (row sums of ``|W|`` unchanged, and since sources in a stratum share a
    sign, the sign of every value still matches its new source). What is
    destroyed is *which* same-kind type an input comes from. Eval ablation only.
    """

    started = time.time()
    n = int(graph["n"])
    post = np.asarray(graph["post"], dtype=np.int64)
    pre = np.asarray(graph["pre"], dtype=np.int64)
    value = np.asarray(graph["value"], dtype=np.float32)
    sign = np.asarray(graph["sign"], dtype=np.int64)
    role = source_roles(graph)
    node_stratum = (sign > 0).astype(np.int64) * 3 + role  # 6 strata
    edge_stratum = node_stratum[pre]
    r_post, r_value, info = _vectorised_stratified_swaps(post, pre, value, edge_stratum, n, rounds, seed)

    order = np.argsort(r_post * n + pre, kind="stable")
    out = dict(graph)
    out["post"] = r_post[order].astype(np.int32)
    out["pre"] = pre[order].astype(np.int32)
    out["value"] = r_value[order].astype(np.float32)
    if "weight" in graph:
        # A synapse count is a fact about a real (pre, post) pair; the null has none.
        out.pop("weight")

    # What the null must keep, measured rather than assumed.
    keys_real = np.sort(post * n + pre)
    keys_null = r_post[order] * n + pre[order]  # sorted queries (see _vectorised_stratified_swaps)
    pos = np.minimum(np.searchsorted(keys_real, keys_null), len(keys_real) - 1)
    overlap = float((keys_real[pos] == keys_null).mean())
    abs_row_real = np.bincount(post, weights=np.abs(value).astype(np.float64), minlength=n)
    abs_row_null = np.bincount(r_post, weights=np.abs(r_value).astype(np.float64), minlength=n)
    inhib_real = np.bincount(post[value < 0], minlength=n)
    inhib_null = np.bincount(r_post[r_value < 0], minlength=n)
    sign_ok = bool(((r_value > 0) == (sign[pre] > 0)).all())
    rho_abs = abs_spectral_radius(r_post, pre, r_value, n)
    rho_signed = signed_spectral_radius(r_post, pre, r_value, n)
    receipt = dict(graph["receipt"])
    receipt.update({
        "null": "stratified_by_source_sign_and_role_value_carried_with_target",
        "null_seed": int(seed),
        "rewire": info,
        "strata": int(len(np.unique(edge_stratum))),
        "edge_overlap_with_real": round(overlap, 6),
        "in_degree_changed_types": int((np.bincount(r_post, minlength=n) != np.bincount(post, minlength=n)).sum()),
        # Sources never move, so this is 0 by construction; measured on the output arrays anyway.
        "out_degree_changed_types": int((np.bincount(out["pre"], minlength=n)
                                         != np.bincount(pre, minlength=n)).sum()),
        "abs_row_sum_max_abs_diff": float(np.abs(abs_row_real - abs_row_null).max()),
        "inhibitory_input_count_changed_types": int((inhib_real != inhib_null).sum()),
        "value_sign_matches_source": sign_ok,
        "duplicate_edges": int(len(keys_null) - len(np.unique(keys_null))),
        "abs_spectral_radius": float(rho_abs["estimate"]),
        "abs_spectral_radius_upper_bound": float(rho_abs["upper_bound"]),
        "signed_spectral_radius_est": float(rho_signed["estimate"]),
        "signed_spectral_method": rho_signed.get("method"),
        "signed_dominant_eigenvalue": rho_signed.get("eigenvalue"),
        "rewire_seconds": round(time.time() - started, 2),
    })
    out["receipt"] = receipt
    return out


# ---------------------------------------------------------------------------
# Sparse operator
# ---------------------------------------------------------------------------
#: Columns per CSR spmm call. torch's CPU CSR x dense kernel streams the dense
#: operand once per row block; with N=11,751 rows a 1,024-column operand (48 MB)
#: falls out of cache, and 128-column slabs measured ~1.8x faster on this CPU
#: (0.47 s -> 0.25 s per spmm at M=1024, 8 threads). Columns are independent,
#: so slabbing changes no value.
SPMM_CHUNK = 128


def spmm(w: torch.Tensor, x: torch.Tensor, chunk: int = SPMM_CHUNK) -> torch.Tensor:
    """``w @ x`` for CSR ``w`` and dense 2-D ``x``, in column slabs of ``chunk``."""

    m = x.shape[1]
    if m <= chunk:
        return torch.sparse.mm(w, x.contiguous())
    out = x.new_empty(w.shape[0], m)
    for c in range(0, m, chunk):
        out[:, c:c + chunk] = torch.sparse.mm(w, x[:, c:c + chunk].contiguous())
    return out


class CSRMatmul(torch.autograd.Function):
    """``y = W @ x`` for a constant CSR ``W``; backward is ``grad_x = W^T @ grad_y``.

    ``W`` and ``W^T`` are both prebuilt CSR tensors, so neither pass
    transposes or converts anything. No gradient flows to ``W`` (the connectome
    weights are measured, not learned).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.w_t = w_t
        return spmm(w, x)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):  # type: ignore[override]
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = spmm(ctx.w_t, grad_y)
        return grad_x, None, None


def csr_arrays(post: np.ndarray, pre: np.ndarray, value: np.ndarray, n: int
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(crow int32, col int32, val float32) of ``W[post, pre] = value``; rows sorted by col."""

    post = np.asarray(post, dtype=np.int64)
    pre = np.asarray(pre, dtype=np.int64)
    order = np.argsort(post * np.int64(n) + pre, kind="stable")
    counts = np.bincount(post, minlength=n)
    crow = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=crow[1:])
    if crow[-1] >= np.iinfo(np.int32).max:
        raise ValueError("nnz does not fit int32 CSR indices")
    return crow.astype(np.int32), pre[order].astype(np.int32), np.asarray(value, dtype=np.float32)[order]


def make_csr(crow: torch.Tensor, col: torch.Tensor, val: torch.Tensor, n: int) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "Sparse CSR tensor support is in beta state"
        return torch.sparse_csr_tensor(crow, col, val, size=(n, n), check_invariants=False)


# ---------------------------------------------------------------------------
# The core
# ---------------------------------------------------------------------------
class FullConnectomeCore(nn.Module):
    """All 11,751 male-CNS cell types as a fixed-weight rate network, read/written per token.

    ``graph`` is a dict from :func:`build_full_graph` (or :func:`rewired_graph`),
    or a shape-only dict ``{"n", "n_in", "n_out", "nnz"}`` when the buffers will
    come from ``load_state_dict`` (they are persistent, so a checkpoint is
    self-contained).
    """

    _BUFFERS = ("crow", "col", "val", "crow_t", "col_t", "val_t")

    def __init__(self, hidden_size: int, graph: dict, steps: int = 4):
        super().__init__()
        shapes = graph_shapes(graph)
        self.hidden_size = int(hidden_size)
        self.n = n = shapes["n"]
        self.n_in = shapes["n_in"]
        self.n_out = shapes["n_out"]
        self.steps = int(steps)
        nnz = shapes["nnz"]
        self.register_buffer("crow", torch.zeros(n + 1, dtype=torch.int32))
        self.register_buffer("col", torch.zeros(nnz, dtype=torch.int32))
        self.register_buffer("val", torch.zeros(nnz, dtype=torch.float32))
        self.register_buffer("crow_t", torch.zeros(n + 1, dtype=torch.int32))
        self.register_buffer("col_t", torch.zeros(nnz, dtype=torch.int32))
        self.register_buffer("val_t", torch.zeros(nnz, dtype=torch.float32))
        self.register_buffer("in_idx", torch.zeros(self.n_in, dtype=torch.int64))
        self.register_buffer("out_idx", torch.zeros(self.n_out, dtype=torch.int64))
        self.register_buffer("sign", torch.ones(n, dtype=torch.int8))

        self.in_norm = nn.RMSNorm(self.hidden_size)
        self.in_proj = nn.Linear(self.hidden_size, self.n_in, bias=False)
        self.out_proj = nn.Linear(self.n_out, self.hidden_size, bias=False)
        self.log_gain_in = nn.Parameter(torch.zeros(n))
        self.log_gain_out = nn.Parameter(torch.zeros(n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.leak_logit = nn.Parameter(torch.zeros(n))  # sigmoid(0) = 0.5
        self.gate = nn.Parameter(torch.zeros(self.hidden_size))
        self._csr_cache: Optional[Tuple[Any, torch.Tensor, torch.Tensor]] = None
        if "post" in graph:
            self.set_graph(graph)

    # -- graph buffers -------------------------------------------------------
    @torch.no_grad()
    def set_graph(self, graph: dict) -> Dict[str, int]:
        """(Re)load ``W``/``W^T`` and the in/out index sets from a graph dict.

        Used at construction and for the rewired-null ablation (same ``n``,
        ``n_in``, ``n_out``; ``nnz`` may differ). Parameters are untouched.
        """

        shapes = graph_shapes(graph)
        if (shapes["n"], shapes["n_in"], shapes["n_out"]) != (self.n, self.n_in, self.n_out):
            raise ValueError(f"graph shapes {shapes} do not match the core ({self.n}, {self.n_in}, {self.n_out})")
        n = self.n
        post, pre, value = graph["post"], graph["pre"], graph["value"]
        crow, col, val = csr_arrays(post, pre, value, n)
        crow_t, col_t, val_t = csr_arrays(pre, post, value, n)
        device = self.val.device
        fdtype = self.val.dtype
        self.crow = torch.from_numpy(crow).to(device)
        self.col = torch.from_numpy(col).to(device)
        self.val = torch.from_numpy(val).to(device=device, dtype=fdtype)
        self.crow_t = torch.from_numpy(crow_t).to(device)
        self.col_t = torch.from_numpy(col_t).to(device)
        self.val_t = torch.from_numpy(val_t).to(device=device, dtype=fdtype)
        self.in_idx = torch.as_tensor(np.asarray(graph["in_idx"], dtype=np.int64)).to(device)
        self.out_idx = torch.as_tensor(np.asarray(graph["out_idx"], dtype=np.int64)).to(device)
        self.sign = torch.as_tensor(np.asarray(graph["sign"], dtype=np.int8)).to(device)
        self._csr_cache = None
        return {"n": n, "nnz": int(len(col))}

    def graph_meta(self) -> Dict[str, int]:
        """What a checkpoint needs to rebuild this module's shapes before ``load_state_dict``."""

        return {"n": self.n, "n_in": self.n_in, "n_out": self.n_out, "nnz": int(self.col.numel()),
                "steps": self.steps}

    def _operators(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse CSR ``W`` and ``W^T`` views over the buffers, rebuilt when a buffer changes.

        The key includes each buffer's storage pointer and version counter:
        ``.to()`` swaps the storage, ``load_state_dict`` copies in place and
        bumps the version, and both therefore invalidate the cache.
        """

        bufs = [getattr(self, name) for name in self._BUFFERS]
        key = tuple((t.data_ptr(), t._version, str(t.device), t.dtype, t.numel()) for t in bufs)
        cache = self._csr_cache
        if cache is None or cache[0] != key:
            w = make_csr(self.crow, self.col, self.val, self.n)
            w_t = make_csr(self.crow_t, self.col_t, self.val_t, self.n)
            cache = (key, w, w_t)
            self._csr_cache = cache
        return cache[1], cache[2]

    def _apply(self, fn, *args, **kwargs):  # .to() / .float() / .double() / .cpu()
        self._csr_cache = None
        return super()._apply(fn, *args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._csr_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def __getstate__(self):
        state = super().__getstate__() if hasattr(super(), "__getstate__") else self.__dict__.copy()
        state = dict(state)
        state["_csr_cache"] = None
        return state

    # -- dynamics ------------------------------------------------------------
    def run(self, u_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rates after ``steps`` updates for afferent drive ``u_in`` (M, n_in) -> (r (N, M), e (M, n_out))."""

        w, w_t = self._operators()
        m = u_in.shape[0]
        dtype = self.val.dtype
        u = u_in.new_zeros(self.n, m, dtype=dtype).index_copy(0, self.in_idx, u_in.t().to(dtype))
        a = torch.sigmoid(self.leak_logit).unsqueeze(1)
        g_in = torch.exp(self.log_gain_in).unsqueeze(1)
        g_out = torch.exp(self.log_gain_out).unsqueeze(1)
        base = u + self.bias.unsqueeze(1)
        r = None
        for k in range(self.steps):
            if r is None:  # r_0 is exactly zero: W r_0 = 0, skip the spmm
                drive = base
                r = a * F.relu(drive)
            else:
                drive = g_in * CSRMatmul.apply(g_out * r, w, w_t) + base
                r = (1 - a) * r + a * F.relu(drive)
        if r is None:
            r = u.new_zeros(self.n, m)
        e = r.index_select(0, self.out_idx).t()  # (M, n_out)
        return r, e

    def forward(self, hidden: torch.Tensor, token_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b, t, h = hidden.shape
        flat = hidden.reshape(b * t, h)
        if token_mask is None:
            sel = torch.arange(b * t, device=hidden.device)
        else:
            sel = token_mask.reshape(-1).to(torch.bool).nonzero(as_tuple=False).squeeze(1)
        m = int(sel.numel())
        info: Dict[str, Any] = {"M": m}
        if m == 0:
            return hidden.clone(), info
        if not torch.is_grad_enabled() and not bool(self.gate.detach().any()):
            # Inference with a closed gate: the write is exactly zero, skip the core.
            info["skipped"] = True
            return hidden.clone(), info
        h_sel = flat.index_select(0, sel)
        u_in = self.in_proj(self.in_norm(h_sel))
        r, e = self.run(u_in)
        e_rms = torch.sqrt(e.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        write = self.out_proj((e / e_rms).to(hidden.dtype))
        new = h_sel + self.gate * write
        out = flat.index_copy(0, sel, new).view(b, t, h)
        with torch.no_grad():
            info.update({
                "mean_rate": float(r.mean()),
                "efferent_rms": float(e.pow(2).mean().sqrt()),
                "fraction_active": float((r > 0).float().mean()),
            })
        return out, info


# ---------------------------------------------------------------------------
# Timing and CLI
# ---------------------------------------------------------------------------
def time_core(graph: dict, columns: Sequence[int] = (512, 1024), steps: int = 4, threads: int = 8,
              hidden_size: int = 320, repeats: int = 2, seed: int = 0) -> Dict[str, Any]:
    """Seconds per forward and per backward of the core on the given graph (fp32, CPU)."""

    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    core = FullConnectomeCore(hidden_size, graph, steps=steps)
    with torch.no_grad():
        core.gate.fill_(0.01)
    results: Dict[str, Any] = {"threads": int(threads), "steps": int(steps), "nnz": int(core.col.numel()),
                               "n": core.n, "hidden_size": hidden_size, "repeats": int(repeats), "columns": {}}
    for m in columns:
        hidden = torch.randn(1, int(m), hidden_size, requires_grad=True)
        # warm-up (builds the CSR views, touches the allocator)
        out, _ = core(hidden)
        out.pow(2).mean().backward()
        fwd, bwd = [], []
        for _ in range(repeats):
            core.zero_grad(set_to_none=True)
            hidden.grad = None
            t0 = time.perf_counter()
            out, info = core(hidden)
            t1 = time.perf_counter()
            out.pow(2).mean().backward()
            t2 = time.perf_counter()
            fwd.append(t1 - t0)
            bwd.append(t2 - t1)
        spmm_flops = 2.0 * core.col.numel() * m * (steps - 1)
        results["columns"][str(m)] = {
            "forward_s": round(min(fwd), 4),
            "backward_s": round(min(bwd), 4),
            "forward_s_all": [round(x, 4) for x in fwd],
            "backward_s_all": [round(x, 4) for x in bwd],
            "spmm_gflop_forward": round(spmm_flops / 1e9, 3),
            "spmm_gflops_forward_effective": round(spmm_flops / 1e9 / min(fwd), 2),
            "fraction_active": info.get("fraction_active"),
        }
    return results


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_json(path: os.PathLike, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("graph", help="build the full graph and write its receipt")
    g.add_argument("--npz", default=str(DEFAULT_NPZ))
    g.add_argument("--min_input_fraction", type=float, default=0.0)
    g.add_argument("--radius", type=float, default=0.9)
    g.add_argument("--rewired", action="store_true", help="also build the stratified null and report it")
    g.add_argument("--output", default=str(EXPANSE / "data" / "connectome_full_graph.receipt.json"))
    t = sub.add_parser("time", help="time forward/backward of the core on the full graph")
    t.add_argument("--npz", default=str(DEFAULT_NPZ))
    t.add_argument("--columns", type=int, nargs="+", default=[512, 1024])
    t.add_argument("--steps", type=int, default=4)
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--repeats", type=int, default=2)
    t.add_argument("--output", default=str(EXPANSE / "data" / "connectome_full_timing.json"))
    args = parser.parse_args(argv)

    graph = build_full_graph(args.npz, **({"min_input_fraction": args.min_input_fraction, "radius": args.radius}
                                          if args.command == "graph" else {}))
    if args.command == "graph":
        payload: Dict[str, Any] = {"real": graph["receipt"]}
        if args.rewired:
            payload["rewired"] = rewired_graph(graph, seed=0)["receipt"]
        write_json(args.output, payload)
        print(json.dumps(_jsonable(payload), indent=2))
        return 0
    result = time_core(graph, columns=args.columns, steps=args.steps, threads=args.threads, repeats=args.repeats)
    result["graph"] = {k: graph["receipt"][k] for k in ("n_types", "n_edges", "npz_sha256", "radius", "threshold")}
    result["torch"] = torch.__version__
    result["dtype"] = "float32"
    write_json(args.output, result)
    print(json.dumps(_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
