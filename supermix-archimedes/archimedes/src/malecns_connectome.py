"""Male CNS connectome -> cell-type graph, for wiring Supermix.

Source: Janelia FlyEM male-cns v1.0 flat connectome (CC-BY), the dataset the
natverse/malecns R package reads through neuPrint. The three bulk files live in
``external/malecns_data`` (see ``docs/V91_MALECNS_CONNECTOME.md`` for URLs and
MD5s).

The body-to-body edge file has ~152M rows and is ~1 GB on disk. It is read as a
memory-mapped Arrow IPC file one record batch at a time, so peak RAM stays near
the size of the reduced type-level edge list rather than the raw file. Only
edges whose pre- and post-synaptic bodies are both *typed* neurons survive; the
fraction of synapses kept is recorded, never silently dropped.

Neurotransmitter -> sign follows the convention of the fly whole-brain models
(Shiu et al. 2024, Lappalainen et al. 2024): acetylcholine excitatory; GABA and
glutamate inhibitory (glutamate acts through GluCl in the central brain);
histamine inhibitory (HisCl, photoreceptor output). Monoamines (dopamine,
serotonin, octopamine) are modulatory; they are signed +1 and flagged in
``sign_is_modulatory`` so a consumer can treat them differently.

Usage::

    python source/malecns_connectome.py build \
        --data_dir "../external/malecns_data" \
        --output datasets/v91_malecns/malecns_types.npz

v93 splits the same typed neurons into two hemispheres (node = (type, side))
and writes one block-structured module graph per module count::

    python source/malecns_connectome.py hemispheres \
        --data_dir "../external/malecns_data" \
        --types datasets/v91_malecns/malecns_types.npz \
        --output_dir datasets/v93_malecns --n_modules 256 384
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

SOURCE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
)
EXPECTED_MD5_B64 = {
    ANNOTATIONS: "UKdxh3DFciDxYLpPQxq4ng==",
    NEUROTRANSMITTERS: "PYQrEv5cSe763lKNfdJKHw==",
    WEIGHTS: "8w6dzKJc/QIb8eez2XVZng==",
}

NT_SIGN = {
    "acetylcholine": 1,
    "glutamate": -1,
    "gaba": -1,
    "histamine": -1,
    "dopamine": 1,
    "serotonin": 1,
    "octopamine": 1,
    "unclear": 1,
}
MODULATORY = frozenset({"dopamine", "serotonin", "octopamine"})


def _mode(values: Iterable[Optional[str]], skip: Tuple[str, ...] = ()) -> Optional[str]:
    counts = Counter(v for v in values if isinstance(v, str) and v and v not in skip)
    if not counts:
        return None
    # Most common, ties broken alphabetically so the build is deterministic.
    best = max(counts.values())
    return sorted(k for k, c in counts.items() if c == best)[0]


def _reduce(keys: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uniq, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=weights).astype(np.int64)
    return uniq, sums


def load_types(data_dir: str) -> Dict[str, object]:
    """Typed neurons and one row of metadata per cell type."""
    import pyarrow.feather as pf

    ann = pf.read_table(
        os.path.join(data_dir, ANNOTATIONS),
        columns=["bodyId", "type", "superclass", "class", "flywireType", "mancType"],
    ).to_pandas()
    ann = ann[ann["type"].notna() & (ann["type"].astype(str).str.len() > 0)]
    nts = pf.read_table(
        os.path.join(data_dir, NEUROTRANSMITTERS),
        columns=["body", "consensus_nt", "celltype_predicted_nt"],
    ).to_pandas()
    ann = ann.merge(nts, left_on="bodyId", right_on="body", how="left")

    type_names = sorted(ann["type"].astype(str).unique())
    type_index = {name: i for i, name in enumerate(type_names)}
    body_ids = ann["bodyId"].to_numpy(np.int64)
    body_type = ann["type"].astype(str).map(type_index).to_numpy(np.int32)
    order = np.argsort(body_ids, kind="stable")

    superclass: List[str] = []
    cls: List[str] = []
    nt: List[str] = []
    n_neurons = np.zeros(len(type_names), dtype=np.int32)
    has_flywire = np.zeros(len(type_names), dtype=bool)
    has_manc = np.zeros(len(type_names), dtype=bool)
    for name, group in ann.groupby(ann["type"].astype(str), sort=True):
        i = type_index[name]
        n_neurons[i] = len(group)
        superclass.append(_mode(group["superclass"]) or "unknown")
        cls.append(_mode(group["class"]) or "unknown")
        # Per-neuron consensus first; fall back to the cell-type prediction.
        nt.append(
            _mode(group["consensus_nt"], skip=("unclear",))
            or _mode(group["celltype_predicted_nt"], skip=("unclear",))
            or "unclear"
        )
        has_flywire[i] = bool(group["flywireType"].notna().any())
        has_manc[i] = bool(group["mancType"].notna().any())

    return {
        "type_names": type_names,
        "body_ids_sorted": body_ids[order],
        "body_type_sorted": body_type[order],
        "superclass": superclass,
        "class": cls,
        "nt": nt,
        "n_neurons": n_neurons,
        "has_flywire": has_flywire,
        "has_manc": has_manc,
    }


def aggregate_edges(
    data_dir: str,
    body_ids_sorted: np.ndarray,
    body_type_sorted: np.ndarray,
    n_types: int,
    flush_every: int = 64,
    progress: bool = True,
) -> Dict[str, object]:
    """Stream the body-level edge list and sum synapses per (pre type, post type)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    source = pa.memory_map(os.path.join(data_dir, WEIGHTS), "r")
    reader = ipc.open_file(source)
    n_batches = reader.num_record_batches
    last = len(body_ids_sorted) - 1

    acc_keys = np.zeros(0, dtype=np.int64)
    acc_w = np.zeros(0, dtype=np.int64)
    pending_k: List[np.ndarray] = []
    pending_w: List[np.ndarray] = []
    rows = kept_rows = 0
    total_w = kept_w = 0
    start = time.time()
    for b in range(n_batches):
        batch = reader.get_batch(b)
        pre = batch.column(0).to_numpy()
        post = batch.column(1).to_numpy()
        w = batch.column(2).to_numpy()
        rows += len(w)
        total_w += int(w.sum())
        ip = np.minimum(np.searchsorted(body_ids_sorted, pre), last)
        iq = np.minimum(np.searchsorted(body_ids_sorted, post), last)
        ok = (body_ids_sorted[ip] == pre) & (body_ids_sorted[iq] == post)
        if ok.any():
            tp = body_type_sorted[ip[ok]].astype(np.int64)
            tq = body_type_sorted[iq[ok]].astype(np.int64)
            wk = w[ok].astype(np.int64)
            kept_rows += int(ok.sum())
            kept_w += int(wk.sum())
            k, s = _reduce(tp * n_types + tq, wk)
            pending_k.append(k)
            pending_w.append(s)
        if len(pending_k) >= flush_every or b == n_batches - 1:
            if pending_k:
                acc_keys, acc_w = _reduce(
                    np.concatenate([acc_keys] + pending_k),
                    np.concatenate([acc_w] + pending_w),
                )
                pending_k, pending_w = [], []
            if progress:
                print(
                    f"batch {b + 1}/{n_batches}  rows {rows:,}  kept {kept_rows:,}"
                    f"  type-edges {len(acc_keys):,}  {time.time() - start:.0f}s",
                    flush=True,
                )
    return {
        "pre": (acc_keys // n_types).astype(np.int32),
        "post": (acc_keys % n_types).astype(np.int32),
        "weight": acc_w,
        "body_edge_rows": rows,
        "body_edge_rows_kept": kept_rows,
        "synapses_total": total_w,
        "synapses_kept": kept_w,
    }


def _md5_b64(path: str) -> str:
    import base64

    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def build(data_dir: str, output: str, verify_md5: bool = True) -> Dict[str, object]:
    if verify_md5:
        for name, expected in EXPECTED_MD5_B64.items():
            got = _md5_b64(os.path.join(data_dir, name))
            if got != expected:
                raise SystemExit(f"MD5 mismatch for {name}: {got} != {expected}")
    types = load_types(data_dir)
    n_types = len(types["type_names"])
    edges = aggregate_edges(
        data_dir, types["body_ids_sorted"], types["body_type_sorted"], n_types
    )
    nt = types["nt"]
    sign = np.array([NT_SIGN.get(x, 1) for x in nt], dtype=np.int8)
    modulatory = np.array([x in MODULATORY for x in nt], dtype=bool)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(
        output,
        type_names=np.array(types["type_names"], dtype=object),
        superclass=np.array(types["superclass"], dtype=object),
        cell_class=np.array(types["class"], dtype=object),
        nt=np.array(nt, dtype=object),
        sign=sign,
        sign_is_modulatory=modulatory,
        n_neurons=types["n_neurons"],
        has_flywire=types["has_flywire"],
        has_manc=types["has_manc"],
        pre=edges["pre"],
        post=edges["post"],
        weight=edges["weight"],
    )
    receipt = {
        "schema": "supermix-v91-malecns-types-v1",
        "source": "Janelia FlyEM male-cns v1.0 flat connectome, minconf 0.5 (CC-BY)",
        "source_url": SOURCE_URL,
        "files_md5_b64": EXPECTED_MD5_B64 if verify_md5 else "not verified",
        "typed_neurons": int(len(types["body_ids_sorted"])),
        "cell_types": n_types,
        "type_edges": int(len(edges["weight"])),
        "body_edge_rows": edges["body_edge_rows"],
        "body_edge_rows_kept": edges["body_edge_rows_kept"],
        "synapses_total": edges["synapses_total"],
        "synapses_kept": edges["synapses_kept"],
        "synapse_fraction_kept": edges["synapses_kept"] / max(1, edges["synapses_total"]),
        "nt_counts_by_type": dict(Counter(nt)),
        "superclass_counts_by_type": dict(Counter(types["superclass"])),
        "sign_convention": {k: v for k, v in NT_SIGN.items()},
        "output": output,
    }
    with open(os.path.splitext(output)[0] + ".receipt.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    return receipt


def load(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# Module graphs: 11,751 cell types -> N sign-homogeneous modules
# ---------------------------------------------------------------------------

#: Functional role of a superclass. Modules never mix roles, so a module graph
#: keeps the sensory -> central -> motor layout of the real CNS.
ROLE_OF_SUPERCLASS = {
    "cb_sensory": "sensory", "vnc_sensory": "sensory", "ol_sensory": "sensory",
    "sensory_ascending": "sensory", "sensory_descending": "sensory",
    "ol_intrinsic": "visual", "visual_projection": "visual",
    "visual_centrifugal": "visual", "visual_projection_tbc": "visual",
    "cb_intrinsic": "central",
    "vnc_intrinsic": "vnc",
    "ascending_neuron": "ascending", "efferent_ascending": "ascending",
    "descending_neuron": "descending", "efferent_descending": "descending",
    "cb_motor": "output", "vnc_motor": "output", "cb_efferent": "output",
    "vnc_efferent": "output", "cb_endocrine": "output", "vnc_endocrine": "output",
}
ROLES = ("sensory", "visual", "central", "vnc", "ascending", "descending", "output")


def _allocate(weights: np.ndarray, total: int, caps: np.ndarray) -> np.ndarray:
    """Largest-remainder split of ``total`` slots, at least one each, at most ``caps``."""

    counts = np.ones(len(weights), dtype=np.int64)
    remaining = total - int(counts.sum())
    if remaining < 0:
        raise ValueError(f"{len(weights)} groups cannot fit in {total} modules")
    share = weights / weights.sum() * remaining
    extra = np.minimum(np.floor(share).astype(np.int64), caps - counts)
    counts += extra
    order = np.argsort(-(share - np.floor(share)))
    while counts.sum() < total:
        progressed = False
        for g in order:
            if counts.sum() >= total:
                break
            if counts[g] < caps[g]:
                counts[g] += 1
                progressed = True
        if not progressed:
            raise ValueError("module budget exceeds the number of cell types")
    return counts


def spectral_embedding(n: int, pre: np.ndarray, post: np.ndarray, weight: np.ndarray,
                       dim: int = 32, seed: int = 0) -> np.ndarray:
    """Row-normalised leading eigenvectors of the symmetrised, log-weighted graph."""

    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    w = np.log1p(weight.astype(np.float64))
    a = sp.coo_matrix((w, (post, pre)), shape=(n, n)).tocsr()
    s = (a + a.T).tocsr()
    degree = np.asarray(s.sum(axis=1)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    norm = sp.diags(inv) @ s @ sp.diags(inv)
    rng = np.random.default_rng(seed)
    _, vecs = eigsh(norm, k=dim, which="LA", v0=rng.standard_normal(n))
    vecs = vecs[:, ::-1]
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)


def cluster_modules(graph: Dict[str, np.ndarray], n_modules: int, seed: int = 0,
                    embedding: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Partition every cell type into ``n_modules`` role- and sign-homogeneous modules.

    Types are grouped by (functional role, Dale sign) first, so no module mixes
    excitatory and inhibitory types or sensory and motor ones. Each group gets a
    share of the module budget proportional to sqrt(types x synapse mass), then
    is split by k-means on a spectral embedding of the whole type graph, so a
    module is a set of types with similar wiring.
    """

    from sklearn.cluster import KMeans

    n = len(graph["type_names"])
    pre, post, weight = graph["pre"], graph["post"], graph["weight"]
    if embedding is None:
        embedding = spectral_embedding(n, pre, post, weight, seed=seed)
    mass = np.bincount(pre, weights=weight, minlength=n) + np.bincount(post, weights=weight, minlength=n)
    roles = np.array([ROLE_OF_SUPERCLASS.get(str(s), "central") for s in graph["superclass"]])
    sign = graph["sign"].astype(np.int64)
    keys = sorted({(r, int(s)) for r, s in zip(roles, sign)}, key=lambda k: (ROLES.index(k[0]), -k[1]))
    members = [np.flatnonzero((roles == r) & (sign == s)) for r, s in keys]
    group_mass = np.array([mass[m].sum() + 1.0 for m in members])
    group_size = np.array([len(m) for m in members], dtype=np.int64)
    budget = _allocate(np.sqrt(group_size * group_mass), n_modules, group_size)

    module_of_type = np.full(n, -1, dtype=np.int64)
    module_role: List[str] = []
    module_sign: List[int] = []
    next_id = 0
    for (role, s), idx, k in zip(keys, members, budget):
        if k == 1:
            labels = np.zeros(len(idx), dtype=np.int64)
        else:
            labels = KMeans(n_clusters=int(k), n_init=4, random_state=seed).fit_predict(embedding[idx])
        # KMeans can leave a cluster empty in degenerate cases; renumber densely.
        _, labels = np.unique(labels, return_inverse=True)
        module_of_type[idx] = next_id + labels
        used = int(labels.max()) + 1
        module_role.extend([role] * used)
        module_sign.extend([int(s)] * used)
        next_id += used
    return {
        "module_of_type": module_of_type,
        "module_role": np.array(module_role, dtype=object),
        "module_sign": np.array(module_sign, dtype=np.int8),
        "n_modules": next_id,
        "groups": [
            {"role": r, "sign": int(s), "types": int(sz), "modules": int(k)}
            for (r, s), sz, k in zip(keys, group_size, budget)
        ],
    }


def module_matrix(graph: Dict[str, np.ndarray], module_of_type: np.ndarray, n_modules: int) -> np.ndarray:
    """``M[post, pre]`` = synapses from module ``pre`` onto module ``post``."""

    mp = module_of_type[graph["pre"]]
    mq = module_of_type[graph["post"]]
    flat = np.bincount(mq * n_modules + mp, weights=graph["weight"].astype(np.float64),
                       minlength=n_modules * n_modules)
    return flat.reshape(n_modules, n_modules)


def threshold_input_fraction(matrix: np.ndarray, min_fraction: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep edges carrying at least ``min_fraction`` of the post module's input.

    The 1% input-fraction cut is the convention FlyWire/hemibrain analyses use
    for a "strong" connection. Returns (post, pre, fraction) for the edges kept.
    """

    total_in = matrix.sum(axis=1, keepdims=True)
    fraction = np.divide(matrix, total_in, out=np.zeros_like(matrix), where=total_in > 0)
    post, pre = np.nonzero(fraction >= min_fraction)
    return post.astype(np.int64), pre.astype(np.int64), fraction[post, pre]


def degree_preserving_rewire(post: np.ndarray, pre: np.ndarray, n: int, swaps_per_edge: int = 10,
                             seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Maslov-Sneppen rewiring of a directed graph.

    Repeatedly picks edges ``a->b`` and ``c->d`` and swaps their targets to
    ``a->d`` and ``c->b``, rejecting swaps that would create a duplicate edge or
    a self-loop. Every node keeps its exact in- and out-degree, and because a
    module's Dale sign belongs to the presynaptic side, every edge keeps its
    sign too. Edge weights travel with the presynaptic end, so each module's
    out-strength distribution is also preserved. What is destroyed is *which*
    module talks to which -- the thing the connectome arm claims matters.
    """

    rng = np.random.default_rng(seed)
    post = post.copy()
    pre = pre.copy()
    present = np.zeros((n, n), dtype=bool)
    present[post, pre] = True
    # Self-loops (a module's recurrence onto itself) are held fixed. Swapping
    # them away would hand the null model a second, unintended difference --
    # no within-module recurrence at all -- and the comparison is supposed to
    # isolate *between*-module wiring.
    movable = np.flatnonzero(post != pre)
    m = len(movable)
    attempts = accepted = 0
    target = swaps_per_edge * m
    while accepted < target and attempts < 20 * target:
        attempts += 1
        i, j = movable[rng.integers(0, m, size=2)]
        a, b = pre[i], post[i]
        c, d = pre[j], post[j]
        if i == j or a == c or b == d:
            continue
        if a == d or c == b:
            continue
        if present[d, a] or present[b, c]:
            continue
        present[b, a] = False
        present[d, c] = False
        present[d, a] = True
        present[b, c] = True
        post[i], post[j] = d, b
        accepted += 1
    return post, pre, {
        "swaps_accepted": int(accepted),
        "swap_attempts": int(attempts),
        "edges_rewired": int(m),
        "self_loops_held": int(len(post) - m),
    }


def stratified_rewire(post: np.ndarray, pre: np.ndarray, fraction: np.ndarray, strata: Optional[np.ndarray],
                      n: int, swaps_per_edge: int = 10, seed: int = 0, *,
                      edge_strata: Optional[np.ndarray] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """The v91 null: Maslov-Sneppen swaps restricted to same-stratum sources.

    Two edges ``a->b`` and ``c->d`` may swap targets only when ``a`` and ``c``
    share a stratum (functional role x Dale sign), and each edge's input
    fraction travels with its **target**, because a fraction is a share of the
    postsynaptic module's input. So every module keeps its in- and out-degree,
    its self-loop, the role and sign of every one of its inputs, and its exact
    multiset of input fractions (row sums unchanged). What is destroyed is
    *which* same-kind module a given input comes from -- the fine wiring,
    reciprocity and clustering.

    The unrestricted :func:`degree_preserving_rewire` carried fractions with
    the source instead, which left 116 of 512 modules with input fractions
    summing above 1 (max 1.95), changed the inhibitory-input count of 429,
    and gave the null a 3.3x larger input-to-output gain -- differences far
    beyond "which module talks to which" (v91 validity review, 2026-09-19).

    ``edge_strata`` (keyword-only, v93) gives every *edge* its own stratum
    instead of deriving it from the source module alone. The two-hemisphere
    null needs this: with node strata (side, role, sign) an L->L edge could
    swap targets with an L->R edge from a same-kind source, moving edges
    between the ipsilateral and commissural blocks and changing each module's
    commissural input count. With the edge stratum (side_pre, role_pre,
    sign_pre, side_post) an LL edge swaps only with an LL edge, LR only with
    LR, so the four block counts and every module's count of inputs from
    each (side, role, sign) are preserved too. When it is given, ``strata``
    may be ``None``. Existing positional callers are unchanged.
    """

    import random

    if strata is None and edge_strata is None:
        raise ValueError("stratified_rewire needs node strata or edge_strata")
    rng = random.Random(seed)
    post_l = post.astype(np.int64).tolist()
    pre_l = pre.astype(np.int64).tolist()
    frac_l = fraction.astype(np.float64).tolist()
    present = set(zip(post_l, pre_l))
    movable = [i for i in range(len(post_l)) if post_l[i] != pre_l[i]]
    by_stratum: Dict[int, List[int]] = {}
    for i in movable:
        stratum = int(edge_strata[i]) if edge_strata is not None else int(strata[pre_l[i]])
        by_stratum.setdefault(stratum, []).append(i)
    groups = [g for g in by_stratum.values() if len(g) > 1]
    sizes = [len(g) for g in groups]
    target = swaps_per_edge * len(movable)
    accepted = attempts = 0
    while accepted < target and attempts < 40 * target:
        attempts += 1
        group = rng.choices(groups, sizes)[0]
        i, j = rng.sample(group, 2)
        a, b, c, d = pre_l[i], post_l[i], pre_l[j], post_l[j]
        if a == c or b == d or a == d or c == b:
            continue
        if (d, a) in present or (b, c) in present:
            continue
        present.discard((b, a))
        present.discard((d, c))
        present.add((d, a))
        present.add((b, c))
        post_l[i], post_l[j] = d, b
        frac_l[i], frac_l[j] = frac_l[j], frac_l[i]
        accepted += 1
    return (
        np.array(post_l, dtype=np.int64),
        np.array(pre_l, dtype=np.int64),
        np.array(frac_l, dtype=fraction.dtype),
        {
            "kind": ("stratified_by_edge_strata_fraction_carried_with_target" if edge_strata is not None
                     else "stratified_by_source_role_and_sign_fraction_carried_with_target"),
            "swaps_accepted": int(accepted),
            "swap_attempts": int(attempts),
            "edges_rewired": int(len(movable)),
            "self_loops_held": int(len(post_l) - len(movable)),
            "strata": int(len(by_stratum)),
        },
    )


def mean_hops(post: np.ndarray, pre: np.ndarray, roles: np.ndarray, n: int,
              source_roles: Tuple[str, ...], sink_roles: Tuple[str, ...]) -> Dict[str, Optional[float]]:
    """Mean shortest synaptic-direction path from source-role to sink-role modules."""

    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path

    # csgraph reads adj[i, j] as an edge i -> j, so rows are the PRE side.
    # (v91's first receipt built it as [post, pre] and walked every edge
    # backwards: it reported 3.03 / 2.60 where the true values are 2.81 / 2.38.)
    adj = sp.coo_matrix((np.ones(len(post)), (pre, post)), shape=(n, n)).tocsr()
    src = np.flatnonzero(np.isin(roles, source_roles))
    dst = np.flatnonzero(np.isin(roles, sink_roles))
    if not len(src) or not len(dst):
        return {"mean": None, "reachable_fraction": None}
    dist = shortest_path(adj, indices=src, unweighted=True)[:, dst]
    finite = dist[np.isfinite(dist)]
    return {
        "mean": float(finite.mean()) if finite.size else None,
        "reachable_fraction": float(finite.size / dist.size),
    }


def graph_diagnostics(post: np.ndarray, pre: np.ndarray, fraction: np.ndarray, sign: np.ndarray,
                      roles: np.ndarray, n: int, reference: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      spectral_radius: float = 0.9, leak: float = 0.5, steps: int = 6) -> Dict[str, object]:
    """Everything a null is supposed to match or break, for the receipt."""

    dense = np.zeros((n, n))
    dense[post, pre] = fraction
    mask = dense > 0
    rho = float(np.max(np.abs(np.linalg.eigvals(dense))))
    scale = spectral_radius / rho if rho > 0 else 1.0
    signed = scale * dense * sign[None, :]
    eig = np.linalg.eigvals(signed)
    off = ~np.eye(n, dtype=bool)
    undirected = ((mask | mask.T) & off).astype(float)
    degree = undirected.sum(1)
    triangles = np.diag(undirected @ undirected @ undirected)
    clustering = np.divide(triangles, degree * (degree - 1), out=np.zeros(n), where=degree > 1).mean()
    total = undirected.sum()
    modularity = sum(
        undirected[np.ix_(roles == r, roles == r)].sum() / total - (degree[roles == r].sum() / total) ** 2
        for r in ROLES
    )
    in_mask = np.isin(roles, ("sensory", "visual", "ascending"))
    out_mask = np.isin(roles, ("descending", "output"))
    step = (1 - leak) * np.eye(n) + leak * signed
    gain = np.zeros((n, n))
    power = np.eye(n)
    for _ in range(steps):
        gain += leak * power
        power = step @ power
    io = gain[np.ix_(out_mask, in_mask)]
    report: Dict[str, object] = {
        "edges": int(mask.sum()),
        "abs_spectral_radius_unscaled": round(rho, 6),
        "scale_to_abs_radius": round(scale, 6),
        "signed_spectral_radius": round(float(np.abs(eig).max()), 6),
        "signed_max_real_eigenvalue": round(float(eig.real.max()), 6),
        "input_fraction_row_sum_max": round(float(dense.sum(1).max()), 6),
        "rows_over_one": int((dense.sum(1) > 1 + 1e-9).sum()),
        "reciprocity": round(float((mask & mask.T & off).sum() / max(1, (mask & off).sum())), 6),
        "clustering": round(float(clustering), 6),
        "role_modularity": round(float(modularity), 6),
        "linear_io_gain_frobenius": round(float(np.linalg.norm(io)), 6),
        "hops_afferent_to_efferent": mean_hops(post, pre, roles, n, ("sensory",), ("descending", "output")),
    }
    if reference is not None:
        r_post, r_pre = reference
        ref = np.zeros((n, n), dtype=bool)
        ref[r_post, r_pre] = True
        report["edge_overlap_with_real"] = round(float((mask & ref).sum() / ref.sum()), 6)
        inhib = sign[None, :] < 0
        report["modules_with_changed_inhibitory_input_count"] = int(
            ((mask & inhib).sum(1) != (ref & inhib).sum(1)).sum()
        )
        blocks = [[(mask[np.ix_(roles == q, roles == p)]).sum() - (ref[np.ix_(roles == q, roles == p)]).sum()
                   for p in ROLES] for q in ROLES]
        report["role_block_edge_count_l1_diff"] = int(np.abs(np.array(blocks)).sum())
    return report


def add_stratified_null(path: str, seed: int = 1) -> Dict[str, object]:
    """Install the stratified null into an existing module npz, in place.

    The connectome arrays (``edge_*``, module metadata) are copied through
    byte-for-byte: a model already training on this file must see exactly the
    wiring it started with. The old degree-only null is kept under
    ``unstratified_rewired_*`` for provenance. The write is atomic.
    """

    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    n = len(arrays["module_sign"])
    roles = np.array([str(r) for r in arrays["module_role"]])
    sign = arrays["module_sign"].astype(np.int64)
    strata_names = sorted({(r, int(s)) for r, s in zip(roles, sign)})
    stratum_of = {k: i for i, k in enumerate(strata_names)}
    strata = np.array([stratum_of[(r, int(s))] for r, s in zip(roles, sign)])
    post, pre, fraction = arrays["edge_post"], arrays["edge_pre"], arrays["edge_fraction"]
    if "unstratified_rewired_post" not in arrays:
        arrays["unstratified_rewired_post"] = arrays["rewired_post"]
        arrays["unstratified_rewired_pre"] = arrays["rewired_pre"]
    r_post, r_pre, r_fraction, info = stratified_rewire(post, pre, fraction, strata, n, seed=seed)
    arrays["rewired_post"], arrays["rewired_pre"], arrays["rewired_fraction"] = r_post, r_pre, r_fraction
    arrays["null_kind"] = np.array(info["kind"], dtype=object)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    sign_f = sign.astype(np.float64)
    diagnostics = {
        "real": graph_diagnostics(post, pre, fraction, sign_f, roles, n),
        "stratified_null": graph_diagnostics(r_post, r_pre, r_fraction, sign_f, roles, n, reference=(post, pre)),
        "unstratified_null": graph_diagnostics(
            arrays["unstratified_rewired_post"], arrays["unstratified_rewired_pre"], fraction, sign_f, roles, n,
            reference=(post, pre),
        ),
    }
    return {"rewire": info, "seed": seed, "diagnostics": diagnostics}


def build_modules(types_path: str, output: str, n_modules: int, min_fraction: float,
                  seed: int = 0, embedding: Optional[np.ndarray] = None) -> Dict[str, object]:
    graph = load(types_path)
    clusters = cluster_modules(graph, n_modules, seed=seed, embedding=embedding)
    n = int(clusters["n_modules"])
    matrix = module_matrix(graph, clusters["module_of_type"], n)
    post, pre, fraction = threshold_input_fraction(matrix, min_fraction)
    r_post, r_pre, rewire = degree_preserving_rewire(post, pre, n, seed=seed + 1)

    sign = clusters["module_sign"].astype(np.int64)
    n_types = np.bincount(clusters["module_of_type"], minlength=n)
    n_neurons = np.bincount(clusters["module_of_type"], weights=graph["n_neurons"], minlength=n)
    # A readable label per module: its biggest member type.
    label = []
    for mod in range(n):
        idx = np.flatnonzero(clusters["module_of_type"] == mod)
        best = idx[np.argmax(graph["n_neurons"][idx])]
        label.append(f"{clusters['module_role'][mod]}:{graph['type_names'][best]}")

    density = len(post) / float(n * n)
    np.savez_compressed(
        output,
        module_of_type=clusters["module_of_type"],
        module_role=clusters["module_role"],
        module_sign=clusters["module_sign"],
        module_label=np.array(label, dtype=object),
        module_n_types=n_types,
        module_n_neurons=n_neurons,
        synapse_matrix=matrix.astype(np.float32),
        edge_post=post, edge_pre=pre, edge_fraction=fraction.astype(np.float32),
        rewired_post=r_post, rewired_pre=r_pre,
    )
    receipt = {
        "schema": "supermix-v91-malecns-modules-v1",
        "types": types_path,
        "n_modules": n,
        "min_input_fraction": min_fraction,
        "edges": int(len(post)),
        "density": round(density, 5),
        "self_loops": int((post == pre).sum()),
        "rewired_self_loops": int((r_post == r_pre).sum()),
        "rewire": rewire,
        "excitatory_modules": int((sign > 0).sum()),
        "inhibitory_modules": int((sign < 0).sum()),
        "groups": clusters["groups"],
        "seed": seed,
        "output": output,
    }
    # Replaces the degree-only null above (kept as unstratified_rewired_*)
    # with the stratified one, and records what each null matches.
    receipt["null"] = add_stratified_null(output, seed=seed + 1)
    with open(os.path.splitext(output)[0] + ".receipt.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    return receipt


# ---------------------------------------------------------------------------
# Two hemispheres (v93): (type, side) nodes, mirror modules, block-aware null
# ---------------------------------------------------------------------------

#: Side codes. L is 0 so the left hemisphere holds module ids 0..N-1 and the
#: right N..2N-1, which keeps `ConnectomeCore.load_graph` (one flat id space)
#: unchanged.
SIDE_NAMES = ("L", "R")
#: Blocks of the module graph, keyed by (side of pre, side of post). LL and
#: RR are ipsilateral (inside one brain); LR (pre left, post right) and RL
#: are the commissural bonds between the two.
BLOCK_NAMES = ("LL", "RR", "LR", "RL")
HEMISPHERE_NULL_KIND = "stratified_by_source_side_role_sign_and_target_side_fraction_carried_with_target"
SIDED_TYPES_NAME = "malecns_sided_types.npz"


def block_of(side_pre: np.ndarray, side_post: np.ndarray) -> np.ndarray:
    """Block code (index into :data:`BLOCK_NAMES`) of every edge, int8."""

    side_pre = np.asarray(side_pre, dtype=np.int64)
    side_post = np.asarray(side_post, dtype=np.int64)
    return np.where(side_pre == side_post, side_pre, 2 + side_pre).astype(np.int8)


def assign_side(soma_side: Iterable[object], root_side: Iterable[object]) -> Dict[str, np.ndarray]:
    """Hemisphere of every body: code 0 L, 1 R, 2 midline, 3 unknown.

    ``somaSide`` wins when it is L or R. Sensory neurons have their soma in the
    periphery and no ``somaSide`` (16,964 typed bodies), but 16,551 of them
    carry an L/R ``rootSide`` -- the side their axon enters the CNS -- so that
    is the fallback. Without it the whole sensory periphery (ol_sensory 6,062,
    vnc_sensory 5,603, cb_sensory 4,756, sensory_ascending 528 bodies), i.e. the
    afferent modules the core reads text through, would vanish. Midline ``M``
    bodies (375, in 161 types that have no L/R member at all) belong to neither
    hemisphere; they and the 413 bodies with no usable side are dropped and
    counted by the caller, never folded into one side.

    Returns ``side`` (int8), ``fallback`` (rootSide decided the side) and
    ``conflict`` (somaSide and rootSide are both L/R and disagree; somaSide
    wins -- one body in v1.0).
    """

    soma = np.asarray([v if isinstance(v, str) else "" for v in soma_side], dtype=object)
    root = np.asarray([v if isinstance(v, str) else "" for v in root_side], dtype=object)
    side = np.full(len(soma), 3, dtype=np.int8)
    fallback = np.zeros(len(soma), dtype=bool)
    for code, name in enumerate(SIDE_NAMES):
        side[soma == name] = code
    for code, name in enumerate(SIDE_NAMES):
        hit = (side == 3) & (root == name)
        side[hit] = code
        fallback |= hit
    side[(side == 3) & ((soma == "M") | (root == "M"))] = 2
    lateral = np.isin(soma, SIDE_NAMES) & np.isin(root, SIDE_NAMES)
    conflict = lateral & (soma != root)
    return {"side": side, "fallback": fallback, "conflict": conflict}


def load_sided_nodes(data_dir: str, type_names: List[str]) -> Dict[str, object]:
    """(type, side) nodes over the typed bodies, with type identity fixed by ``type_names``.

    Per-type metadata (superclass, class, NT, sign) is deliberately *not*
    recomputed here: a type's mode over its left bodies could differ from its
    mode over its right bodies, and the modules are clustered on the v91 type
    graph, so everything type-level is copied by name from ``malecns_types.npz``
    by the caller. This function only decides which side each body belongs to,
    numbers the (type, side) nodes, and counts the bodies that belong to
    neither side.

    Bodies that are dropped still get an id for the edge stream: two
    pseudo-nodes ``n_nodes`` (midline) and ``n_nodes + 1`` (unknown side), so
    the synapses lost by dropping them are counted exactly in the same pass
    and then removed.
    """

    import pyarrow.feather as pf

    ann = pf.read_table(
        os.path.join(data_dir, ANNOTATIONS),
        columns=["bodyId", "type", "superclass", "somaSide", "rootSide"],
    ).to_pandas()
    ann = ann[ann["type"].notna() & (ann["type"].astype(str).str.len() > 0)]
    type_index = {name: i for i, name in enumerate(type_names)}
    names = ann["type"].astype(str)
    missing = sorted(set(names.unique()) - set(type_index))
    if missing:
        raise ValueError(f"{len(missing)} annotation types are not in the type table, e.g. {missing[:5]}")
    body_type = names.map(type_index).to_numpy(np.int64)
    body_ids = ann["bodyId"].to_numpy(np.int64)
    superclass = np.array([s if isinstance(s, str) else "unknown" for s in ann["superclass"]], dtype=object)
    sides = assign_side(ann["somaSide"].tolist(), ann["rootSide"].tolist())
    side, fallback = sides["side"], sides["fallback"]

    kept = side < 2
    keys = body_type[kept] * 2 + side[kept]
    uniq, inverse = np.unique(keys, return_inverse=True)
    node_type = (uniq // 2).astype(np.int32)
    node_side = (uniq % 2).astype(np.int8)
    n_nodes = int(len(uniq))
    node_n_neurons = np.bincount(inverse, minlength=n_nodes).astype(np.int32)
    body_node = np.empty(len(body_ids), dtype=np.int64)
    body_node[kept] = inverse
    body_node[~kept] = n_nodes + (side[~kept].astype(np.int64) - 2)
    order = np.argsort(body_ids, kind="stable")

    n_types = len(type_names)
    on_side = np.zeros((n_types, 2), dtype=bool)
    on_side[node_type, node_side] = True
    midline_types = np.unique(body_type[side == 2])
    counts = {
        "typed_neurons": int(len(body_ids)),
        "bodies_by_side": {
            "L": int((side == 0).sum()), "R": int((side == 1).sum()),
            "M": int((side == 2).sum()), "unknown": int((side == 3).sum()),
        },
        "root_side_fallbacks": int(fallback.sum()),
        "root_side_fallbacks_by_side": {
            "L": int((fallback & (side == 0)).sum()), "R": int((fallback & (side == 1)).sum()),
        },
        "root_side_fallbacks_by_superclass": dict(Counter(superclass[fallback].tolist())),
        "soma_root_side_conflicts_soma_wins": int(sides["conflict"].sum()),
        "dropped_midline_bodies": int((side == 2).sum()),
        "dropped_midline_types": int(len(midline_types)),
        "dropped_midline_types_without_sided_bodies": int((~on_side[midline_types].any(1)).sum()),
        "dropped_midline_bodies_by_superclass": dict(Counter(superclass[side == 2].tolist())),
        "dropped_unknown_side_bodies": int((side == 3).sum()),
        "sided_nodes": n_nodes,
        "nodes_by_side": {"L": int((node_side == 0).sum()), "R": int((node_side == 1).sum())},
        "types_on_both_sides": int(on_side.all(1).sum()),
        "types_left_only": int((on_side[:, 0] & ~on_side[:, 1]).sum()),
        "types_right_only": int((on_side[:, 1] & ~on_side[:, 0]).sum()),
        "types_without_sided_bodies": int((~on_side.any(1)).sum()),
        "single_body_nodes": int((node_n_neurons == 1).sum()),
        "median_bodies_per_node": float(np.median(node_n_neurons)),
    }
    return {
        "node_type": node_type,
        "node_side": node_side,
        "node_n_neurons": node_n_neurons,
        "n_nodes": n_nodes,
        "body_ids_sorted": body_ids[order],
        "body_node_sorted": body_node[order],
        "counts": counts,
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sided_types(data_dir: str, types_path: str, output: str, verify_md5: bool = True) -> Dict[str, object]:
    """The sided type graph: one streaming pass over the edge file with (type, side) nodes.

    Everything type-level is copied from ``types_path`` (the v91
    ``malecns_types.npz``) by type name, so module labels, NT signs and roles
    stay comparable with v91. The stream reuses :func:`aggregate_edges`
    unchanged over ``n_nodes + 2`` ids: the two extra ids are the dropped
    midline and unknown-side bodies, whose synapses are tallied for the
    receipt and then cut from the edge list.
    """

    started = time.time()
    if verify_md5:
        for name, expected in EXPECTED_MD5_B64.items():
            got = _md5_b64(os.path.join(data_dir, name))
            if got != expected:
                raise SystemExit(f"MD5 mismatch for {name}: {got} != {expected}")
    types = load(types_path)
    type_names = [str(x) for x in types["type_names"]]
    nodes = load_sided_nodes(data_dir, type_names)
    n = int(nodes["n_nodes"])
    stream_started = time.time()
    edges = aggregate_edges(data_dir, nodes["body_ids_sorted"], nodes["body_node_sorted"], n + 2)
    stream_seconds = time.time() - stream_started

    pre = edges["pre"].astype(np.int64)
    post = edges["post"].astype(np.int64)
    weight = edges["weight"]
    # Endpoint class: 0 sided node, 1 midline pseudo-node, 2 unknown-side pseudo-node.
    cls_pre = np.clip(pre - n + 1, 0, 2)
    cls_post = np.clip(post - n + 1, 0, 2)
    table = np.bincount(cls_pre * 3 + cls_post, weights=weight, minlength=9).astype(np.int64).reshape(3, 3)
    real = (cls_pre == 0) & (cls_post == 0)
    pre, post, weight = pre[real], post[real], weight[real]
    node_side = nodes["node_side"]
    block = block_of(node_side[pre], node_side[post])
    block_synapses = np.bincount(block, weights=weight, minlength=4).astype(np.int64)
    block_edges = np.bincount(block, minlength=4)
    sided_synapses = int(weight.sum())

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(
        output,
        type_names=types["type_names"],
        superclass=types["superclass"],
        cell_class=types["cell_class"],
        nt=types["nt"],
        sign=types["sign"],
        sign_is_modulatory=types["sign_is_modulatory"],
        n_neurons=types["n_neurons"],
        node_type=nodes["node_type"],
        node_side=node_side,
        node_n_neurons=nodes["node_n_neurons"],
        pre=pre.astype(np.int32),
        post=post.astype(np.int32),
        weight=weight,
    )
    classes = ("sided", "midline", "unknown_side")
    counts = dict(nodes["counts"])
    receipt = {
        "schema": "supermix-v93-malecns-sided-types-v1",
        "source": "Janelia FlyEM male-cns v1.0 flat connectome, minconf 0.5 (CC-BY)",
        "source_url": SOURCE_URL,
        "files_md5_b64": EXPECTED_MD5_B64 if verify_md5 else "not verified",
        "types": types_path,
        "types_sha256": _sha256(types_path),
        "side_rule": "somaSide if in {L,R} else rootSide if in {L,R} else dropped (M counted as midline)",
        **counts,
        "cell_types": len(type_names),
        "sided_edges": int(len(weight)),
        "sided_edges_by_block": {name: int(block_edges[i]) for i, name in enumerate(BLOCK_NAMES)},
        "body_edge_rows": edges["body_edge_rows"],
        "body_edge_rows_kept_typed": edges["body_edge_rows_kept"],
        "synapses_total": edges["synapses_total"],
        "synapses_typed_typed": edges["synapses_kept"],
        "synapses_sided": sided_synapses,
        "synapse_fraction_sided_of_total": sided_synapses / max(1, edges["synapses_total"]),
        "synapse_fraction_sided_of_typed": sided_synapses / max(1, edges["synapses_kept"]),
        "synapses_by_block": {name: int(block_synapses[i]) for i, name in enumerate(BLOCK_NAMES)},
        "synapse_share_by_block": {
            name: round(float(block_synapses[i]) / max(1, sided_synapses), 6) for i, name in enumerate(BLOCK_NAMES)
        },
        "synapses_by_endpoint_class": {
            f"{classes[i]}->{classes[j]}": int(table[i, j]) for i in range(3) for j in range(3)
        },
        "synapses_dropped_midline": int(table[1, :].sum() + table[:, 1].sum() - table[1, 1]),
        "synapses_dropped_unknown_side": int(table[2, :].sum() + table[:, 2].sum() - table[2, 2]),
        "synapses_dropped_total": int(edges["synapses_kept"] - sided_synapses),
        "seconds": {"stream": round(stream_seconds, 1), "total": round(time.time() - started, 1)},
        "output": output,
    }
    with open(os.path.splitext(output)[0] + ".receipt.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    return receipt


def hemisphere_strata(module_side: np.ndarray, module_role: np.ndarray, module_sign: np.ndarray
                      ) -> Tuple[np.ndarray, List[Tuple[int, str, int]]]:
    """Node stratum id of every module under (side, role, sign), and the key list."""

    triples = [(int(a), str(b), int(c)) for a, b, c in zip(module_side, module_role, module_sign)]
    keys = sorted(set(triples))
    index = {k: i for i, k in enumerate(keys)}
    return np.array([index[t] for t in triples], dtype=np.int64), keys


def hemisphere_edge_strata(post: np.ndarray, pre: np.ndarray, module_side: np.ndarray,
                           module_role: np.ndarray, module_sign: np.ndarray) -> np.ndarray:
    """Per-edge stratum (side_pre, role_pre, sign_pre, side_post) for :func:`stratified_rewire`.

    The stratum id is ``node_stratum[pre] * 2 + side[post]``; two edges share
    it exactly when their sources are same-side, same-role, same-sign and
    their targets are on the same side, so a swap never moves an edge between
    the LL / RR / LR / RL blocks.
    """

    node_stratum, _ = hemisphere_strata(module_side, module_role, module_sign)
    side = np.asarray(module_side, dtype=np.int64)
    return node_stratum[np.asarray(pre)] * len(SIDE_NAMES) + side[np.asarray(post)]


def hemisphere_diagnostics(post: np.ndarray, pre: np.ndarray, fraction: np.ndarray, roles: np.ndarray,
                           module_side: np.ndarray, reference: Optional[Tuple[np.ndarray, np.ndarray]] = None
                           ) -> Dict[str, object]:
    """Side-aware companions to :func:`graph_diagnostics` for a mirror-scheme graph.

    ``graph_diagnostics`` treats both hemispheres as one population, so it
    cannot tell whether a null scrambled laterality. This reports what only a
    two-brain graph has: per-block edge counts, self-loops and density; side
    modularity (the role-modularity formula over the side label); homolog
    edge symmetry (the share of left edges whose mirror ``(i+N, j+N)`` exists
    on the right, ipsilateral and commissural separately) and the correlation
    of mirrored fractions; each module's commissural input share; the
    spectral radius of |W| for each side alone and for the joint matrix, so
    the receipt shows how much recurrence the bonds add; and shortest paths
    from left sensory modules to left versus right efferents. With
    ``reference`` it also reports the L1 difference of block counts and how
    many modules changed their commissural in-degree -- both must be 0 for
    the block-aware null.
    """

    side = np.asarray(module_side, dtype=np.int64)
    n2 = len(side)
    n = n2 // 2
    post = np.asarray(post, dtype=np.int64)
    pre = np.asarray(pre, dtype=np.int64)
    fraction = np.asarray(fraction, dtype=np.float64)
    dense = np.zeros((n2, n2))
    dense[post, pre] = fraction
    mask = dense > 0
    block = block_of(side[pre], side[post])
    n_side = np.bincount(side, minlength=2)
    pairs = ((0, 0), (1, 1), (0, 1), (1, 0))
    blocks: Dict[str, object] = {}
    for code, name in enumerate(BLOCK_NAMES):
        sel = block == code
        s_pre, s_post = pairs[code]
        blocks[name] = {
            "edges": int(sel.sum()),
            "self_loops": int((sel & (post == pre)).sum()),
            "density": round(float(sel.sum()) / max(1, int(n_side[s_pre]) * int(n_side[s_post])), 6),
            "fraction_mass": round(float(fraction[sel].sum()), 4),
        }
    off = ~np.eye(n2, dtype=bool)
    undirected = ((mask | mask.T) & off).astype(float)
    degree = undirected.sum(1)
    total = max(undirected.sum(), 1.0)
    side_modularity = sum(
        undirected[np.ix_(side == s, side == s)].sum() / total - (degree[side == s].sum() / total) ** 2
        for s in range(len(SIDE_NAMES))
    )
    mirror_post = np.where(post < n, post + n, post - n)
    mirror_pre = np.where(pre < n, pre + n, pre - n)
    mirrored = mask[mirror_post, mirror_pre]
    ipsi = block <= 1
    contra = block >= 2

    def share(sel: np.ndarray) -> Optional[float]:
        return round(float(mirrored[sel].mean()), 6) if sel.any() else None

    if mirrored.sum() > 1:
        corr = float(np.corrcoef(fraction[mirrored], dense[mirror_post[mirrored], mirror_pre[mirrored]])[0, 1])
    else:
        corr = None
    contra_in = np.bincount(post[contra], weights=fraction[contra], minlength=n2)
    has_input = dense.sum(1) > 0

    def radius(m: np.ndarray) -> float:
        return round(float(np.max(np.abs(np.linalg.eigvals(m)))), 6) if m.size else 0.0

    sided_roles = np.array([f"{r}@{SIDE_NAMES[s]}" for r, s in zip(roles, side)])
    report: Dict[str, object] = {
        "blocks": blocks,
        "side_modularity": round(float(side_modularity), 6),
        "homolog_edge_symmetry": {"all": share(np.ones(len(post), dtype=bool)), "ipsilateral": share(ipsi),
                                  "commissural": share(contra)},
        "homolog_fraction_correlation": None if corr is None else round(corr, 6),
        "commissural_input_share": {
            "mean_over_modules_with_input": round(float(contra_in[has_input].mean()), 6) if has_input.any() else None,
            "max": round(float(contra_in.max()), 6),
            "modules_with_input": int(has_input.sum()),
            "modules_without_commissural_input": int((has_input & (contra_in == 0)).sum()),
        },
        "abs_spectral_radius": {"left": radius(dense[:n, :n]), "right": radius(dense[n:, n:]), "joint": radius(dense)},
        "hops_left_sensory_to_left_efferent": mean_hops(
            post, pre, sided_roles, n2, ("sensory@L",), ("descending@L", "output@L")),
        "hops_left_sensory_to_right_efferent": mean_hops(
            post, pre, sided_roles, n2, ("sensory@L",), ("descending@R", "output@R")),
    }
    if reference is not None:
        r_post, r_pre = (np.asarray(x, dtype=np.int64) for x in reference)
        r_block = block_of(side[r_pre], side[r_post])
        r_contra = r_block >= 2
        report["side_block_edge_count_l1_diff"] = int(
            np.abs(np.bincount(block, minlength=4) - np.bincount(r_block, minlength=4)).sum())
        report["modules_with_changed_commissural_input_count"] = int((
            np.bincount(post[contra], minlength=n2) != np.bincount(r_post[r_contra], minlength=n2)).sum())
    return report


def build_hemispheres(types_path: str, sided_path: str, output: str, n_per_side: int, min_fraction: float,
                      seed: int = 0, embedding: Optional[np.ndarray] = None) -> Dict[str, object]:
    """One block-structured module graph over two hemispheres (the v93 mirror scheme).

    The type graph is clustered once into ``n_per_side`` modules with the v91
    :func:`cluster_modules` (same strata, same seed, so at 512 the partition is
    v91's), and every sided node goes to ``module_of_type[type] + N * side``:
    left module ``i`` and right module ``i`` hold the same cell types and are
    homologous by construction. Clustering each side separately would give
    non-homologous partitions with no natural diagonal for the commissural
    blocks and a null that must rewire cross edges blind.

    Edges come from one ``module_matrix`` over the ``2N`` id space, thresholded
    once at ``min_fraction`` of a module's **total** (ipsilateral plus
    commissural) input, then labelled by block. Thresholding each block
    against its own row sums would inflate commissural fractions about 5x
    (they carry ~21% of input) and push row sums above 1, the defect the v91
    validity review found in the unstratified null.

    A module whose types all live on one side is empty on the other: its slot
    is kept with zero rows and columns and counted in the receipt, so left and
    right ids stay aligned.

    The npz carries everything ``ConnectomeCore.load_graph`` reads today
    (``module_sign``, ``module_role``, ``edge_*``, ``rewired_*``) over the
    ``2N`` ids plus ``module_side``, ``module_label``, ``module_n_types``,
    ``module_n_neurons``, ``homolog``, ``edge_block``, ``synapse_matrix`` and
    ``null_kind``. Because the null's strata include the target side and the
    presynaptic end never moves, ``edge_block[i]`` is also the block of
    rewired edge ``i``.
    """

    started = time.time()
    graph = load(types_path)
    sided = load(sided_path)
    if not np.array_equal(graph["type_names"], sided["type_names"]):
        raise ValueError(f"{types_path} and {sided_path} index different type tables")
    clusters = cluster_modules(graph, n_per_side, seed=seed, embedding=embedding)
    n = int(clusters["n_modules"])
    n2 = 2 * n
    node_type = sided["node_type"].astype(np.int64)
    node_side = sided["node_side"].astype(np.int64)
    node_n_neurons = sided["node_n_neurons"].astype(np.float64)
    module_of_type = clusters["module_of_type"]
    module_of_node = module_of_type[node_type] + n * node_side
    cluster_seconds = time.time() - started

    matrix = module_matrix(sided, module_of_node, n2)
    post, pre, fraction = threshold_input_fraction(matrix, min_fraction)
    module_side = np.repeat(np.arange(len(SIDE_NAMES), dtype=np.int8), n)
    module_role = np.concatenate([clusters["module_role"]] * 2)
    module_sign = np.concatenate([clusters["module_sign"]] * 2).astype(np.int8)
    homolog = np.concatenate([np.arange(n) + n, np.arange(n)]).astype(np.int64)
    module_n_types = np.bincount(module_of_node, minlength=n2)
    module_n_neurons = np.bincount(module_of_node, weights=node_n_neurons, minlength=n2)
    type_names = graph["type_names"]
    label = []
    for mod in range(n2):
        idx = np.flatnonzero(module_of_node == mod)
        role, side_name = str(module_role[mod]), SIDE_NAMES[int(module_side[mod])]
        if len(idx) == 0:
            label.append(f"{role}:<empty>@{side_name}")
        else:
            best = idx[np.argmax(node_n_neurons[idx])]
            label.append(f"{role}:{type_names[node_type[best]]}@{side_name}")
    edge_block = block_of(module_side[pre], module_side[post])
    empty = module_n_types == 0

    rewire_started = time.time()
    node_strata, strata_keys = hemisphere_strata(module_side, module_role, module_sign)
    edge_strata = hemisphere_edge_strata(post, pre, module_side, module_role, module_sign)
    r_post, r_pre, r_fraction, rewire = stratified_rewire(
        post, pre, fraction, node_strata, n2, seed=seed + 1, edge_strata=edge_strata)
    rewire["kind"] = HEMISPHERE_NULL_KIND
    # The v91 null (strata = role x sign, blind to sides) is run for the
    # receipt only, to show what it would do to the blocks; it is not stored.
    roles = np.array([str(r) for r in module_role])
    blind_keys = sorted({(r, int(s)) for r, s in zip(roles, module_sign)})
    blind_index = {k: i for i, k in enumerate(blind_keys)}
    blind_strata = np.array([blind_index[(r, int(s))] for r, s in zip(roles, module_sign)])
    b_post, b_pre, b_fraction, blind = stratified_rewire(post, pre, fraction, blind_strata, n2, seed=seed + 1)
    rewire_seconds = time.time() - rewire_started

    diagnostics_started = time.time()
    sign_f = module_sign.astype(np.float64)
    diagnostics = {
        "real": {
            **graph_diagnostics(post, pre, fraction, sign_f, roles, n2),
            **hemisphere_diagnostics(post, pre, fraction, roles, module_side),
        },
        "stratified_null": {
            **graph_diagnostics(r_post, r_pre, r_fraction, sign_f, roles, n2, reference=(post, pre)),
            **hemisphere_diagnostics(r_post, r_pre, r_fraction, roles, module_side, reference=(post, pre)),
        },
        "side_blind_null": {
            **graph_diagnostics(b_post, b_pre, b_fraction, sign_f, roles, n2, reference=(post, pre)),
            **hemisphere_diagnostics(b_post, b_pre, b_fraction, roles, module_side, reference=(post, pre)),
        },
    }
    diagnostics_seconds = time.time() - diagnostics_started

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(
        output,
        module_of_type=module_of_type,
        module_of_node=module_of_node,
        node_type=sided["node_type"],
        node_side=sided["node_side"],
        node_n_neurons=sided["node_n_neurons"],
        module_side=module_side,
        module_role=module_role,
        module_sign=module_sign,
        module_label=np.array(label, dtype=object),
        module_n_types=module_n_types.astype(np.int64),
        module_n_neurons=module_n_neurons,
        homolog=homolog,
        synapse_matrix=matrix.astype(np.float32),
        edge_post=post, edge_pre=pre, edge_fraction=fraction.astype(np.float32),
        edge_block=edge_block,
        rewired_post=r_post, rewired_pre=r_pre, rewired_fraction=r_fraction.astype(np.float32),
        null_kind=np.array(HEMISPHERE_NULL_KIND, dtype=object),
    )
    per_side = {}
    for code, name in enumerate(SIDE_NAMES):
        sel = module_side == code
        per_side[name] = {
            "modules": int(sel.sum()),
            "empty_modules": int((sel & empty).sum()),
            "excitatory_modules": int((sel & (module_sign > 0)).sum()),
            "inhibitory_modules": int((sel & (module_sign < 0)).sum()),
            "types": int(module_n_types[sel].sum()),
            "neurons": int(module_n_neurons[sel].sum()),
        }
    receipt = {
        "schema": "supermix-v93-malecns-hemispheres-v1",
        "scheme": "mirror",
        "types": types_path,
        "sided_types": sided_path,
        "input_sha256": {"types": _sha256(types_path), "sided_types": _sha256(sided_path)},
        "n_per_side_requested": int(n_per_side),
        "n_per_side": n,
        "n_modules": n2,
        "min_input_fraction": min_fraction,
        "input_fraction_denominator": "total (ipsilateral + commissural) input of the post module",
        "edges": int(len(post)),
        "density": round(len(post) / float(n2 * n2), 5),
        "self_loops": int((post == pre).sum()),
        "rewired_self_loops": int((r_post == r_pre).sum()),
        "blocks": diagnostics["real"]["blocks"],
        "per_side": per_side,
        "empty_module_ids": np.flatnonzero(empty).tolist(),
        "empty_module_labels": [label[i] for i in np.flatnonzero(empty)],
        "groups": clusters["groups"],
        "strata": {"node": len(strata_keys), "edge": int(rewire["strata"])},
        "rewire": rewire,
        "side_blind_rewire": blind,
        "null_kind": HEMISPHERE_NULL_KIND,
        "diagnostics": diagnostics,
        "seed": seed,
        "null_seed": seed + 1,
        "seconds": {
            "cluster": round(cluster_seconds, 1), "rewire": round(rewire_seconds, 1),
            "diagnostics": round(diagnostics_seconds, 1), "total": round(time.time() - started, 1),
        },
        "output": output,
    }
    with open(os.path.splitext(output)[0] + ".receipt.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    return receipt


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="stream the edge list into a cell-type graph")
    b.add_argument("--data_dir", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--skip_md5", action="store_true")
    m = sub.add_parser("modules", help="cluster cell types into module graphs (+ rewired nulls)")
    m.add_argument("--types", required=True, help="npz written by `build`")
    m.add_argument("--n_modules", type=int, nargs="+", required=True)
    m.add_argument("--min_fraction", type=float, default=0.01)
    m.add_argument("--output_prefix", required=True)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("stratify_null", help="install the stratified null into an existing module npz")
    s.add_argument("--modules", required=True)
    s.add_argument("--seed", type=int, default=1)
    h = sub.add_parser("hemispheres", help="v93: (type, side) nodes and mirror module graphs per hemisphere")
    h.add_argument("--data_dir", required=True)
    h.add_argument("--types", required=True, help="v91 malecns_types.npz; type identity is taken from it")
    h.add_argument("--output_dir", required=True, help="never datasets/v91_malecns (v91/v92 hash those files)")
    h.add_argument("--n_modules", type=int, nargs="+", default=[256, 384], help="modules PER SIDE")
    h.add_argument("--min_fraction", type=float, default=0.01)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--skip_md5", action="store_true")
    h.add_argument("--reuse_sided", action="store_true",
                   help=f"skip the edge stream when {SIDED_TYPES_NAME} already exists in --output_dir")
    args = parser.parse_args(argv)
    if args.command == "hemispheres":
        # Forward slashes so the receipts read the same on every platform.
        sided_path = f"{args.output_dir.rstrip('/')}/{SIDED_TYPES_NAME}"
        if args.reuse_sided and os.path.exists(sided_path):
            print(f"reusing {sided_path}", flush=True)
        else:
            receipt = build_sided_types(args.data_dir, args.types, sided_path, verify_md5=not args.skip_md5)
            print(json.dumps(receipt, indent=2), flush=True)
        graph = load(args.types)
        embedding = spectral_embedding(
            len(graph["type_names"]), graph["pre"], graph["post"], graph["weight"], seed=args.seed
        )
        for size in args.n_modules:
            receipt = build_hemispheres(
                args.types, sided_path, f"{args.output_dir.rstrip('/')}/malecns_hemispheres_{size}.npz",
                size, args.min_fraction, seed=args.seed, embedding=embedding,
            )
            print(json.dumps(receipt, indent=2), flush=True)
        return 0
    if args.command == "stratify_null":
        receipt = add_stratified_null(args.modules, seed=args.seed)
        path = os.path.splitext(args.modules)[0] + ".null.receipt.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
        print(json.dumps(receipt, indent=2))
        return 0
    if args.command == "build":
        receipt = build(args.data_dir, args.output, verify_md5=not args.skip_md5)
        print(json.dumps(receipt, indent=2))
    elif args.command == "modules":
        graph = load(args.types)
        embedding = spectral_embedding(
            len(graph["type_names"]), graph["pre"], graph["post"], graph["weight"], seed=args.seed
        )
        for size in args.n_modules:
            receipt = build_modules(
                args.types, f"{args.output_prefix}_{size}.npz", size, args.min_fraction,
                seed=args.seed, embedding=embedding,
            )
            print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
