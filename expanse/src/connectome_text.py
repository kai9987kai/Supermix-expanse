"""Language distillation of the full male-CNS connectome.

## Why this exists

The full-connectome core (``connectome_full.py``) gives Expanse the male-CNS
*wiring* as a fixed dynamical system, but a dynamical system cannot tell the
trunk what a cell type is called, what it releases or whom it talks to. These
rows teach exactly that, as short question/answer turns in the house style
(``supermix-archimedes/corpus/*.jsonl``: one-line replies, <= 60 words,
numeric answers end in ``total N``), so the trunk learns the vocabulary and the
facts of the graph it carries.

v93 (``build_connectome_corpus.py``) asked three numeric questions about the
1,000 largest types only. Expanse asks seven kinds of question about **every**
type whose name is a plain identifier (``[A-Za-z][A-Za-z0-9_\\-]{0,15}``:
11,528 of 11,751; the other 223 are merged labels such as ``DNp51,DNpe019`` or
names with spaces or a leading digit):

    cns_nt           predicted transmitter + excitatory / inhibitory / modulatory
    cns_superclass   superclass (and cell class when annotated)
    cns_top_input    strongest presynaptic type by synapse count, with the count
    cns_top_output   strongest postsynaptic type by synapse count, with the count
    cns_type_count   number of neurons of the type
    cns_in_degree    number of distinct presynaptic types
    cns_path_role    sensory / motor / efferent / descending / ascending / intrinsic

Every fact is read from the npz arrays (all 3,830,931 edges; ties broken by
type name so the answers are a deterministic function of the table) and then
**re-derived by an independent code path** (:func:`verify_rows`) before a row
ships; a mismatch drops the row and is counted in the report.

## Held-out types

10% of the eligible *types* (not rows) are held out: none of their facts
appear in a train row, so eval can ask about a type the model has never been
told about (what it can get right there comes from naming regularities --
``DN*`` descending, ``MN*`` motor -- not from memorised facts). Rows carry
``split`` = ``train`` / ``heldout`` plus ``cell_type`` and a canonical
``answer`` for exact-match scoring.

Data: Janelia FlyEM male-cns v1.0 (https://male-cns.janelia.org/), CC BY 4.0;
Berg et al., Cell (2026). Rows were generated from type-level tables derived
from the flat connectome; the data providers do not endorse this work.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import connectome_full as cf  # noqa: E402

EXPANSE = HERE.parent
DEFAULT_OUTPUT = EXPANSE / "data" / "connectome_rows.jsonl"
ROWS_SCHEMA = "supermix-expanse-connectome-rows-v1"
DOMAIN = "connectome"
SOURCE = "malecns-v1.0"

#: A plain type name (same rule as v93's build_connectome_corpus).
NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{0,15}")

TASKS = ("cns_nt", "cns_superclass", "cns_top_input", "cns_top_output", "cns_type_count", "cns_in_degree",
         "cns_path_role")
NUMERIC_TASKS = ("cns_top_input", "cns_top_output", "cns_type_count", "cns_in_degree")
#: Share of cns_path_role rows drawn from non-intrinsic types (see _rows_for_split).
PATH_ROLE_NON_INTRINSIC_SHARE = 0.5

#: Transmitter -> effect, the v91 sign convention (malecns_connectome.NT_SIGN / MODULATORY).
NT_EFFECT = {
    "acetylcholine": "excitatory",
    "gaba": "inhibitory",
    "glutamate": "inhibitory",
    "histamine": "inhibitory",
    "dopamine": "modulatory",
    "serotonin": "modulatory",
    "octopamine": "modulatory",
}
NT_WORD = {"gaba": "GABA"}

SUPERCLASS_WORDS: Dict[str, str] = {
    "cb_intrinsic": "central brain intrinsic",
    "vnc_intrinsic": "ventral nerve cord intrinsic",
    "ol_intrinsic": "optic lobe intrinsic",
    "ascending_neuron": "ascending neuron",
    "descending_neuron": "descending neuron",
    "visual_projection": "visual projection",
    "visual_centrifugal": "visual centrifugal",
    "ol_sensory": "optic lobe sensory",
    "cb_sensory": "central brain sensory",
    "vnc_sensory": "ventral nerve cord sensory",
    "cb_motor": "central brain motor",
    "vnc_motor": "ventral nerve cord motor",
    "cb_efferent": "central brain efferent",
    "vnc_efferent": "ventral nerve cord efferent",
    "cb_endocrine": "central brain endocrine",
    "vnc_endocrine": "ventral nerve cord endocrine",
    "sensory_ascending": "sensory ascending",
    "sensory_descending": "sensory descending",
    "efferent_ascending": "efferent ascending",
    "efferent_descending": "efferent descending",
}

#: superclass -> (path role, one-clause description of what that role does).
PATH_ROLE: Dict[str, Tuple[str, str]] = {
    "cb_sensory": ("sensory", "it brings signals from sense organs into the central brain"),
    "vnc_sensory": ("sensory", "it brings signals from the body into the ventral nerve cord"),
    "ol_sensory": ("sensory", "it brings light signals from the eye into the optic lobe"),
    "sensory_ascending": ("sensory", "its axon enters the ventral nerve cord and ascends to the brain"),
    "sensory_descending": ("sensory", "its axon enters the brain and descends to the ventral nerve cord"),
    "cb_motor": ("motor", "it sends commands from the brain out to muscles"),
    "vnc_motor": ("motor", "it sends commands from the ventral nerve cord out to muscles"),
    "cb_efferent": ("efferent", "its output leaves the central brain"),
    "vnc_efferent": ("efferent", "its output leaves the ventral nerve cord"),
    "cb_endocrine": ("efferent", "it is an endocrine cell whose output leaves the central brain"),
    "vnc_endocrine": ("efferent", "it is an endocrine cell whose output leaves the ventral nerve cord"),
    "efferent_ascending": ("efferent", "its output leaves the CNS and it also ascends to the brain"),
    "efferent_descending": ("efferent", "its output leaves the CNS and it also descends to the nerve cord"),
    "descending_neuron": ("descending", "it carries signals from the brain down to the ventral nerve cord"),
    "ascending_neuron": ("ascending", "it carries signals from the ventral nerve cord up to the brain"),
    "cb_intrinsic": ("intrinsic", "it stays inside the central brain"),
    "vnc_intrinsic": ("intrinsic", "it stays inside the ventral nerve cord"),
    "ol_intrinsic": ("intrinsic", "it stays inside the optic lobe"),
    "visual_projection": ("intrinsic", "it carries visual signals from the optic lobe to the central brain"),
    "visual_centrifugal": ("intrinsic", "it carries signals from the central brain back to the optic lobe"),
}

TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "cns_nt": (
        "What neurotransmitter does type {t} use?",
        "Which transmitter is predicted for the cell type {t} in the male CNS?",
        "neurotransmitter of type {t}",
        "Is type {t} excitatory or inhibitory?",
        "What does type {t} release at its synapses?",
        "In the male fly CNS, which transmitter does {t} release?",
    ),
    "cns_superclass": (
        "What kind of neuron is type {t} in the male CNS?",
        "Which superclass does the cell type {t} belong to?",
        "superclass of type {t}",
        "Which superclass is {t} in the male fly connectome?",
        "Classify the cell type {t} by superclass.",
    ),
    "cns_top_input": (
        "Which cell type sends the most synapses to type {t}?",
        "What is the strongest input to type {t} in the male CNS?",
        "top input of type {t}",
        "Which presynaptic type connects most strongly onto {t}?",
        "Where does type {t} get most of its input synapses from?",
    ),
    "cns_top_output": (
        "Which cell type receives the most synapses from type {t}?",
        "What is the strongest downstream partner of type {t} in the male CNS?",
        "top output target of type {t}",
        "Where does type {t} send most of its synapses?",
        "Which postsynaptic type does {t} contact most strongly?",
    ),
    "cns_type_count": (
        "How many neurons of type {t} are in the male CNS?",
        "In the male CNS, how many neurons have the cell type {t}?",
        "type {t} neuron count in the male CNS",
        "Count the neurons of cell type {t} in the male CNS connectome.",
        "How many {t} neurons does the male fly CNS contain?",
    ),
    "cns_in_degree": (
        "How many cell types send synapses to type {t}?",
        "From how many presynaptic types does {t} receive input?",
        "number of input types of {t} in the male CNS",
        "Count the distinct input types of {t} in the male CNS.",
        "How many different cell types connect onto type {t}?",
    ),
    "cns_path_role": (
        "Is type {t} a sensory, motor, descending, ascending or intrinsic neuron?",
        "What role does type {t} play in the path from senses to muscles?",
        "path role of type {t}",
        "Does type {t} carry input, output or internal signals in the male CNS?",
        "Is {t} a sensory neuron, a motor neuron or an interneuron?",
    ),
}


def superclass_words(code: str) -> str:
    if code in SUPERCLASS_WORDS:
        return SUPERCLASS_WORDS[code]
    return code.replace("cb_", "central brain ").replace("vnc_", "ventral nerve cord ").replace(
        "ol_", "optic lobe ").replace("_", " ")


def class_words(code: str) -> Optional[str]:
    if not code or code in ("unknown", "nan", "None"):
        return None
    return code.replace("_", " ")


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------
def _strongest_partner(key: np.ndarray, other: np.ndarray, weight: np.ndarray, n: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Per node of ``key``: the ``other`` endpoint with the largest weight (ties -> smaller index = name order).

    Returns (partner int64, weight int64), -1 / 0 where the node has no edge.
    Type names in the npz are sorted, so a smaller index is the alphabetically
    earlier name.
    """

    order = np.lexsort((other, -weight, key))
    k = key[order]
    first = np.ones(len(k), dtype=bool)
    first[1:] = k[1:] != k[:-1]
    partner = np.full(n, -1, dtype=np.int64)
    best = np.zeros(n, dtype=np.int64)
    partner[k[first]] = other[order][first]
    best[k[first]] = weight[order][first]
    return partner, best


def type_facts(graph: dict) -> Dict[str, Any]:
    """Every fact the rows state, per type, from the graph's arrays (all edges)."""

    receipt = graph.get("receipt", {})
    if float(receipt.get("threshold", 0.0)) != 0.0:
        raise ValueError("connectome rows need the full graph (min_input_fraction=0): facts are over all edges")
    if "weight" not in graph:
        raise ValueError("graph has no synapse counts ('weight'); build it with connectome_full.build_full_graph")
    n = int(graph["n"])
    post = np.asarray(graph["post"], dtype=np.int64)
    pre = np.asarray(graph["pre"], dtype=np.int64)
    weight = np.asarray(graph["weight"], dtype=np.int64)
    names = [str(x) for x in graph["type_names"]]
    if names != sorted(names):
        raise ValueError("type_names must be sorted (tie-breaking by index == by name)")
    top_in, top_in_w = _strongest_partner(post, pre, weight, n)
    top_out, top_out_w = _strongest_partner(pre, post, weight, n)
    return {
        "names": names,
        "nt": [str(x) for x in graph["nt"]],
        "superclass": [str(x) for x in graph["superclass"]],
        "cell_class": [str(x) for x in graph["cell_class"]],
        "n_neurons": np.asarray(graph["n_neurons"], dtype=np.int64),
        "top_in": top_in, "top_in_w": top_in_w,
        "top_out": top_out, "top_out_w": top_out_w,
        "in_degree": np.bincount(post, minlength=n).astype(np.int64),
    }


def eligible_types(names: Sequence[str]) -> List[int]:
    return [i for i, name in enumerate(names) if NAME_PATTERN.fullmatch(name)]


def split_types(names: Sequence[str], seed: int, heldout_fraction: float = 0.1) -> Tuple[List[int], List[int]]:
    """Eligible type indices split into (train, heldout) by a seeded shuffle of the sorted names."""

    idx = eligible_types(names)
    rng = random.Random(f"heldout-types-{seed}")
    shuffled = list(idx)
    rng.shuffle(shuffled)
    k = int(round(heldout_fraction * len(shuffled)))
    return sorted(shuffled[k:]), sorted(shuffled[:k])


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def fact_row(facts: Dict[str, Any], task: str, i: int, rng: random.Random) -> Dict[str, str]:
    """One row for (task, type index i): prompt from a random template, reply + canonical answer."""

    names = facts["names"]
    t = names[i]
    prompt = rng.choice(TEMPLATES[task]).format(t=t)
    if task == "cns_nt":
        nt = facts["nt"][i]
        if nt in NT_EFFECT:
            word = NT_WORD.get(nt, nt)
            reply = f"type {t} is predicted to release {word}, so it is {NT_EFFECT[nt]}"
        else:
            reply = f"the transmitter of type {t} is unclear in the male CNS data, so its sign is not known"
        answer = nt
    elif task == "cns_superclass":
        sc = facts["superclass"][i]
        reply = f"type {t} belongs to the {superclass_words(sc)} superclass"
        cls = class_words(facts["cell_class"][i])
        if cls:
            reply += f", class {cls}"
        answer = sc
    elif task in ("cns_top_input", "cns_top_output"):
        partner_key, weight_key = ("top_in", "top_in_w") if task == "cns_top_input" else ("top_out", "top_out_w")
        j = int(facts[partner_key][i])
        w = int(facts[weight_key][i])
        if j < 0:
            reply = (f"type {t} receives no synapses from typed neurons, total 0" if task == "cns_top_input"
                     else f"type {t} makes no synapses onto typed neurons, total 0")
            answer = "none 0"
        else:
            other = "itself" if j == i else f"type {names[j]}"
            syn = "synapse" if w == 1 else "synapses"
            if task == "cns_top_input":
                reply = f"the strongest input to type {t} is {other}, with {w} {syn}, total {w}"
            else:
                reply = f"type {t} sends the most synapses to {other}, {w} {syn}, total {w}"
            answer = f"{names[j]} {w}"
    elif task == "cns_type_count":
        c = int(facts["n_neurons"][i])
        reply = f"type {t} has {c} {'neuron' if c == 1 else 'neurons'} in the male CNS, total {c}"
        answer = str(c)
    elif task == "cns_in_degree":
        k = int(facts["in_degree"][i])
        kind = "cell type" if k == 1 else "cell types"
        reply = f"type {t} receives synapses from {k} {kind}, total {k}"
        answer = str(k)
    elif task == "cns_path_role":
        sc = facts["superclass"][i]
        role, what = PATH_ROLE.get(sc, ("intrinsic", "it stays inside the CNS"))
        article = "an" if role[0] in "aeiou" else "a"
        reply = f"type {t} is {article} {role} neuron, {what}"
        answer = role
    else:
        raise KeyError(task)
    return {"user": prompt, "assistant": reply, "domain": DOMAIN, "task": task, "cell_type": t,
            "answer": answer, "source": SOURCE}


def _rows_for_split(facts: Dict[str, Any], types: Sequence[int], n_rows: int, split: str,
                    rng: random.Random) -> List[Dict[str, str]]:
    """``n_rows`` distinct (task, type) facts, balanced over tasks, types without replacement per task.

    Types are drawn uniformly, except for ``cns_path_role``: 85% of eligible
    types are intrinsic, so a uniform draw would let "intrinsic" score 0.85
    while teaching almost nothing about the sensory / motor / descending /
    ascending names. Up to half of its rows (``PATH_ROLE_NON_INTRINSIC_SHARE``)
    are therefore drawn from the non-intrinsic types; the report gives the
    resulting majority-answer share for both splits.
    """

    rows: List[Dict[str, str]] = []
    base, extra = divmod(n_rows, len(TASKS))
    seen_prompts = set()
    types = list(types)
    for k, task in enumerate(TASKS):
        want = min(base + (1 if k < extra else 0), len(types))
        if task == "cns_path_role":
            other = [i for i in types if PATH_ROLE.get(facts["superclass"][i], ("intrinsic",))[0] != "intrinsic"]
            inner = [i for i in types if PATH_ROLE.get(facts["superclass"][i], ("intrinsic",))[0] == "intrinsic"]
            n_other = min(len(other), int(round(want * PATH_ROLE_NON_INTRINSIC_SHARE)))
            n_inner = min(len(inner), want - n_other)
            chosen = rng.sample(other, n_other) + rng.sample(inner, n_inner)
        else:
            chosen = rng.sample(types, want)
        for i in chosen:
            row = fact_row(facts, task, i, rng)
            if row["user"] in seen_prompts:  # cannot happen (type names are unique); kept as a guard
                continue
            seen_prompts.add(row["user"])
            row["split"] = split
            rows.append(row)
    rng.shuffle(rows)
    return rows


def build_connectome_rows(graph: dict, seed: int, n_rows: int, *, heldout_fraction: float = 0.1,
                          n_heldout_rows: Optional[int] = None) -> List[dict]:
    """Train rows (``n_rows``) over 90% of the eligible types, then held-out rows over the other 10%.

    ``n_heldout_rows`` defaults to the same rows-per-type density as train
    (``n_rows * f / (1 - f)``). Rows are deduplicated by (task, type) and by
    prompt; every fact is exact (see :func:`verify_rows`).
    """

    facts = type_facts(graph)
    train_types, heldout_types = split_types(facts["names"], seed, heldout_fraction)
    if n_heldout_rows is None:
        n_heldout_rows = int(round(n_rows * heldout_fraction / max(1e-9, 1.0 - heldout_fraction)))
    rng = random.Random(seed)
    train = _rows_for_split(facts, train_types, int(n_rows), "train", rng)
    heldout = _rows_for_split(facts, heldout_types, int(n_heldout_rows), "heldout", rng)
    return train + heldout


# ---------------------------------------------------------------------------
# Independent verification
# ---------------------------------------------------------------------------
def verify_rows(rows: Sequence[dict], npz_path: os.PathLike) -> Tuple[List[dict], Dict[str, int]]:
    """Re-derive each row's ``answer`` straight from the raw npz, by a different code path.

    ``type_facts`` works on the sorted graph with one lexsort over all edges;
    this reads the unsorted npz arrays and, per asked type, scans only that
    type's edges (found through a post / pre index built with ``argsort``),
    taking the max weight with name tie-breaking in plain Python. The numeric
    tail (``total N``) of the reply is checked against the fact as well.
    Returns (rows that match, mismatch counts per task).
    """

    data = cf.load_npz(npz_path)
    names = [str(x) for x in data["type_names"]]
    index = {name: i for i, name in enumerate(names)}
    pre = data["pre"].astype(np.int64)
    post = data["post"].astype(np.int64)
    weight = data["weight"].astype(np.int64)
    by_post = np.argsort(post, kind="stable")
    by_pre = np.argsort(pre, kind="stable")
    post_start = np.searchsorted(post[by_post], np.arange(len(names) + 1))
    pre_start = np.searchsorted(pre[by_pre], np.arange(len(names) + 1))

    def strongest(edges: np.ndarray, other: np.ndarray) -> str:
        if len(edges) == 0:
            return "none 0"
        # the largest synapse count; among equal counts the alphabetically first name
        top_w = max(int(weight[e]) for e in edges)
        top_name = min(names[int(other[e])] for e in edges if int(weight[e]) == top_w)
        return f"{top_name} {top_w}"

    def expected(task: str, t: str) -> str:
        i = index[t]
        if task == "cns_nt":
            return str(data["nt"][i])
        if task == "cns_superclass":
            return str(data["superclass"][i])
        if task == "cns_type_count":
            return str(int(data["n_neurons"][i]))
        if task == "cns_in_degree":
            return str(int(len(set(pre[by_post[post_start[i]:post_start[i + 1]]].tolist()))))
        if task == "cns_top_input":
            return strongest(by_post[post_start[i]:post_start[i + 1]], pre)
        if task == "cns_top_output":
            return strongest(by_pre[pre_start[i]:pre_start[i + 1]], post)
        if task == "cns_path_role":
            return PATH_ROLE.get(str(data["superclass"][i]), ("intrinsic", ""))[0]
        raise KeyError(task)

    kept: List[dict] = []
    dropped: Counter = Counter()
    for row in rows:
        want = expected(row["task"], row["cell_type"])
        numeric_ok = True
        if row["task"] in NUMERIC_TASKS:
            # the reply's trailing "total N" must be the fact's number
            tail = re.search(r"total (-?\d+)$", row["assistant"])
            numeric_ok = tail is not None and tail.group(1) == want.split()[-1]
        if row["answer"] == want and numeric_ok:
            kept.append(row)
        else:
            dropped[row["task"]] += 1
    return kept, dict(dropped)


# ---------------------------------------------------------------------------
# Report + CLI
# ---------------------------------------------------------------------------
def rows_report(rows: Sequence[dict], graph: dict, seed: int, n_rows: int, n_heldout_requested: int,
                dropped: Dict[str, int], output: str, heldout_fraction: float) -> Dict[str, Any]:
    names = [str(x) for x in graph["type_names"]]
    train_types, heldout_types = split_types(names, seed, heldout_fraction)
    excluded = [nm for nm in names if not NAME_PATTERN.fullmatch(nm)]
    per_task: Dict[str, Dict[str, int]] = {t: {"train": 0, "heldout": 0} for t in TASKS}
    answers: Dict[str, Dict[str, Counter]] = {sp: {t: Counter() for t in TASKS} for sp in ("train", "heldout")}
    types_by_split: Dict[str, set] = {"train": set(), "heldout": set()}
    templates_used: Dict[str, Counter] = {t: Counter() for t in TASKS}
    template_regex = {t: [re.compile(re.escape(tpl).replace(re.escape("{t}"), r"\S+")) for tpl in TEMPLATES[t]]
                      for t in TASKS}
    for row in rows:
        per_task[row["task"]][row["split"]] += 1
        types_by_split[row["split"]].add(row["cell_type"])
        answers[row["split"]][row["task"]][row["answer"]] += 1
        for k, pattern in enumerate(template_regex[row["task"]]):
            if pattern.fullmatch(row["user"]):
                templates_used[row["task"]][k] += 1
                break
    words_user = [len(r["user"].split()) for r in rows]
    words_asst = [len(r["assistant"].split()) for r in rows]
    heldout_names = [names[i] for i in heldout_types]
    leak = types_by_split["train"] & set(heldout_names)
    return {
        "schema": ROWS_SCHEMA,
        "npz": graph["receipt"]["npz"],
        "npz_sha256": graph["receipt"]["npz_sha256"],
        "seed": int(seed),
        "rows": len(rows),
        "train_rows": sum(1 for r in rows if r["split"] == "train"),
        "heldout_rows": sum(1 for r in rows if r["split"] == "heldout"),
        "requested": {"train": int(n_rows), "heldout": int(n_heldout_requested)},
        "per_task": per_task,
        "tasks": list(TASKS),
        "numeric_tasks_end_in_total": list(NUMERIC_TASKS),
        "types_total": len(names),
        "types_eligible": len(train_types) + len(heldout_types),
        "eligibility": f"name fullmatches {NAME_PATTERN.pattern}",
        "types_excluded": len(excluded),
        "types_excluded_examples": excluded[:12],
        "heldout_fraction": heldout_fraction,
        "types_train_pool": len(train_types),
        "types_heldout_pool": len(heldout_types),
        "distinct_types": {"train": len(types_by_split["train"]), "heldout": len(types_by_split["heldout"]),
                           "all": len(types_by_split["train"] | types_by_split["heldout"])},
        "heldout_types_in_train_rows": len(leak),
        "distinct_prompts": len({r["user"] for r in rows}),
        "distinct_facts": len({(r["task"], r["cell_type"]) for r in rows}),
        # (share, answer) of the most common answer per task: the score of always guessing it
        "majority_answer_share": {
            sp: {t: (round(c.most_common(1)[0][1] / max(1, sum(c.values())), 4), c.most_common(1)[0][0])
                 if c else None for t, c in per.items()}
            for sp, per in answers.items()
        },
        "path_role_non_intrinsic_share": PATH_ROLE_NON_INTRINSIC_SHARE,
        "templates_per_task": {t: len(TEMPLATES[t]) for t in TASKS},
        "template_use": {t: dict(sorted(c.items())) for t, c in templates_used.items()},
        "max_words": {"user": max(words_user, default=0), "assistant": max(words_asst, default=0)},
        "non_ascii_rows": sum(1 for r in rows if not (r["user"] + r["assistant"]).isascii()),
        "multiline_rows": sum(1 for r in rows if "\n" in r["user"] + r["assistant"]),
        "dropped_failing_verification": dropped,
        "verified_by": "connectome_text.verify_rows: every answer re-derived from the raw npz arrays",
        "row_fields": ["user", "assistant", "domain", "task", "split", "cell_type", "answer", "source"],
        "heldout_type_names": heldout_names,
        "output": output,
        "attribution": "Janelia FlyEM male-cns v1.0 (CC BY 4.0); Berg et al., Cell (2026)",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npz", default=str(cf.DEFAULT_NPZ))
    parser.add_argument("--n_rows", type=int, default=6000, help="train rows")
    parser.add_argument("--n_heldout_rows", type=int, default=None)
    parser.add_argument("--heldout_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    graph = cf.build_full_graph(args.npz)
    rows = build_connectome_rows(graph, args.seed, args.n_rows, heldout_fraction=args.heldout_fraction,
                                 n_heldout_rows=args.n_heldout_rows)
    n_heldout = args.n_heldout_rows if args.n_heldout_rows is not None else int(
        round(args.n_rows * args.heldout_fraction / (1.0 - args.heldout_fraction)))
    rows, dropped = verify_rows(rows, args.npz)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    os.replace(tmp, output)
    out_str = str(output.resolve()).replace("\\", "/")
    report = rows_report(rows, graph, args.seed, args.n_rows, n_heldout, dropped, out_str, args.heldout_fraction)
    report_path = output.with_name(output.stem + ".report.json")
    cf.write_json(report_path, report)
    brief = {k: v for k, v in report.items() if k not in ("heldout_type_names", "template_use")}
    print(json.dumps(cf._jsonable(brief), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
