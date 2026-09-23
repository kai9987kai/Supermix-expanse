"""Independently check a maths reply against the question that was asked.

The session's central finding is that only a *checkable* answer resists
recitation: a model reproducing a remembered reply to a novel problem is simply
wrong, and no amount of fluency hides it. `eval_problem_solving.py` applies that
offline, over problems it generates itself.

This module applies it live, to whatever the user typed. It re-derives the truth
from the question and compares, so the interface can say "wrong, the answer is
905" instead of presenting confident arithmetic and leaving the reader to check.

It recognises only the shapes the models were trained on -- twenty-one of them
as of v82, one per task in the v80 corpus: nine arithmetic shapes from v74 and
twelve solver-verified science and mathematics shapes from v79/v80. Anything
else returns `None`, which the interface must render as *not checked* rather
than as correct: an unrecognised question is not a passed one.

Multiplication, division, sequence and two-step were added for v74, which
introduced those tasks. Before that a chat reply to "What is 47 x 6?" showed
NOT CHECKED -- the model's strongest tasks were the ones nothing verified.

**This module is not a verifier and must never be promoted to one.** It
re-derives an expected answer from a *pattern* in the question, not from a
parse of its meaning, so a question whose shape it half-recognises would get a
confident wrong verdict. `nexus_epistemics.ANSWER_VERIFIER_IDS` is the
allowlist of things permitted to certify an answer and it contains exactly one
entry, `grounding_runtime.finalize_grounded_response`; `answer_check` is
deliberately absent from it. The compound-expression trap in
`_is_compound_expression` is the concrete reason: "What is 2 + 3 * 4?" once
parsed as multiplication with expected 12.0 where the truth is 14, because a
lone `A * B` search found `3 * 4` and never saw the `+`. That is now refused
as NOT CHECKED, and `test_answer_check.py` pins it in both directions.

v82 coverage, measured over 840 prompts drawn from `build_omni_corpus.TASKS`
and `eval_problem_solving.GENERATORS` at seed 4242: 799/840 = 0.951 before the
v82 widening, 840/840 = 1.000 after, with zero confident-wrong verdicts in
either. Coverage is not accuracy of the *model*; it is only the fraction of
questions this module is willing to judge at all.

v93 adds eleven benchmark tasks and this module gains a parser for every
shape among them: five science and mathematics forms re-derived from the
question (`impulse`, `ohms_current`, `spring_energy`, `permutations`,
`final_velocity`), three code forms the existing `_code_trace` already runs
once its snippet-start pattern admits `r = sum(range(3, 9))`, and three
connectome lookups (`cns_type_count`, `cns_side_count`, `cns_pair_synapses`)
whose "re-derivation" is the same table lookup the corpus builder did, from
the CC-BY male-CNS arrays. Measured at seed 4242 over all 41 registered tasks:
1,640/1,640 narrow-template prompts parsed and 4,920/4,920 prompts over the
full natural-phrasing bank (held-out forms included), zero confident-wrong
verdicts in either. `_permutations` runs before `_combination_choose` because
"permutations of 9 things taken 2 at a time" contains `taken`, and the
combination reading would have returned 36 for an answer of 72.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

#: Matches the tolerance `eval_problem_solving.is_correct` uses, so the live
#: check and the benchmark cannot disagree about the same answer.
TOLERANCE = 1e-6

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_DECIMAL = r"-?\d+(?:\.\d+)?"


@dataclass
class Check:
    """The verdict on one reply."""

    task: str
    expected: float
    predicted: Optional[float]
    correct: bool

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "expected": self.expected,
            "predicted": self.predicted,
            "correct": self.correct,
        }


def extract_answer(text: str) -> Optional[float]:
    """The reply's answer is the last number it produces.

    Identical rule to `eval_problem_solving.extract_answer`; the scratchpad
    formats all end with the result, so "600 + 200 = 800, 17 + 88 = 105, total
    905" reads as 905 rather than 600.
    """

    matches = _NUMBER.findall(text.replace(",", ""))
    if not matches:
        return None
    raw = matches[-1].rstrip(".")
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            value = float(numerator) / float(denominator)
        else:
            value = float(raw)
        return value if math.isfinite(value) else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


# -- question parsers -------------------------------------------------------
#
# Each returns (task, expected) or None. They are deliberately narrow: a loose
# pattern that half-matched an unrelated question would produce a confident
# wrong verdict, which is worse than no verdict at all.


#: An operator sitting between two numbers -- the only kind that makes an
#: expression compound. `m/s`, `m/s^2` and `x + 14 = 39` do not match, because
#: neither side of the operator is a digit in the first two and `_algebra`
#: claims the third before these parsers run.
#: The trailing operand is a lookahead, not a consumed group: without it
#: `re.findall` over "2 + 3 * 4" consumes the `3` while matching `2 + 3` and
#: then finds only one operator, which is the bug this guard exists to catch.
_INFIX = re.compile(r"\d\s*[-+*/x]\s*(?=-?\d)", re.I)


def _is_compound_expression(question: str) -> bool:
    """True when the question chains two or more infix operators.

    The bare `A op B` parsers below each find *one* operator and compute from
    it. Given "What is 2 + 3 * 4?" the multiplication parser finds `3 * 4` and
    returns 12.0, which is confidently wrong: precedence makes the answer 14.
    Nothing in the corpus asks a compound question, so the correct response is
    to refuse rather than to grow an expression evaluator here -- a partially
    correct evaluator would produce exactly the confident wrong verdict this
    module exists to avoid.
    """

    return len(_INFIX.findall(question)) >= 2


def _binary(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(rf"(?<![\w.+-])({_DECIMAL})\s*([+-])\s*({_DECIMAL})(?!\w|\.\d)", question)
    if not match or "=" in question or _is_compound_expression(question):
        return None
    left, op, right = float(match.group(1)), match.group(2), float(match.group(3))
    return ("arithmetic", float(left + right if op == "+" else left - right))


def _percent(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(rf"({_DECIMAL})\s*%\s*of\s*({_DECIMAL})", question, re.I)
    if not match:
        return None
    return ("percent", float(match.group(1)) * float(match.group(2)) / 100.0)


def _algebra(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(rf"\bx\s*([+*/-])\s*({_DECIMAL})\s*=\s*({_DECIMAL})(?!\w|\.\d)", question, re.I)
    if not match:
        return None
    op, constant, right = match.group(1), float(match.group(2)), float(match.group(3))
    if op in "*/" and constant == 0:
        return None
    if op == "+":
        result = right - constant
    elif op == "-":
        result = right + constant
    elif op == "*":
        result = right / constant
    else:
        result = right * constant
    return ("algebra_one_step", result)


def _average(question: str) -> Optional[Tuple[str, float]]:
    if not re.search(r"\b(average|mean)\b", question, re.I):
        return None
    tail = question.split(":", 1)[-1]
    values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", tail)]
    if len(values) < 2:
        return None
    return ("average", sum(values) / len(values))


def _multiplication(question: str) -> Optional[Tuple[str, float]]:
    """`A x B`, the corpus's multiplication form.

    Must run after `_algebra`: `x` is this corpus's multiplication sign *and*
    its unknown, and only the digit on the left tells them apart.
    """

    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[x*]\s*(-?\d+(?:\.\d+)?)", question, re.I)
    if not match or "=" in question or _is_compound_expression(question):
        return None
    return ("multiplication", float(match.group(1)) * float(match.group(2)))


def _division(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", question)
    if not match or "=" in question or _is_compound_expression(question):
        return None
    divisor = float(match.group(2))
    if divisor == 0:
        # Not checkable rather than an exception; the question has no answer.
        return None
    return ("division", float(match.group(1)) / divisor)


#: The fractions `two_step` is allowed to state in words, and their percentages.
#: Kept in step with `build_scratchpad_math.PERCENT_FRACTIONS`; a form this does
#: not know falls through to `_percent`, which reads the percentage and ignores
#: the second operation, so it must not silently disagree.
_SPOKEN_FRACTIONS = {
    "one tenth": 10.0, "one fifth": 20.0, "one quarter": 25.0, "one half": 50.0,
}

#: How the second operation can be worded. `reduce it by` and `take away` are
#: subtraction; `increase it by` is addition.
_ADD_WORDS = r"add|increase it by"
_SUBTRACT_WORDS = r"subtract|reduce it by|take away"


def _two_step(question: str) -> Optional[Tuple[str, float]]:
    """`P% of N`, then a second operation. Must precede `_percent`, which it contains.

    v88 gave this task five prompt templates where it had one, and the extra
    forms broke the original pattern in three ways at once: the percentage can
    be spelled out (`Take 15 percent of 320`) or spoken as a fraction (`Start
    with one quarter of 320`), the two clauses can be joined by `and then`, a
    semicolon or a full stop, and the operation can be worded (`reduce it by`,
    `increase it by`, `take away`).

    A miss here is not harmless. `_two_step` runs before `_percent` precisely
    because it contains a percent question, so a form this does not match is
    read by `_percent` as `P% of N` -- the first half only -- and every such row
    is then reported WRONG against a correct reply. That is what 174 of 20,391
    checked rows did before this was widened, all of them `two_step`.
    """

    base: Optional[float] = None
    tail = question

    match = re.search(
        rf"({_DECIMAL})\s*(?:%|percent)\s*of\s*({_DECIMAL})", question, re.I)
    if match:
        base = float(match.group(1)) * float(match.group(2)) / 100.0
        tail = question[match.end():]
    else:
        spoken = re.search(
            rf"\b({'|'.join(_SPOKEN_FRACTIONS)})\s+of\s*({_DECIMAL})",
            question, re.I)
        if spoken:
            base = (_SPOKEN_FRACTIONS[spoken.group(1).lower()]
                    * float(spoken.group(2)) / 100.0)
            tail = question[spoken.end():]
    if base is None:
        return None

    operation = re.search(
        rf"(?:then|and then|next|,|;|\.)\s*({_ADD_WORDS}|{_SUBTRACT_WORDS})"
        rf"\s*({_DECIMAL})", tail, re.I)
    if not operation:
        return None
    word, operand = operation.group(1).lower(), float(operation.group(2))
    adds = re.fullmatch(_ADD_WORDS, word, re.I) is not None
    return ("two_step", base + operand if adds else base - operand)


def _sequence(question: str) -> Optional[Tuple[str, float]]:
    """The next term of an arithmetic progression.

    Returns None when the differences are not constant. The corpus only
    contains arithmetic progressions, so anything else is a question this
    cannot verify -- and reporting "not checked" is correct where guessing a
    rule would silently invent a right answer.
    """

    if not re.search(r"\b(next|sequence)\b", question, re.I):
        return None
    tail = question.split(":", 1)[-1]
    values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", tail)]
    if len(values) < 3:
        return None
    steps = {round(b - a, 9) for a, b in zip(values, values[1:])}
    if len(steps) != 1:
        return None
    return ("sequence", values[-1] + steps.pop())


# -- science shapes (v81) ---------------------------------------------------
#
# v80 answers physics correctly and the interface said NOT CHECKED for every
# one of them, because these shapes were never taught to the checker. A model
# whose strongest new capability cannot be verified live is the same gap v76
# closed for multiplication, one domain over.
#
# Each reads the quantities by name and unit, so it matches the corpus's
# phrasings without depending on any single one of them.

def _quantity(question: str, names: str, unit: str) -> Optional[float]:
    """Read `<name> <number> <unit>` in either order, as the corpus writes it."""

    number = r"(-?\d+(?:\.\d+)?)"
    # Both names and unit are alternations, so both must be grouped. Left
    # bare, `(\d+)\s*m|metres?` parses as `(\d+)\s*m` OR `metres?` -- the
    # second branch has no capture group, and match.group(1) is then None.
    for pattern in (rf"(?:{names})\D{{0,24}}?{number}\s*(?:{unit})\b",
                    rf"{number}\s*(?:{unit})\b\D{{0,24}}?(?:{names})"):
        match = re.search(pattern, question, re.I)
        if match:
            return float(match.group(1))
    return None


#: (task, words identifying the target, quantity A, quantity B)
#: Division tasks are named in `_DIVISION_LAWS`; everything else multiplies.
#: Units carry their spelled-out forms. The corpus writes "57 volts and 5
#: amps" as readily as "57 V, 5 A", and a checker that only knew the symbols
#: reported NOT CHECKED for a third of the questions the model answers.
_MASS = r"kg|kilograms?"
_FORCE = r"N|newtons?"
_ANY = r"[a-z/^]*"

_PRODUCT_LAWS = (
    ("force", r"force",
     (r"mass|body|block|object", _MASS), (r"accelerat\w*", r"m/s\^?2")),
    ("momentum", r"momentum",
     (r"mass|object|body", _MASS), (r"velocity|speed|moves|travelling|at", r"m/s")),
    # v82: "Find the work done by 98 N acting over 2 m." names the force only
    # as "by". Measured 32/40 before adding it, 40/40 after.
    ("work", r"work",
     (r"force|done by|by", _FORCE), (r"distance|moves|through|over|acts", r"m|metres?")),
    ("voltage", r"voltage|potential difference",
     (r"current|flows|carrying|drives", r"A|amps?|amperes?"),
     (r"resistance|ohm|through|across|resistor", r"ohms?")),
    ("electrical_power", r"electrical power|power dissipated|power|used at",
     (r"voltage|volts?|runs at|at", r"V|volts?"),
     (r"current|drawing|amps?|and", r"A|amps?|amperes?")),
    ("wave_speed", r"wave speed|speed of|its speed|speed at",
     (r"frequency|at", r"Hz|hertz"), (r"wavelength|with", r"m|metres?")),
    # v82: "produced by 580 N on 116 kg" names the force only as "by".
    # Measured 35/40 before, 40/40 after.
    ("acceleration", r"acceleration|accelerat\w*",
     (r"force|results from|from|produced by|by", _FORCE), (r"mass|body|object|on", _MASS)),
    # v82: "What power corresponds to 1860 joules in 20 seconds?" names the
    # work only as "corresponds to". Measured 28/40 before, 40/40 after.
    ("power", r"power",
     (r"work|corresponds to|delivered|done", r"J|joules?"), (r"time|in|over", r"s|seconds?")),
    ("molarity", r"molarity|concentration|molar",
     (r"mol|moles|solute|of", r"mol|moles"),
     (r"volume|litres?|liters?|dissolved|in", r"L|litres?|liters?")),
    # -- v93 --
    # Placed after the nine above on purpose. `impulse` prompts always name a
    # force, and `ohms_current` prompts always say "current" and usually
    # "voltage", so the `force`, `voltage` and `electrical_power` laws see them
    # first; each then fails on a quantity it needs (kg, amps) and falls
    # through. The reverse cannot happen: a `voltage` prompt carries no number
    # in volts and a `force` prompt no time in seconds, so neither of these
    # can claim one of theirs.
    ("impulse", r"impulse",
     (r"force|push|delivered by|acting|acts|by|from|:", _FORCE),
     (r"time|for|over|lasting|during|in|,", r"s|seconds?")),
    ("ohms_current", r"current",
     (r"voltage|volts?|at|drives|across|applied|when|:", r"V|volts?"),
     (r"resistance|resistor|ohms?|through|across|drives|,", r"ohms?")),
)


_DIVISION_LAWS = frozenset({"acceleration", "power", "molarity", "ohms_current"})


def _science(question: str) -> Optional[Tuple[str, float]]:
    for task, target, (a_names, a_unit), (b_names, b_unit) in _PRODUCT_LAWS:
        if not re.search(target, question, re.I):
            continue
        a = _quantity(question, a_names, a_unit)
        b = _quantity(question, b_names, b_unit)
        if a is None or b is None:
            continue
        if task in _DIVISION_LAWS:
            if b == 0:
                return None   # not checkable rather than an exception
            return (task, a / b)
        return (task, a * b)
    return None


def _kinetic_energy(question: str) -> Optional[Tuple[str, float]]:
    if not re.search(r"kinetic energy", question, re.I):
        return None
    mass = _quantity(question, r"mass|body", r"kg")
    velocity = _quantity(question, r"velocity|speed|moves|at", r"m/s")
    if mass is None or velocity is None:
        return None
    return ("kinetic_energy", 0.5 * mass * velocity * velocity)


def _spring_energy(question: str) -> Optional[Tuple[str, float]]:
    """`E = k x^2 / 2` (v93). Runs before `_science`, as `_kinetic_energy` does.

    The spring constant is the number carrying `N/m`, which is unambiguous;
    the extension is the number in metres that is *not* part of `N/m`, read
    after an extension word so the `m` of `N/m` cannot be taken for it.
    """

    if not re.search(r"spring energy|elastic potential energy|energy (?:is )?stored",
                     question, re.I):
        return None
    constant = re.search(r"(-?\d+(?:\.\d+)?)\s*N/m\b", question, re.I)
    extension = _quantity(
        question,
        r"extension|extended|stretched|pulled|stretch|by|is",
        r"m|metres?|meters?",
    )
    if constant is None or extension is None:
        return None
    k, x = float(constant.group(1)), float(extension)
    return ("spring_energy", 0.5 * k * x * x)


def _final_velocity(question: str) -> Optional[Tuple[str, float]]:
    """`v = u + a t` (v93). Runs before `_science`.

    Three numbers, three units, and the units tell them apart: the
    acceleration carries `m/s^2`, the initial velocity `m/s` with no `^2`
    after it, and the time a bare `s` or `seconds`. Nothing is read by
    position, so any order of the three clauses parses the same way.
    """

    if not re.search(r"final velocity|final speed|velocity after|going after",
                     question, re.I):
        return None
    accel = re.search(r"(-?\d+(?:\.\d+)?)\s*m/s\^?2\b", question, re.I)
    initial = re.search(r"(-?\d+(?:\.\d+)?)\s*m/s\b(?!\^?2)", question, re.I)
    # `(?<![/^])` keeps the `s` of `m/s` and the `2` of `m/s^2` out of it.
    time = re.search(r"(?<![/^\w])(-?\d+(?:\.\d+)?)\s*(?:s|seconds?)\b(?!/|\^)",
                     question, re.I)
    if accel is None or initial is None or time is None:
        return None
    u, a, t = (float(m.group(1)) for m in (initial, accel, time))
    return ("final_velocity", u + a * t)


def _permutations(question: str) -> Optional[Tuple[str, float]]:
    """`P(n, k)`, however the corpus words it (v93).

    Must run before `_combination_choose`: "permutations of 9 things taken 2
    at a time" contains `taken`, and that parser would return C(9, 2) = 36
    for a question whose answer is 72 -- the confident wrong verdict this
    module exists to avoid. So any question that says permutation, arrange or
    ordered is claimed here first, and a form this cannot read returns None
    rather than falling through to the combination reading.
    """

    if not re.search(r"permutation|arrang|ordered|line up", question, re.I):
        return None
    number = r"(\d+)"
    # (pattern, reversed) -- `reversed` says the phrasing states k before n,
    # as `_combination_choose` does. Every corpus and natural form is listed;
    # the order matters only where two could match, and none do.
    for pattern, reverse in (
        (rf"n\s*=\s*{number}\s*k\s*=\s*{number}", False),
        (rf"n\s*{number}\s*,?\s*k\s*{number}", False),
        (rf"of\s*{number}\s*things taken\s*{number}", False),
        (rf"of\s*{number}\s*(?:taken|take)\s*{number}", False),
        (rf"{number}\s*things and line up\s*{number}", False),
        (rf"can\s*{number}\s*items? be arranged in order from\s*{number}", True),
        (rf"arrangements? of\s*{number}\s*(?:are there )?from\s*{number}", True),
        (rf"arrange\s*{number}\s*out of\s*{number}", True),
        (rf"arranging\s*{number}\s*out of\s*{number}", True),
        (rf"{number}\s*items? can be picked from\s*{number}", True),
        (rf"selections? of\s*{number}\s*from\s*{number}", True),
    ):
        match = re.search(pattern, question, re.I)
        if not match:
            continue
        a, b = int(match.group(1)), int(match.group(2))
        n, k = (b, a) if reverse else (a, b)
        if k > n:
            return None   # far more likely misread than genuinely 0
        import math as _math

        return ("permutations", float(_math.perm(n, k)))
    return None


# -- connectome lookups (v93) ------------------------------------------------
#
# The three cns_* shapes ask for a fact, not a computation, so the only honest
# re-derivation is the same lookup the corpus builder did: the answer comes
# from `build_connectome_corpus.population()`, built from the CC-BY male-CNS
# tables. A type or pair outside that population returns None (not checked)
# rather than 0, because "not in the table we teach from" is not "zero
# neurons". Where the tables are absent the parsers are inert.

#: A type name as the corpus writes it: letters, digits, `_` and `-`. The
#: trailing `[A-Za-z0-9_]` keeps a sentence-final hyphen or the `?` out.
_TYPE = r"([A-Za-z][A-Za-z0-9_\-]*[A-Za-z0-9_]|[A-Za-z])"


def _connectome_population():
    try:
        import build_connectome_corpus as cns
    except Exception:  # noqa: BLE001 - the checker degrades, it does not break
        return None
    try:
        if not cns.data_available():
            return None
        return cns.population()
    except Exception:  # noqa: BLE001
        return None


def _cns_pair_synapses(question: str) -> Optional[Tuple[str, float]]:
    if not re.search(r"synapse", question, re.I) or not re.search(r"male CNS", question, re.I):
        return None
    for pattern in (
        rf"from (?:cell )?type {_TYPE} (?:to|onto) (?:cell )?type {_TYPE}",
        rf"type {_TYPE} makes? onto type {_TYPE}",
        rf"from {_TYPE} to {_TYPE}\b",
    ):
        match = re.search(pattern, question)
        if match:
            break
    else:
        return None
    pop = _connectome_population()
    if pop is None:
        return None
    pre, post = match.group(1), match.group(2)
    for a, b, weight in pop.pairs:
        if a == pre and b == post:
            return ("cns_pair_synapses", float(weight))
    return None


def _cns_side_count(question: str) -> Optional[Tuple[str, float]]:
    if not re.search(r"neurons?", question, re.I) or not re.search(r"male CNS", question, re.I):
        return None
    side = re.search(r"\b(left|right)\b", question, re.I)
    if side is None:
        return None
    name = _cns_type_name(question)
    if name is None:
        return None
    pop = _connectome_population()
    if pop is None or name not in pop.n_neurons:
        return None
    table = pop.left if side.group(1).lower() == "left" else pop.right
    return ("cns_side_count", float(table[name]))


def _cns_type_count(question: str) -> Optional[Tuple[str, float]]:
    if not re.search(r"neurons?", question, re.I) or not re.search(r"male CNS", question, re.I):
        return None
    if re.search(r"\b(left|right|side|hemisphere|synapse)", question, re.I):
        return None   # a side count or a pair count, not this shape
    name = _cns_type_name(question)
    if name is None:
        return None
    pop = _connectome_population()
    if pop is None or name not in pop.n_neurons:
        return None
    return ("cns_type_count", float(pop.n_neurons[name]))


def _cns_type_name(question: str) -> Optional[str]:
    """The one type a count question names, or None when the form is unknown."""

    for pattern in (
        rf"(?:cell )?type {_TYPE}",
        rf"(?:the |many )?{_TYPE} neurons",
    ):
        match = re.search(pattern, question)
        if match:
            return match.group(1)
    return None


def _combination_choose(question: str) -> Optional[Tuple[str, float]]:
    """`n choose k`, however the corpus words it.

    The corpus fixes k at 2 so the working can be shown, but this reads
    whatever k is stated rather than assuming it -- an assumption here would
    produce a confident wrong verdict on any other k.
    """

    if not re.search(r"combination|choose|chosen|taken", question, re.I):
        return None
    number = r"(\d+)"
    # (pattern, reversed) -- `reversed` says the phrasing states k before n.
    # v82: this used to pick n = max(a, b) for *every* phrasing, so "30 choose
    # 40" returned C(40, 30) = 847660528 where the truth is 0. A size heuristic
    # cannot tell an impossible question from a reversed one; the word order
    # can, and each of the corpus's four phrasings has a fixed order.
    for pattern, reverse in (
        (rf"{number}\s*choose\s*{number}", False),
        (rf"n\s*=\s*{number}\s*k\s*=\s*{number}", False),
        (rf"of\s*{number}\s*things taken\s*{number}", False),
        (rf"can\s*{number}\s*items? be chosen from\s*{number}", True),
    ):
        match = re.search(pattern, question, re.I)
        if not match:
            continue
        a, b = int(match.group(1)), int(match.group(2))
        n, k = (b, a) if reverse else (a, b)
        if k > n:
            # C(n, k) is 0 here, but a question asking to choose 40 from 30 is
            # far more likely to be one this parser has misread than one whose
            # answer is genuinely 0. NOT CHECKED is the honest verdict.
            return None
        import math as _math

        return ("combination", float(_math.comb(n, k)))
    return None


def _arithmetic_series(question: str) -> Optional[Tuple[str, float]]:
    """Sum of the first n terms of an arithmetic progression."""

    if not re.search(r"arithmetic (?:series|progression)", question, re.I):
        return None
    # v82: "An arithmetic series starts at 15 with common difference 4" never
    # says "first term". Measured 24/40 before adding `starts at`, 40/40 after.
    first = re.search(r"(?:first term|starts? at|beginning at)\s*(?:is\s*)?(-?\d+)",
                      question, re.I)
    difference = re.search(r"(?:common )?difference\s*(?:of\s*)?(-?\d+)", question, re.I)
    terms = re.search(r"(?:sum of|first)\s*(\d+)\s*terms|(?:\bn\s*(\d+))", question, re.I)
    if not (first and difference and terms):
        return None
    count = int(terms.group(1) or terms.group(2))
    a, d = int(first.group(1)), int(difference.group(1))
    last = a + (count - 1) * d
    return ("arithmetic_series", float(count * (a + last) / 2))


def _word_problem(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(
        r"has\s+(\d+).*?get\s+(\d+)\s+more.*?give\s+away\s+(\d+)", question, re.I | re.S
    )
    if not match:
        return None
    start, gain, lose = (int(match.group(i)) for i in (1, 2, 3))
    return ("word_problem", float(start + gain - lose))


#: Order matters, and every entry below is placed against a specific ambiguity:
#:
#: * `_word_problem` and `_average` precede everything numeric, because both
#:   contain bare numbers a naive "a + b" search would seize on.
#: * `_sequence` precedes them too -- "7, 17, 27, 37" is a comma-separated list
#:   of numbers, which is exactly what an average looks like.
#: * `_two_step` precedes `_percent` because it *contains* a percent question.
#: * `_algebra` precedes `_multiplication` because `x` is both this corpus's
#:   multiplication sign and its unknown.
#: * `_kinetic_energy` and `_science` precede the bare-number parsers, because
#:   a physics question carries two numbers and a naive `a x b` search would
#:   seize on them without knowing which law applies. `_kinetic_energy` runs
#:   first of the two: it names a mass and a velocity, which is also what
#:   `momentum` matches on.
def _code_trace(question: str) -> Optional[Tuple[str, float]]:
    """Re-derive a code-tracing answer by executing the snippet in the question.

    Every other parser here re-implements the arithmetic it checks. This one
    does not need to: the question *contains* the program, so the check is to
    run it. That makes it the strongest verifier in this module -- there is no
    second implementation to drift from the first.

    Extraction is deliberately narrow. The snippet is recovered by finding the
    first assignment to a single-letter variable and taking everything from
    there to the trailing question, then the same allowlisted, builtins-free,
    timeout-bounded executor the corpus builder uses decides the value. A
    question this cannot confidently parse returns ``None`` -- *not checked* --
    which is the only safe failure for this module.
    """

    try:
        import build_code_corpus as code
    except Exception:  # noqa: BLE001 - the checker degrades, it does not break
        return None

    # The wrapper names the variable being asked about. Every template phrases
    # it one of these ways, and a question naming none of them is not a code
    # question this should touch.
    target = None
    for pattern in (r"value of ([A-Za-z_]\w*)", r"give (?:the value of )?([A-Za-z_]\w*)",
                    r"what (?:is|does) ([A-Za-z_]\w*)", r"final ([A-Za-z_]\w*)",
                    r"([A-Za-z_]\w*) after this runs", r"([A-Za-z_]\w*) hold at the end",
                    r"\b([A-Za-z_]\w*)\?\s*$"):
        found = re.search(pattern, question, flags=re.IGNORECASE)
        if found:
            target = found.group(1)
            break
    if target is None:
        return None

    # The code is the longest span that parses AND assigns that variable. Trying
    # spans rather than pattern-matching the wrapper means a new phrasing cannot
    # silently produce a wrong value: a mis-trimmed span fails to parse, and a
    # span that parses but never assigns the target is rejected below.
    # The right-hand side of the first assignment is a literal (list, number,
    # negative number) or, since v93's `code_range_sum` (`r = sum(range(3,
    # 9))`), a call to a lower-case name. Which names are callable is the
    # executor's decision, not this pattern's: anything else is refused there.
    start = re.search(r"[A-Za-z_]\w*\s*=\s*(?:[\[\-\d]|[a-z]+\()", question)
    if start is None:
        return None
    body = question[start.start():]

    # Candidate spans end at a whitespace boundary, longest first. An earlier
    # version required the span to end in a digit or a bracket, which silently
    # excluded `code_conditional` -- its snippet ends `else b - a`, on a
    # variable name -- and left two of nine tasks unverifiable. Correctness
    # comes from the executor accepting the span, not from guessing where the
    # prose starts, so the heuristic is gone and only the boundary remains.
    pieces = body.split(" ")
    for size in range(len(pieces), 0, -1):
        chunk = " ".join(pieces[:size]).rstrip(" ?.")
        if not chunk or not re.search(rf"\b{re.escape(target)}\s*=", chunk):
            continue
        result = code.run_snippet(chunk, target)
        if result.ok and result.value is not None:
            return ("code_trace", float(result.value))
    return None


PARSERS: Tuple[Callable[[str], Optional[Tuple[str, float]]], ...] = (
    # First: a code question contains numbers and operators that several
    # parsers below would happily misread as arithmetic. `x = 5` and
    # `range(4)` look like a binary expression to `_binary`.
    _code_trace,
    # v93: the connectome lookups key on "male CNS" and refuse everything
    # else, and a type name such as `5-HTPLP01` or `IN17A052` is exactly the
    # kind of string `_binary` and `_multiplication` would misread as an
    # expression, so they run before any numeric parser.
    _cns_pair_synapses,
    _cns_side_count,
    _cns_type_count,
    _word_problem,
    # v93: before `_combination_choose`, which would otherwise read
    # "permutations of 9 things taken 2 at a time" as C(9, 2).
    _permutations,
    _combination_choose,
    _arithmetic_series,
    _kinetic_energy,
    # v93: before `_science`, whose `acceleration` law matches "accelerates"
    # and whose `work` law matches "work out the spring energy"; both fall
    # through on a missing quantity today, and these make that not matter.
    _spring_energy,
    _final_velocity,
    _science,
    _sequence,
    _average,
    _two_step,
    _algebra,
    _percent,
    _division,
    _multiplication,
    _binary,
)


def parse_question(question: str) -> Optional[Tuple[str, float]]:
    for parser in PARSERS:
        try:
            result = parser(question)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if result is not None:
            return result if math.isfinite(result[1]) else None
    return None


def check(question: str, reply: str) -> Optional[Check]:
    """Verify a reply, or return ``None`` when the question is not checkable.

    ``None`` means *not checked*. The caller must not render it as correct: the
    whole value of this is that a wrong answer is visibly wrong, and quietly
    passing anything unrecognised would destroy that.
    """

    parsed = parse_question(question)
    if parsed is None:
        return None
    task, expected = parsed
    predicted = extract_answer(reply)
    correct = (
        predicted is not None
        and abs(predicted - expected) <= max(TOLERANCE, abs(expected) * 1e-6)
    )
    return Check(task=task, expected=expected, predicted=predicted, correct=correct)


def supported_shapes() -> List[str]:
    """The question forms this can verify, for the interface to advertise.

    Through v81 this listed only the nine arithmetic shapes while `PARSERS`
    already handled twelve science and mathematics shapes, so the interface
    under-advertised what it could check by more than half. Every entry here is
    asserted parseable by `test_answer_check.py`, so the list cannot drift
    ahead of the parsers again -- but note it can still drift *behind* them,
    which is the harmless direction.
    """

    return [
        # arithmetic (v74)
        "Solve this basic math problem: 617 + 288",
        "What is 25% of 840?",
        "Solve for x: x + 14 = 39",
        "A student has 45 marbles. They get 38 more and then give away 27. How many marbles do they have now?",
        "Find the average (mean) of these numbers: 40, 60, 20, 80",
        "What is 25 x 7?",
        "Quick question: 70 / 5",
        "What comes next in the sequence: 7, 17, 27, 37?",
        "What is 50% of 698, then add 28?",
        # physics (v79/v80)
        "Given mass 25 kg and acceleration 4 m/s^2, compute the force.",
        "A force of 580 N acts on a mass of 116 kg. What is the acceleration?",
        "mass 98 kg velocity 3 m/s find momentum",
        "What is the kinetic energy of a 12 kg mass moving at 5 m/s?",
        "Find the work done by 98 N acting over 2 m.",
        "What power corresponds to 1860 joules in 20 seconds?",
        "A current of 5 A flows through a resistance of 57 ohms. What is the voltage?",
        "A device runs at 12 V drawing 3 A. What is the electrical power?",
        "A wave with frequency 40 Hz has wavelength 6 m. What is its speed?",
        # chemistry (v79/v80)
        "What is the molarity of 4 mol of solute dissolved in 2 L?",
        # mathematics (v80)
        "In how many ways can 2 items be chosen from 30?",
        "An arithmetic series starts at 15 with common difference 4. "
        "What is the sum of the first 8 terms?",
        # code tracing (v87) -- checked by RUNNING the snippet rather than by
        # re-implementing it, which makes it the only parser here with no
        # second implementation to drift from the first.
        #
        # All nine code tasks are covered, measured at 108 checked / 0 wrong
        # / 0 unchecked over twelve samples each. Ordinary prose is still left
        # alone: "how are you today" and "x marks the spot" both return None.
        "What does x hold at the end? x = 2\nfor i in range(6): x = x + 8",
        "nums = [4, 10, 15]\nr = sum(nums) Give r.",
        "nums = [8, 3, 4]\nr = nums[0] + nums[2] What is r?",
        # v93 science and mathematics (build_omni_corpus.V93_TASKS)
        "A force of 80 N acts for 6 s. What is the impulse?",
        "A voltage of 376 V is applied across 8 ohm. What is the current?",
        "A spring of constant 18 N/m is stretched 4 m. Find the spring energy.",
        "Find the number of permutations of 9 things taken 2 at a time.",
        "A body moving at 12 m/s accelerates at 3 m/s^2 for 8 s. "
        "What is its final velocity?",
        # v93 code tracing (build_code_corpus.V93_TASKS): still one parser,
        # `_code_trace`, which now also runs `range(a, b)`, `.count` and a
        # negative index because the executor's allowlist admits them.
        "Trace this Python and give r. r = sum(range(3, 9))",
        "nums = [2, 4, 7, 4]\nr = nums.count(4) What is r?",
        "nums = [13, 8, 10]\nr = nums[-2] Give r.",
        # v93 connectome lookups (build_connectome_corpus.TASKS): the truth is
        # read from the CC-BY male-CNS tables the corpus was built from, so
        # these parse only where `datasets/v91_malecns` is on disk.
        "How many neurons of type KCg-m are in the male CNS?",
        "How many neurons of type KCg-m are on the left side of the male CNS?",
        "How many synapses go from type KCg-m to type PAM08 in the male CNS?",
    ]
