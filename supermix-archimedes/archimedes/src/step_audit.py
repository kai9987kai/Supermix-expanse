"""Locate the step where a worked reply stops being arithmetic.

Accuracy says a reply is wrong. It does not say *which* of the five or six
operations inside it went wrong, and for a model this size that is the only
question worth asking. This module reads a reply the way the benchmark's
extractor does -- as text, with no access to the generator -- and checks every
operation it can pin down against exact arithmetic.

Two kinds of step turn up, and the distinction is what this module exists for.

A **written** step states its own operands: ``70 x 5 = 350``. It can be checked
on its own, and a false one is an arithmetic slip at a known position.

An **unwritten** step is one the format performs without writing. Every
decomposed task in this corpus has one. ``multiplication`` emits
``70 x 5 = 350, 9 x 5 = 45, total 395`` and never writes ``350 + 45 = 395``;
``percent`` emits ``times 10 = 60.3, times 5 = 30.15, total 90.45`` and never
writes the sum either. The reply's final number therefore rests on an operation
the model had to do silently, and neither the corpus nor the benchmark has ever
recorded how hard those silent operations are.

`carry_load` scores an unwritten step by whether it needs a carry or a borrow,
because the partials a decomposition produces are usually disjoint in their
digits -- ``350 + 45`` touches the hundreds-and-tens and the tens-and-units, and
the tens column is ``5 + 4``, no carry -- and a carry-free sum is closer to
concatenation than to addition.

**That hypothesis was wrong.** `force`, `multiplication` and `voltage` carry in
a sixth of their silent steps and score 1.000; `two_step`, `power` and
`algebra_one_step` never carry and score below 0.45. The measurement is kept
because it is the one that ruled the idea out, and because the difficulty of a
silent step is still worth reporting even though it does not predict accuracy.

What the module found instead is in the verdicts. Sorting v86's 139 wrong
replies by where they first go wrong put seven tasks entirely in
``arithmetic_slip`` -- a false step at a known position -- and put `average` and
`percent`, the only two tasks whose format performs an operation without writing
it, in the column where every written step is true and the answer is still
wrong. See `docs/V87_WHERE_THE_POINTS_WENT.md`.

Nothing here executes generated text and nothing consults the generator: a
reply is scored from its own characters, so the audit applies equally to corpus
rows and to model output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Sequence, Tuple

# Matches the benchmark's own tolerance (`eval_problem_solving.is_correct`), so
# a step this module calls false is one the benchmark would also call false.
TOLERANCE = 1e-6

NUMBER = r"-?\d+(?:\.\d+)?"

#: ``a OP b = c``. ``x`` is the corpus's multiplication sign; ``*`` and the
#: unicode multiplication sign are accepted because model output is not bound
#: by the corpus's conventions.
BINARY = re.compile(rf"({NUMBER})\s*([+\-x*×/])\s*({NUMBER})\s*=\s*({NUMBER})")

#: ``1 percent of 603 = 6.03``, the anchor every percent reply starts from.
ONE_PERCENT = re.compile(rf"1 percent of ({NUMBER})\s*=\s*({NUMBER})")

#: ``half of 56 = 28``, used by `combination` and `kinetic_energy`.
HALF_OF = re.compile(rf"half of ({NUMBER})\s*=\s*({NUMBER})")

#: ``times 10 = 60.3``. The left operand is implicit -- it is whatever the
#: previous step produced -- so this shape is only checkable in context.
TIMES = re.compile(rf"times ({NUMBER})\s*=\s*({NUMBER})")

#: ``divide by 6, total 34.333333``: the divisor and the quotient are separated
#: by the word `total`, which is also how the reply ends. Kept as one pattern so
#: the quotient is not mistaken for the reply's answer.
DIVIDE_BY = re.compile(rf"divide by ({NUMBER}),\s*total ({NUMBER})")

#: ``sum: 54 then 126 then 174``: running values with no operator between them.
#: Every addition in this shape is unwritten by construction.
RUNNING = re.compile(rf"sum:\s*({NUMBER}(?:\s+then\s+{NUMBER})+)")

TOTAL = re.compile(rf"total\s+({NUMBER})")

OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "x": lambda a, b: a * b,
    "*": lambda a, b: a * b,
    "×": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else None,
}


def _decimal(text: str) -> Optional[Decimal]:
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= TOLERANCE * max(1.0, abs(right))


# ---------------------------------------------------------------------------
# Carry load
# ---------------------------------------------------------------------------


def carry_load(left: float, right: float, operator: str = "+") -> Optional[int]:
    """How many digit columns of ``left OP right`` need a carry or a borrow.

    Zero means the operands are disjoint in their digits and the result can be
    assembled column by column without any information crossing between
    columns. ``350 + 45`` is such a sum: the tens column is ``5 + 4``, which
    does not reach ten, so nothing propagates. ``54 + 72`` is not: the units
    column reaches twelve and the tens column must be told.

    The distinction matters because a decomposition's two partials are usually
    disjoint on purpose -- ``70 x 5`` and ``9 x 5`` put their results in
    different columns -- and a model that can concatenate is not necessarily a
    model that can add.

    Returns ``None`` for anything this cannot decide (non-finite input, or an
    operator other than ``+``/``-``), so an undecidable step is never silently
    counted as easy.
    """

    if operator not in ("+", "-"):
        return None
    a = _decimal(_plain(left))
    b = _decimal(_plain(right))
    if a is None or b is None:
        return None
    if operator == "-":
        b = -b
    # A signed pair is an addition of like signs or a subtraction; normalise to
    # whichever it actually is, because a borrow and a carry are different
    # column events and only one of them can apply.
    if (a >= 0) == (b >= 0):
        return _carries(abs(a), abs(b))
    big, small = (abs(a), abs(b)) if abs(a) >= abs(b) else (abs(b), abs(a))
    return _borrows(big, small)


def _plain(value) -> str:
    """A decimal string with no exponent, for `Decimal` to read exactly."""

    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, int):
        return str(value)
    return format(Decimal(repr(float(value))), "f")


def _columns(left: Decimal, right: Decimal) -> Tuple[List[int], List[int]]:
    """Both operands as digit lists, aligned on the decimal point."""

    scale = max(-left.as_tuple().exponent, -right.as_tuple().exponent, 0)
    factor = Decimal(10) ** scale
    a, b = int(left * factor), int(right * factor)
    width = max(len(str(a)), len(str(b)))
    return ([int(d) for d in str(a).zfill(width)],
            [int(d) for d in str(b).zfill(width)])


def _carries(left: Decimal, right: Decimal) -> int:
    a, b = _columns(left, right)
    carry, count = 0, 0
    for x, y in zip(reversed(a), reversed(b)):
        column = x + y + carry
        carry = 1 if column >= 10 else 0
        count += carry
    return count


def _borrows(big: Decimal, small: Decimal) -> int:
    a, b = _columns(big, small)
    borrow, count = 0, 0
    for x, y in zip(reversed(a), reversed(b)):
        column = x - y - borrow
        borrow = 1 if column < 0 else 0
        count += borrow
    return count


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One operation the reply performs, written or not."""

    kind: str          # binary | one_percent | half | times | divide | running | closing
    text: str          # the characters the step was read from ("" when unwritten)
    operator: str
    left: float
    right: float
    stated: Optional[float]   # what the reply says the result is; None when unwritten
    expected: float           # what the arithmetic gives
    written: bool
    position: int

    @property
    def ok(self) -> bool:
        return self.stated is None or _close(self.stated, self.expected)

    @property
    def carries(self) -> Optional[int]:
        return carry_load(self.left, self.right, self.operator)


@dataclass
class Audit:
    """Everything the audit could establish about one reply."""

    reply: str
    steps: List[Step] = field(default_factory=list)
    answer: Optional[float] = None
    verdict: str = "unreadable"
    unwritten_note: str = ""

    @property
    def written(self) -> List[Step]:
        return [s for s in self.steps if s.written]

    @property
    def unwritten(self) -> List[Step]:
        return [s for s in self.steps if not s.written]

    @property
    def first_bad(self) -> Optional[Step]:
        for step in self.steps:
            if not step.ok:
                return step
        return None

    @property
    def hard_unwritten(self) -> List[Step]:
        """Unwritten steps whose columns actually interact."""

        return [s for s in self.unwritten if (s.carries or 0) > 0]


def _read_written(reply: str) -> List[Step]:
    """Every step the reply states operands for, in the order they appear."""

    steps: List[Step] = []
    anchor: Optional[float] = None   # the value a relative step multiplies

    events: List[Tuple[int, str, tuple]] = []
    for match in BINARY.finditer(reply):
        events.append((match.start(), "binary", match.groups() + (match.group(0),)))
    for match in ONE_PERCENT.finditer(reply):
        events.append((match.start(), "one_percent", match.groups() + (match.group(0),)))
    for match in HALF_OF.finditer(reply):
        events.append((match.start(), "half", match.groups() + (match.group(0),)))
    for match in TIMES.finditer(reply):
        events.append((match.start(), "times", match.groups() + (match.group(0),)))
    for match in DIVIDE_BY.finditer(reply):
        events.append((match.start(), "divide", match.groups() + (match.group(0),)))
    for match in RUNNING.finditer(reply):
        events.append((match.start(), "running", (match.group(1), match.group(0))))
    events.sort()

    # `1 percent of 603 = 6.03` also matches nothing else, but `9 x 9 = 81`
    # inside `velocity squared = 9 x 9 = 81` matches BINARY twice over the same
    # characters in some replies; dropping a step that starts inside the span of
    # the one before it keeps each operation counted once.
    consumed = -1
    for start, kind, groups in events:
        if start < consumed:
            continue
        text = groups[-1]
        consumed = start + len(text)
        if kind == "binary":
            left, operator, right, stated = groups[0], groups[1], groups[2], groups[3]
            expected = OPERATIONS[operator](float(left), float(right))
            if expected is None:
                continue
            step = Step(kind, text, operator, float(left), float(right),
                        float(stated), expected, True, len(steps))
        elif kind == "one_percent":
            base, stated = float(groups[0]), float(groups[1])
            step = Step(kind, text, "/", base, 100.0, stated, base / 100.0,
                        True, len(steps))
        elif kind == "half":
            base, stated = float(groups[0]), float(groups[1])
            step = Step(kind, text, "/", base, 2.0, stated, base / 2.0,
                        True, len(steps))
        elif kind == "running":
            # `sum: 54 then 126 then 174` writes running values and no
            # operands, so every addition inside it is unwritten. The addends
            # are still recoverable -- each is the difference between
            # neighbouring values -- which is what the carry load needs. There
            # is nothing independent to check the stated values against, so
            # these steps carry no verdict, only a difficulty.
            values = [float(v) for v in re.findall(NUMBER, groups[0])]
            for index in range(1, len(values)):
                previous, current = values[index - 1], values[index]
                steps.append(Step("running", "", "+", previous, current - previous,
                                  None, current, False, len(steps)))
            if values:
                anchor = values[-1]
            continue
        elif kind == "times":
            if anchor is None:
                continue
            factor, stated = float(groups[0]), float(groups[1])
            step = Step(kind, text, "x", anchor, factor, stated, anchor * factor,
                        True, len(steps))
        else:  # divide
            if anchor is None:
                continue
            divisor, stated = float(groups[0]), float(groups[1])
            if divisor == 0:
                continue
            step = Step(kind, text, "/", anchor, divisor, stated, anchor / divisor,
                        True, len(steps))
        steps.append(step)
        # A relative step multiplies the *anchor*, not the step before it: in
        # `1 percent of 603 = 6.03, times 10 = 60.3, times 5 = 30.15` both
        # `times` steps scale the one-percent value. Only an absolute step moves
        # the anchor.
        if kind in ("binary", "one_percent", "half"):
            anchor = step.stated
    return steps


def audit(reply: str) -> Audit:
    """Read a reply and locate every operation in it.

    The verdict is one of:

    ``clean``
        every written step is true and the closing total is the result of one
        of them, so the answer was derived in writing.
    ``arithmetic_slip``
        a written step states a false result. `first_bad` gives the earliest.
    ``unwritten_step``
        every written step is true, but the closing total is not any of their
        results. The reply's answer rests on an operation it never wrote.
    ``unsupported``
        the closing total does not follow and nothing was written to support
        it, which is a bare assertion rather than a decomposition.
    """

    result = Audit(reply=reply)
    result.steps = _read_written(reply)
    totals = TOTAL.findall(reply)
    if totals:
        result.answer = float(totals[-1])

    bad = result.first_bad
    if bad is not None:
        result.verdict = "arithmetic_slip"
        return result

    written = result.written
    if result.answer is None:
        result.verdict = "unreadable"
        return result
    if not written:
        result.verdict = "unsupported"
        return result

    # Summing the partials is tried before "the total repeats a written value",
    # because the two explanations collide whenever a partial is zero:
    # `600 - 600 = 0, 76 - 5 = 71, total 71` both repeats a written 71 and sums
    # to it. The sum is the operation that actually happened, and reading it as
    # a repeat would drop the reply out of the carry buckets exactly when the
    # silent addition was easiest -- biasing the comparison this module exists
    # to make.
    combined = _explain_as_sum(written, result.answer)
    if combined is not None:
        parts, total = combined
        left = parts[0]
        for value in parts[1:]:
            step = Step("closing", "", "+", left, value, None, left + value,
                        False, len(result.steps))
            result.steps.append(step)
            left += value
        result.verdict = "unwritten_step"
        result.unwritten_note = (
            f"total {total:g} = " + " + ".join(f"{p:g}" for p in parts)
        )
        return result

    if any(step.stated is not None and _close(step.stated, result.answer)
           for step in written):
        result.verdict = "clean"
        return result

    # The total follows from nothing that was written and no sum of the
    # partials reaches it. The reply asserted its answer.
    result.verdict = "unsupported"
    return result


def _explain_as_sum(written: Sequence[Step],
                    answer: float) -> Optional[Tuple[List[float], float]]:
    """Find the trailing run of written results that adds up to the answer.

    Only a *contiguous suffix* is tried. Searching every subset would find a
    coincidence in almost any reply with five numbers in it, and a coincidence
    named as the model's reasoning is worse than an unexplained gap.
    """

    values = [s.stated for s in written if s.stated is not None]
    for size in range(2, len(values) + 1):
        window = values[-size:]
        if _close(sum(window), answer):
            return list(window), answer
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise(replies: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate audits over a set of scored replies.

    Each reply is a mapping with at least ``task``, ``reply`` and ``correct``.
    The returned table is per task, and its point is the last two columns:
    accuracy when the reply's silent step was carry-free, against accuracy when
    it was not.
    """

    tasks: Dict[str, Dict[str, object]] = {}
    for record in replies:
        task = str(record.get("task", "?"))
        report = audit(str(record.get("reply", "")))
        correct = bool(record.get("correct"))
        bucket = tasks.setdefault(task, {
            "n": 0, "correct": 0,
            "verdicts": {}, "written_steps": 0, "false_steps": 0,
            "correct_with_false_step": 0, "wrong_with_sound_steps": 0,
            "carry_free": {"n": 0, "correct": 0},
            "needs_carry": {"n": 0, "correct": 0},
        })
        bucket["n"] += 1
        bucket["correct"] += int(correct)
        verdicts = bucket["verdicts"]
        verdicts[report.verdict] = verdicts.get(report.verdict, 0) + 1
        bucket["written_steps"] += len(report.written)
        false_steps = sum(0 if s.ok else 1 for s in report.written)
        bucket["false_steps"] += false_steps
        # The two cells that say whether the scratchpad is load-bearing.
        #
        # A right answer above false working means the answer did not come from
        # the working -- v86 replies to `420 / 7` with `320 / 7 = 60, total 60`,
        # and 60 is right while 320/7 is not 60. A wrong answer above sound
        # working is the opposite failure: every step true and the conclusion
        # still missed, which is a planning error rather than an arithmetic one.
        if correct and false_steps:
            bucket["correct_with_false_step"] += 1
        if not correct and report.written and not false_steps:
            bucket["wrong_with_sound_steps"] += 1
        if report.unwritten:
            side = "needs_carry" if report.hard_unwritten else "carry_free"
            bucket[side]["n"] += 1
            bucket[side]["correct"] += int(correct)

    for task, bucket in tasks.items():
        wrong = bucket["n"] - bucket["correct"]
        bucket["wrong"] = wrong
        # Where a wrong reply first goes wrong. `false_step` is an arithmetic
        # error at a known position; `sound_steps` means every written step is
        # true and the error is in an operation the format does not write.
        bucket["wrong_at_a_written_step"] = wrong - bucket["wrong_with_sound_steps"]
        bucket["accuracy"] = round(bucket["correct"] / max(1, bucket["n"]), 4)
        bucket["decorative_working_rate"] = (
            round(bucket["correct_with_false_step"] / bucket["correct"], 4)
            if bucket["correct"] else None
        )
        for side in ("carry_free", "needs_carry"):
            entry = bucket[side]
            entry["accuracy"] = (round(entry["correct"] / entry["n"], 4)
                                 if entry["n"] else None)
    return tasks


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read what `eval_problem_solving --dump_replies` writes and locate the errors.

        python source/eval_problem_solving.py --checkpoint C --novel 630 \
            --dump_replies replies.jsonl
        python source/step_audit.py --replies replies.jsonl
    """

    import argparse
    import json

    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--replies", required=True,
                        help="JSONL from `eval_problem_solving --dump_replies`")
    parser.add_argument("--task", default=None,
                        help="print every wrong reply for this task, with the "
                             "first false step marked")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    with open(args.replies, encoding="utf-8") as handle:
        replies = [json.loads(line) for line in handle if line.strip()]
    table = summarise(replies)

    if args.task:
        for record in replies:
            if record.get("task") != args.task or record.get("correct"):
                continue
            report = audit(str(record.get("reply", "")))
            bad = report.first_bad
            print(f"Q: {record.get('prompt')}")
            print(f"A: {record.get('reply')}")
            if bad is not None:
                print(f"   step {bad.position}: '{bad.text}' is "
                      f"{bad.stated:g}, arithmetic gives {bad.expected:g}")
            elif report.unwritten:
                print(f"   every written step is true; {len(report.unwritten)} "
                      f"operation(s) were performed without being written")
            print(f"   expected {record.get('expected')}, "
                  f"answered {record.get('predicted')}\n")
        return 0

    header = (f"{'task':20s} {'n':>4s} {'acc':>6s} {'wrong':>5s} "
              f"{'at a written step':>17s} {'every step true':>16s} {'right+false':>11s}")
    print(header)
    print("-" * len(header))
    for name, entry in sorted(table.items(), key=lambda kv: kv[1]["accuracy"]):
        print(f"{name:20s} {entry['n']:4d} {entry['accuracy']:6.3f} "
              f"{entry['wrong']:5d} {entry['wrong_at_a_written_step']:17d} "
              f"{entry['wrong_with_sound_steps']:16d} "
              f"{entry['correct_with_false_step']:11d}")

    total = sum(e["n"] for e in table.values())
    correct = sum(e["correct"] for e in table.values())
    decorative = sum(e["correct_with_false_step"] for e in table.values())
    print(f"\n{correct}/{total} correct. {decorative} correct replies "
          f"({decorative / max(1, correct):.1%}) stand above a false written "
          f"step, which is the rate at which the working is decoration.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"schema": "supermix-v87-step-audit-v1",
                       "replies": args.replies, "tasks": table}, handle, indent=2)
        print(f"\nreceipt -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
