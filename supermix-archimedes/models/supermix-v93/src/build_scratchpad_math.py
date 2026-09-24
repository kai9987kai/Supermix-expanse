"""v66: arithmetic that shows its working.

V65 removed the blocker that made arithmetic unrepresentable -- digits are now
separate tokens -- and the model went from emitting no number a quarter of the
time to always answering in the right format at roughly the right magnitude:
51.5 where the truth is 51.333, 374.8 where it is 349.8. Exact accuracy stayed
near zero.

That failure has a shape. Right magnitude with wrong digits is what a model
produces when it is guessing the answer in a single step instead of computing it.
The standard remedy is to make the intermediate work part of the target, so the
model learns a procedure it can execute rather than a mapping it must approximate.

The decomposition here is place-value, in two steps, chosen because it is always
arithmetically valid rather than merely usually valid:

    524 - 305  ->  500 - 300 = 200, 24 - 5 = 19, total 219
    504 - 309  ->  500 - 300 = 200, 4 - 9 = -5, total 195

The second case is the reason for the design. A column method needs borrow
handling and can produce negative digits that have to be carried; splitting into
hundreds and remainder cannot fail, because the two partial results simply add.
A generator that is right only most of the time would teach the model to be wrong
in exactly the cases that are hardest.

The final number of every answer is still the answer, so
`eval_problem_solving.py` scores these models unchanged and v65 and v66 are
directly comparable on the same benchmark.

    python source/build_scratchpad_math.py --output datasets/v66/scratchpad_240k.jsonl --target 240000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

RECEIPT_SCHEMA = "supermix-v66-scratchpad-math-v1"

#: Whether `percent` writes the addition of its two partial products.
#:
#: **Off, because v87 measured this and it made the task worse**, 0.476 -> 0.286.
#: The reasoning for turning it on was sound and the diagnosis behind it still
#: holds: ten of v86's fourteen wrong percent replies had every written step true
#: and a total that followed from none of them, because the final sum was the one
#: operation the format never wrote.
#:
#: Writing it did not make it doable. v87 now gets both parts right and fails the
#: addition it was forced to state:
#:
#:     1 percent of 24 = 0.24, times 10 = 2.4, times 5 = 1.2,
#:     2.4 + 1.2 = 3.2                                    (truth 3.6)
#:
#: These are decimal additions that carry across the point, and carrying is where
#: this model breaks: over all 630 v86 replies a written +/- step that needs a
#: carry is false 0.186 of the time against 0.104 for one that does not
#: (Fisher exact p = 0.033).
#:
#: The lesson v87 paid for: making a step explicit only helps when the model can
#: perform it. An operation left implicit is sometimes reached another way; an
#: operation written down must be executed, and a wrong one poisons the total.
#:
#: The coverage fix in `_scratchpad_percent` is **not** reverted with this. That
#: one closed a real hole -- 12% and 15% appear in a third of the benchmark's
#: percent problems and in none of v86's 40,000 percent rows -- and is orthogonal.
PERCENT_WRITTEN_SUM = False


#: Whether `average` and `percent` show working for their *inner* operations.
#:
#: Off reproduces the v66-v70 corpora exactly. On applies v68's rule to the two
#: tasks that still guessed a step in one shot -- which are also the two v70
#: scored worst on, 33.3% and 58.3% against 91.7-100% for the decomposed tasks.
DECOMPOSE_INNER = False

#: Whether `average` is constrained so its mean always terminates.
#:
#: **Off. Untrained, and it narrows the benchmark as well as the corpus.**
#:
#: The generator draws four to six values and divides by the count. Four and
#: five always terminate; six repeats whenever the sum is not a multiple of
#: three, so **22% of `average` problems ask for a value like
#: `59.333333333333336`**.
#:
#: `build_omni_corpus` states the principle this violates:
#:
#:     Where a task cannot produce a value a digit-level model could plausibly
#:     learn, it is left out rather than padded with noise -- training on
#:     `294.1995` teaches the shape of a float, not physics.
#:
#: Measured on v88, five of `average`'s six wrong replies fail at the final
#: division, and the mismatches are exactly this shape: expected
#: `59.333333333333336`, answered `59.0`; expected `77.5`, answered
#: `76.833333`.
#:
#: On, the last value moves by at most two so the sum is a multiple of three,
#: and no mean repeats. They still reach two decimal places -- dividing by four
#: gives `.25` -- which is exactly representable and is not what this fixes. No
#: randomness is consumed, so the arm cannot shift any later row.
#:
#: **It does NOT narrow the benchmark, and v89 paid for my believing it did.**
#: The omni tasks share their generators with `eval_problem_solving`, so a flag
#: there changes both sides together. The scratchpad tasks do not:
#: `eval_problem_solving._average` is its own function and still asks for
#: repeating means. Turning this on therefore trains the model on a narrower
#: distribution than it is tested on -- a train/eval mismatch of exactly the
#: kind `coverage_audit.py` exists to catch, except that audit compares operand
#: value sets and this is a difference in the shape of the answer.
#:
#: v89 ran with it on. `average` sat at **0/4 on every probe for the whole run**
#: where v88 had reached 0.714, and the replies show structural errors -- an
#: operand added twice, dividing four numbers by five -- that v88 did not make.
#: The mechanism is not understood and is not guessed at here.
#:
#: If this is ever used again, `eval_problem_solving._average` must apply the
#: same constraint, which is the declared benchmark-narrowing
#: `COMBINATION_IN_ENVELOPE` documents. Until then, leave it off.
AVERAGE_TERMINATES = False

#: Whether `average` writes its running total as explicit binary additions.
#:
#: Off reproduces the v66-v81 corpora exactly, including the v80 build. On
#: replaces the one-shot chain
#:
#:     sum: 61 then 124 then 196 then 257, total 257, divide by 4, total 64.25
#:
#: with
#:
#:     61 + 63 = 124, 124 + 72 = 196, 196 + 61 = 257, total 257,
#:     divide by 4, total 64.25
#:
#: The old form states each running total with **neither operand shown**, which
#: is the fixed-dependency scratchpad Cho et al. and Lee et al. 2023 (arXiv
#: 2307.03381, Fig. 8) identify as the failing design: the model must recompute
#: `previous + next` internally and only writes the result. Lee et al. measure
#: 100.0% against 89.9% between two scratchpads that differ only in whether the
#: hard step's operands are written down.
#:
#: **Hypothesis, not a result.** `average` has scored 0.033 under the current
#: format and nothing here has been trained. What *is* measured is the token
#: cost, in the `--token_budget_report` table.
AVERAGE_BINARY_STEPS = False

#: Whether `two_step` resolves its percentage to an exact unit fraction before
#: applying the second operation.
#:
#: Off reproduces v74-v86. On changes
#:
#:     1 percent of 320 = 3.2, times 25 = 80, then 80 - 17 = 63
#:
#: into
#:
#:     25 percent is one quarter, 320 / 4 = 80, then 80 - 17 = 63
#:
#: V86's paired benchmark put `two_step` at 0.300, down from v80's 0.633, and a
#: live failure computed `3.2 x 25 = 85`. The alternative makes the dependency
#: explicit using the exact division shapes the same checkpoint scores at 1.0.
#: It is an untrained v87 arm, not a claimed improvement.
TWO_STEP_DIVISION_TRACE = False

#: Whether raw `average` and `two_step` prompts use a deterministic mixture of
#: semantically equivalent forms. The choice comes from a separate hash of the
#: row identity, so enabling the arm never consumes the problem RNG and cannot
#: silently change operands, answers, or later rows.
#:
#: Off leaves the baseline generator stream unchanged. It is deliberately paired
#: with a held-out paraphrase benchmark in `eval_prompt_robustness.py`.
PROMPT_PARAPHRASES = False

#: Whether `algebra_one_step` resolves the sign in words and splits the arithmetic.
#:
#: Off reproduces the v66-v81 corpora exactly. On changes two things, both of
#: which the observed failures point at:
#:
#: 1. **The sign is resolved in words.** `x + -12 = 22` becomes "add 12 to both
#:    sides", not "subtract -12 from both sides". A model that has to carry a
#:    negative through a subtraction has two chances to get the sign wrong; the
#:    English form has none. Half of all rows have `b < 0` here.
#: 2. **The arithmetic is decomposed by place value**, the way `subtraction`
#:    and `multiplication` already are -- and `multiplication` is the one task
#:    scoring 1.00. The observed failure is a borrow inside one jump:
#:    `x + 29 = 34` produced "34 - 29 = 4" where the truth is 5.
#:
#: The split is tens-and-remainder rather than a column method, so it can never
#: need a borrow: `30 - 20 = 10, 4 - 9 = -5, total 5`. A negative low part is
#: correct and is exactly what the `subtraction` task already emits.
#:
#: **Hypothesis, not a result.** `algebra_one_step` scored 0.30 under the
#: current format; nothing here has been trained.
ALGEBRA_WORD_SIGN = False

#: How the question can be phrased. Matching `build_english_math_dataset` keeps
#: the prompt distribution familiar, so the only thing that changes is the answer.
PROMPT_FORMS = (
    "Solve this basic math problem: {expression}",
    "Quick question: {expression}",
    "Please help with this. {expression}",
    "What is {expression}?",
)

PERCENT_FRACTIONS = {
    10: ("one tenth", 10),
    20: ("one fifth", 5),
    25: ("one quarter", 4),
    50: ("one half", 2),
}

V87_TRAIN_PROMPT_TEMPLATES = {
    "average": (
        "Find the average (mean) of these numbers: {csv}",
        "Could you work out the mean of {natural}?",
        "What is the average for {csv}?",
        "Calculate the mean value of {natural}.",
        "For the numbers {csv}, what is their mean?",
    ),
    "two_step": (
        "What is {pct}% of {base}, then {word} {delta}?",
        "Take {pct} percent of {base} and then {word} {delta}.",
        "Start with {fraction} of {base}, then {natural_word} {delta}. What remains?",
        "After finding {pct}% of {base}, {follow_word} {delta}. What is the result?",
        "Compute {pct} percent of {base}; next, {word} {delta}.",
    ),
}


def _natural_join(values: Sequence[int]) -> str:
    rendered = [str(value) for value in values]
    if len(rendered) == 2:
        return " and ".join(rendered)
    return ", ".join(rendered[:-1]) + ", and " + rendered[-1]


def _prompt_variant_index(seed: int, row_index: int, task: str, count: int) -> int:
    """Choose a prompt form without advancing the problem generator."""

    identity = f"supermix-v87:{seed}:{row_index}:{task}".encode("utf-8")
    digest = hashlib.blake2b(identity, digest_size=8).digest()
    return int.from_bytes(digest, "big") % count


def _paraphrase_prompt(
    item: Dict[str, Any], *, seed: int, row_index: int
) -> Tuple[str, str]:
    """Return a training paraphrase and its stable template id."""

    task = str(item.get("task") or "")
    fields = dict(item.get("_prompt_fields") or {})
    templates = V87_TRAIN_PROMPT_TEMPLATES.get(task)
    if not templates or not fields:
        return str(item["expression"]), "canonical"
    index = _prompt_variant_index(seed, row_index, task, len(templates))
    return templates[index].format(**fields), f"{task}.train.{index}"


def _split_place_value(value: int) -> Tuple[int, int]:
    """Return ``(hundreds_part, remainder)`` so the two sum back to ``value``."""

    hundreds = (abs(value) // 100) * 100
    if value < 0:
        hundreds = -hundreds
    return hundreds, value - hundreds


def _split_tens(value: int) -> Tuple[int, int]:
    """Return ``(tens_part, remainder)`` so the two sum back to ``value``.

    The same trick as :func:`_split_place_value`, one order of magnitude down,
    for the two-digit operands `algebra_one_step` works with. Splitting them at
    hundreds would emit ``0 - 0 = 0`` on every row and decompose nothing.
    """

    tens = (abs(value) // 10) * 10
    if value < 0:
        tens = -tens
    return tens, value - tens


def _scratchpad_addition(rng: random.Random) -> Dict[str, Any]:
    a, b = rng.randint(100, 999), rng.randint(10, 999)
    return _scratchpad_binary(a, b, "+", a + b)


def _scratchpad_subtraction(rng: random.Random) -> Dict[str, Any]:
    a, b = rng.randint(100, 999), rng.randint(10, 999)
    return _scratchpad_binary(a, b, "-", a - b)


def _scratchpad_binary(a: int, b: int, op: str, answer: int) -> Dict[str, Any]:
    a_hundreds, a_rest = _split_place_value(a)
    b_hundreds, b_rest = _split_place_value(b)
    if op == "+":
        high, low = a_hundreds + b_hundreds, a_rest + b_rest
    else:
        high, low = a_hundreds - b_hundreds, a_rest - b_rest
    # The invariant the whole design rests on. If it ever fails the corpus is
    # teaching wrong arithmetic, which is worse than teaching none.
    assert high + low == answer, f"decomposition broke for {a} {op} {b}"

    expression = f"{a} {op} {b}"
    working = (
        f"{a_hundreds} {op} {b_hundreds} = {high}, "
        f"{a_rest} {op} {b_rest} = {low}, total {answer}"
    )
    return {"expression": expression, "answer": answer, "working": working,
            "task": "addition" if op == "+" else "subtraction"}


def _terminating_mean(values: List[int]) -> List[int]:
    """Nudge the last value so the mean has at most one decimal place.

    Only a division by six can repeat here: fours and fives always terminate,
    and `sum / 6` is exact when the sum is a multiple of three. So the last
    value moves by at most two, which keeps it inside the generator's own
    5..99 range and leaves the other values untouched.

    No randomness is consumed, so enabling the arm cannot shift any later row.
    """

    if len(values) % 3 == 0 and sum(values) % 3:
        adjusted = list(values)
        shift = -(sum(values) % 3)
        if adjusted[-1] + shift < 5:
            shift += 3
        adjusted[-1] += shift
        return adjusted
    return values


def _scratchpad_average(rng: random.Random) -> Dict[str, Any]:
    # Four to seven values, not four to five.
    #
    # v67 scored 0% on average, and reading its working showed why: the benchmark
    # generates 4, 5 or 6 values while this emitted only 4 or 5, so every
    # six-number problem was out of distribution. The model did the only thing it
    # could and truncated -- "sum: 28 then 63 then 111 then 141 then 158, total
    # 158, divide by 5" for a six-number prompt -- failing roughly a third of the
    # task by construction. Going to seven covers the benchmark's range with a
    # margin rather than matching its edge exactly.
    # 4-6 matches the benchmark's range exactly. v67 emitted 4-5 against a
    # benchmark testing 4-6 and every six-number problem was out of
    # distribution; the margin to 7 was added then. With decomposed working
    # a seven-value row is long enough to be dropped by turn-aligned packing,
    # which would silently reintroduce that same gap at the top of the range.
    values = [rng.randint(5, 99) for _ in range(rng.choice([4, 5, 6]))]
    if AVERAGE_TERMINATES:
        values = _terminating_mean(values)
    total = sum(values)
    answer = total / len(values)
    joined = ", ".join(str(v) for v in values)
    prompt_fields = {
        "csv": joined,
        "natural": _natural_join(values),
    }

    # Decompose only what fits the sequence budget.
    #
    # v72 measured sequence length as expensive -- 128 to 160 cost 24 points of
    # accuracy on identical data -- so the corpus should fit 128 rather than the
    # run growing to fit the corpus. Decomposed working for six values does not:
    # 81% of those rows exceed 128 tokens and turn-aligned packing drops them,
    # which is the out-of-distribution gap that made v67's average score zero.
    #
    # Six-value rows therefore keep the terse format. Both lengths stay in
    # distribution, and the decomposition applies wherever it is affordable.
    # Reverted to the terse format. Decomposition was worth 4x at sequence
    # length 160, and at 128 it measured *worse* over 500 problems: v70's
    # terse format 24.0% against v73's decomposed 16.0%. The seq-160 result
    # was measured against a floor low enough that almost anything looked
    # like an improvement.
    if DECOMPOSE_INNER and False:
        # v70 scored 33.3% on average against 91.7% on addition, and the
        # difference is the format: addition decomposes each sum into place
        # values, average merely lists the results. v68 named the rule -- a
        # scratchpad helps only where it decomposes the operation -- and this
        # applies it, so every running addition shows the same working the
        # addition task already does.
        # Split the *running total* and add the new value to its remainder.
        #
        # Splitting both operands the way `addition` does would emit
        # "0 + 0 = 0" on every step, because each value here is two-digit and
        # its hundreds part is always zero. Two thirds of the working would be
        # noise the model has to learn to produce. Splitting only the
        # accumulator keeps both steps carrying real arithmetic.
        steps: List[str] = []
        accumulated = 0
        for value in values:
            high, rest = _split_place_value(accumulated)
            partial = rest + value
            accumulated = high + partial
            assert accumulated == sum(values[: len(steps) + 1])
            # Show the hard half, state the easy half.
            #
            # The two-digit addition is where the running sum actually drifts;
            # re-adding the hundreds is prefixing a digit. Writing both doubled
            # the row length and pushed 83.3% of six-value averages past the
            # sequence limit, where turn-aligned packing drops them -- silently
            # recreating the out-of-distribution gap v67 was fixed for.
            steps.append(f"{rest} + {value} = {partial}, running {accumulated}")
        return {
            "expression": f"Find the average (mean) of these numbers: {joined}",
            "answer": answer,
            "working": (
                "; ".join(steps)
                + f"; total {total}, divide by {len(values)}, total {round(answer, 6)}"
            ),
            "task": "average",
            "raw_prompt": True,
            "_prompt_fields": prompt_fields,
        }

    if AVERAGE_BINARY_STEPS:
        # Every addition written as `a + b = c`, with both operands present.
        #
        # The terse form below states running totals only, so the model has to
        # do each addition in its head and write nothing but the result. That
        # is the fixed-dependency scratchpad the cited work identifies as the
        # one that does not work, and `average` has scored 0.033 under it while
        # `word_problem` -- which writes `a + b = c` in full -- scored 0.867 on
        # the same model.
        #
        # The running total does leave the two-digit envelope: with six values
        # of 5-99 it can reach 594, so the last steps are a three-digit plus a
        # two-digit number. That is a real risk and it is not hidden here; the
        # alternative is to also decompose each step by place value, which v73
        # measured as *worse* at sequence length 128 (16.0% against 24.0%),
        # because the rows then stop fitting.
        steps: List[str] = []
        accumulated = values[0]
        for value in values[1:]:
            nxt = accumulated + value
            steps.append(f"{accumulated} + {value} = {nxt}")
            accumulated = nxt
        assert accumulated == total, "binary running sum disagreed with the total"
        working = (
            ", ".join(steps)
            + f", total {total}, divide by {len(values)}, total {round(answer, 6)}"
        )
        return {"expression": f"Find the average (mean) of these numbers: {joined}",
                "answer": answer, "working": working, "task": "average",
                "raw_prompt": True, "_prompt_fields": prompt_fields}

    running: List[str] = []
    accumulated = 0
    for value in values:
        accumulated += value
        running.append(str(accumulated))
    # Six decimals, not four.
    #
    # At four, an average like 100/3 is written 33.3333 while the truth is
    # 33.33333...; the difference is 3.3e-05 and the benchmark's relative
    # tolerance is 3.3e-05, so a model reproducing the corpus exactly scored
    # *wrong*. The task was unwinnable as posed, which is a corpus defect rather
    # than a model failure, and is the likeliest reason v66's average accuracy
    # was a flat 0%. Six decimals puts the stated answer two orders of magnitude
    # inside tolerance.
    working = (
        f"sum: {' then '.join(running)}, total {total}, "
        f"divide by {len(values)}, total {round(answer, 6)}"
    )
    return {"expression": f"Find the average (mean) of these numbers: {joined}",
            "answer": answer, "working": working, "task": "average", "raw_prompt": True,
            "_prompt_fields": prompt_fields}


def _scratchpad_percent(rng: random.Random) -> Dict[str, Any]:
    # These must cover `eval_problem_solving._percent`, which draws from
    # [5, 10, 12, 15, 20, 25]. Until v87 this list was [5, 10, 20, 25, 50], so
    # 12 and 15 appeared in a third of the benchmark's percent problems and in
    # none of the corpus's forty thousand rows. Split by percentage, v86 scored
    # 16/17 on the four it had been taught and 0/13 on the two it had not; the
    # task's reported 0.533 was that average and was read as difficulty for
    # eight versions. 50 is kept -- the benchmark never asks for it, but
    # dropping a percentage the model already handles buys nothing.
    #
    # Anything added here must survive DECOMPOSE_INNER's tens-and-units split,
    # which 12 (10 + 2) and 15 (10 + 5) do.
    pct, base = rng.choice([5, 10, 12, 15, 20, 25, 50]), rng.randint(20, 2000)
    one_percent = base / 100
    answer = pct * one_percent
    if DECOMPOSE_INNER:
        # The multiply was the one-shot step. v68 measured this exact failure:
        # "1 percent of 1049 = 10.49, times 10, total 104.9" -- one percent
        # computed *exactly*, then multiplied by the wrong number. Splitting the
        # multiplier into tens and units gives that step working to show.
        tens, units = (pct // 10) * 10, pct % 10
        part_tens, part_units = one_percent * tens, one_percent * units
        assert abs((part_tens + part_units) - answer) < 1e-9
        # Emit only the parts that exist. 5% splits to tens=0, and 20% to
        # units=0; writing "times 0 = 0.0" would make the model learn to produce
        # a term that is always noise, and two thirds of the multipliers here
        # have a zero half.
        written = [(factor, value) for factor, value
                   in ((tens, part_tens), (units, part_units)) if factor]
        parts = [f"times {factor} = {round(value, 6)}" for factor, value in written]
        if len(written) == 2 and PERCENT_WRITTEN_SUM:
            # Write the sum of the parts.
            #
            # Until v87 this step was silent, and the v86 replies show exactly
            # what that cost. Asked for 15% of 56 the model wrote
            # `times 10 = 5.6, times 5 = 2.8, total 1.8`: both parts right, and
            # then 5.6 + 2.8 produced 1.8. Ten of the fourteen wrong percent
            # replies have every written step true and a total that follows
            # from none of them.
            #
            # `decompose_product` already writes its running sums once there
            # are more than two partials, for the same reason. Two partials
            # there cannot carry -- they are disjoint place values -- but these
            # two can: 5.6 and 2.8 collide in the tenths.
            first, second = written[0][1], written[1][1]
            parts.append(f"{round(first, 6)} + {round(second, 6)} = "
                         f"{round(first + second, 6)}")
        working = (
            f"1 percent of {base} = {round(one_percent, 6)}, "
            + ", ".join(parts)
            + f", total {round(answer, 6)}"
        )
    else:
        working = (
            f"1 percent of {base} = {round(one_percent, 4)}, "
            f"times {pct}, total {round(answer, 4)}"
        )
    return {"expression": f"What is {pct}% of {base}?", "answer": answer,
            "working": working, "task": "percent", "raw_prompt": True}


def _scratchpad_algebra(rng: random.Random) -> Dict[str, Any]:
    """One-step linear equations, solved by undoing the operation.

    The working states the inverse operation before performing it, because the
    step that has to generalise is "subtract the constant from both sides", not
    the subtraction itself. Operand ranges match `eval_problem_solving._algebra`
    so the benchmark is in distribution.
    """

    x, b = rng.randint(-30, 30), rng.randint(-30, 30)
    right = x + b
    if ALGEBRA_WORD_SIGN:
        # Resolve the sign in English, then decompose the arithmetic.
        #
        # `x + -12 = 22` is undone by *adding* 12, and saying so removes the
        # double negative entirely rather than asking the model to evaluate
        # `22 - -12`. The subtraction is then split tens-and-remainder, which
        # is the shape `multiplication` uses and scores 1.00 with.
        word, preposition, op = (
            ("subtract", "from", "-") if b >= 0 else ("add", "to", "+")
        )
        magnitude = abs(b)
        right_tens, right_rest = _split_tens(right)
        magnitude_tens, magnitude_rest = _split_tens(magnitude)
        if b >= 0:
            high, low = right_tens - magnitude_tens, right_rest - magnitude_rest
        else:
            high, low = right_tens + magnitude_tens, right_rest + magnitude_rest
        # The invariant the decomposition rests on. If it ever fails the corpus
        # teaches wrong arithmetic, which is worse than teaching none.
        assert high + low == x, f"decomposition broke for x + {b} = {right}"
        working = (
            f"{word} {magnitude} {preposition} both sides, "
            f"{right_tens} {op} {magnitude_tens} = {high}, "
            f"{right_rest} {op} {magnitude_rest} = {low}, total {x}"
        )
    else:
        working = f"subtract {b} from both sides, {right} - {b} = {x}, total {x}"
    return {"expression": f"Solve for x: x + {b} = {right}", "answer": float(x),
            "working": working, "task": "algebra_one_step", "raw_prompt": True}


def _scratchpad_word_problem(rng: random.Random) -> Dict[str, Any]:
    """Two-step word problems, worked in the order the sentence states them.

    The chain is written out rather than collapsed -- ``a + b = c, c - d = e`` --
    so the second step consumes the first step's result. That is the part a model
    has to learn; a single combined expression would let it pattern-match the
    final number instead.
    """

    start, gain, lose = rng.randint(20, 99), rng.randint(5, 60), rng.randint(5, 60)
    item = rng.choice(["notebooks", "cookies", "marbles", "stickers"])
    after_gain = start + gain
    answer = after_gain - lose
    working = (
        f"{start} + {gain} = {after_gain}, {after_gain} - {lose} = {answer}, "
        f"total {answer}"
    )
    return {
        "expression": (
            f"A student has {start} {item}. They get {gain} more and then give "
            f"away {lose}. How many {item} do they have now?"
        ),
        "answer": float(answer),
        "working": working,
        "task": "word_problem",
        "raw_prompt": True,
    }



def _scratchpad_multiplication(rng: random.Random) -> Dict[str, Any]:
    """Two-digit times one-digit, split over the multiplicand's place values.

    `(tens + units) * b == tens*b + units*b` holds for every pair, so the
    decomposition can never be the thing that is wrong.
    """

    a, b = rng.randint(11, 99), rng.randint(2, 9)
    tens, units = (a // 10) * 10, a % 10
    part_tens, part_units = tens * b, units * b
    answer = a * b
    assert part_tens + part_units == answer
    return {
        "expression": f"{a} x {b}",
        "answer": float(answer),
        "working": (
            f"{tens} x {b} = {part_tens}, {units} x {b} = {part_units}, total {answer}"
        ),
        "task": "multiplication",
    }


def _scratchpad_division(rng: random.Random) -> Dict[str, Any]:
    """Exact division, built backwards so every partial divides cleanly.

    The quotient is chosen first and split into tens and units; the dividend is
    then `b * (tens + units)`. Both partials are therefore exact by
    construction, which a forward-generated problem could not guarantee.
    """

    b = rng.randint(2, 9)
    quotient = rng.randint(11, 60)
    tens, units = (quotient // 10) * 10, quotient % 10
    part_tens, part_units = b * tens, b * units
    dividend = part_tens + part_units
    assert dividend == b * quotient
    return {
        "expression": f"{dividend} / {b}",
        "answer": float(quotient),
        "working": (
            f"{part_tens} / {b} = {tens}, {part_units} / {b} = {units}, total {quotient}"
        ),
        "task": "division",
    }


def _scratchpad_sequence(rng: random.Random) -> Dict[str, Any]:
    """Next term of an arithmetic sequence.

    The step the model has to learn is *finding* the difference, so the working
    states it before using it. Nothing here is a lookup: the terms are new every
    row.
    """

    start, step = rng.randint(3, 40), rng.randint(2, 12)
    terms = [start + step * i for i in range(4)]
    answer = terms[-1] + step
    joined = ", ".join(str(t) for t in terms)
    return {
        "expression": f"What comes next in the sequence: {joined}?",
        "answer": float(answer),
        "working": (
            f"{terms[1]} - {terms[0]} = {step}, difference {step}, "
            f"{terms[-1]} + {step} = {answer}, total {answer}"
        ),
        "task": "sequence",
        "raw_prompt": True,
    }


def _scratchpad_two_step(rng: random.Random) -> Dict[str, Any]:
    """A percentage followed by an addition or subtraction.

    Two different operations chained, where the second consumes the first's
    result. This is the closest thing here to a problem that cannot be answered
    by recognising a single pattern.
    """

    pct = rng.choice([10, 20, 25, 50])
    base = rng.choice([x for x in range(40, 900) if (x * pct) % 100 == 0])
    first = base * pct // 100
    delta = rng.randint(5, 60)
    add = rng.random() < 0.5
    answer = first + delta if add else first - delta
    op, word = ("+", "add") if add else ("-", "subtract")
    fraction, divisor = PERCENT_FRACTIONS[pct]
    if TWO_STEP_DIVISION_TRACE:
        working = (
            f"{pct} percent is {fraction}, {base} / {divisor} = {first}, "
            f"then {first} {op} {delta} = {answer}, total {answer}"
        )
    else:
        working = (
            f"1 percent of {base} = {base / 100:g}, times {pct} = {first}, "
            f"then {first} {op} {delta} = {answer}, total {answer}"
        )
    return {
        "expression": f"What is {pct}% of {base}, then {word} {delta}?",
        "answer": float(answer),
        "working": working,
        "task": "two_step",
        "raw_prompt": True,
        "_prompt_fields": {
            "pct": pct,
            "base": base,
            "word": word,
            "delta": delta,
            "fraction": fraction,
            "natural_word": "add" if add else "take away",
            "follow_word": "increase it by" if add else "reduce it by",
        },
    }


GENERATORS: Tuple[Callable[[random.Random], Dict[str, Any]], ...] = (
    _scratchpad_addition,
    _scratchpad_subtraction,
    _scratchpad_average,
    _scratchpad_percent,
    _scratchpad_algebra,
    _scratchpad_word_problem,
    # v74: problem shapes that cannot be answered by recognising one pattern.
    _scratchpad_multiplication,
    _scratchpad_division,
    _scratchpad_sequence,
    _scratchpad_two_step,
)


def build_rows(target: int, seed: int = 66) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for index in range(target):
        item = GENERATORS[index % len(GENERATORS)](rng)
        if item.get("raw_prompt"):
            user = item["expression"]
            variant = "canonical"
            if PROMPT_PARAPHRASES:
                user, variant = _paraphrase_prompt(item, seed=seed, row_index=index)
        else:
            user = PROMPT_FORMS[rng.randrange(len(PROMPT_FORMS))].format(
                expression=item["expression"]
            )
            variant = "legacy_prompt_form"
        row = {
            "user": user,
            "assistant": item["working"],
            "topic": "basic_math",
            "task": item["task"],
        }
        if PROMPT_PARAPHRASES and item.get("raw_prompt"):
            row["prompt_variant"] = variant
        rows.append(row)
    rng.shuffle(rows)
    return rows


def write(rows: Sequence[Dict[str, Any]], output: Path) -> Dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    words = [len((r["user"] + " " + r["assistant"]).split()) for r in rows]
    words.sort()
    return {
        "schema": RECEIPT_SCHEMA,
        "output": str(output),
        "rows": len(rows),
        "tasks": dict(Counter(r["task"] for r in rows)),
        "median_words": words[len(words) // 2],
        "p95_words": words[int(0.95 * len(words))],
        "format_flags": {
            "decompose_inner": DECOMPOSE_INNER,
            "average_binary_steps": AVERAGE_BINARY_STEPS,
            "average_terminates": AVERAGE_TERMINATES,
            "algebra_word_sign": ALGEBRA_WORD_SIGN,
            "two_step_division_trace": TWO_STEP_DIVISION_TRACE,
            "prompt_paraphrases": PROMPT_PARAPHRASES,
        },
        "non_claims": [
            "Showing working is not reasoning. The model may learn to imitate the "
            "steps without the steps constraining the final number.",
            "The decomposition is two-step place value, which is valid for every "
            "case but is not how a person carries. Nothing here claims it is the "
            "best scratchpad, only that it is always correct.",
            "Every answer is generated, so the corpus is exactly as diverse as its "
            "generator and no more.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(REPO_ROOT / "datasets" / "v66" / "scratchpad_240k.jsonl"))
    parser.add_argument("--target", type=int, default=240000)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument(
        "--decompose-inner",
        action="store_true",
        help="show working for the inner operations of average and percent",
    )
    parser.add_argument(
        "--average_terminates",
        action="store_true",
        help=("constrain average so no mean repeats. 22% of its problems "
              "currently ask for a value like 59.333333333333336, which is "
              "what five of v88's six wrong average replies fail on. NARROWS "
              "THE BENCHMARK TOO -- eval_problem_solving calls this same "
              "generator, so a score measured with this on is not comparable "
              "with v88's average of 0.714."),
    )
    parser.add_argument(
        "--average_binary_steps",
        action="store_true",
        help=("write the average's running total as explicit `a + b = c` steps "
              "instead of bare running totals. The motivation is measured: of "
              "v86's 29 wrong average replies, 16 have every written step true "
              "and a wrong answer, because the running sum is the one part the "
              "format never writes. The effect is not -- no run has used this."),
    )
    parser.add_argument(
        "--algebra_word_sign",
        action="store_true",
        help=("resolve the algebra sign in words and split the arithmetic by "
              "place value. The motivation is measured: all 17 of v86's wrong "
              "algebra replies state the right inverse operation and then get "
              "the arithmetic wrong, and every one of them evaluates a double "
              "negative -- `6 - -12 = 27`, `3 - -15 = 26`. The effect is not: "
              "no run has used this."),
    )
    parser.add_argument(
        "--two_step_division_trace",
        action="store_true",
        help=("resolve 10/20/25/50 percent to exact division before the second "
              "operation (v87 hypothesis; unmeasured)"),
    )
    parser.add_argument(
        "--prompt_paraphrases",
        action="store_true",
        help=("use deterministic training paraphrases for average and two_step "
              "without changing the generated operands (v87 hypothesis)"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    global DECOMPOSE_INNER, AVERAGE_BINARY_STEPS, ALGEBRA_WORD_SIGN
    global AVERAGE_TERMINATES
    global TWO_STEP_DIVISION_TRACE, PROMPT_PARAPHRASES
    DECOMPOSE_INNER = bool(args.decompose_inner)
    AVERAGE_BINARY_STEPS = bool(args.average_binary_steps)
    AVERAGE_TERMINATES = bool(args.average_terminates)
    ALGEBRA_WORD_SIGN = bool(args.algebra_word_sign)
    TWO_STEP_DIVISION_TRACE = bool(args.two_step_division_trace)
    PROMPT_PARAPHRASES = bool(args.prompt_paraphrases)
    rows = build_rows(args.target, args.seed)
    receipt = write(rows, Path(args.output))
    print(json.dumps({k: v for k, v in receipt.items() if k != "non_claims"}, indent=2))
    print()
    for row in rows[:3]:
        print(f"  U: {row['user']}")
        print(f"  A: {row['assistant']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
