"""Generate a scientific training corpus, verified by the NexusMind solver.

## Why this exists

When asked to "deeply expand the model's knowledge" before v74, the honest
answer was that the corpus could not deliver it: the dialogue portion is 19.8%
one repeated fragment, and v74's ten task types are all synthetic arithmetic.
A model trained on that knows arithmetic and nothing about the world.

`nexus_solver.py` changes what is possible. It is a deterministic solver over
`Fraction`/`Decimal` covering mechanics, energy, fluids, thermodynamics,
electromagnetism, waves, chemistry, algebra, geometry and combinatorics, and it
emits a step-by-step derivation with a SHA-256 receipt. It is a **verified
knowledge source**, and it can be asked an unlimited number of questions.

So: generate scientific problems, work them out, and have the solver check
every single one. Any row whose worked answer disagrees with the solver is
dropped, not shipped. The corpus is correct by construction rather than by
assumption -- no previous corpus in this repository had that property.

## Two design decisions taken from earlier failures

**Phrasing varies, because v74 was format-brittle.** v74 scored 0.894 on its
benchmark and got 0 of 5 naturally-typed questions right, because every task
had exactly one template and the model learned the template. Here each task
carries several phrasings, and the one shown to the model is decoupled from the
canonical query handed to the solver. The model sees variety; the oracle always
sees a form it parses.

**The response never ends in a unit.** Answers are extracted as the last number
in the reply. `total 5 m/s^2` extracts as **2**. Every response therefore ends
`total <number>` with nothing after it, and units live in the prose before it.
This is the kind of thing that silently halves a benchmark score.

## What is deliberately excluded

Tasks whose answers are irrational are generated with rounded values and
verified against the solver within a relative tolerance, rather than exactly.
Where a task cannot produce a value a digit-level model could plausibly learn,
it is left out rather than padded with noise -- training on `294.1995` teaches
the shape of a float, not physics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from nexus_solver import solve_problem  # noqa: E402

#: Sequence length every result in this line was produced at.
#:
#: v72 measured raising it from 128 to 160 as costing **24 accuracy points** on
#: identical data, so the corpus fits the run rather than the run growing to fit
#: the corpus. A format change is therefore a budget decision, and
#: `token_budget_report` is what makes the budget visible before a run pays for
#: it.
DEFAULT_SEQUENCE_LENGTH = 128

#: Relative tolerance when comparing a generated answer to the solver's.
#: Exact-rational tasks agree to the bit; the constant-bearing ones (g, pi)
#: are rounded for legibility and must still land within this.
TOLERANCE = 1e-6

#: Standard gravity as the solver uses it, so rounded answers agree.
GRAVITY = 9.80665

#: Consecutive failures to produce a new prompt before a task is judged
#: exhausted. Generous, because a near-full space still yields hits.
EXHAUSTION_MISSES = 5000

#: Whether `combination` keeps every intermediate inside the learned envelope.
#:
#: Off reproduces v80's corpus exactly. The current form multiplies `n` by
#: `n - 1` with `n` up to 60, and `decompose_product` splits only its **first**
#: operand -- so `60 x 59` is emitted as `60 x 59 = 3540, 0 x 59 = 0`, a
#: two-digit by two-digit product in a single jump, followed by `half of 3540 =
#: 1770`, a one-shot halving of a four-digit number. v81's rule is that every
#: intermediate must stay inside two-digit-by-one-digit products; both of these
#: are outside it, and the task scores 0.00.
#:
#: On, the even operand is halved *first* and `n` is capped at 19, so the only
#: product is at most `19 x 9` -- byte-identical in shape to the
#: `multiplication` task that scores 1.00.
#:
#: **This narrows the benchmark as well as the corpus**, because
#: `eval_problem_solving` calls these same generators. A combination score
#: measured with this on is NOT comparable with v80's 0.00, which was measured
#: over n in 4..60. It is the same trade c7041897 already took for
#: `kinetic_energy`, and it must be stated every time the number is quoted.
COMBINATION_IN_ENVELOPE = False

#: Whether prompts are drawn from the wide natural-phrasing bank.
#:
#: Off reproduces v86's corpus exactly: each task shows one of its 4-5 written
#: templates. On, `natural_phrasings.select` widens that to the way a person
#: actually types -- casual register, missing punctuation, lowercase, long-form
#: units, filler openers, numbers before or after the quantity name.
#:
#: The measurement behind it: v74 had exactly one template per task, scored
#: 0.894 on its own benchmark and got **0 of 5** naturally-typed questions
#: right. v86's 4-5 templates per task took that to **14 of 18**. This widens
#: the same lever further.
#:
#: Safe to expand because the phrasing shown to the model is decoupled from the
#: canonical query handed to the solver, so verification is unaffected however
#: the question is worded. And `_pick` performs its narrow `rng.choice` on both
#: paths, so every operand, answer and worked response is bit-identical whether
#: this is on or off -- only the wording moves.
NATURAL_PHRASINGS = False


@dataclass
class OmniProblem:
    task: str
    domain: str
    prompt: str
    response: str
    answer: float
    unit: str
    canonical: str
    params: Dict[str, float] = field(default_factory=dict)

    def to_row(self, keep_canonical: bool = False) -> Dict[str, str]:
        """The training row. ``keep_canonical`` adds the solver's own query.

        Without it, a shipped row cannot be re-verified: the model-facing
        prompt is one of several phrasings and `nexus_solver` parses only the
        canonical form. Measured on the v80 corpus, the solver could re-verify
        **1,256 of 3,000 sampled model-facing prompts (41.9%)**; the rest are
        phrasings it does not parse. Carrying the canonical query makes a
        solver-checked reward loop possible later and costs one short string
        per row.

        Off by default so the emitted corpus is byte-identical to v80's.
        """

        row = {
            "user": self.prompt,
            "assistant": self.response,
            "domain": self.domain,
            "task": self.task,
        }
        if keep_canonical:
            row["canonical"] = self.canonical
        return row


def _number(value: float) -> str:
    """Render a number the way the corpus should read it.

    Integers stay integral: `60`, not `60.0`. A digit-level tokenizer spends
    real capacity on a trailing `.0` that carries no information.
    """

    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{round(value, 4):g}"


def decompose_product(a: int, b: int) -> str:
    """Show the working for `a x b`, by place value.

    v79 learned the physics and failed the arithmetic. Measured on the
    finished model, asking for the force from `mass x acceleration`:

        single-digit operands   12/12 correct
        two-digit                9/12
        three-digit              1/12

    The cause was this module's first version, which wrote
    `167 x 11 = 1837` as a single step. v66 established that this model
    cannot do multi-digit multiplication in one jump; v74's arithmetic
    corpus never asks it to, splitting every product by place value -- and
    it never uses an operand above 99.

    Widening the parameter ranges (needed to stop duplicate prompts) made
    that far worse, since products then reached 24,000 with no working
    shown at all. Two separately-correct fixes interacting badly.

    So the working is written out, with running sums, for operands of any
    size:

        167 x 11 -> "100 x 11 = 1100, 60 x 11 = 660, 7 x 11 = 77,
                     1100 + 660 = 1760, 1760 + 77 = 1837"
    """

    digits = str(abs(a))
    width = len(digits)
    # Every digit position, including zeros. v74 writes `80 x 3 = 240,
    # 0 x 3 = 0` rather than dropping the units term, so the shape is the
    # same for every problem -- and that is the corpus that scored 0.93.
    parts = [int(d) * 10 ** (width - 1 - i) for i, d in enumerate(digits)]
    if a < 0:
        parts = [-p for p in parts]

    pieces = [f"{part} x {b} = {part * b}" for part in parts]
    if len(parts) <= 2:
        # v74's exact form: two partial products, then the total. It does not
        # write the addition out, and the model learned to do it -- adding a
        # running-sum step here would depart from the only format measured to
        # work.
        return ", ".join(pieces)

    running = parts[0] * b
    for part in parts[1:]:
        nxt = running + part * b
        pieces.append(f"{running} + {part * b} = {nxt}")
        running = nxt
    return ", ".join(pieces)


#: Whether the three division tasks split their quotient by place value.
#:
#: **Off, because v87 measured this and it is a design error.** It was turned on
#: for v87 with a real measurement behind it -- accuracy on a single written step
#: falls sevenfold with the number of significant places it must produce, 0.825
#: for `7` against 0.075 for `174` -- and it made all three tasks much worse:
#:
#:     power         0.333 -> 0.048
#:     molarity      0.667 -> 0.333
#:     acceleration  0.762 -> 0.571
#:
#: The reason is that the partial dividends are back-computed from the answer:
#:
#:     model:  6400 / 64 = 100,  420 / 64 = 5,  122 / 64 = 2
#:     truth:  6400 / 64 = 100, 1920 / 64 = 30, 192 / 64 = 3
#:
#: To write `1920` the model must already know the next quotient digit is 30.
#: The format is a valid presentation of a result, not a procedure that can be
#: executed forward, so the model invents dividends and divides them wrongly.
#:
#: `decompose_product` works because it splits an **input**, whose digits are on
#: the page. This splits the **output**. A decomposition helps only when every
#: step it adds is derivable forward from what is already written and inside the
#: model's arithmetic; this satisfies neither.
#:
#: Kept behind a flag rather than deleted so the negative result stays
#: reproducible -- `output/v87_measurements/v87_paired_n630.json` is the receipt.
DECOMPOSE_QUOTIENT = False

#: Whether the three division tasks show long division instead of one jump.
#:
#: **Off. Untrained, and offered as a measured proposal rather than a result.**
#:
#: Division is where v88's remaining errors live. Of the 50 wrong replies in the
#: whole 630-problem benchmark, 24 are the three division tasks -- `power` 15 of
#: 21, `acceleration` 6, `molarity` 3 -- and every one of them fails at the same
#: place, the single one-shot division step:
#:
#:     power = work / time, 5712 / 48 = 114      truth 119
#:
#: `DECOMPOSE_QUOTIENT` above already tried to split this and made it much
#: worse, because each partial dividend was back-computed from the answer. The
#: bar for any replacement is therefore the rule that failure established: every
#: step derivable forward from what is on the page, and inside the model's
#: arithmetic.
#:
#: What changed is the second half. v88's written subtraction is now reliable --
#: 4 of 147 carry-free steps false and 9 of 134 borrowing ones, against v87-era
#: rates of 0.104 and 0.186 -- and multiplication scores 1.000. Long division
#: rests on exactly those two operations, so it is viable now in a way it was
#: not when `decompose_quotient` was written.
#:
#: Measured over 1,200 generated division problems before proposing it:
#:
#:     format               median  p95  max  turns >=128  widest operand
#:     one-shot (today)         31   36   42            0   5 digits
#:     subtract multiples       67   99  107            0   5 digits
#:     long division            79  111  122            0   4 digits
#:
#: `subtract multiples` was rejected on the envelope, not the budget: it asks
#: for `16992 - 11800`, and `subtraction` trains on three digits. Long division
#: keeps 97% of its operands at three digits or fewer because each remainder is
#: bounded by ten times the divisor.
#:
#: What it still does not have is a training run. The digit estimate
#: (`48 into 91 = 1`) is the smallest one available -- one digit, against a
#: two-digit dividend -- but it is an estimate, and nothing here proves the
#: model can make it. Do not enable this and a phrasing arm in the same run.
LONG_DIVISION = False


def long_division(dividend: int, divisor: int, with_total: bool = True) -> str:
    """Show `dividend / divisor` one quotient digit at a time, school method.

    Every number written is already on the page or derived from one that is:

        48 into 57 = 1, 57 - 48 = 9, 48 into 91 = 1, 91 - 48 = 43,
        48 into 432 = 9, 9 x 48 = 432, 432 - 432 = 0, total 119

    `57` is the leading digits of the dividend, `9` is what the subtraction just
    produced, `91` appends the next dividend digit to it. Contrast
    `decompose_quotient`, which needed `1920` -- a number obtainable only by
    already knowing the quotient digit was 30.

    The multiplication line is written only when the digit exceeds one, because
    `1 x 48 = 48` restates its own operand and the subtrahend is visible without
    it. That is worth roughly eight tokens per unit digit, and this format has
    six tokens of headroom against the 128-token turn at its longest.
    """

    text = str(dividend)
    pieces: List[str] = []
    carry = 0
    quotient = ""
    for char in text:
        carry = carry * 10 + int(char)
        digit = carry // divisor
        if digit or quotient:
            quotient += str(digit)
            product = digit * divisor
            # A zero quotient digit still gets a line. Without one the trace
            # jumps from a remainder to a carry two bring-downs later --
            #
            #     76 into 80 = 1, 80 - 76 = 4, 76 into 456 = 6, ... total 106
            #
            # -- and nothing on the page says a digit was skipped, so following
            # it literally gives 16. `test_every_operand_is_already_on_the_page`
            # found this; only a small fraction of quotients carry an internal
            # zero, so it would have survived a spot check of the output.
            pieces.append(f"{divisor} into {carry} = {digit}")
            if digit > 1:
                pieces.append(f"{digit} x {divisor} = {product}")
            # `432 - 432 = 0` states a remainder that is already visible: the
            # product equals the carry on the line above it. Dropping it is
            # worth about eight tokens, and it is what brings the longest turn
            # back under the budget -- three of 18,000 sampled rows reached 130
            # with it, and packing discards those without a word.
            if digit and carry != product:
                pieces.append(f"{carry} - {product} = {carry - product}")
            carry -= product
    body = ", ".join(pieces)
    # The science responses append their own `total`; a second one inside
    # the division would restate the answer and cost tokens for nothing.
    return f"{body}, total {int(quotient or 0)}" if with_total else body


def decompose_quotient(dividend: int, divisor: int) -> str:
    """Show the working for `dividend / divisor`, one quotient place at a time.

    `decompose_product` exists because v79 wrote `167 x 11 = 1837` in one jump
    and scored 0.03. The three division tasks were left whole and have the same
    defect from the other side, which v86 measured without anyone reading it
    that way:

        power         quotient 2-300   0.400
        molarity      quotient 1-60    0.667
        acceleration  quotient 2-60    0.733
        division      quotient 11-60   1.000   <- decomposed

    `division` is the same operation over the same range as the other three and
    scores a full point above `power`. The one thing it does differently is
    split the quotient by place value, so no written step has to produce more
    than one significant digit.

    A sweep over `power` with the format, wording and model held fixed and only
    the quotient's width moving gives the dose-response directly
    (`output/v87_measurements/division_dose_response.json`):

        1-digit quotient   0.725 / 0.750
        2-digit            0.375 / 0.575
        3-digit            0.125 / 0.125

    The divisor's width does almost nothing (0.725 against 0.750), which is
    what makes this a quotient problem rather than an operand problem, and is
    why splitting the *quotient* is the fix rather than splitting the dividend.

        19152 / 76 -> "15200 / 76 = 200, 3800 / 76 = 50, 152 / 76 = 2,
                       200 + 50 = 250, 250 + 2 = 252"

    Every partial divides exactly, by construction rather than by luck: each
    one is `digit * place * divisor`, so its quotient is `digit * place`.
    """

    quotient, remainder = divmod(dividend, divisor)
    if remainder:
        raise ValueError(
            f"{dividend} / {divisor} is not exact; every division task in this "
            "corpus is built backwards from the quotient so that it is"
        )
    digits = str(quotient)
    width = len(digits)
    # Zero places are kept, matching `decompose_product` above. That choice is
    # deliberate there ("the shape is the same for every problem -- and that is
    # the corpus that scored 0.93") and `_scratchpad_division` makes the same
    # one, writing `0 / 7 = 0` when the units digit is zero. Dropping them here
    # would make this the only decomposition in the corpus whose step count
    # varies with the *digits* of the answer rather than its width, and no
    # measurement supports preferring that.
    parts = [int(d) * 10 ** (width - 1 - i) for i, d in enumerate(digits)]
    pieces = [f"{part * divisor} / {divisor} = {part}" for part in parts]
    if len(parts) <= 2:
        # Two partials and no written addition: `_scratchpad_division`'s exact
        # form, which scores 1.000. The sum of two place values cannot carry,
        # so there is nothing for a written step to add.
        return ", ".join(pieces)

    running = parts[0]
    for part in parts[1:]:
        pieces.append(f"{running} + {part} = {running + part}")
        running += part
    return ", ".join(pieces)


def _pick(rng: random.Random, templates: List[str], _task: Optional[str] = None,
          **values) -> str:
    """One phrasing of the question, with the numbers filled in.

    The narrow `rng.choice` happens on **both** paths, so the RNG stream -- and
    therefore every operand, answer and worked response in the corpus -- is
    bit-identical whether or not `NATURAL_PHRASINGS` is on. With the flag on,
    the index that choice produced is fed to `natural_phrasings.select` as part
    of the row's identity instead of being used directly. See that module for
    why a hash rather than a second draw.
    """

    narrow = rng.choice(templates)
    template = narrow
    if NATURAL_PHRASINGS and _task is not None:
        import natural_phrasings  # local: keeps the flag-off path import-free

        template = natural_phrasings.select(_task, templates, narrow, values)
    return template.format(**{k: _number(v) if isinstance(v, (int, float))
                              else v for k, v in values.items()})


# -- task generators --------------------------------------------------------
#
# Each returns a problem whose `canonical` form the solver parses, and whose
# `prompt` is one of several phrasings a person might actually type.


def _force(rng: random.Random) -> OmniProblem:
    mass = rng.randint(11, 99)
    accel = rng.randint(2, 9)
    answer = mass * accel
    prompt = _pick(rng, [
        "A body of mass {m} kg has an acceleration of {a} m/s^2. What is the force?",
        "mass {m} kg acceleration {a} m/s^2 find the force",
        "Find the force on a {m} kg mass accelerating at {a} m/s^2.",
        "What force acts on mass {m} kg with acceleration {a} m/s^2?",
        "Given mass {m} kg and acceleration {a} m/s^2, compute the force.",
    ], m=mass, a=accel, _task="force")
    response = (f"force = mass x acceleration, {decompose_product(mass, accel)}, "
                f"the force is {answer} newtons, total {answer}")
    return OmniProblem("force", "physics", prompt, response, float(answer), "N",
                       f"mass {mass} kg acceleration {accel} m/s^2 find the force",
                       {"m": mass, "a": accel})


def _acceleration(rng: random.Random) -> OmniProblem:
    mass = rng.randint(2, 200)
    accel = rng.randint(2, 60)
    force = mass * accel  # chosen so the division is exact
    prompt = _pick(rng, [
        "A force of {f} N acts on a mass of {m} kg. What is the acceleration?",
        "force {f} N mass {m} kg find acceleration",
        "Find the acceleration produced by {f} N on {m} kg.",
        "What acceleration results from a {f} N force on a {m} kg body?",
    ], f=force, m=mass, _task="acceleration")
    response = (f"acceleration = force / mass, "
                f"{long_division(force, mass, with_total=False) if LONG_DIVISION else decompose_quotient(force, mass) if DECOMPOSE_QUOTIENT else f'{force} / {mass} = {accel}'}, "
                f"the acceleration is {accel} metres per second squared, total {accel}")
    return OmniProblem("acceleration", "physics", prompt, response, float(accel), "m/s^2",
                       f"force {force} N mass {mass} kg find acceleration",
                       {"F": force, "m": mass})


def _momentum(rng: random.Random) -> OmniProblem:
    mass = rng.randint(11, 99)
    velocity = rng.randint(2, 9)
    answer = mass * velocity
    prompt = _pick(rng, [
        "A {m} kg object moves with velocity {v} m/s. What is its momentum?",
        "mass {m} kg velocity {v} m/s find momentum",
        "Find the momentum of a mass {m} kg travelling at velocity {v} m/s.",
        "What is the linear momentum for mass {m} kg and velocity {v} m/s?",
    ], m=mass, v=velocity, _task="momentum")
    response = (f"momentum = mass x velocity, {decompose_product(mass, velocity)}, "
                f"the momentum is {answer} kilogram metres per second, total {answer}")
    return OmniProblem("momentum", "physics", prompt, response, float(answer), "kg*m/s",
                       f"mass {mass} kg velocity {velocity} m/s find momentum",
                       {"m": mass, "v": velocity})


def _kinetic_energy(rng: random.Random) -> OmniProblem:
    # Halve the mass *first*, so the only product is `squared x half_mass` --
    # a two-digit by one-digit multiply, which is v74's proven shape. The
    # previous ranges asked for `25 x 25 = 625` then `30 x 625 = 18750`, both
    # in one step, and the task scored 0.00.
    mass = rng.randrange(2, 20, 2)
    velocity = rng.randint(2, 9)
    squared = velocity * velocity
    half_mass = mass // 2
    answer = half_mass * squared
    prompt = _pick(rng, [
        "A mass of {m} kg moves at velocity {v} m/s. Find the kinetic energy.",
        "mass {m} kg velocity {v} m/s kinetic energy",
        "What is the kinetic energy of a {m} kg body at {v} m/s?",
        "Compute the kinetic energy for mass {m} kg and speed {v} m/s.",
    ], m=mass, v=velocity, _task="kinetic_energy")
    response = (f"kinetic energy = half x mass x velocity squared, "
                f"half of {mass} = {half_mass}, "
                f"velocity squared = {velocity} x {velocity} = {squared}, "
                f"{decompose_product(squared, half_mass)}, "
                f"the kinetic energy is {answer} joules, total {answer}")
    return OmniProblem("kinetic_energy", "physics", prompt, response, float(answer), "J",
                       f"mass {mass} kg velocity {velocity} m/s kinetic energy",
                       {"m": mass, "v": velocity})


def _work(rng: random.Random) -> OmniProblem:
    force = rng.randint(11, 99)
    distance = rng.randint(2, 9)
    answer = force * distance
    prompt = _pick(rng, [
        "A force of {f} N moves an object {d} m. How much work is done?",
        "force {f} N distance {d} m work done",
        "Find the work done by {f} N acting over {d} m.",
        "What work is done when a {f} N force acts through {d} m?",
    ], f=force, d=distance, _task="work")
    response = (f"work = force x distance, {decompose_product(force, distance)}, "
                f"the work done is {answer} joules, total {answer}")
    return OmniProblem("work", "physics", prompt, response, float(answer), "J",
                       f"force {force} N distance {distance} m work done",
                       {"F": force, "d": distance})


def _power(rng: random.Random) -> OmniProblem:
    time = rng.randint(2, 100)
    power = rng.randint(2, 300)
    work = power * time  # exact division
    prompt = _pick(rng, [
        "{w} J of work is done in {t} s. What is the power?",
        "work {w} J time {t} s power",
        "Find the power when {w} J is delivered over {t} s.",
        "What power corresponds to {w} joules in {t} seconds?",
    ], w=work, t=time, _task="power")
    response = (f"power = work / time, "
                f"{long_division(work, time, with_total=False) if LONG_DIVISION else decompose_quotient(work, time) if DECOMPOSE_QUOTIENT else f'{work} / {time} = {power}'}, "
                f"the power is {power} watts, total {power}")
    return OmniProblem("power", "physics", prompt, response, float(power), "W",
                       f"work {work} J time {time} s power",
                       {"W": work, "t": time})


def _voltage(rng: random.Random) -> OmniProblem:
    current = rng.randint(11, 99)
    resistance = rng.randint(2, 9)
    answer = current * resistance
    prompt = _pick(rng, [
        "A current of {i} A flows through {r} ohm. What is the voltage?",
        "current {i} A resistance {r} ohm find voltage",
        "Find the potential difference across {r} ohm carrying {i} A.",
        "What voltage drives {i} A through a {r} ohm resistor?",
    ], i=current, r=resistance, _task="voltage")
    response = (f"voltage = current x resistance, {decompose_product(current, resistance)}, "
                f"the voltage is {answer} volts, total {answer}")
    return OmniProblem("voltage", "physics", prompt, response, float(answer), "V",
                       f"current {current} A resistance {resistance} ohm find voltage",
                       {"I": current, "R": resistance})


def _electrical_power(rng: random.Random) -> OmniProblem:
    voltage = rng.randint(11, 99)
    current = rng.randint(2, 9)
    answer = voltage * current
    prompt = _pick(rng, [
        "A device runs at {v} V drawing {i} A. What is the electrical power?",
        "voltage {v} V current {i} A electrical power",
        "Find the power dissipated at {v} V and {i} A.",
        "What electrical power is used at {v} volts and {i} amps?",
    ], v=voltage, i=current, _task="electrical_power")
    response = (f"power = voltage x current, {decompose_product(voltage, current)}, "
                f"the power is {answer} watts, total {answer}")
    return OmniProblem("electrical_power", "physics", prompt, response, float(answer), "W",
                       f"voltage {voltage} V current {current} A electrical power",
                       {"V": voltage, "I": current})


def _wave_speed(rng: random.Random) -> OmniProblem:
    frequency = rng.randint(11, 99)
    wavelength = rng.randint(2, 9)
    answer = frequency * wavelength
    prompt = _pick(rng, [
        "A wave has frequency {f} Hz and wavelength {w} m. What is its speed?",
        "frequency {f} Hz wavelength {w} m wave speed",
        "Find the speed of a wave of frequency {f} Hz and wavelength {w} m.",
        "What is the wave speed at {f} Hz with a {w} m wavelength?",
    ], f=frequency, w=wavelength, _task="wave_speed")
    response = (f"wave speed = frequency x wavelength, {decompose_product(frequency, wavelength)}, "
                f"the wave speed is {answer} metres per second, total {answer}")
    return OmniProblem("wave_speed", "physics", prompt, response, float(answer), "m/s",
                       f"frequency {frequency} Hz wavelength {wavelength} m wave speed",
                       {"f": frequency, "lambda": wavelength})


def _molarity(rng: random.Random) -> OmniProblem:
    volume = rng.randint(1, 60)
    concentration = rng.randint(1, 60)
    moles = concentration * volume  # exact
    prompt = _pick(rng, [
        "{n} mol of solute is dissolved in {v} L. What is the molarity?",
        "moles {n} mol volume {v} L molarity",
        "Find the concentration of {n} moles in {v} litres.",
        "What is the molar concentration of {n} mol in {v} L of solution?",
    ], n=moles, v=volume, _task="molarity")
    response = (f"molarity = moles / volume, "
                f"{long_division(moles, volume, with_total=False) if LONG_DIVISION else decompose_quotient(moles, volume) if DECOMPOSE_QUOTIENT else f'{moles} / {volume} = {concentration}'}, "
                f"the concentration is {concentration} molar, total {concentration}")
    return OmniProblem("molarity", "chemistry", prompt, response, float(concentration), "M",
                       f"moles {moles} mol volume {volume} L molarity",
                       {"n": moles, "V": volume})


def _combination(rng: random.Random) -> OmniProblem:
    # k is fixed at 2 so the working can actually be shown: C(n,2) is
    # n x (n-1) / 2. The previous version stated `7 choose 2 = 21` with no
    # working at all, leaving the model to memorise the whole table, and it
    # scored 0.00 on 976 rows.
    n = rng.randint(4, 19) if COMBINATION_IN_ENVELOPE else rng.randint(4, 60)
    k = 2
    answer = math.comb(n, k)
    prompt = _pick(rng, [
        "In how many ways can {k} items be chosen from {n}?",
        "n choose k n = {n} k = {k}",
        "Find the number of combinations of {n} things taken {k} at a time.",
        "How many combinations are there of {n} choose {k}?",
    ], n=n, k=k, _task="combination")
    product = n * (n - 1)
    if COMBINATION_IN_ENVELOPE:
        # Halve before multiplying, not after. One of `n` and `n - 1` is
        # always even, so the halving is exact, and it happens while the
        # number is still two-digit instead of four.
        even, odd = (n, n - 1) if n % 2 == 0 else (n - 1, n)
        half = even // 2
        assert odd * half == answer, f"combination decomposition broke for n={n}"
        response = (f"combinations = n x (n - 1) / 2, {n} - 1 = {n - 1}, "
                    f"half of {even} = {half}, "
                    f"{decompose_product(odd, half)}, "
                    f"there are {answer} ways, total {answer}")
    else:
        response = (f"combinations = n x (n - 1) / 2, {n} - 1 = {n - 1}, "
                    f"{decompose_product(n, n - 1)}, "
                    f"half of {product} = {answer}, "
                    f"there are {answer} ways, total {answer}")
    return OmniProblem("combination", "mathematics", prompt, response, float(answer), "",
                       f"n choose k n = {n} k = {k}", {"n": n, "k": k})


def _arithmetic_series(rng: random.Random) -> OmniProblem:
    # Bounded so the worked response fits the 128-token sequence length.
    # Turn-aligned packing DROPS a turn that does not fit, which would
    # silently bias this task toward its shortest problems.
    # Every intermediate stays two-digit, and the term count is even so the
    # halving happens on a small number before the only multiplication. The
    # previous version emitted `5 x 228 = 1140` and `1140 / 2` in single
    # steps and scored 0.00. Two-digit single-step additions are fine:
    # `word_problem` does them and scored 0.87.
    first = rng.randint(1, 20)
    difference = rng.randint(2, 6)
    terms = rng.randrange(4, 12, 2)
    last = first + (terms - 1) * difference
    answer = terms * (first + last) // 2
    prompt = _pick(rng, [
        "An arithmetic series starts at {a} with common difference {d}. "
        "What is the sum of the first {n} terms?",
        "sum of arithmetic series first term {a} common difference {d} n {n}",
        "Find the sum of {n} terms of an arithmetic progression "
        "with first term {a} and difference {d}.",
    ], a=first, d=difference, n=terms, _task="arithmetic_series")
    span = (terms - 1) * difference
    half_terms = terms // 2
    ends = first + last
    response = (f"last term = first + (n - 1) x difference, "
                f"{terms} - 1 = {terms - 1}, "
                # Both operands are single-digit here, so this needs no split.
                f"{terms - 1} x {difference} = {span}, "
                f"{first} + {span} = {last}, "
                f"sum = half of n x (first + last), half of {terms} = {half_terms}, "
                f"{first} + {last} = {ends}, "
                f"{decompose_product(ends, half_terms)}, total {answer}")
    return OmniProblem("arithmetic_series", "mathematics", prompt, response, float(answer), "",
                       f"sum of arithmetic series first term {first} "
                       f"common difference {difference} n {terms}",
                       {"a": first, "d": difference, "n": terms})


# -- v93 generators ----------------------------------------------------------
#
# Five more single-formula tasks, added for v93 (docs/V93_NEUROGENESIS_TWO_
# HEMISPHERES.md, D7). Each mirrors the shape of a task above that already
# scores well, so that every written step stays inside the envelope
# V81_WHAT_THE_MODEL_CAN_LEARN.md established: products are
# two-digit-by-one-digit through `decompose_product`, divisions are built
# backwards from the quotient so they are exact, and additions stay below 999.
#
# They live in `V93_TASKS`, not `TASKS`. `build` draws every task from ONE
# `random.Random(seed)` in `chosen` order, so appending to `TASKS` would move
# the default build's output; `eval_problem_solving` and `coverage_audit`
# register `TASKS` into the 30-task v89 benchmark whose fingerprint
# 3b99a446cd533be9bc5f8ae57d1310b4 every published receipt is paired on; and
# `test_code_corpus.py` derives its 21-task fingerprint from that registry.
# Keeping the twelve where they are is what leaves all three byte-identical.
# `ALL_TASKS` is the lookup table; `--task` accepts any name in it.


def _impulse(rng: random.Random) -> OmniProblem:
    # Same ranges as `force` and `work`: a two-digit force, a one-digit time.
    force = rng.randint(11, 99)
    time = rng.randint(2, 9)
    answer = force * time
    prompt = _pick(rng, [
        "A force of {f} N acts for {t} s. What is the impulse?",
        "force {f} N time {t} s find the impulse",
        "Find the impulse delivered by {f} N applied for {t} s.",
        "What impulse does a {f} N force give over {t} s?",
        "Given force {f} N and time {t} s, compute the impulse.",
    ], f=force, t=time, _task="impulse")
    response = (f"impulse = force x time, {decompose_product(force, time)}, "
                f"the impulse is {answer} newton seconds, total {answer}")
    return OmniProblem("impulse", "physics", prompt, response, float(answer), "N*s",
                       f"force {force} N time {time} s find the impulse",
                       {"F": force, "t": time})


def _ohms_current(rng: random.Random) -> OmniProblem:
    # Built backwards from the quotient, as `acceleration` and `division` are,
    # and over `division`'s exact ranges (divisor 2-9, quotient 11-60), the
    # shape that scored 1.000 on v86. The written division follows the same
    # three-way switch as the other division tasks so `--long_division`
    # reaches it.
    resistance = rng.randint(2, 9)
    current = rng.randint(11, 60)
    voltage = current * resistance  # exact division
    prompt = _pick(rng, [
        "A voltage of {v} V is applied across {r} ohm. What is the current?",
        "voltage {v} V resistance {r} ohm find the current",
        "Find the current when {v} V drives a {r} ohm resistor.",
        "What current flows through {r} ohm at {v} V?",
        "Given voltage {v} V and resistance {r} ohm, compute the current.",
    ], v=voltage, r=resistance, _task="ohms_current")
    response = (f"current = voltage / resistance, "
                f"{long_division(voltage, resistance, with_total=False) if LONG_DIVISION else decompose_quotient(voltage, resistance) if DECOMPOSE_QUOTIENT else f'{voltage} / {resistance} = {current}'}, "
                f"the current is {current} amperes, total {current}")
    return OmniProblem("ohms_current", "physics", prompt, response, float(current), "A",
                       f"voltage {voltage} V resistance {resistance} ohm find the current",
                       {"V": voltage, "R": resistance})


def _spring_energy(rng: random.Random) -> OmniProblem:
    # `kinetic_energy`'s decomposition exactly: halve the even constant first,
    # square the one-digit extension, then one two-digit-by-one-digit product.
    constant = rng.randrange(2, 20, 2)
    extension = rng.randint(2, 9)
    squared = extension * extension
    half_constant = constant // 2
    answer = half_constant * squared
    prompt = _pick(rng, [
        "A spring of constant {k} N/m is stretched {x} m. Find the spring energy.",
        "spring constant {k} N/m extension {x} m spring energy",
        "What is the elastic potential energy of a {k} N/m spring extended {x} m?",
        "Compute the spring energy for spring constant {k} N/m and extension {x} m.",
    ], k=constant, x=extension, _task="spring_energy")
    response = (f"spring energy = half x spring constant x extension squared, "
                f"half of {constant} = {half_constant}, "
                f"extension squared = {extension} x {extension} = {squared}, "
                f"{decompose_product(squared, half_constant)}, "
                f"the spring energy is {answer} joules, total {answer}")
    return OmniProblem("spring_energy", "physics", prompt, response, float(answer), "J",
                       f"spring constant {constant} N/m extension {extension} m spring energy",
                       {"k": constant, "x": extension})


def _permutations(rng: random.Random) -> OmniProblem:
    # k is fixed at 2, as `combination` fixes it, so the working is one product
    # `n x (n - 1)`. n stops at 10 because `decompose_product` splits only its
    # first operand: `12 x 11` would be written `10 x 11 = 110, 2 x 11 = 22`,
    # and a ten-times-two-digit partial is not a shape any scored task
    # contains. That leaves seven operand sets, which the report shows as
    # heavy repetition -- the same order as v74's 712 x 56 multiplication task
    # (0.93) rather than the 24,000 x 1.7 omni build (0.03).
    n = rng.randint(4, 10)
    k = 2
    answer = math.perm(n, k)
    prompt = _pick(rng, [
        "In how many ways can {k} items be arranged in order from {n}?",
        "permutations n = {n} k = {k}",
        "Find the number of permutations of {n} things taken {k} at a time.",
        "How many ordered arrangements of {k} are there from {n} items?",
    ], n=n, k=k, _task="permutations")
    response = (f"permutations = n x (n - 1), {n} - 1 = {n - 1}, "
                f"{decompose_product(n, n - 1)}, "
                f"there are {answer} arrangements, total {answer}")
    return OmniProblem("permutations", "mathematics", prompt, response, float(answer), "",
                       f"permutations n = {n} k = {k}", {"n": n, "k": k})


def _final_velocity(rng: random.Random) -> OmniProblem:
    # v = u + a t. Both factors of the product are one digit, as
    # `arithmetic_series` writes `(n - 1) x d`, and the one addition is
    # two-digit plus two-digit with a sum under 150, which `word_problem`
    # does at 0.87. Verified through `science_plan`'s formula registry
    # (`constant_acceleration.final_velocity`), the one v71 target this
    # corpus uses; its parser needs the explicit assumption phrase and consumes
    # every word, hence the fixed wording of the canonical query.
    initial = rng.randint(10, 60)
    accel = rng.randint(2, 9)
    time = rng.randint(2, 9)
    gained = accel * time
    answer = initial + gained
    prompt = _pick(rng, [
        "A body moving at {u} m/s accelerates at {a} m/s^2 for {t} s. "
        "What is its final velocity?",
        "initial velocity {u} m/s acceleration {a} m/s^2 time {t} s final velocity",
        "Find the final velocity from initial velocity {u} m/s, "
        "acceleration {a} m/s^2 and time {t} s.",
        "Starting at {u} m/s with acceleration {a} m/s^2, "
        "what is the velocity after {t} s?",
    ], u=initial, a=accel, t=time, _task="final_velocity")
    response = (f"final velocity = initial velocity + acceleration x time, "
                f"{accel} x {time} = {gained}, {initial} + {gained} = {answer}, "
                f"the final velocity is {answer} metres per second, total {answer}")
    return OmniProblem("final_velocity", "physics", prompt, response, float(answer), "m/s",
                       f"assuming constant acceleration an object has initial velocity "
                       f"{initial} m/s acceleration {accel} m/s^2 for {time} s "
                       f"find the final velocity",
                       {"u": initial, "a": accel, "t": time})


#: Every v89 generator, by task name. Frozen: see the note above `_impulse`.
TASKS: Dict[str, Callable[[random.Random], OmniProblem]] = {
    "force": _force,
    "acceleration": _acceleration,
    "momentum": _momentum,
    "kinetic_energy": _kinetic_energy,
    "work": _work,
    "power": _power,
    "voltage": _voltage,
    "electrical_power": _electrical_power,
    "wave_speed": _wave_speed,
    "molarity": _molarity,
    "combination": _combination,
    "arithmetic_series": _arithmetic_series,
}

#: The tasks added for v93. Built only when named with `--task`; never part of
#: a default build, so `python build_omni_corpus.py` still writes v89's corpus.
V93_TASKS: Dict[str, Callable[[random.Random], OmniProblem]] = {
    "impulse": _impulse,
    "ohms_current": _ohms_current,
    "spring_energy": _spring_energy,
    "permutations": _permutations,
    "final_velocity": _final_velocity,
}

#: Every generator this module can build, for lookup by name.
ALL_TASKS: Dict[str, Callable[[random.Random], OmniProblem]] = {**TASKS, **V93_TASKS}


# -- verification -----------------------------------------------------------


def verify(problem: OmniProblem) -> bool:
    """Check a generated problem against the solver.

    This is the point of the module. The generator computes an answer; the
    solver computes it independently, exactly, from a parsed query. If they
    disagree the row is wrong and must not be trained on.
    """

    result = solve_problem(problem.canonical)
    if not result.solved or result.answer_value is None:
        return False
    expected = problem.answer
    if expected == 0:
        return abs(result.answer_value) <= TOLERANCE
    return abs(result.answer_value - expected) / abs(expected) <= TOLERANCE


def extract_answer(text: str) -> Optional[float]:
    """The last number in a reply, matching the benchmark's rule."""

    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(matches[-1]) if matches else None


# -- the token budget (v82) -------------------------------------------------
#
# V67 lost its six-value `average` rows without anyone noticing: they were
# longer than the block, turn-aligned packing dropped them, and the task then
# scored 0% on exactly the sizes that had gone missing. Nothing reported the
# loss. V72 then measured that growing the block from 128 to 160 costs 24
# accuracy points, so "just make it longer" is not available either.
#
# A format change is therefore a budget decision, and this is the guard.


def _budget_tokenizer(digit_tokens: bool = True):
    """A tokenizer whose *segmentation* matches training's.

    Length depends only on the regex, not on the vocabulary -- `encode` emits
    exactly one id per regex match, `<unk>` included -- so an empty vocabulary
    measures the same token counts the real one would, and this needs no
    corpus pass and no checkpoint.
    """

    import mimomix_text  # local: pulls in torch, which the generators do not need

    return mimomix_text.WordTokenizer([], digit_tokens=digit_tokens)


def token_budget_report(
    rows: Sequence[Dict[str, str]],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    digit_tokens: bool = True,
) -> Dict[str, Any]:
    """Per task: response length, turn length, and what packing would drop.

    ``dropped_fraction`` is the share of rows whose full
    ``<bos><user> ... <assistant> ... <eos>`` turn exceeds ``sequence_length``.
    Those rows do not shorten -- they **disappear**, and they are always a
    task's longest ones, so the model meets its hardest cases first at
    evaluation.

    Measured on the current generators with the v80 tokenizer settings
    (digit_tokens on, sequence_length 128): `arithmetic_series` responses run
    to a median of 81 tokens and a max of 84, `combination` 60/65,
    `kinetic_energy` 54/56, and every other task is at or below 38.
    """

    tokenizer = _budget_tokenizer(digit_tokens)
    per_task: Dict[str, List[Tuple[int, int]]] = {}
    for row in rows:
        response_tokens = len(tokenizer.pattern.findall(row["assistant"]))
        turn_tokens = len(tokenizer.encode_turn(row["user"], row["assistant"])[0])
        per_task.setdefault(row.get("task", "?"), []).append(
            (response_tokens, turn_tokens)
        )

    tasks: Dict[str, Any] = {}
    worst = 0.0
    for name, pairs in sorted(per_task.items()):
        responses = sorted(p[0] for p in pairs)
        turns = sorted(p[1] for p in pairs)
        count = len(responses)
        index95 = min(count - 1, int(0.95 * count))
        dropped = sum(1 for t in turns if t > sequence_length)
        fraction = round(dropped / count, 6)
        worst = max(worst, fraction)
        tasks[name] = {
            "rows": count,
            "response_median": responses[count // 2],
            "response_p95": responses[index95],
            "response_max": responses[-1],
            "turn_median": turns[count // 2],
            "turn_p95": turns[index95],
            "turn_max": turns[-1],
            "dropped_by_turn_aligned_packing": dropped,
            "dropped_fraction": fraction,
        }

    return {
        "sequence_length": sequence_length,
        "digit_tokens": digit_tokens,
        "tasks": tasks,
        "worst_dropped_fraction": round(worst, 6),
        "note": (
            "turn-aligned packing drops a turn longer than sequence_length "
            "rather than truncating it; the dropped rows are a task's longest, "
            "so any non-zero fraction here biases that task toward its easiest "
            "instances"
        ),
    }


# -- retry rows (v82) -------------------------------------------------------


#: The word that marks a correction. One word, not a bracketed token: with
#: digit-level tokenisation `[retry]` becomes four symbols the model has to
#: learn to emit in order, and `<retry>` would collide with the special-token
#: convention that `decode` strips.
RETRY_MARKER = "no"

#: A step of worked arithmetic: `30 x 3 = 90`.
_STEP = re.compile(r"(-?\d+) ([-+x/]) (-?\d+) = (-?\d+)")


def inject_retry(response: str, rng: random.Random) -> Optional[str]:
    """Insert one wrong step, the marker, and the correct step.

    ``30 x 3 = 90, 9 x 3 = 27, total 117`` becomes
    ``30 x 3 = 80, no, 30 x 3 = 90, 9 x 3 = 27, total 117``.

    Ye et al. 2024 ("Physics of Language Models 2.2", arXiv 2408.16293) report
    iGSM-med accuracy rising from 78% at retry_rate 0 to about 86% at 0.2 and
    94% at 0.5, and find masking the wrong tokens out of the loss unnecessary.
    The hypothesis is that a model which has only ever seen correct chains has
    no represented way to recover from its own first mistake.

    **The wrong number is never last.** The benchmark extracts the last number
    in the reply, so a retry in the final step would score every row wrong.
    This only ever rewrites a step that is followed by more text, and returns
    ``None`` when the response has no such step -- the caller keeps the
    original rather than shipping something it has not checked.
    """

    candidates = [m for m in _STEP.finditer(response) if m.end() < len(response) - 1]
    if not candidates:
        return None
    match = rng.choice(candidates)
    left, operator, right, correct = match.groups()
    value = int(correct)
    offsets = [d for d in (-30, -20, -10, -9, -3, -2, -1, 1, 2, 3, 9, 10, 20, 30)
               if value + d != value]
    wrong = value + rng.choice(offsets)
    if wrong == value:  # unreachable, but a silent no-op retry would be worse
        return None
    step = f"{left} {operator} {right} = "
    return (
        response[: match.start()]
        + f"{step}{wrong}, {RETRY_MARKER}, {step}{correct}"
        + response[match.end():]
    )


# -- balanced operands (v82) ------------------------------------------------


def _carry_count(a: int, b: int, operator: str) -> int:
    """Carries in ``a + b`` or borrows in ``a - b``, column by column.

    Lee et al. 2023 Fig. 3 stratify by digit length **and** carry count, and
    report that uniform random sampling "performs relatively poorly even for
    2-digit addition" because the hard buckets are rare. Counting carries is
    how a bucket gets identified.

    Negative operands and non-additive operators return 0: a borrow is not
    defined for them and guessing would put rows in the wrong bucket.
    """

    if operator not in "+-" or a < 0 or b < 0:
        return 0
    if operator == "-" and a < b:
        return 0
    carries = 0
    borrow = 0
    while a > 0 or b > 0:
        digit_a, digit_b = a % 10, b % 10
        if operator == "+":
            borrow = 1 if digit_a + digit_b + borrow >= 10 else 0
        else:
            borrow = 1 if digit_a - borrow < digit_b else 0
        carries += borrow
        a //= 10
        b //= 10
    return carries


def operand_bucket(problem: "OmniProblem") -> str:
    """``d<digits>_c<carries>``: the stratum a row belongs to.

    ``digits`` is the widest integer operand in the question, and ``carries``
    counts every carry and borrow the worked response performs. Both are read
    off the row itself, so the same function buckets every task without a
    per-task table that could drift from the generators.
    """

    integers = [abs(int(v)) for v in problem.params.values()
                if isinstance(v, (int, float)) and float(v).is_integer()]
    digits = max((len(str(v)) for v in integers), default=1)
    steps = list(_STEP.finditer(problem.response))
    carries = sum(
        _carry_count(int(m.group(1)), int(m.group(3)), m.group(2)) for m in steps
    )

    # The carry a v74-shaped product hides.
    #
    # `decompose_product` writes `40 x 3 = 120, 7 x 3 = 21` and stops -- the
    # addition of the partial products is left for the model to do in its head,
    # deliberately, because that is the form that scored 0.93. So for the
    # multiplicative tasks the *only* carry in the problem is the one nobody
    # writes down, and a bucketing that read the written steps alone would put
    # every row in one bucket and balance nothing. `47 x 3` needs no carry;
    # `47 x 9` (360 + 63) needs one.
    products = [int(m.group(4)) for m in steps if m.group(2) == "x"]
    if len(products) > 1 and not any(m.group(2) == "+" for m in steps):
        running = products[0]
        for value in products[1:]:
            carries += _carry_count(running, value, "+")
            running += value

    return f"d{digits}_c{min(carries, 4)}"


# -- train-set priming (v82) ------------------------------------------------


class _HarderRandom:
    """A random source whose every range reaches one notch further.

    Jelassi et al. 2023 (arXiv 2306.15400) prime a training set with a few
    dozen examples harder than the test range, so the test range stops being
    the edge of the distribution and becomes its interior. V67 hit the same
    thing from the other side: it generated 4-5 values against a benchmark
    testing 4-6, and every six-value problem was out of distribution.

    Wrapping the RNG rather than writing harder generators means the **format
    is identical by construction** -- the same generator body runs, so a primed
    row differs from an ordinary one only in its numbers. A widened row is
    still solver-verified like any other, and one that cannot be verified is
    dropped like any other.
    """

    def __init__(self, rng: random.Random, widen: float = 0.1):
        self._rng = rng
        self._widen = widen

    def _extend(self, low: int, high: int) -> int:
        return high + max(1, int(round((high - low) * self._widen)))

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, self._extend(a, b))

    def randrange(self, start: int, stop: Optional[int] = None, step: int = 1) -> int:
        if stop is None:
            start, stop = 0, start
        # Extend by whole steps, so a range of even numbers stays even.
        extra = max(step, int(round((stop - start) * self._widen / step)) * step)
        return self._rng.randrange(start, stop + extra, step)

    def choice(self, seq):
        return self._rng.choice(seq)

    def random(self) -> float:
        return self._rng.random()

    def shuffle(self, seq) -> None:
        self._rng.shuffle(seq)


#: Attempts before a balanced build stops trying to fill its rare buckets.
BALANCE_MISSES = 20000


def _balanced_sample(name: str, per_task: int, draw):
    """Rejection-sample so digit x carry buckets are as equal as they can be.

    Two passes. The first draws ``per_task`` rows the ordinary way and records
    the histogram the generator produces on its own -- that is the "before" the
    report shows, and it is the distribution every corpus in this line has
    trained on. The second refills from scratch with a per-bucket cap, so the
    rare high-carry buckets are represented instead of being crowded out.

    Buckets that the generator simply cannot produce enough of do not stall the
    build: the miss budget runs out, the shortfall is reported, and
    ``equalised`` says plainly whether the target was reached.
    """

    first_pass = []
    for _ in range(per_task):
        problem = draw(name)
        if problem is not None:
            first_pass.append(problem)
    before = Counter(operand_bucket(p) for p in first_pass)
    if not before:
        return [], {}, {}, False

    target = math.ceil(per_task / len(before))
    kept: List["OmniProblem"] = []
    after: Counter = Counter()
    for problem in first_pass:
        bucket = operand_bucket(problem)
        if after[bucket] < target and len(kept) < per_task:
            kept.append(problem)
            after[bucket] += 1

    misses = 0
    while len(kept) < per_task and misses < BALANCE_MISSES:
        problem = draw(name)
        if problem is None:
            misses += 1
            continue
        bucket = operand_bucket(problem)
        if after[bucket] >= target:
            misses += 1
            continue
        kept.append(problem)
        after[bucket] += 1
        misses = 0

    equalised = len(kept) == per_task and len(set(after.values())) <= 2
    return kept, dict(sorted(before.items())), dict(sorted(after.items())), equalised


def build(per_task: int, seed: int, tasks: Optional[List[str]] = None,
          repeat: bool = True, retry_rate: float = 0.0,
          balanced_operands: bool = False, priming_fraction: float = 0.0,
          keep_canonical: bool = False,
          sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
          token_budget: bool = False):
    """Generate, verify, and return (rows, report).

    Every option added in v82 defaults to the value that reproduces the v80
    corpus exactly, and each one is a hypothesis with a citation rather than a
    measured improvement:

    ``retry_rate``
        Fraction of rows carrying one wrong step, the marker word, and the
        correction (Ye et al. 2024).
    ``balanced_operands``
        Rejection-sample so digit-length x carry-count buckets are equal
        (Lee et al. 2023 Fig. 3).
    ``priming_fraction``
        Fraction of rows drawn one notch beyond the benchmark's range, in the
        identical format (Jelassi et al. 2023).
    ``keep_canonical``
        Ship the solver's own query alongside the row.
    ``token_budget``
        Add the per-task length table to the report.
    """

    rng = random.Random(seed)
    harder = _HarderRandom(rng)
    # The default is the v89 twelve. A v93 task is built only when asked for
    # by name, so this line is what keeps a default build byte-identical.
    chosen = tasks or list(TASKS)
    unknown = [name for name in chosen if name not in ALL_TASKS]
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(unknown)}")
    rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    dropped: Dict[str, int] = {}
    short: Dict[str, Dict[str, object]] = {}
    distinct: Dict[str, int] = {}
    retried: Dict[str, int] = {}
    primed: Dict[str, int] = {}
    balance: Dict[str, Dict[str, object]] = {}

    def draw(name: str):
        """One verified problem, or None. Counts its own drops and priming."""

        source = rng
        if priming_fraction > 0 and rng.random() < priming_fraction:
            source = harder
            primed[name] = primed.get(name, 0) + 1
        problem = ALL_TASKS[name](source)
        if not verify(problem) or extract_answer(problem.response) != problem.answer:
            dropped[name] = dropped.get(name, 0) + 1
            if source is harder:
                primed[name] -= 1
            return None
        return problem

    def emit(name: str, problem: "OmniProblem") -> Dict[str, str]:
        """Turn a verified problem into a row, optionally with a retry."""

        row = problem.to_row(keep_canonical=keep_canonical)
        if retry_rate > 0 and rng.random() < retry_rate:
            candidate = inject_retry(row["assistant"], rng)
            # The retry must not have moved the answer. A row whose last number
            # changed would score every correct model as wrong, so it is simply
            # left alone rather than shipped unverified.
            if candidate is not None and extract_answer(candidate) == problem.answer:
                row["assistant"] = candidate
                retried[name] = retried.get(name, 0) + 1
        return row

    for name in chosen:
        made = 0
        seen = set()
        if repeat and balanced_operands:
            problems, before, after, reached = _balanced_sample(
                name, per_task, draw
            )
            for problem in problems:
                seen.add(problem.prompt)
                rows.append(emit(name, problem))
            counts[name] = len(problems)
            distinct[name] = len(seen)
            balance[name] = {
                "buckets_before": before,
                "buckets_after": after,
                "equalised": reached,
            }
            if len(problems) < per_task:
                short[name] = {"asked": per_task, "produced": len(problems),
                               "reason": "balancing could not fill every bucket"}
            continue
        if repeat:
            # Repetition is how this model learns an algorithm.
            #
            # v74's multiplication task -- the one that scored 0.93 -- holds
            # only **712 distinct operand pairs repeated 56x each** over
            # 40,000 rows. The omni corpus was built the opposite way, 24,000
            # combinations each appearing about 1.7 times, and that task
            # scored 0.03.
            #
            # Uniqueness is the right goal for a *knowledge* corpus, where a
            # repeated fact is memorised. It is the wrong goal for a
            # *procedure*, where the benchmark draws unseen operands from the
            # same space and the model has to learn the method. Distinct
            # counts are still reported, so the repetition factor is visible
            # rather than hidden.
            for _ in range(per_task):
                problem = draw(name)
                if problem is None:
                    continue
                seen.add(problem.prompt)
                rows.append(emit(name, problem))
                made += 1
            counts[name] = made
            distinct[name] = len(seen)
            continue

        # Stop when the generator stops producing prompts it has not already
        # produced. A task's parameter space is finite -- `combination` over
        # n<=40, k<=8 has only about a thousand distinct questions -- and
        # asking for more than it holds would either spin or, worse, ship the
        # same question hundreds of times. Duplicated rows are exactly what a
        # recitation-proof benchmark exists to punish, so the shortfall is
        # reported instead of padded.
        misses = 0
        while made < per_task and misses < EXHAUSTION_MISSES:
            # `draw` folds in verification and the answer-parses check: the
            # response must parse to the right answer, or the benchmark will
            # score a correct model as wrong.
            problem = draw(name)
            if problem is None:
                misses += 1
                continue
            key = problem.prompt
            if key in seen:
                misses += 1
                continue
            seen.add(key)
            rows.append(emit(name, problem))
            made += 1
            misses = 0
        counts[name] = made
        if made < per_task:
            short[name] = {"asked": per_task, "produced": made,
                           "reason": "parameter space exhausted"}

    report = {
        "schema": "supermix-v79-omni-corpus-v1",
        "seed": seed,
        "rows": len(rows),
        "per_task": counts,
        "dropped_failing_verification": dropped,
        "tasks": len(chosen),
        "short_of_requested": short,
        "distinct_prompts": distinct,
        "repetition": {k: round(counts[k] / v, 1)
                       for k, v in distinct.items() if v},
        "verified_by": "nexus_solver.solve_problem",
        "tolerance": TOLERANCE,
        "options": {
            "repeat": repeat,
            "retry_rate": retry_rate,
            # What happened, not what was asked for. Balancing only runs on the
            # repeating path, and a receipt that records a request as though it
            # were an effect is how an experiment that never ran gets written up
            # as a null result. The CLI refuses this pair outright; the library
            # entry point is still reachable, so it reports the truth.
            "balanced_operands": bool(balanced_operands and repeat),
            "priming_fraction": priming_fraction,
            "keep_canonical": keep_canonical,
            "combination_in_envelope": COMBINATION_IN_ENVELOPE,
            "natural_phrasings": NATURAL_PHRASINGS,
            "decompose_quotient": DECOMPOSE_QUOTIENT,
            "long_division": LONG_DIVISION,
        },
    }
    if balanced_operands and not repeat:
        report["balanced_operands_skipped"] = (
            "requested, but operand balancing is implemented only for "
            "repeat=True; the uniqueness path rejects draws on prompt novelty, "
            "not bucket occupancy, so no rows were stratified"
        )
    if retry_rate > 0:
        report["retry_rows"] = retried
        report["retry_marker"] = RETRY_MARKER
    if priming_fraction > 0:
        report["priming_rows"] = primed
    if balanced_operands:
        report["operand_balance"] = balance
    if token_budget:
        report["token_budget"] = token_budget_report(rows, sequence_length)
    return rows, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per_task", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--output", default="datasets/v79/v79_omni.jsonl")
    parser.add_argument("--report", default=None)
    parser.add_argument("--unique", action="store_true",
                        help=("emit only distinct prompts. Off by default: v74's "
                              "multiplication task repeated 712 pairs 56x each and "
                              "scored 0.93, while the diverse omni build scored 0.03"))
    parser.add_argument("--task", action="append", default=[],
                        help=("restrict to these tasks; repeatable. The default is "
                              "the twelve v89 tasks; the v93 tasks "
                              f"({', '.join(V93_TASKS)}) are built only when named"))
    parser.add_argument("--retry_rate", type=float, default=0.0,
                        help=("fraction of rows carrying one wrong step, the "
                              "marker word and the correction (Ye et al. 2024, "
                              "arXiv 2408.16293). Hypothesis, unmeasured here"))
    parser.add_argument("--priming_fraction", type=float, default=0.0,
                        help=("fraction of rows drawn one notch beyond the "
                              "benchmark's range in the identical format "
                              "(Jelassi et al. 2023). Hypothesis, unmeasured here"))
    parser.add_argument("--balanced_operands", action="store_true",
                        help=("equalise digit-length x carry-count buckets by "
                              "rejection sampling (Lee et al. 2023 Fig. 3). "
                              "Hypothesis, unmeasured here"))
    parser.add_argument("--keep_canonical", action="store_true",
                        help="ship the solver's own query in each row")
    parser.add_argument("--token_budget_report", action="store_true",
                        help=("measure per-task response and turn lengths and "
                              "what turn-aligned packing would drop"))
    parser.add_argument("--sequence_length", type=int,
                        default=DEFAULT_SEQUENCE_LENGTH,
                        help="the block size the token budget is measured against")
    parser.add_argument("--combination_in_envelope", action="store_true",
                        help=("halve before multiplying and cap n at 19, so no "
                              "intermediate leaves the two-digit-by-one-digit "
                              "envelope. NARROWS THE BENCHMARK TOO -- scores are "
                              "not comparable with v80's"))
    parser.add_argument("--natural_phrasings", action="store_true",
                        help=("widen each task from its four or five textbook "
                              "templates to fifteen, including the casual "
                              "register a question actually arrives in. Changes "
                              "the prompt and nothing else: operands, answers, "
                              "worked responses and the canonical query the "
                              "solver parses are all bit-identical. The last "
                              "three forms per task are withheld for "
                              "eval_natural_phrasing.py"))
    parser.add_argument("--long_division", action="store_true",
                        help=("show the three division tasks' working one "
                              "quotient digit at a time, school method. Every "
                              "number is on the page or one bring-down from it, "
                              "which is what decompose_quotient lacked. Targets "
                              "24 of the 29 division errors that make up 58% of "
                              "v88's remaining failures. UNTRAINED"))
    parser.add_argument("--decompose_quotient", action="store_true",
                        help=("split the three division tasks' quotient by "
                              "place value. MEASURED HARMFUL in v87 -- power "
                              "0.333 -> 0.048 -- because the partial dividends "
                              "are back-computed from the answer. Kept only so "
                              "the negative result stays reproducible"))
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Balancing is implemented only on the repeating path: the uniqueness loop
    # rejects a draw on prompt novelty, not on bucket occupancy, so
    # `--balanced_operands` was silently doing nothing under `--unique` while
    # the report still recorded it as applied. An A/B of `--unique` against
    # `--unique --balanced_operands` produced byte-identical corpora, so the
    # arm would have measured an exact null by construction and the receipt
    # would have said the flag was on. Refuse the pair rather than ship that.
    if args.unique and args.balanced_operands:
        parser.error(
            "--balanced_operands is not implemented with --unique: balancing "
            "stratifies repeated draws, and the uniqueness path never calls it. "
            "Drop one of the two."
        )
    global COMBINATION_IN_ENVELOPE, NATURAL_PHRASINGS, DECOMPOSE_QUOTIENT
    global LONG_DIVISION
    COMBINATION_IN_ENVELOPE = bool(args.combination_in_envelope)
    NATURAL_PHRASINGS = bool(args.natural_phrasings)
    DECOMPOSE_QUOTIENT = bool(args.decompose_quotient)
    LONG_DIVISION = bool(args.long_division)
    rows, report = build(args.per_task, args.seed, args.task or None,
                         repeat=not args.unique,
                         retry_rate=args.retry_rate,
                         balanced_operands=args.balanced_operands,
                         priming_fraction=args.priming_fraction,
                         keep_canonical=args.keep_canonical,
                         sequence_length=args.sequence_length,
                         token_budget=args.token_budget_report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    report["output"] = str(output)
    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {len(rows):,} rows to {output}")
    for name, count in sorted(report["per_task"].items()):
        note = ""
        if report["dropped_failing_verification"].get(name):
            note = f"  ({report['dropped_failing_verification'][name]} dropped)"
        print(f"  {name:<20} {count:,}{note}")

    budget = report.get("token_budget")
    if budget:
        print(f"\ntoken budget at sequence_length {budget['sequence_length']}")
        print(f"  {'task':<20} {'resp med':>8} {'p95':>5} {'max':>5} "
              f"{'turn max':>9} {'dropped':>8}")
        for name, stats in budget["tasks"].items():
            print(f"  {name:<20} {stats['response_median']:>8} "
                  f"{stats['response_p95']:>5} {stats['response_max']:>5} "
                  f"{stats['turn_max']:>9} {stats['dropped_fraction']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
