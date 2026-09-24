"""How a person actually types a question, for the twelve solver-verified tasks.

## The measurement this exists to move

v74 carried **one** template per task. It scored 0.894 on its own benchmark --
which generates prompts in the corpus's own format -- and answered **0 of 5**
naturally-typed questions. v79 gave each task four or five phrasings; v86 then
answered **14 of 18** hand-typed questions on the same harness
(`output/v85_measurements/natural_phrasing.json`). So phrasing variety is a
lever that has already moved once, by a large amount, for a corpus change that
cost nothing at inference time.

Four or five phrasings is still a template set. Every one of them is written in
the register of a physics textbook: capitalised, punctuated, units abbreviated,
the quantity named before its number. None of them is how the question arrives
in a chat box. This module widens each task to 12-15 forms that cover, per task:

* casual register and lowercase -- ``whats the force on a 12 kg object``
* contractions -- ``what's``, ``whats`` (the tokenizer's word pattern keeps
  ``what's`` as a single token, so a contraction costs the same as its word)
* missing punctuation, and question-first vs command-first order
* filler openers -- ``hey can you work out ...``
* polite framings -- ``please could you find ...``
* units long-form (``kilograms``, ``metres per second squared``) and short
  (``kg``, ``m/s^2``)
* the number before the quantity name (``25 kilograms accelerating at ...``)
  as well as after it (``the mass is 25 kg``)

## What is deliberately NOT touched

**The canonical query.** `build_omni_corpus.OmniProblem.canonical` is the string
handed to `nexus_solver`, and it is built from the parameters, not from the
phrasing. Widening this bank therefore cannot weaken verification: the oracle
still sees a form it parses, whatever the model is shown. That decoupling is the
reason it is safe to add register that no solver could ever parse.

**The response.** Every reply still ends ``total <number>``, still decomposes
inside the two-digit-by-one-digit envelope, and is byte-identical to the one the
same problem would have produced without this module. Only the question changes.

## Why the selection is a hash and not a draw

`build_scratchpad_math._prompt_variant_index` established the pattern in this
repository: choose the paraphrase from a hash of the row's identity so that
"enabling the arm never consumes the problem RNG and cannot silently change
operands, answers, or later rows."

The same rule here, for the same reason. `build_omni_corpus` still makes its
ordinary `rng.choice` over the narrow list, so the RNG stream is bit-identical
with the flag on and off; the narrow index it produces is then fed in as part of
the identity rather than used directly. A v87 corpus built with the flag on is
therefore a **single-variable** change from v86: same operands, same answers,
same worked responses, different prompt strings. If the score moves, phrasing is
the only thing that could have moved it.

A consequence worth stating: the identity is ``(task, narrow index, values)``,
so a given problem appears under as many distinct phrasings as the narrow list
has entries -- four or five -- not all fifteen. Across a task every phrasing is
seen with hundreds of different problems, which is what stops the model tying a
phrasing to an answer; and each problem is seen under several phrasings, which
is what teaches that the answer does not depend on the wording.
"""

from __future__ import annotations

import contextlib
import hashlib
from typing import Dict, Mapping, Sequence, Tuple

__all__ = ["EXTRA_PHRASINGS", "bank", "select", "variant_index"]


#: Phrasings **added** to each task's existing list, never replacing it.
#:
#: Storing only the additions is what keeps the two lists from drifting apart.
#: `bank()` concatenates a task's own templates with these, so the five forms
#: v86 trained on -- and the terse labelled forms `prompt_normaliser` rewrites
#: *into* -- remain in the corpus by construction. A test pins that.
#:
#: Placeholder names match each generator's `_pick` call exactly; a mismatch
#: raises `KeyError` at generation time rather than shipping a literal `{m}`.
EXTRA_PHRASINGS: Dict[str, Tuple[str, ...]] = {
    "force": (
        "whats the force on a {m} kg object accelerating at {a} m/s^2",
        "what's the force if the mass is {m} kilograms and the acceleration "
        "is {a} metres per second squared",
        "hey can you work out the force for a {m} kg mass at {a} m/s^2",
        "please could you find the force on {m} kg accelerating at {a} m/s^2",
        "work out the force: mass {m} kg, acceleration {a} m/s^2",
        "i have a {m} kg mass speeding up at {a} m/s^2, what force is that",
        "{m} kilograms accelerating at {a} metres per second squared, "
        "what is the force",
        "how much force does it take to accelerate {m} kg at {a} m/s^2",
        "the mass is {m} kg and the acceleration is {a} m/s^2 so what is the force",
        "force please: {m} kg at {a} m/s^2",
    ),
    "acceleration": (
        "whats the acceleration of a {m} kg mass pushed with {f} N",
        "how fast does {m} kg accelerate under {f} newtons",
        "hey can you work out the acceleration for {f} N on {m} kg",
        "please could you find the acceleration when {f} newtons acts "
        "on {m} kilograms",
        "work out the acceleration: force {f} N, mass {m} kg",
        "a {m} kg body feels {f} N, whats its acceleration",
        "{f} newtons on {m} kilograms, what is the acceleration",
        "if the force is {f} N and the mass is {m} kg what is the acceleration",
        "acceleration please: {f} N acting on {m} kg",
        "how quickly does a {m} kg object speed up when pushed with {f} N",
    ),
    "momentum": (
        "whats the momentum of a {m} kg object moving at {v} m/s",
        "how much momentum does a {m} kg trolley at {v} metres per second have",
        "hey can you work out the momentum for {m} kg at {v} m/s",
        "please could you find the momentum of {m} kilograms travelling "
        "at {v} metres per second",
        "work out the momentum: mass {m} kg, velocity {v} m/s",
        "a {m} kg cart rolling at {v} m/s, whats its momentum",
        "{m} kilograms moving at {v} metres per second, what is the momentum",
        "if the mass is {m} kg and the speed is {v} m/s what is the momentum",
        "momentum please: {m} kg at {v} m/s",
        "the object weighs {m} kg and moves at {v} m/s so what is its momentum",
    ),
    "kinetic_energy": (
        "whats the kinetic energy of a {m} kg body at {v} m/s",
        "how much kinetic energy does {m} kg moving at {v} m/s have",
        "hey can you work out the kinetic energy for {m} kg at {v} m/s",
        "please could you find the kinetic energy of {m} kilograms "
        "at {v} metres per second",
        "work out the kinetic energy: mass {m} kg, velocity {v} m/s",
        "a {m} kg mass travelling at {v} m/s, whats its kinetic energy",
        "{m} kilograms at {v} metres per second, what is the kinetic energy",
        "if the mass is {m} kg and the speed is {v} m/s what is the kinetic energy",
        "kinetic energy please: {m} kg moving at {v} m/s",
    ),
    "work": (
        "whats the work done by {f} N over {d} m",
        "how much work is done pushing with {f} newtons for {d} metres",
        "hey can you work out the work done by {f} N across {d} m",
        "please could you find the work done when {f} newtons acts over {d} metres",
        "work done: force {f} N, distance {d} m",
        "a force of {f} N moves something {d} m, how much work is that",
        "{f} newtons over {d} metres, what is the work done",
        "if the force is {f} N and the distance is {d} m what is the work",
        "work done please: {f} N through {d} m",
    ),
    "power": (
        "whats the power if {w} J is done in {t} s",
        "how much power is that, {w} joules in {t} seconds",
        "hey can you work out the power for {w} J over {t} s",
        "please could you find the power when {w} joules is delivered "
        "in {t} seconds",
        "power: work {w} J, time {t} s",
        "{w} joules in {t} seconds, what is the power",
        "if the work is {w} J and the time is {t} s what is the power",
        "power please: {w} J in {t} s",
        "{w} J of work took {t} s, whats the power",
    ),
    "voltage": (
        "whats the voltage across a {r} ohm resistor carrying {i} A",
        "how much voltage drives {i} amps through {r} ohms",
        "hey can you work out the voltage for {i} A through {r} ohm",
        "please could you find the voltage when {i} amps flows through {r} ohms",
        "voltage: current {i} A, resistance {r} ohm",
        "{i} amps through {r} ohms, what is the voltage",
        "if the current is {i} A and the resistance is {r} ohm "
        "what is the voltage",
        "voltage please: {i} A and {r} ohm",
        "a {r} ohm resistor carries {i} A, whats the voltage",
    ),
    "electrical_power": (
        "whats the electrical power at {v} V and {i} A",
        "how much power does a {v} volt supply at {i} amps give",
        "hey can you work out the electrical power for {v} V drawing {i} A",
        "please could you find the electrical power at {v} volts and {i} amps",
        "electrical power: voltage {v} V, current {i} A",
        "{v} volts at {i} amps, what is the electrical power",
        "if the voltage is {v} V and the current is {i} A "
        "what is the electrical power",
        "electrical power please: {v} V and {i} A",
        "a {v} V battery drives {i} A, whats the electrical power",
    ),
    "wave_speed": (
        "whats the wave speed at {f} Hz with a {w} m wavelength",
        "how fast does a wave go at {f} hertz and {w} metres",
        "hey can you work out the wave speed for {f} Hz and {w} m",
        "please could you find the wave speed of a {f} hertz wave "
        "with wavelength {w} metres",
        "wave speed: frequency {f} Hz, wavelength {w} m",
        "{f} hertz and a {w} metre wavelength, what is the wave speed",
        "if the frequency is {f} Hz and the wavelength is {w} m "
        "what is the wave speed",
        "wave speed please: {f} Hz, {w} m",
        "a wave at {f} Hz has a {w} m wavelength, whats its speed",
    ),
    "molarity": (
        "whats the molarity of {n} mol in {v} L",
        "how concentrated is {n} moles in {v} litres",
        "hey can you work out the molarity for {n} mol in {v} L",
        "please could you find the concentration of {n} moles in {v} litres",
        "molarity: moles {n} mol, volume {v} L",
        "{n} moles dissolved in {v} litres, what is the molarity",
        "if there are {n} mol in {v} L what is the concentration",
        "molarity please: {n} mol in {v} L",
        "i dissolved {n} mol in {v} L, whats the molarity",
    ),
    "combination": (
        "how many ways can i pick {k} from {n}",
        "whats {n} choose {k}",
        "hey can you work out {n} choose {k}",
        "please could you find how many ways {k} items can be chosen from {n}",
        "combinations: n {n}, k {k}",
        "if i have {n} things and pick {k}, how many combinations",
        "how many ways can {k} items be picked from {n}",
        "number of combinations of {n} choose {k} please",
        "choosing {k} out of {n}, how many ways is that",
    ),
    # Kept terse on purpose. Measured on the current generators, an
    # `arithmetic_series` turn already runs to 111 tokens of the 128 the run
    # packs to, and turn-aligned packing DROPS a turn that does not fit rather
    # than truncating it -- so a wordy phrasing here would silently delete the
    # task's longest problems. `bank_fits_the_budget` in the tests is what keeps
    # this honest; every form below is no longer than the existing prose one.
    "arithmetic_series": (
        "sum the first {n} terms starting at {a} step {d}",
        "whats the sum of {n} terms from {a} going up by {d}",
        "arithmetic series: first {a}, difference {d}, {n} terms, sum?",
        "hey can you sum {n} terms starting {a} rising by {d}",
        "please could you find the sum of {n} terms, first {a}, difference {d}",
        "add up {n} terms beginning at {a} increasing by {d}",
        "series sum please: start {a}, step {d}, {n} terms",
        "if a series starts at {a} and rises by {d}, what do {n} terms add to",
        "total of {n} terms from {a} with common difference {d}",
    ),
    # -- v93 tasks (build_omni_corpus.V93_TASKS) -------------------------------
    #
    # Same register and the same held-out rule as the twelve above: the last
    # three of each tuple never reach a corpus, so `eval_natural_phrasing` has
    # a phrasing the model has not seen for every new task as well. Placeholders
    # are the `_pick` keyword names of each generator; `test_v93_corpus.py`
    # renders every form through its generator to pin that.
    "impulse": (
        "whats the impulse from {f} N acting for {t} s",
        "how much impulse does a {f} newton force give in {t} seconds",
        "hey can you work out the impulse for {f} N over {t} s",
        "please could you find the impulse when {f} newtons acts for {t} seconds",
        "work out the impulse: force {f} N, time {t} s",
        "a {f} N push lasting {t} s, whats the impulse",
        "{f} newtons for {t} seconds, what is the impulse",
        "if the force is {f} N and it acts for {t} s what is the impulse",
        "impulse please: {f} N for {t} s",
        "the force is {f} N and the time is {t} s so what is the impulse",
    ),
    "ohms_current": (
        "whats the current through {r} ohm at {v} V",
        "how much current flows when {v} volts is put across {r} ohms",
        "hey can you work out the current for {v} V and {r} ohm",
        "please could you find the current when {v} volts drives {r} ohms",
        "work out the current: voltage {v} V, resistance {r} ohm",
        "a {r} ohm resistor at {v} V, whats the current",
        "{v} volts across {r} ohms, what is the current",
        "if the voltage is {v} V and the resistance is {r} ohm what is the current",
        "current please: {v} V across {r} ohm",
        "the voltage is {v} V and the resistance is {r} ohm so what is the current",
    ),
    "spring_energy": (
        "whats the spring energy of a {k} N/m spring stretched {x} m",
        "how much energy is stored in a {k} N/m spring pulled {x} metres",
        "hey can you work out the spring energy for {k} N/m and {x} m",
        "please could you find the elastic potential energy of a {k} N/m "
        "spring extended {x} metres",
        "work out the spring energy: spring constant {k} N/m, extension {x} m",
        "a spring with constant {k} N/m stretched by {x} m, whats its spring energy",
        "spring constant {k} N/m and extension {x} m, what is the spring energy",
        "if k is {k} N/m and the extension is {x} m what is the spring energy",
        "spring energy please: {k} N/m stretched {x} m",
        "the spring constant is {k} N/m and it is stretched {x} m so what is "
        "the spring energy",
    ),
    "permutations": (
        "how many ways can i arrange {k} out of {n} in order",
        "whats the number of permutations of {n} taken {k}",
        "hey can you work out the permutations of {n} take {k}",
        "please could you find how many ordered ways {k} items can be picked from {n}",
        "permutations: n {n}, k {k}",
        "if i have {n} things and line up {k} of them, how many permutations",
        "how many ordered selections of {k} from {n} are there",
        "number of permutations of {n} take {k} please",
        "arranging {k} out of {n} in order, how many ways is that",
    ),
    "final_velocity": (
        "whats the final velocity if it starts at {u} m/s and accelerates "
        "at {a} m/s^2 for {t} s",
        "how fast is it going after {t} s from {u} m/s at {a} m/s^2",
        "hey can you work out the final velocity for {u} m/s, {a} m/s^2, {t} s",
        "please could you find the final speed from {u} m/s after {t} s "
        "at {a} m/s^2",
        "work out the final velocity: initial {u} m/s, acceleration {a} m/s^2, "
        "time {t} s",
        "a car at {u} m/s speeds up at {a} m/s^2 for {t} s, whats its final velocity",
        "{u} m/s plus {a} m/s^2 for {t} s, what is the final velocity",
        "if it starts at {u} m/s and gains {a} m/s^2 for {t} s what is the "
        "final velocity",
        "final velocity please: {u} m/s, {a} m/s^2, {t} s",
        # Kept under the 32-token prompt ceiling with room to spare: the
        # longest form here is 26 tokens, so 26 + 96 generated stays inside the
        # 128-position context the model was trained to.
        "initial velocity {u} m/s, acceleration {a} m/s^2, time {t} s, "
        "so what is the final velocity",
    ),
}


#: How many of each task's natural phrasings are withheld from training.
#:
#: Widening a phrasing bank and then scoring on the same bank measures nothing:
#: a model can memorise fifteen templates as easily as five, and the benchmark
#: would report that as understanding. These last few forms per task are never
#: built into a corpus, so `eval_natural_phrasing.py` can ask the question that
#: matters -- does the answer survive a phrasing the model has **never seen** --
#: rather than the one that flatters it.
#:
#: Three per task leaves seven or eight for training and gives the benchmark
#: 36 held-out forms across twelve tasks (51 across seventeen once the five v93
#: tasks are counted). Taken from the end of each tuple, so appending a
#: phrasing extends the training set and never silently moves a form out of
#: the benchmark.
HELD_OUT_PER_TASK = 3


#: Set by `held_out_only()`. When true, `bank` returns *only* the withheld
#: forms, so an ordinary generator renders a held-out phrasing using its own
#: keyword arguments. Rendering the template from `OmniProblem.params` instead
#: would not work: the params dict keys are physics symbols (`F`, `W`, `I`)
#: while the templates use the lowercase names `_pick` is called with.
_HELD_OUT_ONLY = False


@contextlib.contextmanager
def held_out_only():
    """Make every generator emit a phrasing that training never contained.

    Used by `eval_natural_phrasing.py`. The generator's RNG stream does not
    depend on which phrasing is chosen, so the same seed inside and outside this
    block yields the same operands and the same answer with a different prompt --
    which is what makes that benchmark paired.
    """

    global _HELD_OUT_ONLY
    previous = _HELD_OUT_ONLY
    _HELD_OUT_ONLY = True
    try:
        yield
    finally:
        _HELD_OUT_ONLY = previous


def held_out(task: str) -> Tuple[str, ...]:
    """The phrasings withheld from training, for evaluation only.

    Never call this from a corpus builder. `test_natural_phrasings.py` asserts
    that no built corpus contains any string this returns.
    """

    extra = EXTRA_PHRASINGS.get(task, ())
    if len(extra) <= HELD_OUT_PER_TASK:
        return ()
    return extra[-HELD_OUT_PER_TASK:]


def bank(task: str, templates: Sequence[str],
         include_held_out: bool = False) -> Tuple[str, ...]:
    """The task's own templates first, then the natural ones.

    Concatenation rather than replacement, so the forms v86 was trained on --
    and the forms `prompt_normaliser` rewrites *into* -- cannot be dropped by
    editing this file. An unknown task simply keeps its own templates, which is
    what lets a new generator be added without touching this module.

    The last `HELD_OUT_PER_TASK` natural forms are excluded unless asked for,
    so the training corpus and the phrasing benchmark cannot overlap.
    """

    withheld = held_out(task)
    if _HELD_OUT_ONLY and withheld:
        return withheld
    extra = EXTRA_PHRASINGS.get(task, ())
    if not include_held_out and withheld:
        extra = extra[:-len(withheld)]
    return tuple(templates) + extra


def variant_index(task: str, narrow_index: int, values: Mapping[str, object],
                  count: int) -> int:
    """Which phrasing this row gets, without consuming any randomness.

    The identity is the task, the index the ordinary `rng.choice` produced over
    the narrow list, and the row's parameter values. All three are already
    fixed by the time this is called, so the answer is deterministic and the
    generator's RNG stream is untouched -- see the module docstring for why
    that matters.
    """

    if count <= 0:
        raise ValueError("a phrasing bank cannot be empty")
    rendered = ";".join(f"{k}={values[k]}" for k in sorted(values))
    identity = f"supermix-v87-phrasing:{task}:{narrow_index}:{rendered}"
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % count


def select(task: str, templates: Sequence[str], narrow: str,
           values: Mapping[str, object]) -> str:
    """The natural phrasing for a row whose narrow pick was ``narrow``.

    ``narrow`` is the string `rng.choice(templates)` returned. Feeding it back
    in rather than discarding it is what gives a repeated problem more than one
    phrasing: the same operands drawn again get a different narrow index, and so
    a different natural form.
    """

    wide = bank(task, templates)
    try:
        narrow_index = list(templates).index(narrow)
    except ValueError:  # a caller passed a string not in its own list
        narrow_index = 0
    return wide[variant_index(task, narrow_index, values, len(wide))]
