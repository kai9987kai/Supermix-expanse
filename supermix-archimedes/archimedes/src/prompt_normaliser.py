"""Rewrite a naturally-typed question into the shape v74 was trained on.

v74 scores **0.894** on the n=500 benchmark and produced nonsense for every
naturally-phrased question typed into the chat interface. Both are true, and
the reason is that the benchmark generates prompts in the corpus's own format
while a person does not:

    "What is 47 x 6?"      -> 40 x 6 = 240, 7 x 6 = 42, total 282   correct
    "what is 47 times 6"   -> 400 x 6 = 200, 7 x 6 = 42, total 242  wrong

Probing which features actually matter, rather than assuming:

| feature                          | matters |
|----------------------------------|---------|
| operator token (`x` vs `times`)  | **yes** |
| a lead-in phrase being present   | **yes** |
| capitalisation                   | no      |
| trailing question mark           | no      |

`"47 x 6"` with no lead-in was read as algebra ("subtract 6 from both sides"),
so the lead-in is doing real work: it selects the task, not just the register.

## What this is and is not

It is a **presentation** fix. It maps how a person writes an operation onto the
token the model was trained on. It does not compute anything, it never alters a
number, and it never invents operands -- if it cannot recognise the shape it
returns the text untouched so ordinary conversation still reaches the model.

It does **not** make the model more capable, and it must not be described as
doing so. A question the model gets wrong in the training format stays wrong
here: `"What is 15% of 240?"` returns 26.0 (should be 36) both before and
after normalisation, because `percent` genuinely scores 0.75.

The rewrite is reported to the caller so the interface can show what was
actually asked. Silently changing someone's question and presenting the answer
as a reply to what they typed would misrepresent the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

#: Lead-ins drawn verbatim from the corpus, one per arithmetic task. The model
#: accepts any of them for any task, but using each task's own lead-in keeps the
#: rewritten prompt inside the distribution it was trained on.
LEAD_IN = {
    "multiplication": "What is {a} x {b}?",
    "division": "Quick question: {a} / {b}",
    "addition": "Please help with this. {a} + {b}",
    "subtraction": "Solve this basic math problem: {a} - {b}",
}

NUMBER = r"-?\d+(?:\.\d+)?"
REQUEST_PREFIX = (
    r"(?:(?:please\s+)?(?:what\s+is|what's|calculate|compute|evaluate)|"
    r"quick\s+question:|please\s+help\s+with\s+this\.|"
    r"solve\s+this\s+basic\s+math\s+problem:)\s+"
)
NUMBER_LIST = rf"{NUMBER}(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+){NUMBER})+"

#: Terse labelled forms taken verbatim from `build_omni_corpus`, one per science
#: task. Each corpus task carries four or five phrasings; these are the ones that
#: name every quantity explicitly, so a rewrite cannot be misread as a different
#: task once the units are stripped.
SCIENCE_LEAD_IN = {
    "force": "Given mass {m} kg and acceleration {a} m/s^2, compute the force.",
    "acceleration": "force {f} N mass {m} kg find acceleration",
    "momentum": "mass {m} kg velocity {v} m/s find momentum",
    "kinetic_energy": "mass {m} kg velocity {v} m/s kinetic energy",
    "work": "force {f} N distance {d} m work done",
    "electrical_power": "voltage {u} V current {i} A electrical power",
    "voltage": "current {i} A resistance {r} ohm find voltage",
    "power": "work {w} J time {t} s power",
}

#: The two counting tasks, which carry no units and so cannot be reached by the
#: quantity patterns above.
#:
#: v88's held-out phrasing benchmark put `arithmetic_series` at **0/25** and
#: `combination` at **2/25** on wordings withheld from training, against 25/25
#: for both on their trained wordings. They do not fail by computing wrongly.
#: They stop being a maths model and answer from the dialogue corpus:
#:
#:     total of 8 terms from 8 with common difference 2
#:       -> "Practical answer: Sure. I can propose a concrete plan if you want."
#:     choosing 2 out of 54, how many ways is that
#:       -> "Recommended path: Got it."
#:
#: Every other task in that benchmark scores 19/25 or better, so this is not a
#: general weakness -- it is these two shapes falling closer to the 96,108
#: language rows than to their own 40,000. A presentation fix is the right tool:
#: the model answers both perfectly once the question is in its own form.
COUNTING_LEAD_IN = {
    "combination": "In how many ways can {k} items be chosen from {n}?",
    "arithmetic_series": ("An arithmetic series starts at {a} with common "
                          "difference {d}. What is the sum of the first {n} "
                          "terms?"),
}

#: Words that mark a counting question. A rule fires only when one of these is
#: present *and* the operands can be located by role, never by position -- see
#: `_counting` for why position is not usable here.
CHOOSE_WORDS = r"choose|choosing|chose|pick|picking|select|selecting|combination|ways|options"
SERIES_WORDS = r"series|progression|terms?"

#: v91 Cognitive Lead-in formats for Pearlian causal DAG, proof audit, DoT, and conformal stopping
COGNITIVE_LEAD_IN = {
    "causal_intervention": "Given scenario {scenario}, compute causal query P({outcome} | do({treatment}={val})).",
    "proof_verify": "Verify proof derivation: {trace}",
    "diffusion_thought": "Denoise continuous thought latent for: {prompt}",
    "conformal_stopping": "Evaluate conformal stopping at step {step} of {budget} with verifier {verifier} and entropy {entropy}.",
}

#: Quantity patterns, ordered so the more specific unit wins. `m/s^2` must be
#: tried before `m/s`, and both before a bare `m`, or an acceleration is read as
#: a velocity and a velocity as a distance.
QUANTITY_PATTERNS = (
    ("a", rf"({NUMBER})\s*(?:m\s*/\s*s\s*(?:\^|\*\*)?\s*2|m/s²|"
          rf"met(?:re|er)s?\s+per\s+second\s+squared)"),
    ("v", rf"({NUMBER})\s*(?:m\s*/\s*s(?![\^²0-9])|met(?:re|er)s?\s+per\s+second(?!\s+squared))"),
    ("m", rf"({NUMBER})\s*(?:kg\b|kilogram(?:me)?s?\b)"),
    ("f", rf"({NUMBER})\s*(?:N\b|newtons?\b)"),
    ("u", rf"({NUMBER})\s*(?:V\b|volts?\b)"),
    ("i", rf"({NUMBER})\s*(?:A\b|amp(?:ere)?s?\b)"),
    ("r", rf"({NUMBER})\s*(?:ohms?\b|Ω)"),
    ("w", rf"({NUMBER})\s*(?:J\b|joules?\b)"),
    ("t", rf"({NUMBER})\s*(?:s\b|seconds?\b)"),
    ("d", rf"({NUMBER})\s*(?:m\b|met(?:re|er)s?\b)"),
)

#: What each task asks for, and what it needs to be answerable. A target whose
#: quantities are not all present is left alone rather than guessed at.
SCIENCE_TARGETS = (
    ("kinetic_energy", r"kinetic\s+energ", ("m", "v")),
    ("electrical_power", r"(?:electrical\s+power|power)", ("u", "i")),
    ("power", r"power", ("w", "t")),
    ("voltage", r"(?:voltage|potential\s+difference)", ("i", "r")),
    ("momentum", r"momentum", ("m", "v")),
    ("acceleration", r"accelerat", ("f", "m")),
    ("work", r"work", ("f", "d")),
    ("force", r"force", ("m", "a")),
)

# Match the whole quantity-labelled sentence. Free-form prose may contain
# constraints, additional targets or different physical assumptions; it must
# reach the model intact rather than lose those details in a template rewrite.
SCIENCE_SHAPES = {
    "force": (
        r"a body of mass @m@ has an acceleration of @a@\. what is the force",
        r"mass @m@ (?:acceleration|accelerating at) @a@,? find the force",
        r"find the force on a @m@ mass accelerating at @a@",
        r"what force acts on mass @m@ with acceleration @a@",
        r"given mass @m@ and acceleration @a@, compute the force",
        r"if something weighs @m@ and speeds up at @a@, what force(?: is that)?",
    ),
    "acceleration": (
        r"a force of @f@ acts on a mass of @m@\. what is the acceleration",
        r"force @f@ mass @m@ find acceleration",
        r"find the acceleration produced by @f@ on @m@",
        r"what acceleration results from a @f@ force on a @m@ body",
        r"a @m@ mass is pushed with @f@\. how fast does it accelerate",
    ),
    "momentum": (
        r"a @m@ object moves with velocity @v@\. what is its momentum",
        r"mass @m@ velocity @v@ find momentum",
        r"find the momentum of a mass @m@ travelling at velocity @v@",
        r"what is the linear momentum for mass @m@ and velocity @v@",
        r"how much momentum does a @m@ (?:trolley|body|object) moving at @v@ have",
        r"a @m@ (?:trolley|body|object) at @v@, what is its momentum",
    ),
    "kinetic_energy": (
        r"a mass of @m@ moves at velocity @v@\. find the kinetic energy",
        r"mass @m@ velocity @v@ kinetic energy",
        r"(?:what is the )?kinetic energy of a @m@ body at @v@",
        r"compute the kinetic energy for mass @m@ and speed @v@",
    ),
    "work": (
        r"a force of @f@ moves an object @d@\. how much work is done",
        r"force @f@ distance @d@ work done",
        r"find the work done by @f@ acting over @d@",
        r"what work is done when a @f@ force acts through @d@",
        r"work done pushing with @f@ over @d@",
    ),
    "power": (
        r"@w@ of work(?: is done)? in @t@[.,] what (?:is the )?power",
        r"work @w@ time @t@ power",
        r"find the power when @w@ is delivered over @t@",
        r"what power corresponds to @w@ in @t@",
    ),
    "voltage": (
        r"a current of @i@ flows through @r@\. what is the voltage",
        r"current @i@ resistance @r@ find voltage",
        r"find the potential difference across @r@ carrying @i@",
        r"what voltage drives @i@ through a @r@ resistor",
    ),
    "electrical_power": (
        r"a device runs at @u@ drawing @i@\. what is the electrical power",
        r"voltage @u@ current @i@ electrical power",
        r"find the power dissipated at @u@ and @i@",
        r"what electrical power is used at @u@ and @i@",
        r"a @u@ battery drives @i@\. what's the power",
    ),
}


@dataclass(frozen=True)
class Normalised:
    """The prompt to send, and an honest record of what was done to it."""

    prompt: str
    rule: Optional[str] = None
    original: Optional[str] = None

    @property
    def changed(self) -> bool:
        return self.rule is not None and self.prompt != self.original


def _numbers(text: str) -> List[str]:
    return re.findall(NUMBER, text)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _request(pattern: str, text: str, prefix: str = REQUEST_PREFIX):
    return re.fullmatch(rf"(?:{prefix})?{pattern}\s*[?.!]?", text, flags=re.IGNORECASE)


def _binary(text: str) -> Optional[Normalised]:
    """`A <op> B` in any of the ways people write it."""

    # "subtract A from B" names its operands in the opposite order to "B - A",
    # so it cannot be handled by the symmetric `A op B` scan below.
    reversed_subtraction = _request(
        rf"subtract(?:ing)?\s+({NUMBER})\s+from\s+({NUMBER})", text
    )
    if reversed_subtraction:
        return Normalised(
            LEAD_IN["subtraction"].format(
                a=reversed_subtraction.group(2), b=reversed_subtraction.group(1)
            ),
            "subtraction",
        )

    operators = [
        ("multiplication", r"(?:x|\*|times|multiplied\s+by)"),
        ("division", r"(?:/|÷|divided\s+by|over)"),
        ("addition", r"(?:\+|plus|added\s+to)"),
        ("subtraction", r"(?:-|minus|take\s+away|less)"),
    ]
    for task, pattern in operators:
        # Require the operator to be delimited so "6 - 2" matches but the
        # minus sign inside "-12" does not.
        match = _request(
            rf"({NUMBER})\s*(?:{pattern})\s*({NUMBER})",
            text,
        )
        if not match:
            continue
        a, b = match.group(1), match.group(2)
        return Normalised(LEAD_IN[task].format(a=a, b=b), task)
    return None


def _quantities(text: str) -> tuple[dict, str]:
    """Every quantity the text names by its unit, keyed by symbol.

    A unit is consumed once matched, so a single number cannot be read as two
    different quantities. `"7 m/s"` is a velocity and is then unavailable as a
    distance, which is what stops `m/s` being harvested twice.
    """

    found: dict = {}
    remaining = text
    for symbol, pattern in QUANTITY_PATTERNS:
        matches = list(re.finditer(pattern, remaining, flags=re.IGNORECASE))
        if len(matches) > 1:
            return {}, text
        if matches:
            match = matches[0]
            found[symbol] = match.group(1)
            remaining = remaining[: match.start()] + f"@{symbol}@" + remaining[match.end():]
    return found, remaining


def _science(text: str) -> Optional[Normalised]:
    """Rewrite a physics question into the terse labelled corpus form.

    Deliberately conservative, for the reason the module docstring gives: a
    wrong rewrite is worse than none. A rule fires only when the text names the
    target *and* every quantity that target needs, each anchored to its unit.
    "What force do you feel in a lift?" names a target and no quantities, so it
    goes through untouched to ordinary conversation.
    """

    lowered = text.lower()
    quantities, template = _quantities(text)
    if not quantities:
        return None
    for task, target_pattern, required in SCIENCE_TARGETS:
        if not re.search(target_pattern, lowered):
            continue
        if not all(symbol in quantities for symbol in required):
            continue
        if not any(re.fullmatch(pattern, template.strip(" ?.!\t\n"), flags=re.IGNORECASE)
                   for pattern in SCIENCE_SHAPES[task]):
            continue
        values = {symbol: quantities[symbol] for symbol in required}
        return Normalised(SCIENCE_LEAD_IN[task].format(**values), task)
    return None


def _counting(text: str) -> Optional[Normalised]:
    """Rewrite a combination or arithmetic-series question into the corpus form.

    Operands are found **by role, never by position**. `54 choose 2` and
    `choosing 2 out of 54` name the same problem in opposite orders, and a rule
    that took the first number as `n` would silently answer the second one
    backwards -- which is worse than not firing, because the reply looks
    confident and the working looks right.

    So: `combination` anchors `n` to a from/of/out-of preposition and treats the
    other number as `k`, and falls back to size ordering only when no
    preposition is present (`54 choose 2`), where the pool is necessarily the
    larger. `arithmetic_series` anchors each of its three operands to its own
    marker word and fires only when all three are found.

    Conservative in the same way `_science` is: a question that names the task
    but does not carry its operands goes through untouched. `"how many ways are
    there to solve this"` must reach ordinary conversation.
    """

    lowered = text.lower()

    # -- arithmetic series: needs a first term, a difference and a count ------
    if re.search(SERIES_WORDS, lowered):
        count = re.search(rf"({NUMBER})\s+terms?\b", lowered) or re.search(
            rf"\b(?:first|sum\s+of|total\s+of|add\s+up)\s+({NUMBER})\b", lowered)
        # `sum of the first 10 terms` counts terms; it does not name the first
        # term. The negative lookahead is what separates the two readings, and
        # without it "first 10 terms, start 9" is rewritten as starting at 10.
        first = re.search(
            rf"(?:start(?:s|ing)?(?:\s+at)?|begin(?:s|ning)?(?:\s+at)?|"
            rf"first(?:\s+term)?|from)\s+({NUMBER})(?![\d.])(?!\s*terms?\b)",
            lowered)
        step = re.search(
            rf"(?:common\s+difference|difference|step|rising\s+by|rises\s+by|"
            rf"increasing\s+by|increases\s+by|going\s+up\s+by|goes\s+up\s+by)"
            rf"\s+(?:of\s+)?({NUMBER})", lowered)
        if count and first and step:
            return Normalised(
                COUNTING_LEAD_IN["arithmetic_series"].format(
                    a=first.group(1), d=step.group(1), n=count.group(1)),
                "arithmetic_series", text)

    # -- combination: needs a chosen count and a pool ------------------------
    if re.search(CHOOSE_WORDS, lowered):
        pool = re.search(
            rf"(?:from|out\s+of|of)\s+({NUMBER})", lowered)
        if pool:
            others = [value for value in re.findall(NUMBER, lowered)
                      if value != pool.group(1)]
            # `combinations of 2 from 54` puts a number after `of` that is the
            # chosen count, not the pool; the later preposition wins.
            later = re.findall(rf"(?:from|out\s+of)\s+({NUMBER})", lowered)
            if later:
                pool_value = later[-1]
                others = [v for v in re.findall(NUMBER, lowered) if v != pool_value]
            else:
                pool_value = pool.group(1)
            if len(others) == 1:
                chosen, size = others[0], pool_value
                if float(chosen) < float(size):
                    return Normalised(
                        COUNTING_LEAD_IN["combination"].format(k=chosen, n=size),
                        "combination", text)
        else:
            # `54 choose 2` -- no preposition, so the pool is the larger. The
            # match must be the *whole* question: "choose a number between 1 and
            # 10" also contains `choose` and two numbers, and rewriting that into
            # a combination question would answer something nobody asked.
            adjacent = re.fullmatch(
                rf"\s*({NUMBER})\s+choose\s+({NUMBER})\s*", lowered)
            if adjacent:
                low, high = sorted(adjacent.groups(), key=float)
                return Normalised(
                    COUNTING_LEAD_IN["combination"].format(k=low, n=high),
                    "combination", text)
    return None


def _cognitive(source: str, original_text: str) -> Optional[Normalised]:
    """Rewrite complete cognitive requests without supplying missing facts.

    Identifiers and numeric literals are data, so preserve their spelling.
    Proofs and thought payloads may contain indentation or quoted whitespace;
    match those against the original instead of the whitespace-cleaned source.
    """

    # 1. Causal intervention query: P(Y | do(X = x))
    causal_match = re.fullmatch(
        r"(?:(?:please\s+)?(?:compute|calculate|what\s+is)\s+(?:the\s+)?(?:causal|interventional)\s+effect\s+(?:on\s+)?(?P<outcome>[A-Za-z_]\w*)\s+if\s+(?:we\s+)?|"
        r"(?:please\s+)?compute\s+causal\s+query:?\s*)"
        rf"do\s*(?P<paren>\()?\s*(?P<treatment>[A-Za-z_]\w*)\s*=\s*(?P<val>{NUMBER})\s*(?(paren)\))"
        r"(?(outcome)|\s+(?:on|find)\s*(?P<query_outcome>[A-Za-z_]\w*))"
        r"\s+in\s+(?P<scenario>[A-Za-z_]\w*)\s*[?.!]?",
        source,
        flags=re.IGNORECASE,
    )
    if causal_match:
        return Normalised(
            COGNITIVE_LEAD_IN["causal_intervention"].format(
                scenario=causal_match.group("scenario"),
                outcome=causal_match.group("outcome") or causal_match.group("query_outcome"),
                treatment=causal_match.group("treatment"), val=causal_match.group("val"),
            ),
            "causal_intervention",
            original_text,
        )

    # Counterfactuals require the factual state and structural assumptions.
    # The former template invented an outcome of 3.8 and a default scenario.
    # Leave such requests intact for the model to handle the missing context.

    # 3. Proof verification request
    proof_match = re.fullmatch(
        r"\s*(?:please\s+)?(?:verify|check)\s+(?:the\s+)?(?:proof(?:\s+derivation)?|derivation|steps):[ \t]*(.+)",
        original_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if proof_match and proof_match.group(1).strip():
        trace = proof_match.group(1)
        return Normalised(
            COGNITIVE_LEAD_IN["proof_verify"].format(trace=trace),
            "proof_verify",
            original_text,
        )

    # 4. Diffusion-of-Thought command
    dot_match = re.fullmatch(
        r"\s*(?:(?:please\s+)?(?:denoise\s+(?:continuous\s+)?(?:thought|reasoning)(?:\s+latent)?(?:\s+plan)?|crystallize\s+(?:thought|reasoning)\s+plan))\s*(?:for|on)?:[ \t]*(.+)",
        original_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if dot_match and dot_match.group(1).strip():
        prompt = dot_match.group(1)
        return Normalised(
            COGNITIVE_LEAD_IN["diffusion_thought"].format(prompt=prompt),
            "diffusion_thought",
            original_text,
        )

    # 5. Conformal early exit check
    conf_match = re.fullmatch(
        r"(?:(?:please\s+)?(?:evaluate\s+conformal\s+stopping|conformal\s+early\s+exit\s+check)):\s*step\s*(\d+)\s*(?:of|/)\s*(\d+)\s*,?\s*verifier\s*(-?\d+(?:\.\d+)?)\s*,?\s*entropy\s*(-?\d+(?:\.\d+)?)\s*[?.!]?",
        source,
        flags=re.IGNORECASE,
    )
    if conf_match:
        step, budget, verifier, entropy = conf_match.groups()
        return Normalised(
            COGNITIVE_LEAD_IN["conformal_stopping"].format(
                step=step, budget=budget, verifier=verifier, entropy=entropy
            ),
            "conformal_stopping",
            original_text,
        )

    return None


def normalise(text: str) -> Normalised:
    """Rewrite `text` into the corpus format, or return it unchanged.

    Rules are ordered most-specific first: a two-step question contains a
    percent question, and a percent question contains numbers that would
    otherwise look like an average.
    """

    if not text or not text.strip():
        return Normalised(text, None, text)
    source = _clean(text)
    lowered = source.lower()

    # Two-step: a percentage followed by a further operation.
    two_step = _request(
        rf"({NUMBER})\s*(?:%|percent)\s*of\s*({NUMBER})\s*,?\s*"
        rf"then\s*(add|subtract|plus|minus)\s*({NUMBER})",
        lowered,
    )
    if two_step:
        percent, whole, operation, operand = two_step.groups()
        word = "add" if operation in ("add", "plus") else "subtract"
        return Normalised(
            f"What is {percent}% of {whole}, then {word} {operand}?",
            "two_step",
            text,
        )

    percent = _request(
        rf"({NUMBER})\s*(?:%|percent)\s*(?:of)\s*({NUMBER})", lowered
    )
    if percent:
        return Normalised(
            f"What is {percent.group(1)}% of {percent.group(2)}?", "percent", text
        )

    average = _request(
        rf"(?:the\s+)?(?:average(?:\s*\(mean\))?|mean)\s+of\s+"
        rf"(?:these\s+numbers:\s*)?({NUMBER_LIST})",
        lowered, prefix=rf"(?:{REQUEST_PREFIX}|find\s+)",
    )
    if average:
        joined = ", ".join(_numbers(average.group(1)))
        return Normalised(
            f"Find the average (mean) of these numbers: {joined}", "average", text,
        )

    sequence = _request(
        rf"(?:what\s+comes\s+next(?:\s+in\s+the\s+sequence)?|"
        rf"continue\s+(?:the\s+)?sequence|(?:find\s+)?the\s+next\s+"
        rf"(?:number|term)\s+in\s+the\s+sequence)\s*:?\s*({NUMBER_LIST})",
        lowered, prefix=r"please\s+",
    )
    if sequence:
        values = _numbers(sequence.group(1))
        if len(values) >= 3:
            joined = ", ".join(values)
            return Normalised(
                f"What comes next in the sequence: {joined}?", "sequence", text
            )

    # Algebra is already written the way the corpus writes it, and rewriting an
    # equation risks reordering its sides. Only the lead-in is normalised.
    algebra = _request(
        rf"x\s*([+\-*/])\s*({NUMBER})\s*=\s*({NUMBER})", lowered,
        prefix=r"(?:please\s+)?solve\s+for\s+x:\s*",
    )
    if algebra:
        operator, operand, result = algebra.groups()
        return Normalised(
            f"Solve for x: x {operator} {operand} = {result}", "algebra_one_step", text
        )

    # Cognitive requests before science/binary scan.
    cognitive = _cognitive(source, text)
    if cognitive is not None:
        return cognitive

    # Science before the binary scan. "A 30 kg mass is pushed with 90 N" would
    # otherwise be harvested as an arithmetic pair by the `A op B` search.
    science = _science(source)
    if science is not None:
        return Normalised(science.prompt, science.rule, text)

    # Counting before the binary scan too. "choosing 2 out of 54" carries two
    # numbers and no operator, which the `A op B` search would leave alone, but
    # "sum 10 terms starting at 9 going up by 4" carries three and could be
    # harvested as an average.
    counting = _counting(source)
    if counting is not None:
        return Normalised(counting.prompt, counting.rule, text)

    binary = _binary(source)
    if binary is not None:
        return Normalised(binary.prompt, binary.rule, text)

    # A word problem, or ordinary conversation. Both go through untouched: the
    # corpus's word problems are written in plain prose already, and rewriting
    # conversation would be pure damage.
    return Normalised(text, None, text)
