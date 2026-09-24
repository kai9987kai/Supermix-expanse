"""Generate an executable-Python corpus, verified by running the code.

## Why this exists

v86 scores 0.779 across 21 tasks and has **zero** coding ability, for the
simple reason that there is no code in the corpus. Nothing here has ever asked
the model to read a snippet.

Code is the cheapest thing in this repository to verify exactly: you *run it*.
That is the same contract `build_omni_corpus` has with `nexus_solver`, with
CPython as the checker instead. A generator builds a snippet from a template,
works the answer out step by step in the model's own format, and the
interpreter then executes the snippet it actually shows the model. Disagreement
drops the row.

The check is not circular, and it is worth being precise about why, because the
drop rate is 0.00% and a 0% drop rate usually means a verifier that passes
everything. The generator computes its answer with Python *expressions*
(`s + n * k`); the checker executes the *string it renders into the prompt*.
Those two agree only if the rendered snippet says what the generator thinks it
says. An off-by-one in a `range`, a variable renamed in the template but not in
the derivation, a loop bound written from the wrong parameter -- each of those
is a mismatch between expression and string, and each is caught.
`test_code_corpus.py` mutates a template and asserts the row is dropped, which
is the only way to know the checker is load-bearing rather than decorative.

## What the envelope allowed, and what it ruled out

Every constraint from V81_WHAT_THE_MODEL_CAN_LEARN applies unchanged, and two
of them removed whole task families before a line was written:

**Character-level string work is unlearnable here and is not attempted.**
`s = "banana"; r = s.count("a")` cannot be learned by this model at all: the
tokenizer's `[A-Za-z]+` rule makes `banana` a *single* token, so the letters
are not visible to the model in any representation. It would be a task whose
answer is a function of information the input does not carry. Counting *list
elements* (`len(words)`) is fine and appears in `code_index`, because each word
is its own token.

**Snippets are one statement per line and never nest more than one level.**
Tokens carry their leading whitespace, so `"\\n    x"` and `" x"` are different
symbols and a deeply indented corpus builds a parallel vocabulary of
indent-prefixed identifiers. The generators here emit exactly six line-initial
forms in total (`\\nfor`, `\\n    for`, `\\nwhile`, `\\nif`, `\\nelse`, `\\nr`),
all of them high frequency; `test_code_corpus.py` pins that set so a future
task cannot quietly widen it.

**Every intermediate stays two-digit.** Loop counts, steps and list values are
bounded so no running total passes 99 and no product leaves
two-digit-by-one-digit. `code_nested_loop` caps both ranges at 9 for a reason
that is easy to miss: `decompose_product` writes `10 x 7 = 70, 2 x 7 = 14` and
deliberately does *not* write the total, because that is the form that scored
0.93. That is fine when the product is the answer, and wrong when it is an
operand for a later step -- the model would have to carry an unwritten number.
So the outer range is single-digit and the iteration count is one written step.

## Repetition, and where this family is weakest

v74's multiplication task holds 712 distinct problems repeated ~56x and scored
0.93; the same operation with 24,000 unique problems scored 0.03. Distinct
capacity is therefore a design parameter, not an accident, and
`distinct_capacity()` reports it per task. At `--per_task 20000` the family
runs from 28x repetition (`code_divmod`) down to 3.9x (`code_index`), and
`code_index` is the one to narrow first if the family underperforms: its list
carries a distractor element that multiplies the prompt space without adding a
step to the procedure.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

# Imported rather than re-implemented on purpose. `decompose_product` is the
# exact working format that scored 0.93 on v74's multiplication task, down to
# whether the addition of the partial products is written out; a second copy of
# it here would drift from that measurement the first time either file changed.
from build_omni_corpus import (  # noqa: E402
    DEFAULT_SEQUENCE_LENGTH,
    decompose_product,
    extract_answer,
    token_budget_report,
)

#: Wall clock a single snippet gets before it is abandoned.
#:
#: Every snippet these generators emit is bounded by construction -- the
#: longest runs 270 iterations -- so this never fires in practice. It exists
#: because "the templates are all safe" is an assumption, and an assumption
#: that hangs the build for good is worse than one that drops a row.
DEFAULT_TIMEOUT_SECONDS = 2.0

#: Refuse a snippet longer than this before parsing it. Generated snippets top
#: out near 90 characters.
MAX_SOURCE_CHARS = 400

#: Consecutive failures to produce a new prompt before a task is judged
#: exhausted, matching `build_omni_corpus.EXHAUSTION_MISSES`.
EXHAUSTION_MISSES = 5000


# -- the checker ------------------------------------------------------------
#
# Running generated code is the whole point, so the sandbox is written to be
# read: an allowlist of syntax, an allowlist of names, no builtins beyond six
# pure functions, and a wall clock. Nothing here ever executes a string that
# came from outside this module -- `verify` runs `problem.canonical`, and the
# only thing that writes a `canonical` is a generator in `TASKS`.


class SnippetRejected(Exception):
    """The snippet uses syntax the corpus does not permit."""


#: Node types a corpus snippet may contain. Matched by class *name* rather than
#: by class object so this survives the AST changes between Python versions
#: (`ast.Index` disappeared from parse output in 3.9).
ALLOWED_NODES = frozenset({
    "Module", "Expr", "Assign", "AugAssign", "For", "While", "If", "Break",
    "Continue", "Pass", "Name", "Store", "Load", "Constant", "BinOp", "UnaryOp",
    "USub", "UAdd", "Add", "Sub", "Mult", "FloorDiv", "Mod", "Compare", "Eq",
    "NotEq", "Lt", "LtE", "Gt", "GtE", "List", "Subscript", "Index", "Call",
    # `IfExp` is the conditional expression `a if cond else b`, used by
    # `code_conditional` because the three-line block form left a 34-token
    # prompt where the model's 128-token context allows 32 alongside a
    # full-length generation.
    #
    # Worth recording how this was found: adding the ternary without this entry
    # sent the build's drop rate from 0.000 to 0.111 -- exactly one task in
    # nine -- because every `code_conditional` row failed validation and was
    # discarded. The verifier caught a change its author had not thought
    # through, which is the whole argument for validating snippets this module
    # generates itself.
    "IfExp",
    # `Attribute` is admitted for v93's `code_list_count` (`nums.count(v)`),
    # and only in the shape `validate_snippet` checks below: a method named in
    # `SAFE_METHODS`, called on a plain name. An attribute that is not a call
    # on an allowlisted method -- `().__class__`, `x.__dict__` -- is still
    # refused, so the sandbox-escape surface this node type usually opens is
    # not opened here.
    "Attribute",
})

#: The only callables a snippet may name. All six are pure, total on the
#: arguments these templates pass, and cannot reach the filesystem, the network
#: or the interpreter's own state.
SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "sum": sum,
}

#: The only methods a snippet may call, by attribute name. `list.count` is
#: pure and total; it is the one v93 needs.
SAFE_METHODS = frozenset({"count"})


def validate_snippet(source: str) -> ast.Module:
    """Parse `source` and refuse anything outside the allowlist.

    Belt and braces: these snippets are built from templates in this file, so
    in normal operation nothing can reach here that is not already safe. The
    validator is what makes that a *checked* claim instead of a stated one --
    if someone later adds a template with an attribute access or an import, the
    build fails loudly at that template rather than silently gaining a
    capability.
    """

    if len(source) > MAX_SOURCE_CHARS:
        raise SnippetRejected(f"snippet is {len(source)} chars, limit {MAX_SOURCE_CHARS}")
    try:
        tree = ast.parse(source, filename="<code-corpus>", mode="exec")
    except SyntaxError as exc:  # a template that does not parse is a bug, not a row
        raise SnippetRejected(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        name = type(node).__name__
        if name not in ALLOWED_NODES:
            raise SnippetRejected(f"disallowed syntax: {name}")
        if isinstance(node, ast.Name):
            # `_`-prefixed names are how sandbox escapes are written
            # (`().__class__`, `__builtins__`); no template needs one.
            if node.id.startswith("_"):
                raise SnippetRejected(f"disallowed name: {node.id}")
        if isinstance(node, ast.Attribute):
            # Only `<name>.<safe method>`; anything else is the escape shape.
            if node.attr not in SAFE_METHODS or not isinstance(node.value, ast.Name):
                raise SnippetRejected(f"disallowed attribute: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                pass  # checked as an Attribute above
            elif not isinstance(node.func, ast.Name):
                raise SnippetRejected("only direct calls to allowed builtins")
            elif node.func.id not in SAFE_BUILTINS:
                raise SnippetRejected(f"disallowed call: {node.func.id}")
            if node.keywords:
                raise SnippetRejected("keyword arguments are not permitted")
    return tree


@dataclass
class SnippetResult:
    """What happened when the interpreter ran a snippet."""

    ok: bool
    value: Optional[float] = None
    reason: str = ""


def run_snippet(source: str, target: str,
                timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> SnippetResult:
    """Execute `source` and read `target` out of the resulting namespace.

    Restricted three ways: the AST allowlist above, a `__builtins__` mapping
    holding six pure functions, and a wall clock. The snippet runs on a daemon
    thread so a hang cannot wedge the build -- a thread that overruns is
    abandoned rather than joined, which is the honest thing to do given CPython
    offers no way to kill one, and is safe here because the thread can only
    touch its own namespace.

    A non-numeric or non-integral result is a failure, not a row: the whole
    corpus answers with the last integer in the reply.
    """

    try:
        validate_snippet(source)
        compiled = compile(source, "<code-corpus>", "exec")
    except SnippetRejected as exc:
        return SnippetResult(False, None, str(exc))
    except (SyntaxError, ValueError) as exc:
        return SnippetResult(False, None, f"compile failed: {exc}")

    outcome: Dict[str, Any] = {}

    def _run() -> None:
        namespace: Dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS)}
        try:
            exec(compiled, namespace)  # noqa: S102 - the point of the module
        except BaseException as exc:  # noqa: BLE001 - any failure drops the row
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            return
        if target not in namespace:
            outcome["error"] = f"snippet never assigned {target}"
            return
        outcome["value"] = namespace[target]

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        return SnippetResult(False, None, f"timed out after {timeout_seconds}s")
    if "error" in outcome:
        return SnippetResult(False, None, outcome["error"])

    value = outcome.get("value")
    if isinstance(value, bool) or not isinstance(value, int):
        # bool is an int subclass; `r = True` would silently score as 1.
        return SnippetResult(False, None, f"{target} is {type(value).__name__}, not int")
    return SnippetResult(True, float(value), "")


@dataclass
class CodeProblem:
    """One snippet, its worked derivation, and the value the interpreter got.

    `canonical` is the executable source itself, which is the exact parallel to
    `OmniProblem.canonical` holding the solver's own query: it is the form the
    checker consumes, and it is decoupled from the phrasing the model sees.
    """

    task: str
    domain: str
    prompt: str
    response: str
    answer: float
    unit: str
    canonical: str
    target: str
    params: Dict[str, float] = field(default_factory=dict)

    def to_row(self, keep_canonical: bool = False) -> Dict[str, str]:
        row = {
            "user": self.prompt,
            "assistant": self.response,
            "domain": self.domain,
            "task": self.task,
        }
        if keep_canonical:
            # Carrying the source makes a shipped row re-runnable months later
            # without regenerating it, which is what the omni corpus wanted
            # `keep_canonical` for and could only get for 41.9% of its rows.
            # Here it is 100%: the checker parses exactly what it emitted.
            row["canonical"] = self.canonical
            row["target"] = self.target
        return row


DOMAIN = "programming"


def _pick(rng: random.Random, templates: Sequence[str], **values: str) -> str:
    """Choose one phrasing. Values are pre-rendered strings, never numbers.

    v74 carried one template per task, scored 0.894 on its own benchmark and
    answered 0 of 5 naturally-typed questions. Every task below carries five.
    """

    return rng.choice(list(templates)).format(**values)


#: The five ways a trace question gets asked. Two put the code first and three
#: put it last, because a model that only ever saw `<instruction><code>` learns
#: the position as part of the task.
#: Every template wraps the snippet on ONE side only, never both.
#:
#: The budget here is not the 128-token turn -- it is the model's context. A
#: prompt must leave room for a full-length generation, because
#: `max_position_embeddings` is 128 and a reply that runs to the cap would ask
#: for a position the rotary table does not have. `test_eval_v82` asserts
#: `longest_prompt + DEFAULT_MAX_NEW_TOKENS <= 128`, so at a 96-token cap the
#: longest prompt must fit in 32.
#:
#: The dropped template was "Run this snippet in your head. {code} Give the
#: value of {var}." -- a lead-in AND a trailing instruction around a two-line
#: snippet, which reached 39 tokens on `code_conditional`. One wrapper says the
#: same thing. Phrasing variety matters (v74 had one template per task and
#: scored 0 of 5 on naturally-typed questions), so the count is kept at five by
#: replacing it rather than deleting it.
TRACE_TEMPLATES = (
    "Trace this Python and give {var}. {code}",
    "{code} What is {var} after this runs?",
    "What does {var} hold at the end? {code}",
    "{code} Give the value of {var}.",
    "Python: {code} What is {var}?",
)

#: The same question, worded tightly, for tasks whose snippet is already long.
#:
#: Measured, and the result was not what it looked like. `code_nested_loop`
#: reached 35 tokens against the 32 the context allows beside a full-length
#: generation, and the obvious suspects cost nothing at all: dropping the
#: four-space indent to one space saved 0 tokens, and fixing the start value at
#: 0 saved 0. The tokenizer normalises runs of whitespace, and every digit is
#: already its own token under `--digit_tokens`.
#:
#: The entire overhead was the wrapper. "What does c hold at the end?" costs
#: four tokens more than "What is c?", which is the whole overrun. So the
#: snippet is left exactly as it is -- it is the thing being taught -- and only
#: the words around it are trimmed.
#:
#: Kept at five entries because phrasing variety is what stops the model
#: learning a template instead of a task: v74 had one phrasing per task and
#: answered 0 of 5 naturally-typed questions.
COMPACT_TRACE_TEMPLATES = (
    "{code} What is {var}?",
    "{code} Give {var}.",
    "Python: {code} {var}?",
    "Trace this. {code} {var}?",
    "{code} Final {var}?",
)


# -- task generators --------------------------------------------------------


def _loop_add(rng: random.Random) -> CodeProblem:
    """`x = s` then `n` additions of `k`. The plainest accumulator there is.

    `n` stops at 7 and `s + n * k` at 72 for two separate reasons: the running
    total must stay two-digit (rule 1), and each written step costs about eight
    tokens, so an eighth step pushed the longest turns past the 128-token block
    where turn-aligned packing drops them outright.
    """

    start = rng.randint(0, 9)
    steps = rng.randint(3, 7)
    increment = rng.randint(2, 9)
    source = f"x = {start}\nfor i in range({steps}): x = x + {increment}"
    prompt = _pick(rng, TRACE_TEMPLATES, code=source, var="x")

    value = start
    pieces = [f"the loop runs {steps} times", f"x starts at {start}"]
    for _ in range(steps):
        pieces.append(f"{value} + {increment} = {value + increment}")
        value += increment
    pieces.append(f"x is {value}")
    response = ", ".join(pieces) + f", total {value}"
    return CodeProblem("code_loop_add", DOMAIN, prompt, response, float(value), "",
                       source, "x",
                       {"start": start, "steps": steps, "increment": increment})


def _loop_subtract(rng: random.Random) -> CodeProblem:
    """The same accumulator downwards, so the model does not learn `for` = `+`.

    The start is redrawn until the final value is non-negative: a corpus of
    negative results would teach the sign as a property of the loop rather than
    of the arithmetic, and the benchmark's `-?\\d+` extraction makes a stray
    minus a wrong answer rather than a near miss.
    """

    steps = rng.randint(3, 7)
    decrement = rng.randint(2, 9)
    span = steps * decrement
    start = rng.randint(span, 99)
    source = f"x = {start}\nfor i in range({steps}): x = x - {decrement}"
    prompt = _pick(rng, TRACE_TEMPLATES, code=source, var="x")

    value = start
    pieces = [f"the loop runs {steps} times", f"x starts at {start}"]
    for _ in range(steps):
        pieces.append(f"{value} - {decrement} = {value - decrement}")
        value -= decrement
    pieces.append(f"x is {value}")
    response = ", ".join(pieces) + f", total {value}"
    return CodeProblem("code_loop_subtract", DOMAIN, prompt, response, float(value), "",
                       source, "x",
                       {"start": start, "steps": steps, "decrement": decrement})


#: Values a literal list is built from. Three drawn without replacement out of
#: fifteen gives 2,730 distinct lists -- 7.3x repetition at `--per_task 20000`,
#: the same order as v74's winning 56x rather than the omni build's 1.7x.
LIST_POOL = range(2, 17)

#: `code_index` draws from a narrower pool because its prompt space is already
#: multiplied by the choice of index pair; see the module docstring.
INDEX_POOL = range(2, 14)


def _list_sum(rng: random.Random) -> CodeProblem:
    """`sum` over a three-item literal, with **the operands written out**.

    This is the task the omni corpus got wrong. v86's `average` writes
    `sum: 48 then 137 then 170` -- a running total with no operands -- and its
    individual additions are correct 1.5% of the time while its division is
    correct 69%. Writing `48 + 89 = 137` instead is the entire difference, and
    it is why a list-sum task earns its place next to an arithmetic one.
    """

    values = rng.sample(list(LIST_POOL), 3)
    literal = ", ".join(str(v) for v in values)
    source = f"nums = [{literal}]\nr = sum(nums)"
    prompt = _pick(rng, COMPACT_TRACE_TEMPLATES, code=source, var="r")

    running = values[0]
    pieces = ["sum adds the items in order", f"r starts at {running}"]
    for value in values[1:]:
        pieces.append(f"{running} + {value} = {running + value}")
        running += value
    pieces.append(f"r is {running}")
    response = ", ".join(pieces) + f", total {running}"
    return CodeProblem("code_list_sum", DOMAIN, prompt, response, float(running), "",
                       source, "r", {f"v{i}": v for i, v in enumerate(values)})


def _list_extreme(rng: random.Random) -> CodeProblem:
    """`max` or `min` over the same literal, as a chain of pairwise compares.

    Both directions in one task rather than two, so the model has to read which
    builtin was called instead of learning "a list question means the biggest
    number". The canonical source records which was asked; the phrasing does
    not mention it at all.
    """

    values = rng.sample(list(LIST_POOL), 3)
    literal = ", ".join(str(v) for v in values)
    wants_max = rng.random() < 0.5
    builtin = "max" if wants_max else "min"
    word = "larger" if wants_max else "smaller"
    source = f"nums = [{literal}]\nr = {builtin}(nums)"
    prompt = _pick(rng, COMPACT_TRACE_TEMPLATES, code=source, var="r")

    best = values[0]
    pieces = [f"{builtin} keeps the {word} of each pair"]
    for value in values[1:]:
        chosen = max(best, value) if wants_max else min(best, value)
        pieces.append(f"compare {best} and {value}, the {word} is {chosen}")
        best = chosen
    pieces.append(f"r is {best}")
    response = ", ".join(pieces) + f", total {best}"
    return CodeProblem("code_list_extreme", DOMAIN, prompt, response, float(best), "",
                       source, "r",
                       {f"v{i}": v for i, v in enumerate(values)})


def _index(rng: random.Random) -> CodeProblem:
    """Two positions read out of a list and added.

    The derivation states `count positions from zero` every time. Zero-based
    indexing is the one fact in this family that is a convention rather than a
    computation, and a convention the derivation never names is a convention
    the model has to infer from the numbers.
    """

    values = rng.sample(list(INDEX_POOL), 3)
    literal = ", ".join(str(v) for v in values)
    first, second = sorted(rng.sample(range(3), 2))
    source = f"nums = [{literal}]\nr = nums[{first}] + nums[{second}]"
    prompt = _pick(rng, COMPACT_TRACE_TEMPLATES, code=source, var="r")

    left, right = values[first], values[second]
    answer = left + right
    response = (f"count positions from zero, "
                f"nums [ {first} ] is {left}, nums [ {second} ] is {right}, "
                f"{left} + {right} = {answer}, r is {answer}, total {answer}")
    return CodeProblem("code_index", DOMAIN, prompt, response, float(answer), "",
                       source, "r",
                       {f"v{i}": v for i, v in enumerate(values)},)


def _conditional(rng: random.Random) -> CodeProblem:
    """`if a > b` picking which subtraction to run.

    Both operands stay two-digit and the branch is the only decision, so a
    wrong answer localises: taking the wrong branch gives a negative number,
    which is visibly different from an arithmetic slip.
    """

    a = rng.randint(11, 60)
    b = rng.randint(11, 60)
    while b == a:
        b = rng.randint(11, 60)
    # A conditional *expression*, not a three-line if/else block.
    #
    # The constraint this hit is the model's CONTEXT, not the turn budget:
    # `max_position_embeddings` is 128, so a prompt must leave room for a
    # full-length generation, or a reply running to the 96-token cap asks the
    # rotary table for a position it does not have. The block form reached 34
    # tokens against the 32 that leaves, and joining its lines with spaces is
    # not valid Python -- the interpreter rejects it, which is the executor
    # doing its job.
    #
    # `--digit_tokens` is why this is tight at all: every digit is its own
    # token, so `a = 59; b = 50` spends six tokens on the numbers alone.
    #
    # The ternary keeps what the task teaches -- both branches are visible, and
    # taking the wrong one still yields a negative number, so a branch error
    # stays distinguishable from an arithmetic slip.
    source = f"a = {a}; b = {b}\nr = a - b if a > b else b - a"
    prompt = _pick(rng, COMPACT_TRACE_TEMPLATES, code=source, var="r")

    if a > b:
        branch = (f"{a} is greater than {b}, so the if branch runs, r = a - b")
        answer = a - b
        step = f"{a} - {b} = {answer}"
    else:
        branch = (f"{a} is not greater than {b}, so the else branch runs, r = b - a")
        answer = b - a
        step = f"{b} - {a} = {answer}"
    response = (f"compare a and b, {branch}, {step}, "
                f"r is {answer}, total {answer}")
    return CodeProblem("code_conditional", DOMAIN, prompt, response, float(answer), "",
                       source, "r", {"a": a, "b": b})


def _divmod_task(rng: random.Random) -> CodeProblem:
    """`a // b` or `a % b`, worked as one multiplication and one subtraction.

    `a` is capped at `10 * b - 1` so the quotient is always a single digit.
    That is not tidiness: `decompose_product` writes two partial products and
    stops, by design, so a two-digit quotient would put the product itself
    (`40 x 2 = 80, 9 x 2 = 18` -> 98) nowhere in the text while the next step
    subtracts it. Every number this derivation uses is a number it wrote.
    """

    b = rng.randint(3, 9)
    a = rng.randint(10, min(99, 10 * b - 1))
    quotient, remainder = divmod(a, b)
    floor_division = rng.random() < 0.5
    operator = "//" if floor_division else "%"
    source = f"a = {a}; b = {b}; r = a {operator} b"
    prompt = _pick(rng, COMPACT_TRACE_TEMPLATES, code=source, var="r")

    working = (f"{a} {operator} {b}, count whole {b} s inside {a}, "
               f"{decompose_product(quotient, b)}, "
               f"{a} - {quotient * b} = {remainder}, "
               f"{remainder} is less than {b}")
    if floor_division:
        answer = quotient
        response = (f"{working}, the quotient is {quotient}, "
                    f"r is {quotient}, total {quotient}")
    else:
        answer = remainder
        response = (f"{working}, the remainder is {remainder}, "
                    f"r is {remainder}, total {remainder}")
    return CodeProblem("code_divmod", DOMAIN, prompt, response, float(answer), "",
                       source, "r", {"a": a, "b": b, "q": quotient})


def _nested_loop(rng: random.Random) -> CodeProblem:
    """How many times a doubly-nested body runs, plus a starting count.

    Both ranges are single-digit so `decompose_product(outer, inner)` is one
    written step and the iteration count is a number on the page before the
    final addition consumes it. `start + outer * inner` is capped at 99 by
    redrawing, keeping that addition a two-digit single step.
    """

    outer = rng.randint(2, 9)
    inner = rng.randint(2, 9)
    start = rng.randint(0, 9)
    while start + outer * inner > 99:
        outer = rng.randint(2, 9)
        inner = rng.randint(2, 9)
    iterations = outer * inner
    answer = start + iterations
    source = (f"c = {start}\n"
              f"for i in range({outer}):\n"
              f"    for j in range({inner}): c = c + 1")
    prompt = _pick(rng, COMPACT_TRACE_TEMPLATES, code=source, var="c")

    response = (f"the inner loop runs {inner} times for each outer step, "
                f"the outer loop runs {outer} times, "
                f"{decompose_product(outer, inner)} iterations, "
                f"c starts at {start}, {start} + {iterations} = {answer}, "
                f"c is {answer}, total {answer}")
    return CodeProblem("code_nested_loop", DOMAIN, prompt, response, float(answer), "",
                       source, "c",
                       {"outer": outer, "inner": inner, "start": start})


def _while_accumulate(rng: random.Random) -> CodeProblem:
    """A `while` that stops the first time the total passes a threshold.

    Written as `while` rather than `for ... if ...: break` for a tokenizer
    reason, not a semantic one: the `break` form needs an indented two-statement
    body, which adds `\\n    x` and `\\n    if` to the vocabulary as symbols
    unrelated to ` x` and ` if`. The `while` form is one line and teaches the
    same thing -- accumulate, test, exit.

    The comparison is restated after *every* step. Emitting only the final one
    would leave the stopping rule implicit, and rule 4 of the envelope is that
    a scratchpad helps only where it decomposes the operation being performed.
    """

    # Three or four iterations, not three to five.
    #
    # Each iteration emits two pieces -- the addition and the restated
    # comparison -- so the reply grows about twice as fast per step as the other
    # code tasks. Measured over 1,080 rows with the v86 tokenizer, five
    # iterations reached a 129-token turn against the 128-token budget, and
    # `_build_turn_aligned_tensors` drops a turn longer than the block *without
    # reporting it*. Five of 1,080 rows were being silently discarded.
    #
    # Capping iterations rather than dropping the restated comparison is
    # deliberate: the comparison is what makes the stopping rule explicit, and
    # rule 4 of the envelope is that a scratchpad helps only where it decomposes
    # the operation. Losing a little capacity is the cheaper trade.
    step = rng.randint(5, 19)
    iterations = rng.randint(3, 4)
    threshold = rng.randint((iterations - 1) * step, iterations * step - 1)
    source = f"x = 0\nwhile x <= {threshold}: x = x + {step}"
    prompt = _pick(rng, TRACE_TEMPLATES, code=source, var="x")

    value = 0
    pieces = ["x starts at 0"]
    for _ in range(iterations):
        nxt = value + step
        pieces.append(f"{value} + {step} = {nxt}")
        value = nxt
        if value <= threshold:
            pieces.append(f"{value} is not past {threshold}")
        else:
            pieces.append(f"{value} is past {threshold}, the loop stops")
    pieces.append(f"x is {value}")
    response = ", ".join(pieces) + f", total {value}"
    return CodeProblem("code_while_accumulate", DOMAIN, prompt, response, float(value), "",
                       source, "x",
                       {"step": step, "threshold": threshold, "iterations": iterations})


# -- v93 generators ----------------------------------------------------------
#
# Three more tasks (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md, D7), verified by
# execution like the nine above. They live in `V93_TASKS` rather than `TASKS`
# for the reason `build_omni_corpus` gives: `build` draws every task from one
# RNG in `chosen` order, so a default build stays byte-identical only if the
# default list does; and the benchmark registry built from `TASKS` carries the
# fingerprint every published receipt is paired on.

#: The most terms a `code_range_sum` trace writes out. Measured with the
#: training segmentation against the benchmark's 96-token generation cap
#: (`eval_problem_solving.DEFAULT_MAX_NEW_TOKENS`): the longest ten-term
#: trace, `range(3, 13)`, is 95 reply tokens including EOS, and an eleven-term
#: one is 97 or more. One step per addition is the rule this family rests on,
#: so the term count is bounded rather than the trace compressed. The map's
#: suggestion of `range(13)` alone would have needed thirteen terms and 115
#: tokens, a reply the benchmark could never finish.
RANGE_SUM_MAX_TERMS = 10

#: `range` stops below this. Sums stay two-digit: `sum(range(1, 13)) = 78`.
RANGE_SUM_MAX_STOP = 13


def _range_sum(rng: random.Random) -> CodeProblem:
    """`r = sum(range(n))` or `r = sum(range(a, n))`, every addition written.

    The one-argument form alone has eight legal instances under the term
    bound, which is a lookup table rather than a procedure. The two-argument
    form is what makes it a procedure: the trace has to read where `range`
    starts as well as where it stops, and the sum then depends on both. A
    start of 0 is written as `range(n)`, so both spellings the model will meet
    are taught, and the trace names the first and last values so nothing about
    the half-open interval is left to be inferred.
    """

    stop = rng.randint(3, RANGE_SUM_MAX_STOP)
    # The start must leave at least two terms and at most the trace budget.
    start = rng.randint(max(0, stop - RANGE_SUM_MAX_TERMS), stop - 2)
    call = f"range({stop})" if start == 0 else f"range({start}, {stop})"
    source = f"r = sum({call})"
    prompt = _pick(rng, TRACE_TEMPLATES, code=source, var="r")

    terms = list(range(start, stop))
    running = terms[0]
    pieces = [f"range counts from {start} up to {stop - 1}", f"r starts at {running}"]
    for term in terms[1:]:
        pieces.append(f"{running} + {term} = {running + term}")
        running += term
    pieces.append(f"r is {running}")
    response = ", ".join(pieces) + f", total {running}"
    return CodeProblem("code_range_sum", DOMAIN, prompt, response, float(running), "",
                       source, "r", {"start": start, "stop": stop})


#: Values a `code_list_count` list is built from. Six values in four slots keep
#: the space near the family's other list tasks (4,020 distinct problems,
#: 5.0x repetition at 20,000 rows) rather than the 96,040 a five-slot list
#: over eight values would give.
COUNT_POOL = range(2, 8)
COUNT_LENGTH = 4


def _list_count(rng: random.Random) -> CodeProblem:
    """`r = nums.count(v)` over a four-item literal, one verdict per item.

    The counted value appears one, two or three times; the other slots hold
    values that differ from it, so every `matches` / `does not` in the trace is
    decidable from the two numbers on the line. `.count` is the first method
    call in this family, admitted through `SAFE_METHODS` rather than by
    widening the call rule in general.
    """

    value = rng.choice(list(COUNT_POOL))
    hits = rng.randint(1, 3)
    positions = set(rng.sample(range(COUNT_LENGTH), hits))
    others = [v for v in COUNT_POOL if v != value]
    items = [value if i in positions else rng.choice(others) for i in range(COUNT_LENGTH)]
    literal = ", ".join(str(v) for v in items)
    source = f"nums = [{literal}]\nr = nums.count({value})"
    prompt = _pick(rng, COMPACT_TRACE_TEMPLATES, code=source, var="r")

    pieces = [f"count checks each item against {value}"]
    for item in items:
        pieces.append(f"{item} matches" if item == value else f"{item} does not")
    pieces.append(f"r is {hits}")
    response = ", ".join(pieces) + f", total {hits}"
    return CodeProblem("code_list_count", DOMAIN, prompt, response, float(hits), "",
                       source, "r", {"value": value, "hits": hits,
                                     **{f"v{i}": v for i, v in enumerate(items)}})


def _neg_index(rng: random.Random) -> CodeProblem:
    """`r = nums[-k]`: one position read from the end of a three-item list.

    `code_index` states `count positions from zero` on every row because
    zero-based indexing is a convention the numbers alone do not reveal. The
    negative form is the same kind of fact from the other end, and the trace
    walks back one position at a time so `nums [ -2 ]` is reached from
    `nums [ -1 ]` rather than asserted.
    """

    values = rng.sample(list(INDEX_POOL), 3)
    literal = ", ".join(str(v) for v in values)
    k = rng.randint(1, 3)
    source = f"nums = [{literal}]\nr = nums[-{k}]"
    prompt = _pick(rng, COMPACT_TRACE_TEMPLATES, code=source, var="r")

    pieces = ["negative positions count from the end"]
    for back in range(1, k + 1):
        pieces.append(f"nums [ -{back} ] is {values[-back]}")
    answer = values[-k]
    pieces.append(f"r is {answer}")
    response = ", ".join(pieces) + f", total {answer}"
    return CodeProblem("code_neg_index", DOMAIN, prompt, response, float(answer), "",
                       source, "r", {"k": k, **{f"v{i}": v for i, v in enumerate(values)}})


#: Every v89 generator, by task name. The `code_` prefix keeps them
#: distinguishable in a receipt and guarantees they never shadow an arithmetic
#: or omni task when `eval_problem_solving` merges the three families into one
#: dict. Frozen: see the note above `RANGE_SUM_MAX_TERMS`.
TASKS: Dict[str, Callable[[random.Random], CodeProblem]] = {
    "code_loop_add": _loop_add,
    "code_loop_subtract": _loop_subtract,
    "code_list_sum": _list_sum,
    "code_list_extreme": _list_extreme,
    "code_index": _index,
    "code_conditional": _conditional,
    "code_divmod": _divmod_task,
    "code_nested_loop": _nested_loop,
    "code_while_accumulate": _while_accumulate,
}

#: The tasks added for v93. Built only when named with `--task`.
V93_TASKS: Dict[str, Callable[[random.Random], CodeProblem]] = {
    "code_range_sum": _range_sum,
    "code_list_count": _list_count,
    "code_neg_index": _neg_index,
}

#: Every generator this module can build, for lookup by name.
ALL_TASKS: Dict[str, Callable[[random.Random], CodeProblem]] = {**TASKS, **V93_TASKS}


#: How many distinct problems each generator can produce, counted from the
#: parameter ranges in its body rather than sampled.
#:
#: This is reported because v74's multiplication task scored 0.93 with 712
#: distinct problems and 0.03 with 24,000 of the same operation. A task whose
#: capacity drifts upward silently is a task whose score will fall for a reason
#: nobody attributes to the corpus. `test_code_corpus.py` samples each
#: generator and asserts the number below is not an undercount.
def distinct_capacity() -> Dict[str, int]:
    pool = len(LIST_POOL)
    index_pool = len(INDEX_POOL)
    return {
        # start(10) x steps(5) x increment(8)
        "code_loop_add": 10 * 5 * 8,
        # sum over steps(3..7), decrement(2..9) of the legal starts 99-span+1
        "code_loop_subtract": sum(
            max(0, 99 - steps * dec + 1)
            for steps in range(3, 8) for dec in range(2, 10)
        ),
        # ordered 3-subsets of the pool
        "code_list_sum": pool * (pool - 1) * (pool - 2),
        # the same, doubled by max/min
        "code_list_extreme": 2 * pool * (pool - 1) * (pool - 2),
        # ordered 3-subsets of the narrower pool x 3 index pairs
        "code_index": index_pool * (index_pool - 1) * (index_pool - 2) * 3,
        # a(50) x b(50) minus the diagonal
        "code_conditional": 50 * 50 - 50,
        # sum over b of the legal a range, doubled by // and %
        "code_divmod": 2 * sum(min(99, 10 * b - 1) - 10 + 1 for b in range(3, 10)),
        # (outer, inner) pairs with a legal start
        "code_nested_loop": sum(
            1 for outer in range(2, 10) for inner in range(2, 10)
            for start in range(10) if start + outer * inner <= 99
        ),
        # step(15) x iterations(3) x the `step` thresholds each admits
        "code_while_accumulate": sum(
            step for step in range(5, 20) for _ in range(3)
        ),
        # -- v93 --
        # (start, stop) pairs with 2..RANGE_SUM_MAX_TERMS terms; 71 today
        "code_range_sum": sum(
            (stop - 2) - max(0, stop - RANGE_SUM_MAX_TERMS) + 1
            for stop in range(3, RANGE_SUM_MAX_STOP + 1)
        ),
        # value(6) x sum over hits of C(4, hits) x 5^(4 - hits)
        "code_list_count": len(COUNT_POOL) * sum(
            math.comb(COUNT_LENGTH, hits) * (len(COUNT_POOL) - 1) ** (COUNT_LENGTH - hits)
            for hits in range(1, 4)
        ),
        # ordered 3-subsets of the narrower pool x 3 offsets
        "code_neg_index": index_pool * (index_pool - 1) * (index_pool - 2) * 3,
    }


# -- verification -----------------------------------------------------------


def verify(problem: CodeProblem,
           timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> SnippetResult:
    """Run the snippet the prompt shows and compare it to the worked answer.

    Returns the full `SnippetResult` rather than a bool so the build can report
    *why* rows were dropped. A verifier whose failures are an unlabelled count
    cannot tell a broken template from an exhausted parameter range.
    """

    result = run_snippet(problem.canonical, problem.target, timeout_seconds)
    if not result.ok:
        return result
    if result.value != problem.answer:
        return SnippetResult(
            False, result.value,
            f"interpreter got {result.value}, derivation says {problem.answer}",
        )
    return result


#: The words a snippet line may begin with. Pinned because tokens carry their
#: leading whitespace: `"\n    for"` is a different symbol from `" for"`, so
#: every new line-initial form is a new vocabulary entry the model must learn
#: separately. Six is cheap; an indented multi-statement body would not be.
LINE_INITIAL_FORMS = frozenset({
    "for", "    for", "while", "if", "else", "r",
})


def line_initial_forms(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    """Count the distinct indent-prefixed line starts a corpus contains."""

    counts: Dict[str, int] = {}
    for row in rows:
        for line in row["user"].split("\n")[1:]:
            word = line.split(" ")[0] if line.strip() else ""
            # Keep the indentation, because that is what the tokenizer keeps.
            leading = len(line) - len(line.lstrip(" "))
            form = " " * leading + word.strip()
            if form:
                counts[form] = counts.get(form, 0) + 1
    return dict(sorted(counts.items()))


# -- corpus construction ----------------------------------------------------


def build(per_task: int, seed: int, tasks: Optional[Sequence[str]] = None,
          repeat: bool = True, keep_canonical: bool = False,
          timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
          sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
          token_budget: bool = False) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Generate, execute, verify, and return (rows, report).

    `repeat=True` is the default for the reason `build_omni_corpus` documents
    at length: repetition is how this model learns a procedure, and the
    benchmark draws unseen parameters from the same space so recall cannot pass
    it. `repeat=False` emits distinct prompts only and reports the shortfall
    when a generator's space runs out.
    """

    rng = random.Random(seed)
    # The default is the v89 nine; a v93 task is built only when named.
    chosen = list(tasks or TASKS)
    unknown = [name for name in chosen if name not in ALL_TASKS]
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(unknown)}")

    rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    dropped: Dict[str, int] = {}
    reasons: Dict[str, Dict[str, int]] = {}
    distinct: Dict[str, int] = {}
    short: Dict[str, Dict[str, object]] = {}
    attempts: Dict[str, int] = {}

    def draw(name: str) -> Optional[CodeProblem]:
        """One executed-and-agreed problem, or None. Counts its own drops."""

        attempts[name] = attempts.get(name, 0) + 1
        problem = ALL_TASKS[name](rng)
        result = verify(problem, timeout_seconds)
        parsed = extract_answer(problem.response)
        if not result.ok:
            dropped[name] = dropped.get(name, 0) + 1
            bucket = reasons.setdefault(name, {})
            bucket[result.reason] = bucket.get(result.reason, 0) + 1
            return None
        if parsed != problem.answer:
            # A right answer the benchmark cannot read scores as wrong, so this
            # is as fatal as an execution mismatch and is counted separately.
            dropped[name] = dropped.get(name, 0) + 1
            bucket = reasons.setdefault(name, {})
            key = f"response parses to {parsed}, answer is {problem.answer}"
            bucket[key] = bucket.get(key, 0) + 1
            return None
        return problem

    for name in chosen:
        made = 0
        seen: set = set()
        if repeat:
            for _ in range(per_task):
                problem = draw(name)
                if problem is None:
                    continue
                seen.add(problem.prompt)
                rows.append(problem.to_row(keep_canonical=keep_canonical))
                made += 1
        else:
            misses = 0
            while made < per_task and misses < EXHAUSTION_MISSES:
                problem = draw(name)
                if problem is None or problem.prompt in seen:
                    misses += 1
                    continue
                seen.add(problem.prompt)
                rows.append(problem.to_row(keep_canonical=keep_canonical))
                made += 1
                misses = 0
            if made < per_task:
                short[name] = {"asked": per_task, "produced": made,
                               "reason": "parameter space exhausted"}
        counts[name] = made
        distinct[name] = len(seen)

    total_attempts = sum(attempts.values())
    total_dropped = sum(dropped.values())
    report: Dict[str, Any] = {
        "schema": "supermix-v87-code-corpus-v1",
        "seed": seed,
        "rows": len(rows),
        "tasks": len(chosen),
        "per_task": counts,
        "distinct_prompts": distinct,
        "repetition": {k: round(counts[k] / v, 1) for k, v in distinct.items() if v},
        "distinct_capacity": {k: v for k, v in distinct_capacity().items()
                              if k in counts},
        "attempts": attempts,
        "dropped_failing_execution": dropped,
        "drop_reasons": reasons,
        "drop_rate": round(total_dropped / total_attempts, 6) if total_attempts else 0.0,
        "short_of_requested": short,
        "verified_by": "cpython exec in a restricted namespace",
        "verification": {
            "builtins": sorted(SAFE_BUILTINS),
            "methods": sorted(SAFE_METHODS),
            "timeout_seconds": timeout_seconds,
            "allowed_syntax": sorted(ALLOWED_NODES),
            "note": (
                "the checker executes the snippet the PROMPT shows, not the "
                "expression the generator used, so a template that renders "
                "something other than what it computes is dropped"
            ),
        },
        "line_initial_forms": line_initial_forms(rows),
        "options": {
            "repeat": repeat,
            "keep_canonical": keep_canonical,
        },
    }
    if token_budget:
        report["token_budget"] = token_budget_report(rows, sequence_length)
    return rows, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per_task", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--output", default="datasets/v87/v87_code.jsonl")
    parser.add_argument("--report", default=None)
    parser.add_argument("--unique", action="store_true",
                        help=("emit only distinct prompts. Off by default: v74's "
                              "multiplication task repeated 712 pairs 56x each and "
                              "scored 0.93, while a 24,000-unique build of the same "
                              "operation scored 0.03"))
    parser.add_argument("--task", action="append", default=[],
                        help=("restrict to these tasks; repeatable. The default is "
                              "the nine v89 tasks; the v93 tasks "
                              f"({', '.join(V93_TASKS)}) are built only when named"))
    parser.add_argument("--keep_canonical", action="store_true",
                        help="ship the executable source and target variable per row")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                        help="wall clock a single snippet gets before it is dropped")
    parser.add_argument("--token_budget_report", action="store_true",
                        help=("measure per-task response and turn lengths and what "
                              "turn-aligned packing would drop"))
    parser.add_argument("--sequence_length", type=int,
                        default=DEFAULT_SEQUENCE_LENGTH,
                        help="the block size the token budget is measured against")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, report = build(args.per_task, args.seed, args.task or None,
                         repeat=not args.unique,
                         keep_canonical=args.keep_canonical,
                         timeout_seconds=args.timeout,
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
    print(f"drop rate {report['drop_rate']:.4%} "
          f"({sum(report['dropped_failing_execution'].values()):,} of "
          f"{sum(report['attempts'].values()):,} attempts)")
    print(f"  {'task':<24} {'rows':>7} {'distinct':>9} {'x':>6} {'capacity':>9} "
          f"{'dropped':>8}")
    capacity = report["distinct_capacity"]
    for name, count in sorted(report["per_task"].items()):
        print(f"  {name:<24} {count:>7,} {report['distinct_prompts'][name]:>9,} "
              f"{report['repetition'].get(name, 0):>6.1f} "
              f"{capacity.get(name, 0):>9,} "
              f"{report['dropped_failing_execution'].get(name, 0):>8,}")

    budget = report.get("token_budget")
    if budget:
        print(f"\ntoken budget at sequence_length {budget['sequence_length']}")
        print(f"  {'task':<24} {'resp med':>8} {'p95':>5} {'max':>5} "
              f"{'turn max':>9} {'dropped':>8}")
        for name, stats in budget["tasks"].items():
            print(f"  {name:<24} {stats['response_median']:>8} "
                  f"{stats['response_p95']:>5} {stats['response_max']:>5} "
                  f"{stats['turn_max']:>9} {stats['dropped_fraction']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
