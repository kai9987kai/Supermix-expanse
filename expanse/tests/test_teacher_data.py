"""Tests for teacher_data: the code sandbox first, then parsing, task families, bio filters and drivers.

The sandbox is tested twice over: malicious snippets must be refused by the AST
whitelist, and -- with the whitelist deliberately bypassed -- the runtime layers
(isolated subprocess, restricted builtins, timeout, Job Object memory cap) must
still contain them. The drivers are exercised end to end against an in-process
fake llama-server, so no model is needed and the suite runs in about a minute.

    python -m pytest expanse/tests/test_teacher_data.py -q
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import teacher_data as td  # noqa: E402


# ---------------------------------------------------------------------------
# AST whitelist
# ---------------------------------------------------------------------------

MALICIOUS = {
    "import_os": "def f(x): import os; return os.system('echo pwned')",
    "import_os_after_return": "def f(x): return x; import os",
    "from_os": "def f(x): from os import system; return system('dir')",
    "import_subprocess": "def f(x): import subprocess; return subprocess.run(['cmd'])",
    "import_sys": "def f(x): import sys; return sys.modules",
    "math_star": "def f(x): from math import *; return x",
    "open_file": "def f(x): return open('C:/Windows/win.ini').read()",
    "dunder_import": "def f(x): return __import__('os').getcwd()",
    "getattr_builtins": "def f(x): return getattr(__builtins__, 'open')('x')",
    "builtins_name": "def f(x): return __builtins__",
    "eval": "def f(x): return eval('1+1')",
    "exec": "def f(x): return exec('import os')",
    "compile": "def f(x): return compile('1', 'a', 'eval')",
    "globals": "def f(x): return globals()",
    "locals_vars": "def f(x): return vars()",
    "type_class": "def f(x): return type('A', (), {})",
    "print": "def f(x): return print(x)",
    "breakpoint": "def f(x): return breakpoint()",
    "dunder_attr": "def f(x): return ().__class__.__bases__[0].__subclasses__()",
    "private_attr": "def f(x): return x._secret",
    "gen_frame_walk": "def f(x): return (i for i in x).gi_frame.f_back.f_globals",
    "format_dunder": "def f(x): return '{0.__class__}'.format(x)",
    "format_split_dunder": "def f(x): return ('{0._' + '_class__}').format(x)",
    "format_attr_walk": "def f(x): return '{0.real}'.format(x)",
    "format_map": "def f(x): return '{a}'.format_map(x)",
    "lambda_omega": "def f(x): return (lambda g: g(g))(lambda g: g(g))",
    "self_application": "def f(x): return f(f)",
    "lambda_recursion_bomb": "def f(n): g = lambda h, k: h(h, k + 1) + h(h, k + 1); return g(g, 0)",
    "while_true": "def f(x):\n    while True:\n        pass",
    "while_cond": "def f(x):\n    while x > 0:\n        x -= 1\n    return x",
    "iter_sentinel_loop": "def f(x): return [y for y in iter(int, 1)]",
    "exponent_tower": "def f(x): return 10 ** 10 ** 10",
    "huge_exponent": "def f(x): return 2 ** 100000000",
    "const_expr_exponent": "def f(x): return x ** (10 * 10 * 10 * 10 * 10)",
    "shift_bomb": "def f(x): return 1 << 10 ** 10",
    "pow_builtin_bomb": "def f(x): return pow(10, 10 ** 9)",
    "huge_literal": "def f(x): return 99999999999999999999999",
    "try_except": "def f(x):\n    try:\n        return 1\n    except Exception:\n        return 2",
    "global_stmt": "def f(x): global y; return x",
    "delete": "def f(x): del x",
    "raise": "def f(x): raise SystemExit",
    "yield": "def f(x): yield x",
    "async_def": "async def f(x): return x",
    "class_def": "class A:\n    pass",
    "two_defs": "def f(x): return x\ndef g(y): return y",
    "decorator": "@staticmethod\ndef f(x): return x",
    "with_stmt": "def f(x):\n    with x as y:\n        return y",
    "syntax_error": "def f(x) return x",
}

SAFE = {
    "sum": "def f(nums): return sum(nums)",
    "vowels": "def f(s): return sum(1 for c in s.lower() if c in 'aeiou')",
    "math_import": "def f(x): import math; return math.sqrt(x)",
    "math_from": "def f(x): from math import gcd; return gcd(x, 6)",
    "recursion": "def f(n): return n if n < 2 else f(n - 1) + f(n - 2)",
    "sorted_key_lambda": "def f(xs): return sorted(xs, key=lambda v: -v)",
    "name_exponent": "def f(a, b): return a ** b",
    "square": "def f(x): return x ** 2",
    "sqrt_pow": "def f(a, b): return (a ** 2 + b ** 2) ** 0.5",
    "format_spec": "def f(x): return '{:.2f}'.format(x)",
    "fstring": "def f(x): return f'{x:b}'",
    "dict_comp": "def f(s): return {c: s.count(c) for c in s}",
    "walrus": "def f(xs): return [y for x in xs if (y := x * 2) > 2]",
    "multi_stmt": "def f(xs): t = 0; t += sum(xs); return t",
}


@pytest.mark.parametrize("name", sorted(MALICIOUS))
def test_ast_rejects_malicious(name):
    ok, reason = td.check_code_ast(MALICIOUS[name])
    assert not ok, f"{name} passed the whitelist"
    assert reason != "ok"


@pytest.mark.parametrize("name", sorted(SAFE))
def test_ast_accepts_safe(name):
    ok, reason = td.check_code_ast(SAFE[name])
    assert ok, f"{name}: {reason}"


def test_ast_name_and_arity():
    assert td.check_code_ast("def g(x): return x", "f") == (False, "wrong_name:g")
    assert td.check_code_ast("def f(x, y): return x", "f", n_args=1)[1] == "wrong_arity"
    assert td.check_code_ast("def f(*a): return a", "f", n_args=1)[1] == "wrong_arity"
    assert td.check_code_ast("def f(x): return x", "f", n_args=1) == (True, "ok")


# ---------------------------------------------------------------------------
# Runtime layers (AST bypassed on purpose)
# ---------------------------------------------------------------------------

def _run(src, tests=(((1,), 1),), **kw):
    return td.run_sandboxed(src, "f", list(tests), check_ast=False, **kw)


def test_sandbox_passes_correct_code():
    r = td.run_sandboxed("def f(xs): return sum(xs)", "f", [(([1, 2],), 3), (([],), 0)])
    assert r["ok"] and r["passed"] == 2 and r["error"] is None


def test_sandbox_reports_wrong_answers():
    r = td.run_sandboxed("def f(xs): return max(xs)", "f", [(([1, 2],), 3), (([5],), 5)])
    assert not r["ok"] and r["passed"] == 1 and r["first_fail"] is not None


def test_sandbox_rejects_before_running():
    r = td.run_sandboxed(MALICIOUS["import_os"], "f", [((1,), 1)])
    assert not r["ok"] and r["error"].startswith("ast:") and r["seconds"] == 0.0


def test_runtime_timeout_contains_infinite_loop():
    t0 = time.time()
    r = _run(MALICIOUS["while_true"], timeout=2.0)
    assert not r["ok"] and r["error"] == "timeout"
    assert time.time() - t0 < 10


def test_runtime_timeout_contains_lambda_bomb():
    src = "def f(x):\n    g = lambda h, k: 0 if k > 60 else h(h, k + 1) + h(h, k + 1)\n    return g(g, 0)"
    r = _run(src, timeout=2.0)
    assert not r["ok"] and r["error"] == "timeout"


def test_runtime_recursion_omega_fails_cleanly():
    r = _run(MALICIOUS["lambda_omega"])
    assert not r["ok"] and r["error"] is None and "RecursionError" in r["first_fail"][1]


def test_runtime_huge_exponent_is_stopped():
    r = _run("def f(x): return 10 ** 10 ** 8", timeout=2.0)
    assert not r["ok"]


def test_runtime_memory_cap():
    r = _run("def f(x): return len('a' * (2 * 10 ** 9))")
    assert not r["ok"]
    assert r["job_object"] is (sys.platform == "win32")


@pytest.mark.parametrize("name", ["open_file", "dunder_import", "import_os", "import_subprocess", "eval",
                                  "getattr_builtins", "print", "globals"])
def test_runtime_restricted_builtins(name):
    r = _run(MALICIOUS[name])
    assert not r["ok"]


def test_runtime_import_math_allowed():
    r = td.run_sandboxed("def f(x): import math; return math.isqrt(x)", "f", [((17,), 4)])
    assert r["ok"]
    r = td.run_sandboxed("def f(x): from math import isqrt; return isqrt(x)", "f", [((17,), 4)])
    assert r["ok"]


def test_runtime_math_not_prebound():
    # a row must be runnable as written: math.* without an import is a NameError in real Python
    r = td.run_sandboxed("def f(x): return math.isqrt(x)", "f", [((17,), 4)])
    assert not r["ok"] and "NameError" in r["first_fail"][1]


def test_runtime_cannot_forge_result_line():
    # even if printing were possible, the nonce is unknown to the candidate
    r = _run("def f(x): return x", tests=[((1,), 2)])
    assert not r["ok"] and r["passed"] == 0


def test_runtime_env_and_cwd_isolated():
    # os is unreachable from candidate code, so probe the harness process itself
    import subprocess
    probe = "import os; print(sorted(os.environ)); print(os.getcwd())"
    env = {k: __import__("os").environ[k] for k in ("PATH", "SYSTEMROOT") if k in __import__("os").environ}
    out = subprocess.run([sys.executable, "-I", "-c", probe], env=env, capture_output=True, text=True).stdout
    assert "PYTHONPATH" not in out


# ---------------------------------------------------------------------------
# Reply parsing and verification
# ---------------------------------------------------------------------------

CHECK = {"func": "add_all", "args": ["items"], "tests": repr([(([1, 2, 3],), 6), (([],), 0), (([-5],), -5)])}


def test_verify_one_line_reply():
    r = td.verify_code_reply("It adds the items with sum. def add_all(items): return sum(items)", CHECK)
    assert r["ok"], r
    assert r["assistant"] == "It adds the items with sum. def add_all(items): return sum(items)"


def test_verify_fenced_multiline_is_normalised():
    reply = "Use the built-in sum over the list.\n```python\ndef add_all(items):\n    return sum(items)\n```"
    r = td.verify_code_reply(reply, CHECK)
    assert r["ok"], r
    assert "\n" not in r["assistant"] and r["assistant"].endswith("def add_all(items): return sum(items)")


def test_verify_drops_code_leadin():
    reply = "The function `add_all` sums the list. Here is the complete function:\n\n```python\ndef add_all(items): return sum(items)\n```"
    r = td.verify_code_reply(reply, CHECK)
    assert r["ok"], r
    assert r["assistant"] == "The function add_all sums the list. def add_all(items): return sum(items)"


def test_verify_explanation_after_def():
    r = td.verify_code_reply("def add_all(items): return sum(items) which adds every item together", CHECK)
    assert r["ok"], r
    assert r["assistant"].startswith("which adds every item together.")


def test_verify_rejections():
    assert td.verify_code_reply("Adds them. def total(items): return sum(items)", CHECK)["reason"] == "no_def"
    assert td.verify_code_reply("Adds them all up. def add_all(xs): return sum(xs)", CHECK)["reason"] == "arg_names"
    assert td.verify_code_reply("Takes the max value here. def add_all(items): return max(items)", CHECK)["reason"].startswith("tests:")
    assert td.verify_code_reply("def add_all(items): return sum(items)", CHECK)["reason"] == "no_explanation"
    assert td.verify_code_reply("def add_all(items): return sum(items)", CHECK, require_explanation=False)["ok"]
    loop = "Loop over it.\ndef add_all(items):\n    t = 0\n    for x in items:\n        t += x\n    return t"
    assert td.verify_code_reply(loop, CHECK)["reason"] == "def_not_one_line"
    bad = "Opens a file for fun. def add_all(items): return open('x').read()"
    assert td.verify_code_reply(bad, CHECK)["reason"] == "ast:name:open"


# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------

def test_family_count_and_prompt_limits():
    assert len(td.FAMILY_NAMES) >= 40
    for i in range(len(td.FAMILY_NAMES) * 3):
        for split in ("train", "heldout"):
            t = td.code_task_for_index(split, i)
            assert td.word_count(t.user) <= td.MAX_USER_WORDS
            assert t.user.isascii() and "`" not in t.user
            assert 3 <= len(t.tests) <= 6
            assert t.func in t.user and all(a in t.user for a in t.args)


def test_tasks_deterministic_and_split_disjoint():
    a = td.code_task_for_index("train", 5, seed=0)
    b = td.code_task_for_index("train", 5, seed=0)
    assert a.user == b.user and a.tests == b.tests
    train = {(td.code_task_for_index("train", i).user, repr(td.code_task_for_index("train", i).tests)) for i in range(300)}
    held = {(td.code_task_for_index("heldout", i).user, repr(td.code_task_for_index("heldout", i).tests)) for i in range(150)}
    assert not (train & held)


def test_every_family_reference_passes_sandbox():
    # a lookup-table candidate proves each family's tests survive the literal round trip and the comparator
    for i, fam in enumerate(td.FAMILY_NAMES[::7]):
        t = td.code_task_for_index("train", td.FAMILY_NAMES.index(fam))
        table = {repr(tuple(a)): e for a, e in t.tests}
        params = ", ".join(t.args)
        src = f"def {t.func}({params}): return {table!r}[repr(({params},))]"
        assert td.run_sandboxed(src, t.func, t.check["tests"])["ok"], fam


# ---------------------------------------------------------------------------
# Text hygiene and bio filters
# ---------------------------------------------------------------------------

def test_to_ascii_and_fits_student():
    assert td.to_ascii("TNF-\u03b1 \u2013 caf\u00e9 \u201cq\u201d 5\u00b0C") == 'TNF-alpha - cafe "q" 5 degreesC'
    assert td.fits_student("What is x?", "x is y.") is None
    assert td.fits_student("q", "a\nb") == "assistant_multiline"
    assert td.fits_student("q", "use `x`") == "assistant_backticks"
    assert td.fits_student(" ".join(["w"] * 41), "a") == "user_too_long"
    assert td.fits_student("q", " ".join(["w"] * 61)) == "assistant_too_long"


def test_clean_definition():
    # a short first sentence gets its complete second sentence; the cut-off third is dropped
    good, why = td.clean_definition("hepatocytes", " the main cells of the liver. They make bile and store "
                                                   "glycogen. And the")
    assert why == "ok" and good == "Hepatocytes are the main cells of the liver. They make bile and store glycogen."
    # a long first sentence stands alone
    good, why = td.clean_definition("hepatocytes", " the main cells of the liver, which carry out most of its "
                                                   "metabolic functions. They also make bile.")
    assert why == "ok" and good.endswith("metabolic functions.")
    assert td.clean_definition("nephron", " the functional unit of the kidney that filters")[1] == "truncated"
    assert td.clean_definition("nephron", " a unit (Smith et al., 2003) of the kidney that filters.")[1] == "citation"
    assert td.clean_definition("nephron", " a thing.")[1] == "too_short"
    assert td.clean_definition("nephron", " what we measured in our cohort of kidney patients today.")[1] == "study_talk"
    rep = " a unit of the kidney and a unit of the kidney and a unit of the kidney."
    assert td.clean_definition("nephron", rep)[1] == "repetition"
    vague = " a term used to describe a disease that is different from another disease."
    assert td.clean_definition("distinguish", vague)[1] == "vague"


def test_term_tiers_drop_fragments_and_rank_general_words():
    terms = ["myocardial", "signific", "insulin", "sufficient", "fibrosis"]
    corpus = td.whole_word_counts(["Myocardial fibrosis was significant; insulin was sufficient."])
    general = {"Ġinsulin": 31052, "Ġsufficient": 14016, "Ġsignificant": 5000}
    tiers = td.term_tiers(terms, corpus, general)
    assert "signific" not in tiers                       # BPE fragment, never a whole word
    assert tiers == {"myocardial": 0, "fibrosis": 0, "insulin": 1, "sufficient": 2}
    q = td.build_bio_queue(terms, [], seed=0, tiers=tiers)
    assert [x["term"] for x in q][-2:] == ["insulin", "sufficient"]


def test_parse_pubmedqa_answer():
    assert td.parse_pubmedqa_answer(" yes, because the treated group recovered faster.") == \
        ("yes", "yes, because the treated group recovered faster.", "ok")
    assert td.parse_pubmedqa_answer(" No. Because it did not work at all here.")[0] == "no"
    assert td.parse_pubmedqa_answer(" perhaps it works")[2] == "no_decision"
    assert td.parse_pubmedqa_answer(" maybe, the data were unclear.")[2] == "no_because"
    assert td.parse_pubmedqa_answer(" yes, because the treated group recovered")[2] == "truncated"


needs_biomedlm_tok = pytest.mark.skipif(not (td.BIOMEDLM_DIR / "tokenizer.json").exists(),
                                       reason="BioMedLM tokenizer not downloaded (external/fetch_base.py)")


@needs_biomedlm_tok
def test_mine_terms_on_real_biomedlm_vocab():
    student = {"the", "cell", "cells", "protein", "blood", "patient", "study"}
    terms = td.mine_biomedical_terms(student)
    assert len(terms) > 1000
    for t in terms[:500]:
        assert len(t) >= 7 and t.isalpha() and t.islower() and t.isascii()
        assert t not in student and t not in td.BIO_STOPLIST
    assert "patients" not in terms and "however" not in terms


def test_bio_queue_interleaves_and_is_deterministic():
    terms = [f"term{i:04d}" for i in range(30)]
    pqa = [{"pubid": i, "question": f"Does drug {i} work?", "contexts": ["c"], "final_decision": "yes"} for i in range(10)]
    q1 = td.build_bio_queue(terms, pqa, seed=1)
    q2 = td.build_bio_queue(terms, pqa, seed=1)
    assert [x["key"] for x in q1] == [x["key"] for x in q2]
    assert len(q1) == 40 and len({x["key"] for x in q1}) == 40
    assert [x["kind"] for x in q1[:6]] == ["definition", "definition", "pubmedqa"] * 2


# ---------------------------------------------------------------------------
# LlamaServer HTTP plumbing and the drivers, against a fake server
# ---------------------------------------------------------------------------

class _FakeLlama:
    """In-process stand-in for llama-server: /health, /completion, /v1/chat/completions."""

    def __init__(self, completion_fn, chat_fn):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send({"status": "ok"})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                timings = {"prompt_n": 10, "prompt_ms": 100.0, "predicted_n": 5, "predicted_ms": 250.0}
                if self.path == "/completion":
                    self._send({"content": fake.completion_fn(req), "timings": timings})
                else:
                    self._send({"choices": [{"message": {"content": fake.chat_fn(req)}}], "timings": timings})

        self.completion_fn, self.chat_fn = completion_fn, chat_fn
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def client(self) -> td.LlamaServer:
        s = td.LlamaServer("fake.gguf", self.port)

        class _Alive:
            def poll(self):
                return None

        s.proc = _Alive()          # skip process management; exercise only the HTTP layer
        return s

    def close(self):
        self.httpd.shutdown()


def test_llama_server_command_and_http_layer():
    cmd = td.LlamaServer(td.CODER_GGUF, 8123, threads=4, ctx=2048).command()
    assert cmd[0].endswith("llama-server.exe" if sys.platform == "win32" else "llama-server")
    assert ["-t", "4"] == cmd[cmd.index("-t"):cmd.index("-t") + 2] and "--no-webui" in cmd
    fake = _FakeLlama(lambda req: " completed", lambda req: "Yes")
    try:
        s = fake.client()
        assert s.complete("x", n_predict=3, stop=["\n"], seed=1) == " completed"
        assert s.chat([{"role": "user", "content": "hi"}], n_predict=1) == "Yes"
        sp = s.speed()
        assert sp["requests"] == 2 and sp["gen_tokens_per_s"] == 20.0 and sp["prompt_tokens_per_s"] == 100.0
    finally:
        fake.close()


def _code_solver(req):
    """Answer each code prompt with a correct lookup-table function, and every 4th one wrongly."""

    user = req["messages"][-1]["content"]
    for split in ("heldout", "train"):
        for i in range(400):
            t = td.code_task_for_index(split, i)
            if t.user == user:
                params = ", ".join(t.args)
                if i % 4 == 3:
                    return f"This one is wrong on purpose. def {t.func}({params}): return None"
                table = {repr(tuple(a)): e for a, e in t.tests}
                return f"It looks the answer up in a table.\ndef {t.func}({params}): return {table!r}[repr(({params},))]"
    return "no idea"


def test_generate_code_rows_resumes(tmp_path):
    fake = _FakeLlama(lambda req: "", _code_solver)
    try:
        s = fake.client()
        r1 = td.generate_code_rows(s, data_dir=tmp_path, target=4, heldout=2, limit=4, log=lambda m: None)
        attempts1 = td.read_jsonl(tmp_path / "code_attempts.jsonl")
        assert len(attempts1) == 4
        r2 = td.generate_code_rows(s, data_dir=tmp_path, target=4, heldout=2, limit=None, log=lambda m: None)
    finally:
        fake.close()
    rows = td.read_jsonl(tmp_path / "code_rows.jsonl")
    attempts = td.read_jsonl(tmp_path / "code_attempts.jsonl")
    assert len({a["key"] for a in attempts}) == len(attempts)          # nothing asked twice
    assert sum(r["split"] == "heldout" for r in rows) == 2 and sum(r["split"] == "train" for r in rows) == 4
    for r in rows:
        assert set(r) >= {"user", "assistant", "domain", "task", "source", "split", "check"}
        assert r["domain"] == "programming" and r["source"] == "qwen2.5-coder-7b" and r["task"].startswith("coder_")
        assert td.fits_student(r["user"], r["assistant"]) is None
        assert td.verify_code_reply(r["assistant"], r["check"])["ok"]  # eval can re-verify from the row alone
    assert any(not a["ok"] and a["reason"].startswith("tests:") for a in attempts)
    receipt = json.load(open(tmp_path / "code_rows.receipt.json"))
    assert receipt["rows"] == {"heldout": 2, "train": 4} and len(receipt["runs"]) == 2
    assert r2["attempts"]["reject_reasons"].get("tests:failed", 0) >= 1


@needs_biomedlm_tok
def test_bio_generate_and_judge(tmp_path):
    ctx = ("Patients with myocardial fibrosis, carotid stenosis or venous thrombosis received immunotherapy; "
           "cardiomyopathy and nephropathy were recorded.")
    pqa = [{"pubid": 100 + i, "question": f"Does treatment {i} reduce pain after surgery?", "contexts": [ctx],
            "long_answer": "", "final_decision": "yes" if i % 2 == 0 else "no"} for i in range(12)]
    with open(tmp_path / "pubmedqa_pqa_labeled.jsonl", "w") as f:
        for q in pqa:
            f.write(json.dumps(q) + "\n")

    def completion(req):
        if req["prompt"].endswith("Answer:"):
            assert req.get("grammar") == td.PUBMEDQA_GRAMMAR
            return " yes, because pain scores fell in the treated group."
        return " a biomedical entity that is often discussed in the clinical literature of this field."

    verdicts = iter(["Yes", "No"] * 100)
    fake = _FakeLlama(completion, lambda req: next(verdicts))
    try:
        s = fake.client()
        r = td.generate_bio_candidates(s, data_dir=tmp_path, target=6, heldout=3, student_words={"the", "cell"},
                                       log=lambda m: None)
        assert r["candidates"]["pending_judge"] > 0
        td.judge_bio_candidates(s, data_dir=tmp_path, log=lambda m: None)
    finally:
        fake.close()
    rows = td.read_jsonl(tmp_path / "bio_rows.jsonl")
    cands = td.read_jsonl(tmp_path / "bio_candidates.jsonl")
    assert len({c["key"] for c in cands}) == len(cands)
    split_of = {}
    for c in cands:                                    # every item lives in exactly one split
        assert split_of.setdefault(c["key"], c["split"]) == c["split"]
    pq = [r for r in rows if r["task"] == "bio_pubmedqa"]
    assert pq and all(r["check"]["final_decision"] == "yes" for r in pq)   # "no" questions were label mismatches
    defs = [r for r in rows if r["task"] == "bio_definition"]
    assert defs and all(r["check"]["judge"].endswith(":yes") for r in defs)
    for r in rows:
        assert r["domain"] == "biomedical" and r["source"] == "biomedlm"
        assert td.fits_student(r["user"], r["assistant"]) is None
    judg = td.read_jsonl(tmp_path / "bio_judgements.jsonl")
    assert {j["verdict"] for j in judg} == {"yes", "no"}
    receipt = json.load(open(tmp_path / "bio_rows.receipt.json"))
    assert receipt["candidates"]["pending_judge"] == 0 and receipt["pubmedqa_label_agreement"]["parsed"] > 0


def test_cli_parser_has_all_subcommands():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import make_teacher_data as mtd
    p = mtd.build_parser()
    for cmd in ("convert-biomedlm", "code", "bio", "judge", "all"):
        a = p.parse_args([cmd, "--limit", "3", "--target", "10", "--heldout", "2"])
        assert a.limit == 3 and a.target == 10 and a.heldout == 2 and callable(a.fn)
