"""Teacher corpora for Supermix Expanse: verified code rows and filtered biomedical rows.

Expanse grafts two large teachers into a 47.6M student -- Qwen2.5-Coder-7B-Instruct
(code) and BioMedLM 2.7B (biomedicine). Grafting their layer-0/1 MLPs gives the
student new *capacity*; this module produces the *text* that stage 2 distils
through it. Both teachers run as local GGUF models under ``llama-server`` on the
CPU, so nothing here needs a GPU or a network API.

The one idea the whole module is built around is the same one `answer_check`
is built around in Archimedes: **only checkable text is allowed in**.

* **Code rows are executed, not trusted.** Every task family has our own
  reference implementation and our own test cases; the teacher only ever sees
  the English prompt. Its one-line ``def`` is parsed, passed through an AST
  whitelist, then run against the tests in a separate, isolated Python process
  (``python -I -X utf8 -c HARNESS``, 3 s wall clock, empty environment, a
  256 MB Windows Job Object memory cap, restricted builtins). A row is kept
  only if every test passes. The student therefore never trains on a function
  that is wrong on our tests -- fluency cannot smuggle a bug in.
* **Biomedical rows are cross-examined.** BioMedLM is a *base* LM with no
  instruction tuning, so it is prompted few-shot. PubMedQA answers are kept
  only when their yes/no/maybe agrees with the expert ``final_decision``; term
  definitions are kept only when the *other* teacher (Qwen2.5-Coder, a
  different model family trained on different data) judges the statement
  accurate. Neither filter proves truth, but each removes the teacher's
  unforced errors, and they are independent of each other.

Every row fits the student's house style: user <= 40 words, assistant <= 60
words, one line, ASCII only, no markdown. Every generator is resumable and
append-only: attempts are logged with a stable key (``train:17``,
``term:hepatocytes``), a rerun skips what is already logged, and a crash loses
at most the attempt in flight. Receipts (json) record counts, reject reasons,
pass rates and measured teacher tokens/s.

Held-out rows are item-disjoint from training rows: code held-out prompts come
from a disjoint seed stream per family (``seed|heldout|i`` vs ``seed|train|i``,
plus an exact-prompt collision check); biomedical held-out rows are whole terms
and whole PubMedQA questions that never appear in training.

The sandbox (`check_code_ast`, `run_sandboxed`, `verify_code_reply`) is also the
one `eval_expanse.py` uses to score the student's code, so teacher filtering and
student evaluation apply the same definition of "passes".
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paths (DESIGN.md "Paths" table)
# ---------------------------------------------------------------------------

EXP = Path(__file__).resolve().parents[2]
EXPANSE = EXP / "expanse"
DATA = EXPANSE / "data"
EXTERNAL = EXP / "external"
LLAMA_DIR = EXTERNAL / "llama.cpp"
EXE = ".exe" if os.name == "nt" else ""
LLAMA_SERVER = LLAMA_DIR / f"llama-server{EXE}"
LLAMA_PERPLEXITY = LLAMA_DIR / f"llama-perplexity{EXE}"
LLAMA_SRC = EXTERNAL / "llama.cpp-src"
LLAMA_TAG = "b11115"
TEACHERS = EXTERNAL / "teachers"
CODER_GGUF = TEACHERS / "gguf" / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
BIOMEDLM_DIR = TEACHERS / "biomedlm"
BIOMEDLM_FETCH_RECEIPT = BIOMEDLM_DIR / "fetch.receipt.json"
BIOMEDLM_GGUF = TEACHERS / "gguf" / "biomedlm-q8_0.gguf"
ARCH_CKPT = EXTERNAL / "base" / "supermix_archimedes.pt"
LOGS = EXTERNAL / "logs"

CODER_SOURCE = "qwen2.5-coder-7b"
BIO_SOURCE = "biomedlm"
CODER_PORT = 8091
BIO_PORT = 8092

MAX_USER_WORDS = 40
MAX_ASSISTANT_WORDS = 60


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[dict]:
    """Read a jsonl file; a torn last line (crash mid-write) is skipped, not fatal."""

    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    """Append one row and flush it to disk, so a kill loses nothing already written."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    os.replace(tmp, path)


def sha256_file(path: Path, chunk: int = 16 * 2**20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Text hygiene: the student's house style
# ---------------------------------------------------------------------------

_ASCII_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "―": "-", "−": "-", "‐": "-", "‑": "-",
    "…": "...", "×": "x", "·": ".", "•": "-",
    "°": " degrees", "µ": "u", "μ": "u", "±": "+/-",
    "≤": "<=", "≥": ">=", "≠": "!=", "→": "->", "←": "<-",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "κ": "kappa", "λ": "lambda", "ω": "omega", "Δ": "delta",
    " ": " ", " ": " ", " ": " ", "​": "",
}


def to_ascii(text: str) -> str:
    """Transliterate what has an obvious ASCII spelling, drop the rest.

    NFKD splits accented letters into base + combining mark, so "é" -> "e";
    the explicit map handles typographic punctuation and the Greek letters that
    biomedical text uses as words (alpha-helix, TNF-alpha).
    """

    text = "".join(_ASCII_MAP.get(ch, ch) for ch in text)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_markdown(text: str) -> str:
    """Remove markdown decoration from *prose* (never apply to code: ``*`` is multiplication)."""

    text = text.replace("`", "")
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"(?m)^\s*(#{1,6}|[-*+]|\d+[.)])\s+", "", text)
    return text


def word_count(text: str) -> int:
    return len(text.split())


def fits_student(user: str, assistant: str) -> Optional[str]:
    """Return None when a row fits the student's format, else the reason it does not."""

    for name, text, cap in (("user", user, MAX_USER_WORDS), ("assistant", assistant, MAX_ASSISTANT_WORDS)):
        if not text or not text.strip():
            return f"{name}_empty"
        if "\n" in text or "\r" in text:
            return f"{name}_multiline"
        if not text.isascii():
            return f"{name}_non_ascii"
        if "`" in text:
            return f"{name}_backticks"
        if word_count(text) > cap:
            return f"{name}_too_long"
    return None


# ---------------------------------------------------------------------------
# llama-server
# ---------------------------------------------------------------------------

class LlamaServer:
    """Run ``llama-server`` for one GGUF model as a context manager.

    The process is started without a shell, stdout/stderr go to a log file (a
    pipe nobody reads would eventually block the server), readiness is
    ``/health`` returning 200, and ``__exit__`` always terminates it -- a
    leaked 4.7 GB server would starve the other agents sharing this machine.

    ``complete`` (``/completion``, raw prompt) is for the base-LM teacher;
    ``chat`` (``/v1/chat/completions``, the GGUF's own chat template) is for
    the instruct teacher. Both return the generated text and fold the server's
    own ``timings`` into ``self.stats`` so receipts can report tokens/s.
    """

    def __init__(self, model_path, port: int = CODER_PORT, threads: int = 6, ctx: int = 2048, *,
                 exe: Path = LLAMA_SERVER, host: str = "127.0.0.1", extra_args: Sequence[str] = (),
                 log_path: Optional[Path] = None, startup_timeout: float = 900.0,
                 request_timeout: float = 900.0, parallel: int = 1):
        self.model_path = Path(model_path)
        #: Server slots. ``ctx`` is per slot; the server gets ``ctx * parallel``.
        #: On this CPU decoding is memory-bound, so a batch of slots shares one
        #: pass over the weights and multiplies throughput (see ordered_prefetch).
        self.parallel = max(1, int(parallel))
        self._stats_lock = threading.Lock()
        self.port = int(port)
        self.threads = int(threads)
        self.ctx = int(ctx)
        self.exe = Path(exe)
        self.host = host
        self.extra_args = list(extra_args)
        self.log_path = Path(log_path) if log_path else LOGS / f"llama-server-{self.model_path.stem}-{self.port}.log"
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.proc: Optional[subprocess.Popen] = None
        self._log = None
        self.load_seconds: Optional[float] = None
        self.last_response: Optional[dict] = None
        self.stats = {"requests": 0, "prompt_tokens": 0, "prompt_ms": 0.0,
                      "predicted_tokens": 0, "predicted_ms": 0.0, "wall_s": 0.0}

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def command(self) -> List[str]:
        # -np N with -c ctx*N: N slots of ``ctx`` tokens each (N=1: the whole context
        # belongs to the one request in flight).
        # -cram 0: no host-RAM prompt cache (default 8 GiB) -- RAM is shared.
        # -fit off: keep the ctx we ask for instead of auto-fitting.
        return [str(self.exe), "-m", str(self.model_path), "--host", self.host, "--port", str(self.port),
                "-t", str(self.threads), "-c", str(self.ctx * self.parallel), "-np", str(self.parallel),
                "-cram", "0", "-fit", "off",
                "--no-webui"] + self.extra_args

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _health(self, timeout: float = 2.0) -> Optional[int]:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=timeout) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return None

    def start(self) -> None:
        if not self.exe.exists():
            raise FileNotFoundError(self.exe)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        if self._health() is not None:
            raise RuntimeError(f"port {self.port} already answers /health; another server is running there")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        t0 = time.time()
        self.proc = subprocess.Popen(self.command(), stdout=self._log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, cwd=str(self.exe.parent), creationflags=flags)
        while True:
            code = self.proc.poll()
            if code is not None:
                self._close_log()
                raise RuntimeError(f"llama-server exited with code {code} during startup; see {self.log_path}")
            if self._health() == 200:
                break
            if time.time() - t0 > self.startup_timeout:
                self.stop()
                raise TimeoutError(f"llama-server not healthy after {self.startup_timeout:.0f}s; see {self.log_path}")
            time.sleep(0.5)
        self.load_seconds = time.time() - t0

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=20)
        self.proc = None
        self._close_log()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    # -- requests ----------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("llama-server is not running")
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
            out = json.loads(r.read().decode("utf-8"))
        timings = out.get("timings") or {}
        with self._stats_lock:
            self.stats["wall_s"] += time.time() - t0
            self.stats["requests"] += 1
            self.stats["prompt_tokens"] += int(timings.get("prompt_n", 0) or 0)
            self.stats["prompt_ms"] += float(timings.get("prompt_ms", 0.0) or 0.0)
            self.stats["predicted_tokens"] += int(timings.get("predicted_n", 0) or 0)
            self.stats["predicted_ms"] += float(timings.get("predicted_ms", 0.0) or 0.0)
            self.last_response = out
        return out

    def complete(self, prompt: str, n_predict: int = 64, temperature: float = 0.3,
                 stop: Optional[Sequence[str]] = None, seed: Optional[int] = None, **extra) -> str:
        payload = {"prompt": prompt, "n_predict": int(n_predict), "temperature": float(temperature),
                   "cache_prompt": True}
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = int(seed)
        payload.update(extra)
        return self._post("/completion", payload).get("content", "")

    def chat(self, messages: Sequence[dict], n_predict: int = 128, temperature: float = 0.3,
             stop: Optional[Sequence[str]] = None, seed: Optional[int] = None, **extra) -> str:
        payload = {"messages": list(messages), "max_tokens": int(n_predict), "temperature": float(temperature),
                   "cache_prompt": True}
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = int(seed)
        payload.update(extra)
        out = self._post("/v1/chat/completions", payload)
        try:
            return out["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError):
            return ""

    def speed(self) -> dict:
        """Tokens/s as measured by the server itself (excludes HTTP and Python overhead)."""

        with self._stats_lock:
            s = dict(self.stats)
        s["gen_tokens_per_s"] = round(s["predicted_tokens"] / (s["predicted_ms"] / 1000.0), 3) if s["predicted_ms"] else None
        s["prompt_tokens_per_s"] = round(s["prompt_tokens"] / (s["prompt_ms"] / 1000.0), 3) if s["prompt_ms"] else None
        s["load_seconds"] = round(self.load_seconds, 2) if self.load_seconds is not None else None
        s["model"] = self.model_path.name
        s["threads"] = self.threads
        s["ctx"] = self.ctx
        s["parallel"] = self.parallel
        s["wall_s"] = round(s["wall_s"], 2)
        s["prompt_ms"] = round(s["prompt_ms"], 1)
        s["predicted_ms"] = round(s["predicted_ms"], 1)
        return s


# ---------------------------------------------------------------------------
# Code sandbox, layer 1: AST whitelist
# ---------------------------------------------------------------------------

#: Node types a small pure function may contain. Everything else -- while,
#: try, with, class, global/nonlocal, del, raise, assert, yield, await, async,
#: match -- is refused. ``while`` is refused outright rather than only
#: ``while True``: a one-line ``def`` cannot contain it anyway, and a loop
#: condition that is "never true" is not decidable.
_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Expr, ast.Assign, ast.AugAssign,
    ast.AnnAssign, ast.Pass, ast.If, ast.For, ast.Break, ast.Continue, ast.Import, ast.ImportFrom, ast.alias,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.Attribute, ast.Subscript, ast.Slice, ast.Starred,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.keyword, ast.Lambda,
    ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension, ast.JoinedStr, ast.FormattedValue, ast.NamedExpr,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr,
    ast.BitXor, ast.BitAnd, ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
)

#: Names that reach outside the function's own arithmetic: I/O, dynamic code,
#: introspection, interpreter state, process control, or the builtins module.
FORBIDDEN_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "hasattr", "input", "breakpoint", "help", "exit", "quit", "memoryview", "os", "sys",
    "subprocess", "builtins", "__builtins__", "type", "object", "super", "classmethod", "staticmethod",
    "property", "dir", "print", "iter", "next", "aiter", "anext", "importlib", "shutil", "socket", "ctypes",
    "pathlib", "io", "signal", "threading", "multiprocessing", "pickle", "marshal", "inspect", "gc",
    "copyreg", "code", "codeop", "posix", "nt", "resource", "platform", "__loader__", "__spec__",
    "__file__", "__name__", "__dict__", "__class__", "license", "credits", "copyright",
})

#: Attribute prefixes that walk from an object to its frame, code or globals
#: (generator/coroutine frames, tracebacks, code objects, legacy func_*).
_FORBIDDEN_ATTR_PREFIXES = ("_", "gi_", "f_", "cr_", "ag_", "tb_", "co_", "func_", "im_")
_FORBIDDEN_ATTRS = frozenset({"mro", "format_map"})
#: A str.format field that indexes or walks attributes ("{0.x}", "{0[k]}") can read object internals.
_FORMAT_FIELD_WALK = re.compile(r"\{[^{}!:]*[.\[]")

MAX_INT_CONSTANT = 10**12
MAX_EXPONENT = 1000


def _const_number(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const_number(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    return None


def _contains(node: ast.AST, kinds) -> bool:
    return any(isinstance(n, kinds) for n in ast.walk(node))


def check_code_ast(src: str, func_name: Optional[str] = None, n_args: Optional[int] = None) -> Tuple[bool, str]:
    """Whitelist-check candidate source. Returns (ok, reason); reason is "ok" on success.

    The module must be exactly one plain ``def`` (named ``func_name`` when given,
    no decorators). Inside it only `_ALLOWED_NODES` may appear, no identifier
    may be dunder or in `FORBIDDEN_NAMES`, attributes may not be private or
    frame-walking, the only importable module is ``math``, and three
    resource-bomb shapes are refused statically: exponent towers / huge constant
    exponents, huge integer literals, and self-application (``f(f)``,
    immediately-invoked lambdas) which is how lambda recursion bombs are built.
    The runtime layers (`run_sandboxed`) back every one of these up.
    """

    try:
        tree = ast.parse(src, mode="exec")
    except (SyntaxError, ValueError) as e:
        return False, f"syntax: {type(e).__name__}"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "not_single_def"
    fn = tree.body[0]
    if func_name is not None and fn.name != func_name:
        return False, f"wrong_name:{fn.name}"
    if fn.decorator_list:
        return False, "decorator"
    if n_args is not None:
        a = fn.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or len(a.args) != n_args:
            return False, "wrong_arity"

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False, f"node:{type(node).__name__}"
        # identifiers of every kind
        ident = None
        if isinstance(node, ast.Name):
            ident = node.id
        elif isinstance(node, ast.arg):
            ident = node.arg
        elif isinstance(node, ast.FunctionDef):
            ident = node.name
        elif isinstance(node, ast.keyword):
            ident = node.arg
        elif isinstance(node, ast.alias):
            ident = node.asname or node.name
        if ident is not None and (ident.startswith("__") or ident in FORBIDDEN_NAMES):
            return False, f"name:{ident}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith(_FORBIDDEN_ATTR_PREFIXES) or node.attr in _FORBIDDEN_ATTRS:
                return False, f"attr:{node.attr}"
            if node.attr == "format":
                recv = node.value
                if not (isinstance(recv, ast.Constant) and isinstance(recv.value, str)):
                    return False, "format_receiver"
                if _FORMAT_FIELD_WALK.search(recv.value):
                    return False, "format_field_walk"
        if isinstance(node, ast.Import):
            if any(a.name != "math" for a in node.names):
                return False, "import"
        if isinstance(node, ast.ImportFrom):
            if node.module != "math" or node.level != 0 or any(a.name == "*" for a in node.names):
                return False, "import"
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, str) and "__" in v:
                return False, "dunder_string"
            if isinstance(v, (bytes, bytearray)) and b"__" in v:
                return False, "dunder_string"
            if isinstance(v, int) and not isinstance(v, bool) and abs(v) > MAX_INT_CONSTANT:
                return False, "huge_constant"
            if isinstance(v, complex):
                return False, "complex"
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.LShift)):
            if any(isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Pow, ast.LShift)) for n in ast.walk(node.right)):
                return False, "exponent_tower"
            e = _const_number(node.right)
            if e is not None:
                if abs(e) > MAX_EXPONENT:
                    return False, "huge_exponent"
            elif not _contains(node.right, (ast.Name,)):
                # a constant *expression* exponent (e.g. 10*10*10*10) is never needed; refuse rather than evaluate it
                return False, "constant_expression_exponent"
        if isinstance(node, ast.Call):
            if isinstance(node.func, (ast.Lambda, ast.Call)):
                return False, "self_application"
            if isinstance(node.func, ast.Name):
                callee = node.func.id
                for a in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(a, ast.Name) and a.id == callee:
                        return False, "self_application"
                if callee == "pow" and len(node.args) >= 2:
                    e = _const_number(node.args[1])
                    if (e is not None and abs(e) > MAX_EXPONENT) or _contains(node.args[1], (ast.BinOp,)):
                        return False, "huge_exponent"
    return True, "ok"


# ---------------------------------------------------------------------------
# Code sandbox, layers 2-4: isolated subprocess, restricted builtins, job object
# ---------------------------------------------------------------------------

#: Runs inside ``python -I -X utf8 -c``. Reads a Python literal from stdin
#: (``ast.literal_eval`` -- never ``eval``), execs the candidate with a
#: restricted builtins dict whose ``__import__`` can only return ``math``, and
#: prints one result line prefixed by a nonce the candidate cannot see.
SANDBOX_HARNESS = r'''
import ast, json, math, sys
def _main():
    data = ast.literal_eval(sys.stdin.read())
    nonce = data["nonce"]
    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "math" and level == 0:
            return math
        raise ImportError("only math may be imported")
    import builtins as _b
    safe = {k: getattr(_b, k) for k in (
        "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "frozenset",
        "int", "isinstance", "len", "list", "map", "max", "min", "ord", "pow", "range", "reversed", "round",
        "set", "sorted", "str", "sum", "tuple", "zip", "bin", "hex", "oct", "hash", "slice", "repr", "format",
        "ascii", "callable", "bytes", "True", "False", "None", "Exception", "ValueError", "TypeError",
        "IndexError", "KeyError", "ZeroDivisionError", "ArithmeticError", "StopIteration")}
    safe["__import__"] = _import
    # math is importable but NOT pre-bound: a row must be runnable Python as written
    ns = {"__builtins__": safe, "__name__": "candidate"}
    out = {"passed": 0, "total": len(data["tests"]), "error": None, "first_fail": None}
    try:
        exec(compile(data["src"], "<candidate>", "exec"), ns)
        fn = ns[data["func"]]
    except BaseException as e:
        out["error"] = "load:" + type(e).__name__
        sys.stdout.write(nonce + json.dumps(out) + "\n")
        return
    def _norm(x):
        if isinstance(x, (list, tuple)):
            return [_norm(v) for v in x]
        if isinstance(x, dict):
            return {k: _norm(v) for k, v in x.items()}
        return x
    def _eq(got, exp):
        if isinstance(exp, bool) or isinstance(got, bool):
            return type(got) is type(exp) and got == exp
        if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
            return math.isclose(got, exp, rel_tol=1e-6, abs_tol=1e-9)
        if isinstance(exp, (list, tuple)) and isinstance(got, (list, tuple)):
            return len(got) == len(exp) and all(_eq(g, e) for g, e in zip(got, exp))
        if isinstance(exp, dict) and isinstance(got, dict):
            return set(got) == set(exp) and all(_eq(got[k], exp[k]) for k in exp)
        return type(got) is type(exp) and _norm(got) == _norm(exp)
    for args, expected in data["tests"]:
        try:
            got = fn(*args)
            ok = _eq(got, expected)
        except BaseException as e:
            got, ok = "raised " + type(e).__name__, False
        if ok:
            out["passed"] += 1
        elif out["first_fail"] is None:
            out["first_fail"] = (repr(args)[:120], repr(got)[:120], repr(expected)[:120])
    sys.stdout.write(nonce + json.dumps(out) + "\n")
_main()
'''

SANDBOX_TIMEOUT_S = 3.0
SANDBOX_MEMORY_BYTES = 256 * 2**20


def _job_object_limit(proc: subprocess.Popen, memory_bytes: int):
    """Put `proc` in a Windows Job Object: memory cap, no child processes, killed with the job.

    Returns the job handle (keep it alive until the process ends, then close it)
    or None when unavailable. There is a sub-millisecond window between process
    creation and assignment; the interpreter's own startup is far longer than
    that, and the AST layer has already run, so candidate code never executes
    unconfined in practice.
    """

    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = EXTENDED()
        # PROCESS_MEMORY 0x100 | ACTIVE_PROCESS 0x8 | KILL_ON_JOB_CLOSE 0x2000 | DIE_ON_UNHANDLED_EXCEPTION 0x400
        info.BasicLimitInformation.LimitFlags = 0x100 | 0x8 | 0x2000 | 0x400
        info.BasicLimitInformation.ActiveProcessLimit = 1
        info.ProcessMemoryLimit = int(memory_bytes)
        ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        ok = ok and k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))
        if not ok:
            k32.CloseHandle(wintypes.HANDLE(job))
            return None
        return job
    except Exception:
        return None


def _close_handle(handle) -> None:
    if handle:
        import ctypes
        from ctypes import wintypes
        ctypes.WinDLL("kernel32").CloseHandle(wintypes.HANDLE(handle))


def run_sandboxed(src: str, func_name: str, tests, timeout: float = SANDBOX_TIMEOUT_S,
                  memory_bytes: int = SANDBOX_MEMORY_BYTES, check_ast: bool = True) -> dict:
    """Execute `src` and call `func_name(*args)` for every ``(args, expected)`` in `tests`.

    `tests` is a list or its ``repr`` string. Returns ``{"ok", "passed", "total",
    "error", "first_fail", "seconds", "job_object"}``; ``ok`` means every test
    passed. ``check_ast=False`` exists only so the tests can prove the runtime
    layers hold on their own; production callers never pass it.
    """

    if isinstance(tests, str):
        tests = ast.literal_eval(tests)
    tests = [(tuple(a), e) for a, e in tests]
    if check_ast:
        ok, reason = check_code_ast(src, func_name)
        if not ok:
            return {"ok": False, "passed": 0, "total": len(tests), "error": "ast:" + reason, "first_fail": None,
                    "seconds": 0.0, "job_object": None}
    nonce = "@@" + secrets.token_hex(8) + "@@"
    payload = repr({"src": src, "func": func_name, "tests": tests, "nonce": nonce})
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT") if k in os.environ}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="expanse_sbx_") as cwd:
        proc = subprocess.Popen([sys.executable, "-I", "-X", "utf8", "-c", SANDBOX_HARNESS],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=cwd, env=env, creationflags=flags)
        job = _job_object_limit(proc, memory_bytes)
        try:
            out, _err = proc.communicate(payload.encode("utf-8"), timeout=timeout)
            error = None
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _err = proc.communicate()
            error = "timeout"
        finally:
            _close_handle(job)
    seconds = round(time.time() - t0, 3)
    result = {"ok": False, "passed": 0, "total": len(tests), "error": error, "first_fail": None,
              "seconds": seconds, "job_object": job is not None}
    if error:
        return result
    text = out.decode("utf-8", "replace")
    lines = [ln for ln in text.splitlines() if ln.startswith(nonce)]
    if len(lines) != 1:
        result["error"] = f"no_result(rc={proc.returncode})"
        return result
    got = json.loads(lines[0][len(nonce):])
    result.update(passed=got["passed"], error=got["error"], first_fail=got["first_fail"])
    result["ok"] = got["error"] is None and got["passed"] == len(tests) and len(tests) > 0
    return result


# ---------------------------------------------------------------------------
# Reply parsing and verification
# ---------------------------------------------------------------------------

_LEADING_LABEL = re.compile(r"^(explanation|answer|solution|here is|here's)\s*[:\-]\s*", re.I)
#: "... Here is the complete function:" style lead-ins to the code, dropped from the explanation.
_TRAILING_LEADIN = re.compile(r"(^|(?<=[.!?]))\s*(here is|here's|below is)[^.!?]*[:.]?\s*$", re.I)


def _one_line_def(code: str, func_name: str) -> Optional[Tuple[str, str]]:
    """Return (the def as ONE line, the text after it), or None when that is impossible.

    A reply that already has the def on one line keeps the teacher's own text.
    A def whose body the teacher broke onto indented lines is rebuilt as
    ``def f(args): stmt; stmt`` from the AST, which is only possible when every
    body statement is a simple statement.
    """

    lines = code.split("\n")
    first = lines[0].strip().rstrip(";")
    try:
        tree = ast.parse(first)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
            return first, "\n".join(lines[1:])
    except SyntaxError:
        pass
    block = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() == "" or ln.startswith((" ", "\t")):
            block.append(ln)
        else:
            break
    if len(block) == 1:
        return None
    try:
        tree = ast.parse("\n".join(block).strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    fn = tree.body[0]
    simple = (ast.Return, ast.Assign, ast.AugAssign, ast.Expr, ast.Pass, ast.Import, ast.ImportFrom)
    if fn.name != func_name or not all(isinstance(s, simple) for s in fn.body) or fn.decorator_list:
        return None
    line = f"def {fn.name}({ast.unparse(fn.args)}): " + "; ".join(ast.unparse(s) for s in fn.body)
    return line, "\n".join(lines[len(block):])


def _longest_def_prefix(line: str) -> Optional[Tuple[str, str]]:
    """``"def f(x): return x + 1 which adds one"`` -> ("def f(x): return x + 1", "which adds one").

    The longest space-delimited prefix that parses as exactly one def wins, so a
    longer valid expression is never cut short.
    """

    cuts = [i for i, ch in enumerate(line) if ch == " "]
    for i in reversed(cuts):
        head = line[:i].rstrip().rstrip(";")
        try:
            tree = ast.parse(head)
        except SyntaxError:
            continue
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
            return head, line[i:].strip()
    return None


def _clean_explanation(text: str) -> str:
    text = one_line(strip_markdown(text))
    text = _LEADING_LABEL.sub("", text).strip()
    text = _TRAILING_LEADIN.sub("", text).strip()
    text = re.sub(r"^[\s:;,.\-]+|[\s:;,\-]+$", "", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def extract_function(reply: str, func_name: str) -> Tuple[str, str]:
    """Split a teacher/student reply into (explanation, one-line def). Raises ValueError(reason).

    Tolerated deviations from the requested format, all normalised to
    "explanation def": code fences/backticks, the def on its own line, a body
    broken onto indented simple-statement lines, and the explanation placed
    *after* the def instead of before it.
    """

    text = reply.replace("\r", "")
    text = re.sub(r"```[A-Za-z0-9_+-]*", "\n", text).replace("`", "")
    m = re.search(rf"\bdef\s+{re.escape(func_name)}\s*\(", text)
    if not m:
        raise ValueError("no_def")
    before = text[:m.start()]
    code = text[m.start():]
    found = _one_line_def(code, func_name)
    if found is None:
        first, _, rest = code.partition("\n")
        cut = _longest_def_prefix(first.strip())
        if cut is None:
            raise ValueError("def_not_one_line")
        line, after = cut[0], one_line(cut[1] + " " + rest)
    else:
        line, after = found[0], one_line(found[1])
    explanation = _clean_explanation(before)
    if word_count(explanation) < 3 and word_count(after) >= 3:
        explanation = _clean_explanation(after)
    return explanation, line


def verify_code_reply(reply: str, check: dict, require_explanation: bool = True,
                      timeout: float = SANDBOX_TIMEOUT_S) -> dict:
    """Parse, whitelist and execute a reply against ``check = {"func", "tests", "args"?}``.

    Returns ``{"ok", "reason", "assistant", "passed", "total", ...}``. ``assistant``
    is the normalised one-line reply (explanation + def) when parsing succeeded.
    `eval_expanse.py` calls this with ``require_explanation=False`` on student output.
    """

    func = check["func"]
    tests = check["tests"]
    n_tests = len(ast.literal_eval(tests)) if isinstance(tests, str) else len(tests)
    res = {"ok": False, "reason": None, "assistant": None, "passed": 0, "total": n_tests}
    try:
        explanation, line = extract_function(reply, func)
    except ValueError as e:
        res["reason"] = str(e)
        return res
    line = to_ascii(line)
    explanation = to_ascii(explanation)
    assistant = one_line(f"{explanation} {line}")
    res["assistant"] = assistant
    if require_explanation and word_count(explanation) < 3:
        res["reason"] = "no_explanation"
        return res
    arity = len(check["args"]) if check.get("args") else None
    ok, why = check_code_ast(line, func, n_args=arity)
    if not ok:
        res["reason"] = "ast:" + why
        return res
    if check.get("args"):
        got_args = [a.arg for a in ast.parse(line).body[0].args.args]
        if got_args != list(check["args"]):
            res["reason"] = "arg_names"
            return res
    run = run_sandboxed(line, func, tests, timeout=timeout)
    res.update(passed=run["passed"], total=run["total"], sandbox_error=run["error"],
               first_fail=run["first_fail"], seconds=run["seconds"])
    if not run["ok"]:
        res["reason"] = "tests:" + (run["error"] or f"{run['passed']}/{run['total']}")
        return res
    res["ok"] = True
    res["reason"] = "ok"
    return res


# ---------------------------------------------------------------------------
# Code task families: our prompts, our reference implementations, our tests
# ---------------------------------------------------------------------------

WORDS = ("apple river stone cloud tiger lemon piano rocket garden silver window candle orange planet forest "
         "yellow bridge castle dragon eagle falcon guitar hammer island jacket kitten ladder magnet needle ocean "
         "pencil quartz rabbit saddle tomato umbrella violet walnut zebra anchor bottle cactus meadow pepper "
         "summer winter spring autumn copper basket pillow mirror").split()
PALINDROMES = ("level", "radar", "civic", "rotor", "kayak", "refer", "noon", "madam", "Racecar", "Stats",
               "Anna", "deified", "wow", "a")

NUM_ARGS = ("nums", "values", "xs", "numbers", "items", "data", "arr", "lst", "seq", "vals")
STR_ARGS = ("s", "text", "word", "string", "phrase", "t", "line", "msg")
SENT_ARGS = ("sentence", "text", "s", "phrase", "line", "words")
INT_ARGS = ("n", "k", "x", "num", "m", "value")


@dataclass
class CodeFamily:
    """One task family: English spec + our reference function + our input generator.

    ``desc`` is a third-person clause ("returns the sum of ...") with ``{a}``,
    ``{b}``, ``{c}`` for argument names and ``{k}`` for a family constant drawn
    by ``params``; ``gen(rng, p)`` draws one argument tuple and ``ref(p, *args)``
    computes the expected value. ``edge`` are argument tuples always tested.
    """

    name: str
    fns: Tuple[str, ...]
    args: Tuple[Tuple[str, ...], ...]
    desc: str
    gen: Callable[[random.Random, dict], tuple]
    ref: Callable[..., Any]
    params: Optional[Callable[[random.Random], dict]] = None
    edge: Tuple[tuple, ...] = ()


FAMILIES: Dict[str, CodeFamily] = {}


def _fam(name, fns, args, desc, gen, ref, params=None, edge=()):
    FAMILIES[name] = CodeFamily(name, tuple(fns), tuple(tuple(a) for a in args), desc, gen, ref, params, tuple(edge))


def _ints(rng, lo_n=0, hi_n=8, lo=-20, hi=20):
    return [rng.randint(lo, hi) for _ in range(rng.randint(lo_n, hi_n))]


def _sentence(rng, lo=1, hi=6):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi)))


def _word(rng):
    w = rng.choice(WORDS)
    return w if rng.random() < 0.7 else w.capitalize()


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _first_unique(s):
    c = Counter(s)
    return next((ch for ch in s if c[ch] == 1), None)


def _is_prime(n):
    return n >= 2 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# lists of numbers
_fam("sum_list", ["sum_list", "list_sum", "add_all", "total_of", "sum_values"], [NUM_ARGS],
     "returns the sum of the numbers in the list {a}", lambda r, p: (_ints(r),), lambda p, a: sum(a), edge=[([],)])
_fam("product_list", ["product", "product_of", "multiply_all", "list_product"], [NUM_ARGS],
     "returns the product of the numbers in the list {a}, or 1 for an empty list",
     lambda r, p: (_ints(r, 0, 6, -9, 9),), lambda p, a: math.prod(a), edge=[([],)])
_fam("average", ["average", "mean", "avg", "mean_of"], [NUM_ARGS],
     "returns the average (arithmetic mean) of the non-empty list of numbers {a}",
     lambda r, p: (_ints(r, 1, 8),), lambda p, a: sum(a) / len(a))
_fam("max_of_list", ["largest", "max_value", "biggest", "find_max"], [NUM_ARGS],
     "returns the largest number in the non-empty list {a}", lambda r, p: (_ints(r, 1, 8),), lambda p, a: max(a))
_fam("min_of_list", ["smallest", "min_value", "find_min", "lowest"], [NUM_ARGS],
     "returns the smallest number in the non-empty list {a}", lambda r, p: (_ints(r, 1, 8),), lambda p, a: min(a))
_fam("list_range", ["spread", "value_range", "max_minus_min", "range_of"], [NUM_ARGS],
     "returns the difference between the largest and the smallest number in the non-empty list {a}",
     lambda r, p: (_ints(r, 1, 8),), lambda p, a: max(a) - min(a))
_fam("count_evens", ["count_evens", "num_even", "even_count", "how_many_even"], [NUM_ARGS],
     "returns how many even numbers are in the list {a}", lambda r, p: (_ints(r),),
     lambda p, a: sum(1 for x in a if x % 2 == 0), edge=[([],)])
_fam("sum_evens", ["sum_evens", "even_sum", "add_evens", "total_even"], [NUM_ARGS],
     "returns the sum of the even numbers in the list {a}", lambda r, p: (_ints(r),),
     lambda p, a: sum(x for x in a if x % 2 == 0), edge=[([],)])
_fam("filter_evens", ["only_evens", "keep_even", "evens", "filter_even"], [NUM_ARGS],
     "returns a new list with only the even numbers from {a}, in their original order",
     lambda r, p: (_ints(r),), lambda p, a: [x for x in a if x % 2 == 0], edge=[([],)])
_fam("filter_positive", ["positives", "keep_positive", "only_positive", "drop_non_positive"], [NUM_ARGS],
     "returns a new list containing only the numbers in {a} that are greater than zero, in order",
     lambda r, p: (_ints(r),), lambda p, a: [x for x in a if x > 0], edge=[([0, -1],)])
_fam("count_greater", ["count_above", "count_greater", "num_above", "how_many_above"], [NUM_ARGS, ("t", "limit", "threshold", "cutoff")],
     "returns how many numbers in the list {a} are strictly greater than {b}",
     lambda r, p: (_ints(r), r.randint(-10, 10)), lambda p, a, t: sum(1 for x in a if x > t), edge=[([], 0)])
_fam("above_k", ["above", "greater_than", "keep_big", "filter_above"], [NUM_ARGS],
     "returns a list of the numbers in {a} that are greater than {k}, keeping their order",
     lambda r, p: (_ints(r),), lambda p, a: [x for x in a if x > p["k"]], params=lambda r: {"k": r.randint(-5, 10)})
_fam("scale_list", ["scale", "multiply_each", "times_k", "scaled"], [NUM_ARGS],
     "returns a new list with every number in {a} multiplied by {k}",
     lambda r, p: (_ints(r),), lambda p, a: [x * p["k"] for x in a], params=lambda r: {"k": r.randint(2, 9)}, edge=[([],)])
_fam("add_to_each", ["add_k", "shift", "offset_all", "plus_k"], [NUM_ARGS],
     "returns a new list where {k} is added to every number in {a}",
     lambda r, p: (_ints(r),), lambda p, a: [x + p["k"] for x in a], params=lambda r: {"k": r.randint(1, 20)}, edge=[([],)])
_fam("square_each", ["squares", "square_all", "squared", "square_list"], [NUM_ARGS],
     "returns a list of the squares of the numbers in {a}", lambda r, p: (_ints(r),), lambda p, a: [x * x for x in a])
_fam("abs_each", ["absolutes", "abs_all", "magnitudes", "abs_list"], [NUM_ARGS],
     "returns a list of the absolute values of the numbers in {a}", lambda r, p: (_ints(r),), lambda p, a: [abs(x) for x in a])
_fam("running_total", ["running_total", "cumulative", "prefix_sums", "cumsum"], [NUM_ARGS],
     "returns the running totals of the list {a}, so [1, 2, 3] gives [1, 3, 6]",
     lambda r, p: (_ints(r),), lambda p, a: [sum(a[:i + 1]) for i in range(len(a))], edge=[([],)])
_fam("dedupe", ["dedupe", "unique", "remove_duplicates", "distinct"], [NUM_ARGS],
     "returns the items of the list {a} with duplicates removed, keeping the order of first occurrence",
     lambda r, p: (_ints(r, 0, 10, 0, 6),), lambda p, a: list(dict.fromkeys(a)), edge=[([],)])
_fam("second_largest", ["second_largest", "runner_up", "second_max", "next_biggest"], [NUM_ARGS],
     "returns the second largest distinct value in the list {a}, which has at least two distinct values",
     lambda r, p: (lambda xs: (xs + [max(xs)] * r.randint(0, 2),))(r.sample(range(-20, 40), r.randint(2, 7))),
     lambda p, a: sorted(set(a))[-2], edge=[([5, 5, 3],)])
_fam("index_of_max", ["argmax", "index_of_max", "max_index", "where_max"], [NUM_ARGS],
     "returns the index of the first occurrence of the largest value in the non-empty list {a}",
     lambda r, p: (_ints(r, 1, 8, 0, 9),), lambda p, a: a.index(max(a)))
_fam("is_sorted", ["is_sorted", "in_order", "ascending", "sorted_check"], [NUM_ARGS],
     "returns True if the list {a} is sorted in non-decreasing order and False otherwise",
     lambda r, p: ((sorted(_ints(r)) if r.random() < 0.5 else _ints(r)),),
     lambda p, a: all(a[i] <= a[i + 1] for i in range(len(a) - 1)), edge=[([],), ([2, 2, 3],)])
_fam("flatten", ["flatten", "flatten_once", "join_lists", "merge_sublists"], [("lists", "nested", "groups", "rows")],
     "returns a single list made by joining the inner lists of the list of lists {a} in order",
     lambda r, p: ([_ints(r, 0, 3, 0, 9) for _ in range(r.randint(0, 4))],), lambda p, a: [x for s in a for x in s],
     edge=[([],), ([[1], [], [2, 3]],)])
_fam("dot_product", ["dot", "dot_product", "inner", "sum_products"], [("a", "u", "xs", "v1"), ("b", "v", "ys", "v2")],
     "returns the dot product of the equal-length number lists {a} and {b}",
     lambda r, p: (lambda n: (_ints(r, n, n, -9, 9), _ints(r, n, n, -9, 9)))(r.randint(0, 5)),
     lambda p, a, b: sum(x * y for x, y in zip(a, b)))
_fam("common_sorted", ["common", "shared", "intersection", "in_both"], [("a", "xs", "first", "left"), ("b", "ys", "second", "right")],
     "returns a sorted list of the distinct values that appear in both lists {a} and {b}",
     lambda r, p: (_ints(r, 0, 7, 0, 9), _ints(r, 0, 7, 0, 9)), lambda p, a, b: sorted(set(a) & set(b)))
_fam("merge_sorted", ["merge", "merge_sorted", "combine_sorted", "merged"], [("a", "xs", "first", "left"), ("b", "ys", "second", "right")],
     "returns one sorted list containing all the elements of the sorted lists {a} and {b}",
     lambda r, p: (sorted(_ints(r, 0, 5)), sorted(_ints(r, 0, 5))), lambda p, a, b: sorted(a + b))
_fam("pairwise_diff", ["diffs", "differences", "steps", "deltas"], [NUM_ARGS],
     "returns a list of the differences between consecutive numbers in {a}, each later value minus the one before",
     lambda r, p: (_ints(r, 1, 7),), lambda p, a: [a[i + 1] - a[i] for i in range(len(a) - 1)])
_fam("last_k", ["last_items", "tail", "last_few", "take_last"], [NUM_ARGS],
     "returns the last {k} items of the list {a}, or the whole list if it is shorter than that",
     lambda r, p: (_ints(r, 0, 8),), lambda p, a: a[-p["k"]:], params=lambda r: {"k": r.randint(2, 4)}, edge=[([1],)])
_fam("every_other", ["every_other", "even_positions", "alternate", "skip_one"], [NUM_ARGS],
     "returns every other element of the list {a}, starting with the first one",
     lambda r, p: (_ints(r, 0, 9),), lambda p, a: a[::2])
_fam("rotate_left", ["rotate_left", "rotate", "shift_left", "cycle_once"], [NUM_ARGS],
     "returns the list {a} rotated left by one position, so the first item moves to the end; an empty list stays empty",
     lambda r, p: (_ints(r, 1, 7),), lambda p, a: a[1:] + a[:1], edge=[([],)])
_fam("min_max", ["min_max", "extremes", "bounds", "low_high"], [NUM_ARGS],
     "returns a tuple (smallest, largest) for the non-empty list of numbers {a}",
     lambda r, p: (_ints(r, 1, 8),), lambda p, a: (min(a), max(a)))
_fam("median", ["median", "middle_value", "find_median", "med"], [NUM_ARGS],
     "returns the median of the non-empty list of numbers {a}, averaging the two middle values when the length is even",
     lambda r, p: (_ints(r, 1, 8),), lambda p, a: _median(a))
_fam("zip_dict", ["to_dict", "pair_up", "make_map", "zip_dict"], [("keys", "names", "ks", "labels"), ("values", "vals", "vs", "items")],
     "returns a dictionary that maps each item of {a} to the item at the same position in {b}",
     lambda r, p: (lambda n: (r.sample(WORDS, n), _ints(r, n, n, 0, 50)))(r.randint(0, 4)),
     lambda p, a, b: dict(zip(a, b)))
# strings
_fam("count_vowels", ["count_vowels", "vowel_count", "num_vowels", "vowels_in"], [STR_ARGS],
     "returns the number of vowels (a, e, i, o, u, in either case) in the string {a}",
     lambda r, p: (_sentence(r, 1, 3) if r.random() < 0.7 else _sentence(r, 1, 3).upper(),),
     lambda p, a: sum(1 for c in a.lower() if c in "aeiou"), edge=[("",), ("rhythm",)])
_fam("reverse_string", ["reverse", "reversed_text", "backwards", "flip"], [STR_ARGS],
     "returns the string {a} reversed", lambda r, p: (_sentence(r, 1, 3),), lambda p, a: a[::-1], edge=[("",)])
_fam("reverse_words", ["reverse_words", "flip_words", "words_backwards", "reverse_order"], [SENT_ARGS],
     "returns the words of the sentence {a} in reverse order, joined by single spaces",
     lambda r, p: (_sentence(r, 1, 6),), lambda p, a: " ".join(a.split()[::-1]))
_fam("is_palindrome", ["is_palindrome", "palindrome", "reads_same", "is_pal"], [STR_ARGS],
     "returns True if the string {a} reads the same forwards and backwards ignoring case, and False otherwise",
     lambda r, p: (r.choice(PALINDROMES) if r.random() < 0.5 else _word(r),),
     lambda p, a: a.lower() == a.lower()[::-1], edge=[("Noon",)])
_fam("word_lengths", ["word_lengths", "lengths", "len_of_words", "word_sizes"], [SENT_ARGS],
     "returns a list with the length of each word in the sentence {a}, where words are separated by spaces",
     lambda r, p: (_sentence(r, 1, 6),), lambda p, a: [len(w) for w in a.split()])
_fam("count_words", ["count_words", "word_count", "num_words", "how_many_words"], [SENT_ARGS],
     "returns the number of words in the string {a}, splitting on whitespace",
     lambda r, p: (_sentence(r, 0, 7),), lambda p, a: len(a.split()), edge=[("",), ("  two   words ",)])
_fam("capitalize_words", ["capitalize_words", "title_words", "cap_each", "upper_first"], [SENT_ARGS],
     "returns the sentence {a} with the first letter of every word in upper case",
     lambda r, p: (_sentence(r, 1, 5),), lambda p, a: " ".join(w.capitalize() for w in a.split(" ")))
_fam("count_char", ["count_char", "occurrences", "char_count", "times_seen"], [STR_ARGS, ("c", "ch", "char", "letter")],
     "returns how many times the character {b} appears in the string {a}",
     lambda r, p: (lambda s: (s, r.choice(s) if s and r.random() < 0.8 else "z"))(_sentence(r, 1, 3)),
     lambda p, a, c: a.count(c))
_fam("remove_char", ["remove_char", "strip_char", "without", "delete_char"], [STR_ARGS, ("c", "ch", "char", "letter")],
     "returns the string {a} with every occurrence of the character {b} removed",
     lambda r, p: (lambda s: (s, r.choice(s) if s else "a"))(_sentence(r, 1, 3)), lambda p, a, c: a.replace(c, ""))
_fam("remove_vowels", ["remove_vowels", "disemvowel", "no_vowels", "strip_vowels"], [STR_ARGS],
     "returns the string {a} with all vowels (a, e, i, o, u, in either case) removed",
     lambda r, p: (_sentence(r, 1, 3).title() if r.random() < 0.3 else _sentence(r, 1, 3),),
     lambda p, a: "".join(c for c in a if c.lower() not in "aeiou"))
_fam("longest_word", ["longest_word", "longest", "biggest_word", "max_word"], [SENT_ARGS],
     "returns the longest word in the sentence {a}, choosing the first one if there is a tie",
     lambda r, p: (_sentence(r, 1, 6),), lambda p, a: max(a.split(), key=len))
_fam("acronym", ["acronym", "initials", "abbreviate", "first_letters"], [SENT_ARGS],
     "returns the acronym of the phrase {a}: the first letter of each word, upper-cased and joined together",
     lambda r, p: (_sentence(r, 1, 5),), lambda p, a: "".join(w[0].upper() for w in a.split()))
_fam("is_anagram", ["is_anagram", "anagrams", "same_letters", "anagram_check"], [("a", "s1", "first", "w1"), ("b", "s2", "second", "w2")],
     "returns True if the strings {a} and {b} are anagrams of each other ignoring case, and False otherwise",
     lambda r, p: (lambda w: (w, ("".join(r.sample(w, len(w))) if r.random() < 0.5 else r.choice(WORDS))))(r.choice(WORDS)),
     lambda p, a, b: sorted(a.lower()) == sorted(b.lower()), edge=[("Listen", "silent")])
_fam("char_frequency", ["char_freq", "letter_counts", "frequencies", "count_chars"], [STR_ARGS],
     "returns a dictionary mapping each character of the string {a} to the number of times it occurs",
     lambda r, p: (r.choice(WORDS),), lambda p, a: dict(Counter(a)), edge=[("",)])
_fam("count_upper", ["count_upper", "num_capitals", "upper_count", "capitals"], [STR_ARGS],
     "returns the number of upper-case letters in the string {a}",
     lambda r, p: ("".join(c.upper() if r.random() < 0.3 else c for c in _sentence(r, 1, 3)),),
     lambda p, a: sum(1 for c in a if c.isupper()))
_fam("repeat_string", ["repeat", "repeat_text", "times", "echo"], [STR_ARGS, ("n", "k", "count", "times")],
     "returns the string {a} repeated {b} times with nothing in between",
     lambda r, p: (r.choice(WORDS)[: r.randint(1, 4)], r.randint(0, 4)), lambda p, a, n: a * n)
_fam("first_unique_char", ["first_unique", "first_single", "non_repeating", "lonely_char"], [STR_ARGS],
     "returns the first character of the string {a} that occurs exactly once in it, or None if there is none",
     lambda r, p: (r.choice(WORDS) if r.random() < 0.7 else "aabb",), lambda p, a: _first_unique(a), edge=[("",)])
# integers and formulas
_fam("is_prime", ["is_prime", "prime", "check_prime", "is_prime_number"], [INT_ARGS],
     "returns True if the integer {a} is a prime number and False otherwise",
     lambda r, p: (r.randint(-3, 200),), lambda p, n: _is_prime(n), edge=[(0,), (1,), (2,), (97,)])
_fam("gcd", ["gcd", "greatest_common_divisor", "hcf", "common_divisor"], [("a", "x", "m", "p"), ("b", "y", "n", "q")],
     "returns the greatest common divisor of the non-negative integers {a} and {b}, which are not both zero",
     lambda r, p: (r.randint(0, 120), r.randint(1, 120)), lambda p, a, b: math.gcd(a, b), edge=[(0, 7)])
_fam("fibonacci", ["fib", "fibonacci", "nth_fib", "fibo"], [INT_ARGS],
     "returns the {a}-th Fibonacci number, where the 0th is 0 and the 1st is 1",
     lambda r, p: (r.randint(0, 20),), lambda p, n: _fib(n), edge=[(0,), (1,), (2,)])
_fam("factorial", ["factorial", "fact", "n_factorial", "product_up_to"], [INT_ARGS],
     "returns the factorial of the non-negative integer {a}", lambda r, p: (r.randint(0, 12),),
     lambda p, n: math.factorial(n), edge=[(0,), (1,)])
_fam("digit_sum", ["digit_sum", "sum_digits", "add_digits", "digits_total"], [INT_ARGS],
     "returns the sum of the decimal digits of the non-negative integer {a}",
     lambda r, p: (r.randint(0, 99999),), lambda p, n: sum(int(d) for d in str(n)), edge=[(0,)])
_fam("clamp", ["clamp", "limit", "bound", "clip"], [("x", "value", "v", "n"), ("lo", "low", "minimum", "floor"), ("hi", "high", "maximum", "ceiling")],
     "returns {a} limited to the range from {b} to {c} inclusive, where {b} is not greater than {c}",
     lambda r, p: (lambda lo: (r.randint(-30, 30), lo, lo + r.randint(0, 20)))(r.randint(-15, 5)),
     lambda p, x, lo, hi: max(lo, min(hi, x)))
_fam("c_to_f", ["c_to_f", "celsius_to_fahrenheit", "to_fahrenheit", "ctof"], [("c", "celsius", "temp", "deg")],
     "converts the temperature {a} from degrees Celsius to degrees Fahrenheit and returns it",
     lambda r, p: (r.choice([r.randint(-40, 100), r.randint(-400, 1000) / 10]),), lambda p, c: c * 9 / 5 + 32, edge=[(0,), (100,)])
_fam("f_to_c", ["f_to_c", "fahrenheit_to_celsius", "to_celsius", "ftoc"], [("f", "fahrenheit", "temp", "deg")],
     "converts the temperature {a} from degrees Fahrenheit to degrees Celsius and returns it",
     lambda r, p: (r.randint(-40, 212),), lambda p, f: (f - 32) * 5 / 9, edge=[(32,), (212,)])
_fam("is_power_of_two", ["is_power_of_two", "power_of_two", "is_pow2", "pow2_check"], [INT_ARGS],
     "returns True if the positive integer {a} is a power of two (1, 2, 4, 8, ...) and False otherwise",
     lambda r, p: (r.choice([2 ** r.randint(0, 12), r.randint(1, 5000)]),), lambda p, n: n & (n - 1) == 0,
     edge=[(1,), (6,)])
_fam("leap_year", ["is_leap", "leap_year", "is_leap_year", "leap"], [("year", "y", "yr", "n")],
     "returns True if {a} is a leap year in the Gregorian calendar and False otherwise",
     lambda r, p: (r.choice([r.randint(1800, 2400), r.choice([1900, 2000, 2100, 2400])]),),
     lambda p, y: y % 4 == 0 and (y % 100 != 0 or y % 400 == 0), edge=[(1900,), (2000,), (2024,)])
_fam("sum_of_squares", ["sum_of_squares", "square_sum", "squares_total", "sum_squares"], [INT_ARGS],
     "returns the sum of the squares of the integers from 1 to {a}, for a non-negative integer {a}",
     lambda r, p: (r.randint(0, 30),), lambda p, n: sum(i * i for i in range(1, n + 1)), edge=[(0,)])
_fam("triangle_number", ["triangle", "triangular", "sum_to", "sum_up_to"], [INT_ARGS],
     "returns the sum of all integers from 1 to {a} as an integer, for a non-negative integer {a}",
     lambda r, p: (r.randint(0, 500),), lambda p, n: n * (n + 1) // 2, edge=[(0,)])
_fam("to_binary", ["to_binary", "binary", "bin_string", "as_binary"], [INT_ARGS],
     "returns the binary representation of the non-negative integer {a} as a string without the 0b prefix",
     lambda r, p: (r.randint(0, 1000),), lambda p, n: format(n, "b"), edge=[(0,), (1,)])
_fam("hypotenuse", ["hypotenuse", "hyp", "long_side", "diagonal"], [("a", "x", "leg1", "p"), ("b", "y", "leg2", "q")],
     "returns the length of the hypotenuse of a right triangle whose two legs are {a} and {b}",
     lambda r, p: (r.randint(1, 20), r.randint(1, 20)), lambda p, a, b: math.hypot(a, b), edge=[(3, 4)])
_fam("circle_area", ["circle_area", "area", "disc_area", "area_of_circle"], [("r", "radius", "rad", "size")],
     "returns the area of a circle with radius {a}, using math.pi",
     lambda r, p: (r.choice([r.randint(0, 20), r.randint(1, 100) / 10]),), lambda p, x: math.pi * x * x)
_fam("is_even", ["is_even", "even", "check_even", "even_number"], [INT_ARGS],
     "returns True if the integer {a} is even and False otherwise", lambda r, p: (r.randint(-50, 50),),
     lambda p, n: n % 2 == 0, edge=[(0,), (-3,)])
_fam("count_digits", ["count_digits", "num_digits", "digit_count", "length_of_number"], [INT_ARGS],
     "returns the number of decimal digits in the non-negative integer {a}, where 0 has one digit",
     lambda r, p: (r.randint(0, 10 ** r.randint(1, 9)),), lambda p, n: len(str(n)), edge=[(0,), (10,)])
_fam("power", ["power", "raise_to", "pow_int", "exponent"], [("base", "b", "x", "a"), ("exp", "e", "n", "k")],
     "returns {a} raised to the non-negative integer power {b}",
     lambda r, p: (r.randint(-9, 9), r.randint(0, 8)), lambda p, b, e: b ** e, edge=[(0, 0), (5, 0)])
_fam("sign", ["sign", "signum", "sign_of", "polarity"], [INT_ARGS],
     "returns 1 if the number {a} is positive, -1 if it is negative and 0 if it is zero",
     lambda r, p: (r.randint(-100, 100),), lambda p, n: (n > 0) - (n < 0), edge=[(0,)])

FAMILY_NAMES: Tuple[str, ...] = tuple(FAMILIES)

CODE_TEMPLATES = (
    "Write a Python function {sig} that {desc}.",
    "In Python, write {sig}, which {desc}.",
    "Implement {sig} in Python so that it {desc}.",
    "Python: define {sig} that {desc}.",
    "Give me a one-line Python function {sig} that {desc}.",
)

CODE_SYSTEM_PROMPT = (
    "You are a concise Python tutor. Reply on ONE line of at most 60 words: first one short plain-English "
    "sentence explaining the idea, then the complete function written as a single line of the form "
    "def name(args): return expression. Use exactly the function name and argument names given. "
    "No newlines, no markdown, no backticks, no comments, no type hints, no print. The only allowed import is "
    "math, written inside the function: def name(args): import math; return expression.\n"
    "Example reply: Add up the list with the built-in sum. def total(xs): return sum(xs)\n"
    "Example reply: Use the square root from the math module. def root(x): import math; return math.sqrt(x)"
)


@dataclass
class CodeTask:
    family: str
    func: str
    args: Tuple[str, ...]
    user: str
    tests: List[Tuple[tuple, Any]] = field(default_factory=list)

    @property
    def check(self) -> dict:
        return {"func": self.func, "args": list(self.args), "tests": repr(self.tests)}


def make_code_task(family: str, rng: random.Random, n_tests: int = 6) -> CodeTask:
    """Draw one concrete prompt + test set from a family (names, constants, inputs all random)."""

    fam = FAMILIES[family]
    params = fam.params(rng) if fam.params else {}
    func = rng.choice(fam.fns)
    names: List[str] = []
    for pool in fam.args:
        choices = [n for n in pool if n not in names and n != func]
        names.append(rng.choice(choices))
    fmt = {"a": names[0], "b": names[1] if len(names) > 1 else "", "c": names[2] if len(names) > 2 else "", **params}
    desc = fam.desc.format(**fmt)
    sig = f"{func}({', '.join(names)})"
    templates = list(CODE_TEMPLATES)
    rng.shuffle(templates)
    user = None
    for t in templates:
        cand = t.format(sig=sig, desc=desc)
        if word_count(cand) <= MAX_USER_WORDS:
            user = cand
            break
    if user is None:
        raise ValueError(f"family {family} cannot fit a {MAX_USER_WORDS}-word prompt")
    tests, seen = [], set()
    for args in list(fam.edge):
        key = repr(args)
        if key not in seen:
            seen.add(key)
            tests.append((tuple(args), fam.ref(params, *args)))
    tries = 0
    while len(tests) < n_tests and tries < 60:
        tries += 1
        args = fam.gen(rng, params)
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        tests.append((tuple(args), fam.ref(params, *args)))
    return CodeTask(family, func, tuple(names), user, tests)


def code_task_for_index(split: str, index: int, seed: int = 0) -> CodeTask:
    """Deterministic task for attempt ``index`` of ``split``: families cycle, seeds are split-disjoint."""

    family = FAMILY_NAMES[index % len(FAMILY_NAMES)]
    rng = random.Random(f"expanse-code|{seed}|{split}|{index}")
    return make_code_task(family, rng)


def _key_seed(key: str) -> int:
    """Stable per-item sampling seed, so a rerun of the same key asks the teacher the same way."""

    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def ordered_prefetch(items: Iterable, call: Callable, parallel: int = 1):
    """Yield ``(item, call(item))`` in input order with up to ``parallel`` calls in flight.

    Only the teacher request runs in worker threads; the caller consumes results
    one at a time in its own thread, so verification, split assignment and every
    append to the resumable jsonl logs stay sequential and deterministic. The
    ``items`` generator is also advanced in the caller's thread. When the caller
    stops early, in-flight requests are awaited and dropped -- nothing was logged
    for them, so a rerun simply retries those keys. A dead server re-raises.
    """

    if parallel <= 1:
        for item in items:
            yield item, call(item)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    source = iter(items)
    pending: deque = deque()
    pool = ThreadPoolExecutor(max_workers=parallel)
    try:
        def fill() -> None:
            while len(pending) < parallel:
                try:
                    nxt = next(source)
                except StopIteration:
                    return
                pending.append((nxt, pool.submit(call, nxt)))

        fill()
        while pending:
            item, fut = pending.popleft()
            result = fut.result()
            yield item, result
            fill()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _teacher_call(server: LlamaServer, method: str, *args, **kwargs) -> Optional[str]:
    """Call ``server.complete``/``server.chat``; None on a transient HTTP error, re-raise if the server died."""

    try:
        return getattr(server, method)(*args, **kwargs)
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
        if server.proc is None or server.proc.poll() is not None:
            raise
        return None


def generate_code_rows(server: LlamaServer, *, data_dir: Path = DATA, target: int = 1500, heldout: int = 150,
                       limit: Optional[int] = None, seed: int = 0, temperature: float = 0.3,
                       n_predict: int = 160, log: Callable[[str], None] = print) -> dict:
    """Ask the coder teacher for verified one-line functions until the targets are met.

    Held-out prompts are filled first (they are few and eval needs them), then
    train. ``limit`` caps teacher calls in this invocation. Every attempt is
    logged to ``code_attempts.jsonl`` under key ``split:index``; passing rows go
    to ``code_rows.jsonl``. Rerunning resumes after the last logged attempt.
    Requests run ``server.parallel`` at a time (`ordered_prefetch`); results are
    verified and logged in key order.
    """

    data_dir = Path(data_dir)
    rows_path, attempts_path = data_dir / "code_rows.jsonl", data_dir / "code_attempts.jsonl"
    rows = read_jsonl(rows_path)
    done = {a["key"] for a in read_jsonl(attempts_path)}
    users = {r["user"] for r in rows}
    counts = Counter(r["split"] for r in rows)
    calls, t0 = 0, time.time()
    session = Counter()

    def call(key_task):
        key, task = key_task
        messages = [{"role": "system", "content": CODE_SYSTEM_PROMPT}, {"role": "user", "content": task.user}]
        return _teacher_call(server, "chat", messages, n_predict=n_predict, temperature=temperature,
                             seed=_key_seed(key))

    for split, want in (("heldout", heldout), ("train", target)):
        if counts[split] >= want or (limit is not None and calls >= limit):
            continue
        max_index = 50 * max(want, 1) + 1000

        def tasks(split=split, max_index=max_index):
            for index in range(max_index + 1):
                key = f"{split}:{index}"
                if key in done:
                    continue
                task = code_task_for_index(split, index, seed)
                if task.user in users:
                    append_jsonl(attempts_path, {"key": key, "family": task.family, "ok": False, "reason": "duplicate_prompt"})
                    done.add(key)
                    continue
                yield key, task

        finished = False
        for (key, task), reply in ordered_prefetch(tasks(), call, server.parallel):
            calls += 1
            if reply is None:           # transient server error: not logged, so a rerun retries this key
                session["server_error"] += 1
            elif task.user in users:    # an identical prompt passed while this one was in flight
                append_jsonl(attempts_path, {"key": key, "family": task.family, "ok": False, "reason": "duplicate_prompt"})
                done.add(key)
            else:
                res = verify_code_reply(reply, task.check)
                reason = res["reason"]
                if res["ok"]:
                    bad = fits_student(task.user, res["assistant"])
                    if bad:
                        res["ok"], reason = False, bad
                if res["ok"]:
                    row = {"user": task.user, "assistant": res["assistant"], "domain": "programming",
                           "task": f"coder_{task.family}", "source": CODER_SOURCE, "split": split, "check": task.check}
                    append_jsonl(rows_path, row)
                    users.add(task.user)
                    counts[split] += 1
                session[reason] += 1
                append_jsonl(attempts_path, {"key": key, "family": task.family, "ok": bool(res["ok"]), "reason": reason,
                                             "reply": reply[:600], "passed": res.get("passed"), "total": res.get("total")})
                done.add(key)
                if calls % 10 == 0 or res["ok"] is False and calls < 5:
                    sp = server.speed()
                    log(f"[code] calls {calls} train {counts['train']}/{target} heldout {counts['heldout']}/{heldout} "
                        f"last={reason} gen {sp['gen_tokens_per_s']} tok/s  {time.time() - t0:.0f}s")
            if counts[split] >= want or (limit is not None and calls >= limit):
                finished = True
                break
        if not finished and counts[split] < want:
            log(f"[code] {split}: gave up after {max_index} indices")
    summary = {"calls": calls, "session_reasons": dict(session), "seconds": round(time.time() - t0, 1),
               "speed": server.speed(), "temperature": temperature, "seed": seed, "parallel": server.parallel}
    receipt = write_code_receipt(data_dir, run=summary, target=target, heldout=heldout)
    log(f"[code] done: {receipt['rows']}  pass rate {receipt['attempts']['pass_rate']}")
    return receipt


def _reason_key(reason: str) -> str:
    """Bucket reject reasons for receipts: ``tests:4/6`` -> ``tests:failed``, ``ast:name:open`` -> ``ast:name``."""

    reason = reason or "none"
    if reason.startswith("tests:"):
        detail = reason[6:]
        return "tests:failed" if re.fullmatch(r"\d+/\d+", detail) else "tests:" + detail.split(":")[0]
    if reason.startswith("ast:"):
        return ":".join(reason.split(":")[:2])
    return reason


def write_code_receipt(data_dir: Path, run: Optional[dict] = None, target: int = 0, heldout: int = 0) -> dict:
    """Recount the files (never trust in-memory counters) and append this run's stats."""

    data_dir = Path(data_dir)
    rows = read_jsonl(data_dir / "code_rows.jsonl")
    attempts = read_jsonl(data_dir / "code_attempts.jsonl")
    path = data_dir / "code_rows.receipt.json"
    old = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    judged = [a for a in attempts if a.get("reason") != "duplicate_prompt"]
    reasons = Counter(_reason_key(a["reason"]) for a in judged if not a["ok"])
    receipt = {
        "schema": "expanse-teacher-code-v1",
        "teacher": {"name": "Qwen2.5-Coder-7B-Instruct (Q4_K_M GGUF)", "gguf": str(CODER_GGUF),
                    "note": "Qwen3-Coder has no 7B/8B release; Qwen2.5-Coder-7B-Instruct substituted (see DESIGN.md)"},
        "rows": dict(Counter(r["split"] for r in rows)),
        "targets": {"train": target or old.get("targets", {}).get("train"), "heldout": heldout or old.get("targets", {}).get("heldout")},
        "families_total": len(FAMILY_NAMES),
        "families_covered": {s: len({r["task"] for r in rows if r["split"] == s}) for s in ("train", "heldout")},
        "per_family_train": dict(sorted(Counter(r["task"] for r in rows if r["split"] == "train").items())),
        "attempts": {"total": len(judged), "passed": sum(1 for a in judged if a["ok"]),
                     "pass_rate": round(sum(1 for a in judged if a["ok"]) / max(1, len(judged)), 4),
                     "reject_reasons": dict(reasons.most_common())},
        "verification": {"ast_whitelist": True, "subprocess": "python -I -X utf8 -c HARNESS", "timeout_s": SANDBOX_TIMEOUT_S,
                         "memory_cap_bytes": SANDBOX_MEMORY_BYTES, "env": "PATH+SYSTEMROOT only",
                         "tests_per_task": 6, "rule": "all tests pass"},
        "system_prompt": CODE_SYSTEM_PROMPT,
        "runs": old.get("runs", []) + ([run] if run else []),
    }
    write_json_atomic(path, receipt)
    return receipt


# ---------------------------------------------------------------------------
# Biomedical: term mining
# ---------------------------------------------------------------------------

#: Frequent PubMed words that are English/academic rather than biomedical.
#: The student vocabulary is only ~4.7k words, so it cannot be the whole
#: "common English" list on its own; this stoplist covers the gap for the
#: words a 28,896-token PubMed BPE vocabulary actually contains.
BIO_STOPLIST = frozenset("""
patients patient however therefore significant significantly respectively previously including compared
associated demonstrated performed observed increased decreased analysis analyses results conclusion
conclusions background methods objective objectives purpose participants treatment treatments hospital
university department evaluated evaluate evaluation reported determine determined investigate investigated
investigation additional different important potential presence approximately following underwent received
whereas although moreover furthermore addition regarding according available because between including
within without several various provide provided providing obtained compared comparison comparing studies
because related relationship relatively similar similarly suggest suggests suggested suggesting indicate
indicates indicated indicating showed shown increase decrease reduced reduction increasing decreasing
multiple individual individuals population populations previous currently current recently present
presented presents general generally especially particular particularly possible possibly required require
requires requiring include includes included involving involved involves further finally overall total
subjects subject control controls controlled group groups groupings children adults elderly women
effects effect effective effectively efficacy outcome outcomes measured measure measures measurement
characteristics characteristic factors factor features feature associated association associations
approach approaches technique techniques strategy strategies examined examine examining identify
identified identification developed development developing therapy therapeutic clinical clinically
research researchers experimental experiment experiments procedure procedures observed observation
observations significance statistically statistical regression analyzed analysed assessed assessment
assessing questionnaire questionnaires survey surveys interview interviews national international
program programs programme programmes management medical medicine health healthcare quality
secondary primary prospective retrospective consecutive randomized randomised controlled trial trials
followed period periods duration baseline months weeks years days hours minutes compared versus
receiving received underwent undergoing consisted consisting containing contained demonstrate
demonstrates confirmed confirm confirms revealed reveals whether therefore thereby therein moreover
""".split())


def _inflections(w: str) -> set:
    out = {w, w + "s", w + "es", w + "ed", w + "d", w + "ing", w + "er", w + "ers", w + "est", w + "ly",
           w + "ness", w + "ment", w + "ments", w + "al", w + "ally", w + "ity", w + "ion", w + "ions"}
    if w.endswith("e"):
        out |= {w[:-1] + "ing", w[:-1] + "ion", w[:-1] + "ions", w[:-1] + "ation", w[:-1] + "ations"}
    if w.endswith("y"):
        out |= {w[:-1] + "ies", w[:-1] + "ied", w[:-1] + "ily", w[:-1] + "iness"}
    if len(w) >= 3:
        out |= {w + w[-1] + "ed", w + w[-1] + "ing", w + w[-1] + "er"}
    return out


def load_student_words(ckpt: Path = ARCH_CKPT) -> set:
    """Lower-cased alphabetic words of the Archimedes student vocabulary (read from its checkpoint)."""

    import torch  # local: only this function needs torch

    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    tokens = payload["tokenizer"]["tokens"]
    words = {t.strip().lower() for t in tokens if t.strip().isalpha()}
    del payload
    return words


def common_english(student_words: Iterable[str]) -> set:
    common = set(BIO_STOPLIST)
    for w in student_words:
        common |= _inflections(w)
    return common


def mine_biomedical_terms(student_words: Iterable[str], tokenizer_json: Path = BIOMEDLM_DIR / "tokenizer.json",
                          min_len: int = 7) -> List[str]:
    """BioMedLM's own single-token words that the student cannot already spell.

    A GPT-2 BPE vocabulary built on PubMed promotes a word to a single ``Ġ``
    token only when it is very frequent there, so these are the field's core
    terms. Returned in vocabulary-id order, i.e. most frequent merge first.
    """

    vocab = json.load(open(tokenizer_json, encoding="utf-8"))["model"]["vocab"]
    student_words = set(student_words)
    common = common_english(student_words)
    out = []
    for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        if not tok.startswith("Ġ"):
            continue
        w = tok[1:]
        if len(w) < min_len or not (w.isascii() and w.isalpha() and w.islower()):
            continue
        if w in student_words or w in common:
            continue
        out.append(w)
    return out


QWEN_TOKENIZER = TEACHERS / "qwen2.5-coder-7b-instruct" / "tokenizer.json"
#: Qwen ids below this are frequent general-web words ("develop", "success", "sufficient").
GENERAL_COMMON_ID = 30000


def whole_word_counts(texts: Iterable[str]) -> Counter:
    return Counter(re.findall(r"[a-z]+", " ".join(texts).lower()))


def term_tiers(terms: Sequence[str], corpus_counts: Counter, general_vocab: Optional[dict] = None) -> Dict[str, int]:
    """Rank mined terms by how biomedical they are; terms absent from the corpus are dropped.

    Nearly half of the single ``Ġ``-tokens of a PubMed BPE vocabulary are word
    *fragments* ("signific", "concentr", "anaphyl") that the merges later extend
    into real words; a fragment never occurs as a whole word in running text, so
    requiring one whole-word occurrence in the PubMedQA abstracts removes them.
    The rest are tiered with a general-purpose vocabulary (Qwen's): a word that
    web-scale BPE never promoted to a single token is specialist vocabulary.

      tier 0  whole word in the corpus, not a single general-vocab token ("myocardial")
      tier 1  single general-vocab token, but a rare one (id >= 30000: "insulin", "apoptosis")
      tier 2  frequent general-web word (id < 30000: "sufficient") -- used last, if ever
    """

    tiers = {}
    for t in terms:
        if corpus_counts.get(t, 0) < 1:
            continue
        gid = general_vocab.get("Ġ" + t) if general_vocab else None
        tiers[t] = 0 if gid is None else (1 if gid >= GENERAL_COMMON_ID else 2)
    return tiers


# ---------------------------------------------------------------------------
# Biomedical: prompts
# ---------------------------------------------------------------------------

DEFINITION_FEWSHOT = (
    "Short, accurate definitions from a biomedical glossary.\n\n"
    "Insulin is a peptide hormone secreted by the beta cells of the pancreas that lowers blood glucose by "
    "promoting its uptake into muscle, fat and liver cells.\n\n"
    "Erythrocytes are red blood cells, which carry oxygen from the lungs to the tissues bound to the protein "
    "hemoglobin.\n\n"
    "Apoptosis is a regulated form of programmed cell death in which the cell shrinks, its chromatin condenses "
    "and its fragments are removed without causing inflammation.\n\n"
    "Bronchitis is inflammation of the lining of the bronchial tubes, which usually causes cough and mucus "
    "production.\n\n"
)

DEFINITION_USER_TEMPLATES = (
    "What {be} {term}?",
    "Define {term}.",
    "What does the term {term} mean in biomedicine?",
    "Explain {term} in one sentence.",
    "In medicine and biology, what {be} {term}?",
    "Give a short definition of {term}.",
)

PUBMEDQA_FEWSHOT = (
    "Context: In a randomized trial of 240 adults with mild hypertension, twelve weeks of daily brisk walking "
    "lowered systolic blood pressure by a mean of 6 mmHg compared with usual activity.\n"
    "Question: Does regular brisk walking lower blood pressure in adults with mild hypertension?\n"
    "Answer: yes, because the walking group had clearly lower systolic pressure than the control group after "
    "twelve weeks.\n\n"
    "Context: Among 410 volunteers randomly assigned to daily vitamin C or placebo for one winter, the number "
    "of colds per person was 1.8 in both groups and symptom duration did not differ.\n"
    "Question: Does daily vitamin C supplementation prevent the common cold?\n"
    "Answer: no, because volunteers taking vitamin C caught as many colds as those taking placebo.\n\n"
    "Context: In a small cohort of 38 patients, higher serum ferritin was weakly linked to longer hospital "
    "stay, but the association lost significance after adjustment for age and infection severity.\n"
    "Question: Does high serum ferritin predict a longer hospital stay?\n"
    "Answer: maybe, because the weak link disappeared once age and infection severity were taken into account.\n\n"
)

PUBMEDQA_USER_TEMPLATES = ("{q}", "{q} Answer yes, no or maybe.", "Biomedical question: {q}")

#: GBNF for the PubMedQA answer. A base LM told "Answer:" tends to restate the
#: question instead of answering it, so the *shape* is constrained: the first
#: word must be yes/no/maybe and the reason must start with "because". The
#: decision itself is still the model's own (its probabilities renormalised over
#: the three words), and a row is kept only if it agrees with the expert label.
PUBMEDQA_GRAMMAR = 'root ::= " " ("yes" | "no" | "maybe") ", because " [^\\n]+ "\\n"'

JUDGE_SYSTEM = ("You are a strict biomedical fact checker. You answer with exactly one word: yes or no.")
JUDGE_TEMPLATE = "Statement: {statement}\nIs this statement accurate? yes/no"

#: Citation debris a PubMed-trained LM emits; never allowed in a row.
_CITATION = re.compile(r"et al|\[\d|http|www\.|\bdoi\b|@", re.I)
#: First-person / paper-structure talk; allowed in a PubMedQA reason, not in a glossary definition.
_STUDY_TALK = re.compile(r"\bwe\b|\bour\b|\bthis study\b|\bFig\b|\bFigure\b|\bTable\b|\(\d", re.I)
#: Definitions that describe the word instead of the thing ("X is a term used to describe ...").
_VAGUE = re.compile(r"^\w+ (is|are) (a |the )?(term|word|name)s? (used|that|for|which)|used to describe", re.I)


def _plural(term: str) -> bool:
    return term.endswith("s") and not term.endswith(("ss", "is", "us", "sis", "itis", "ics"))


def _repetitive(text: str) -> bool:
    toks = text.lower().split()
    grams = Counter(tuple(toks[i:i + 3]) for i in range(len(toks) - 2))
    return any(c >= 2 for c in grams.values())


def clean_definition(term: str, completion: str) -> Tuple[Optional[str], str]:
    """Turn ``"{Term} is" + completion`` into a 1-2 sentence definition, or (None, reason)."""

    verb = "are" if _plural(term) else "is"
    if completion and not completion[0].isspace():
        completion = " " + completion
    text = one_line(strip_markdown(to_ascii(f"{term.capitalize()} {verb}{completion}")))
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    # a sentence counts only if it ends in terminal punctuation; the generation
    # cap (48 tokens) usually cuts the last one mid-way
    complete = [s for s in sentences if s.endswith((".", "!", "?"))]
    if not complete or not sentences[0].endswith((".", "!", "?")):
        return None, "truncated"
    out = complete[0]
    if word_count(out) < 14 and len(complete) > 1 and word_count(out + " " + complete[1]) <= MAX_ASSISTANT_WORDS:
        out = out + " " + complete[1]
    n = word_count(out)
    if n < 7:
        return None, "too_short"
    if n > MAX_ASSISTANT_WORDS:
        return None, "too_long"
    if _CITATION.search(out):
        return None, "citation"
    if _STUDY_TALK.search(out):
        return None, "study_talk"
    if _VAGUE.search(out):
        return None, "vague"
    if _repetitive(out):
        return None, "repetition"
    if out.lower().count(term.lower()) > 2:
        return None, "term_repeated"
    return out, "ok"


def parse_pubmedqa_answer(completion: str) -> Tuple[Optional[str], Optional[str], str]:
    """``" yes, because ..."`` -> ("yes", "yes, because ...", "ok") or (None, None, reason)."""

    text = one_line(strip_markdown(to_ascii(completion)))
    m = re.match(r"^(yes|no|maybe)\b[\s,;:.-]*(.*)$", text, re.I)
    if not m:
        return None, None, "no_decision"
    decision, reason = m.group(1).lower(), m.group(2).strip()
    if not reason.lower().startswith("because"):
        return decision, None, "no_because"
    reason = re.split(r"(?<=[.!?])\s", reason)[0].strip()
    if not reason.endswith((".", "!", "?")):
        return decision, None, "truncated"
    answer = f"{decision}, {reason}"
    if word_count(answer) < 6:
        return decision, None, "too_short"
    if word_count(answer) > MAX_ASSISTANT_WORDS:
        return decision, None, "too_long"
    if _CITATION.search(reason):
        return decision, None, "citation"
    if _repetitive(reason):
        return decision, None, "repetition"
    return decision, answer, "ok"


def load_pubmedqa(data_dir: Path = DATA) -> List[dict]:
    """PubMedQA ``pqa_labeled`` (1,000 expert-labelled questions), cached as jsonl in data_dir."""

    cache = Path(data_dir) / "pubmedqa_pqa_labeled.jsonl"
    if cache.exists():
        return read_jsonl(cache)
    from datasets import load_dataset  # local: network + heavy import only on first use

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = []
    for ex in ds:
        rows.append({"pubid": int(ex["pubid"]), "question": ex["question"],
                     "contexts": list(ex["context"]["contexts"]), "long_answer": ex["long_answer"],
                     "final_decision": ex["final_decision"]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    os.replace(tmp, cache)
    return rows


def pubmedqa_prompt(item: dict, max_context_words: int = 220) -> str:
    ctx = one_line(to_ascii(" ".join(item["contexts"])))
    words = ctx.split()
    if len(words) > max_context_words:
        ctx = " ".join(words[:max_context_words]) + " ..."
    return f"{PUBMEDQA_FEWSHOT}Context: {ctx}\nQuestion: {one_line(to_ascii(item['question']))}\nAnswer:"


def build_bio_queue(terms: Sequence[str], pubmedqa: Sequence[dict], seed: int = 0,
                    tiers: Optional[Dict[str, int]] = None) -> List[dict]:
    """Deterministic processing order: 2 terms : 1 question, each shuffled with `seed`.

    With `tiers` (see `term_tiers`) only tiered terms are used, lower tier first,
    shuffled within each tier; without it every term is used, shuffled.
    """

    rng = random.Random(f"expanse-bio|{seed}")
    ordered: List[str] = []
    if tiers is None:
        ordered = list(terms)
        rng.shuffle(ordered)
    else:
        for tier in sorted(set(tiers.values())):
            group = [t for t in terms if tiers.get(t) == tier]
            rng.shuffle(group)
            ordered += group
    term_items = [{"key": f"term:{t}", "kind": "definition", "term": t} for t in ordered]
    qs = [q for q in pubmedqa if word_count(one_line(to_ascii(q["question"]))) <= MAX_USER_WORDS - 5]
    qs = sorted(qs, key=lambda q: q["pubid"])
    rng.shuffle(qs)
    q_items = [{"key": f"pqa:{q['pubid']}", "kind": "pubmedqa", "item": q} for q in qs]
    queue, ti, qi = [], 0, 0
    while ti < len(term_items) or qi < len(q_items):
        for _ in range(2):
            if ti < len(term_items):
                queue.append(term_items[ti])
                ti += 1
        if qi < len(q_items):
            queue.append(q_items[qi])
            qi += 1
    return queue


# ---------------------------------------------------------------------------
# Biomedical: generation (BioMedLM) and judging (Qwen2.5-Coder)
# ---------------------------------------------------------------------------

def _bio_state(data_dir: Path) -> dict:
    """Recount bio progress from the append-only files."""

    cands = read_jsonl(Path(data_dir) / "bio_candidates.jsonl")
    judg = {j["key"]: j for j in read_jsonl(Path(data_dir) / "bio_judgements.jsonl")}
    rows = read_jsonl(Path(data_dir) / "bio_rows.jsonl")
    row_keys = {r["check"]["key"] for r in rows}
    pending = [c for c in cands if c["status"] == "pending_judge" and c["key"] not in judg]
    n_judged = sum(1 for c in cands if c["status"] == "pending_judge" and c["key"] in judg)
    n_yes = sum(1 for c in cands if c["status"] == "pending_judge" and judg.get(c["key"], {}).get("verdict") == "yes")
    return {"candidates": cands, "judgements": judg, "rows": rows, "row_keys": row_keys, "pending": pending,
            "accept_rate": (n_yes / n_judged) if n_judged >= 20 else 0.7,
            "counts": Counter(r["split"] for r in rows),
            "pending_counts": Counter(c["split"] for c in pending)}


def generate_bio_candidates(server: LlamaServer, *, data_dir: Path = DATA, target: int = 1500, heldout: int = 150,
                            limit: Optional[int] = None, seed: int = 0, student_words: Optional[set] = None,
                            log: Callable[[str], None] = print) -> dict:
    """Prompt BioMedLM through the bio queue until expected accepted rows meet the targets.

    PubMedQA answers whose decision matches the expert label become rows at
    once; definitions become ``pending_judge`` candidates for `judge_bio_candidates`.
    "Expected" counts pending definitions at the running judge accept rate.
    Split is assigned when an item is first processed: held-out until the
    held-out quota is (expectedly) full, then train -- so every term/question
    lives in exactly one split.
    """

    data_dir = Path(data_dir)
    cand_path, rows_path = data_dir / "bio_candidates.jsonl", data_dir / "bio_rows.jsonl"
    st = _bio_state(data_dir)
    processed = {c["key"] for c in st["candidates"]}
    rate = st["accept_rate"]
    counts, pend = Counter(st["counts"]), Counter(st["pending_counts"])
    if student_words is None:
        student_words = load_student_words()
    terms = mine_biomedical_terms(student_words)
    pqa = load_pubmedqa(data_dir)
    corpus = whole_word_counts(" ".join(q["contexts"]) + " " + q.get("long_answer", "") for q in pqa)
    general = json.load(open(QWEN_TOKENIZER, encoding="utf-8"))["model"]["vocab"] if QWEN_TOKENIZER.exists() else None
    tiers = term_tiers(terms, corpus, general)
    queue = build_bio_queue(terms, pqa, seed=seed, tiers=tiers)
    tier_counts = dict(sorted(Counter(tiers.values()).items()))
    log(f"[bio] terms mined {len(terms)} (whole-word in corpus {len(tiers)}, tiers {tier_counts})  pubmedqa {len(pqa)}  "
        f"queue {len(queue)}  processed {len(processed)}  rows {dict(counts)}  pending {dict(pend)}  "
        f"accept_rate {rate:.2f}")
    calls, t0, session = 0, time.time(), Counter()

    def expected(split):
        return counts[split] + rate * pend[split]

    def call(pos_item):
        _, item = pos_item
        req_seed = _key_seed(item["key"])
        if item["kind"] == "definition":
            term = item["term"]
            verb = "are" if _plural(term) else "is"
            return _teacher_call(server, "complete", DEFINITION_FEWSHOT + f"{term.capitalize()} {verb}",
                                 n_predict=48, temperature=0.3, stop=["\n"], seed=req_seed)
        return _teacher_call(server, "complete", pubmedqa_prompt(item["item"]), n_predict=56, temperature=0.2,
                             stop=["\n"], seed=req_seed, grammar=PUBMEDQA_GRAMMAR)

    def todo():
        for pos, item in enumerate(queue):
            if item["key"] not in processed:
                yield pos, item

    # The teacher call depends only on the item, so it can run ahead; the split
    # is still assigned here, in queue order, from the running expectation.
    for (pos, item), completion in ordered_prefetch(todo(), call, server.parallel):
        if expected("train") >= target and expected("heldout") >= heldout:
            break
        if limit is not None and calls >= limit:
            break
        calls += 1
        if completion is None:
            session["server_error"] += 1
            continue
        split = "heldout" if expected("heldout") < heldout else "train"
        rng = random.Random(f"expanse-bio-user|{seed}|{item['key']}")
        cand = {"key": item["key"], "kind": item["kind"], "split": split, "status": "rejected", "reason": None}
        if item["kind"] == "definition":
            term = item["term"]
            verb = "are" if _plural(term) else "is"
            definition, why = clean_definition(term, completion)
            user = rng.choice(DEFINITION_USER_TEMPLATES).format(term=term, be=verb)
            cand.update(raw=completion[:400], reason=why, user=user)
            if definition is not None:
                bad = fits_student(user, definition)
                if bad:
                    cand["reason"] = bad
                else:
                    cand.update(status="pending_judge", assistant=definition,
                                row={"user": user, "assistant": definition, "domain": "biomedical",
                                     "task": "bio_definition", "source": BIO_SOURCE, "split": split,
                                     "check": {"key": item["key"], "term": term}})
                    pend[split] += 1
        else:
            q = item["item"]
            decision, answer, why = parse_pubmedqa_answer(completion)
            question = one_line(to_ascii(q["question"]))
            user = rng.choice(PUBMEDQA_USER_TEMPLATES).format(q=question)
            if word_count(user) > MAX_USER_WORDS:
                user = question
            cand.update(raw=completion[:400], reason=why, user=user, decision=decision,
                        final_decision=q["final_decision"])
            if answer is not None:
                if decision != q["final_decision"]:
                    cand["reason"] = "label_mismatch"
                elif fits_student(user, answer):
                    cand["reason"] = fits_student(user, answer)
                else:
                    row = {"user": user, "assistant": answer, "domain": "biomedical", "task": "bio_pubmedqa",
                           "source": BIO_SOURCE, "split": split,
                           "check": {"key": item["key"], "pubid": q["pubid"], "final_decision": q["final_decision"]}}
                    cand.update(status="accepted", assistant=answer)
                    if item["key"] not in st["row_keys"]:
                        append_jsonl(rows_path, row)
                        st["row_keys"].add(item["key"])
                        counts[split] += 1
        session[f"{item['kind']}:{cand['reason']}"] += 1
        append_jsonl(cand_path, cand)
        processed.add(item["key"])
        if calls % 10 == 0:
            sp = server.speed()
            log(f"[bio] calls {calls} pos {pos} rows {dict(counts)} pending {dict(pend)} "
                f"gen {sp['gen_tokens_per_s']} tok/s prompt {sp['prompt_tokens_per_s']} tok/s {time.time() - t0:.0f}s")
    exhausted = all(it["key"] in processed for it in queue)
    run = {"phase": "generate", "calls": calls, "session_reasons": dict(session), "seconds": round(time.time() - t0, 1),
           "speed": server.speed(), "queue_exhausted": exhausted, "terms_mined": len(terms),
           "terms_whole_word": len(tiers), "term_tiers": tier_counts, "seed": seed}
    receipt = write_bio_receipt(data_dir, run=run, target=target, heldout=heldout)
    receipt["queue_exhausted"] = exhausted
    return receipt


def judge_statement(server: LlamaServer, statement: str) -> str:
    """One-token cross-teacher verdict: "yes", "no" or "unclear"."""

    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_TEMPLATE.format(statement=statement)}]
    out = _teacher_call(server, "chat", messages, n_predict=1, temperature=0.0, seed=0)
    if out is None:
        return "error"
    out = out.strip().lower()
    if out.startswith("yes"):
        return "yes"
    if out.startswith("no"):
        return "no"
    return "unclear"


def judge_bio_candidates(server: LlamaServer, *, data_dir: Path = DATA, limit: Optional[int] = None,
                         log: Callable[[str], None] = print) -> dict:
    """Have the coder teacher judge every pending definition; "yes" rows are appended to bio_rows.jsonl."""

    data_dir = Path(data_dir)
    st = _bio_state(data_dir)
    calls, t0, verdicts = 0, time.time(), Counter()
    pending = st["pending"] if limit is None else st["pending"][:limit]
    for cand, verdict in ordered_prefetch(pending, lambda c: judge_statement(server, c["assistant"]), server.parallel):
        calls += 1
        verdicts[verdict] += 1
        if verdict == "error":          # transient: leave it pending for the next judge run
            continue
        if verdict == "yes" and cand["key"] not in st["row_keys"]:
            row = dict(cand["row"])
            row["check"] = dict(row["check"], judge=f"{CODER_SOURCE}:yes")
            append_jsonl(data_dir / "bio_rows.jsonl", row)
            st["row_keys"].add(cand["key"])
        append_jsonl(data_dir / "bio_judgements.jsonl", {"key": cand["key"], "verdict": verdict, "split": cand["split"]})
        if calls % 20 == 0:
            log(f"[judge] {calls}/{len(st['pending'])} {dict(verdicts)} prompt {server.speed()['prompt_tokens_per_s']} "
                f"tok/s {time.time() - t0:.0f}s")
    run = {"phase": "judge", "calls": calls, "verdicts": dict(verdicts), "seconds": round(time.time() - t0, 1),
           "speed": server.speed()}
    receipt = write_bio_receipt(data_dir, run=run)
    log(f"[judge] done {dict(verdicts)} rows {receipt['rows']}")
    return receipt


def write_bio_receipt(data_dir: Path, run: Optional[dict] = None, target: int = 0, heldout: int = 0) -> dict:
    data_dir = Path(data_dir)
    st = _bio_state(data_dir)
    path = data_dir / "bio_rows.receipt.json"
    old = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    cands = st["candidates"]
    judg = st["judgements"]
    receipt = {
        "schema": "expanse-teacher-bio-v1",
        "teacher": {"name": "BioMedLM 2.7B (stanford-crfm/BioMedLM, Q8_0 GGUF)", "gguf": str(BIOMEDLM_GGUF)},
        "judge": {"name": "Qwen2.5-Coder-7B-Instruct (Q4_K_M GGUF)", "prompt": JUDGE_TEMPLATE, "system": JUDGE_SYSTEM,
                  "rule": "keep definitions judged 'yes' (1 token, temperature 0)"},
        "rows": dict(Counter(r["split"] for r in st["rows"])),
        "rows_by_task": {s: dict(Counter(r["task"] for r in st["rows"] if r["split"] == s)) for s in ("train", "heldout")},
        "targets": {"train": target or old.get("targets", {}).get("train"), "heldout": heldout or old.get("targets", {}).get("heldout")},
        "candidates": {"total": len(cands),
                       "by_kind_reason": dict(Counter(f"{c['kind']}:{c['reason']}" for c in cands).most_common()),
                       "pending_judge": len(st["pending"])},
        "judge_verdicts": dict(Counter(j["verdict"] for j in judg.values())),
        "judge_accept_rate": round(st["accept_rate"], 4),
        "pubmedqa_label_agreement": _pqa_agreement(cands),
        "runs": old.get("runs", []) + ([run] if run else []),
    }
    write_json_atomic(path, receipt)
    return receipt


def _pqa_agreement(cands: List[dict]) -> dict:
    parsed = [c for c in cands if c["kind"] == "pubmedqa" and c.get("decision")]
    agree = sum(1 for c in parsed if c["decision"] == c["final_decision"])
    return {"parsed": len(parsed), "agree": agree, "rate": round(agree / max(1, len(parsed)), 4)}


# ---------------------------------------------------------------------------
# BioMedLM -> GGUF
# ---------------------------------------------------------------------------

#: Runs the pinned llama.cpp converter in its own process. The only change it
#: makes is a fallback in ``get_vocab_base_pre``: llama.cpp identifies a BPE
#: pre-tokenizer by hashing the token ids of a probe string, and BioMedLM's
#: PubMed vocabulary is not in its table. Its tokenizer.json declares the GPT-2
#: ByteLevel pre-tokenizer with the GPT-2 regex, so "gpt-2" is the exact
#: answer; the fallback refuses anything else.
CONVERT_RUNNER = r'''
import json, runpy, sys
src, hf_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, src + "/gguf-py")
sys.path.insert(0, src)
import conversion.base as base
pre = json.load(open(hf_dir + "/tokenizer.json", encoding="utf-8")).get("pre_tokenizer") or {}
_orig = base.TextModel.get_vocab_base_pre
def _patched(self, tokenizer):
    try:
        return _orig(self, tokenizer)
    except NotImplementedError:
        if pre.get("type") == "ByteLevel" and pre.get("use_regex", True):
            print("[expanse] BPE pre-tokenizer hash not in llama.cpp's table; tokenizer.json is GPT-2 ByteLevel "
                  "with the GPT-2 regex -> 'gpt-2'", flush=True)
            return "gpt-2"
        raise
base.TextModel.get_vocab_base_pre = _patched
sys.argv = [src + "/convert_hf_to_gguf.py"] + sys.argv[3:]
runpy.run_path(src + "/convert_hf_to_gguf.py", run_name="__main__")
'''


def wait_for_file(path: Path, max_minutes: float = 40.0, poll_s: float = 60.0,
                  log: Callable[[str], None] = print) -> bool:
    t0 = time.time()
    while not Path(path).exists():
        if time.time() - t0 > max_minutes * 60:
            return False
        log(f"[wait] {path} not there yet ({(time.time() - t0) / 60:.1f} min)")
        time.sleep(min(poll_s, 60.0))
    return True


def ensure_llama_src(dest: Path = LLAMA_SRC, tag: str = LLAMA_TAG, log: Callable[[str], None] = print) -> dict:
    """Sparse, blobless, depth-1 checkout of the converter and gguf-py at the pinned tag."""

    dest = Path(dest)
    if not (dest / "convert_hf_to_gguf.py").exists():
        if not dest.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", "--branch", tag,
                            "https://github.com/ggml-org/llama.cpp", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "init", "--no-cone"], check=True)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", "/convert_hf_to_gguf.py",
                        "/conversion/", "/gguf-py/", "/requirements/", "/requirements.txt", "/LICENSE"], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", tag], check=True)
    commit = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    tag_commit = subprocess.run(["git", "-C", str(dest), "rev-parse", f"{tag}^{{commit}}"], capture_output=True,
                                text=True).stdout.strip()
    log(f"[convert] llama.cpp-src at {commit[:12]} (tag {tag} -> {tag_commit[:12]})")
    return {"path": str(dest), "tag": tag, "commit": commit, "tag_commit": tag_commit}


def convert_biomedlm(*, hf_dir: Path = BIOMEDLM_DIR, out: Path = BIOMEDLM_GGUF, outtype: str = "q8_0",
                     force: bool = False, wait_minutes: float = 40.0, log: Callable[[str], None] = print) -> dict:
    """HF (bf16 safetensors, attention scaling folded) -> Q8_0 GGUF via the pinned converter."""

    out = Path(out)
    receipt_path = out.with_suffix(".receipt.json")
    if out.exists() and receipt_path.exists() and not force:
        log(f"[convert] {out} exists; skipping (use --force to redo)")
        return json.load(open(receipt_path, encoding="utf-8"))
    if not wait_for_file(Path(hf_dir) / "fetch.receipt.json", max_minutes=wait_minutes, log=log):
        raise TimeoutError("BioMedLM fetch receipt did not appear")
    cfg = json.load(open(Path(hf_dir) / "config.json", encoding="utf-8"))
    if cfg.get("scale_attn_by_inverse_layer_idx"):
        raise RuntimeError("config still has scale_attn_by_inverse_layer_idx=true; llama.cpp's gpt2 graph would "
                           "silently ignore it -- the fetch step must fold it first")
    src = ensure_llama_src(log=log)
    partial = out.with_name(out.name + ".partial")
    cmd = [sys.executable, "-X", "utf8", "-c", CONVERT_RUNNER, str(LLAMA_SRC), str(hf_dir), str(hf_dir),
           "--outtype", outtype, "--outfile", str(partial)]
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / "convert_biomedlm.log"
    t0 = time.time()
    with open(log_path, "wb") as lf:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    seconds = round(time.time() - t0, 1)
    if rc != 0 or not partial.exists():
        raise RuntimeError(f"conversion failed (rc={rc}); see {log_path}")
    os.replace(partial, out)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    receipt = {
        "schema": "expanse-biomedlm-gguf-v1", "source_dir": str(hf_dir),
        "fetch_receipt": json.load(open(Path(hf_dir) / "fetch.receipt.json", encoding="utf-8")),
        "llama_cpp_src": src, "outtype": outtype, "out": str(out), "bytes": out.stat().st_size,
        "sha256": sha256_file(out), "seconds": seconds, "log": str(log_path),
        "pre_tokenizer_fallback": "[expanse] BPE pre-tokenizer hash" in log_text,
        "command": " ".join(["python -c CONVERT_RUNNER"] + cmd[5:]),
    }
    write_json_atomic(receipt_path, receipt)
    log(f"[convert] wrote {out} ({receipt['bytes'] / 2**30:.2f} GiB) in {seconds:.0f}s")
    return receipt


#: ~1,700 words of original biomedical prose (written for this project, so it is
#: certainly not in BioMedLM's training data) for the perplexity sanity check.
BIOMED_PPL_TEXT = """\
The human heart is a muscular pump that moves blood through two circuits arranged in series. The right side of the heart receives deoxygenated blood from the systemic veins and pumps it through the pulmonary artery to the lungs, where carbon dioxide diffuses out of the blood and oxygen diffuses in across the thin alveolar membrane. The left side receives oxygenated blood from the pulmonary veins and ejects it into the aorta, from which it is distributed to every organ. Each heartbeat is initiated by pacemaker cells in the sinoatrial node, whose membrane potential drifts slowly toward threshold because of an inward current carried mainly by sodium ions through hyperpolarization-activated channels. When threshold is reached, voltage-gated calcium channels open and an action potential spreads across the atria, pauses briefly at the atrioventricular node, and is then conducted rapidly through the bundle of His and the Purkinje fibers to the ventricular myocardium. The pause at the atrioventricular node allows the atria to finish contracting before the ventricles begin, which improves ventricular filling.

Cardiac output is the product of heart rate and stroke volume. Stroke volume depends on preload, afterload and contractility. According to the Frank-Starling mechanism, a greater end-diastolic volume stretches the myocardial fibers and increases the force of the next contraction, so that the output of the two ventricles remains matched over many beats. Sympathetic stimulation increases both heart rate and contractility through beta-1 adrenergic receptors, which raise intracellular cyclic AMP and enhance calcium entry during the plateau phase of the ventricular action potential. Parasympathetic stimulation through the vagus nerve slows the heart by releasing acetylcholine, which opens potassium channels in nodal cells and makes their resting potential more negative.

Hypertension is defined as a sustained elevation of arterial blood pressure and is one of the most important modifiable risk factors for stroke, myocardial infarction, heart failure and chronic kidney disease. In most patients no single cause can be identified, and the condition is called primary or essential hypertension. Secondary hypertension may result from renal artery stenosis, primary aldosteronism, pheochromocytoma, obstructive sleep apnea or the use of certain drugs. Treatment begins with lifestyle measures such as reducing dietary sodium, losing excess weight, limiting alcohol and increasing physical activity. First-line medications include thiazide diuretics, angiotensin-converting enzyme inhibitors, angiotensin receptor blockers and dihydropyridine calcium channel blockers. Angiotensin-converting enzyme inhibitors block the conversion of angiotensin I to angiotensin II, a potent vasoconstrictor that also stimulates the release of aldosterone from the adrenal cortex. A dry cough is a common side effect because the same enzyme normally degrades bradykinin.

Diabetes mellitus is a group of metabolic disorders characterized by chronic hyperglycemia. Type 1 diabetes results from autoimmune destruction of the insulin-producing beta cells of the pancreatic islets, so affected patients depend on exogenous insulin for survival. Type 2 diabetes is far more common and arises from a combination of insulin resistance in muscle, liver and adipose tissue and a progressive failure of beta cells to compensate. Obesity, physical inactivity and genetic susceptibility all contribute. Glycated hemoglobin reflects the average blood glucose concentration over the preceding two to three months and is widely used both for diagnosis and for monitoring treatment. Metformin reduces hepatic glucose production and is usually the first drug prescribed. Newer agents include inhibitors of the sodium-glucose cotransporter 2, which increase the excretion of glucose in the urine, and agonists of the glucagon-like peptide 1 receptor, which enhance glucose-dependent insulin secretion, slow gastric emptying and reduce appetite. Long-term complications of diabetes are divided into microvascular disease, which affects the retina, kidneys and peripheral nerves, and macrovascular disease, which accelerates atherosclerosis in the coronary, cerebral and peripheral arteries.

The kidneys regulate the volume and composition of the extracellular fluid. Each kidney contains about one million nephrons. Blood is filtered at the glomerulus, a tuft of capillaries surrounded by Bowman's capsule, and the filtrate then passes through the proximal tubule, the loop of Henle, the distal tubule and the collecting duct. The proximal tubule reabsorbs most of the filtered sodium, water, glucose and amino acids. The loop of Henle creates a concentration gradient in the medullary interstitium through countercurrent multiplication, and antidiuretic hormone controls the final concentration of the urine by inserting aquaporin water channels into the membranes of collecting duct cells. The glomerular filtration rate is the best overall index of kidney function and is usually estimated from the serum creatinine concentration together with age and sex. Chronic kidney disease is classified into stages according to the estimated filtration rate and the amount of albumin in the urine.

The immune system protects the body against infection through innate and adaptive mechanisms. Innate immunity includes physical barriers such as the skin and mucous membranes, antimicrobial peptides, the complement system and phagocytic cells such as neutrophils and macrophages. These cells recognize conserved molecular patterns on microbes through pattern recognition receptors, including the toll-like receptors, and respond within minutes to hours. Adaptive immunity is slower to develop but highly specific and capable of memory. B lymphocytes produce antibodies that neutralize toxins and viruses and mark bacteria for destruction, while T lymphocytes recognize peptide fragments presented by major histocompatibility complex molecules. Cytotoxic T cells carry the CD8 coreceptor and kill virus-infected cells, and helper T cells carry the CD4 coreceptor and coordinate the responses of other immune cells by releasing cytokines. Vaccination exploits immunological memory: exposure to an inactivated pathogen, a purified antigen or messenger RNA encoding an antigen primes the adaptive immune system so that a later encounter with the real pathogen produces a faster and stronger response.

Antibiotic resistance is a growing threat to public health. Bacteria can become resistant through mutations in the genes encoding drug targets, through enzymes that inactivate the drug, through efflux pumps that remove the drug from the cell, or through reduced permeability of the outer membrane. Beta-lactamase enzymes, for example, hydrolyze the beta-lactam ring of penicillins and cephalosporins. Resistance genes are often carried on plasmids and can spread between different bacterial species by horizontal gene transfer. Methicillin-resistant Staphylococcus aureus produces an altered penicillin-binding protein with low affinity for beta-lactam antibiotics. Carbapenem-resistant Enterobacteriaceae are particularly worrying because few effective treatment options remain. Strategies to slow the spread of resistance include prescribing antibiotics only when they are needed, choosing the narrowest effective spectrum, completing appropriate courses, improving infection control in hospitals and developing new antimicrobial agents.

Cancer arises when cells accumulate genetic and epigenetic changes that allow them to proliferate without normal restraint. Oncogenes such as mutant forms of RAS drive cell division, whereas tumor suppressor genes such as TP53 and RB1 normally halt the cell cycle or trigger apoptosis in response to DNA damage. Most cancers require several mutations, which helps to explain why incidence rises steeply with age. Malignant tumors invade surrounding tissues and can metastasize through the blood or lymphatic vessels to distant organs. Treatment may involve surgery, radiotherapy, cytotoxic chemotherapy, targeted drugs directed against specific molecular abnormalities, and immunotherapy. Immune checkpoint inhibitors are antibodies that block inhibitory receptors such as PD-1 on T cells or its ligand PD-L1 on tumor cells, releasing a brake on the antitumor immune response. They have produced durable responses in melanoma, lung cancer and several other tumors, but they can also cause autoimmune side effects in the skin, gut, liver and endocrine glands.

The nervous system transmits information through electrical and chemical signals. A neuron at rest maintains a negative membrane potential because the sodium-potassium pump keeps potassium concentrated inside the cell and sodium outside, and because the resting membrane is much more permeable to potassium. When a stimulus depolarizes the axon hillock to threshold, voltage-gated sodium channels open and an action potential propagates along the axon. Myelin, produced by oligodendrocytes in the central nervous system and by Schwann cells in the peripheral nervous system, insulates the axon and allows saltatory conduction between the nodes of Ranvier. At the synapse, the arriving action potential opens voltage-gated calcium channels, and the resulting calcium influx triggers the fusion of synaptic vesicles with the presynaptic membrane and the release of neurotransmitter. Glutamate is the main excitatory neurotransmitter in the brain, and gamma-aminobutyric acid is the main inhibitory one. Multiple sclerosis is an autoimmune disease in which inflammation damages the myelin of the central nervous system, slowing or blocking conduction and producing symptoms such as visual loss, weakness, numbness and problems with balance.

The liver performs hundreds of metabolic functions. It stores glycogen after meals and releases glucose during fasting, synthesizes plasma proteins such as albumin and clotting factors, produces bile to aid the digestion of fats, and metabolizes drugs and toxins, largely through the cytochrome P450 family of enzymes. Chronic liver injury from alcohol, viral hepatitis or fatty liver disease can lead to fibrosis and eventually cirrhosis, in which the normal architecture of the liver is replaced by scar tissue and regenerative nodules. Complications of cirrhosis include portal hypertension, ascites, variceal bleeding, hepatic encephalopathy and an increased risk of hepatocellular carcinoma.
"""


def run_perplexity(gguf: Path = BIOMEDLM_GGUF, text: str = BIOMED_PPL_TEXT, ctx: int = 512, threads: int = 4,
                   data_dir: Path = DATA, log: Callable[[str], None] = print) -> dict:
    """``llama-perplexity`` over our own biomedical text; returns {"ppl", "ppl_err", "chunks", ...}."""

    text_path = Path(data_dir) / "biomed_ppl_text.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    cmd = [str(LLAMA_PERPLEXITY), "-m", str(gguf), "-f", str(text_path), "-c", str(ctx), "-b", str(ctx),
           "-t", str(threads)]
    t0 = time.time()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=str(LLAMA_DIR), creationflags=flags)
    out = proc.stdout + proc.stderr
    (LOGS / "biomedlm_perplexity.log").write_text(out, encoding="utf-8")
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", out)
    chunks = re.search(r"calculating perplexity over (\d+) chunks", out)
    res = {"ppl": float(m.group(1)) if m else None, "ppl_err": float(m.group(2)) if m else None,
           "chunks": int(chunks.group(1)) if chunks else None, "ctx": ctx, "threads": threads,
           "tokens_scored_approx": (int(chunks.group(1)) * ctx // 2) if chunks else None,
           "text_words": word_count(text), "seconds": round(time.time() - t0, 1), "returncode": proc.returncode,
           "text_path": str(text_path)}
    log(f"[ppl] {res}")
    return res


BIO_SAMPLE_PROMPTS = ("Insulin resistance is", "The main function of the kidney is", "Tuberculosis is caused by")
