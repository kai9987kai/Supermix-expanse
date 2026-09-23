import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


FEAT_DIM = 128
MODEL_CLASSES = 10
VALID_FEATURE_MODES = (
    "legacy",
    "context_v2",
    "context_v3",
    "context_v4",
    "context_v5",
    "context_mix_v1",
    "context_mix_v2_mm",
    "context_mix_v3",
    "context_mix_v4",
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[^\w\s]")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
CREATIVE_HINT_RE = re.compile(r"\b(creative|imagine|story|brainstorm|novel|metaphor|analogy|invent)\b", re.I)
CONCISE_HINT_RE = re.compile(r"\b(short|brief|concise|one line|tldr|quick answer)\b", re.I)
ANALYST_HINT_RE = re.compile(r"\b(step by step|tradeoff|analy[sz]e|reason|plan|debug|diagnose)\b", re.I)
FOLLOWUP_EDIT_HINT_RE = re.compile(
    r"\b(it|that|this|same|again|continue|deeper|expand|shorter|longer|rewrite|rephrase|refine|improve|make it)\b",
    re.I,
)
AMBIGUOUS_EDIT_RE = re.compile(
    r"\b(make it better|improve it|fix this|change it|same but better|more like that|do that again)\b",
    re.I,
)
SHORTEN_REQUEST_RE = re.compile(r"\b(shorter|brief|concise|trim|compress|tldr|one line|one sentence)\b", re.I)
EXPAND_REQUEST_RE = re.compile(r"\b(deeper|expand|elaborate|more detail|longer|unpack|walk through)\b", re.I)
REWRITE_REQUEST_RE = re.compile(r"\b(rewrite|rephrase|clearer|polish|fix the wording|paraphrase)\b", re.I)
CONTINUE_REQUEST_RE = re.compile(r"\b(continue|go on|keep going|next part|what next)\b", re.I)
REASONING_REQUEST_RE = re.compile(r"\b(step by step|why|how|derive|prove|debug|reason|analy[sz]e|tradeoff)\b", re.I)
CREATIVE_REQUEST_RE = re.compile(r"\b(creative|story|metaphor|analogy|brainstorm|invent|vivid|novel)\b", re.I)
NUMBER_RE = re.compile(r"\b-?\d+(?:\.\d+)?\b")
LITERARY_HINT_RE = re.compile(
    r"\b(excerpt|novel|book|chapter|character|theme|tone|narrat|prose|literary|joyce|finnegans|portmanteau|modernist|allusion-heavy)\b",
    re.I,
)
SCIENCE_HINT_RE = re.compile(
    r"\b(science|scientific|physics|chemistry|biology|geology|astronomy|ecology|cell|atom|molecule|energy|force|gravity|photosynthesis|ecosystem|planet|star|weather|climate|experiment|hypothesis)\b",
    re.I,
)
ENGLISH_HINT_RE = re.compile(
    r"\b(english|grammar|punctuation|spelling|vocabulary|sentence|paragraph|essay|rewrite|paraphrase|formal|informal|proofread|edit|thesis|topic sentence)\b",
    re.I,
)
MATH_HINT_RE = re.compile(
    r"\b(math|mathematics|arithmetic|add|subtract|multiply|divide|fraction|decimal|percent|percentage|ratio|algebra|equation|solve|simplify|calculate|average|mean)\b|[0-9]+\s*[%+\-*/=]",
    re.I,
)
SCRIPTURE_REF_RE = re.compile(
    r"\b(?:genesis|exodus|leviticus|numbers|deuteronomy|joshua|judges|ruth|samuel|kings|chronicles|ezra|nehemiah|esther|job|psalm|psalms|proverbs|ecclesiastes|isaiah|jeremiah|ezekiel|daniel|hosea|joel|amos|obadiah|jonah|micah|nahum|habakkuk|zephaniah|haggai|zechariah|malachi|matthew|mark|luke|john|acts|romans|corinthians|galatians|ephesians|philippians|colossians|thessalonians|timothy|titus|philemon|hebrews|james|peter|jude|revelation)\s+\d+(?::\d+(?:-\d+)?)?\b",
    re.I,
)
SCRIPTURE_HINT_RE = re.compile(
    r"\b(bible|scripture|verse|verses|chapter|gospel|psalm|psalms|parable|apostle|prophet|genesis|exodus|john|matthew|romans|revelation|old testament|new testament|kjv)\b",
    re.I,
)
DICTIONARY_HINT_RE = re.compile(
    r"\b(dictionary|define|definition|meaning|meanings|vocabulary|word meaning|synonym|synonyms|antonym|antonyms|part of speech|pronunciation|etymology|lemma)\b",
    re.I,
)
EXACT_LOOKUP_HINT_RE = re.compile(
    r"\b(quote|exact|verbatim|give me the text|text of|what does .* say|definition of|define|meaning of|word meaning|part of speech)\b",
    re.I,
)
CODE_HINT_RE = re.compile(
    r"`[^`]+`|\b(traceback|stack|error|exception|python|sql|api|model|train|javascript|typescript|java|c\+\+|c#|bash|regex|json|yaml|docker|kubernetes)\b",
    re.I,
)
MEDIA_IMAGE_PATH_RE = re.compile(r"(?:^|\|)\s*path=([^|\n]+)")
MEDIA_VIDEO_PATH_RE = re.compile(r"(?:^|\|)\s*video=([^|\n]+)")
MEDIA_3D_PATH_RE = re.compile(r"(?:^|\|)\s*model3d=([^|\n]+)")
ROLE_LINE_RE = re.compile(r"^(user|assistant|system|tool|memory \d+ user|memory \d+ assistant)\s*:\s*(.*)$", re.I)
EXPLICIT_TARGET_RE = re.compile(r"`[^`]{2,}`|\"[^\"]{2,}\"|'[^']{2,}'")
PROGRAMMING_HINT_RE = re.compile(
    r"```|`[^`]+`|\b(def|class|import|from|return|async|await|SELECT|INSERT|UPDATE|DELETE|function|const|let|var|public|private|try|except|catch|lambda|pip|npm|pytest|traceback|stack trace|segfault|nullpointer|keyerror|indexerror)\b",
    re.I,
)
IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
CAMEL_OR_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")
CODE_LINE_SHAPE_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CREATIVE_WORDS = {
    "creative",
    "story",
    "metaphor",
    "analogy",
    "imagine",
    "brainstorm",
    "novel",
    "invent",
    "vivid",
}
ANALYTIC_WORDS = {
    "step",
    "first",
    "second",
    "plan",
    "diagnose",
    "debug",
    "verify",
    "measure",
    "because",
    "therefore",
    "tradeoff",
}
ENGLISH_WORDS = {
    "english",
    "grammar",
    "punctuation",
    "spelling",
    "vocabulary",
    "word",
    "words",
    "sentence",
    "sentences",
    "paragraph",
    "essay",
    "rewrite",
    "paraphrase",
    "proofread",
    "edit",
    "revise",
    "concise",
    "formal",
    "informal",
    "tone",
    "voice",
    "thesis",
    "transition",
    "topic",
    "subject",
    "verb",
    "adjective",
    "adverb",
    "noun",
    "pronoun",
    "preposition",
    "article",
    "tense",
    "plural",
    "singular",
    "synonym",
    "antonym",
    "definition",
    "capitalize",
    "capitalization",
    "comma",
    "period",
    "semicolon",
    "colon",
    "apostrophe",
}
MATH_WORDS = {
    "math",
    "mathematics",
    "arithmetic",
    "number",
    "numbers",
    "calculate",
    "calculation",
    "solve",
    "simplify",
    "equation",
    "algebra",
    "fraction",
    "fractions",
    "decimal",
    "decimals",
    "percent",
    "percentage",
    "ratio",
    "proportion",
    "average",
    "mean",
    "median",
    "sum",
    "difference",
    "product",
    "quotient",
    "multiply",
    "multiplied",
    "divide",
    "divided",
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "remainder",
    "integer",
    "positive",
    "negative",
    "greater",
    "less",
    "equals",
    "units",
    "convert",
    "conversion",
}
DICTIONARY_WORDS = {
    "dictionary",
    "define",
    "definition",
    "meaning",
    "meanings",
    "vocabulary",
    "word",
    "words",
    "synonym",
    "synonyms",
    "antonym",
    "antonyms",
    "lexicon",
    "lemma",
    "pronunciation",
    "etymology",
    "usage",
    "example",
    "examples",
    "noun",
    "verb",
    "adjective",
    "adverb",
}
SCRIPTURE_WORDS = {
    "bible",
    "scripture",
    "scriptures",
    "verse",
    "verses",
    "chapter",
    "chapters",
    "gospel",
    "gospels",
    "psalm",
    "psalms",
    "parable",
    "apostle",
    "apostles",
    "prophet",
    "prophets",
    "covenant",
    "kingdom",
    "grace",
    "mercy",
    "faith",
    "prayer",
    "salvation",
    "lord",
    "god",
    "jesus",
    "christ",
    "kjv",
    "testament",
    "genesis",
    "exodus",
    "isaiah",
    "matthew",
    "john",
    "romans",
    "revelation",
}
CLARIFICATION_PHRASES = (
    "which part",
    "what part",
    "what exactly",
    "what do you want me to",
    "which answer",
    "paste the text",
    "share the text",
    "what topic",
    "what would you like",
    "what are you referring to",
)
SCIENCE_WORDS = {
    "science",
    "scientific",
    "physics",
    "chemistry",
    "biology",
    "geology",
    "earth",
    "astronomy",
    "ecology",
    "experiment",
    "hypothesis",
    "theory",
    "evidence",
    "observation",
    "variable",
    "control",
    "data",
    "measurement",
    "si",
    "unit",
    "units",
    "mass",
    "volume",
    "density",
    "matter",
    "solid",
    "liquid",
    "gas",
    "plasma",
    "atom",
    "proton",
    "neutron",
    "electron",
    "molecule",
    "compound",
    "element",
    "mixture",
    "reaction",
    "chemical",
    "physical",
    "energy",
    "force",
    "motion",
    "velocity",
    "acceleration",
    "gravity",
    "friction",
    "work",
    "power",
    "circuit",
    "current",
    "voltage",
    "resistance",
    "light",
    "sound",
    "wave",
    "frequency",
    "wavelength",
    "cell",
    "dna",
    "gene",
    "organism",
    "tissue",
    "organ",
    "system",
    "photosynthesis",
    "respiration",
    "ecosystem",
    "habitat",
    "species",
    "planet",
    "moon",
    "star",
    "sun",
    "solar",
    "orbit",
    "rotation",
    "revolution",
    "weather",
    "climate",
    "water cycle",
    "rock cycle",
    "cave",
    "tower",
    "pub",
    "Evil",
    "United Kingdom",
    "tractor",
    "WiFi"
    "News",
    "Nitrogen",
    "Oxygen"
}
PROGRAMMING_WORDS = {
    "python",
    "javascript",
    "typescript",
    "java",
    "kotlin",
    "swift",
    "rust",
    "go",
    "golang",
    "ruby",
    "php",
    "c",
    "c++",
    "c#",
    "sql",
    "bash",
    "powershell",
    "shell",
    "html",
    "css",
    "json",
    "yaml",
    "xml",
    "api",
    "http",
    "rest",
    "graphql",
    "docker",
    "kubernetes",
    "git",
    "pip",
    "npm",
    "conda",
    "venv",
    "pytest",
    "unittest",
    "pandas",
    "numpy",
    "torch",
    "tensorflow",
    "exception",
    "traceback",
    "stack",
    "debug",
    "compile",
    "runtime",
    "function",
    "method",
    "class",
    "module",
    "package",
    "import",
    "return",
    "loop",
    "array",
    "list",
    "dict",
    "object",
    "database",
    "schema",
    "query",
    "index",
    "migration",
}
CONTEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "same",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "us",
    "we",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
}


@dataclass
class ConversationExample:
    context: str
    response: str
    label: Optional[int] = None


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _tokens(text: str, max_tokens: int = 384) -> List[str]:
    return TOKEN_RE.findall(text.lower())[:max_tokens]


def _hash_add(vec: torch.Tensor, key: str, weight: float, dim: int = FEAT_DIM, sign_bit: int = 1) -> None:
    h = _stable_hash(key)
    idx = h % dim
    sign = 1.0 if ((h >> sign_bit) & 1) == 0 else -1.0
    vec[idx] += float(weight) * sign


def _identifier_subtokens(token: str, max_parts: int = 8) -> List[str]:
    raw = token.strip("`'\"")
    if not raw:
        return []
    parts: List[str] = []
    for chunk in re.split(r"[_\-./:]+", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        matches = CAMEL_OR_WORD_RE.findall(chunk)
        if matches:
            parts.extend(matches)
        else:
            parts.append(chunk)
    out: List[str] = []
    for p in parts:
        p = p.lower().strip()
        if len(p) >= 2:
            out.append(p)
        if len(out) >= max_parts:
            break
    return out


def _looks_code_like_line(line: str) -> bool:
    s = line.rstrip()
    if not s:
        return False
    if s.startswith("```"):
        return True
    if s.startswith(("    ", "\t")):
        return True
    if re.search(r"\b(def|class|import|from|return|if|else|for|while|try|except|catch|function|const|let|var|SELECT|INSERT|UPDATE|DELETE)\b", s):
        return True
    if s.count("{") + s.count("}") + s.count("(") + s.count(")") + s.count(";") >= 2:
        return True
    if "=>" in s or "::" in s or "->" in s:
        return True
    return False


def _featurize_text_impl(text: str, dim: int = FEAT_DIM) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    toks = _tokens(text)
    if not toks:
        return vec

    # Unigrams
    for tok in toks:
        h = _stable_hash("u|" + tok)
        idx = h % dim
        sign = 1.0 if ((h >> 1) & 1) == 0 else -1.0
        vec[idx] += sign

    # Bigrams add light phrase sensitivity
    for i in range(len(toks) - 1):
        bg = toks[i] + "__" + toks[i + 1]
        h = _stable_hash("b|" + bg)
        idx = h % dim
        sign = 1.0 if ((h >> 2) & 1) == 0 else -1.0
        vec[idx] += 0.75 * sign

    # Domain-aware vocabulary expansion for programming / technical prompts.
    code_hits = 0
    creative_hits = 0
    analytic_hits = 0
    science_hits = 0
    english_hits = 0
    math_hits = 0
    scripture_hits = 0
    dict_hits = 0
    for tok in toks:
        if tok in PROGRAMMING_WORDS:
            code_hits += 1
            _hash_add(vec, "kw|code|" + tok, 1.10, dim=dim, sign_bit=3)
        if tok in CREATIVE_WORDS:
            creative_hits += 1
            _hash_add(vec, "kw|creative|" + tok, 0.55, dim=dim, sign_bit=4)
        if tok in ANALYTIC_WORDS:
            analytic_hits += 1
            _hash_add(vec, "kw|analytic|" + tok, 0.55, dim=dim, sign_bit=5)
        if tok in SCIENCE_WORDS:
            science_hits += 1
            _hash_add(vec, "kw|science|" + tok, 0.82, dim=dim, sign_bit=46)
        if tok in ENGLISH_WORDS:
            english_hits += 1
            _hash_add(vec, "kw|english|" + tok, 0.75, dim=dim, sign_bit=26)
        if tok in MATH_WORDS:
            math_hits += 1
            _hash_add(vec, "kw|math|" + tok, 0.90, dim=dim, sign_bit=27)
        if tok in SCRIPTURE_WORDS:
            scripture_hits += 1
            _hash_add(vec, "kw|scripture|" + tok, 0.80, dim=dim, sign_bit=64)
        if tok in DICTIONARY_WORDS:
            dict_hits += 1
            _hash_add(vec, "kw|dictionary|" + tok, 0.72, dim=dim, sign_bit=62)

    # Identifier subtoken hashing improves coverage of programming vocab without a fixed tokenizer vocab.
    raw_identifiers = IDENTIFIER_TOKEN_RE.findall(text)
    for ident in raw_identifiers[:96]:
        subtoks = _identifier_subtokens(ident, max_parts=8)
        if not subtoks:
            continue
        if len(subtoks) >= 2:
            code_hits += 1
        for sub in subtoks:
            _hash_add(vec, "id|" + sub, 0.40, dim=dim, sign_bit=6)
        if len(ident) >= 5:
            head = ident[:4].lower()
            tail = ident[-4:].lower()
            _hash_add(vec, "idh|" + head, 0.18, dim=dim, sign_bit=7)
            _hash_add(vec, "idt|" + tail, 0.18, dim=dim, sign_bit=8)

    if PROGRAMMING_HINT_RE.search(text):
        code_hits += 2
        _hash_add(vec, "domain|code_hint", 0.90, dim=dim, sign_bit=9)
    if SCIENCE_HINT_RE.search(text):
        science_hits += 2
        _hash_add(vec, "domain|science_hint", 0.78, dim=dim, sign_bit=47)
    if ENGLISH_HINT_RE.search(text):
        english_hits += 2
        _hash_add(vec, "domain|english_hint", 0.70, dim=dim, sign_bit=28)
    if MATH_HINT_RE.search(text):
        math_hits += 2
        _hash_add(vec, "domain|math_hint", 0.85, dim=dim, sign_bit=29)
    if SCRIPTURE_HINT_RE.search(text) or SCRIPTURE_REF_RE.search(text):
        scripture_hits += 2
        _hash_add(vec, "domain|scripture_hint", 0.78, dim=dim, sign_bit=65)
    if DICTIONARY_HINT_RE.search(text):
        dict_hits += 2
        _hash_add(vec, "domain|dictionary_hint", 0.72, dim=dim, sign_bit=63)

    # Add lightweight code-line structure features (shape hashing) for snippets and stack traces.
    code_like_lines = 0
    for line in (text or "").splitlines()[:16]:
        if not _looks_code_like_line(line):
            continue
        code_like_lines += 1
        shape = line.strip()
        shape = CODE_LINE_SHAPE_WORD_RE.sub("ID", shape)
        shape = re.sub(r"\d+", "N", shape)
        shape = re.sub(r"\"[^\"]*\"|'[^']*'", "STR", shape)
        shape = re.sub(r"\s+", " ", shape)[:96]
        if shape:
            _hash_add(vec, "line|" + shape, 0.65, dim=dim, sign_bit=10)

    if code_hits > 0 or code_like_lines > 0:
        _hash_add(vec, "domain|code", min(1.25, 0.18 * (code_hits + code_like_lines)), dim=dim, sign_bit=11)
    if science_hits > 0:
        _hash_add(vec, "domain|science", min(1.0, 0.15 * science_hits), dim=dim, sign_bit=48)
    if english_hits > 0:
        _hash_add(vec, "domain|english", min(0.95, 0.14 * english_hits), dim=dim, sign_bit=30)
    if math_hits > 0:
        _hash_add(vec, "domain|math", min(1.05, 0.16 * math_hits), dim=dim, sign_bit=31)
    if scripture_hits > 0:
        _hash_add(vec, "domain|scripture", min(1.0, 0.15 * scripture_hits), dim=dim, sign_bit=66)
    if dict_hits > 0:
        _hash_add(vec, "domain|dictionary", min(0.95, 0.14 * dict_hits), dim=dim, sign_bit=53)
    if creative_hits > 0:
        _hash_add(vec, "domain|creative", min(0.70, 0.12 * creative_hits), dim=dim, sign_bit=12)
    if analytic_hits > 0:
        _hash_add(vec, "domain|analytic", min(0.70, 0.12 * analytic_hits), dim=dim, sign_bit=13)

    n = torch.norm(vec, p=2)
    if n > 0:
        vec = vec / n
    return vec


@lru_cache(maxsize=50000)
def _cached_feat_tuple(text: str) -> Tuple[float, ...]:
    vec = _featurize_text_impl(text, dim=FEAT_DIM)
    return tuple(float(v) for v in vec.tolist())


def featurize_text(text: str, dim: int = FEAT_DIM) -> torch.Tensor:
    clean = (text or "").strip()
    if not clean:
        return torch.zeros(dim, dtype=torch.float32)
    if dim == FEAT_DIM:
        return torch.tensor(_cached_feat_tuple(clean), dtype=torch.float32)
    return _featurize_text_impl(clean, dim=dim)


def _split_role_line(line: str) -> Tuple[str, str]:
    m = ROLE_LINE_RE.match(line.strip())
    if not m:
        return "text", line.strip()
    role = str(m.group(1)).strip().lower()
    content = str(m.group(2)).strip()
    return role, content


def featurize_context_v2(context_text: str, dim: int = FEAT_DIM, max_lines: int = 32) -> torch.Tensor:
    """
    Conversation-aware context featurizer:
    - role-sensitive encoding (user/assistant/system/tool/memory)
    - recency weighting on latest lines
    - intent hints (question + imperative markers)
    """
    raw = (context_text or "").strip()
    if not raw:
        return torch.zeros(dim, dtype=torch.float32)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return featurize_text(raw, dim=dim)
    lines = lines[-max_lines:]

    role_weights = {
        "user": 1.20,
        "assistant": 0.95,
        "system": 0.85,
        "tool": 0.88,
        "memory_user": 1.05,
        "memory_assistant": 0.90,
        "text": 1.00,
    }
    acc = torch.zeros(dim, dtype=torch.float32)
    n = len(lines)
    question_boost = featurize_text("[question_intent]", dim=dim)
    action_boost = featurize_text("[action_request]", dim=dim)

    for idx, line in enumerate(lines):
        role, content = _split_role_line(line)
        if not content:
            continue

        role_key = role
        if role.startswith("memory ") and " user" in role:
            role_key = "memory_user"
        elif role.startswith("memory ") and " assistant" in role:
            role_key = "memory_assistant"
        if role_key not in role_weights:
            role_key = "text"

        # Emphasize recent lines to improve conversation state tracking.
        recency = 0.55 + 0.45 * float(idx + 1) / float(max(1, n))
        weight = recency * float(role_weights[role_key])
        acc += weight * featurize_text(f"[{role_key}] {content}", dim=dim)

        lower = content.lower()
        if "?" in content:
            acc += 0.035 * question_boost
        if re.search(r"\b(please|can you|could you|help|build|fix|make|optimi[sz]e)\b", lower):
            acc += 0.030 * action_boost

    norm = torch.norm(acc, p=2)
    if norm > 0:
        acc = acc / norm
    return acc


def featurize_context_v3(context_text: str, dim: int = FEAT_DIM, max_lines: int = 40) -> torch.Tensor:
    """
    Stronger conversation-state encoder:
    - starts from v2 role/recency encoding
    - adds role-transition and turn-position features
    - boosts follow-up/coreference cues using recent dialogue structure
    """
    raw = (context_text or "").strip()
    if not raw:
        return torch.zeros(dim, dtype=torch.float32)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return featurize_text(raw, dim=dim)
    lines = lines[-max_lines:]

    # Base encoder keeps backwards-compatible behavior while adding richer dialogue signals.
    acc = 0.72 * featurize_context_v2("\n".join(lines), dim=dim, max_lines=max_lines)
    n = len(lines)
    parsed: List[Tuple[str, str]] = []
    for line in lines:
        role, content = _split_role_line(line)
        role_key = role
        if role.startswith("memory ") and " user" in role:
            role_key = "memory_user"
        elif role.startswith("memory ") and " assistant" in role:
            role_key = "memory_assistant"
        if role_key not in {"user", "assistant", "system", "tool", "memory_user", "memory_assistant"}:
            role_key = "text"
        parsed.append((role_key, content))

    prev_role: Optional[str] = None
    user_turns = 0
    assistant_turns = 0
    latest_user = ""
    latest_assistant = ""
    question_count = 0
    imperative_count = 0

    for i, (role_key, content) in enumerate(parsed):
        if not content:
            continue
        recency = 0.45 + 0.55 * float(i + 1) / float(max(1, n))
        pos_bin = min(5, int(6 * float(i) / float(max(1, n))))
        token_count = len(_tokens(content, max_tokens=96))
        len_bin = "short" if token_count <= 10 else ("med" if token_count <= 30 else "long")

        _hash_add(acc, f"ctxv3|role_pos|{role_key}|p{pos_bin}", 0.14 * recency, dim=dim, sign_bit=14)
        _hash_add(acc, f"ctxv3|len|{role_key}|{len_bin}", 0.10 * recency, dim=dim, sign_bit=15)

        if prev_role is not None:
            _hash_add(acc, f"ctxv3|trans|{prev_role}>{role_key}", 0.22 * recency, dim=dim, sign_bit=16)
        prev_role = role_key

        lower = content.lower()
        if role_key in {"user", "memory_user"}:
            user_turns += 1
            latest_user = content
            if "?" in content:
                question_count += 1
            if re.search(r"\b(please|can you|could you|help|make|add|improve|continue|explain|rewrite|optimi[sz]e)\b", lower):
                imperative_count += 1
        elif role_key in {"assistant", "memory_assistant"}:
            assistant_turns += 1
            latest_assistant = content

        if LITERARY_HINT_RE.search(content):
            _hash_add(acc, "ctxv3|domain|literary", 0.16 * recency, dim=dim, sign_bit=17)
        if PROGRAMMING_HINT_RE.search(content):
            _hash_add(acc, "ctxv3|domain|code", 0.18 * recency, dim=dim, sign_bit=18)

    if latest_user:
        user_lower = latest_user.lower()
        if re.search(r"\b(it|that|this|they|them|he|she|those|these|same|previous|above)\b", user_lower):
            _hash_add(acc, "ctxv3|followup|coref", 0.28, dim=dim, sign_bit=19)
            if latest_assistant:
                acc += 0.12 * featurize_text("[recent_assistant] " + latest_assistant, dim=dim)
        if re.search(r"\b(continue|go on|more|deeper|expand|another|next)\b", user_lower):
            _hash_add(acc, "ctxv3|followup|continue", 0.24, dim=dim, sign_bit=20)
        if LITERARY_HINT_RE.search(latest_user):
            _hash_add(acc, "ctxv3|latest_user|literary", 0.22, dim=dim, sign_bit=21)
        if PROGRAMMING_HINT_RE.search(latest_user):
            _hash_add(acc, "ctxv3|latest_user|code", 0.24, dim=dim, sign_bit=22)

    dialogue_balance = float(min(user_turns, assistant_turns)) / float(max(1, max(user_turns, assistant_turns)))
    _hash_add(acc, f"ctxv3|balance|b{int(dialogue_balance * 4)}", 0.12, dim=dim, sign_bit=23)
    if question_count > 0:
        _hash_add(acc, f"ctxv3|question_count|{min(3, question_count)}", 0.10, dim=dim, sign_bit=24)
    if imperative_count > 0:
        _hash_add(acc, f"ctxv3|imperative_count|{min(3, imperative_count)}", 0.10, dim=dim, sign_bit=25)

    norm = torch.norm(acc, p=2)
    if norm > 0:
        acc = acc / norm
    return acc


def featurize_context_v4(context_text: str, dim: int = FEAT_DIM, max_lines: int = 48) -> torch.Tensor:
    """
    Conversation+task-aware encoder:
    - builds on v3 dialogue signals
    - adds domain continuity (english/math/literary/code)
    - emphasizes latest user task intent (fix/rewrite/solve/explain/continue)
    """
    raw = (context_text or "").strip()
    if not raw:
        return torch.zeros(dim, dtype=torch.float32)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return featurize_text(raw, dim=dim)
    lines = lines[-max_lines:]

    acc = 0.76 * featurize_context_v3("\n".join(lines), dim=dim, max_lines=max_lines)

    parsed: List[Tuple[str, str]] = []
    for line in lines:
        role, content = _split_role_line(line)
        role_key = role
        if role.startswith("memory ") and " user" in role:
            role_key = "memory_user"
        elif role.startswith("memory ") and " assistant" in role:
            role_key = "memory_assistant"
        if role_key not in {"user", "assistant", "system", "tool", "memory_user", "memory_assistant"}:
            role_key = "text"
        parsed.append((role_key, content))

    domain_counts = {"code": 0, "literary": 0, "science": 0, "english": 0, "math": 0, "scripture": 0}
    latest_user = ""
    latest_assistant = ""
    n = len(parsed)

    for i, (role_key, content) in enumerate(parsed):
        if not content:
            continue
        recency = 0.40 + 0.60 * float(i + 1) / float(max(1, n))
        lower = content.lower()

        if PROGRAMMING_HINT_RE.search(content):
            domain_counts["code"] += 1
            _hash_add(acc, f"ctxv4|domain_line|{role_key}|code", 0.12 * recency, dim=dim, sign_bit=32)
        if LITERARY_HINT_RE.search(content):
            domain_counts["literary"] += 1
            _hash_add(acc, f"ctxv4|domain_line|{role_key}|literary", 0.11 * recency, dim=dim, sign_bit=33)
        if SCIENCE_HINT_RE.search(content):
            domain_counts["science"] += 1
            _hash_add(acc, f"ctxv4|domain_line|{role_key}|science", 0.12 * recency, dim=dim, sign_bit=49)
        if ENGLISH_HINT_RE.search(content):
            domain_counts["english"] += 1
            _hash_add(acc, f"ctxv4|domain_line|{role_key}|english", 0.12 * recency, dim=dim, sign_bit=34)
        if MATH_HINT_RE.search(content):
            domain_counts["math"] += 1
            _hash_add(acc, f"ctxv4|domain_line|{role_key}|math", 0.13 * recency, dim=dim, sign_bit=35)
        if SCRIPTURE_HINT_RE.search(content) or SCRIPTURE_REF_RE.search(content):
            domain_counts["scripture"] += 1
            _hash_add(acc, f"ctxv4|domain_line|{role_key}|scripture", 0.12 * recency, dim=dim, sign_bit=67)

        if role_key in {"user", "memory_user"}:
            latest_user = content
            if re.search(r"\b(fix|correct|proofread|rewrite|rephrase|summarize|expand|continue|translate)\b", lower):
                _hash_add(acc, "ctxv4|intent|language_edit", 0.18 * recency, dim=dim, sign_bit=36)
            if re.search(r"\b(solve|calculate|work out|simplify|show steps|equation|fraction|percent)\b", lower):
                _hash_add(acc, "ctxv4|intent|math_solve", 0.22 * recency, dim=dim, sign_bit=37)
            if re.search(r"\b(write|draft|email|essay|paragraph|sentence)\b", lower):
                _hash_add(acc, "ctxv4|intent|writing_task", 0.16 * recency, dim=dim, sign_bit=38)
            if re.search(r"\b(bible|scripture|verse|chapter|gospel|psalm|parable|meaning of .*:\d+)\b", lower) or SCRIPTURE_REF_RE.search(content):
                _hash_add(acc, "ctxv4|intent|scripture_lookup", 0.20 * recency, dim=dim, sign_bit=68)
        if re.search(r"\b(code|python|javascript|bug|error|stack trace)\b", lower):
            _hash_add(acc, "ctxv4|intent|code_task", 0.18 * recency, dim=dim, sign_bit=39)
        if re.search(r"\b(science|physics|chemistry|biology|experiment|explain why|concept)\b", lower):
            _hash_add(acc, "ctxv4|intent|science_explain", 0.18 * recency, dim=dim, sign_bit=50)
        if re.search(r"\b(prayer|devotional|scripture reflection|bible study|verse meaning)\b", lower):
            _hash_add(acc, "ctxv4|intent|scripture_reflection", 0.16 * recency, dim=dim, sign_bit=69)
        if role_key in {"assistant", "memory_assistant"}:
            latest_assistant = content

        token_count = len(_tokens(content, max_tokens=128))
        if token_count >= 40:
            _hash_add(acc, f"ctxv4|len_bucket|{role_key}|long", 0.08 * recency, dim=dim, sign_bit=40)

    if latest_user:
        u = latest_user.lower()
        if latest_assistant:
            # Encourage continuity for follow-up edits ("make it shorter", "fix grammar", "check math").
            if re.search(r"\b(make it|shorter|longer|more creative|clearer|better|fix that|correct that|check again)\b", u):
                _hash_add(acc, "ctxv4|followup|refine_prior_answer", 0.24, dim=dim, sign_bit=41)
                acc += 0.10 * featurize_text("[assistant_context] " + latest_assistant[:320], dim=dim)
        if re.search(r"\b(why|because|explain|show steps|how)\b", u):
            _hash_add(acc, "ctxv4|needs_explanation", 0.16, dim=dim, sign_bit=42)
        if re.search(r"\b(example|sample|template)\b", u):
            _hash_add(acc, "ctxv4|needs_example", 0.14, dim=dim, sign_bit=43)

    for domain, count in domain_counts.items():
        if count > 0:
            _hash_add(acc, f"ctxv4|domain_total|{domain}|{min(4, count)}", 0.10, dim=dim, sign_bit=44)

    # Cross-domain hybrid prompts are common ("write a clearer explanation of this math").
    active_domains = [k for k, v in domain_counts.items() if v > 0]
    if len(active_domains) >= 2:
        pair_key = "|".join(sorted(active_domains)[:3])
        _hash_add(acc, "ctxv4|multidomain|" + pair_key, 0.16, dim=dim, sign_bit=45)

    norm = torch.norm(acc, p=2)
    if norm > 0:
        acc = acc / norm
    return acc


def featurize_context_v5(context_text: str, dim: int = FEAT_DIM, max_lines: int = 56) -> torch.Tensor:
    """
    Reasoning-chain & empathy-aware encoder:
    - builds on v4 task/domain signals
    - detects chain-of-thought patterns (step-by-step, numbered reasoning)
    - recognizes analogy usage ("like", "imagine", "think of it as")
    - detects debate/argument structure (pro/con, "on the other hand")
    - recognizes empathetic / emotionally supportive language
    - adds conversation-depth awareness (long multi-turn vs. single-shot)
    """
    raw = (context_text or "").strip()
    if not raw:
        return torch.zeros(dim, dtype=torch.float32)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return featurize_text(raw, dim=dim)
    lines = lines[-max_lines:]

    acc = 0.78 * featurize_context_v4("\n".join(lines), dim=dim, max_lines=max_lines)

    parsed: List[Tuple[str, str]] = []
    for line in lines:
        role, content = _split_role_line(line)
        role_key = role
        if role.startswith("memory ") and " user" in role:
            role_key = "memory_user"
        elif role.startswith("memory ") and " assistant" in role:
            role_key = "memory_assistant"
        if role_key not in {"user", "assistant", "system", "tool", "memory_user", "memory_assistant"}:
            role_key = "text"
        parsed.append((role_key, content))

    n = len(parsed)
    cot_signals = 0
    analogy_signals = 0
    debate_signals = 0
    empathy_signals = 0
    depth_count = 0
    latest_user = ""

    for i, (role_key, content) in enumerate(parsed):
        if not content:
            continue
        lower = content.lower()
        recency = 0.35 + 0.65 * float(i + 1) / float(max(1, n))
        depth_count += 1

        if role_key == "system":
            if lower.startswith("conversation_tags="):
                tag_blob = lower.split("=", 1)[1]
                for tag in [t.strip() for t in tag_blob.split(",") if t.strip()]:
                    _hash_add(acc, f"ctxv5|ctrl|{tag}", 0.16 * recency, dim=dim, sign_bit=78)
            if lower.startswith("topic_terms="):
                acc += 0.08 * featurize_text("[topic_terms] " + content[:180], dim=dim)
            if lower.startswith("last_assistant_focus="):
                acc += 0.09 * featurize_text("[assistant_focus] " + content[:180], dim=dim)

        # Chain-of-thought detection
        if re.search(r"\b(step\s*\d|first|second|third|therefore|because|so\b|thus|hence|consequently)\b", lower):
            cot_signals += 1
            _hash_add(acc, f"ctxv5|cot|{role_key}", 0.18 * recency, dim=dim, sign_bit=64)
        if re.search(r"^\s*\d+[.)]", content):
            cot_signals += 1
            _hash_add(acc, f"ctxv5|numbered_steps|{role_key}", 0.14 * recency, dim=dim, sign_bit=65)

        # Analogy detection
        if re.search(r"\b(like|imagine|think of it as|analogy|metaphor|picture this|similar to)\b", lower):
            analogy_signals += 1
            _hash_add(acc, f"ctxv5|analogy|{role_key}", 0.16 * recency, dim=dim, sign_bit=66)

        # Debate/argument detection
        if re.search(r"\b(on the other hand|however|conversely|pro|con|argument|counterargument|in favor|against)\b", lower):
            debate_signals += 1
            _hash_add(acc, f"ctxv5|debate|{role_key}", 0.14 * recency, dim=dim, sign_bit=67)

        # Empathy detection
        if re.search(r"\b(feel|feeling|sorry|understand|support|care|empathy|comfort|tough|hard time|overwhelm|stress)\b", lower):
            empathy_signals += 1
            _hash_add(acc, f"ctxv5|empathy|{role_key}", 0.15 * recency, dim=dim, sign_bit=68)

        if role_key in {"user", "memory_user"}:
            latest_user = content

    # Global signal injection
    if cot_signals >= 2:
        _hash_add(acc, "ctxv5|mode|reasoning_heavy", 0.22, dim=dim, sign_bit=69)
    if analogy_signals >= 1:
        _hash_add(acc, "ctxv5|mode|analogy_active", 0.18, dim=dim, sign_bit=70)
    if debate_signals >= 1:
        _hash_add(acc, "ctxv5|mode|debate_active", 0.16, dim=dim, sign_bit=71)
    if empathy_signals >= 1:
        _hash_add(acc, "ctxv5|mode|empathy_active", 0.20, dim=dim, sign_bit=72)

    # Conversation depth awareness
    depth_bin = min(5, depth_count // 4)
    _hash_add(acc, f"ctxv5|depth|d{depth_bin}", 0.12, dim=dim, sign_bit=73)

    # Latest user intent refinements
    if latest_user:
        u = latest_user.lower()
        if re.search(r"\b(explain|why|how does|what is|define|describe)\b", u):
            _hash_add(acc, "ctxv5|intent|explanation", 0.20, dim=dim, sign_bit=74)
        if re.search(r"\b(story|continue|write|creative|poem|fiction)\b", u):
            _hash_add(acc, "ctxv5|intent|creative_gen", 0.18, dim=dim, sign_bit=75)
        if re.search(r"\b(debate|argue|pros? and cons?|both sides)\b", u):
            _hash_add(acc, "ctxv5|intent|debate_req", 0.18, dim=dim, sign_bit=76)
        if re.search(r"\b(help|scared|anxious|sad|upset|lonely|stressed|overwhelmed)\b", u):
            _hash_add(acc, "ctxv5|intent|emotional_support", 0.22, dim=dim, sign_bit=77)

    norm = torch.norm(acc, p=2)
    if norm > 0:
        acc = acc / norm
    return acc


def featurize_context_mix_v1(context_text: str, dim: int = FEAT_DIM) -> torch.Tensor:
    """
    Multi-view text encoder (lightweight "multi-model" training aspect):
    blends legacy/v2/v3/v4/v5 features with domain-adaptive weighting.
    """
    raw = (context_text or "").strip()
    if not raw:
        return torch.zeros(dim, dtype=torch.float32)

    f0 = featurize_text(raw, dim=dim)
    f2 = featurize_context_v2(raw, dim=dim)
    f3 = featurize_context_v3(raw, dim=dim)
    f4 = featurize_context_v4(raw, dim=dim)
    f5 = featurize_context_v5(raw, dim=dim)

    flags = _domain_flags(raw)
    w0, w2, w3, w4, w5 = 0.12, 0.18, 0.22, 0.24, 0.24
    if flags["code"]:
        w3 += 0.04
        w4 += 0.05
        w0 -= 0.03
    if flags["english"] or flags["math"]:
        w4 += 0.06
        w5 += 0.04
        w2 -= 0.03
    if flags["literary"]:
        w3 += 0.04
        w5 += 0.05
    total = max(1e-6, w0 + w2 + w3 + w4 + w5)
    acc = (w0 * f0 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5) / total
    n = torch.norm(acc, p=2)
    if n > 0:
        acc = acc / n
    return acc


def _project_numeric_features(prefix: str, values: Sequence[float], dim: int = FEAT_DIM) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    for i, raw_v in enumerate(values):
        try:
            v = float(raw_v)
        except Exception:
            continue
        # Keep the projection numerically stable and bounded.
        v = max(-4.0, min(4.0, v))
        vv = math.tanh(v)
        if abs(vv) < 1e-9:
            continue
        for proj in range(3):
            h = _stable_hash(f"mm|{prefix}|{i}|{proj}")
            idx = h % dim
            sign = 1.0 if ((h >> (proj + 1)) & 1) == 0 else -1.0
            vec[idx] += float(0.50 + 0.15 * proj) * vv * sign
        # Add quantized buckets for coarse invariance.
        q = int(round(vv * 8.0))
        _hash_add(vec, f"mmq|{prefix}|{i}|{q}", 0.18, dim=dim, sign_bit=51 + (i % 7))
    n = torch.norm(vec, p=2)
    if n > 0:
        vec = vec / n
    return vec


def _media_paths_from_context_text(context_text: str) -> Dict[str, str]:
    # Extract media paths (image, video, 3D model) from context text containing [Media] tag
    text = context_text or ""
    if "[Media]" not in text:
        return {}
    out: Dict[str, str] = {}
    
    # Search for image path using MEDIA_IMAGE_PATH_RE regex pattern
    m = MEDIA_IMAGE_PATH_RE.search(text)
    if m:
        out["image"] = m.group(1).strip()
    
    # Search for video path using MEDIA_VIDEO_PATH_RE regex pattern
    m = MEDIA_VIDEO_PATH_RE.search(text)
    if m:
        out["video"] = m.group(1).strip()
    
    # Search for 3D model path using MEDIA_3D_PATH_RE regex pattern
    m = MEDIA_3D_PATH_RE.search(text)
    if m:
        out["model3d"] = m.group(1).strip()
    return out


@lru_cache(maxsize=8192)
def _cached_media_feat_tuple(context_text: str) -> Tuple[float, ...]:
    paths = _media_paths_from_context_text(context_text)
    if not paths:
        return tuple(0.0 for _ in range(FEAT_DIM))

    acc = torch.zeros(FEAT_DIM, dtype=torch.float32)
    try:
        if paths.get("image"):
            from image_feature_utils import extract_image_numeric_features  # lazy import

            acc += 0.34 * _project_numeric_features("image", extract_image_numeric_features(paths["image"]), dim=FEAT_DIM)
            _hash_add(acc, "mm|has_image", 0.20, dim=FEAT_DIM, sign_bit=58)
    except Exception:
        pass
    try:
        if paths.get("video"):
            from video_feature_utils import extract_video_numeric_features  # lazy import

            acc += 0.34 * _project_numeric_features("video", extract_video_numeric_features(paths["video"]), dim=FEAT_DIM)
            _hash_add(acc, "mm|has_video", 0.20, dim=FEAT_DIM, sign_bit=59)
    except Exception:
        pass
    try:
        if paths.get("model3d"):
            from mesh_feature_utils import extract_3d_model_numeric_features  # lazy import

            acc += 0.34 * _project_numeric_features("model3d", extract_3d_model_numeric_features(paths["model3d"]), dim=FEAT_DIM)
            _hash_add(acc, "mm|has_3d", 0.20, dim=FEAT_DIM, sign_bit=60)
    except Exception:
        pass

    # Cross-media tokens help differentiate single-modality vs mixed examples.
    if len(paths) >= 2:
        kinds = "|".join(sorted(paths.keys()))
        _hash_add(acc, "mm|combo|" + kinds, 0.22, dim=FEAT_DIM, sign_bit=61)

    n = torch.norm(acc, p=2)
    if n > 0:
        acc = acc / n
    return tuple(float(v) for v in acc.tolist())


def featurize_context_mix_v2_mm(context_text: str, dim: int = FEAT_DIM) -> torch.Tensor:
    """
    Multi-view + multimodal fusion encoder.
    Uses the text multi-view encoder plus a dedicated numeric media encoder path
    for local image/video/3D rows when present.
    """
    raw = (context_text or "").strip()
    if not raw:
        return torch.zeros(dim, dtype=torch.float32)

    base = featurize_context_mix_v1(raw, dim=dim)
    if dim != FEAT_DIM:
        # Numeric media fusion path is implemented for the canonical feature width.
        return base

    media_vec = torch.tensor(_cached_media_feat_tuple(raw), dtype=torch.float32)
    media_norm = torch.norm(media_vec, p=2).item()
    if media_norm <= 1e-8:
        return base

    flags = _domain_flags(raw)
    base_w = 0.78
    media_w = 0.30
    if flags["science"]:
        media_w += 0.04
    if flags["literary"] or flags["english"]:
        base_w += 0.02
    acc = base_w * base + media_w * media_vec
    if flags["science"] and ("video=" in raw or "model3d=" in raw or "path=" in raw):
        acc += 0.06 * featurize_text("[multimodal_science_example]", dim=dim)
    n = torch.norm(acc, p=2)
    if n > 0:
        acc = acc / n
    return acc


def featurize_context_mix_v3(context_text: str, dim: int = FEAT_DIM) -> torch.Tensor:
    """
    Smarter runtime context encoder:
    - starts from multimodal/context mix v2
    - reads control tags injected by build_context()
    - adds explicit task/edit intent anchors for follow-up requests
    """
    raw = (context_text or "").strip()
    if not raw:
        return torch.zeros(dim, dtype=torch.float32)

    base = featurize_context_mix_v2_mm(raw, dim=dim)
    if dim != FEAT_DIM:
        return base

    acc = 0.76 * base + 0.20 * featurize_context_v5(raw, dim=dim)
    latest_user = ""
    latest_assistant = ""
    tags: List[str] = []
    ambiguity_tags: List[str] = []
    topic_terms: List[str] = []
    assistant_focus: List[str] = []
    previous_request = ""
    expected_act = ""

    for line in [ln.strip() for ln in raw.splitlines() if ln.strip()]:
        role, content = _split_role_line(line)
        low = content.lower()
        if role == "system":
            if low.startswith("conversation_tags="):
                tags = [t.strip() for t in low.split("=", 1)[1].split(",") if t.strip()]
            elif low.startswith("ambiguity_tags="):
                ambiguity_tags = [t.strip() for t in low.split("=", 1)[1].split(",") if t.strip()]
            elif low.startswith("expected_act="):
                expected_act = content.split("=", 1)[1].strip().lower()
            elif low.startswith("topic_terms="):
                topic_terms = [t.strip() for t in content.split("=", 1)[1].split(",") if t.strip()]
            elif low.startswith("last_assistant_focus="):
                assistant_focus = [t.strip() for t in content.split("=", 1)[1].split(",") if t.strip()]
            elif low.startswith("previous_user_request="):
                previous_request = content.split("=", 1)[1].strip()
            continue
        if role == "user" and content:
            latest_user = content
        elif role == "assistant" and content:
            latest_assistant = content

    for tag in tags:
        acc += 0.035 * featurize_text(f"[ctx_tag] {tag}", dim=dim)
    for tag in ambiguity_tags:
        acc += 0.032 * featurize_text(f"[ctx_ambiguity] {tag}", dim=dim)
    if topic_terms:
        acc += 0.09 * featurize_text("[ctx_topic] " + " ".join(topic_terms[:8]), dim=dim)
    if assistant_focus:
        acc += 0.08 * featurize_text("[ctx_focus] " + " ".join(assistant_focus[:6]), dim=dim)
    if previous_request:
        acc += 0.06 * featurize_text("[ctx_prev_user] " + previous_request[:220], dim=dim)
    if expected_act:
        acc += 0.07 * featurize_text("[ctx_expected_act] " + expected_act, dim=dim)
        if expected_act == "clarify":
            acc += 0.08 * featurize_text("[clarify_request] resolve referent ask precise question", dim=dim)

    latest_user_low = latest_user.lower()
    if "followup" in tags and latest_assistant:
        acc += 0.10 * featurize_text("[followup_anchor] " + latest_assistant[:240], dim=dim)
    if "shorten" in tags:
        acc += 0.08 * featurize_text("[shorten_request] brief concise shorter trim", dim=dim)
    if "expand" in tags or "continue" in tags:
        acc += 0.08 * featurize_text("[expand_request] deeper elaborate continue explain more", dim=dim)
    if "rewrite" in tags:
        acc += 0.08 * featurize_text("[rewrite_request] clearer rewrite rephrase polish", dim=dim)
    if "reasoning" in tags or REASONING_REQUEST_RE.search(latest_user_low):
        acc += 0.09 * featurize_text("[reasoning_request] step by step because therefore verify", dim=dim)
    if "creative" in tags or CREATIVE_REQUEST_RE.search(latest_user_low):
        acc += 0.09 * featurize_text("[creative_request] analogy metaphor vivid story imagine", dim=dim)

    flags = _domain_flags(raw)
    if flags["code"]:
        acc += 0.05 * featurize_text("[domain_code_focus]", dim=dim)
    if flags["math"] or flags["science"]:
        acc += 0.04 * featurize_text("[domain_reasoning_focus]", dim=dim)
    if flags["english"] or flags["literary"]:
        acc += 0.04 * featurize_text("[domain_writing_focus]", dim=dim)

    n = torch.norm(acc, p=2)
    if n > 0:
        acc = acc / n
    return acc


def featurize_context_mix_v4(context_text: str, dim: int = FEAT_DIM) -> torch.Tensor:
    """
    Frontier control-aware encoder:
    - starts from context_mix_v3
    - reads explicit control tags for reasoning depth, creativity, recency, and source quality
    - adds light anchors for research-heavy and latest-information prompts
    """
    raw = (context_text or "").strip()
    if not raw:
        return torch.zeros(dim, dtype=torch.float32)

    base = featurize_context_mix_v3(raw, dim=dim)
    if dim != FEAT_DIM:
        return base

    acc = 0.78 * base + 0.16 * featurize_context_v5(raw, dim=dim)

    reasoning_budget = ""
    creativity_level = ""
    knowledge_recency = ""
    source_quality = ""
    task_mode = ""
    research_tags: List[str] = []
    latest_user = ""

    for line in [ln.strip() for ln in raw.splitlines() if ln.strip()]:
        role, content = _split_role_line(line)
        low = content.lower()
        if role == "system":
            if low.startswith("reasoning_budget="):
                reasoning_budget = content.split("=", 1)[1].strip().lower()
            elif low.startswith("creativity_level="):
                creativity_level = content.split("=", 1)[1].strip().lower()
            elif low.startswith("knowledge_recency="):
                knowledge_recency = content.split("=", 1)[1].strip().lower()
            elif low.startswith("source_quality="):
                source_quality = content.split("=", 1)[1].strip().lower()
            elif low.startswith("task_mode="):
                task_mode = content.split("=", 1)[1].strip().lower()
            elif low.startswith("research_tags="):
                research_tags = [tok.strip().lower() for tok in content.split("=", 1)[1].split(",") if tok.strip()]
            continue
        if role == "user" and content:
            latest_user = content

    if reasoning_budget:
        acc += 0.07 * featurize_text(f"[reasoning_budget] {reasoning_budget}", dim=dim)
        if reasoning_budget == "deep":
            acc += 0.06 * featurize_text("[deep_reasoning] decompose verify compare synthesize", dim=dim)
        elif reasoning_budget == "medium":
            acc += 0.04 * featurize_text("[medium_reasoning] explain compare answer", dim=dim)
        elif reasoning_budget == "short":
            acc += 0.04 * featurize_text("[short_reasoning] direct answer concise", dim=dim)

    if creativity_level:
        acc += 0.07 * featurize_text(f"[creativity_level] {creativity_level}", dim=dim)
        if creativity_level == "vivid":
            acc += 0.05 * featurize_text("[creative_style] vivid metaphor analogy scene twist", dim=dim)
        elif creativity_level == "balanced":
            acc += 0.04 * featurize_text("[creative_style] polished engaging but grounded", dim=dim)
        elif creativity_level == "precise":
            acc += 0.04 * featurize_text("[creative_style] precise grounded technical", dim=dim)

    if knowledge_recency:
        acc += 0.07 * featurize_text(f"[knowledge_recency] {knowledge_recency}", dim=dim)
        if knowledge_recency == "latest":
            acc += 0.06 * featurize_text("[latest_info] current recent update newest paper official release", dim=dim)
        elif knowledge_recency == "recent":
            acc += 0.04 * featurize_text("[recent_info] recent official information", dim=dim)
        elif knowledge_recency == "evergreen":
            acc += 0.03 * featurize_text("[evergreen_info] stable concept fundamentals", dim=dim)

    if source_quality:
        acc += 0.06 * featurize_text(f"[source_quality] {source_quality}", dim=dim)
        if source_quality == "research":
            acc += 0.05 * featurize_text("[research_style] empirical result ablation benchmark tradeoff", dim=dim)
        elif source_quality == "official":
            acc += 0.04 * featurize_text("[official_style] documentation exact behavior supported guidance", dim=dim)
        elif source_quality == "synthetic":
            acc += 0.03 * featurize_text("[synthetic_style] generated candidate draft", dim=dim)

    if task_mode:
        acc += 0.07 * featurize_text(f"[task_mode] {task_mode}", dim=dim)
        if task_mode == "knowledge":
            acc += 0.04 * featurize_text("[knowledge_task] facts concepts research docs", dim=dim)
        elif task_mode == "creative":
            acc += 0.04 * featurize_text("[creative_task] brainstorm story concept", dim=dim)
        elif task_mode == "reasoning":
            acc += 0.04 * featurize_text("[reasoning_task] derive debug evaluate", dim=dim)
        elif task_mode == "coding":
            acc += 0.04 * featurize_text("[coding_task] code api bug patch", dim=dim)

    for tag in research_tags[:8]:
        acc += 0.03 * featurize_text(f"[research_tag] {tag}", dim=dim)

    latest_low = latest_user.lower()
    if ("latest" in latest_low or "newest" in latest_low or "research paper" in latest_low or "arxiv" in latest_low):
        acc += 0.05 * featurize_text("[research_latest_request] latest paper current results", dim=dim)
    if "creative" in latest_low and ("paper" in latest_low or "research" in latest_low):
        acc += 0.04 * featurize_text("[creative_research_blend] analogy explain vividly but correctly", dim=dim)

    n = torch.norm(acc, p=2)
    if n > 0:
        acc = acc / n
    return acc


def resolve_feature_mode(feature_mode: str, smarter_auto: bool = False) -> str:
    mode = str(feature_mode or "legacy").strip().lower()
    if mode not in VALID_FEATURE_MODES:
        mode = "legacy"
    if smarter_auto and mode in {"legacy", "context_v2", "context_mix_v3"}:
        return "context_mix_v4"
    return mode


def text_to_model_input(text: str, feature_mode: str = "legacy") -> torch.Tensor:
    # ChampionNet expects (B, T, 128)
    mode = resolve_feature_mode(feature_mode, smarter_auto=False)
    if mode == "context_mix_v4":
        return featurize_context_mix_v4(text).view(1, 1, FEAT_DIM)
    if mode == "context_mix_v2_mm":
        return featurize_context_mix_v2_mm(text).view(1, 1, FEAT_DIM)
    if mode == "context_mix_v3":
        return featurize_context_mix_v3(text).view(1, 1, FEAT_DIM)
    if mode == "context_mix_v1":
        return featurize_context_mix_v1(text).view(1, 1, FEAT_DIM)
    if mode == "context_v5":
        return featurize_context_v5(text).view(1, 1, FEAT_DIM)
    if mode == "context_v4":
        return featurize_context_v4(text).view(1, 1, FEAT_DIM)
    if mode == "context_v3":
        return featurize_context_v3(text).view(1, 1, FEAT_DIM)
    if mode == "context_v2":
        return featurize_context_v2(text).view(1, 1, FEAT_DIM)
    return featurize_text(text).view(1, 1, FEAT_DIM)


def _salient_terms(text: str, max_terms: int = 8) -> List[str]:
    terms: List[str] = []
    seen: Set[str] = set()
    for tok in _tokens(text, max_tokens=192):
        tok = tok.lower().strip()
        if not re.fullmatch(r"[a-z0-9_+\-']+", tok):
            continue
        if tok in CONTEXT_STOPWORDS:
            continue
        if len(tok) < 3 and not any(ch.isdigit() for ch in tok):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms


def _content_term_set(text: str, max_terms: int = 16) -> Set[str]:
    return set(_salient_terms(text, max_terms=max_terms))


def _recent_anchor_text(recent_assistant_messages: Sequence[str]) -> str:
    for msg in reversed(recent_assistant_messages):
        text = str(msg or "").strip()
        if text:
            return text
    return ""


def _ambiguity_profile(query_text: str, recent_assistant_messages: Sequence[str]) -> Dict[str, Any]:
    query = str(query_text or "").strip()
    lower = query.lower()
    anchor = _recent_anchor_text(recent_assistant_messages)
    content_terms = _content_term_set(query, max_terms=12)
    explicit_target = bool(
        EXPLICIT_TARGET_RE.search(query)
        or "`" in query
        or "\n" in query
        or (":" in query and len(query) >= 20)
    )
    short_query = len(content_terms) <= 2 and len(query.split()) <= 5
    vague_followup = bool(FOLLOWUP_EDIT_HINT_RE.search(lower)) and len(content_terms) <= 4
    generic_edit = bool(AMBIGUOUS_EDIT_RE.search(lower))
    conflict = bool(SHORTEN_REQUEST_RE.search(lower) and (EXPAND_REQUEST_RE.search(lower) or CONTINUE_REQUEST_RE.search(lower)))
    missing_anchor = not bool(anchor) and (generic_edit or vague_followup or short_query)
    needs_clarification = bool(conflict or (missing_anchor and not explicit_target))
    tags: List[str] = []
    if missing_anchor:
        tags.append("missing_anchor")
    if generic_edit:
        tags.append("generic_edit")
    if vague_followup:
        tags.append("vague_followup")
    if short_query:
        tags.append("short_query")
    if conflict:
        tags.append("conflicting_constraints")
    return {
        "needs_clarification": needs_clarification,
        "has_anchor": bool(anchor),
        "conflict": conflict,
        "explicit_target": explicit_target,
        "tags": tags,
    }


def _expected_conversation_act(history: Sequence[Tuple[str, str]], user_text: str) -> str:
    recent_assistant = [history[-1][1]] if history else []
    ambiguity = _ambiguity_profile(user_text, recent_assistant)
    if ambiguity["needs_clarification"]:
        return "clarify"

    profile = _followup_request_profile(user_text)
    if profile["shorten"]:
        return "shorten"
    if profile["expand"]:
        return "expand"
    if profile["rewrite"]:
        return "rewrite"
    if profile["continue"]:
        return "continue"
    if profile["reasoning"]:
        return "reason"
    if profile["creative"]:
        return "create"
    return "answer"


def _context_control_tags(history: Sequence[Tuple[str, str]], user_text: str) -> List[str]:
    query = (user_text or "").strip().lower()
    tags: List[str] = []
    if history and FOLLOWUP_EDIT_HINT_RE.search(query):
        tags.append("followup")
    ambiguity = _ambiguity_profile(user_text, [history[-1][1]] if history else [])
    if ambiguity["needs_clarification"]:
        tags.append("clarify")
    if SHORTEN_REQUEST_RE.search(query):
        tags.append("shorten")
    if EXPAND_REQUEST_RE.search(query):
        tags.append("expand")
    if REWRITE_REQUEST_RE.search(query):
        tags.append("rewrite")
    if CONTINUE_REQUEST_RE.search(query):
        tags.append("continue")
    if REASONING_REQUEST_RE.search(query) or ANALYST_HINT_RE.search(query):
        tags.append("reasoning")
    if CREATIVE_REQUEST_RE.search(query) or CREATIVE_HINT_RE.search(query):
        tags.append("creative")

    flags = _domain_flags(user_text)
    for domain in ("code", "literary", "science", "english", "math", "scripture", "dictionary"):
        if flags.get(domain, False):
            tags.append(domain)
    return tags


def build_context(history: Sequence[Tuple[str, str]], user_text: str, max_turns: int = 4) -> str:
    recent = list(history[-max_turns:])
    parts: List[str] = []
    tags = _context_control_tags(recent, user_text)
    if tags:
        parts.append(f"System: conversation_tags={','.join(tags)}")
    ambiguity = _ambiguity_profile(user_text, [recent[-1][1]] if recent else [])
    if ambiguity["tags"]:
        parts.append(f"System: ambiguity_tags={','.join(ambiguity['tags'])}")
    expected_act = _expected_conversation_act(recent, user_text)
    if expected_act:
        parts.append(f"System: expected_act={expected_act}")
    topic_terms = _salient_terms(" ".join([user_text] + [u for u, _a in recent[-2:]]))
    if topic_terms:
        parts.append(f"System: topic_terms={', '.join(topic_terms)}")
    if recent:
        last_user, last_assistant = recent[-1]
        last_assistant_focus = _salient_terms(last_assistant, max_terms=6)
        if last_assistant_focus:
            parts.append(f"System: last_assistant_focus={', '.join(last_assistant_focus)}")
        if last_user:
            parts.append(f"System: previous_user_request={last_user[:180]}")
    for user, assistant in recent:
        parts.append(f"User: {user}")
        parts.append(f"Assistant: {assistant}")
    parts.append(f"User: {user_text}")
    return "\n".join(parts)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = _coerce_text(item)
            if text:
                out.append(text)
        return " ".join(out).strip()
    if isinstance(value, dict):
        if "text" in value:
            return _coerce_text(value["text"])
        if "content" in value:
            return _coerce_text(value["content"])
        return ""
    return ""


def _media_context_prefix_from_record(record: Dict[str, Any]) -> str:
    image_path = _coerce_text(record.get("image_path"))
    image_desc = _coerce_text(record.get("image_desc"))
    image_caption = _coerce_text(record.get("image_caption") or record.get("caption"))
    image_tags_raw = record.get("image_tags") or record.get("tags")
    image_tags = _coerce_text(image_tags_raw)
    video_path = _coerce_text(record.get("video_path"))
    video_desc = _coerce_text(record.get("video_desc"))
    video_caption = _coerce_text(record.get("video_caption"))
    video_tags = _coerce_text(record.get("video_tags"))
    model3d_path = _coerce_text(record.get("model3d_path") or record.get("mesh_path") or record.get("obj_path"))
    model3d_desc = _coerce_text(record.get("model3d_desc") or record.get("mesh_desc"))
    model3d_caption = _coerce_text(record.get("model3d_caption"))
    model3d_tags = _coerce_text(record.get("model3d_tags"))

    if image_path and not image_desc:
        try:
            from image_feature_utils import describe_image_for_text  # lazy import, only used for image rows

            image_desc = describe_image_for_text(image_path)
        except Exception:
            image_desc = ""

    if video_path and not video_desc:
        try:
            from video_feature_utils import describe_video_for_text  # lazy import

            video_desc = describe_video_for_text(video_path)
        except Exception:
            video_desc = ""

    if model3d_path and not model3d_desc:
        try:
            from mesh_feature_utils import describe_3d_model_for_text  # lazy import

            model3d_desc = describe_3d_model_for_text(model3d_path)
        except Exception:
            model3d_desc = ""

    parts: List[str] = []
    if image_path:
        parts.append(f"path={image_path}")
    if image_desc:
        parts.append(f"desc={image_desc}")
    if image_caption:
        parts.append(f"caption={image_caption}")
    if image_tags:
        parts.append(f"tags={image_tags}")
    if video_path:
        parts.append(f"video={video_path}")
    if video_desc:
        parts.append(f"video_desc={video_desc}")
    if video_caption:
        parts.append(f"video_caption={video_caption}")
    if video_tags:
        parts.append(f"video_tags={video_tags}")
    if model3d_path:
        parts.append(f"model3d={model3d_path}")
    if model3d_desc:
        parts.append(f"model3d_desc={model3d_desc}")
    if model3d_caption:
        parts.append(f"model3d_caption={model3d_caption}")
    if model3d_tags:
        parts.append(f"model3d_tags={model3d_tags}")
    if not parts:
        return ""
    return "[Media] " + " | ".join(parts)


def _extract_flat_pair(record: Dict[str, Any]) -> Optional[ConversationExample]:
    user_keys = ("user", "prompt", "input", "question", "instruction")
    assistant_keys = ("assistant", "response", "output", "answer", "completion", "target")

    user_text = ""
    for key in user_keys:
        if key in record:
            user_text = _coerce_text(record.get(key))
            if user_text:
                break

    response_text = ""
    for key in assistant_keys:
        if key in record:
            response_text = _coerce_text(record.get(key))
            if response_text:
                break

    if not user_text or not response_text:
        return None

    label_value = record.get("label")
    label: Optional[int] = None
    if isinstance(label_value, int):
        label = label_value
    elif isinstance(label_value, str) and label_value.strip().isdigit():
        label = int(label_value.strip())

    media_prefix = _media_context_prefix_from_record(record)
    if media_prefix:
        context = f"User: {media_prefix}\nUser: {user_text}"
    else:
        context = f"User: {user_text}"

    return ConversationExample(context=context, response=response_text, label=label)


def _extract_messages(record: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    for key in ("messages", "conversation", "chat"):
        value = record.get(key)
        if isinstance(value, list):
            return [msg for msg in value if isinstance(msg, dict)]
    return None


def _extract_from_messages(messages: List[Dict[str, Any]], max_history_msgs: int = 8) -> List[ConversationExample]:
    examples: List[ConversationExample] = []
    hist: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None

    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        text = _coerce_text(msg.get("content", msg.get("text")))
        if not text:
            continue

        if role == "user":
            pending_user = text
            continue

        if role == "assistant" and pending_user is not None:
            local_hist = hist[-max_history_msgs:]
            context_lines: List[str] = []
            for r, t in local_hist:
                context_lines.append(f"{r.capitalize()}: {t}")
            context_lines.append(f"User: {pending_user}")
            examples.append(ConversationExample(context="\n".join(context_lines), response=text))

            hist.append(("user", pending_user))
            hist.append(("assistant", text))
            pending_user = None
            continue

        if role in ("assistant", "system", "tool"):
            hist.append((role, text))

    return examples


def load_conversation_examples(path: str) -> List[ConversationExample]:
    examples: List[ConversationExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(record, dict):
                continue

            pair = _extract_flat_pair(record)
            if pair is not None:
                examples.append(pair)
                continue

            msgs = _extract_messages(record)
            if msgs:
                examples.extend(_extract_from_messages(msgs))

    if not examples:
        raise ValueError(
            "No conversation examples found. Provide JSONL records with "
            "`user`+`assistant` fields or a `messages` list."
        )
    return examples


def _kmeans_assign(vectors: torch.Tensor, k: int, iters: int = 25, seed: int = 42) -> torch.Tensor:
    n = vectors.shape[0]
    if n == 0:
        raise ValueError("Cannot cluster an empty dataset.")

    k = max(1, min(k, n))
    gen = torch.Generator(device=vectors.device)
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen, device=vectors.device)
    centers = vectors[perm[:k]].clone()

    for _ in range(iters):
        dists = torch.cdist(vectors, centers, p=2)
        labels = dists.argmin(dim=1)
        next_centers = centers.clone()
        for ci in range(k):
            mask = labels == ci
            if torch.any(mask):
                next_centers[ci] = vectors[mask].mean(dim=0)
            else:
                ridx = int(torch.randint(0, n, (1,), generator=gen, device=vectors.device).item())
                next_centers[ci] = vectors[ridx]

        shift = torch.norm(next_centers - centers, p=2).item()
        centers = next_centers
        if shift < 1e-4:
            break

    dists = torch.cdist(vectors, centers, p=2)
    return dists.argmin(dim=1)


def assign_labels(examples: Sequence[ConversationExample], seed: int = 42) -> Tuple[torch.Tensor, str]:
    provided = True
    labels: List[int] = []
    for ex in examples:
        if ex.label is None:
            provided = False
            break
        labels.append(int(ex.label))

    if provided:
        fixed: List[int] = []
        for label in labels:
            if label < 0:
                fixed.append(0)
            elif label >= MODEL_CLASSES:
                fixed.append(MODEL_CLASSES - 1)
            else:
                fixed.append(label)
        return torch.tensor(fixed, dtype=torch.long), "provided_label"

    response_vecs = torch.stack([featurize_text(ex.response) for ex in examples], dim=0)
    k = min(MODEL_CLASSES, max(2, min(len(examples), 10)))
    auto_labels = _kmeans_assign(response_vecs, k=k, seed=seed).to(torch.long)
    return auto_labels, f"kmeans_{k}"


def build_training_tensors(
    examples: Sequence[ConversationExample],
    labels: torch.Tensor,
    feature_mode: str = "legacy",
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = resolve_feature_mode(feature_mode, smarter_auto=False)
    if mode == "context_mix_v4":
        feats = [featurize_context_mix_v4(ex.context) for ex in examples]
    elif mode == "context_mix_v2_mm":
        feats = [featurize_context_mix_v2_mm(ex.context) for ex in examples]
    elif mode == "context_mix_v3":
        feats = [featurize_context_mix_v3(ex.context) for ex in examples]
    elif mode == "context_mix_v1":
        feats = [featurize_context_mix_v1(ex.context) for ex in examples]
    elif mode == "context_v5":
        feats = [featurize_context_v5(ex.context) for ex in examples]
    elif mode == "context_v4":
        feats = [featurize_context_v4(ex.context) for ex in examples]
    elif mode == "context_v3":
        feats = [featurize_context_v3(ex.context) for ex in examples]
    elif mode == "context_v2":
        feats = [featurize_context_v2(ex.context) for ex in examples]
    else:
        feats = [featurize_text(ex.context) for ex in examples]
    x = torch.stack(feats, dim=0).unsqueeze(1)  # (N,1,128)
    y = labels.to(torch.long)
    if y.numel() != x.shape[0]:
        raise ValueError("Label count does not match feature count.")
    return x, y


def build_bucket_metadata(
    examples: Sequence[ConversationExample],
    labels: torch.Tensor,
    max_candidates_per_bucket: int = 64,
    feature_mode: str = "legacy",
) -> Dict[str, Any]:
    label_counts = defaultdict(int)
    for label in labels.tolist():
        label_counts[int(label)] += 1
    priors = {
        str(label): float(count / max(1, labels.numel()))
        for label, count in sorted(label_counts.items())
    }

    if int(max_candidates_per_bucket) <= 0:
        # Fast mode for very large training runs: skip bucket feature materialization.
        return {"buckets": {}, "label_priors": priors}

    mode = resolve_feature_mode(feature_mode, smarter_auto=False)
    if mode == "context_mix_v4":
        ctx_featurizer = featurize_context_mix_v4
    elif mode == "context_mix_v2_mm":
        ctx_featurizer = featurize_context_mix_v2_mm
    elif mode == "context_mix_v3":
        ctx_featurizer = featurize_context_mix_v3
    elif mode == "context_mix_v1":
        ctx_featurizer = featurize_context_mix_v1
    elif mode == "context_v5":
        ctx_featurizer = featurize_context_v5
    elif mode == "context_v4":
        ctx_featurizer = featurize_context_v4
    elif mode == "context_v3":
        ctx_featurizer = featurize_context_v3
    elif mode == "context_v2":
        ctx_featurizer = featurize_context_v2
    else:
        ctx_featurizer = featurize_text

    grouped: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for ex, label in zip(examples, labels.tolist()):
        key = ex.response.strip()
        if not key:
            continue
        bucket = grouped[int(label)]
        row = bucket.get(key)
        if row is None:
            row = {
                "count": 0,
                "ctx_sum": torch.zeros(FEAT_DIM, dtype=torch.float32),
                "resp_sum": torch.zeros(FEAT_DIM, dtype=torch.float32),
            }
            bucket[key] = row
        row["count"] += 1
        row["ctx_sum"] += ctx_featurizer(ex.context)
        row["resp_sum"] += featurize_text(ex.response)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for label, rows_dict in grouped.items():
        sorted_items = sorted(rows_dict.items(), key=lambda kv: (-int(kv[1]["count"]), kv[0]))
        rows: List[Dict[str, Any]] = []
        for response, stats in sorted_items[:max_candidates_per_bucket]:
            cnt = max(1, int(stats["count"]))
            ctx_vec = (stats["ctx_sum"] / cnt).tolist()
            resp_vec = (stats["resp_sum"] / cnt).tolist()
            rows.append(
                {
                    "text": response,
                    "count": cnt,
                    "vec": resp_vec,
                    "ctx_vec": ctx_vec,
                }
            )
        buckets[str(label)] = rows

    return {"buckets": buckets, "label_priors": priors}


def choose_bucket_from_logits(
    logits: torch.Tensor,
    available_labels: Sequence[int],
    temperature: float = 0.0,
) -> int:
    if not available_labels:
        return int(torch.argmax(logits).item())

    idx = torch.tensor(list(available_labels), dtype=torch.long, device=logits.device)
    sliced = logits.index_select(0, idx)
    if temperature <= 0:
        return int(idx[int(torch.argmax(sliced).item())].item())

    probs = torch.softmax(sliced / max(temperature, 1e-6), dim=0)
    choice = int(torch.multinomial(probs, num_samples=1).item())
    return int(idx[choice].item())


def infer_style_mode(query_text: str, requested_mode: str = "auto") -> str:
    mode = str(requested_mode or "auto").strip().lower()
    if mode in {"balanced", "creative", "concise", "analyst"}:
        return mode

    query = (query_text or "").strip().lower()
    if not query:
        return "balanced"
    if CONCISE_HINT_RE.search(query):
        return "concise"
    if CREATIVE_HINT_RE.search(query):
        return "creative"
    if LITERARY_HINT_RE.search(query):
        if re.search(r"\b(continue|rewrite|style|voice|creative)\b", query):
            return "creative"
        return "analyst"
    if SCIENCE_HINT_RE.search(query):
        if re.search(r"\b(explain|why|how|compare|experiment|hypothesis)\b", query):
            return "analyst"
        return "balanced"
    if MATH_HINT_RE.search(query):
        if re.search(r"\b(word problem|story problem|creative)\b", query):
            return "creative"
        return "analyst"
    if ENGLISH_HINT_RE.search(query):
        if re.search(r"\b(write|draft|creative|story|poem|metaphor)\b", query):
            return "creative"
        return "analyst"
    if SCRIPTURE_HINT_RE.search(query) or SCRIPTURE_REF_RE.search(query):
        if re.search(r"\b(quote|exact|verbatim|text of|what does .* say)\b", query):
            return "concise"
        if re.search(r"\b(prayer|poem|devotional|reflection|creative|story)\b", query):
            return "creative"
        if re.search(r"\b(explain|meaning|context|compare|verse|chapter|quote)\b", query):
            return "analyst"
        return "balanced"
    if DICTIONARY_HINT_RE.search(query):
        if re.search(r"\b(define|definition|meaning|part of speech|synonym|antonym)\b", query):
            return "concise"
        if re.search(r"\b(use in a sentence|example sentence|creative example|mnemonic)\b", query):
            return "creative"
        return "analyst"
    if ANALYST_HINT_RE.search(query) or CODE_HINT_RE.search(query):
        return "analyst"
    return "balanced"


def _followup_request_profile(query_text: str) -> Dict[str, bool]:
    query = (query_text or "").strip().lower()
    return {
        "followup": bool(FOLLOWUP_EDIT_HINT_RE.search(query)),
        "shorten": bool(SHORTEN_REQUEST_RE.search(query)),
        "expand": bool(EXPAND_REQUEST_RE.search(query)),
        "rewrite": bool(REWRITE_REQUEST_RE.search(query)),
        "continue": bool(CONTINUE_REQUEST_RE.search(query)),
        "reasoning": bool(REASONING_REQUEST_RE.search(query) or ANALYST_HINT_RE.search(query)),
        "creative": bool(CREATIVE_REQUEST_RE.search(query) or CREATIVE_HINT_RE.search(query)),
    }


def _clarification_question_for_query(query_text: str, recent_assistant_messages: Sequence[str]) -> str:
    profile = _followup_request_profile(query_text)
    anchor = _recent_anchor_text(recent_assistant_messages)
    if profile["shorten"]:
        return "What should I shorten? Paste the text or point me to the answer you want condensed."
    if profile["expand"] or profile["continue"]:
        return "What should I expand on? Point me to the topic or paste the text you want me to continue."
    if profile["rewrite"]:
        return "What should I rewrite? Paste the text or mention the answer you want rephrased."
    if profile["creative"]:
        return "What topic do you want me to make more creative?"
    if profile["reasoning"]:
        return "Which problem or claim do you want me to reason through step by step?"
    if anchor:
        focus_terms = ", ".join(_salient_terms(anchor, max_terms=4))
        if focus_terms:
            return f"Which part should I focus on: {focus_terms}?"
    return "What are you referring to? A topic, question, or text snippet would let me answer precisely."


def _clarification_response_signal(text: str) -> float:
    raw = str(text or "").strip().lower()
    if not raw:
        return 0.0
    score = 0.0
    if "?" in raw:
        score += 0.35
    for phrase in CLARIFICATION_PHRASES:
        if phrase in raw:
            score += 0.18
    if re.search(r"\b(paste|share|point me to|tell me which|show me)\b", raw):
        score += 0.15
    return min(1.0, score)


def _query_view_texts(query_text: str, recent_assistant_messages: Sequence[str]) -> List[str]:
    query = str(query_text or "").strip()
    if not query:
        return []
    profile = _followup_request_profile(query)
    ambiguity = _ambiguity_profile(query, recent_assistant_messages)
    anchor = _recent_anchor_text(recent_assistant_messages)
    views = [query]
    if profile["followup"] and anchor:
        views.append(f"{query}\nanchor: {anchor[:240]}")
        focus_terms = _salient_terms(anchor, max_terms=6)
        if focus_terms:
            views.append(f"{query}\nfocus: {' '.join(focus_terms)}")
    if profile["shorten"]:
        views.append(f"{query}\nbrief concise shorter trim")
    if profile["expand"] or profile["continue"]:
        views.append(f"{query}\ndeeper elaborate explain more continue")
    if profile["rewrite"]:
        views.append(f"{query}\nrewrite rephrase clearer preserve meaning")
    if profile["reasoning"]:
        views.append(f"{query}\nstep by step because verify tradeoff")
    if profile["creative"]:
        views.append(f"{query}\nmetaphor analogy vivid fresh angle")
    if ambiguity["needs_clarification"]:
        views.append(f"{query}\nclarify exact referent or source text")
    out: List[str] = []
    seen: Set[str] = set()
    for view in views:
        key = " ".join(view.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(view)
    return out[:5]


def _followup_constraint_score(
    query_text: str,
    candidate_text: str,
    anchor_text: str,
    reasoning_signal: float = 0.0,
    creative_signal: float = 0.0,
) -> float:
    profile = _followup_request_profile(query_text)
    if not profile["followup"] or not anchor_text or not candidate_text:
        return 0.0

    anchor_terms = _content_term_set(anchor_text, max_terms=14)
    candidate_terms = _content_term_set(candidate_text, max_terms=14)
    anchor_overlap = 0.0
    if anchor_terms:
        anchor_overlap = float(len(anchor_terms & candidate_terms)) / float(max(1, len(anchor_terms)))

    anchor_numbers = set(NUMBER_RE.findall(anchor_text))
    candidate_numbers = set(NUMBER_RE.findall(candidate_text))
    number_overlap = 0.0
    if anchor_numbers:
        number_overlap = float(len(anchor_numbers & candidate_numbers)) / float(max(1, len(anchor_numbers)))

    anchor_words = max(1.0, float(len(anchor_text.split())))
    cand_words = max(1.0, float(len(candidate_text.split())))
    len_ratio = cand_words / anchor_words
    text_sim = float(SequenceMatcher(None, anchor_text.lower(), candidate_text.lower()).ratio())

    score = 0.24 * anchor_overlap + 0.12 * number_overlap
    if profile["shorten"]:
        if len_ratio < 0.92:
            score += 0.34 * min(1.0, (0.92 - len_ratio) / 0.52)
        else:
            score -= 0.10 * min(1.0, (len_ratio - 0.92) / 0.85)
    if profile["expand"] or profile["continue"]:
        if len_ratio > 1.05:
            score += 0.24 * min(1.0, (len_ratio - 1.05) / 1.10)
    if profile["rewrite"]:
        if 0.22 <= text_sim <= 0.92:
            score += 0.22
        elif text_sim > 0.98:
            score -= 0.12
    if profile["reasoning"]:
        score += 0.22 * float(reasoning_signal)
    if profile["creative"]:
        score += 0.20 * float(creative_signal)
    return float(max(-0.30, min(1.25, score)))


def _token_set(text: str, max_tokens: int = 96) -> Set[str]:
    out: Set[str] = set()
    for tok in _tokens(text, max_tokens=max_tokens):
        if re.fullmatch(r"[a-z0-9_'-]+", tok):
            out.add(tok)
    return out


def _normalize_01(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = torch.min(values)
    hi = torch.max(values)
    span = float((hi - lo).item())
    if span < 1e-6:
        return torch.zeros_like(values)
    return (values - lo) / (span + 1e-6)


def _domain_flags(text: str) -> Dict[str, bool]:
    t = text or ""
    return {
        "code": bool(PROGRAMMING_HINT_RE.search(t) or CODE_HINT_RE.search(t)),
        "literary": bool(LITERARY_HINT_RE.search(t)),
        "science": bool(SCIENCE_HINT_RE.search(t)),
        "english": bool(ENGLISH_HINT_RE.search(t)),
        "math": bool(MATH_HINT_RE.search(t)),
        "scripture": bool(SCRIPTURE_HINT_RE.search(t) or SCRIPTURE_REF_RE.search(t)),
        "dictionary": bool(DICTIONARY_HINT_RE.search(t)),
    }


def rank_response_candidates(
    candidates: Sequence[Dict[str, Any]],
    query_text: str,
    recent_assistant_messages: Sequence[str],
    style_mode: str = "balanced",
) -> Tuple[List[int], torch.Tensor]:
    if not candidates:
        return [], torch.zeros(0, dtype=torch.float32)

    query_views = _query_view_texts(query_text, recent_assistant_messages)
    q = featurize_text(query_text)
    cand_resp_vecs = torch.tensor([row["vec"] for row in candidates], dtype=torch.float32)
    cand_ctx_vecs = torch.tensor(
        [row.get("ctx_vec", row["vec"]) for row in candidates],
        dtype=torch.float32,
    )
    sim_resp = torch.mv(cand_resp_vecs, q)
    sim_ctx = torch.mv(cand_ctx_vecs, q)

    recent = [msg.strip() for msg in recent_assistant_messages[-4:] if msg.strip()]
    if recent:
        recent_vecs = torch.stack([featurize_text(msg) for msg in recent], dim=0)
        sim_recent = torch.mm(cand_resp_vecs, recent_vecs.t()).max(dim=1).values
    else:
        sim_recent = torch.zeros(cand_resp_vecs.shape[0], dtype=torch.float32)
    if len(query_views) >= 2:
        view_vecs = torch.stack([featurize_text(view) for view in query_views], dim=0)
        view_resp = torch.mm(cand_resp_vecs, view_vecs.t())
        view_ctx = torch.mm(cand_ctx_vecs, view_vecs.t())
        view_consistency_signal = _normalize_01(
            0.55 * view_ctx.mean(dim=1)
            + 0.45 * view_resp.mean(dim=1)
            + 0.20 * torch.minimum(view_ctx.min(dim=1).values, view_resp.min(dim=1).values)
        )
    else:
        view_consistency_signal = torch.zeros(cand_resp_vecs.shape[0], dtype=torch.float32)

    query_tokens = _token_set(query_text, max_tokens=64)
    overlap_vals: List[float] = []
    creative_vals: List[float] = []
    analytic_vals: List[float] = []
    concise_vals: List[float] = []
    diversity_vals: List[float] = []
    code_vals: List[float] = []
    literary_vals: List[float] = []
    science_vals: List[float] = []
    english_vals: List[float] = []
    math_vals: List[float] = []
    scripture_vals: List[float] = []
    dictionary_vals: List[float] = []
    exact_lookup_vals: List[float] = []
    reference_style_vals: List[float] = []
    reasoning_depth_vals: List[float] = []
    empathy_vals: List[float] = []
    word_count_vals: List[float] = []
    q_domains = _domain_flags(query_text)
    for row in candidates:
        text = str(row.get("text", ""))
        toks = _token_set(text, max_tokens=96)
        union_n = max(1, len(query_tokens | toks))
        overlap_vals.append(float(len(query_tokens & toks) / union_n))

        creative_hits = len(toks & CREATIVE_WORDS)
        analytic_hits = len(toks & ANALYTIC_WORDS)
        token_len = max(1, len(toks))
        lower_text = text.lower()

        creative_signal = min(1.0, 0.35 * creative_hits + (0.12 if ":" in text else 0.0) + (0.10 if ";" in text else 0.0))
        analytic_signal = min(
            1.0,
            0.28 * analytic_hits
            + (0.15 if re.search(r"\b(step|first|second|third)\b", lower_text) else 0.0)
            + (0.10 if re.search(r"\b(because|therefore|verify)\b", lower_text) else 0.0),
        )
        concise_signal = min(1.0, 18.0 / float(token_len))
        diversity_signal = min(
            1.0,
            0.55 * float(len(toks)) / 40.0
            + (0.15 if ";" in text else 0.0)
            + (0.12 if ":" in text else 0.0)
            + (0.10 if re.search(r"\b(for example|for instance|consider|imagine)\b", lower_text) else 0.0),
        )
        dflags = _domain_flags(text)
        code_signal = 1.0 if dflags["code"] else 0.0
        literary_signal = 1.0 if dflags["literary"] else 0.0
        science_signal = 1.0 if dflags["science"] else 0.0
        english_signal = 1.0 if dflags["english"] else 0.0
        math_signal = 1.0 if dflags["math"] else 0.0
        scripture_signal = 1.0 if dflags.get("scripture", False) else 0.0
        dictionary_signal = 1.0 if dflags["dictionary"] else 0.0
        # Soft boosts for obvious math formatting and English-editing language.
        if re.search(r"[0-9]+\s*(?:[+\-*/=]|%)", text):
            math_signal = min(1.0, math_signal + 0.5)
        if re.search(r"\b(grammar|punctuation|spelling|sentence|paragraph|rewrite|proofread)\b", lower_text):
            english_signal = min(1.0, english_signal + 0.5)
        if re.search(r"\b(means|definition|defined as|synonym|antonym|dictionary|vocabulary|part of speech|usage)\b", lower_text):
            dictionary_signal = min(1.0, dictionary_signal + 0.5)
        if SCRIPTURE_REF_RE.search(text) or re.search(r"\b(kjv|bible|scripture|verse|chapter|gospel|psalm|parable|apostle|prophet)\b", lower_text):
            scripture_signal = min(1.0, scripture_signal + 0.5)
        reference_style = 0.0
        if SCRIPTURE_REF_RE.search(text):
            reference_style += 0.65
        if re.search(r"\b(kjv|scripture|bible|verse|chapter)\b", lower_text):
            reference_style += 0.20
        if re.search(r"\b(definition|defined as|part of speech|synonym|antonym|lemma|usage)\b", lower_text):
            reference_style += 0.45
        if re.search(r"^[A-Z][^\\n]{0,80}:\\s", text):
            reference_style += 0.10
        reference_style = min(1.0, reference_style)

        exact_lookup_signal = 0.0
        if reference_style > 0:
            exact_lookup_signal += 0.45 * reference_style
        if dictionary_signal > 0:
            exact_lookup_signal += 0.25 * dictionary_signal
        if scripture_signal > 0:
            exact_lookup_signal += 0.25 * scripture_signal
        if concise_signal > 0:
            exact_lookup_signal += 0.15 * concise_signal
        exact_lookup_signal = min(1.0, exact_lookup_signal)

        # Reasoning depth: multi-step structure indicators in the response.
        reasoning_depth = 0.0
        if re.search(r"\b(step\s*\d|first|then|next|finally|therefore|thus|hence|consequently)\b", lower_text):
            reasoning_depth += 0.35
        if re.search(r"^\s*\d+[.)]\s", text, re.MULTILINE):
            reasoning_depth += 0.30
        if re.search(r"\b(because|since|due to|as a result)\b", lower_text):
            reasoning_depth += 0.20
        if re.search(r"\b(let me|let's|consider|suppose|assume)\b", lower_text):
            reasoning_depth += 0.15
        reasoning_depth = min(1.0, reasoning_depth)

        # Empathy: emotionally supportive language indicators.
        empathy_score = 0.0
        if re.search(r"\b(I hear you|I understand|that must be|it's okay|completely valid|you're not alone)\b", lower_text):
            empathy_score += 0.40
        if re.search(r"\b(feel|feeling|support|comfort|care|empathy|compassion)\b", lower_text):
            empathy_score += 0.25
        if re.search(r"\b(tough|hard time|overwhelm|stress|anxious|worry)\b", lower_text):
            empathy_score += 0.20
        if re.search(r"\b(proud of|grateful|appreciate|thank)\b", lower_text):
            empathy_score += 0.15
        empathy_score = min(1.0, empathy_score)

        creative_vals.append(creative_signal)
        analytic_vals.append(analytic_signal)
        concise_vals.append(concise_signal)
        diversity_vals.append(diversity_signal)
        code_vals.append(code_signal)
        literary_vals.append(literary_signal)
        science_vals.append(science_signal)
        english_vals.append(english_signal)
        math_vals.append(math_signal)
        scripture_vals.append(scripture_signal)
        dictionary_vals.append(dictionary_signal)
        exact_lookup_vals.append(exact_lookup_signal)
        reference_style_vals.append(reference_style)
        reasoning_depth_vals.append(reasoning_depth)
        empathy_vals.append(empathy_score)
        word_count_vals.append(float(max(1, len(text.split()))))

    lex_sim = torch.tensor(overlap_vals, dtype=torch.float32)
    creative_signal = _normalize_01(torch.tensor(creative_vals, dtype=torch.float32))
    analytic_signal = _normalize_01(torch.tensor(analytic_vals, dtype=torch.float32))
    concise_signal = _normalize_01(torch.tensor(concise_vals, dtype=torch.float32))
    diversity_signal = _normalize_01(torch.tensor(diversity_vals, dtype=torch.float32))
    code_signal = torch.tensor(code_vals, dtype=torch.float32)
    literary_signal = torch.tensor(literary_vals, dtype=torch.float32)
    science_signal = torch.tensor(science_vals, dtype=torch.float32)
    english_signal = torch.tensor(english_vals, dtype=torch.float32)
    math_signal = torch.tensor(math_vals, dtype=torch.float32)
    scripture_signal = torch.tensor(scripture_vals, dtype=torch.float32)
    dictionary_signal = torch.tensor(dictionary_vals, dtype=torch.float32)
    exact_lookup_signal = _normalize_01(torch.tensor(exact_lookup_vals, dtype=torch.float32))
    reference_style_signal = torch.tensor(reference_style_vals, dtype=torch.float32)
    reasoning_depth_signal = _normalize_01(torch.tensor(reasoning_depth_vals, dtype=torch.float32))
    empathy_signal = _normalize_01(torch.tensor(empathy_vals, dtype=torch.float32))
    word_count_signal = torch.tensor(word_count_vals, dtype=torch.float32)

    freq_penalty = torch.tensor([math.log1p(float(row.get("count", 1))) for row in candidates], dtype=torch.float32)
    bucket_bonus = torch.tensor([float(row.get("bucket_score", 0.0)) for row in candidates], dtype=torch.float32)
    domain_alignment = torch.zeros(len(candidates), dtype=torch.float32)
    if q_domains["code"]:
        domain_alignment += 0.24 * code_signal
    if q_domains["literary"]:
        domain_alignment += 0.22 * literary_signal
    if q_domains["science"]:
        domain_alignment += 0.22 * science_signal
    if q_domains["english"]:
        domain_alignment += 0.22 * english_signal
    if q_domains["math"]:
        domain_alignment += 0.24 * math_signal
    if q_domains.get("scripture", False):
        domain_alignment += 0.22 * scripture_signal
    if q_domains["dictionary"]:
        domain_alignment += 0.20 * dictionary_signal

    # Detect if query suggests reasoning or empathy needs
    q_lower = query_text.lower() if query_text else ""
    q_needs_reasoning = bool(re.search(r"\b(solve|step|explain|why|how|calculate|prove|derive|reason)\b", q_lower))
    q_needs_empathy = bool(re.search(r"\b(feel|sad|anxious|scared|upset|help me|stressed|overwhelmed|lonely)\b", q_lower))
    q_needs_exact_lookup = bool(
        EXACT_LOOKUP_HINT_RE.search(q_lower)
        or q_domains.get("scripture", False)
        or q_domains["dictionary"]
    )
    q_prefers_reference = bool(q_domains.get("scripture", False) or re.search(r"\b(reference|ref|chapter|verse|citation|cite)\b", q_lower))
    followup_profile = _followup_request_profile(query_text)
    anchor_signal = torch.zeros(len(candidates), dtype=torch.float32)
    edit_match_signal = torch.zeros(len(candidates), dtype=torch.float32)
    if recent:
        anchor_text = recent[-1]
        anchor_signal = _normalize_01(torch.mv(cand_resp_vecs, featurize_text(anchor_text)))
        anchor_word_count = max(1.0, float(len(anchor_text.split())))
        edit_vals: List[float] = []
        for idx, row in enumerate(candidates):
            text = str(row.get("text", ""))
            score = 0.20 * float(anchor_signal[idx].item())
            score += _followup_constraint_score(
                query_text=query_text,
                candidate_text=text,
                anchor_text=anchor_text,
                reasoning_signal=float(reasoning_depth_signal[idx].item()),
                creative_signal=float(creative_signal[idx].item()),
            )
            if followup_profile["expand"] or followup_profile["continue"]:
                word_count = float(word_count_signal[idx].item())
                if word_count > 1.05 * anchor_word_count:
                    score += 0.08
            edit_vals.append(score)
        edit_match_signal = _normalize_01(torch.tensor(edit_vals, dtype=torch.float32))
    followup_bonus = (
        0.10 * anchor_signal + 0.14 * edit_match_signal + 0.08 * view_consistency_signal
    ) if followup_profile["followup"] else 0.0

    mode = infer_style_mode(query_text, requested_mode=style_mode)
    if mode == "creative":
        scores = (
            0.48 * sim_ctx
            + 0.22 * sim_resp
            + 0.10 * lex_sim
            + 0.18 * bucket_bonus
            + 0.14 * creative_signal
            + 0.07 * diversity_signal
            + 0.08 * domain_alignment
            + 0.05 * view_consistency_signal
            + (0.05 * exact_lookup_signal if q_needs_exact_lookup else 0.0)
            + (0.04 * reference_style_signal if q_prefers_reference else 0.0)
            + (0.10 * reasoning_depth_signal if q_needs_reasoning else 0.04 * reasoning_depth_signal)
            + (0.12 * empathy_signal if q_needs_empathy else 0.03 * empathy_signal)
            + followup_bonus
            - (0.08 * reference_style_signal if (q_needs_exact_lookup and not q_prefers_reference) else 0.0)
            - 0.22 * sim_recent
            - 0.02 * freq_penalty
        )
    elif mode == "concise":
        scores = (
            0.60 * sim_ctx
            + 0.24 * sim_resp
            + 0.12 * lex_sim
            + 0.18 * concise_signal
            + 0.14 * bucket_bonus
            + 0.07 * domain_alignment
            + 0.05 * view_consistency_signal
            + (0.12 * exact_lookup_signal if q_needs_exact_lookup else 0.02 * exact_lookup_signal)
            + (0.08 * reference_style_signal if q_prefers_reference else 0.0)
            + (0.08 * reasoning_depth_signal if q_needs_reasoning else 0.02 * reasoning_depth_signal)
            + (0.10 * empathy_signal if q_needs_empathy else 0.02 * empathy_signal)
            + followup_bonus
            - (0.05 * creative_signal if q_needs_exact_lookup else 0.0)
            - 0.27 * sim_recent
            - 0.03 * freq_penalty
        )
    elif mode == "analyst":
        scores = (
            0.62 * sim_ctx
            + 0.27 * sim_resp
            + 0.16 * lex_sim
            + 0.14 * analytic_signal
            + 0.14 * bucket_bonus
            + 0.10 * domain_alignment
            + 0.07 * view_consistency_signal
            + (0.10 * exact_lookup_signal if q_needs_exact_lookup else 0.03 * exact_lookup_signal)
            + (0.08 * reference_style_signal if q_prefers_reference else 0.0)
            + 0.12 * reasoning_depth_signal
            + (0.08 * empathy_signal if q_needs_empathy else 0.02 * empathy_signal)
            + followup_bonus
            - 0.26 * sim_recent
            - 0.02 * freq_penalty
        )
    else:
        scores = (
            0.60 * sim_ctx
            + 0.25 * sim_resp
            + 0.10 * lex_sim
            + 0.18 * bucket_bonus
            + 0.08 * domain_alignment
            + 0.06 * view_consistency_signal
            + (0.10 * exact_lookup_signal if q_needs_exact_lookup else 0.03 * exact_lookup_signal)
            + (0.06 * reference_style_signal if q_prefers_reference else 0.0)
            + (0.10 * reasoning_depth_signal if q_needs_reasoning else 0.04 * reasoning_depth_signal)
            + (0.10 * empathy_signal if q_needs_empathy else 0.03 * empathy_signal)
            + followup_bonus
            - (0.04 * creative_signal if q_needs_exact_lookup else 0.0)
            - 0.28 * sim_recent
            - 0.02 * freq_penalty
        )

    ranked = torch.argsort(scores, descending=True).tolist()
    return ranked, scores


def _stylize_response_text(text: str, query_text: str, style_mode: str, creativity: float) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    mode = infer_style_mode(query_text, requested_mode=style_mode)
    if mode == "concise":
        return _to_concise_answer(raw)

    if mode == "analyst":
        return _to_analyst_answer(raw)

    if mode != "creative":
        return raw

    c = max(0.0, min(1.0, float(creativity)))
    if c <= 0.0:
        return raw

    intros = [
        "Creative take:",
        "Idea sketch:",
        "Fresh angle:",
        "Possibility:",
        "Here's an interesting way to think about it:",
        "Let me paint a picture:",
        "Consider this perspective:",
        "Imagine it this way:",
        "Here's what makes this fascinating:",
        "The elegant answer is:",
        "A fresh take on this:",
        "The key realization is:",
    ]
    analogies = [
        "debugging a system by isolating one variable at a time",
        "laying tracks before running the train",
        "building scaffolding before finishing the facade",
        "tuning an instrument before performing",
        "a master chef tasting as they cook, adjusting spices gradually",
        "an archaeologist carefully brushing away sand to reveal the treasure",
        "a detective connecting red strings on a corkboard",
        "a gardener pruning branches so the tree grows stronger",
        "a cartographer drawing a map while exploring unknown territory",
        "an orchestra conductor bringing different instruments into harmony",
        "a sculptor revealing the form already hidden in the marble",
        "a chess player thinking three moves ahead before touching a piece",
    ]
    h = _stable_hash((query_text or "") + "|" + raw)
    intro = intros[h % len(intros)]
    if c < 0.30:
        return f"{intro} {raw}"

    analogy = analogies[(h >> 7) % len(analogies)]
    if c < 0.65:
        return f"{intro} {raw} Think of it like {analogy}."

    # High creativity: add a thought-provoking follow-up question
    follow_ups = [
        "What would you do differently if you saw it from this angle?",
        "Could there be an even deeper layer we haven't considered?",
        "What does this tell us about the bigger picture?",
        "How might this change how you approach the problem?",
        "What's the one thing here that surprises you most?",
    ]
    follow_up = follow_ups[(h >> 14) % len(follow_ups)]
    return f"{intro} {raw} Think of it like {analogy}. {follow_up}"


def _to_concise_answer(text: str, max_words: int = 24) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    first = raw
    chunks = [c.strip() for c in SENTENCE_RE.findall(raw) if c.strip()]
    if chunks:
        first = chunks[0]
    words = first.split()
    if len(words) > max_words:
        first = " ".join(words[:max_words]).rstrip(" ,;:.") + "."
    return first


def _to_analyst_answer(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if re.search(r"(^|\s)(1\)|1\.|first\b)", raw.lower()):
        return raw
    chunks = [c.strip() for c in SENTENCE_RE.findall(raw) if c.strip()]
    if len(chunks) >= 3:
        return f"1) {chunks[0]} 2) {chunks[1]} 3) {chunks[2]}"
    if len(chunks) >= 2:
        return f"1) {chunks[0]} 2) {chunks[1]}"
    return f"1) {raw}"


def _reasoning_signal(text: str) -> float:
    raw = (text or "").strip().lower()
    if not raw:
        return 0.0
    score = 0.0
    if re.search(r"(^|\s)(1\)|1\.|first\b|step\s*1)", raw):
        score += 0.35
    if re.search(r"\b(then|next|finally|therefore|thus|because|since|verify)\b", raw):
        score += 0.35
    if re.search(r"\b(let me|consider|assume|tradeoff|edge case)\b", raw):
        score += 0.20
    return min(1.0, score)


def _maybe_refine_selected_response(
    text: str,
    query_text: str,
    recent_assistant_messages: Sequence[str],
    style_mode: str,
    creativity: float,
) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    profile = _followup_request_profile(query_text)
    anchor = ""
    for msg in reversed(recent_assistant_messages):
        msg = msg.strip()
        if msg:
            anchor = msg
            break

    if profile["followup"] and anchor:
        anchor_clean = cleanup_response_text(anchor)
        if profile["shorten"]:
            if len(raw.split()) < len(anchor_clean.split()):
                return _to_concise_answer(raw, max_words=18)
            return _to_concise_answer(anchor_clean, max_words=18)
        if profile["creative"] and len(query_text.split()) <= 14:
            return _stylize_response_text(
                anchor_clean,
                query_text=query_text,
                style_mode="creative",
                creativity=max(0.55, float(creativity)),
            )
        if profile["expand"] or profile["continue"] or profile["reasoning"]:
            base = raw if len(raw.split()) >= len(anchor_clean.split()) else anchor_clean
            return _to_analyst_answer(base)
        if profile["rewrite"]:
            if SequenceMatcher(None, anchor_clean.lower(), raw.lower()).ratio() > 0.98:
                return cleanup_response_text(anchor_clean)

    styled = _stylize_response_text(raw, query_text=query_text, style_mode=style_mode, creativity=creativity)
    if profile["reasoning"] and _reasoning_signal(styled) < 0.30:
        return _to_analyst_answer(styled)
    return styled


def pick_response(
    candidates: Sequence[Dict[str, Any]],
    query_text: str,
    recent_assistant_messages: Sequence[str],
    response_temperature: float = 0.0,
    style_mode: str = "balanced",
    creativity: float = 0.0,
) -> str:
    if not candidates:
        return "I do not have a good response for that yet."

    ambiguity = _ambiguity_profile(query_text, recent_assistant_messages)
    ranked, scores = rank_response_candidates(
        candidates=candidates,
        query_text=query_text,
        recent_assistant_messages=recent_assistant_messages,
        style_mode=style_mode,
    )
    if ambiguity["needs_clarification"] and (ambiguity["conflict"] or not ambiguity["has_anchor"]):
        top_score = float(scores[ranked[0]].item()) if ranked else -1e9
        if top_score < 0.55 or not ambiguity["has_anchor"]:
            return _clarification_question_for_query(query_text, recent_assistant_messages)
    blocked = set(msg.strip() for msg in recent_assistant_messages[-2:] if msg.strip())
    filtered = [i for i in ranked if str(candidates[i].get("text", "")).strip() not in blocked]
    if not filtered:
        filtered = ranked
    if not filtered:
        return "I do not know how to answer that yet."

    if response_temperature > 0:
        top_n = min(4, len(filtered))
        top_idx = filtered[:top_n]
        top_scores = scores[top_idx]
        probs = torch.softmax(top_scores / max(response_temperature, 1e-6), dim=0)
        chosen = top_idx[int(torch.multinomial(probs, num_samples=1).item())]
        text = str(candidates[chosen].get("text", "")).strip()
        if text:
            return _maybe_refine_selected_response(
                text,
                query_text=query_text,
                recent_assistant_messages=recent_assistant_messages,
                style_mode=style_mode,
                creativity=creativity,
            )

    for i in ranked:
        text = str(candidates[i].get("text", "")).strip()
        if text and text not in blocked:
            return _maybe_refine_selected_response(
                text,
                query_text=query_text,
                recent_assistant_messages=recent_assistant_messages,
                style_mode=style_mode,
                creativity=creativity,
            )

    best = filtered[0]
    text = str(candidates[best].get("text", "")).strip()
    if not text:
        return "I do not know how to answer that yet."
    return _maybe_refine_selected_response(
        text,
        query_text=query_text,
        recent_assistant_messages=recent_assistant_messages,
        style_mode=style_mode,
        creativity=creativity,
    )


def cleanup_response_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    chunks = [c.strip() for c in SENTENCE_RE.findall(raw) if c.strip()]
    if not chunks:
        return raw

    out: List[str] = []
    seen_tail: List[str] = []
    seen_vecs: List[torch.Tensor] = []
    seen_token_sets: List[Set[str]] = []
    filler = {"ok", "okay", "sure", "got it", "understood", "right", "yes"}
    greeting_chunks = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    for chunk in chunks:
        chunk_clean = re.sub(r"^(then|also|and|okay|sure|got it|understood)[\s,.:;-]+", "", chunk, flags=re.I).strip()
        chunk_clean = re.sub(
            r"^(recommended path|short answer|creative take|idea sketch|fresh angle|possibility)\s*:\s*",
            "",
            chunk_clean,
            flags=re.I,
        ).strip()
        norm = re.sub(r"\s+", " ", chunk_clean.lower()).strip(" .!?;,")
        if not norm:
            continue
        if norm in filler:
            continue
        if norm in greeting_chunks and out:
            continue
        if norm.startswith("then ") and norm[5:] in filler:
            continue
        if seen_tail and norm == seen_tail[-1]:
            continue
        if norm.startswith("then ") and seen_tail and norm[5:] == seen_tail[-1]:
            continue
        vec = featurize_text(norm)
        if any(float(torch.dot(vec, prev).item()) > 0.95 for prev in seen_vecs):
            continue
        token_set = _token_set(norm, max_tokens=48)
        if token_set:
            is_near_duplicate = False
            for prev_set in seen_token_sets:
                inter = len(token_set & prev_set)
                union = max(1, len(token_set | prev_set))
                min_size = max(1, min(len(token_set), len(prev_set)))
                if (inter / union) > 0.72:
                    is_near_duplicate = True
                    break
                if inter >= 6 and (inter / min_size) > 0.86:
                    is_near_duplicate = True
                    break
            if is_near_duplicate:
                continue
        out.append(chunk_clean or chunk)
        seen_tail.append(norm)
        seen_vecs.append(vec)
        seen_token_sets.append(token_set)
        if len(seen_vecs) > 6:
            seen_vecs = seen_vecs[-6:]
        if len(seen_token_sets) > 6:
            seen_token_sets = seen_token_sets[-6:]
        if len(seen_tail) > 4:
            seen_tail = seen_tail[-4:]

    cleaned = " ".join(out).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or raw


def summarize_label_stats(labels: torch.Tensor) -> str:
    if labels.numel() == 0:
        return "no labels"
    counts = defaultdict(int)
    for label in labels.tolist():
        counts[int(label)] += 1
    parts = [f"{label}:{count}" for label, count in sorted(counts.items())]
    return ", ".join(parts)
