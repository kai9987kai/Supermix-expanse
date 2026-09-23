"""Byte-level BPE tokenizer for Supermix Expanse v3.

## Why this exists

Expanse v1/v2 speak through Archimedes' ``WordTokenizer``
(``supermix-archimedes/archimedes/src/mimomix_text.py``): one id per whole word
(with its leading whitespace), 10,951 of them. Its own docstring names the
catch -- *the vocabulary is the ceiling on what the model can say*. On the
Expanse corpora that ceiling is hit everywhere: every bio dev row contains at
least one ``<unk>``, and a code reply that needs an identifier the vocabulary
never saw cannot be written at all (code writing scored 0%). A word-level
table cannot be grown fast enough to follow the data; a subword one never
needs to, because every byte string has an encoding.

``BPETokenizer`` wraps a Hugging Face ``tokenizers.Tokenizer``:

* **byte-level BPE**, full 256-byte initial alphabet, so any UTF-8 text encodes
  without ``<unk>`` and ``decode(encode(t)) == t`` (tests/test_bpe_tokenizer.py);
* pre-tokenizer ``Sequence([Split(Regex(" ?\\d"), "isolated"),
  ByteLevel(add_prefix_space=False)])``: every digit is its own pre-token (the
  first digit of a number keeps the preceding space, " 123" -> " 1", "2", "3",
  so numbers do not cost an extra space token), so no merge can ever fuse digits -- the same property ``WordTokenizer``'s
  ``digit_tokens=True`` gives (``mimomix_text.DIGIT_TOKEN_PATTERN`` explains
  why whole-number tokens put arithmetic out of reach);
* ByteLevel decoder; the six ``SPECIAL_TOKENS`` at ids 0-5 with exactly the
  strings (and so the ``PAD BOS EOS UNK USER ASSISTANT`` constants) of the
  word tokenizer, so ``encode_turn``, the ``-100`` prompt mask, ``PAD``
  padding and ``greedy_decode``'s ``EOS`` stop are unchanged.

It is duck-typed to ``WordTokenizer`` (``tokens``, ``index``, ``vocab_size``,
``digit_tokens``, ``reverse_digits``, ``encode``, ``decode``, ``encode_turn``,
``unknown_rate``, ``vocabulary_report``, ``to_dict``/``from_dict``), so every
consumer in ``expanse/`` works unchanged; checkpoints tell the two apart by
``tokenizer["kind"] == "bpe"`` (``expanse_core.tokenizer_from_dict``).

Two details that are easy to get wrong:

* **Special-token strings in user text are text.** By default a
  ``tokenizers.Tokenizer`` matches its added special tokens anywhere in the
  input, so a prompt containing the literal characters ``<eos>`` would encode
  to the real EOS id -- a round-trip break and a prompt-injection hole the word
  tokenizer never had (it splits ``<eos>`` into ``<``, ``eos``, ``>``).
  ``encode_special_tokens = True`` turns that off; it is a runtime flag that
  ``to_str()`` does not serialise, so it is set on every construction.
* ``tokens`` is the id-ordered list of **decoded** token strings (``" the"``,
  not the byte-level ``"Ġthe"``), matching ``WordTokenizer.tokens``, so the
  v3 retokeniser can compare new tokens with old ones directly. A token that
  is part of a multi-byte UTF-8 character decodes to U+FFFD, so several ids
  can share one decoded string; ``index`` keeps the lowest id for each.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

_ARCH_SRC = Path(__file__).resolve().parents[2] / "supermix-archimedes" / "archimedes" / "src"
if str(_ARCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ARCH_SRC))

from mimomix_text import ASSISTANT, BOS, EOS, PAD, SPECIAL_TOKENS, UNK, USER  # noqa: E402

KIND = "bpe"
PRE_TOKENIZER_SPEC = "Sequence([Split(Regex(r' ?\d'), isolated), ByteLevel(add_prefix_space=False)])"

__all__ = ["BPETokenizer", "train_bpe", "new_bpe_model", "KIND", "PRE_TOKENIZER_SPEC", "SPECIAL_TOKENS",
           "PAD", "BOS", "EOS", "UNK", "USER", "ASSISTANT"]


def new_bpe_model() -> Tokenizer:
    """An untrained ``Tokenizer`` with the v3 pipeline (model, pre-tokenizer, decoder)."""
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        # one digit per pre-token, the first keeping its leading space (" 1", "2", "3");
        # Digits(individual_digits=True) would split the space off as a token of its own
        pre_tokenizers.Split(Regex(r" ?\d"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tk.decoder = decoders.ByteLevel()
    return tk


class BPETokenizer:
    """Byte-level BPE with ``WordTokenizer``'s interface (see module doc)."""

    kind = KIND

    def __init__(self, tokenizer: Tokenizer):
        for i, s in enumerate(SPECIAL_TOKENS):
            got = tokenizer.token_to_id(s)
            if got != i:
                raise ValueError(f"special token {s!r} must have id {i}, has {got}")
        # see module doc: special strings inside user text must stay text
        tokenizer.encode_special_tokens = True
        self._tk = tokenizer
        #: Numbers are split into single digits by the pre-tokenizer (never merged).
        self.digit_tokens = True
        #: Digits are always written in reading order (``WordTokenizer`` option not used by v3).
        self.reverse_digits = False
        n = int(tokenizer.get_vocab_size(with_added_tokens=True))
        self.tokens: List[str] = list(tokenizer.decode_batch([[i] for i in range(n)], skip_special_tokens=False))
        self.tokens[:len(SPECIAL_TOKENS)] = list(SPECIAL_TOKENS)
        self.index: Dict[str, int] = {}
        for i, t in enumerate(self.tokens):
            self.index.setdefault(t, i)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BPETokenizer":
        if payload.get("kind") != KIND:
            raise ValueError(f"not a {KIND} tokenizer dict (kind={payload.get('kind')!r})")
        raw = payload["json"]
        return cls(Tokenizer.from_str(raw if isinstance(raw, str) else json.dumps(raw)))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": KIND, "json": self._tk.to_str(), "digit_tokens": True}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def backend(self) -> Tokenizer:
        """The wrapped ``tokenizers.Tokenizer`` (read-only use)."""
        return self._tk

    # -- encoding ----------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return list(self._tk.encode(text, add_special_tokens=False).ids)

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [list(e.ids) for e in self._tk.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        """Text of ``ids``; specials (structure, not text) and out-of-range ids are dropped."""
        n, lo = len(self.tokens), len(SPECIAL_TOKENS)
        keep = [int(i) for i in ids if lo <= int(i) < n]
        return self._tk.decode(keep, skip_special_tokens=True) if keep else ""

    def unknown_rate(self, text: str) -> float:
        """Always 0.0: byte-level BPE has no ``<unk>`` (every byte is in the alphabet)."""
        return 0.0

    def vocabulary_report(self, texts: Sequence[str]) -> Dict[str, object]:
        total = sum(len(e) for e in self.encode_batch(list(texts))) if texts else 0
        return {
            "kind": KIND,
            "vocab_size": self.vocab_size,
            "special_tokens": len(SPECIAL_TOKENS),
            "digit_tokens": self.digit_tokens,
            "reverse_digits": self.reverse_digits,
            "sampled_tokens": total,
            "unknown_tokens": 0,
            "coverage": 1.0,
            "note": "byte-level BPE: every UTF-8 string has an encoding, nothing maps to <unk>",
        }

    # -- chat formatting ---------------------------------------------------
    def encode_turn(self, user: str, assistant: Optional[str] = None) -> Tuple[List[int], int]:
        """``([BOS, USER] + encode(user) + [ASSISTANT] (+ encode(assistant) + [EOS]), prompt_len)``,
        exactly ``WordTokenizer.encode_turn``'s structure."""
        ids = [BOS, USER] + self.encode(user) + [ASSISTANT]
        prompt_length = len(ids)
        if assistant is not None:
            ids = ids + self.encode(assistant) + [EOS]
        return ids, prompt_length


def train_bpe(texts: Iterable[str], vocab_size: int = 24000, min_frequency: int = 2) -> BPETokenizer:
    """Train the v3 byte-level BPE on ``texts`` (one string per item).

    The trainer starts from the six specials (ids 0-5, in ``SPECIAL_TOKENS``
    order) plus the 256-symbol byte alphabet, then learns merges until the
    vocabulary reaches ``vocab_size`` or no pair occurs ``min_frequency``
    times -- so the result can be smaller than asked for on a small corpus.
    """
    texts = list(texts)
    tk = new_bpe_model()
    trainer = trainers.BpeTrainer(vocab_size=int(vocab_size), min_frequency=int(min_frequency),
                                  special_tokens=list(SPECIAL_TOKENS), show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(texts, trainer, length=len(texts))
    return BPETokenizer(tk)
