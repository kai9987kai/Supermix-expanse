"""Text tokenisation and corpus construction for the v57 talking MiMoMix model.

V53 built a full decoder-only language model and never trained it on language.
Its API says so in its own docstring: the backends are randomly initialised and
their text is noise. This module supplies the missing half -- a tokenizer and a
corpus -- so `MiMoMixModel` can be trained to actually generate text.

## The tokenizer

`WordTokenizer` is a whitespace-preserving word tokenizer, not BPE. Each token
carries its own leading whitespace, so `"".join(tokens)` reconstructs the input
exactly and decoding needs no spacing heuristics. `assert_roundtrip` pins that.

It is deliberately simple because the corpus is: the local chat database uses
292 distinct word types across 4.6M word tokens. A subword vocabulary would buy
nothing on data with that shape, and byte-level would multiply the sequence
length by five for no gain.

**The vocabulary is the ceiling on what the model can say.** A word the
tokenizer has never seen encodes to `<unk>` and can never be generated. That is a
property of the available corpus, not a design choice, and
`vocabulary_report` reports it so the limit travels with the checkpoint.

## The corpus

`databases/llm_chat.db` holds 120,000 `(user_text, response_text)` pairs, 21M
characters. It is templated: 37,543 distinct responses over 120,000 rows. A model
trained on it learns to hold a turn in that register -- coding-assistant small
talk -- and learns nothing else. It has no world knowledge to learn from here.

Prompt tokens are masked out of the loss with `-100`, which
`torch.nn.functional.cross_entropy` ignores by default, so the model is trained to
*produce* replies rather than to reproduce the user's own words.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "ASSISTANT",
    "BOS",
    "EOS",
    "PAD",
    "UNK",
    "USER",
    "ChatCorpus",
    "WordTokenizer",
    "assert_roundtrip",
    "build_training_tensors",
    "load_chat_pairs",
    "load_chat_pairs_jsonl",
    "reverse_digit_runs",
    "turn_alignment_report",
]

PAD, BOS, EOS, UNK, USER, ASSISTANT = 0, 1, 2, 3, 4, 5
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>")

#: Each alternative consumes its own leading whitespace, so joining the matches
#: reproduces the input byte for byte. The trailing ``\s+`` catches a run of
#: whitespace at the end that no token would otherwise absorb.
TOKEN_PATTERN = re.compile(r"\s*[A-Za-z]+(?:'[A-Za-z]+)?|\s*\d+|\s*[^\sA-Za-z\d]|\s+")

#: The same pattern with digit runs split into single digits.
#:
#: The default `\s*\d+` makes "498" one opaque token, which puts arithmetic out
#: of reach in principle: answering "498 - 419" would require a memorised lookup
#: from token(498) x token(419) to token(79), with no way to see that 498 is made
#: of a 4, a 9 and an 8. Measured on a 240,000-row arithmetic corpus, **8,588 of
#: 9,058 distinct tokens (94.8%) were numbers**, and a model trained under it
#: scored 1.7% on arithmetic -- the same 1.7% on problems it had trained on,
#: because there is nothing there to memorise either.
#:
#: Splitting digits replaces those 8,588 symbols with ten, and makes column-wise
#: arithmetic something a model can represent. Roundtrip is preserved: each digit
#: still carries its own leading whitespace, so joining the matches reproduces
#: the input byte for byte.
DIGIT_TOKEN_PATTERN = re.compile(r"\s*[A-Za-z]+(?:'[A-Za-z]+)?|\s*\d|\s*[^\sA-Za-z\d]|\s+")

#: A maximal run of digits, for the least-significant-digit-first option below.
DIGIT_RUN_PATTERN = re.compile(r"\d+")


def reverse_digit_runs(text: str) -> str:
    """Reverse every maximal run of digits in ``text``.

    ``"124 + 72 = 196"`` becomes ``"421 + 27 = 691"``. The transform is an
    **involution** -- applying it twice restores the input -- which is what lets
    :meth:`WordTokenizer.decode` undo it with the same function, and it never
    changes how the tokenizer's regex segments the string, because a reversed
    digit run is still a digit run of the same length.
    """

    return DIGIT_RUN_PATTERN.sub(lambda m: m.group(0)[::-1], text)


class WordTokenizer:
    """Whitespace-preserving word tokenizer with a fixed vocabulary."""

    def __init__(self, tokens: Sequence[str], digit_tokens: bool = False,
                 reverse_digits: bool = False):
        self.tokens: List[str] = list(SPECIAL_TOKENS) + [
            token for token in tokens if token not in SPECIAL_TOKENS
        ]
        self.index: Dict[str, int] = {token: i for i, token in enumerate(self.tokens)}
        #: Whether numbers are split into digits. Travels with the checkpoint,
        #: because a tokenizer reloaded under the other setting would segment
        #: every number differently and silently mis-encode the vocabulary.
        self.digit_tokens = bool(digit_tokens)
        #: Whether numbers are written least-significant digit first.
        #:
        #: Lee et al. 2023 (arXiv 2307.03381) report a NanoGPT of 10.6M
        #: parameters reaching 100% on three-digit addition at roughly 2,500
        #: training samples with reversed digits, and never reaching it without.
        #: The reason offered is that addition is computed right to left, so a
        #: left-to-right decoder writing "1" of "124" first must have already
        #: resolved every carry it has not yet emitted.
        #:
        #: Doing it here rather than in the corpus builders means the builders,
        #: `answer_check` and `eval_problem_solving` need no change at all: the
        #: reversal is applied on the way in and undone on the way out, so every
        #: string outside this class is in ordinary reading order.
        #:
        #: **Unmeasured in this repository.** No Supermix run has trained under
        #: it. It travels with the checkpoint for the same reason
        #: ``digit_tokens`` does -- a tokenizer reloaded under the other setting
        #: would silently mis-encode every number in the corpus.
        self.reverse_digits = bool(reverse_digits)

    @property
    def pattern(self):
        return DIGIT_TOKEN_PATTERN if self.digit_tokens else TOKEN_PATTERN

    def _pieces(self, text: str) -> List[str]:
        """Tokenize, applying the digit reversal first when it is enabled."""

        if self.reverse_digits:
            text = reverse_digit_runs(text)
        return self.pattern.findall(text)

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, texts: Iterable[str], max_vocab: int = 16384, min_count: int = 1,
              digit_tokens: bool = False, reverse_digits: bool = False) -> "WordTokenizer":
        """Build a vocabulary, covering each word with and without leading space.

        Tokens carry their own leading whitespace, so ``"Got"`` and ``" Got"`` are
        different strings. A word seen only mid-sentence therefore has no
        sentence-initial form in the vocabulary, and encoding a string that
        *starts* with it yields `<unk>`.

        That is not hypothetical: building this vocabulary from
        ``user + " " + assistant`` put only the space-prefixed forms in it, and
        17,709 of 19,600 replies then began with `<unk>`. Worse than the lost
        text, it flattered perplexity, because `<unk>` became a trivially
        predictable target at the position after `<assistant>`.

        So every kept token is admitted in both forms. It roughly doubles a
        whitespace-heavy vocabulary and removes the failure entirely.
        """

        pattern = DIGIT_TOKEN_PATTERN if digit_tokens else TOKEN_PATTERN
        counter: Counter = Counter()
        for text in texts:
            # Count the *encoded* form, or a reversed-digit vocabulary would be
            # built over strings the encoder never produces. With whole-number
            # tokens that matters: "124" and "421" are different symbols.
            if reverse_digits:
                text = reverse_digit_runs(text)
            counter.update(pattern.findall(text))

        kept: List[str] = []
        seen = set()
        for token, count in counter.most_common():
            if count < min_count:
                break
            for variant in (token, token.lstrip()):
                if variant and variant not in seen:
                    seen.add(variant)
                    kept.append(variant)
            if len(kept) >= max_vocab:
                break
        return cls(kept[:max_vocab], digit_tokens=digit_tokens,
                   reverse_digits=reverse_digits)

    @classmethod
    def extend(cls, base: "WordTokenizer", texts: Iterable[str], max_new: int,
               min_count: int = 1) -> "WordTokenizer":
        """Append new tokens to ``base`` without moving any existing id (v93 D5).

        `build` orders ids by frequency, so building over a corpus that differs
        by a single row can renumber thousands of tokens, and
        `load_initial_weights` then refuses the checkpoint ("different
        vocabulary") -- correctly, because id *n* would denote a different
        word. A warm start on a bigger corpus therefore cannot rebuild; it has
        to *extend*: ``base.tokens`` is copied as an exact prefix (same ids,
        same order, the specials included) and only tokens the base lacks are
        appended, in the same most-common order and with the same
        both-spacing-forms admission `build` uses, up to ``max_new`` of them.

        Segmentation follows ``base.pattern`` and ``base.reverse_digits``, so
        the new tokens are counted over the strings the encoder will actually
        produce. ``to_dict``/``from_dict`` are untouched: an extended
        tokenizer serialises exactly like a built one, and nothing records
        where the prefix ended -- the checkpoint it was extended from does.

        A corpus the base already covers adds nothing, which is what makes a
        crash resume of an extended run rebuild the identical token list.
        """

        counter: Counter = Counter()
        for text in texts:
            if base.reverse_digits:
                text = reverse_digit_runs(text)
            counter.update(base.pattern.findall(text))

        kept: List[str] = list(base.tokens)
        seen = set(kept)
        added = 0
        max_new = max(0, int(max_new))
        for token, count in counter.most_common():
            if count < min_count or added >= max_new:
                break
            for variant in (token, token.lstrip()):
                if variant and variant not in seen and added < max_new:
                    seen.add(variant)
                    kept.append(variant)
                    added += 1

        # Built the way `from_dict` builds, not through `__init__`: the
        # constructor re-derives the special-token prefix, and the contract
        # here is that the id list is a byte-identical prefix copy.
        instance = cls.__new__(cls)
        instance.tokens = kept
        instance.index = {token: i for i, token in enumerate(kept)}
        instance.digit_tokens = bool(base.digit_tokens)
        instance.reverse_digits = bool(base.reverse_digits)
        return instance

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "tokens": self.tokens,
            "digit_tokens": self.digit_tokens,
        }
        # Written only when enabled, so a default-built tokenizer serialises
        # byte-identically to every checkpoint saved before v82.
        if self.reverse_digits:
            payload["reverse_digits"] = True
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "WordTokenizer":
        tokens = list(payload["tokens"])  # type: ignore[arg-type]
        instance = cls.__new__(cls)
        instance.tokens = tokens
        instance.index = {token: i for i, token in enumerate(tokens)}
        # Absent in every pre-v65 checkpoint, which all used whole-number tokens.
        instance.digit_tokens = bool(payload.get("digit_tokens", False))
        # Absent in every pre-v82 checkpoint, which all wrote numbers forwards.
        instance.reverse_digits = bool(payload.get("reverse_digits", False))
        return instance

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "WordTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- encoding ----------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        return [self.index.get(piece, UNK) for piece in self._pieces(text)]

    def decode(self, ids: Sequence[int]) -> str:
        out: List[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id < len(SPECIAL_TOKENS):
                continue  # specials are structure, not text
            if 0 <= token_id < len(self.tokens):
                out.append(self.tokens[token_id])
        text = "".join(out)
        if self.reverse_digits:
            # The same involution that encode applied, undoing it. A partially
            # generated number decodes as a partially reversed one, which is
            # the honest rendering: the model has not written the rest yet.
            text = reverse_digit_runs(text)
        return text

    def unknown_rate(self, text: str) -> float:
        pieces = self._pieces(text)
        if not pieces:
            return 0.0
        return sum(1 for piece in pieces if piece not in self.index) / len(pieces)

    def vocabulary_report(self, texts: Sequence[str]) -> Dict[str, object]:
        """What the tokenizer can and cannot represent on a given sample."""

        total = 0
        unknown = 0
        for text in texts:
            pieces = self._pieces(text)
            total += len(pieces)
            unknown += sum(1 for piece in pieces if piece not in self.index)
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": len(SPECIAL_TOKENS),
            "digit_tokens": self.digit_tokens,
            "reverse_digits": self.reverse_digits,
            "sampled_tokens": total,
            "unknown_tokens": unknown,
            "coverage": round(1.0 - unknown / max(1, total), 6),
            "note": (
                "a word outside this vocabulary encodes to <unk> and can never be "
                "generated; the vocabulary is the ceiling on what the model can say"
            ),
        }

    # -- chat formatting ---------------------------------------------------

    def encode_turn(self, user: str, assistant: Optional[str] = None) -> Tuple[List[int], int]:
        """Return ``(ids, prompt_length)`` for one training turn.

        The prompt is ``<bos><user> ... <assistant>``; ``prompt_length`` is where
        the reply starts, so the caller can mask the prompt out of the loss.
        """

        ids = [BOS, USER] + self.encode(user) + [ASSISTANT]
        prompt_length = len(ids)
        if assistant is not None:
            ids = ids + self.encode(assistant) + [EOS]
        return ids, prompt_length


def assert_roundtrip(tokenizer: WordTokenizer, samples: Sequence[str]) -> None:
    """Encoding then decoding must reproduce the text exactly.

    Only holds for text whose tokens are all in the vocabulary; an out-of-vocab
    word becomes `<unk>` and is genuinely lost, which is the point of tracking it.
    """

    for text in samples:
        if tokenizer.unknown_rate(text) > 0:
            continue
        restored = tokenizer.decode(tokenizer.encode(text))
        if restored != text:
            raise AssertionError(f"round trip changed the text:\n  in  {text!r}\n  out {restored!r}")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass
class ChatCorpus:
    """Train/validation split of `(user, assistant)` pairs, split by row."""

    train: List[Tuple[str, str]]
    validation: List[Tuple[str, str]]
    source: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "train_pairs": len(self.train),
            "validation_pairs": len(self.validation),
            "train_characters": sum(len(u) + len(a) for u, a in self.train),
        }


def load_chat_pairs_jsonl(
    path: str,
    limit: Optional[int] = None,
    validation_fraction: float = 0.02,
    seed: int = 57,
    min_response_characters: int = 8,
    user_key: str = "user",
    assistant_key: str = "assistant",
) -> ChatCorpus:
    """Read `(user, assistant)` pairs from a JSONL corpus.

    Same contract and same row split as :func:`load_chat_pairs`, which reads
    SQLite. It exists because the corpora with real linguistic diversity in this
    repo are JSONL, not SQLite, and the 292-word ceiling v58 names as its binding
    constraint is a property of the one database the trainers happened to read --
    not of the text available on disk.

    Malformed lines are skipped rather than fatal: these files are pipeline
    output, and one bad row should not cost a training run. The count of skipped
    lines is not returned, so callers that care should validate separately.
    """

    pairs: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            user = str(record.get(user_key) or "").strip()
            assistant = str(record.get(assistant_key) or "").strip()
            if not user or len(assistant) < min_response_characters:
                continue
            pairs.append((user, assistant))
            if limit and len(pairs) >= limit:
                break

    if not pairs:
        raise ValueError(f"no usable (user, assistant) pairs in {path!r}")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator).tolist()
    pairs = [pairs[i] for i in order]
    cut = max(1, int(len(pairs) * validation_fraction))
    return ChatCorpus(train=pairs[cut:], validation=pairs[:cut], source=path)


def load_chat_pairs(
    database: str,
    limit: Optional[int] = None,
    validation_fraction: float = 0.02,
    seed: int = 57,
    min_response_characters: int = 8,
) -> ChatCorpus:
    """Read `(user_text, response_text)` pairs out of the local chat database.

    The split is by **row**, and rows are shuffled first, so a validation pair's
    exact response can still appear in training -- the corpus is templated and
    only 37,543 of the 120,000 responses are distinct. Held-out perplexity here
    therefore measures fit to the template distribution, not generalisation to
    unseen language, and the training receipt says so.
    """

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        query = "SELECT user_text, response_text FROM llm_entries"
        if limit:
            query += f" LIMIT {int(limit)}"
        pairs = [
            (user.strip(), response.strip())
            for user, response in connection.execute(query)
            if user and response and len(response.strip()) >= min_response_characters
        ]
    finally:
        connection.close()

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator).tolist()
    pairs = [pairs[i] for i in order]
    cut = max(1, int(len(pairs) * validation_fraction))
    return ChatCorpus(train=pairs[cut:], validation=pairs[:cut], source=database)


def build_training_tensors(
    pairs: Sequence[Tuple[str, str]],
    tokenizer: WordTokenizer,
    sequence_length: int = 128,
    mask_prompt: bool = True,
    turn_aligned: bool = False,
    stats: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack turns into fixed-length `(input_ids, labels)` blocks.

    With ``turn_aligned=False`` (the default, and what every result up to v62 was
    produced under) turns are concatenated into one stream and chopped on a fixed
    stride, so no compute is wasted on padding.

    That stride is blind to turn boundaries, and the cost is not small. Measured
    on the v63 corpus at ``sequence_length=128``: **56.0% of supervised tokens
    land in a block that does not contain their own prompt**, and 879 of 21,673
    blocks contain no turn start at all. Over half the training signal therefore
    teaches the model to continue a reply without having seen the question, which
    is the same thing as teaching it to emit the corpus's most likely reply
    unconditionally.

    On a templated corpus that is nearly harmless -- the modal reply is usually
    the right one -- which is why v57 through v60 trained happily this way. On a
    corpus spanning several domains it is fatal, and it shows up exactly as
    observed in v62 and v63: fluent sentences, ignoring the prompt.

    ``turn_aligned=True`` gives every turn its own block, padded to
    ``sequence_length``, so no supervised token is ever orphaned from its prompt.
    It costs padding compute and drops turns longer than the block, both of which
    the receipt should record.

    Pass a dict as ``stats`` to find out how many turns were dropped. The
    docstring above promised the receipt should record it and nothing ever
    returned the number, so an over-length task could vanish from training with
    no trace -- which is exactly how v67 lost its six-value ``average`` rows.
    Leaving ``stats`` at ``None`` keeps the old two-value return and the old
    behaviour untouched.
    """

    if turn_aligned:
        return _build_turn_aligned_tensors(
            pairs, tokenizer, sequence_length, mask_prompt, stats
        )

    if stats is not None:
        stats.update({
            "packing": "stream",
            "turns": len(pairs),
            "dropped_over_length": 0,
            "dropped_fraction": 0.0,
            "note": "stream packing truncates the tail of the corpus, not whole turns",
        })

    stream_ids: List[int] = []
    stream_labels: List[int] = []
    for user, assistant in pairs:
        ids, prompt_length = tokenizer.encode_turn(user, assistant)
        labels = list(ids)
        if mask_prompt:
            for position in range(min(prompt_length, len(labels))):
                labels[position] = -100
        stream_ids.extend(ids)
        stream_labels.extend(labels)

    usable = (len(stream_ids) // sequence_length) * sequence_length
    if usable == 0:
        raise ValueError("not enough tokens for a single sequence")
    dtype = compact_dtype(tokenizer.vocab_size)
    input_ids = torch.tensor(stream_ids[:usable], dtype=dtype).view(-1, sequence_length)
    labels = torch.tensor(stream_labels[:usable], dtype=dtype).view(-1, sequence_length)
    return input_ids, labels


#: Largest token id storable in int16, leaving room for the -100 ignore label.
_INT16_LIMIT = 32000


def compact_dtype(vocab_size: int) -> torch.dtype:
    """The narrowest integer type that can hold this vocabulary.

    The packed corpus is the single largest allocation a run makes, and it was
    stored as int64 -- 8 bytes for a token id below 9,000. v79 held
    866,748 x 128 x 2 tensors, so **1.78 GB** of the trainer's 4.44 GB
    footprint, on a 15.6 GB machine that was already 25.6 GB committed. It
    spent hours at 17 s/step against a 4 s/step norm, faulting its own corpus
    back from the pagefile.

    int16 holds every id in this repository's vocabularies (8,551 for v79,
    16,384 at the `--max_vocab` ceiling) and the -100 ignore label, and cuts
    that 1.78 GB to **0.44 GB**. int32 is the fallback for a vocabulary that
    would not fit, which still halves it.

    Batches are cast to long on the way into the model, which costs a copy of
    16 x 128 values per step -- nothing against the pagefile traffic it
    removes.
    """

    return torch.int16 if vocab_size < _INT16_LIMIT else torch.int32


def _build_turn_aligned_tensors(
    pairs: Sequence[Tuple[str, str]],
    tokenizer: WordTokenizer,
    sequence_length: int,
    mask_prompt: bool,
    stats: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One turn per block, padded, so every reply sees its own prompt.

    Turns that do not fit are dropped rather than truncated. A truncated reply
    would train the model to stop mid-sentence, and a truncated *prompt* would
    reintroduce the very conditioning gap this exists to close.

    The drop is not neutral. It removes a task's *longest* rows, so a task whose
    format is near the limit loses its hardest instances and the model meets
    them for the first time at evaluation. ``stats`` carries the count out.
    """

    blocks: List[List[int]] = []
    block_labels: List[List[int]] = []
    dropped = 0
    for user, assistant in pairs:
        ids, prompt_length = tokenizer.encode_turn(user, assistant)
        if len(ids) > sequence_length:
            dropped += 1
            continue
        labels = list(ids)
        if mask_prompt:
            for position in range(min(prompt_length, len(labels))):
                labels[position] = -100
        padding = sequence_length - len(ids)
        blocks.append(ids + [PAD] * padding)
        block_labels.append(labels + [-100] * padding)

    if stats is not None:
        stats.update({
            "packing": "turn_aligned",
            "turns": len(pairs),
            "kept": len(blocks),
            "dropped_over_length": dropped,
            "dropped_fraction": round(dropped / max(1, len(pairs)), 6),
            "sequence_length": sequence_length,
            "note": (
                "dropped turns are the longest ones, so a task near the limit "
                "loses its hardest rows and meets them first at evaluation"
            ),
        })

    if not blocks:
        raise ValueError(
            f"no turn fits in {sequence_length} tokens; raise sequence_length"
        )
    dtype = compact_dtype(tokenizer.vocab_size)
    return (
        torch.tensor(blocks, dtype=dtype),
        torch.tensor(block_labels, dtype=dtype),
    )


def turn_alignment_report(
    pairs: Sequence[Tuple[str, str]],
    tokenizer: WordTokenizer,
    sequence_length: int = 128,
) -> Dict[str, object]:
    """What turn-aligned packing would drop, without building the tensors.

    Same rule as :func:`_build_turn_aligned_tensors` -- a turn longer than the
    block is dropped -- so a corpus builder can be told the cost of a format
    change before a run pays for it.
    """

    lengths = [len(tokenizer.encode_turn(user, assistant)[0]) for user, assistant in pairs]
    if not lengths:
        return {"turns": 0, "dropped_over_length": 0, "dropped_fraction": 0.0}
    ordered = sorted(lengths)
    dropped = sum(1 for length in ordered if length > sequence_length)
    return {
        "turns": len(ordered),
        "sequence_length": sequence_length,
        "median_turn_tokens": ordered[len(ordered) // 2],
        "p95_turn_tokens": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_turn_tokens": ordered[-1],
        "dropped_over_length": dropped,
        "dropped_fraction": round(dropped / len(ordered), 6),
    }
