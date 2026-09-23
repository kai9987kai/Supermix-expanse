# Supermix Expanse v3 — subword tokenizer + more data (spec)

Goal: remove the word-level vocabulary bottleneck (every bio dev row has at least one `<unk>`, code writing is 0%) and train on much more verified data. v3 starts from a trained Expanse checkpoint (v1 schema `supermix-expanse-v1`, or v2 schema `supermix-expanse-v2` — both must work), swaps its tokenizer for a byte-level BPE, re-initialises only the tied embedding / LM-head rows, lengthens the context, and continues training on an enlarged corpus.

Repo: `C:/Users/kai99/Desktop/supermix-expanse/Supermix-expanse` (local branch `expanse-v3-subword`, do NOT push). Paths below are relative to it; `expanse_core.PATHS` resolves everything relative to the repo root. The v1 checkpoint is at `expanse/checkpoints/supermix_expanse.pt`, Archimedes at `external/base/supermix_archimedes.pt` (both hard links, present). CPU only, fp32 only, 16 GB RAM shared with a running training job: keep tests small, `torch.set_num_threads(2)`, never delete files you did not create.

## Facts

* Word tokenizer: `supermix-archimedes/archimedes/src/mimomix_text.py::WordTokenizer`. Specials ids 0-5 = `<pad> <bos> <eos> <unk> <user> <assistant>` (`SPECIAL_TOKENS`, constants `PAD BOS EOS UNK USER ASSISTANT`). `encode_turn(user, assistant=None)` -> `([BOS, USER] + encode(user) + [ASSISTANT] (+ encode(assistant) + [EOS]), prompt_len)`. Tokens carry their leading whitespace (`" word"`), digits are single tokens (`digit_tokens=True`).
* Only `embed_tokens.weight` and `lm_head.weight` (one tied Parameter, (V, 320)) depend on vocabulary size. Config: `vocab_size`, `max_position_embeddings=128`, `native_context=128`, `rope_scaling="none"`, `sliding_window=64`.
* Checkpoint tokenizer is stored as `payload["tokenizer"] = tok.to_dict()`; loaders call `text_utils.WordTokenizer.from_dict` (`expanse_core.load_expanse`, `consolidation_v2.load_v2`).
* Tokenizer consumers: `encode_turn`, `encode`, `decode`, `vocab_size`, `tokens`, `index`, `unknown_rate`, `digit_tokens`, `to_dict`/`from_dict` (grep `tok.` in `expanse/`).
* Verified data generators (Kai's MIT code) live in `C:/Users/kai99/Desktop/supermix-expanse/supermix-archimedes/models/supermix-v93/src/` (NOT yet in the repo): `build_omni_corpus.py --per_task N --seed S --unique --output F` (12 science tasks, solver-verified), `build_code_corpus.py --per_task N --seed S --unique --output F` (9 execution-verified code-reading tasks), `build_scratchpad_math.py --target N --seed S --output F`. Rows: `{"user","assistant","task",...}` in the house style. They take seconds for thousands of rows.

## A. Tokenizer + retokenisation (agent A)

1. `expanse/src/bpe_tokenizer.py`: `class BPETokenizer` wrapping `tokenizers.Tokenizer` — byte-level BPE; pre-tokenizer `Sequence([Digits(individual_digits=True), ByteLevel(add_prefix_space=False)])`; ByteLevel decoder; the 6 specials at ids 0-5 with the exact `SPECIAL_TOKENS` strings. Same duck-typed API as `WordTokenizer` (`tokens` = id-ordered list of *decoded* token strings, `index`, `vocab_size`, `digit_tokens=True`, `reverse_digits=False`, `encode`, `decode` (drops specials), `encode_turn` with identical structure, `unknown_rate` -> 0.0, `to_dict()` -> `{"kind": "bpe", "json": <tokenizer json>, "digit_tokens": True}`, `from_dict`). `train_bpe(texts, vocab_size=24000, min_frequency=2) -> BPETokenizer`. Round-trip `decode(encode(t)) == t` for ASCII/UTF-8 text.
2. `expanse_core.tokenizer_from_dict(d)`: `kind == "bpe"` -> `BPETokenizer.from_dict`, else `WordTokenizer.from_dict`. Use it in `load_expanse` and `consolidation_v2.load_v2`.
3. `expanse/retokenize_v3.py --inp <v1|v2 ckpt> --out expanse/checkpoints/supermix_expanse_v3_init.pt --vocab 24000 [--context auto|N]`:
   * Tokenizer training text = user + assistant of every **train-split** row the v3 trainer will use (replay corpus, `data/v3/fresh_*.jsonl`, `data/code_rows.jsonl`, `data/bio_rows.clean.jsonl`, `data/v3/connectome_rows_v3.jsonl` or `data/connectome_rows.jsonl`, fly rows from `train_expanse.make_fly_rows`). Never dev/held-out rows.
   * New embedding row for each non-special token, by priority: (a) the decoded string is an old token -> copy; (b) old-tokenise the string -> mean of its non-`<unk>` rows; (c) fragment: mean of old rows whose token contains the fragment (stripped, lowercase, ≥3 chars; ≤64 rows); (d) old mean + 0.1·std·N(0,1). Rows from (b)-(d) rescaled to the median old-row norm. Specials copy old rows 0-5. Report counts per rule.
   * Context: `auto` = smallest multiple of 32 ≥ the 99th percentile of encoded training-row length, capped at 256; set `vocab_size`, `max_position_embeddings`, `native_context`.
   * Keep the base schema (v1 -> `ec.save_expanse`; v2 -> `consolidation_v2.save_v2`) with the new tokenizer dict; receipt `retokenize_v3` block (vocab, rules counts, context, length percentiles old vs new, tokens-per-word, sha of base). Sanity: load back with `tokenizer_from_dict`, run one forward, greedy-decode 2 prompts.
   * Tests `expanse/tests/test_bpe_tokenizer.py`: specials at 0-5, round-trip, digits split, `encode_turn` structure and prompt_len, to/from dict, `tokenizer_from_dict` dispatch, tiny retokenise on a small synthetic model if feasible.

## B. Data + trainer (agent B)

1. Vendor the builders: copy the `.py` files the three builders need (follow their imports) from `supermix-archimedes/models/supermix-v93/src/` into the repo's `supermix-archimedes/models/supermix-v93/src/` (plus a README noting provenance/MIT).
2. `expanse/make_v3_data.py`: runs the three builders with new seeds (not 79/87/66) — targets `--omni 10000 --code 8000 --math 6000` (per_task = ceil(target / n_tasks)); drops rows whose `user` equals a replay-corpus prompt, an eval `heldout_problems` prompt, or a duplicate; assigns `split` = heldout for 3% by stable hash of `user`, else train; writes `data/v3/fresh_omni.jsonl`, `fresh_code.jsonl`, `fresh_math.jsonl` (`source` "fresh") + `data/v3/fresh.report.json`. Also regenerates connectome rows with more rows but **the same held-out cell types** as v1 (`connectome_text` with the same seed/held-out fraction, `--n_rows 12000`) into `data/v3/connectome_rows_v3.jsonl`; assert held-out type set equals v1's.
3. `train_expanse.py`:
   * `SOURCES` gains `"fresh"`; `load_corpus` reads `data/v3/fresh_*.jsonl` when present (train/dev by the existing hash split of train rows; heldout rows never trained) and prefers `data/v3/connectome_rows_v3.jsonl` when present (dev = held-out types, as before).
   * Load v1 **or** v2 checkpoints (`consolidation_v2.load_v2` for schema v2; v2 native-block params in the graft group) and save with the matching saver.
   * Tokenizer via `tokenizer_from_dict`; `--seq` defaults to the checkpoint's `max_position_embeddings`.
   * `kd_arch` is automatically disabled (logged) when the tokenizer is not Archimedes' word tokenizer (its logits index a different vocabulary).
   * `--freeze_trunk_steps N`: for the first N steps only the tied embedding/LM-head rows and graft/v2 params update (zero other grads); logged.
   * Keep everything else (resume, partials, eval, receipts) working; `--smoke` must pass on a v3-init checkpoint and on the v1 checkpoint.
4. `expanse/compare_models.py` (exists): score each model with `seq = max(128, model.config.max_position_embeddings)` and load tokenizers via `tokenizer_from_dict`.

Both: run your new tests + the existing fast suites you touched (`tests/test_connectome_full.py`, `tests/test_consolidation_v2.py`, relevant parts of `tests/test_expanse_core.py`). No long jobs; the orchestrator runs data generation, retokenisation and training.
