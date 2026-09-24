# Vendored: Supermix v93 corpus builders

The three verified data generators Expanse v3 draws fresh rows from (`expanse/make_v3_data.py`,
`expanse/V3_DESIGN.md` B.1), plus exactly the modules they import:

| file | role |
|---|---|
| `build_omni_corpus.py` | 12 science tasks (v89 set); every worked answer is re-derived by `nexus_solver` and dropped on disagreement |
| `build_code_corpus.py` | 9 code-reading tasks; every answer comes from executing the snippet the prompt shows (restricted namespace, timeout) |
| `build_scratchpad_math.py` | place-value "show your working" arithmetic |
| `nexus_solver.py`, `science_plan.py` | the deterministic solver `build_omni_corpus` verifies against |
| `natural_phrasings.py` | the wide phrasing bank behind `build_omni_corpus --natural_phrasings` |
| `mimomix_text.py` | lazy import of `build_omni_corpus.token_budget_report` only (identical to `archimedes/src/mimomix_text.py`) |

Copied byte-for-byte from [kai9987kai/supermix-archimedes](https://github.com/kai9987kai/supermix-archimedes)
`models/supermix-v93/src/` at commit `1952cf1c92cb506fff60f9ee0c055c2365988e5e` (the same commit as the rest of
this folder), MIT licensed -- see `../../../LICENSE` (Copyright (c) 2026 Kai Piper). Nothing here is modified;
fix upstream and re-vendor rather than editing in place, so the rows stay reproducible from that commit.

SHA-256 at vendoring:

    09ab0de0eca1a52ae9c62cecc9679af681dc4ac55194abb39724e4960ee41692  build_omni_corpus.py
    7f019386406048e78c930594b74a5fa6dd1316736b395e2714cbd7ed9fd21730  build_code_corpus.py
    379a2e2251a100717d0c869864303537c42d991cea4978fc2befe6244e129c90  build_scratchpad_math.py
    8d16ebb465d45fef7915b5009bde8daa0299c8176324563e8ec9ab2c683ee1bb  nexus_solver.py
    effb37997081491c5f63af595651120b97477ad903dd5b5be8d2bc514b9fe373  science_plan.py
    db633eeb32d7f84e6f41dfb9b6210d427aac0b98674af8af9f037748e050e32c  natural_phrasings.py
    2333da1daf0c22f05ef0fe022d40b9c2fc962ddfef335ac67b4277746299cab9  mimomix_text.py

(LF line endings; `make_v3_data.py` records each builder's hash in `data/v3/fresh.report.json`.)
