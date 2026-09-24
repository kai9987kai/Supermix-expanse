# Expanse model comparison

Same rows for every model (v1 training dev split) and the same generation items. **nats/char** is comparable across tokenizers; nats/token only within one tokenizer.

| metric | v1 | v2 | v2_gates0 |
|---|---|---|---|
| nats/char (rows all models cover) replay | 0.1656 | 0.1316 | 0.1324 |
| nats/char (rows all models cover) fly | 0.3931 | 0.3685 | 0.3686 |
| nats/char (rows all models cover) code | 1.2081 | 1.1809 | 1.1811 |
| nats/char (rows all models cover) bio | - | - | - |
| nats/char (rows all models cover) connectome | 0.1810 | 0.1756 | 0.1757 |
| nats/char (all rows) replay | 0.1656 | 0.1316 | 0.1324 |
| nats/char (all rows) fly | 0.3931 | 0.3685 | 0.3686 |
| nats/char (all rows) code | 1.2184 | 1.1914 | 1.1916 |
| nats/char (all rows) bio | 0.9898 | 0.9670 | 0.9672 |
| nats/char (all rows) connectome | 0.1820 | 0.1766 | 0.1767 |
| nats/token (all rows) replay | 0.4182 | 0.3323 | 0.3344 |
| nats/token (all rows) fly | 1.1983 | 1.1234 | 1.1238 |
| nats/token (all rows) code | 4.0362 | 3.9468 | 3.9474 |
| nats/token (all rows) bio | 4.9488 | 4.8350 | 4.8361 |
| nats/token (all rows) connectome | 0.6135 | 0.5951 | 0.5954 |
| exact-answer accuracy | 0.2000 | 0.2800 | 0.2800 |
| code pass rate | 0.0000 | 0.0000 | 0.0000 |
| bio token-F1 | 0.2850 | 0.3425 | 0.3415 |
| PubMedQA accuracy | 0.0000 | 0.0000 | 0.0000 |
| connectome exact match | 0.1600 | 0.1600 | 0.1600 |
| connectome token-F1 | 0.6738 | 0.6738 | 0.6738 |

Generation items per metric: {"exact": 25, "code": 25, "bio": 25, "pubmedqa": 25, "connectome": 25}; greedy, max 64 new tokens.
