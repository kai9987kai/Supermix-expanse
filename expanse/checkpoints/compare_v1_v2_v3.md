# Expanse model comparison

Same rows for every model (v1 training dev split) and the same generation items. **nats/char** is comparable across tokenizers; nats/token only within one tokenizer.

| metric | v1 | v2 | v2_gates0 | v3 |
|---|---|---|---|---|
| nats/char (rows all models cover) replay | 0.1656 | 0.1316 | 0.1324 | 0.0333 |
| nats/char (rows all models cover) fly | 0.3931 | 0.3685 | 0.3686 | 0.3136 |
| nats/char (rows all models cover) code | 1.1455 | 1.1178 | 1.1181 | 0.4733 |
| nats/char (rows all models cover) bio | - | - | - | - |
| nats/char (rows all models cover) connectome | 0.1810 | 0.1756 | 0.1757 | 0.1241 |
| nats/char (all rows) replay | 0.1656 | 0.1316 | 0.1324 | 0.0333 |
| nats/char (all rows) fly | 0.3931 | 0.3685 | 0.3686 | 0.3136 |
| nats/char (all rows) code | 1.1754 | 1.1481 | 1.1484 | 0.5147 |
| nats/char (all rows) bio | 0.9958 | 0.9734 | 0.9736 | 1.3696 |
| nats/char (all rows) connectome | 0.1820 | 0.1766 | 0.1767 | 0.1262 |
| nats/token (all rows) replay | 0.4182 | 0.3323 | 0.3344 | 0.0841 |
| nats/token (all rows) fly | 1.1983 | 1.1234 | 1.1238 | 0.9562 |
| nats/token (all rows) code | 3.9057 | 3.8150 | 3.8161 | 1.8432 |
| nats/token (all rows) bio | 4.9714 | 4.8598 | 4.8609 | 5.8276 |
| nats/token (all rows) connectome | 0.6135 | 0.5951 | 0.5954 | 0.4254 |
| exact-answer accuracy | 0.2000 | 0.2800 | 0.2800 | 0.3200 |
| code pass rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bio token-F1 | 0.2850 | 0.3425 | 0.3415 | 0.2203 |
| PubMedQA accuracy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| connectome exact match | 0.1600 | 0.1600 | 0.1600 | 0.3600 |
| connectome token-F1 | 0.6738 | 0.6738 | 0.6738 | 0.8136 |

Generation items per metric: {"exact": 25, "code": 25, "bio": 25, "pubmedqa": 25, "connectome": 25}; greedy, max 64 new tokens.
Scoring length / tokenizer per model: v1 128 (word), v2 128 (word), v2_gates0 128 (word), v3 128 (bpe).
