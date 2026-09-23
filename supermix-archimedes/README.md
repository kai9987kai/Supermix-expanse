# Vendored: supermix-archimedes

Python source (`archimedes/src/`, incl. `champion/`) and the replay corpus (`corpus/`) of
[kai9987kai/supermix-archimedes](https://github.com/kai9987kai/supermix-archimedes) at commit `1952cf1c92cb506fff60f9ee0c055c2365988e5e`, MIT (see `LICENSE`).
Expanse imports the Archimedes model, tokenizer and MiMoMix trunk from here, and trains on the replay corpus.
The Archimedes checkpoint itself is downloaded by `external/fetch_base.py` from
[Kai9987kai/archimedes-final-model](https://huggingface.co/Kai9987kai/archimedes-final-model).
