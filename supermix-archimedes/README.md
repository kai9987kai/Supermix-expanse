# Vendored: supermix-archimedes

Python source (`archimedes/src/`, incl. `champion/`) and the replay corpus (`corpus/`) of
[kai9987kai/supermix-archimedes](https://github.com/kai9987kai/supermix-archimedes) at commit `1952cf1c92cb506fff60f9ee0c055c2365988e5e`, MIT (see `LICENSE`).
Expanse imports the Archimedes model, tokenizer and MiMoMix trunk from here, and trains on the replay corpus.
`models/supermix-v93/src/` holds the three verified corpus builders (and the modules they import) from the same
commit; Expanse v3 reruns them with new seeds (`expanse/make_v3_data.py`, see that folder's README).
The Archimedes checkpoint itself is downloaded by `external/fetch_base.py` from
[Kai9987kai/archimedes-final-model](https://huggingface.co/Kai9987kai/archimedes-final-model).
