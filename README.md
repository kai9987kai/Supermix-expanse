# Supermix Expanse

[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kai9987kai%2Fsupermix--expanse-FFD21E)](https://huggingface.co/Kai9987kai/supermix-expanse)

**Supermix Expanse** grafts, distils and fine-tunes five sources into one 132.6M-parameter PyTorch model, built and trained entirely on a laptop CPU:

| Source | How it enters Expanse |
|---|---|
| [Archimedes final](https://huggingface.co/Kai9987kai/archimedes-final-model) ([repo](https://github.com/kai9987kai/supermix-archimedes)) — v93 trunk, v87 experts, FlyCore, Omni v48/v38 | the trunk everything is grafted onto; a frozen copy is the anti-forgetting teacher |
| [Omni Collective v7 Frontier](https://huggingface.co/Kai9987kai/supermix-omni-collective-v7-frontier) | frozen 77.6M branch bridged into the residual stream + intent/domain distillation |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 1,500 sandbox-verified code rows + 16 grafted MoE experts from its layer-0/1 MLP neurons |
| [BioMedLM 2.7B](https://huggingface.co/stanford-crfm/BioMedLM) | ~1,400 filtered biomedical rows + 16 grafted MoE experts from its layer-0/1 MLP neurons |
| The **full Janelia male CNS v1.0 connectome** ([natverse/malecns](https://github.com/natverse/malecns), [male-cns.janelia.org](https://male-cns.janelia.org/)) | a per-token recurrent core wired as the whole male fly CNS — all 11,751 cell types, all 3,830,931 type→type edges — plus 6,667 connectome fact rows |

**➡ The trained model is on Hugging Face: <https://huggingface.co/Kai9987kai/supermix-expanse>** (checkpoint, model card, receipts). This repository holds the full project: code, the teacher corpora actually used, the connectome graph, receipts and logs.

> **Status: experimental research model.** Expanse keeps and improves Archimedes' own skills, but its new code-writing and biomedical abilities are weak after one CPU training run — see [Results](#results). Qwen has no Qwen3-Coder at 7-8B, so the official Qwen2.5-Coder-7B-Instruct was used; BioMedLM replaced BioMistral-7B.

## How it works

```mermaid
flowchart TD
    P[Prompt] --> T[Word tokenizer + embeddings<br/>10,951 words, 320-d]
    T --> L02[Trunk layers 0-2<br/>attention + MoE, 88 slots in L1-L2]
    L02 --> G{{Graft site after layer 2<br/>each graft writes through a gate that starts at 0}}
    G --> F[FlyCore<br/>22 fly brains, causal]
    G --> O[Omni v48/v38<br/>prompt encoders]
    G --> O7[Omni v7<br/>frozen 77.6M branch]
    G --> C[Male CNS core<br/>11,751 cell types, 3.83M edges]
    F & O & O7 & C --> L35[Trunk layers 3-5<br/>global attention + MoE]
    L35 --> K[Thinking core + v93 CNS] --> N[Next word]
```

* **Male CNS core** (`expanse/src/connectome_full.py`). Each token's layer-2 state drives the 1,277 sensory / visual / ascending cell types; activity runs 4 recurrent steps through every type→type edge of the male CNS v1.0 (122.3M typed synapses), each weighted by its **measured input fraction** and signed by the presynaptic neurotransmitter (Dale's law); the 713 descending / motor / efferent types are read out and written back through a zero-initialised gate. The wiring is fixed from data; only per-type gain, bias and leak (plus in/out projections) learn. Per-position, so causal and KV-cache exact. Sparse CSR in 128-column slabs: ~0.33 s forward + 0.35 s backward per 1,024 tokens on CPU.
* **Grafted experts** (`expanse/src/donor_graft.py`). New MoE slots (72 → 88 in layers 1-2) filled with 96-neuron experts carved from Qwen-Coder / BioMedLM layer-0/1 MLPs via ridge maps between single-token states, fold-in init (GELU emulated exactly as `silu(1.702z)/1.702`), local function-fit refinement, load-matched wake biases, dormant until 10-50% of training.
* **Omni v7 branch** (`expanse/src/omni_v7_branch.py`). The frozen v7 net reads the prompt once; its 988-d conclusion is bridged in, and its intent/domain predictions are distilled into the trunk.
* **FlyCore causality fix.** Archimedes' `FlyCore.sense` averaged over *all* positions (future tokens and padding included) — a future-token leak. Expanse senses per position.
* Every graft is born function-preserving: with the fly graft off, the grafted model reproduces Archimedes' logits with max |Δ| = 0.0.

[`expanse/DESIGN.md`](expanse/DESIGN.md) is the full spec.

## Results

Dev loss per source (nats/token on reply tokens; connectome dev = held-out *cell types*):

| source | Archimedes | Expanse |
|---|---|---|
| Archimedes replay (science / code tracing / arithmetic) | 0.787 | **0.418** |
| male-CNS connectome facts (rows both tokenizers cover) | 3.522 | **0.571** |
| fly rows | 4.609 | **1.198** |
| Qwen code rows | — (vocabulary cannot encode them) | 4.04 (from 9.86) |
| BioMedLM bio rows | — | 4.95 (from 8.25) |

* Agreement with Omni v7: intent 72.7%, domain 73.1%. Final gate magnitudes: CNS core 0.086, Omni v7 0.076.
* **Honest graft numbers:** teacher→student representation maps are weak (held-out R² 0.05-0.15); only **1 of 32** grafted experts reproduces its teacher neuron group on held-out tokens (R² 0.21). Most transferred knowledge comes from distillation.
* **Teacher data quality:** code answers kept only if they pass our own tests in a sandbox (92.8% pass); BioMedLM definitions kept if Qwen judged them accurate (94% — a lenient judge); PubMedQA answers capped because BioMedLM answers almost everything "yes" (59.3% on held-out vs a 60.5% always-yes baseline).
* Sample answers: arithmetic and code-tracing prompts are answered correctly in the house style; connectome questions are answered in the right format but sometimes to the wrong question; writing new functions and defining biomedical terms is not usable yet.


**Held-out generation** (greedy; 150 items per metric; full report: [`expanse/checkpoints/eval_report.md`](expanse/checkpoints/eval_report.md)):

| metric | Archimedes | Expanse |
|---|---|---|
| replay-style problems with freshly drawn numbers (exact answer) | 21.3% | 22.0% |
| code pass rate (our sandbox tests) | 0% | 0% |
| biomedical token-F1 vs BioMedLM's answers | 0.088 | **0.313** |
| PubMedQA accuracy (always-yes baseline 60.5%) | 0% | 0% (never answers yes/no/maybe) |
| connectome exact match on held-out cell types | 0% | **17.3%** |
| connectome token-F1 | 0.340 | **0.681** |

**Ablations** (Expanse dev loss with one component removed at inference; code = Expanse-covered rows, others = rows both models cover):

| removed | replay | fly | code | connectome |
|---|---|---|---|---|
| nothing (Expanse) | 0.418 | 1.198 | 4.006 | 0.571 |
| male-CNS core gate → 0 | 0.447 | 1.233 | 4.088 | 0.596 |
| male-CNS core on the degree/sign-preserving **rewired** graph | 0.437 | 1.229 | 4.075 | 0.592 |
| Omni v7 gate → 0 | 0.429 | 1.217 | 4.025 | 0.583 |
| Qwen/BioMedLM donor experts dead | 0.419 | 1.199 | 4.005 | 0.570 |
| FlyCore off | 0.419 | 1.198 | 4.006 | 0.571 |

What this shows:
* The **male-CNS core is the component the model relies on most**, on every source including code; swapping in a matched random graph loses 65-90% of its contribution, so the trained model depends on the fly's *specific* wiring. (Whether the real wiring *trains* better than a null needs a matched training run on the rewired graph — not yet done.)
* **Omni v7** helps a little everywhere.
* The **grafted donor experts and FlyCore contribute nothing measurable** — the new code/bio knowledge came from distillation, not from the weight grafts.
* Gains are real for connectome facts and biomedical wording; code writing and PubMedQA-style answering did not transfer at this scale. Bio dev loss is omitted above because every bio dev row contains at least one word outside Expanse's vocabulary.

## v2 and v3

Two follow-ups to v1 were trained on the same laptop CPU and compared on **the same held-out rows** (v1's dev split) with `expanse/compare_models.py`, which reports loss **per reply character** so word-level (v1/v2) and BPE (v3) models are comparable. Full tables: [`compare_v1_v2.md`](expanse/checkpoints/compare_v1_v2.md), [`compare_v1_v2_v3.md`](expanse/checkpoints/compare_v1_v2_v3.md).

**v2 — native latent consolidation** (`expanse/src/consolidation_v2.py`, [`V2_DESIGN.md`](expanse/V2_DESIGN.md)). Three zero-gated consolidation blocks (shared 512-d latent, top-2/8 latent MoE, learned memory, +7.3M params) trained for 800 steps against a temporary fusion bank of Archimedes, donor-expert, Omni, FlyCore and male-CNS signals, with a teacher-free final phase. Two bugs were fixed before training (a closed gate could never receive gradient; the distill loss was ~300x the LM loss). Result: v2 is slightly better than v1 (its own held-out dev loss 1.511 -> 1.446; bio token-F1 0.285 -> 0.343), but **zeroing the new blocks removes almost none of the gain** (replay 0.1316 vs 0.1324 nats/char) — the improvement came from 800 more training steps, not from the consolidation blocks.

**v3 — subword tokenizer + more data** (`expanse/src/bpe_tokenizer.py`, `expanse/retokenize_v3.py`, `expanse/make_v3_data.py`, [`V3_DESIGN.md`](expanse/V3_DESIGN.md)). v1 re-tokenised with a byte-level BPE (8,864 tokens, digits kept separate with the leading space on the first digit, **no `<unk>`** — v1 had one in every bio dev row), embeddings re-initialised from the old word embeddings, then 5,000 steps (trunk frozen for the first 300) on 42k rows: ~18k fresh solver/execution-verified problems from the Supermix builders (rows that restate a replay or evaluation problem under new wording were blocked: ~2,000 of them), 5,150 verified Qwen code rows, 2,700 filtered BioMedLM rows and 12k connectome facts. Same seed as v1, so v1's dev rows stayed unseen.

| nats per reply character (lower is better) | v1 | v2 | **v3** |
|---|---|---|---|
| replay (science / code tracing / arithmetic) | 0.166 | 0.132* | **0.033** |
| code | 1.146 | 1.118 | **0.473** |
| male-CNS connectome (held-out cell types) | 0.181 | 0.176 | **0.124** |
| fly | 0.393 | 0.369 | **0.314** |
| bio (all rows) | 0.996 | 0.973 | 1.370 |

| held-out generation (25 items each) | v1 | v2 | **v3** |
|---|---|---|---|
| replay-style problems with fresh numbers | 20% | 28% | **32%** |
| connectome exact match / token-F1 | 16% / 0.674 | 16% / 0.674 | **36% / 0.814** |
| biomedical token-F1 | 0.285 | **0.343** | 0.220 |
| code pass rate, PubMedQA | 0% | 0% | 0% |

\* v2 used a different split seed, so part of its replay gain is on rows it trained on.

v3 is the strongest version on replay, code loss, fly and the connectome, and now writes code in the right shape (*"Find the list of the list. def max_of_value(xs): return max(x)"*) though not yet correctly enough to pass the tests. It **regressed on biomedical answers**: connectome rows (12k) swamp bio rows (2.4k), and part of v1's lower bio loss is `<unk>` making rare terms cheap. Next step: rebalance the mix and continue training.

```bash
# v2 (after the v1 pipeline)
python expanse/train_consolidation_v2.py --steps 800 --batch 4 --threads 8
# v3: fresh data -> more teacher rows -> retokenise v1 -> train -> compare v1/v2/v3
bash expanse/run_v3.sh 480
```

## Set it up yourself

### Requirements

* Python 3.10+ (developed on 3.12), PyTorch 2.4+ (CPU is fine; developed on torch 2.11 CPU).
* **Use fp32.** On CPUs without fast bf16 kernels (e.g. Windows ARM64), bf16 matmul is ~100× slower than fp32.
* Chatting / evaluating: ~3 GB RAM. Full reproduction: 16 GB RAM, ~25 GB free disk.

```bash
git clone https://github.com/kai9987kai/Supermix-expanse.git
cd Supermix-expanse
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option A — chat with the trained model (5 minutes)

```bash
hf download Kai9987kai/supermix-expanse supermix_expanse_v3.pt --local-dir expanse/checkpoints
python expanse/expanse_chat.py --checkpoint expanse/checkpoints/supermix_expanse_v3.pt   # opens http://127.0.0.1:7861
```

`supermix_expanse.pt` (v1, word-level) is also on the Hub; v3 is the recommended checkpoint.

The chat UI streams answers and has live switches for each graft (CNS core, Omni v7, FlyCore, donor experts). To also compare against Archimedes, download its checkpoint first with `python external/fetch_base.py` (or just `hf download Kai9987kai/archimedes-final-model supermix_archimedes.pt --local-dir external/base`).

From Python:

```python
import sys, torch; sys.path.insert(0, "expanse/src")
import expanse_core as ec
model, tok, _ = ec.load_expanse("expanse/checkpoints/supermix_expanse_v3.pt")  # v1 or v3; word or BPE tokenizer
q = "whats the impulse from 40 N acting for 6 s"
ids, _ = tok.encode_turn(q, None)
out = ec.greedy_decode(model, torch.tensor([ids]), max_new_tokens=80,
                       omni_features=ec.arch.OmniCore.featurize([q]),
                       omni7_state=model.omni7_state_for([q]))
print(tok.decode(out))   # impulse = force x time, 40 x 6 = 240, ... total 240
```

Single-turn, 128-token context; v3 uses a byte-level BPE (v1: word-level vocabulary). The checkpoint is a pickled PyTorch payload (`torch.load(weights_only=False)`) — only load files you trust.

### Option B — rebuild and retrain it (≈ 6-7 h on an 8-core laptop CPU)

The repo already ships the teacher corpora (`expanse/data/`), the male-CNS cell-type graph (`expanse/data/malecns_types.npz`) and the Archimedes/Omni source it depends on, so the teacher steps are optional.

1. **Sources** — Archimedes final, Omni v7, teacher configs and the Qwen coder GGUF (~5.2 GB):
   ```bash
   python external/fetch_base.py
   ```
2. **(Optional) regenerate the teacher corpora** instead of using the shipped ones:
   * Teacher weight slices by HTTP range reads (Qwen: embeddings + layers 0-1, ~2 GB; BioMedLM: full model re-encoded to bf16, ~5.3 GB — the fp32 original is never stored):
     ```bash
     python external/fetch_teacher_weights.py qwen
     python external/fetch_teacher_weights.py biomedlm
     ```
     `qwen` slices and BioMedLM weights are also needed for step 3's expert grafts.
   * A llama.cpp release for your platform from <https://github.com/ggml-org/llama.cpp/releases> (built with `b11115`, `win-cpu-arm64`), unzipped into `external/llama.cpp/` so that `external/llama.cpp/llama-server[.exe]` exists.
   * Generate (BioMedLM → Q8_0 GGUF, verified code rows, BioMedLM rows, Qwen fact-check); ~4 h on CPU:
     ```bash
     python expanse/make_teacher_data.py all --target 1500 --heldout 150 --threads 8 --parallel 4
     ```
3. **Clean, build, calibrate, train, evaluate** — in one unattended script (resumable; it skips finished stages and uses the shipped corpora when present):
   ```bash
   bash expanse/run_pipeline.sh 240          # 240 = training wall budget in minutes
   ```
   or step by step:
   ```bash
   python expanse/clean_bio_rows.py
   python expanse/build_expanse.py --threads 8
   python expanse/train_expanse.py --steps 3221 --fly_rows 1500 --dev_cap 100 --eval_every 250 --save_every 50 --threads 8
   python expanse/eval_expanse.py --threads 8
   ```
   Training writes a rolling resumable checkpoint every 50 steps; rerun the same command to continue after an interruption.
4. **Tests:** `python -m pytest expanse/tests -q` (the core/graft tests need the Archimedes checkpoint and teacher slices from steps 1-2).

**Rebuilding the connectome graph from raw data** (optional): download `body-annotations-male-cns-v1.0-minconf-0.5.feather`, `body-neurotransmitters-male-cns-v1.0.feather` and `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1.05 GB) from <https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/> into a folder, then:

```bash
python supermix-archimedes/archimedes/src/malecns_connectome.py build --data_dir <folder> --output expanse/data/malecns_types.npz
```

## Repository layout

```
expanse/                 Expanse code (src/), CLIs, tests, DESIGN.md, run_pipeline.sh, expanse_chat.py
expanse/data/            teacher corpora, connectome graph + fact rows, PubMedQA eval set, data receipts
expanse/checkpoints/     build/training receipts (+ eval report); the .pt lives on Hugging Face
external/                source fetch scripts, vendored Omni v7 model code, run logs
supermix-archimedes/     vendored Archimedes source + replay corpus (kai9987kai/supermix-archimedes, MIT)
```

## License

* Code: MIT (see [LICENSE](LICENSE)).
* Model weights and derived data: composite terms — see [LICENSE-MODEL.md](LICENSE-MODEL.md). The weights contain parameters derived from BioMedLM and were trained on its outputs, so the **BigScience OpenRAIL-M use restrictions apply, including: not for medical advice or medical results interpretation**. Qwen2.5-Coder-7B-Instruct is Apache-2.0. The male CNS connectome data is CC BY 4.0 (Janelia FlyEM Project Team and collaborators). PubMedQA is MIT.

## Citations

* Male CNS connectome v1.0 — Janelia FlyEM Project Team and the Cambridge Drosophila Connectomics Group, <https://male-cns.janelia.org/>; natverse `malecns`.
* Bolton et al., *BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text* (2024).
* Hui et al., *Qwen2.5-Coder Technical Report* (2024).
* Jin et al., *PubMedQA* (EMNLP 2019).
* Lappalainen et al., *Connectome-constrained networks predict neural activity across the fly visual system* (Nature 2024).
