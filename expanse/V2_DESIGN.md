# Supermix Expanse v2 — latent consolidation design

Expanse v2 is an experimental continuation of the trained v1 model. It does **not** raw-average Qwen, BioMedLM, Omni and Archimedes parameters. Those systems use incompatible hidden spaces and architectures. Instead, v2 learns a temporary common representation during training and bakes that representation into new native Expanse blocks.

## Goal

The v1 model is already one checkpoint, but several source systems still enter through identifiable grafts or branches. v2 asks a stronger question:

> Can the useful signals from those heterogeneous systems be consolidated into new native computation that remains useful after the training-only fusion machinery is removed?

The intended final runtime is therefore:

```text
token embeddings
      |
Archimedes / Expanse layer 0
      |
layer 1 -> native consolidation block
      |       320 -> 512 shared latent
      |       top-2 / 8 latent MoE
      |       learned latent memory
      |       512 -> 320 gated residual
      |
layer 2 -> native consolidation block
      |
existing Fly / Omni / male-CNS graft site
      |
layers 3-4 -> native consolidation block after layer 4
      |
remaining trunk / thinking core
      |
language head
```

The default native v2 stack adds about **7.33M runtime parameters** (three blocks at about 2.44M each) to the ~132.6M v1 checkpoint. The teacher-fusion projectors are training-only and are not saved into the v2 runtime checkpoint.

## Heterogeneous teacher signals

At each selected student depth, `InternalSourceExtractor` builds prompt-level views from systems already present in v1:

- **arch** — the current Expanse/Archimedes residual representation.
- **qwen** — response of the Qwen-derived donor experts at that depth when present.
- **biomedlm** — response of the BioMedLM-derived donor experts when present.
- **omni7** — the frozen Omni Collective v7 988-dimensional prompt state.
- **omni** — the existing Omni prompt features.
- **fly** — the one-position FlyCore residual signal.
- **cns** — the one-position full male-CNS residual signal.

Qwen/BioMed donor probes are source-specific signals, not claims that a 320-d donor expert output is an exact reconstruction of the original multi-billion-parameter teacher hidden state.

## Shared latent and source routing

Each source receives a learned low-rank projector into a 512-dimensional normalized latent space. A trainable router computes a sample-specific weighted consensus instead of blindly averaging sources.

Weak domain priors only break symmetry:

- code -> Qwen
- biomedical -> BioMedLM
- fly -> FlyCore
- connectome -> male-CNS
- replay/general -> Archimedes + Omni

The priors remain trainable. Source-balance regularization discourages collapse onto a single source.

Projector alignment uses multi-view InfoNCE where batch size permits, a cosine fallback for batch size 1, and a variance floor to reduce trivial representation collapse.

## Native latent blocks

`NativeLatentBlock` is the part that survives deployment. Each block contains:

1. RMS-normalized 320-d trunk input.
2. 320 -> 512 projection.
3. Eight 512-d SwiGLU latent experts with top-2 routing.
4. Twenty-four learned 512-d memory slots.
5. 512 -> 320 projection.
6. A per-channel output gate.

The output gate is initialized to exactly zero. The implementation branches around the residual projection while the entire gate is zero, so attaching v2 is an exact identity operation rather than merely numerically close.

The blocks are per-token. They do not mix sequence positions, so they preserve the causal/KV-cache contract of the base model.

## Training objective

The trainer combines the existing Expanse language objective with representation consolidation:

```text
L =
    L_language
  + lambda_distill * L_student_to_fused_latent
  + lambda_align   * L_source_alignment
  + lambda_native  * L_native_router_balance
  + lambda_source  * L_source_router_balance
```

The student-to-target term combines cosine distance, Smooth L1 and relational batch geometry.

Default scalar weights are intentionally conservative and should be tuned from evidence rather than treated as canonical:

```text
distill               0.35
source alignment      0.10
native router balance 0.01
source router balance 0.003
```

## Four training phases

The default schedule is:

1. **Alignment / warm-up** — train source projectors and student latent geometry while native output gates start at zero.
2. **Joint consolidation** — language loss, source alignment and student distillation train together.
3. **Bake** at 72% — freeze the teacher-fusion bank. Its target can no longer co-adapt indefinitely; its distillation weight decays toward the teacher-free boundary.
4. **Teacher-free finish** at 90% — source extraction, fusion and representation losses are completely disabled. The last 10% optimizes only the self-contained runtime model and its native router regularizer.

The final `supermix_expanse_v2.pt` does not contain the training-only `TeacherFusionBank`.

## Run

Start from the trained v1 checkpoint:

```bash
python expanse/train_consolidation_v2.py --steps 800 --batch 4 --threads 8
```

Structural/training smoke run:

```bash
python expanse/train_consolidation_v2.py --smoke --threads 2
```

For a faster experiment that omits FlyCore/full-CNS one-vector source probes:

```bash
python expanse/train_consolidation_v2.py --skip_slow_sources
```

Load a completed v2 checkpoint:

```python
import sys
sys.path.insert(0, "expanse/src")
import consolidation_v2 as cv2

model, tokenizer, payload = cv2.load_v2(
    "expanse/checkpoints/supermix_expanse_v2.pt"
)
```

## What must be measured before calling v2 better

The code path being present is not evidence of capability improvement. A v2 checkpoint should only be described as improved after comparing it against the exact v1 checkpoint on held-out data.

At minimum record:

- per-source dev loss on replay, fly, code, biomedical and connectome rows;
- the existing held-out generation/code/BioMed/connectome evaluation from v1;
- native-block ablation with the v2 output gates zeroed;
- source ablations during training or a matched rerun without each source;
- routing entropy and source usage, looking for collapse;
- catastrophic-forgetting checks on the original Archimedes replay tasks;
- wall time, checkpoint size and generation latency.

A useful consolidation result is not merely lower training loss. The stronger evidence is: v2 beats or matches v1 on held-out tasks, the native-block ablation removes part of that gain, and the final checkpoint retains it after the training-only fusion bank has been removed.

## Current validation status

The branch includes synthetic unit tests for:

- bit-exact identity at zero output gates;
- top-k latent routing normalization;
- hook integration and checkpoint registration;
- heterogeneous fusion shapes and gradients;
- config round-trip;
- donor-source extraction;
- a strictly teacher-free final schedule.

These tests validate mechanics, not model quality. A full v2 training run and held-out benchmark are still required before making performance claims.
