# Supermix Expanse — composite license

Supermix Expanse combines components under different terms. Each applies to the part it covers; where they overlap, the most restrictive use terms apply to the model as a whole.

| Component | Terms |
|---|---|
| Code in `expanse/`, `supermix-archimedes/`, `external/` | MIT License, Copyright (c) 2026 Kai Piper (see `LICENSE`) |
| Archimedes / Supermix / Omni Collective weights inside the checkpoint | Kai Piper; released with this model |
| Parameters derived from **BioMedLM** (grafted MoE experts in layers 1-2) and training on BioMedLM outputs | **BigScience OpenRAIL-M** (bigscience-bloom-rail-1.0), <https://huggingface.co/spaces/bigscience/license> |
| Parameters derived from Qwen2.5-Coder-7B-Instruct and training on its outputs | Apache License 2.0, <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE> |
| Male CNS v1.0 connectome data (wiring of the CNS core, connectome fact rows) | CC BY 4.0 — Janelia FlyEM Project Team and collaborators, <https://male-cns.janelia.org/> |
| PubMedQA-derived evaluation items (`data/pubmedqa_eval.jsonl`) | MIT (PubMedQA, Jin et al. 2019) |

## Use restrictions (inherited from BigScience OpenRAIL-M)

Because the model weights include parameters derived from BioMedLM, the use-based restrictions of the BigScience OpenRAIL-M license (its Attachment A) apply to this model and to any derivative of it, and must be passed on to anyone you redistribute it to. Among other things, you may **not** use Supermix Expanse:

* to provide medical advice or medical results interpretation;
* in any way that violates applicable laws or regulations;
* for any other use prohibited by Attachment A of <https://huggingface.co/spaces/bigscience/license>.

This is an experimental research model. It is provided "as is", without warranty of any kind, and its outputs (including biomedical and connectome statements) may be wrong.
