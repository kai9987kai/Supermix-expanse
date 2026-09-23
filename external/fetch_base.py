"""Fetch the student-side sources and the coder teacher GGUF (resumable)."""
import sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT = str(Path(__file__).resolve().parent)
JOBS = [
    ("Kai9987kai/archimedes-final-model", ["supermix_archimedes.pt", "supermix_archimedes_grafted.receipt.json", "README.md"], ROOT + r"\base"),
    ("Kai9987kai/supermix-omni-collective-v7-frontier", [
        "omni_collective_v7_model.py", "omni_collective_v5_model.py", "omni_collective_v4_model.py", "omni_collective_model.py",
        "image_feature_utils.py", "image_recognition_model.py", "math_equation_model.py", "protein_folding_model.py",
        "train_omni_collective_v7.py", "omni_collective_v7_frontier_summary.json", "omni_collective_v7_frontier_meta.json",
        "omni_collective_v7_frontier.pth", "README.md"], ROOT + r"\omni_v7"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "model.safetensors.index.json", "LICENSE"], ROOT + r"\teachers\qwen2.5-coder-7b-instruct"),
    ("stanford-crfm/BioMedLM", ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "README.md"],
        ROOT + r"\teachers\biomedlm"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", ["qwen2.5-coder-7b-instruct-q4_k_m.gguf", "LICENSE"], ROOT + r"\teachers\gguf"),
]
for repo, files, dest in JOBS:
    for f in files:
        for attempt in range(5):
            try:
                t = time.time(); p = hf_hub_download(repo, f, local_dir=dest)
                print(f"OK {repo}/{f} -> {p} ({time.time()-t:.0f}s)", flush=True); break
            except Exception as e:
                print(f"RETRY {attempt} {repo}/{f}: {e}", flush=True); time.sleep(10 * (attempt + 1))
        else:
            print(f"FAILED {repo}/{f}", flush=True); sys.exit(1)
print("ALL DONE", flush=True)
