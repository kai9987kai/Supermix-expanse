#!/usr/bin/env bash
# Unattended Supermix Expanse v3 pipeline (resumable: finished stages are skipped).
#   bash expanse/run_v3.sh [TRAIN_MINUTES]
# fresh verified data -> more teacher rows -> clean bio -> wait for disk -> retokenise v1
# -> train v3 (seed 2026 = v1's dev split, so the v1 comparison stays clean) -> compare v1/v2/v3.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
TRAIN_MINUTES="${1:-480}"
LOGS=../external/logs
PLOG=$LOGS/v3_pipeline.log
mkdir -p "$LOGS" data/v3 checkpoints
say() { echo "$(date +%H:%M:%S) [v3] $*" | tee -a "$PLOG"; }
fail() { say "FAILED: $*"; exit 1; }
free_gb() { powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1MB)" | tr -d '\r'; }

say "start (train budget ${TRAIN_MINUTES} min)"

# 1. fresh verified rows (Supermix builders) + connectome rows v3 (same held-out cell types as v1)
if [ ! -f data/v3/fresh.report.json ] || [ ! -f data/v3/connectome_rows_v3.jsonl ]; then
  say "fresh data"
  python -u make_v3_data.py --omni 10000 --code 8000 --math 6000 > $LOGS/v3_data.log 2>&1 || fail "make_v3_data (v3_data.log)"
fi
say "fresh data ready: $(wc -l data/v3/*.jsonl | tail -1)"

# 2-3. more teacher rows (append-only, resumable), then fact-check and clean
say "teacher code rows -> 5000 train"
python -u make_teacher_data.py code --target 5000 --heldout 150 --threads 8 --parallel 4 > $LOGS/v3_teacher_code.log 2>&1 || fail "code rows (v3_teacher_code.log)"
say "teacher bio rows -> 3000 train"
python -u make_teacher_data.py bio --target 3000 --heldout 150 --threads 8 --parallel 4 > $LOGS/v3_teacher_bio.log 2>&1 || fail "bio rows (v3_teacher_bio.log)"
python -u make_teacher_data.py judge --threads 8 --parallel 4 > $LOGS/v3_teacher_judge.log 2>&1 || fail "judge (v3_teacher_judge.log)"
python -u clean_bio_rows.py > $LOGS/v3_clean_bio.log 2>&1 || fail "clean_bio_rows"
say "teacher data: code $(wc -l < data/code_rows.jsonl) rows, bio clean $(wc -l < data/bio_rows.clean.jsonl) rows"

# 4. the retokenised init, rolling partials and the final checkpoint need ~2 GB at peak
while [ "$(free_gb)" -lt 3500 ]; do
  say "waiting for disk: $(free_gb) MB free, need 3500 MB (see the cleanup command in the chat)"
  sleep 600
done

# 5. BPE tokenizer + re-initialised embeddings on the v1 checkpoint
if [ ! -f checkpoints/supermix_expanse_v3_init.pt ]; then
  say "retokenise v1"
  python -u retokenize_v3.py --inp checkpoints/supermix_expanse.pt --out checkpoints/supermix_expanse_v3_init.pt \
    --vocab 16000 --max_rows_per_source 12000 > $LOGS/v3_retokenize.log 2>&1 || fail "retokenize_v3 (v3_retokenize.log)"
fi
say "v3 init: $(grep -a -E 'rules|context|vocab' $LOGS/v3_retokenize.log | tail -3 | tr '\n' ' ')"

# 6. train v3 (embeddings aligned first with the trunk frozen); budget-capped, resumable
if [ ! -f checkpoints/supermix_expanse_v3.pt ]; then
  say "train v3"
  python -u train_expanse.py --inp checkpoints/supermix_expanse_v3_init.pt --out checkpoints/supermix_expanse_v3.pt \
    --steps 5000 --seed 2026 --fly_rows 1500 --dev_cap 100 --max_rows_per_source 12000 --freeze_trunk_steps 300 \
    --lr_trunk 5e-5 --eval_every 500 --save_every 100 --max_minutes $((TRAIN_MINUTES + 60)) --finalize_on_budget \
    --threads 8 > $LOGS/v3_train.log 2>&1 || fail "train v3 (v3_train.log)"
fi
say "trained: $(grep -a 'eval final' $LOGS/v3_train.log | tail -1 | cut -c1-240)"

# 7. v1 vs v2 vs v3 on the same rows (per-character loss) + held-out generation
if [ ! -f checkpoints/compare_v1_v2_v3.md ]; then
  say "compare v1 / v2 / v3"
  python -u compare_models.py --model v1=checkpoints/supermix_expanse.pt --model v2=checkpoints/supermix_expanse_v2.pt \
    --model v3=checkpoints/supermix_expanse_v3.pt --gen_limit 25 --max_new_tokens 64 \
    --out checkpoints/compare_v1_v2_v3.json > $LOGS/v3_compare.log 2>&1 || fail "compare (v3_compare.log)"
fi
say "V3 PIPELINE DONE"
