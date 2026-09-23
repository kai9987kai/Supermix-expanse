#!/usr/bin/env bash
# Unattended Supermix Expanse pipeline: teacher data -> Stage 1 build -> step-time
# calibration -> Stage 2 training sized to a wall budget -> evaluation.
# Every stage is resumable; rerunning this script skips finished stages.
#   bash expanse/run_pipeline.sh [TRAIN_MINUTES]
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
TRAIN_MINUTES="${1:-240}"
LOGS=../external/logs
PLOG=$LOGS/pipeline.log
say() { echo "$(date +%H:%M:%S) [pipeline] $*" | tee -a "$PLOG"; }

# 1. teacher data (started separately; wait for it to finish)
mkdir -p $LOGS
if [ -s data/code_rows.jsonl ] && [ -s data/bio_rows.jsonl ] && ! grep -q "EXIT=" $LOGS/teacher_all.log 2>/dev/null; then
  echo "EXIT=0  (corpora already present; teacher run skipped)" >> $LOGS/teacher_all.log
fi
if ! grep -q "EXIT=" $LOGS/teacher_all.log 2>/dev/null; then
  say "waiting for teacher data"
  until grep -q "EXIT=" $LOGS/teacher_all.log 2>/dev/null; do sleep 60; done
fi
code=$(grep -o "EXIT=[0-9]*" $LOGS/teacher_all.log | tail -1 | cut -d= -f2)
say "teacher data finished with exit $code; rows: code $(wc -l < data/code_rows.jsonl) bio $(wc -l < data/bio_rows.jsonl)"
[ "$code" = "0" ] || { say "teacher data failed; stopping"; exit 1; }

# 1b. clean the bio corpus (yes-bias cap, fragment terms) + unbiased PubMedQA eval set
python -u clean_bio_rows.py > $LOGS/clean_bio.log 2>&1 || { say "clean_bio_rows failed (see clean_bio.log)"; exit 1; }
say "bio cleaned: $(PYTHONIOENCODING=utf-8 python -c "import json;r=json.load(open('data/bio_rows.clean.report.json'));print('kept',r['kept_rows'],'of',r['input_rows'],'dropped',r['dropped'],'pubmedqa eval',r['pubmedqa_eval'])")"

# 2. Stage 1 build
if [ ! -f checkpoints/supermix_expanse_grafted.pt ]; then
  say "stage 1 build"
  python -u build_expanse.py --threads 8 > $LOGS/build.log 2>&1 || { say "build failed (see build.log)"; exit 1; }
  say "build done: $(grep -E 'saved|function preservation' $LOGS/build.log | tr '\n' ' ')"
fi

COMMON="--fly_rows 1500 --dev_cap 100 --threads 8"

# 3. calibrate seconds/step (also fills the kd_arch and omni7 caches the real run reuses)
if [ ! -f checkpoints/calib/steps.txt ]; then
  say "calibrating step time"
  python -u train_expanse.py $COMMON --steps 1000 --stop_after 20 --eval_every 100000 --save_every 0 \
    --out checkpoints/calib/supermix_expanse_calib.pt > $LOGS/calib.log 2>&1 || { say "calibration failed (see calib.log)"; exit 1; }
  sps=$(grep -o "[0-9.]*s/step" $LOGS/calib.log | tail -1 | tr -d 's/step')
  train_rows=$(grep -o "packed train ([0-9]*" $LOGS/calib.log | grep -o "[0-9]*$")
  steps=$(python -c "
sps=float('$sps'); rows=int('$train_rows'); budget=$TRAIN_MINUTES*60
epoch=rows//8
s=int(budget/sps)
s=max(min(s, 2*epoch), min(800, 2*epoch))
print(s)")
  mkdir -p checkpoints/calib
  echo "$steps" > checkpoints/calib/steps.txt
  say "calibration: ${sps}s/step, train rows $train_rows (epoch = $((train_rows/8)) steps) -> $steps steps"
fi
STEPS=$(cat checkpoints/calib/steps.txt)

# 4. Stage 2 training (resumes from its rolling partial if interrupted)
if [ ! -f checkpoints/supermix_expanse.pt ]; then
  say "stage 2 training: $STEPS steps"
  python -u train_expanse.py $COMMON --steps "$STEPS" --eval_every 250 --save_every 50 \
    --max_minutes $((TRAIN_MINUTES + 45)) --finalize_on_budget > $LOGS/train.log 2>&1 \
    || { say "training failed (see train.log)"; exit 1; }
  say "training done: $(grep -E 'eval final' $LOGS/train.log | tail -1)"
fi

# 5. evaluation
if [ ! -f checkpoints/eval_report.md ]; then
  say "evaluation"
  python -u eval_expanse.py --threads 8 > $LOGS/eval.log 2>&1 || { say "eval failed (see eval.log)"; exit 1; }
fi
say "PIPELINE DONE"
