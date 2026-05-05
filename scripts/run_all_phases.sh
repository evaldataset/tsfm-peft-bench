#!/bin/bash
# Phase 1+2 자동 실행 스크립트: domain → rank → locus → mechanism checkpoints
# Usage: bash scripts/run_all_phases.sh <model> <gpu_id>
# Example: bash scripts/run_all_phases.sh chronos 0

set -e
MODEL=$1
GPU=$2
DOMAINS="ett_m1,smd,finance,physionet"
SEEDS="42,123,7,3407,2024"

export PYTHONPATH=.

echo "=== [$MODEL] GPU $GPU: Domain mode ==="
.venv/bin/python scripts/run_expansion.py --mode domain --models "$MODEL" \
  --domains "$DOMAINS" --seeds "$SEEDS" --gpu "$GPU" --skip_unavailable_models

echo "=== [$MODEL] GPU $GPU: Rank mode ==="
.venv/bin/python scripts/run_expansion.py --mode rank --models "$MODEL" \
  --domains "$DOMAINS" --seeds "$SEEDS" --gpu "$GPU" --skip_unavailable_models

echo "=== [$MODEL] GPU $GPU: Locus mode ==="
.venv/bin/python scripts/run_expansion.py --mode locus --models "$MODEL" \
  --domains "$DOMAINS" --seeds "$SEEDS" --gpu "$GPU" --skip_unavailable_models

echo "=== [$MODEL] GPU $GPU: Mechanism checkpoints ==="
# Success/failure 셀 재실행 with checkpoint saving
.venv/bin/python scripts/run_expansion.py --mode domain --models "$MODEL" \
  --domains "$DOMAINS" --seeds 42,123 --gpu "$GPU" \
  --force_rerun --save_checkpoints --checkpoint_dir checkpoints/mechanism \
  --methods head_only,lora,adapter --skip_unavailable_models

echo "=== [$MODEL] 모든 phase 완료! ==="
