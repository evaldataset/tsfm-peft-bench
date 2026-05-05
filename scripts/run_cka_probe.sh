#!/usr/bin/env bash
# CKA 궤적 프로브 실행 스크립트.
# 발산 케이스(Chronos+LoRA+ETTm1)와 성공 케이스를 비교 실행.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== CKA 궤적 프로브 시작 ==="
echo "프로젝트 루트: ${PROJECT_ROOT}"

# ─── 발산 케이스: Chronos+LoRA+ETTm1 (전 시드) ──────────────────
echo "[1/6] 발산 케이스: chronos+lora+ett_m1 seed=42"
python scripts/run_cka_trajectory.py \
    --model chronos \
    --method lora \
    --domain ett_m1 \
    --seed 42 \
    --gpu 0 \
    --epochs 10 \
    --cka_every_n 2

echo "[2/6] 성공 케이스: chronos+adapter+ett_m1 seed=42"
python scripts/run_cka_trajectory.py \
    --model chronos \
    --method adapter \
    --domain ett_m1 \
    --seed 42 \
    --gpu 0 \
    --epochs 10 \
    --cka_every_n 2

echo "[3/6] 성공 케이스: chronos+head_only+finance seed=42"
python scripts/run_cka_trajectory.py \
    --model chronos \
    --method head_only \
    --domain finance \
    --seed 42 \
    --gpu 0 \
    --epochs 10 \
    --cka_every_n 2

echo "[4/6] 성공 케이스: moment+lora+finance seed=42"
python scripts/run_cka_trajectory.py \
    --model moment \
    --method lora \
    --domain finance \
    --seed 42 \
    --gpu 0 \
    --epochs 10 \
    --cka_every_n 2

echo "[5/6] Moirai 케이스: moirai+lora+smd seed=42"
python scripts/run_cka_trajectory.py \
    --model moirai \
    --method lora \
    --domain smd \
    --seed 42 \
    --gpu 0 \
    --epochs 10 \
    --cka_every_n 2

echo "[6/6] Moirai 케이스: moirai+full_fine_tuning+ett_m1 seed=42"
python scripts/run_cka_trajectory.py \
    --model moirai \
    --method full_fine_tuning \
    --domain ett_m1 \
    --seed 42 \
    --gpu 0 \
    --epochs 10 \
    --cka_every_n 2

echo ""
echo "=== 모든 CKA 궤적 프로브 완료 ==="
echo "결과 위치: results/cka_trajectories/"
echo ""
echo "분석 실행:"
echo "  python scripts/analyze_cka_trajectories.py"
