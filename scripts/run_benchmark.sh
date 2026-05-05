#!/usr/bin/env bash
# ============================================================================
# DEPRECATED — 이 스크립트는 구형 벤치마크 (ETT만, 3 seeds)를 실행합니다.
# 논문의 모든 결과는 ``scripts/run_expansion.py``로 생성됩니다.
#
# 새 entry point:
#   python scripts/run_expansion.py --models chronos,moment,moirai \
#       --mode domain --seeds 42,123,7,2024,3407
# ============================================================================
echo "[DEPRECATED] run_benchmark.sh: 논문 실험은 run_expansion.py를 사용하세요." >&2
echo "[DEPRECATED] 5초 후 계속됩니다. Ctrl+C로 중단 가능." >&2
sleep 5

set -euo pipefail

# ─── 기본 설정 ──────────────────────────────────────────────
MODELS=("chronos" "moment")
ADAPTATIONS=("zero_shot" "head" "lora" "prefix" "full_ft")
DATASETS=("ett_m1" "ett_h1")
SEEDS=(42 123 456)

DRY_RUN=false
LOG_DIR="results/benchmark_logs"

# ─── CLI 인자 파싱 ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --seeds)
            IFS=' ' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --models)
            IFS=' ' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --adaptations)
            IFS=' ' read -ra ADAPTATIONS <<< "$2"
            shift 2
            ;;
        --datasets)
            IFS=' ' read -ra DATASETS <<< "$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--seeds '42 123'] [--models 'chronos moment'] [--adaptations 'lora head'] [--datasets 'ett_m1'] [--log-dir DIR]"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ─── 로그 디렉토리 생성 ─────────────────────────────────────
mkdir -p "$LOG_DIR"

# ─── 실험 조합 계산 ─────────────────────────────────────────
TOTAL=0
for _m in "${MODELS[@]}"; do
    for _a in "${ADAPTATIONS[@]}"; do
        for _d in "${DATASETS[@]}"; do
            for _s in "${SEEDS[@]}"; do
                TOTAL=$((TOTAL + 1))
            done
        done
    done
done

echo "============================================"
echo "  TSFM-PEFT Benchmark Runner"
echo "============================================"
echo "  Models:      ${MODELS[*]}"
echo "  Adaptations: ${ADAPTATIONS[*]}"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Seeds:       ${SEEDS[*]}"
echo "  Total runs:  ${TOTAL}"
echo "  Dry run:     ${DRY_RUN}"
echo "  Log dir:     ${LOG_DIR}"
echo "============================================"

# ─── 실행 루프 ──────────────────────────────────────────────
RUN_COUNT=0
FAILED=0
SUCCEEDED=0
SKIPPED=0

for model in "${MODELS[@]}"; do
    for adaptation in "${ADAPTATIONS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                RUN_COUNT=$((RUN_COUNT + 1))
                RUN_ID="${model}_${adaptation}_${dataset}_seed${seed}"
                LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

                echo "[${RUN_COUNT}/${TOTAL}] ${RUN_ID}"

                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "  → (dry run) python scripts/train.py model=${model} adaptation=${adaptation} data=${dataset} seed=${seed}"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                # 이미 결과가 있으면 건너뛰기
                RESULT_DIR="results/${model}_${adaptation}_${dataset}"
                if [[ -f "${RESULT_DIR}/best.pt" ]]; then
                    echo "  → 이미 완료됨, 건너뜁니다: ${RESULT_DIR}/best.pt"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                # 학습 실행
                if python scripts/train.py \
                    model="${model}" \
                    adaptation="${adaptation}" \
                    data="${dataset}" \
                    seed="${seed}" \
                    > "${LOG_FILE}" 2>&1; then
                    echo "  → 성공 (로그: ${LOG_FILE})"
                    SUCCEEDED=$((SUCCEEDED + 1))
                else
                    echo "  → 실패 (로그: ${LOG_FILE})"
                    FAILED=$((FAILED + 1))
                fi
            done
        done
    done
done

# ─── 요약 출력 ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "  벤치마크 완료"
echo "============================================"
echo "  전체:   ${TOTAL}"
echo "  성공:   ${SUCCEEDED}"
echo "  실패:   ${FAILED}"
echo "  건너뜀: ${SKIPPED}"
echo "============================================"

if [[ ${FAILED} -gt 0 ]]; then
    echo "WARNING: ${FAILED} 건의 실험이 실패했습니다. 로그를 확인하세요: ${LOG_DIR}/"
    exit 1
fi
