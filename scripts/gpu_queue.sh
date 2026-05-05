#!/usr/bin/env bash
# GPU 메모리를 모니터링하여 유휴 GPU에 대기 중인 실험을 자동 실행하는 스크립트
# 사용법: nohup bash scripts/gpu_queue.sh &
set -euo pipefail

# ─── 경로 설정 ───────────────────────────────────────────────────────────────
PROJECT_ROOT="$(pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/gpu_queue.log"
FLAG_DIR="/tmp/gpu_queue_flags"

mkdir -p "${LOG_DIR}"
mkdir -p "${FLAG_DIR}"

# ─── 대기 작업 정의 ──────────────────────────────────────────────────────────

# GPU 2가 해제되면: gradient probe 실행
GPU2_CMD="CUDA_VISIBLE_DEVICES=2 python scripts/gradient_probe.py \
  --cells 'chronos:lora:ett_m1,chronos:adapter:finance,moirai:lora:ett_m1,moirai:adapter:smd,moment:lora:ett_m1,moment:adapter:finance' \
  --seeds 42,123 --epochs 3 --gpu 0 \
  --output_dir results/gradient_analysis"

# GPU 1이 해제되면: DoRA Chronos+MOMENT 실행
GPU1_CMD="CUDA_VISIBLE_DEVICES=1 python scripts/run_expansion.py \
  --mode domain --models chronos,moment \
  --methods dora \
  --domains ett_m1,finance,smd,physionet \
  --seeds 42,123,7,2024,3407 --gpu 0"

# GPU 3이 해제되면: DoRA Moirai 실행
GPU3_CMD="CUDA_VISIBLE_DEVICES=3 python scripts/run_expansion.py \
  --mode domain --models moirai \
  --methods dora \
  --domains ett_m1,finance,smd,physionet \
  --seeds 42,123,7,2024,3407 --gpu 0"

# ─── 유틸리티 함수 ───────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# GPU의 사용 메모리(MB)를 반환
gpu_memory_used() {
    local gpu_id="$1"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | sed -n "$((gpu_id + 1))p" \
        | tr -d ' '
}

# 작업 시작: nohup으로 백그라운드 실행하고 플래그 파일 생성
launch_job() {
    local gpu_id="$1"
    local cmd="$2"
    local flag_file="${FLAG_DIR}/gpu${gpu_id}.launched"

    log "GPU ${gpu_id} 해제 감지 — 작업 시작"
    log "명령: ${cmd}"

    touch "${flag_file}"
    # 백그라운드 실행; stdout/stderr를 로그에 추가
    nohup bash -c "cd '${PROJECT_ROOT}' && ${cmd}" \
        >> "${LOG_DIR}/gpu${gpu_id}_job.log" 2>&1 &

    log "GPU ${gpu_id} 작업 PID $! 로 실행됨"
}

# ─── 메인 루프 ───────────────────────────────────────────────────────────────

log "GPU 큐 모니터 시작 (폴링 간격: 60초)"
log "대기 작업: GPU 1, 2, 3"

MEMORY_THRESHOLD=2048  # 2GB 미만이면 유휴 상태로 판단

while true; do
    # 모든 작업이 완료되었는지 확인
    launched=0
    [[ -f "${FLAG_DIR}/gpu1.launched" ]] && ((launched++)) || true
    [[ -f "${FLAG_DIR}/gpu2.launched" ]] && ((launched++)) || true
    [[ -f "${FLAG_DIR}/gpu3.launched" ]] && ((launched++)) || true

    if [[ "${launched}" -eq 3 ]]; then
        log "모든 작업(3개)이 실행됨 — 모니터 종료"
        exit 0
    fi

    # 각 GPU 상태 확인 및 작업 시작
    for gpu_id in 1 2 3; do
        flag_file="${FLAG_DIR}/gpu${gpu_id}.launched"
        [[ -f "${flag_file}" ]] && continue  # 이미 실행됨

        mem_used=$(gpu_memory_used "${gpu_id}" 2>/dev/null || echo "9999")

        if [[ "${mem_used}" -lt "${MEMORY_THRESHOLD}" ]]; then
            case "${gpu_id}" in
                1) launch_job 1 "${GPU1_CMD}" ;;
                2) launch_job 2 "${GPU2_CMD}" ;;
                3) launch_job 3 "${GPU3_CMD}" ;;
            esac
        else
            log "GPU ${gpu_id} 사용 중 (${mem_used}MB 사용) — 대기"
        fi
    done

    sleep 60
done
