from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(gpu_id: int = 0) -> torch.device:
    """사용 가능한 디바이스 반환.

    Args:
        gpu_id: GPU 인덱스.

    Returns:
        torch.device 객체.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        gpu_name = torch.cuda.get_device_name(gpu_id)
        _, total_mem = torch.cuda.mem_get_info(gpu_id)
        gpu_mem = total_mem / 1e9
        logger.info(f"GPU 사용: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        logger.info("CPU 사용")

    return device


def log_gpu_memory(prefix: str = "") -> dict[str, float]:
    """현재 GPU 메모리 사용량 로깅.

    Args:
        prefix: 로그 메시지 접두사.

    Returns:
        메모리 정보 딕셔너리 (GB 단위).
    """
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    info = {
        "gpu_allocated_gb": round(allocated, 3),
        "gpu_reserved_gb": round(reserved, 3),
        "gpu_max_allocated_gb": round(max_allocated, 3),
    }

    logger.info(
        f"{prefix}GPU 메모리: allocated={allocated:.3f}GB, reserved={reserved:.3f}GB, max={max_allocated:.3f}GB"
    )

    return info
