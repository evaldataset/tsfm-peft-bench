from __future__ import annotations

"""FIM(Fisher Information Matrix) 대각선 기반 도메인 지문 계산 모듈.

손으로 만든 5차원 이동 프로파일 대신, 모델별 민감도 지형에서 원리적으로 도출한
도메인 임베딩을 제공한다.
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def compute_fim_diagonal(
    model: nn.Module,
    dataset: Dataset[Any],
    n_samples: int = 100,
    device: str = "cuda",
) -> NDArray[np.floating]:
    """Fisher Information Matrix 대각선 추정.

    각 샘플에 대해 손실의 파라미터 그래디언트를 계산하고,
    요소별로 제곱하여 평균함으로써 FIM 대각선을 추정한다.

    FIM_ii = E[ (d log p / d theta_i)^2 ]

    전체 파라미터(requires_grad 여부 무관)에 대해 계산하여
    적응 결정 이전의 사전학습 모델의 민감도 지형 전체를 포착한다.

    Args:
        model: 평가 대상 PyTorch 모델 (nn.Module).
        dataset: ``{"context": Tensor, "target": Tensor}`` 를 반환하는 데이터셋.
        n_samples: FIM 추정에 사용할 샘플 수.
        device: 연산에 사용할 디바이스 문자열.

    Returns:
        파라미터당 FIM 대각선 값으로 구성된 1D NumPy 배열.
        순서는 ``model.parameters()`` 순서와 동일.

    Raises:
        ValueError: 데이터셋이 비었거나 n_samples가 1 미만일 때.
        RuntimeError: forward 패스에서 loss를 얻지 못했을 때.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples는 1 이상이어야 합니다. 현재: {n_samples}")
    if len(dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("데이터셋이 비어 있습니다.")

    actual_device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 파라미터 총 개수 파악 (requires_grad 무관)
    all_params = list(model.parameters())
    n_params = sum(p.numel() for p in all_params)
    fim_accum = np.zeros(n_params, dtype=np.float64)

    # 원래 requires_grad 상태 저장 후 전체 활성화
    orig_requires_grad = [p.requires_grad for p in all_params]
    for p in all_params:
        p.requires_grad_(True)

    model.train()
    model.to(actual_device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_done = 0

    for batch in loader:
        if n_done >= n_samples:
            break

        context: torch.Tensor = batch["context"].to(actual_device)
        target: torch.Tensor = batch["target"].to(actual_device)

        # 그래디언트 초기화
        model.zero_grad()

        try:
            outputs: dict[str, torch.Tensor] = model(  # type: ignore[call-arg]
                context=context, target=target
            )
            loss: torch.Tensor = outputs["loss"]
        except Exception as exc:
            logger.warning("샘플 %d에서 forward 패스 실패, 건너뜀: %s", n_done, exc)
            continue

        if not loss.requires_grad:
            # loss가 미분 불가이면 MSE로 대체
            try:
                pred = outputs.get("pred")
                if pred is not None:
                    loss = torch.nn.functional.mse_loss(
                        pred.to(dtype=torch.float32),
                        target.to(dtype=torch.float32),
                    )
                else:
                    logger.warning("샘플 %d: loss와 pred 모두 미분 불가. 건너뜀.", n_done)
                    continue
            except Exception as exc:
                logger.warning("샘플 %d: 대체 loss 계산 실패: %s", n_done, exc)
                continue

        try:
            loss.backward()
        except Exception as exc:
            logger.warning("샘플 %d에서 backward 실패, 건너뜀: %s", n_done, exc)
            model.zero_grad()
            continue

        # 그래디언트 수집 → FIM 누적
        grad_sq = np.empty(n_params, dtype=np.float64)
        offset = 0
        for p in all_params:
            numel = p.numel()
            if p.grad is not None:
                g = p.grad.detach().cpu().to(torch.float32).numpy().ravel()
                grad_sq[offset : offset + numel] = g ** 2
            else:
                grad_sq[offset : offset + numel] = 0.0
            offset += numel

        fim_accum += grad_sq
        model.zero_grad()
        n_done += 1

        if n_done % 10 == 0:
            logger.debug("FIM 샘플 처리: %d / %d", n_done, n_samples)

    # 원래 requires_grad 상태 복원
    for p, orig in zip(all_params, orig_requires_grad):
        p.requires_grad_(orig)

    if n_done == 0:
        raise RuntimeError("유효한 샘플이 하나도 없어 FIM을 계산할 수 없습니다.")

    fim_diagonal = (fim_accum / n_done).astype(np.float32)
    logger.info("FIM 대각선 계산 완료: %d 샘플, %d 파라미터", n_done, n_params)
    return fim_diagonal


def fim_to_layer_profile(
    fim_diagonal: NDArray[np.floating],
    model: nn.Module,
) -> dict[str, float]:
    """FIM 대각선을 레이어별 L2 노름으로 집계.

    Args:
        fim_diagonal: ``compute_fim_diagonal`` 이 반환한 1D FIM 대각선 배열.
        model: FIM을 계산한 원본 모델 (파라미터 순서가 동일해야 함).

    Returns:
        ``{레이어_이름: FIM_대각선의_L2_노름}`` 딕셔너리.
        각 항목은 해당 레이어의 모든 파라미터 FIM 값의 L2 노름.

    Raises:
        ValueError: fim_diagonal 길이가 모델 파라미터 수와 다를 때.
    """
    n_params = sum(p.numel() for p in model.parameters())
    if len(fim_diagonal) != n_params:
        raise ValueError(
            f"fim_diagonal 길이({len(fim_diagonal)})가 "
            f"모델 파라미터 수({n_params})와 다릅니다."
        )

    profile: dict[str, float] = {}
    offset = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        layer_fim = fim_diagonal[offset : offset + numel]
        norm = float(np.sqrt(np.sum(layer_fim ** 2)))
        profile[name] = norm
        offset += numel

    return profile


def fim_distance(
    fim_a: NDArray[np.floating],
    fim_b: NDArray[np.floating],
    metric: str = "cosine",
) -> float:
    """두 FIM 지문 사이의 거리를 계산.

    Args:
        fim_a: 첫 번째 FIM 대각선 배열.
        fim_b: 두 번째 FIM 대각선 배열.
        metric: 거리 측정 방법. ``"cosine"``, ``"euclidean"``, ``"correlation"`` 중 하나.

    Returns:
        두 FIM 지문 사이의 거리 (float). 코사인/상관의 경우 [0, 2] 범위.

    Raises:
        ValueError: fim_a와 fim_b의 길이가 다르거나 metric이 유효하지 않을 때.
    """
    if len(fim_a) != len(fim_b):
        raise ValueError(
            f"fim_a 길이({len(fim_a)})와 fim_b 길이({len(fim_b)})가 다릅니다."
        )
    if metric not in ("cosine", "euclidean", "correlation"):
        raise ValueError(
            f"지원하지 않는 metric입니다: {metric}. "
            '"cosine", "euclidean", "correlation" 중 선택하세요.'
        )

    a = fim_a.astype(np.float64)
    b = fim_b.astype(np.float64)

    if metric == "euclidean":
        return float(np.sqrt(np.sum((a - b) ** 2)))

    if metric == "cosine":
        norm_a = np.sqrt(np.sum(a ** 2))
        norm_b = np.sqrt(np.sum(b ** 2))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 1.0  # 영벡터는 최대 거리로 처리
        cosine_sim = np.sum(a * b) / (norm_a * norm_b)
        cosine_sim = float(np.clip(cosine_sim, -1.0, 1.0))
        return 1.0 - cosine_sim

    # correlation
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    norm_a = np.sqrt(np.sum(a_centered ** 2))
    norm_b = np.sqrt(np.sum(b_centered ** 2))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    corr = float(np.sum(a_centered * b_centered) / (norm_a * norm_b))
    corr = float(np.clip(corr, -1.0, 1.0))
    return 1.0 - corr
