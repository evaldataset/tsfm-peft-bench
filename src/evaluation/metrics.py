from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error.

    Args:
        pred: 예측값, shape (batch, horizon) 또는 (batch, horizon, channels).
        target: 실제값, pred와 동일 shape.

    Returns:
        스칼라 MAE 텐서.

    Raises:
        ValueError: shape이 다를 때.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    return torch.mean(torch.abs(pred - target))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error.

    Args:
        pred: 예측값, shape (batch, horizon) 또는 (batch, horizon, channels).
        target: 실제값, pred와 동일 shape.

    Returns:
        스칼라 MSE 텐서.

    Raises:
        ValueError: shape이 다를 때.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    return torch.mean((pred - target) ** 2)


def mase(
    pred: torch.Tensor,
    target: torch.Tensor,
    insample: torch.Tensor,
    seasonality: int = 1,
) -> torch.Tensor:
    """Mean Absolute Scaled Error.

    Args:
        pred: 예측값, shape (batch, horizon).
        target: 실제값, shape (batch, horizon).
        insample: 학습 데이터(context), shape (batch, context_len).
        seasonality: 계절성 주기 (1이면 naive 예측).

    Returns:
        스칼라 MASE 텐서.

    Raises:
        ValueError: shape 불일치 또는 seasonality가 0 이하일 때.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    if seasonality <= 0:
        raise ValueError(f"seasonality는 양수여야 합니다: {seasonality}")

    naive_errors = torch.abs(insample[:, seasonality:] - insample[:, :-seasonality])
    scale = torch.mean(naive_errors, dim=1, keepdim=True)
    scale = torch.clamp(scale, min=1e-8)

    errors = torch.abs(pred - target)
    return torch.mean(errors / scale)


def crps_empirical(
    samples: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Continuous Ranked Probability Score (경험적).

    Args:
        samples: 확률적 예측 샘플, shape (batch, num_samples, horizon).
        target: 실제값, shape (batch, horizon).

    Returns:
        스칼라 CRPS 텐서.
    """
    num_samples = samples.shape[1]
    target = target.unsqueeze(1)  # (batch, 1, horizon)

    abs_diff = torch.mean(torch.abs(samples - target))

    # 샘플 간 거리
    samples_sorted, _ = torch.sort(samples, dim=1)
    gini = torch.tensor(0.0, device=samples.device)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            gini += torch.mean(torch.abs(samples_sorted[:, i] - samples_sorted[:, j]))

    if num_samples > 1:
        gini = gini / (num_samples * (num_samples - 1) / 2)

    return abs_diff - 0.5 * gini


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    insample: torch.Tensor | None = None,
    samples: torch.Tensor | None = None,
    seasonality: int = 1,
) -> dict[str, float]:
    """모든 메트릭 계산.

    Args:
        pred: 점 예측값.
        target: 실제값.
        insample: 학습 데이터 (MASE용, Optional).
        samples: 확률적 예측 샘플 (CRPS용, Optional).
        seasonality: 계절성 주기.

    Returns:
        메트릭 이름→값 딕셔너리.
    """
    results: dict[str, float] = {
        "mae": mae(pred, target).item(),
        "mse": mse(pred, target).item(),
    }

    if insample is not None:
        results["mase"] = mase(pred, target, insample, seasonality).item()

    if samples is not None:
        results["crps"] = crps_empirical(samples, target).item()

    return results
