from __future__ import annotations

# pyright: reportMissingImports=false

import pytest
import torch
import torch.nn as nn

from src.evaluation.cka import CKAAnalyzer, cka, hsic, linear_kernel, rbf_kernel
from src.evaluation.metrics import compute_metrics, crps_empirical, mae, mase, mse


class TestMetrics:
    """예측 평가 지표의 정상/오류 경로를 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_mae_known_values(self) -> None:
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[0.0, 2.0], [1.0, 4.0]])
        result = mae(pred, target)
        torch.testing.assert_close(result, torch.tensor(0.75))

    def test_mae_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            _ = mae(torch.ones(2, 3), torch.ones(2, 2))

    def test_mse_known_values(self) -> None:
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[0.0, 2.0], [1.0, 4.0]])
        result = mse(pred, target)
        torch.testing.assert_close(result, torch.tensor(1.25))

    def test_mse_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            _ = mse(torch.ones(2, 3), torch.ones(2, 2))

    def test_mase_known_values(self) -> None:
        pred = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        target = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
        insample = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        result = mase(pred, target, insample, seasonality=1)
        torch.testing.assert_close(result, torch.tensor(1.5))

    def test_mase_invalid_seasonality_raises(self) -> None:
        with pytest.raises(ValueError):
            _ = mase(
                torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 4), seasonality=0
            )

    def test_crps_empirical_deterministic_samples(self) -> None:
        target = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        samples = target.unsqueeze(1).repeat(1, 4, 1)
        result = crps_empirical(samples, target)
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_compute_metrics_keys(self) -> None:
        pred = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target = torch.tensor([[1.5, 1.5]], dtype=torch.float32)
        insample = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
        samples = pred.unsqueeze(1).repeat(1, 3, 1)

        result = compute_metrics(
            pred, target, insample=insample, samples=samples, seasonality=1
        )
        assert set(result.keys()) == {"mae", "mse", "mase", "crps"}


class TestCKAFunctions:
    """CKA 관련 핵심 수학 함수의 성질을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_linear_and_rbf_kernel_shapes(self) -> None:
        x = torch.randn(5, 3)
        k_lin = linear_kernel(x)
        k_rbf = rbf_kernel(x)
        assert k_lin.shape == (5, 5)
        assert k_rbf.shape == (5, 5)

    def test_hsic_non_negative(self) -> None:
        x = torch.randn(8, 4)
        k = linear_kernel(x)
        val = hsic(k, k)
        assert val.item() >= 0.0

    def test_cka_identical_is_one(self) -> None:
        x = torch.randn(32, 6)
        val = cka(x, x)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_cka_orthogonal_like_is_near_zero(self) -> None:
        x = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
        y = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]])
        val = cka(x, y)
        assert abs(val) < 1e-6

    def test_cka_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            _ = cka(torch.randn(4, 2), torch.randn(5, 2))


class _SimpleCKAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.relu(self.encoder(x)))


class TestCKAAnalyzer:
    """CKAAnalyzer의 훅 등록/활성화 수집/표현 비교를 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_hook_registration_and_activation_capture(self) -> None:
        model = _SimpleCKAModel()
        analyzer = CKAAnalyzer(model, layer_patterns=["encoder", "proj"])
        analyzer.register_hooks()

        x = torch.randn(3, 4)
        _ = model(x)
        activations = analyzer.get_activations()

        assert "encoder" in activations
        assert "proj" in activations
        assert activations["encoder"].shape[0] == 3

        analyzer.clear_activations()
        assert analyzer.get_activations() == {}
        analyzer.remove_hooks()

    def test_compare_representations(self) -> None:
        model = _SimpleCKAModel()
        analyzer = CKAAnalyzer(model, layer_patterns=["encoder", "proj"])

        before = {
            "encoder": torch.randn(2, 4),
            "proj": torch.randn(2, 2),
        }
        after = {
            "encoder": before["encoder"].clone(),
            "proj": before["proj"].clone(),
        }
        result = analyzer.compare_representations(before, after)
        assert set(result.keys()) == {"encoder", "proj"}
        assert result["encoder"] == pytest.approx(1.0, abs=1e-4)
        assert result["proj"] == pytest.approx(1.0, abs=1e-4)
