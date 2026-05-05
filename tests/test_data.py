from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray

from src.data.ett import ETTConfig, ETTDataset, load_ett
from src.data.shift import ShiftGenerator, ShiftSeverity, ShiftType
from src.data.shift_metrics import (
    ShiftProfile,
    acf,
    compute_shift_profile,
    dominant_period,
    normalized_psd,
    permutation_entropy,
    wasserstein_1d,
)


@pytest.fixture
def base_series() -> NDArray[np.float32]:
    return np.linspace(0.0, 10.0, 64, dtype=np.float32)


class TestETTConfig:
    """ETT 설정 객체의 기본/사용자 지정 값을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_defaults(self) -> None:
        config = ETTConfig()
        assert config.dataset == "ETTm1"
        assert config.path == "data/ETT-small/ETTm1.csv"
        assert config.target_col == "OT"
        assert config.context_length == 512
        assert config.prediction_length == 96
        assert config.train_ratio == 0.6
        assert config.val_ratio == 0.2
        assert config.test_ratio == 0.2

    def test_custom_values(self) -> None:
        config = ETTConfig(
            dataset="ETTh1",
            path="custom.csv",
            target_col="target",
            context_length=24,
            prediction_length=12,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
        )
        assert config.dataset == "ETTh1"
        assert config.path == "custom.csv"
        assert config.target_col == "target"
        assert config.context_length == 24
        assert config.prediction_length == 12
        assert config.train_ratio == 0.7


class TestETTDataset:
    """ETTDataset 슬라이딩 윈도우 동작과 오류 처리를 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_length_and_getitem(self, base_series: NDArray[np.float32]) -> None:
        dataset = ETTDataset(
            base_series, context_length=8, prediction_length=4, stride=2
        )
        assert len(dataset) == ((64 - 12) // 2) + 1

        sample = dataset[1]
        assert set(sample.keys()) == {"context", "target"}
        assert sample["context"].shape == (8,)
        assert sample["target"].shape == (4,)

        expected_context = torch.tensor(base_series[2:10], dtype=torch.float32)
        expected_target = torch.tensor(base_series[10:14], dtype=torch.float32)
        torch.testing.assert_close(sample["context"], expected_context)
        torch.testing.assert_close(sample["target"], expected_target)

    def test_invalid_stride_raises(self, base_series: NDArray[np.float32]) -> None:
        with pytest.raises(ValueError):
            _ = ETTDataset(base_series, context_length=4, prediction_length=2, stride=0)

    def test_short_data_raises(self) -> None:
        data = np.arange(5, dtype=np.float32)
        with pytest.raises(ValueError):
            _ = ETTDataset(data, context_length=4, prediction_length=3)

    def test_out_of_range_index_raises(self, base_series: NDArray[np.float32]) -> None:
        dataset = ETTDataset(base_series, context_length=8, prediction_length=4)
        with pytest.raises(IndexError):
            _ = dataset[-1]
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]


class TestLoadETT:
    """CSV 기반 ETT 로더의 분할/검증 로직을 테스트한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_load_success(self, tmp_path: Path) -> None:
        path = tmp_path / "ett.csv"
        df = pd.DataFrame({"OT": np.linspace(0, 1, 120, dtype=np.float32)})
        df.to_csv(path, index=False)

        cfg = ETTConfig(
            path=str(path),
            context_length=16,
            prediction_length=8,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        train_ds, val_ds, test_ds = load_ett(cfg)
        assert isinstance(train_ds, ETTDataset)
        assert isinstance(val_ds, ETTDataset)
        assert isinstance(test_ds, ETTDataset)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_load_missing_file_raises(self) -> None:
        cfg = ETTConfig(path="/tmp/non-existing-file.csv")
        with pytest.raises(FileNotFoundError):
            _ = load_ett(cfg)

    def test_load_invalid_ratio_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ett.csv"
        pd.DataFrame({"OT": np.arange(200, dtype=np.float32)}).to_csv(path, index=False)
        cfg = ETTConfig(path=str(path), train_ratio=0.7, val_ratio=0.2, test_ratio=0.2)
        with pytest.raises(ValueError):
            _ = load_ett(cfg)

    def test_load_missing_target_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ett.csv"
        pd.DataFrame({"X": np.arange(200, dtype=np.float32)}).to_csv(path, index=False)
        cfg = ETTConfig(path=str(path), target_col="OT")
        with pytest.raises(ValueError):
            _ = load_ett(cfg)


class TestShiftEnums:
    """분포 이동 열거형의 값 무결성을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_shift_type_values(self) -> None:
        assert ShiftType.AMPLITUDE.value == "amplitude"
        assert ShiftType.SPECTRAL.value == "spectral"
        assert ShiftType.IRREGULARITY.value == "irregularity"
        assert ShiftType.NONSTATIONARITY.value == "nonstationarity"

    def test_shift_severity_values(self) -> None:
        assert ShiftSeverity.MILD.value == "mild"
        assert ShiftSeverity.STRONG.value == "strong"


class TestShiftGenerator:
    """ShiftGenerator의 모든 이동 유형/강도 조합을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    @pytest.fixture
    def generator(self) -> ShiftGenerator:
        return ShiftGenerator(seed=123)

    def test_all_shift_type_and_severity_combinations(
        self, generator: ShiftGenerator
    ) -> None:
        ts = np.sin(np.linspace(0.0, 8.0 * np.pi, 128, dtype=np.float32))
        for shift_type in ShiftType:
            for severity in ShiftSeverity:
                shifted = generator.apply(ts, shift_type=shift_type, severity=severity)
                assert shifted.shape == ts.shape
                assert shifted.dtype == np.float32
                assert not np.allclose(shifted, ts)

    def test_invalid_dimension_raises(self, generator: ShiftGenerator) -> None:
        ts = np.zeros((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            _ = generator.apply(ts, ShiftType.AMPLITUDE, ShiftSeverity.MILD)


class TestShiftMetrics:
    """분포 이동 지표 함수들의 정상/에러 경로를 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_wasserstein_1d(self) -> None:
        a = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert wasserstein_1d(a, b) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            _ = wasserstein_1d(np.array([]), b)

    def test_normalized_psd(self) -> None:
        ts = np.sin(np.linspace(0.0, 4.0 * np.pi, 64, dtype=np.float64))
        psd = normalized_psd(ts)
        assert psd.ndim == 1
        assert float(np.sum(psd)) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            _ = normalized_psd(np.zeros((10, 2), dtype=np.float64))

    def test_acf(self) -> None:
        ts = np.arange(20, dtype=np.float64)
        out = acf(ts, max_lag=5)
        assert out.shape == (5,)
        with pytest.raises(ValueError):
            _ = acf(np.zeros((10, 2), dtype=np.float64), max_lag=3)
        with pytest.raises(ValueError):
            _ = acf(ts, max_lag=0)

    def test_dominant_period(self) -> None:
        x = np.arange(100, dtype=np.float64)
        ts = np.sin(2 * np.pi * x / 10.0)
        period = dominant_period(ts)
        assert period == pytest.approx(10.0, rel=0.1)
        with pytest.raises(ValueError):
            _ = dominant_period(np.zeros((3, 3), dtype=np.float64))

    def test_permutation_entropy(self) -> None:
        ts = np.linspace(0.0, 1.0, 30, dtype=np.float64)
        val = permutation_entropy(ts, order=3, delay=1)
        assert -1e-9 <= val <= 1.0 + 1e-9
        with pytest.raises(ValueError):
            _ = permutation_entropy(np.zeros((3, 3), dtype=np.float64))
        with pytest.raises(ValueError):
            _ = permutation_entropy(ts, order=1)
        with pytest.raises(ValueError):
            _ = permutation_entropy(ts, order=3, delay=0)

    def test_compute_shift_profile_valid(self) -> None:
        source = np.sin(np.linspace(0.0, 8.0 * np.pi, 128, dtype=np.float64))
        target = source * 1.5 + 0.2
        profile = compute_shift_profile(source, target)
        assert isinstance(profile, ShiftProfile)

        values: dict[str, Any] = {
            field.name: getattr(profile, field.name) for field in fields(ShiftProfile)
        }
        assert np.isfinite(values["amplitude_delta_log_std"])
        assert np.isfinite(values["amplitude_delta_mean"])
        non_negative_keys = {
            "amplitude_w1",
            "spectral_w1",
            "spectral_bandpower_l1",
            "acf_distance",
            "acf_delta_period",
            "irregularity_missing_rate",
            "irregularity_perm_entropy_diff",
            "nonstationarity_kpss_diff",
            "nonstationarity_changepoint_diff",
        }
        for key in non_negative_keys:
            assert values[key] >= 0.0

    def test_compute_shift_profile_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            _ = compute_shift_profile(np.zeros((4, 2)), np.zeros(8))
        with pytest.raises(ValueError):
            _ = compute_shift_profile(np.array([], dtype=np.float64), np.ones(4))
