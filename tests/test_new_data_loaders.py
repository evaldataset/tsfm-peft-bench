from __future__ import annotations

# pyright: reportMissingImports=false

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray

from src.data.finance import FinanceConfig, FinanceDataset, load_finance
from src.data.physionet import PhysioNetConfig, PhysioNetDataset, load_physionet
from src.data.smd import SMDConfig, SMDDataset, load_smd


@pytest.fixture
def base_series() -> NDArray[np.float32]:
    return np.linspace(0.0, 10.0, 200, dtype=np.float32)


# ========== PhysioNet Tests ==========


class TestPhysioNetConfig:
    """PhysioNet 설정 객체의 기본값을 검증한다."""

    def test_defaults(self) -> None:
        config = PhysioNetConfig()
        assert config.dataset == "PhysioNet"
        assert config.data_dir == "data/physionet"
        assert config.target_col == "HR"
        assert config.context_length == 48
        assert config.prediction_length == 12
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15

    def test_custom_values(self) -> None:
        config = PhysioNetConfig(
            dataset="Custom",
            data_dir="/tmp/phys",
            target_col="SpO2",
            context_length=96,
            prediction_length=24,
        )
        assert config.target_col == "SpO2"
        assert config.context_length == 96


class TestPhysioNetDataset:
    """PhysioNetDataset 슬라이딩 윈도우 동작을 검증한다."""

    def test_length_and_getitem(self, base_series: NDArray[np.float32]) -> None:
        dataset = PhysioNetDataset(
            base_series, context_length=16, prediction_length=8, stride=2
        )
        assert len(dataset) == ((200 - 24) // 2) + 1

        sample = dataset[0]
        assert set(sample.keys()) == {"context", "target"}
        assert sample["context"].shape == (16,)
        assert sample["target"].shape == (8,)

    def test_invalid_stride_raises(self, base_series: NDArray[np.float32]) -> None:
        with pytest.raises(ValueError):
            _ = PhysioNetDataset(
                base_series, context_length=4, prediction_length=2, stride=0
            )

    def test_short_data_raises(self) -> None:
        data = np.arange(5, dtype=np.float32)
        with pytest.raises(ValueError):
            _ = PhysioNetDataset(data, context_length=4, prediction_length=3)

    def test_out_of_range_index_raises(self, base_series: NDArray[np.float32]) -> None:
        dataset = PhysioNetDataset(base_series, context_length=16, prediction_length=8)
        with pytest.raises(IndexError):
            _ = dataset[-1]
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

    def test_invalid_data_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _ = PhysioNetDataset(
                [1.0, 2.0, 3.0],  # type: ignore[arg-type]
                context_length=2,
                prediction_length=1,
            )


class TestLoadPhysioNet:
    """PhysioNet 로더의 분할/검증 로직을 테스트한다."""

    def test_load_success(self, tmp_path: Path) -> None:
        phys_dir = tmp_path / "physionet"
        phys_dir.mkdir()
        for i in range(3):
            df = pd.DataFrame(
                {"HR": np.linspace(0, 1, 200, dtype=np.float32) + i * 0.1}
            )
            df.to_csv(phys_dir / f"patient_{i}.csv", index=False)

        cfg = PhysioNetConfig(
            data_dir=str(phys_dir),
            target_col="HR",
            context_length=16,
            prediction_length=8,
        )
        train_ds, val_ds, test_ds = load_physionet(cfg)
        assert isinstance(train_ds, PhysioNetDataset)
        assert isinstance(val_ds, PhysioNetDataset)
        assert isinstance(test_ds, PhysioNetDataset)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_load_missing_dir_raises(self) -> None:
        cfg = PhysioNetConfig(data_dir="/tmp/nonexistent_physionet_dir_xyz")
        with pytest.raises(FileNotFoundError):
            _ = load_physionet(cfg)

    def test_load_empty_dir_raises(self, tmp_path: Path) -> None:
        phys_dir = tmp_path / "empty_physionet"
        phys_dir.mkdir()
        cfg = PhysioNetConfig(data_dir=str(phys_dir))
        with pytest.raises(FileNotFoundError):
            _ = load_physionet(cfg)

    def test_load_missing_target_col_raises(self, tmp_path: Path) -> None:
        phys_dir = tmp_path / "physionet"
        phys_dir.mkdir()
        pd.DataFrame({"X": np.arange(200, dtype=np.float32)}).to_csv(
            phys_dir / "patient.csv", index=False
        )
        cfg = PhysioNetConfig(data_dir=str(phys_dir), target_col="HR")
        with pytest.raises(ValueError):
            _ = load_physionet(cfg)

    def test_load_invalid_ratio_raises(self, tmp_path: Path) -> None:
        phys_dir = tmp_path / "physionet"
        phys_dir.mkdir()
        pd.DataFrame({"HR": np.arange(200, dtype=np.float32)}).to_csv(
            phys_dir / "patient.csv", index=False
        )
        cfg = PhysioNetConfig(
            data_dir=str(phys_dir),
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        with pytest.raises(ValueError):
            _ = load_physionet(cfg)


# ========== SMD / PSM Tests ==========


class TestSMDConfig:
    """SMD/PSM 설정 객체의 기본값을 검증한다."""

    def test_defaults(self) -> None:
        config = SMDConfig()
        assert config.dataset == "SMD"
        assert config.path == "data/SMD"
        assert config.target_col == 0
        assert config.context_length == 100
        assert config.prediction_length == 25
        assert config.val_ratio == 0.2

    def test_psm_config(self) -> None:
        config = SMDConfig(dataset="PSM", path="data/PSM", target_col=1)
        assert config.dataset == "PSM"
        assert config.target_col == 1


class TestSMDDataset:
    """SMDDataset 슬라이딩 윈도우 동작을 검증한다."""

    def test_length_and_getitem(self, base_series: NDArray[np.float32]) -> None:
        dataset = SMDDataset(
            base_series, context_length=20, prediction_length=10, stride=5
        )
        expected = ((200 - 30) // 5) + 1
        assert len(dataset) == expected

        sample = dataset[0]
        assert set(sample.keys()) == {"context", "target"}
        assert sample["context"].shape == (20,)
        assert sample["target"].shape == (10,)

    def test_invalid_stride_raises(self, base_series: NDArray[np.float32]) -> None:
        with pytest.raises(ValueError):
            _ = SMDDataset(
                base_series, context_length=10, prediction_length=5, stride=-1
            )

    def test_short_data_raises(self) -> None:
        data = np.arange(5, dtype=np.float32)
        with pytest.raises(ValueError):
            _ = SMDDataset(data, context_length=4, prediction_length=3)


class TestLoadSMD:
    """SMD 로더의 파일 읽기 및 분할 로직을 테스트한다."""

    def test_load_smd_success(self, tmp_path: Path) -> None:
        smd_dir = tmp_path / "SMD"
        train_dir = smd_dir / "train"
        test_dir = smd_dir / "test"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        # SMD format: space-separated, no headers, 38 cols
        n_cols = 5
        n_rows = 300
        for name in ["machine-1-1.txt", "machine-1-2.txt"]:
            data = np.random.randn(n_rows, n_cols)
            np.savetxt(train_dir / name, data, delimiter=",")
            np.savetxt(test_dir / name, data, delimiter=",")

        cfg = SMDConfig(
            dataset="SMD",
            path=str(smd_dir),
            target_col=0,
            context_length=20,
            prediction_length=10,
        )
        train_ds, val_ds, test_ds = load_smd(cfg)
        assert isinstance(train_ds, SMDDataset)
        assert isinstance(val_ds, SMDDataset)
        assert isinstance(test_ds, SMDDataset)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_load_psm_success(self, tmp_path: Path) -> None:
        psm_dir = tmp_path / "PSM"
        psm_dir.mkdir()

        n_rows = 300
        train_df = pd.DataFrame(
            {
                "timestamp_(min)": range(n_rows),
                "feature_0": np.random.randn(n_rows).astype(np.float32),
                "feature_1": np.random.randn(n_rows).astype(np.float32),
            }
        )
        test_df = pd.DataFrame(
            {
                "timestamp_(min)": range(n_rows),
                "feature_0": np.random.randn(n_rows).astype(np.float32),
                "feature_1": np.random.randn(n_rows).astype(np.float32),
            }
        )
        train_df.to_csv(psm_dir / "train.csv", index=False)
        test_df.to_csv(psm_dir / "test.csv", index=False)

        cfg = SMDConfig(
            dataset="PSM",
            path=str(psm_dir),
            target_col="feature_0",
            context_length=20,
            prediction_length=10,
        )
        train_ds, val_ds, test_ds = load_smd(cfg)
        assert isinstance(train_ds, SMDDataset)
        assert len(train_ds) > 0

    def test_load_smd_missing_dir_raises(self) -> None:
        cfg = SMDConfig(path="/tmp/nonexistent_smd_xyz")
        with pytest.raises(FileNotFoundError):
            _ = load_smd(cfg)

    def test_load_psm_missing_files_raises(self, tmp_path: Path) -> None:
        psm_dir = tmp_path / "PSM_empty"
        psm_dir.mkdir()
        cfg = SMDConfig(dataset="PSM", path=str(psm_dir))
        with pytest.raises(FileNotFoundError):
            _ = load_smd(cfg)

    def test_unsupported_dataset_raises(self, tmp_path: Path) -> None:
        cfg = SMDConfig(dataset="INVALID", path=str(tmp_path))
        with pytest.raises(ValueError):
            _ = load_smd(cfg)


# ========== Finance Tests ==========


class TestFinanceConfig:
    """Finance 설정 객체의 기본값을 검증한다."""

    def test_defaults(self) -> None:
        config = FinanceConfig()
        assert config.dataset == "ExchangeRate"
        assert config.path == "data/exchange_rate/exchange_rate.csv"
        assert config.target_col == "AUD"
        assert config.context_length == 96
        assert config.prediction_length == 24
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.1
        assert config.test_ratio == 0.2

    def test_custom_values(self) -> None:
        config = FinanceConfig(
            dataset="Custom",
            path="/tmp/fin.csv",
            target_col=2,
            context_length=48,
        )
        assert config.target_col == 2
        assert config.context_length == 48


class TestFinanceDataset:
    """FinanceDataset 슬라이딩 윈도우 동작을 검증한다."""

    def test_length_and_getitem(self, base_series: NDArray[np.float32]) -> None:
        dataset = FinanceDataset(
            base_series, context_length=24, prediction_length=12, stride=3
        )
        expected = ((200 - 36) // 3) + 1
        assert len(dataset) == expected

        sample = dataset[0]
        assert set(sample.keys()) == {"context", "target"}
        assert sample["context"].shape == (24,)
        assert sample["target"].shape == (12,)

    def test_short_data_raises(self) -> None:
        data = np.arange(10, dtype=np.float32)
        with pytest.raises(ValueError):
            _ = FinanceDataset(data, context_length=8, prediction_length=6)


class TestLoadFinance:
    """Finance 로더의 분할/검증 로직을 테스트한다."""

    def test_load_success(self, tmp_path: Path) -> None:
        fin_dir = tmp_path / "exchange_rate"
        fin_dir.mkdir()
        path = fin_dir / "exchange_rate.csv"

        n_rows = 500
        df = pd.DataFrame(
            {
                "date": pd.date_range("2010-01-01", periods=n_rows, freq="D"),
                "0": np.random.randn(n_rows).astype(np.float32),
                "1": np.random.randn(n_rows).astype(np.float32),
                "2": np.random.randn(n_rows).astype(np.float32),
            }
        )
        df.to_csv(path, index=False)

        cfg = FinanceConfig(
            path=str(path),
            target_col="0",
            context_length=24,
            prediction_length=12,
        )
        train_ds, val_ds, test_ds = load_finance(cfg)
        assert isinstance(train_ds, FinanceDataset)
        assert isinstance(val_ds, FinanceDataset)
        assert isinstance(test_ds, FinanceDataset)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_load_with_int_column(self, tmp_path: Path) -> None:
        path = tmp_path / "fin.csv"
        n_rows = 500
        df = pd.DataFrame(
            {
                "col_a": np.random.randn(n_rows).astype(np.float32),
                "col_b": np.random.randn(n_rows).astype(np.float32),
            }
        )
        df.to_csv(path, index=False)

        cfg = FinanceConfig(
            path=str(path),
            target_col=0,
            context_length=24,
            prediction_length=12,
        )
        train_ds, val_ds, test_ds = load_finance(cfg)
        assert len(train_ds) > 0

    def test_load_missing_file_raises(self) -> None:
        cfg = FinanceConfig(path="/tmp/nonexistent_finance_xyz.csv")
        with pytest.raises(FileNotFoundError):
            _ = load_finance(cfg)

    def test_load_invalid_ratio_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "fin.csv"
        pd.DataFrame({"0": np.arange(500, dtype=np.float32)}).to_csv(path, index=False)
        cfg = FinanceConfig(
            path=str(path),
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.3,
        )
        with pytest.raises(ValueError):
            _ = load_finance(cfg)

    def test_load_missing_target_col_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "fin.csv"
        pd.DataFrame({"X": np.arange(500, dtype=np.float32)}).to_csv(path, index=False)
        cfg = FinanceConfig(path=str(path), target_col="nonexistent")
        with pytest.raises(ValueError):
            _ = load_finance(cfg)


# ========== Adapter Tests ==========


class TestAdapterConfig:
    """Adapter 설정 객체의 기본값을 검증한다."""

    def test_defaults(self) -> None:
        from src.adaptation.adapter import AdapterAdaptationConfig

        config = AdapterAdaptationConfig()
        assert config.bottleneck_size == 64
        assert config.task_type == "SEQ_2_SEQ_LM"


class TestBottleneckAdapterLinear:
    """BottleneckAdapterLinear의 forward 및 requires_grad를 검증한다."""

    def test_forward_residual(self) -> None:
        from src.adaptation.adapter import _BottleneckAdapterLinear

        base = torch.nn.Linear(8, 8)
        adapter = _BottleneckAdapterLinear(base, bottleneck_size=4)

        x = torch.randn(2, 8)
        out = adapter(x)
        assert out.shape == (2, 8)

        # Base params frozen
        for p in adapter.base.parameters():
            assert p.requires_grad is False

        # Adapter params trainable
        for p in adapter.down.parameters():
            assert p.requires_grad is True
        for p in adapter.up.parameters():
            assert p.requires_grad is True

    def test_output_differs_from_base(self) -> None:
        from src.adaptation.adapter import _BottleneckAdapterLinear

        base = torch.nn.Linear(8, 8)
        x = torch.randn(2, 8)
        base_out = base(x)

        adapter = _BottleneckAdapterLinear(base, bottleneck_size=4)
        adapter_out = adapter(x)

        # Adapter adds residual, so output should differ from base
        assert not torch.allclose(base_out.detach(), adapter_out.detach())


class TestApplyAdapterFallback:
    """PEFT 없이 수동 bottleneck adapter를 적용하는 로직을 검증한다."""

    def test_manual_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.adaptation.adapter import (
            AdapterAdaptationConfig,
            _apply_bottleneck_fallback,
        )

        # Create a simple model with known linear layers
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.Linear(8, 4),
        )

        cfg = AdapterAdaptationConfig(bottleneck_size=4)
        result = _apply_bottleneck_fallback(model, cfg)

        # Should have wrapped linear layers
        trainable = sum(1 for p in result.parameters() if p.requires_grad)
        assert trainable > 0
