from __future__ import annotations

# pyright: reportMissingImports=false

import os
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.utils.device import get_device, log_gpu_memory
from src.utils.seed import seed_everything
from src.utils import logging as logging_utils


class TestSeedUtils:
    """시드 고정 유틸리티의 재현성을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_seed_everything_deterministic(self) -> None:
        seed_everything(123)
        py_a = random.random()
        np_a = np.random.rand(3)
        torch_a = torch.rand(3)

        seed_everything(123)
        py_b = random.random()
        np_b = np.random.rand(3)
        torch_b = torch.rand(3)

        assert py_a == py_b
        assert os.environ["PYTHONHASHSEED"] == "123"
        assert np.allclose(np_a, np_b)
        torch.testing.assert_close(torch_a, torch_b)


class TestDeviceUtils:
    """디바이스 선택 및 GPU 메모리 로깅의 CPU 경로를 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_get_device_cpu_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device = get_device()
        assert device.type == "cpu"

    def test_log_gpu_memory_cpu_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert log_gpu_memory() == {}


class TestWandbLogging:
    """wandb 초기화/메트릭 로깅/종료 동작을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_init_wandb_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_run = SimpleNamespace(
            log=Mock(), finish=Mock(), config=SimpleNamespace(update=Mock())
        )
        wandb_module = SimpleNamespace(init=Mock(return_value=fake_run))
        monkeypatch.setattr(logging_utils, "_wandb_run", None)
        monkeypatch.setattr(
            logging_utils.importlib, "import_module", lambda _: wandb_module
        )

        logging_utils.init_wandb(project="p", name="n", config={"a": 1}, tags=["t"])

        wandb_module.init.assert_called_once()
        assert logging_utils._wandb_run is fake_run

    def test_init_wandb_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_import_error(_: str) -> object:
            raise ImportError("no wandb")

        monkeypatch.setattr(logging_utils, "_wandb_run", None)
        monkeypatch.setattr(
            logging_utils.importlib, "import_module", raise_import_error
        )

        logging_utils.init_wandb()
        assert logging_utils._wandb_run is None

    def test_log_metrics_with_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_run = SimpleNamespace(log=Mock(), config=SimpleNamespace(update=Mock()))
        monkeypatch.setattr(logging_utils, "_wandb_run", fake_run)

        logging_utils.log_metrics({"loss": 1.0}, step=5, prefix="train/")

        fake_run.log.assert_called_once_with({"train/loss": 1.0}, step=5)

    def test_log_metrics_without_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(logging_utils, "_wandb_run", None)
        logging_utils.log_metrics({"acc": 0.9}, step=1)

    def test_log_config_with_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_update = Mock()
        fake_run = SimpleNamespace(config=SimpleNamespace(update=cfg_update))
        monkeypatch.setattr(logging_utils, "_wandb_run", fake_run)

        logging_utils.log_config({"lr": 1e-3})
        cfg_update.assert_called_once_with({"lr": 1e-3})

    def test_finish_wandb_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_run = SimpleNamespace(finish=Mock())
        monkeypatch.setattr(logging_utils, "_wandb_run", fake_run)

        logging_utils.finish_wandb()

        fake_run.finish.assert_called_once()
        assert logging_utils._wandb_run is None
