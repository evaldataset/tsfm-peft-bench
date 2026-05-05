from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float32]


class DictConfig(Protocol):
    def __getattr__(self, name: str) -> object: ...


class _TimesFmHparamsClassProtocol(Protocol):
    def __call__(
        self,
        *,
        context_len: int,
        horizon_len: int,
        input_patch_len: int,
        output_patch_len: int,
        num_layers: int,
        num_heads: int,
        model_dims: int,
        backend: str,
        per_core_batch_size: int,
    ) -> object: ...


class _TimesFmCheckpointClassProtocol(Protocol):
    def __call__(self, *, huggingface_repo_id: str) -> object: ...


class _TimesFmInstanceProtocol(Protocol):
    _model: nn.Module

    def forecast(
        self,
        *,
        inputs: list[FloatArray],
        freq: list[int],
    ) -> tuple[object, object]: ...


class _TimesFmClassProtocol(Protocol):
    def __call__(
        self, *, hparams: object, checkpoint: object
    ) -> _TimesFmInstanceProtocol: ...


class _TimesFmModuleProtocol(Protocol):
    TimesFm: _TimesFmClassProtocol
    TimesFmHparams: _TimesFmHparamsClassProtocol
    TimesFmCheckpoint: _TimesFmCheckpointClassProtocol


@dataclass
class TimesFMWrapperConfig:
    """TimesFM 래퍼 설정.

    Args:
        hf_id: HuggingFace 모델 ID.
        context_length: 입력 컨텍스트 길이.
        prediction_length: 기본 예측 길이.
        backend: TimesFM 런타임 백엔드("gpu" 또는 "cpu").
        freq: 시계열 주기 인코딩(0=high, 1=medium, 2=low).
        per_core_batch_size: TimesFM 내부 배치 크기.

    Returns:
        TimesFMWrapperConfig 인스턴스.

    Raises:
        ValueError: 길이/주기 파라미터가 유효 범위를 벗어날 때.
    """

    hf_id: str = "google/timesfm-1.0-200m-pytorch"
    context_length: int = 512
    prediction_length: int = 128
    backend: str = "gpu"
    freq: int = 0
    per_core_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.hf_id == "":
            raise ValueError("hf_id는 비어 있지 않은 문자열이어야 합니다.")
        if self.context_length < 1:
            raise ValueError(
                f"context_length는 1 이상이어야 합니다. 현재: {self.context_length}"
            )
        if self.prediction_length < 1:
            raise ValueError(
                f"prediction_length는 1 이상이어야 합니다. 현재: {self.prediction_length}"
            )
        if self.freq not in (0, 1, 2):
            raise ValueError(f"freq는 0, 1, 2 중 하나여야 합니다. 현재: {self.freq}")
        if self.per_core_batch_size < 1:
            raise ValueError(
                f"per_core_batch_size는 1 이상이어야 합니다. 현재: {self.per_core_batch_size}"
            )

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> TimesFMWrapperConfig:
        """Hydra DictConfig를 TimesFM 래퍼 설정으로 변환.

        Args:
            config: 모델 설정 DictConfig.

        Returns:
            TimesFMWrapperConfig 인스턴스.

        Raises:
            ValueError: ``hf_id``가 문자열이 아닐 때.
        """

        hf_id_obj = getattr(config, "hf_id", "google/timesfm-1.0-200m-pytorch")
        if not isinstance(hf_id_obj, str) or hf_id_obj == "":
            raise ValueError("config.hf_id는 비어 있지 않은 문자열이어야 합니다.")

        return cls(
            hf_id=hf_id_obj,
            context_length=int(cast(int, getattr(config, "context_length", 512))),
            prediction_length=int(cast(int, getattr(config, "prediction_length", 128))),
            backend=str(cast(str, getattr(config, "backend", "gpu"))),
            freq=int(cast(int, getattr(config, "freq", 0))),
            per_core_batch_size=int(
                cast(int, getattr(config, "per_core_batch_size", 32))
            ),
        )


class TimesFMWrapper:
    """TimesFM 기반 시계열 모델 래퍼.

    Args:
        config: Hydra 모델 설정.

    Returns:
        TimesFMWrapper 인스턴스.

    Raises:
        ValueError: 설정 값이 유효하지 않을 때.
    """

    _INPUT_PATCH_LEN: int = 32
    _OUTPUT_PATCH_LEN: int = 128
    _NUM_LAYERS: int = 20
    _NUM_HEADS: int = 16
    _MODEL_DIMS: int = 1280

    def __init__(self, config: DictConfig) -> None:
        self.config: TimesFMWrapperConfig = TimesFMWrapperConfig.from_dict_config(
            config
        )
        self.model: _TimesFmInstanceProtocol | None = None
        self.backbone: nn.Module | None = None

    def load(self) -> None:
        """TimesFM 패키지 동적 로드 후 모델 초기화.

        Args:
            None.

        Returns:
            None.

        Raises:
            ImportError: ``timesfm`` 패키지를 찾을 수 없을 때.
            ValueError: 내부 백본 추출에 실패했을 때.
        """

        try:
            timesfm_module = cast(
                _TimesFmModuleProtocol,
                cast(object, import_module("timesfm")),
            )
        except ImportError as exc:
            raise ImportError(
                "timesfm 패키지를 찾을 수 없습니다. `pip install timesfm`를 확인하세요."
            ) from exc

        backend = self.config.backend
        if backend == "gpu" and not torch.cuda.is_available():
            logger.warning("CUDA를 사용할 수 없어 TimesFM backend를 cpu로 전환합니다.")
            backend = "cpu"

        hparams = timesfm_module.TimesFmHparams(
            context_len=self.config.context_length,
            horizon_len=self.config.prediction_length,
            input_patch_len=self._INPUT_PATCH_LEN,
            output_patch_len=self._OUTPUT_PATCH_LEN,
            num_layers=self._NUM_LAYERS,
            num_heads=self._NUM_HEADS,
            model_dims=self._MODEL_DIMS,
            backend=backend,
            per_core_batch_size=self.config.per_core_batch_size,
        )
        checkpoint = timesfm_module.TimesFmCheckpoint(
            huggingface_repo_id=self.config.hf_id
        )
        self.model = timesfm_module.TimesFm(hparams=hparams, checkpoint=checkpoint)

        if not hasattr(self.model, "_model"):
            raise ValueError("TimesFM 내부 백본(_model)을 찾을 수 없습니다.")

        backbone_obj = cast(object, getattr(self.model, "_model"))
        if not isinstance(backbone_obj, nn.Module):
            raise ValueError("TimesFM _model이 nn.Module 타입이 아닙니다.")

        self.backbone = backbone_obj
        logger.info("TimesFM 모델 로드 완료: %s", self.config.hf_id)

    def get_backbone(self) -> nn.Module:
        """PEFT 적용 대상 백본 모듈 반환.

        Args:
            None.

        Returns:
            TimesFM 내부 PatchedTimeSeriesDecoder 모듈.

        Raises:
            ValueError: 모델이 아직 로드되지 않았을 때.
        """

        if self.backbone is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")
        return self.backbone

    def _prepare_context_batch(self, context: torch.Tensor) -> torch.Tensor:
        """컨텍스트 텐서를 배치 형태로 정규화.

        Args:
            context: shape ``(batch, context_len)`` 또는 ``(context_len,)``.

        Returns:
            shape ``(batch, context_len)`` 텐서.

        Raises:
            ValueError: context 차원이 유효하지 않을 때.
        """

        if context.ndim == 1:
            return context.unsqueeze(0)
        if context.ndim == 2:
            return context
        raise ValueError(
            f"context는 1D 또는 2D여야 합니다. 현재 shape: {context.shape}"
        )

    def _prepare_target_batch(self, target: torch.Tensor) -> torch.Tensor:
        """타깃 텐서를 배치 형태로 정규화.

        Args:
            target: shape ``(batch, pred_len)`` 또는 ``(pred_len,)``.

        Returns:
            shape ``(batch, pred_len)`` 텐서.

        Raises:
            ValueError: target 차원이 유효하지 않을 때.
        """

        if target.ndim == 1:
            return target.unsqueeze(0)
        if target.ndim == 2:
            return target
        raise ValueError(f"target은 1D 또는 2D여야 합니다. 현재 shape: {target.shape}")

    def _get_backbone_device(self) -> torch.device:
        """백본 파라미터 기준 디바이스를 반환.

        Args:
            None.

        Returns:
            백본이 위치한 torch.device.
        """

        backbone = self.get_backbone()
        try:
            return next(backbone.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _extract_backbone_prediction(
        self,
        output: object,
        batch_size: int,
        horizon: int,
    ) -> torch.Tensor:
        """백본 출력 객체에서 예측 텐서를 추출.

        Args:
            output: 백본 forward 결과 객체.
            batch_size: 배치 크기.
            horizon: 타깃 예측 길이.

        Returns:
            shape ``(batch, horizon)`` 예측 텐서.

        Raises:
            ValueError: 출력에서 예측 텐서를 추출할 수 없을 때.
        """

        pred: torch.Tensor | None = None
        if isinstance(output, torch.Tensor):
            pred = output
        elif isinstance(output, dict):
            if "pred" in output and isinstance(output["pred"], torch.Tensor):
                pred = output["pred"]
            elif "prediction" in output and isinstance(
                output["prediction"], torch.Tensor
            ):
                pred = output["prediction"]
            elif "mean" in output and isinstance(output["mean"], torch.Tensor):
                pred = output["mean"]
        else:
            for attr_name in ("pred", "prediction", "mean", "logits"):
                attr_obj = getattr(output, attr_name, None)
                if isinstance(attr_obj, torch.Tensor):
                    pred = attr_obj
                    break

        if pred is None:
            raise ValueError("TimesFM 백본 출력에서 예측 텐서를 찾지 못했습니다.")

        if pred.ndim == 3:
            pred = pred[..., 0]
        elif pred.ndim != 2:
            raise ValueError(
                f"TimesFM 백본 예측 shape이 예상과 다릅니다: {tuple(pred.shape)}"
            )

        if pred.shape[0] != batch_size:
            raise ValueError(
                f"TimesFM 백본 배치 크기가 다릅니다: pred={pred.shape[0]}, expected={batch_size}"
            )
        if pred.shape[-1] < horizon:
            raise ValueError(
                f"TimesFM 백본 출력 길이({pred.shape[-1]})가 target 길이({horizon})보다 짧습니다."
            )
        return pred[:, :horizon]

    def _forward_with_backbone(
        self,
        context_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """내부 백본 경로를 통한 미분 가능 forward를 시도.

        Args:
            context_batch: shape ``(batch, context_len)`` 컨텍스트.
            target_batch: shape ``(batch, pred_len)`` 타깃.

        Returns:
            ``{"loss": loss, "pred": pred}`` 딕셔너리.

        Raises:
            ValueError: 백본 출력 파싱에 실패했을 때.
            RuntimeError: 백본 호출이 실패했을 때.
        """

        backbone = self.get_backbone()
        model_device = self._get_backbone_device()
        horizon = int(target_batch.shape[-1])
        batch_size = context_batch.shape[0]

        context_model = context_batch.to(device=model_device, dtype=torch.float32)
        remainder = context_model.shape[-1] % self._INPUT_PATCH_LEN
        if remainder != 0:
            pad_len = self._INPUT_PATCH_LEN - remainder
            context_model = F.pad(context_model, (pad_len, 0))

        # paddings length must = input_ts length + horizon_len for decode()
        full_pad_len = context_model.shape[-1] + horizon
        input_padding = torch.zeros(
            batch_size, full_pad_len,
            dtype=torch.long, device=model_device,
        )
        freq_tensor = torch.zeros(
            batch_size, 1, dtype=torch.long, device=model_device,
        )

        output: object | None = None
        call_errors: list[str] = []

        # Try decode() first (returns mean/quantile predictions directly)
        decode_fn = getattr(backbone, 'decode', None)
        if callable(decode_fn):
            try:
                mean_output, full_output = decode_fn(
                    input_ts=context_model,
                    paddings=input_padding.float(),
                    freq=freq_tensor,
                    horizon_len=horizon,
                    output_patch_len=self._OUTPUT_PATCH_LEN,
                    return_forecast_on_context=False,
                )
                # mean_output shape: (batch, horizon)
                pred = mean_output[:, :horizon]
                target_for_loss = target_batch.to(device=pred.device, dtype=pred.dtype)
                loss = F.mse_loss(pred, target_for_loss)
                pred_out = pred.to(device=target_batch.device, dtype=target_batch.dtype)
                return {"loss": loss, "pred": pred_out}
            except (TypeError, RuntimeError, AttributeError) as exc:
                call_errors.append(f"decode: {type(exc).__name__}: {exc}")

        # Try forward() (returns internal representations)
        call_candidates: list[dict[str, object]] = [
            {
                "input_ts": context_model,
                "input_padding": input_padding,
                "freq": freq_tensor,
            },
        ]

        for kwargs in call_candidates:
            try:
                output = cast(object, backbone(**kwargs))
                break
            except (TypeError, RuntimeError, AttributeError) as exc:
                call_errors.append(f"forward: {type(exc).__name__}: {exc}")

        if output is None:
            raise RuntimeError(
                f"TimesFM 백본 호출에 실패했습니다. "
                f"시도한 경로 수={len(call_candidates) + 1}, "
                f"마지막 오류={call_errors[-1] if call_errors else 'unknown'}"
            )

        pred = self._extract_backbone_prediction(
            output=output,
            batch_size=batch_size,
            horizon=horizon,
        )
        target_for_loss = target_batch.to(device=pred.device, dtype=pred.dtype)
        loss = F.mse_loss(pred, target_for_loss)

        pred_out = pred.to(device=target_batch.device, dtype=target_batch.dtype)
        return {"loss": loss, "pred": pred_out}

    def _forward_with_forecast(
        self,
        context_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """forecast API를 통한 fallback forward(비미분 경로).

        Args:
            context_batch: shape ``(batch, context_len)`` 컨텍스트.
            target_batch: shape ``(batch, pred_len)`` 타깃.

        Returns:
            ``{"loss": loss, "pred": pred}`` 딕셔너리.
        """

        horizon = int(target_batch.shape[-1])
        pred = self.predict(context=context_batch, prediction_length=horizon)
        target_for_loss = target_batch.to(device=pred.device, dtype=pred.dtype)
        loss = F.mse_loss(pred, target_for_loss)
        pred_out = pred.to(device=target_batch.device, dtype=target_batch.dtype)
        return {"loss": loss, "pred": pred_out}

    def predict(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor:
        """TimesFM forecast() 기반 점 예측 수행.

        Args:
            context: 입력 컨텍스트, shape ``(batch, context_len)`` 또는 ``(context_len,)``.
            prediction_length: 요청 예측 길이.

        Returns:
            shape ``(batch, prediction_length)`` 점예측 텐서.

        Raises:
            ValueError: 모델이 로드되지 않았거나 요청 길이가 유효하지 않을 때.
        """

        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")
        if prediction_length < 1:
            raise ValueError(
                f"prediction_length는 1 이상이어야 합니다. 현재: {prediction_length}"
            )

        context_batch = self._prepare_context_batch(context)
        context_cpu = context_batch.detach().to(device="cpu", dtype=torch.float32)
        inputs: list[FloatArray] = []
        for idx in range(context_cpu.shape[0]):
            values: list[float] = [float(v.item()) for v in context_cpu[idx]]
            context_np: FloatArray = np.asarray(values, dtype=np.float32)
            inputs.append(context_np)
        freq_list = [self.config.freq for _ in range(context_cpu.shape[0])]

        mean_fc_obj, _ = self.model.forecast(inputs=inputs, freq=freq_list)
        pred_all = torch.as_tensor(
            mean_fc_obj, dtype=torch.float32, device=context.device
        )

        if pred_all.ndim != 2:
            raise ValueError(
                f"TimesFM forecast 출력 shape이 예상과 다릅니다: {tuple(pred_all.shape)}"
            )
        if prediction_length > pred_all.shape[-1]:
            raise ValueError(
                f"요청 예측 길이({prediction_length})가 모델 출력 길이({pred_all.shape[-1]})보다 깁니다."
            )
        return pred_all[:, :prediction_length]

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """TimesFM 학습/평가용 forward 패스.

        Args:
            context: 입력 컨텍스트, shape ``(batch, context_len)`` 또는 ``(context_len,)``.
            target: 예측 타깃, shape ``(batch, prediction_len)`` 또는 ``(prediction_len,)``.

        Returns:
            ``{"loss": loss, "pred": pred}`` 딕셔너리.

        Raises:
            ValueError: 모델이 로드되지 않았거나 입력 shape이 유효하지 않을 때.
        """

        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")

        context_batch = self._prepare_context_batch(context)
        target_batch = self._prepare_target_batch(target)
        if context_batch.shape[0] != target_batch.shape[0]:
            raise ValueError(
                f"배치 크기가 다릅니다: context={context_batch.shape[0]}, target={target_batch.shape[0]}"
            )

        try:
            return self._forward_with_backbone(
                context_batch=context_batch,
                target_batch=target_batch,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "TimesFM 내부 백본 forward 경로를 사용할 수 없어 forecast fallback을 사용합니다: %s",
                exc,
            )
            return self._forward_with_forecast(
                context_batch=context_batch,
                target_batch=target_batch,
            )

    def __call__(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """forward() 로 위임하여 인스턴스 호출 지원.

        Args:
            context: 입력 컨텍스트.
            target: 예측 타깃.

        Returns:
            ``{"loss": loss, "pred": pred}`` 딕셔너리.
        """

        return self.forward(context=context, target=target)

    def eval(self) -> TimesFMWrapper:
        """모델을 평가 모드로 전환.

        Args:
            None.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        _ = self.get_backbone().eval()
        model_obj = cast(object, self.model)
        if isinstance(model_obj, nn.Module):
            _ = model_obj.eval()
        return self

    def train(self, mode: bool = True) -> TimesFMWrapper:
        """모델을 학습 모드로 전환.

        Args:
            mode: True이면 학습 모드, False이면 평가 모드.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        _ = self.get_backbone().train(mode)
        model_obj = cast(object, self.model)
        if isinstance(model_obj, nn.Module):
            _ = model_obj.train(mode)
        return self

    def to(self, device: torch.device | str) -> TimesFMWrapper:
        """모델을 지정된 디바이스로 이동.

        Args:
            device: 대상 디바이스.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        _ = self.get_backbone().to(device)
        model_obj = cast(object, self.model)
        if isinstance(model_obj, nn.Module):
            _ = model_obj.to(device)
        # Sync TimesFM internal _device so forecast() creates tensors on the right device
        if self.model is not None and hasattr(self.model, "_device"):
            self.model._device = torch.device(device)  # type: ignore[union-attr]
        return self

    def parameters(self) -> object:
        """모델 전체 파라미터 반환.

        Args:
            None.

        Returns:
            모델 파라미터 이터레이터.

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        return self.get_backbone().parameters()

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """현재 학습 가능한 파라미터 목록 반환.

        Args:
            None.

        Returns:
            ``requires_grad=True`` 파라미터 리스트.

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        return [p for p in self.get_backbone().parameters() if p.requires_grad]
