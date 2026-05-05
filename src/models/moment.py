from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DictConfig(Protocol):
    def __getattr__(self, name: str) -> object: ...


class _MomentOutputProtocol(Protocol):
    forecast: torch.Tensor


class _MomentPipelineInstanceProtocol(Protocol):
    encoder: nn.Module

    def init(self) -> None: ...

    def __call__(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> _MomentOutputProtocol: ...

    def parameters(self) -> object: ...


class _MomentPipelineClassProtocol(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> _MomentPipelineInstanceProtocol: ...


class _MomentModuleProtocol(Protocol):
    MOMENTPipeline: _MomentPipelineClassProtocol


@dataclass
class MOMENTWrapperConfig:
    """MOMENT 래퍼 설정.

    Args:
        hf_id: HuggingFace 모델 ID.
        context_length: 입력 컨텍스트 길이.
        prediction_length: 예측 길이.
        freeze_encoder: 인코더 고정 여부.
        freeze_embedder: 임베더 고정 여부.
        freeze_head: 헤드 고정 여부.
        head_dropout: 예측 헤드 드롭아웃.

    Returns:
        MOMENTWrapperConfig 인스턴스.

    Raises:
        ValueError: 길이 파라미터가 1 미만일 때.
    """

    hf_id: str
    context_length: int = 512
    prediction_length: int = 96
    freeze_encoder: bool = False
    freeze_embedder: bool = True
    freeze_head: bool = False
    head_dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.context_length < 1:
            raise ValueError(
                f"context_length는 1 이상이어야 합니다. 현재: {self.context_length}"
            )
        if self.prediction_length < 1:
            raise ValueError(
                f"prediction_length는 1 이상이어야 합니다. 현재: {self.prediction_length}"
            )

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> MOMENTWrapperConfig:
        """Hydra DictConfig를 MOMENT 래퍼 설정으로 변환.

        Args:
            config: 모델 설정 DictConfig.

        Returns:
            MOMENTWrapperConfig 인스턴스.

        Raises:
            ValueError: ``hf_id``가 없을 때.
        """

        hf_id = getattr(config, "hf_id", None)
        if not isinstance(hf_id, str) or hf_id == "":
            raise ValueError("config.hf_id는 비어 있지 않은 문자열이어야 합니다.")

        return cls(
            hf_id=hf_id,
            context_length=int(cast(int, getattr(config, "context_length", 512))),
            prediction_length=int(cast(int, getattr(config, "prediction_length", 96))),
            freeze_encoder=bool(cast(bool, getattr(config, "freeze_encoder", False))),
            freeze_embedder=bool(cast(bool, getattr(config, "freeze_embedder", True))),
            freeze_head=bool(cast(bool, getattr(config, "freeze_head", False))),
            head_dropout=float(cast(float, getattr(config, "head_dropout", 0.1))),
        )


class MOMENTWrapper:
    """MOMENT 시계열 기반모델 래퍼.

    Args:
        config: Hydra 모델 설정.

    Returns:
        MOMENTWrapper 인스턴스.

    Raises:
        ValueError: 설정 값이 유효하지 않을 때.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config: MOMENTWrapperConfig = MOMENTWrapperConfig.from_dict_config(config)
        self.model: _MomentPipelineInstanceProtocol | None = None
        self.backbone: nn.Module | None = None

    def load(self) -> None:
        """HuggingFace에서 MOMENT 파이프라인과 백본 로드.

        Args:
            None.

        Returns:
            None.

        Raises:
            ImportError: ``momentfm`` 패키지를 찾을 수 없을 때.
            ValueError: 백본 추출에 실패했을 때.
        """

        try:
            moment_module = cast(
                _MomentModuleProtocol,
                cast(object, import_module("momentfm")),
            )
        except ImportError as exc:
            raise ImportError(
                "momentfm 패키지를 찾을 수 없습니다. `pip install git+https://github.com/moment-timeseries-foundation-model/moment.git`를 확인하세요."
            ) from exc

        pipeline_cls = moment_module.MOMENTPipeline
        model_kwargs: dict[str, object] = {
            "task_name": "forecasting",
            "forecast_horizon": self.config.prediction_length,
            "freeze_encoder": self.config.freeze_encoder,
            "freeze_embedder": self.config.freeze_embedder,
            "freeze_head": self.config.freeze_head,
            "head_dropout": self.config.head_dropout,
        }
        self.model = pipeline_cls.from_pretrained(
            self.config.hf_id,
            model_kwargs=model_kwargs,
        )

        init_fn = getattr(self.model, "init", None)
        if callable(init_fn):
            _ = init_fn()

        self.backbone = self.model.encoder

        logger.info("MOMENT 모델 로드 완료: %s", self.config.hf_id)

    def get_backbone(self) -> nn.Module:
        """PEFT 적용 대상 백본 모듈 반환.

        Args:
            None.

        Returns:
            MOMENT 인코더 모듈.

        Raises:
            ValueError: 모델이 아직 로드되지 않았을 때.
        """

        if self.backbone is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")
        return self.backbone

    def _prepare_context(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MOMENT 입력 포맷으로 컨텍스트를 변환.

        Args:
            context: shape ``(batch, context_len)`` 또는 ``(context_len,)``.

        Returns:
            ``(x_enc, input_mask)`` 튜플.

        Raises:
            ValueError: context 차원이 유효하지 않을 때.
        """

        if context.ndim == 1:
            context_batch = context.unsqueeze(0)
        elif context.ndim == 2:
            context_batch = context
        else:
            raise ValueError(
                f"context는 1D 또는 2D여야 합니다. 현재 shape: {context.shape}"
            )

        x_enc = context_batch.to(dtype=torch.float32).unsqueeze(1)
        input_mask = torch.ones(
            (context_batch.shape[0], context_batch.shape[1]),
            dtype=torch.long,
            device=context_batch.device,
        )
        return x_enc, input_mask

    def _extract_forecast(self, output: _MomentOutputProtocol) -> torch.Tensor:
        """MOMENT 출력에서 예측 텐서 추출.

        Args:
            output: MOMENT 출력 객체.

        Returns:
            shape ``(batch, horizon)`` 예측 텐서.

        Raises:
            ValueError: forecast 텐서 shape이 유효하지 않을 때.
        """

        forecast = output.forecast
        if forecast.ndim == 3 and forecast.shape[1] == 1:
            return forecast[:, 0, :]
        if forecast.ndim == 2:
            return forecast
        raise ValueError(f"MOMENT forecast shape이 예상과 다릅니다: {forecast.shape}")

    def predict(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor:
        """MOMENT zero-shot 점 예측 수행.

        Args:
            context: 입력 컨텍스트, shape ``(batch, context_len)`` 또는 ``(context_len,)``.
            prediction_length: 요청 예측 길이.

        Returns:
            shape ``(batch, prediction_length)`` 점예측 텐서.

        Raises:
            ValueError: 모델이 로드되지 않았거나 요청 길이가 지원 범위를 벗어날 때.
        """

        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")

        x_enc, input_mask = self._prepare_context(context)
        output = self.model(x_enc=x_enc, input_mask=input_mask)
        forecast = self._extract_forecast(output)

        if prediction_length > forecast.shape[-1]:
            raise ValueError(
                f"요청 예측 길이({prediction_length})가 모델 출력 길이({forecast.shape[-1]})보다 깁니다."
            )
        return forecast[:, :prediction_length].to(
            device=context.device, dtype=torch.float32
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

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """MOMENT 학습용 forward 패스.

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

        if target.ndim == 1:
            target_batch = target.unsqueeze(0)
        elif target.ndim == 2:
            target_batch = target
        else:
            raise ValueError(
                f"target은 1D 또는 2D여야 합니다. 현재 shape: {target.shape}"
            )

        x_enc, input_mask = self._prepare_context(context)
        if x_enc.shape[0] != target_batch.shape[0]:
            raise ValueError(
                f"배치 크기가 다릅니다: context={x_enc.shape[0]}, target={target_batch.shape[0]}"
            )

        output = self.model(x_enc=x_enc, input_mask=input_mask)
        forecast = self._extract_forecast(output)

        horizon = target_batch.shape[-1]
        if horizon > forecast.shape[-1]:
            raise ValueError(
                f"target 길이({horizon})가 모델 출력 길이({forecast.shape[-1]})보다 깁니다."
            )

        pred = forecast[:, :horizon]
        target_for_loss = target_batch.to(device=pred.device, dtype=pred.dtype)
        loss = F.mse_loss(pred, target_for_loss)

        pred = pred.to(device=target.device, dtype=target.dtype)
        return {"loss": loss, "pred": pred}

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """현재 학습 가능한 파라미터 목록 반환.

        Args:
            None.

        Returns:
            ``requires_grad=True`` 파라미터 리스트.

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")
        model_obj = cast(nn.Module, cast(object, self.model))
        return [p for p in model_obj.parameters() if p.requires_grad]

    def eval(self) -> MOMENTWrapper:
        """모델을 평가 모드로 전환.

        Args:
            None.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        self.get_backbone().eval()
        if self.model is not None:
            model_module = cast(nn.Module, cast(object, self.model))
            model_module.eval()
        return self

    def train(self, mode: bool = True) -> MOMENTWrapper:
        """모델을 학습 모드로 전환.

        Args:
            mode: True이면 학습 모드, False이면 평가 모드.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        self.get_backbone().train(mode)
        if self.model is not None:
            model_module = cast(nn.Module, cast(object, self.model))
            model_module.train(mode)
        return self

    def to(self, device: torch.device | str) -> MOMENTWrapper:
        """모델을 지정된 디바이스로 이동.

        Args:
            device: 대상 디바이스.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        self.get_backbone().to(device)
        if self.model is not None:
            model_module = cast(nn.Module, cast(object, self.model))
            model_module.to(device)
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

        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")
        model_obj = cast(nn.Module, cast(object, self.model))
        return model_obj.parameters()
