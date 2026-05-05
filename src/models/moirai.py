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


class _MoiraiOutputProtocol(Protocol):
    samples: torch.Tensor


class _MoiraiModuleInstanceProtocol(Protocol):
    def parameters(self) -> object: ...


class _MoiraiModuleClassProtocol(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> _MoiraiModuleInstanceProtocol: ...


class _MoiraiForecastInstanceProtocol(Protocol):
    def predict(self, **kwargs: object) -> torch.Tensor | _MoiraiOutputProtocol: ...

    def __call__(self, **kwargs: object) -> torch.Tensor | _MoiraiOutputProtocol: ...

    def parameters(self) -> object: ...


class _MoiraiForecastFactoryProtocol(Protocol):
    def __call__(
        self, *args: object, **kwargs: object
    ) -> _MoiraiForecastInstanceProtocol: ...


class _MoiraiModuleProtocol(Protocol):
    MoiraiForecast: _MoiraiForecastFactoryProtocol
    MoiraiModule: _MoiraiModuleClassProtocol


@dataclass
class MoiraiWrapperConfig:
    """Moirai 래퍼 설정.

    Args:
        hf_id: HuggingFace 모델 ID.
        context_length: 입력 컨텍스트 길이.
        prediction_length: 예측 길이.
        patch_size: 패치 크기 또는 ``"auto"``.
        num_samples: 샘플링 개수.

    Returns:
        MoiraiWrapperConfig 인스턴스.

    Raises:
        ValueError: 길이/샘플링 설정이 1 미만이거나 ``hf_id``가 비었을 때.
    """

    hf_id: str = "Salesforce/moirai-1.1-R-base"
    context_length: int = 512
    prediction_length: int = 96
    patch_size: int | str = "auto"
    num_samples: int = 100

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
        if self.num_samples < 1:
            raise ValueError(
                f"num_samples는 1 이상이어야 합니다. 현재: {self.num_samples}"
            )
        if isinstance(self.patch_size, int) and self.patch_size < 1:
            raise ValueError(
                f"patch_size가 정수라면 1 이상이어야 합니다. 현재: {self.patch_size}"
            )
        if isinstance(self.patch_size, str) and self.patch_size == "":
            raise ValueError("patch_size가 문자열이라면 비어 있지 않아야 합니다.")

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> MoiraiWrapperConfig:
        """Hydra DictConfig를 Moirai 래퍼 설정으로 변환.

        Args:
            config: 모델 설정 DictConfig.

        Returns:
            MoiraiWrapperConfig 인스턴스.

        Raises:
            ValueError: ``hf_id``가 문자열이 아닐 때.
        """

        hf_id_obj = getattr(config, "hf_id", "Salesforce/moirai-1.1-R-base")
        if not isinstance(hf_id_obj, str) or hf_id_obj == "":
            raise ValueError("config.hf_id는 비어 있지 않은 문자열이어야 합니다.")

        patch_size_obj = getattr(config, "patch_size", "auto")
        if not isinstance(patch_size_obj, (int, str)):
            raise ValueError("config.patch_size는 int 또는 str 이어야 합니다.")

        return cls(
            hf_id=hf_id_obj,
            context_length=int(cast(int, getattr(config, "context_length", 512))),
            prediction_length=int(cast(int, getattr(config, "prediction_length", 96))),
            patch_size=patch_size_obj,
            num_samples=int(cast(int, getattr(config, "num_samples", 100))),
        )


class MoiraiWrapper:
    """Moirai 기반모델 래퍼.

    Args:
        config: Hydra 모델 설정.

    Returns:
        MoiraiWrapper 인스턴스.

    Raises:
        ValueError: 설정 값이 유효하지 않을 때.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config: MoiraiWrapperConfig = MoiraiWrapperConfig.from_dict_config(config)
        self.module: _MoiraiModuleInstanceProtocol | None = None
        self.forecast_model: _MoiraiForecastInstanceProtocol | None = None

    def _build_forecast_model(
        self,
        forecast_factory: _MoiraiForecastFactoryProtocol,
        module_obj: _MoiraiModuleInstanceProtocol,
    ) -> _MoiraiForecastInstanceProtocol:
        """MoiraiForecast 인스턴스를 다양한 시그니처로 생성.

        Args:
            forecast_factory: ``MoiraiForecast`` 생성자.
            module_obj: ``MoiraiModule.from_pretrained`` 결과 객체.

        Returns:
            구성된 ``MoiraiForecast`` 인스턴스.

        Raises:
            ValueError: 지원되는 시그니처로 생성에 실패했을 때.
        """

        full_kwargs: dict[str, object] = {
            "prediction_length": self.config.prediction_length,
            "context_length": self.config.context_length,
            "target_dim": 1,
            "feat_dynamic_real_dim": 0,
            "past_feat_dynamic_real_dim": 0,
            "patch_size": self.config.patch_size,
            "num_samples": self.config.num_samples,
            "module": module_obj,
        }
        attempts: list[tuple[tuple[object, ...], dict[str, object]]] = [
            ((), full_kwargs),
            ((), {k: v for k, v in full_kwargs.items() if k != "module"}  
             | {"module": module_obj}),
            (
                (),
                {
                    "module": module_obj,
                    "prediction_length": self.config.prediction_length,
                    "context_length": self.config.context_length,
                    "target_dim": 1,
                    "feat_dynamic_real_dim": 0,
                    "past_feat_dynamic_real_dim": 0,
                },
            ),
            ((module_obj,), {
                "prediction_length": self.config.prediction_length,
                "context_length": self.config.context_length,
                "target_dim": 1,
                "feat_dynamic_real_dim": 0,
                "past_feat_dynamic_real_dim": 0,
                "patch_size": self.config.patch_size,
                "num_samples": self.config.num_samples,
            }),
        ]
        for args, kwargs in attempts:
            try:
                return forecast_factory(*args, **kwargs)
            except TypeError:
                continue

        raise ValueError(
            "MoiraiForecast를 생성할 수 없습니다. uni2ts 버전 호환성을 확인하세요."
        )

    def load(self) -> None:
        """HuggingFace에서 Moirai 모듈과 예측 래퍼 로드.

        Args:
            None.

        Returns:
            None.

        Raises:
            ImportError: ``uni2ts`` 패키지를 찾을 수 없을 때.
        """

        try:
            moirai_module = cast(
                _MoiraiModuleProtocol,
                cast(object, import_module("uni2ts.model.moirai")),
            )
        except ImportError as exc:
            raise ImportError(
                "uni2ts 패키지를 찾을 수 없습니다. `pip install uni2ts`를 확인하세요."
            ) from exc

        module_cls = moirai_module.MoiraiModule
        self.module = module_cls.from_pretrained(self.config.hf_id)
        self.forecast_model = self._build_forecast_model(
            forecast_factory=moirai_module.MoiraiForecast,
            module_obj=self.module,
        )

        logger.info("Moirai 모델 로드 완료: %s", self.config.hf_id)

    def get_backbone(self) -> nn.Module:
        """PEFT 적용 대상 백본 모듈 반환.

        Args:
            None.

        Returns:
            MoiraiModule 백본 모듈.

        Raises:
            ValueError: 모델이 아직 로드되지 않았을 때.
        """

        if self.module is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")
        return cast(nn.Module, cast(object, self.module))

    def _prepare_inputs(
        self,
        context: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Moirai 입력 포맷으로 context (및 target)를 변환.

        ``patch_size="auto"`` 인 경우 ``past_target`` 은
        ``(batch, context_length + prediction_length, 1)`` 형태여야 합니다.
        target 이 없으면 prediction_length 만큼 0 으로 패딩합니다.

        Args:
            context: shape ``(batch, context_len)`` 또는 ``(context_len,)``.
            target: (선택) shape ``(batch, pred_len)`` 또는 ``(pred_len,)``.

        Returns:
            ``(past_target, past_observed_target, past_is_pad)`` 튜플.

        Raises:
            ValueError: context 차원이 유효하지 않을 때.
        """

        if context.ndim == 1:
            ctx = context.unsqueeze(0)
        elif context.ndim == 2:
            ctx = context
        else:
            raise ValueError(
                f"context는 1D 또는 2D여야 합니다. 현재 shape: {context.shape}"
            )
        ctx = ctx.to(dtype=torch.float32)
        batch_size = ctx.shape[0]
        pred_len = self.config.prediction_length

        if target is not None:
            tgt = target.unsqueeze(0) if target.ndim == 1 else target
            tgt = tgt.to(dtype=torch.float32)
            full_seq = torch.cat([ctx, tgt], dim=1)  # (B, ctx+pred, )
            obs_len = ctx.shape[1] + tgt.shape[1]
        elif self.config.patch_size == "auto":
            # auto patch requires ctx+pred length; pad future with zeros
            pad = torch.zeros(
                batch_size, pred_len, dtype=ctx.dtype, device=ctx.device
            )
            full_seq = torch.cat([ctx, pad], dim=1)
            obs_len = ctx.shape[1]  # only context is observed
        else:
            full_seq = ctx
            obs_len = ctx.shape[1]

        past_target = full_seq.unsqueeze(-1)  # (B, T, 1)
        seq_len = full_seq.shape[1]

        # observed mask: 1 where real data, 0 where padding
        obs_mask = torch.zeros(
            batch_size, seq_len, 1, dtype=torch.bool, device=ctx.device
        )
        obs_mask[:, :obs_len, :] = True
        past_observed_target = obs_mask  # keep as bool

        past_is_pad = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=ctx.device
        )

        return past_target, past_observed_target, past_is_pad

    def _extract_samples(
        self,
        output: torch.Tensor | _MoiraiOutputProtocol,
    ) -> torch.Tensor:
        """Moirai 출력에서 샘플 텐서를 추출.

        Args:
            output: MoiraiForecast 출력.

        Returns:
            shape ``(batch, num_samples, horizon)`` 샘플 텐서.

        Raises:
            ValueError: 샘플 텐서 shape이 유효하지 않을 때.
        """

        samples = output if isinstance(output, torch.Tensor) else output.samples
        if samples.ndim == 4 and samples.shape[-1] == 1:
            return samples.squeeze(-1)
        if samples.ndim == 3:
            return samples
        raise ValueError(f"Moirai 샘플 출력 shape이 예상과 다릅니다: {samples.shape}")

    def _run_forecast(
        self,
        past_target: torch.Tensor,
        past_observed_target: torch.Tensor,
        past_is_pad: torch.Tensor,
    ) -> torch.Tensor:
        """MoiraiForecast를 호출해 샘플 출력 획득.

        Args:
            past_target: 과거 타깃 텐서, shape ``(B, T, 1)``.
            past_observed_target: 관측 마스크 텐서.
            past_is_pad: 패딩 마스크 텐서.

        Returns:
            shape ``(batch, num_samples, horizon)`` 샘플 텐서.

        Raises:
            ValueError: 모델이 로드되지 않았거나 호출 실패 시.
        """

        if self.forecast_model is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")

        kwargs: dict[str, object] = {
            "past_target": past_target,
            "past_observed_target": past_observed_target,
            "past_is_pad": past_is_pad,
            "num_samples": self.config.num_samples,
        }
        output = cast(
            torch.Tensor | _MoiraiOutputProtocol,
            self.forecast_model(**kwargs),
        )
        return self._extract_samples(output)

    def predict(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor:
        """Moirai zero-shot 점 예측 수행.

        Args:
            context: 입력 컨텍스트, shape ``(batch, context_len)`` 또는 ``(context_len,)``.
            prediction_length: 요청 예측 길이.

        Returns:
            shape ``(batch, prediction_length)`` 점예측 텐서.

        Raises:
            ValueError: 모델이 로드되지 않았거나 요청 길이가 지원 범위를 벗어날 때.
        """

        if prediction_length < 1:
            raise ValueError(
                f"prediction_length는 1 이상이어야 합니다. 현재: {prediction_length}"
            )

        past_target, past_obs, past_is_pad = self._prepare_inputs(context)
        samples = self._run_forecast(
            past_target=past_target,
            past_observed_target=past_obs,
            past_is_pad=past_is_pad,
        )
        pred = samples.median(dim=1).values
        if prediction_length > pred.shape[-1]:
            raise ValueError(
                f"요청 예측 길이({prediction_length})가 모델 출력 길이({pred.shape[-1]})보다 깁니다."
            )
        return pred[:, :prediction_length].to(
            device=context.device, dtype=torch.float32
        )

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Moirai 학습용 forward 패스.

        MoiraiForecast 내부 ``_val_loss`` 를 활용하여 미분 가능한 NLL 손실을
        계산합니다. 예측값은 샘플링 기반(non-differentiable)으로
        메트릭 계산용으로만 사용합니다.

        Args:
            context: 입력 컨텍스트, shape ``(batch, context_len)`` 또는 ``(context_len,)``.
            target: 예측 타깃, shape ``(batch, prediction_len)`` 또는 ``(prediction_len,)``.

        Returns:
            ``{"loss": loss, "pred": pred}`` 딕셔너리.

        Raises:
            ValueError: 모델이 로드되지 않았거나 입력 shape이 유효하지 않을 때.
        """

        if target.ndim == 1:
            target_batch = target.unsqueeze(0)
        elif target.ndim == 2:
            target_batch = target
        else:
            raise ValueError(
                f"target은 1D 또는 2D여야 합니다. 현재 shape: {target.shape}"
            )

        past_target, past_obs, past_is_pad = self._prepare_inputs(
            context, target_batch
        )

        # ---- 미분 가능 손실 경로 (NLL via _val_loss) ----
        forecast_mod = self.forecast_model
        _val_loss_fn = getattr(forecast_mod, "_val_loss", None)
        if callable(_val_loss_fn):
            try:
                # _val_loss expects past_length = ctx + pred
                patch_sizes = getattr(
                    cast(nn.Module, cast(object, self.module)), "patch_sizes", [32]
                )
                # Use first patch size for simplicity
                ps = int(patch_sizes[0])
                val_loss = _val_loss_fn(
                    patch_size=ps,
                    target=past_target,
                    observed_target=past_obs,
                    is_pad=past_is_pad,
                )
                loss = val_loss.mean()  # (batch,) -> scalar
            except (TypeError, RuntimeError, AttributeError):
                # Fallback to MSE
                loss = None
        else:
            loss = None

        # ---- 예측값 (non-differentiable, 메트릭용) ----
        with torch.no_grad():
            samples = self._run_forecast(
                past_target=past_target,
                past_observed_target=past_obs,
                past_is_pad=past_is_pad,
            )
            pred = samples.median(dim=1).values
            horizon = target_batch.shape[-1]
            pred = pred[:, :horizon] if horizon <= pred.shape[-1] else pred

        # Fallback MSE loss if _val_loss path failed
        if loss is None:
            # 미분 가능 경로: gradient 활성화 상태에서 재추론하여 MSE 계산
            try:
                diff_samples = self._run_forecast(
                    past_target=past_target,
                    past_observed_target=past_obs,
                    past_is_pad=past_is_pad,
                )
                diff_pred = diff_samples.mean(dim=1)
                horizon = target_batch.shape[-1]
                diff_pred = diff_pred[:, :horizon] if horizon <= diff_pred.shape[-1] else diff_pred
                target_for_loss = target_batch.to(
                    device=diff_pred.device, dtype=diff_pred.dtype
                )
                loss = F.mse_loss(diff_pred, target_for_loss)
            except RuntimeError:
                # 최후 수단: 모델 업데이트 불가 경고 후 zero loss 반환
                logger.warning(
                    "Moirai fallback loss: 미분 가능 경로 실패. "
                    "이 배치에서 모델 파라미터가 업데이트되지 않습니다."
                )
                loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred = pred.to(device=target.device, dtype=target.dtype)
        return {"loss": loss, "pred": pred}

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

    def eval(self) -> MoiraiWrapper:
        """모델을 평가 모드로 전환.

        Args:
            None.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        _ = self.get_backbone().eval()
        if self.forecast_model is not None:
            forecast_module = cast(nn.Module, cast(object, self.forecast_model))
            _ = forecast_module.eval()
        return self

    def train(self, mode: bool = True) -> MoiraiWrapper:
        """모델을 학습 모드로 전환.

        Args:
            mode: True이면 학습 모드, False이면 평가 모드.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        _ = self.get_backbone().train(mode)
        if self.forecast_model is not None:
            forecast_module = cast(nn.Module, cast(object, self.forecast_model))
            _ = forecast_module.train(mode)
        return self

    def to(self, device: torch.device | str) -> MoiraiWrapper:
        """모델을 지정된 디바이스로 이동.

        Args:
            device: 대상 디바이스.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        _ = self.get_backbone().to(device)
        if self.forecast_model is not None:
            forecast_module = cast(nn.Module, cast(object, self.forecast_model))
            _ = forecast_module.to(device)
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
