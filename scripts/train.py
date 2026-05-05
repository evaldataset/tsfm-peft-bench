from __future__ import annotations

# pyright: reportMissingImports=false, reportUnusedImport=false, reportCallIssue=false

# ============================================================================
# DEV/DEBUG ENTRY POINT — 단일 (model, adaptation, data) 조합용.
# 논문의 모든 실험 결과는 ``scripts/run_expansion.py``로 생성됩니다.
# 이 스크립트는 ETT 도메인 + Chronos/MOMENT 모델만 검증돼 있습니다.
# 다른 도메인/모델 조합은 ``run_expansion.py`` 또는 직접 wrapper API를 사용하세요.
# ============================================================================

import logging
import math
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.adaptation.head import apply_head_only
from src.adaptation.lora import LoRAAdaptationConfig, LoRALocus, apply_lora
from src.adaptation.prefix import PrefixAdaptationConfig, apply_prefix_tuning
from src.data.ett import ETTConfig, load_ett
from src.evaluation.metrics import compute_metrics
from src.models.chronos import ChronosWrapper
from src.models.moment import MOMENTWrapper
from src.utils.device import get_device, log_gpu_memory
from src.utils.logging import finish_wandb, init_wandb, log_metrics
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """로깅 기본 설정을 초기화.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """샘플 배치를 텐서로 스택하여 반환.

    Args:
        batch: ``context``/``target`` 키를 갖는 샘플 리스트.

    Returns:
        배치 텐서 딕셔너리.

    Raises:
        ValueError: 빈 배치가 들어왔을 때.
    """

    if len(batch) == 0:
        raise ValueError("빈 배치는 허용되지 않습니다.")

    context = torch.stack(
        [sample["context"].to(torch.float32) for sample in batch], dim=0
    )
    target = torch.stack(
        [sample["target"].to(torch.float32) for sample in batch], dim=0
    )
    return {"context": context, "target": target}


def _build_ett_dataloaders(
    cfg: DictConfig,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    """ETT 데이터셋을 로드하고 DataLoader를 생성.

    Args:
        cfg: Hydra 전체 설정.

    Returns:
        ``(train_loader, val_loader, test_loader)`` 튜플.

    Raises:
        ValueError: 데이터셋 설정이 ETT 계열이 아닐 때.
    """

    if not str(cfg.data.name).startswith("ett_"):
        raise ValueError(
            f"현재 스크립트는 ETT 데이터셋만 지원합니다. 현재: {cfg.data.name}"
        )

    data_cfg = ETTConfig(
        dataset=str(cfg.data.dataset),
        path=str(cfg.data.path),
        target_col=str(cfg.data.target_col),
        context_length=int(cfg.data.context_length),
        prediction_length=int(cfg.data.prediction_length),
        train_ratio=float(cfg.data.train_ratio),
        val_ratio=float(cfg.data.val_ratio),
        test_ratio=float(cfg.data.test_ratio),
    )
    train_ds, val_ds, test_ds = load_ett(data_cfg)

    batch_size = int(cfg.training.batch_size)
    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    test_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    return train_loader, val_loader, test_loader


def _build_wrapper(cfg: DictConfig) -> ChronosWrapper | MOMENTWrapper:
    """모델 이름에 따라 래퍼를 생성하고 로드.

    Args:
        cfg: Hydra 전체 설정.

    Returns:
        로드 완료된 모델 래퍼.

    Raises:
        ValueError: 지원하지 않는 모델 이름일 때.
    """

    model_name = str(cfg.model.name)
    if model_name == "chronos":
        wrapper: ChronosWrapper | MOMENTWrapper = ChronosWrapper(cfg.model)
    elif model_name == "moment":
        wrapper = MOMENTWrapper(cfg.model)
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    wrapper.load()
    return wrapper


def _apply_full_fine_tuning(backbone: torch.nn.Module) -> None:
    """백본의 모든 파라미터를 학습 가능으로 설정.

    Args:
        backbone: 학습 대상 백본 모듈.

    Returns:
        None.

    Raises:
        None.
    """

    for parameter in backbone.parameters():
        parameter.requires_grad = True


def _apply_adaptation(cfg: DictConfig, wrapper: ChronosWrapper | MOMENTWrapper) -> None:
    """설정에 맞춰 적응 기법을 적용.

    Args:
        cfg: Hydra 전체 설정.
        wrapper: 로드된 모델 래퍼.

    Returns:
        None.

    Raises:
        ValueError: 지원하지 않는 적응 기법일 때.
    """

    method = str(cfg.adaptation.method)
    backbone = wrapper.get_backbone()

    if method == "zero_shot":
        logger.info("zero_shot 모드: 학습을 건너뜁니다.")
        return

    if method == "head_only":
        _ = apply_head_only(backbone)
        return

    if method == "lora":
        locus = LoRALocus(str(cfg.adaptation.locus))
        target_modules: list[str] | None = None
        if cfg.adaptation.get("target_modules") is not None:
            target_modules = list(cfg.adaptation.target_modules)

        lora_cfg = LoRAAdaptationConfig(
            rank=int(cfg.adaptation.rank),
            alpha=int(cfg.adaptation.alpha),
            dropout=float(cfg.adaptation.dropout),
            locus=locus,
            target_modules=target_modules,
            task_type=str(cfg.adaptation.get("task_type", "SEQ_2_SEQ_LM")),
            layers=str(cfg.adaptation.get("layers", "all")),
            num_layers=int(cfg.model.get("num_layers", 12)),
        )
        adapted = apply_lora(backbone, lora_cfg)
        wrapper.backbone = adapted
        if hasattr(wrapper, "model") and getattr(wrapper, "model", None) is not None:
            model_obj = getattr(wrapper, "model")
            if hasattr(model_obj, "encoder"):
                setattr(model_obj, "encoder", adapted)
        return

    if method == "prefix_tuning":
        prefix_cfg = PrefixAdaptationConfig(
            num_virtual_tokens=int(cfg.adaptation.num_virtual_tokens),
            task_type=str(cfg.adaptation.get("task_type", "SEQ_2_SEQ_LM")),
        )
        adapted = apply_prefix_tuning(backbone, prefix_cfg)
        wrapper.backbone = adapted
        if hasattr(wrapper, "model") and getattr(wrapper, "model", None) is not None:
            model_obj = getattr(wrapper, "model")
            if hasattr(model_obj, "encoder"):
                setattr(model_obj, "encoder", adapted)
        return

    if method == "full_fine_tuning":
        _apply_full_fine_tuning(backbone)
        return

    raise ValueError(f"지원하지 않는 adaptation.method 입니다: {method}")


def _move_to_device(
    wrapper: ChronosWrapper | MOMENTWrapper, device: torch.device
) -> None:
    """백본 모듈을 지정 디바이스로 이동.

    Args:
        wrapper: 모델 래퍼.
        device: 대상 디바이스.

    Returns:
        None.

    Raises:
        None.
    """

    backbone = wrapper.get_backbone()
    _ = backbone.to(device)


def _build_optimizer_scheduler(
    cfg: DictConfig,
    wrapper: ChronosWrapper | MOMENTWrapper,
    steps_per_epoch: int,
) -> tuple[AdamW, LambdaLR, int]:
    """AdamW와 선형 warmup 스케줄러를 생성.

    Args:
        cfg: Hydra 전체 설정.
        wrapper: 모델 래퍼.
        steps_per_epoch: 에폭당 미니배치 수.

    Returns:
        ``(optimizer, scheduler, total_update_steps)`` 튜플.

    Raises:
        ValueError: 학습 가능한 파라미터가 없을 때.
    """

    trainable_parameters = wrapper.get_trainable_parameters()
    if len(trainable_parameters) == 0:
        raise ValueError(
            "학습 가능한 파라미터가 없습니다. adaptation 설정을 확인하세요."
        )

    optimizer = AdamW(
        trainable_parameters,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    accumulation_steps = max(1, int(cfg.training.gradient_accumulation_steps))
    total_update_steps = math.ceil(
        (steps_per_epoch * int(cfg.training.epochs)) / accumulation_steps
    )
    warmup_steps = int(cfg.training.warmup_steps)

    def lr_lambda(current_step: int) -> float:
        """현재 스텝의 학습률 배율을 계산.

        Args:
            current_step: 현재 업데이트 스텝.

        Returns:
            학습률 배율 값.

        Raises:
            None.
        """

        if total_update_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        remain_steps = max(1, total_update_steps - warmup_steps)
        remain_ratio = (total_update_steps - current_step) / float(remain_steps)
        return max(0.0, remain_ratio)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler, total_update_steps


def _predict_batch(
    wrapper: ChronosWrapper | MOMENTWrapper,
    context: torch.Tensor,
    target: torch.Tensor,
    method: str,
    fp16: bool,
    prediction_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """배치 단위 예측과 손실 텐서를 계산.

    Args:
        wrapper: 모델 래퍼.
        context: 입력 컨텍스트 배치.
        target: 타깃 배치.
        method: 적응 방식.
        fp16: mixed precision 사용 여부.
        prediction_length: 예측 길이.

    Returns:
        ``(pred, loss)`` 튜플.

    Raises:
        RuntimeError: forward 결과에 loss 키가 없을 때.
    """

    if method == "zero_shot":
        pred = wrapper.predict(context=context, prediction_length=prediction_length)
        loss = torch.mean((pred - target) ** 2)
        return pred, loss

    autocast_enabled = fp16 and context.is_cuda
    with torch.cuda.amp.autocast(enabled=autocast_enabled):
        outputs = wrapper(context=context, target=target)

    if "loss" not in outputs:
        raise RuntimeError("모델 forward 결과에 loss 키가 없습니다.")

    pred = outputs["pred"]
    loss = outputs["loss"]
    return pred, loss


def _evaluate(
    wrapper: ChronosWrapper | MOMENTWrapper,
    data_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    method: str,
    fp16: bool,
    prediction_length: int,
) -> tuple[float, dict[str, float]]:
    """검증/테스트 로더에서 손실과 메트릭을 계산.

    Args:
        wrapper: 모델 래퍼.
        data_loader: 평가용 DataLoader.
        device: 실행 디바이스.
        method: 적응 방식.
        fp16: mixed precision 사용 여부.
        prediction_length: 예측 길이.

    Returns:
        ``(mean_loss, metrics)`` 튜플.

    Raises:
        ValueError: 평가 로더가 비어 있을 때.
    """

    wrapper.eval()
    if len(data_loader) == 0:
        raise ValueError("평가 DataLoader가 비어 있습니다.")

    losses: list[float] = []
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)

            pred, loss = _predict_batch(
                wrapper=wrapper,
                context=context,
                target=target,
                method=method,
                fp16=fp16,
                prediction_length=prediction_length,
            )
            losses.append(float(loss.detach().item()))
            all_preds.append(pred.detach().to("cpu"))
            all_targets.append(target.detach().to("cpu"))

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(pred=preds, target=targets)
    mean_loss = float(sum(losses) / len(losses))
    return mean_loss, metrics


def _save_checkpoint(
    checkpoint_path: Path,
    cfg: DictConfig,
    wrapper: ChronosWrapper | MOMENTWrapper,
    best_val_loss: float,
    test_metrics: dict[str, float],
) -> None:
    """학습 결과 체크포인트를 저장.

    Args:
        checkpoint_path: 저장 파일 경로.
        cfg: Hydra 전체 설정.
        wrapper: 모델 래퍼.
        best_val_loss: 최적 검증 손실.
        test_metrics: 테스트 메트릭.

    Returns:
        None.

    Raises:
        OSError: 파일 저장 실패 시.
    """

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": str(cfg.model.name),
        "adaptation_method": str(cfg.adaptation.method),
        "adaptation_config": OmegaConf.to_container(cfg.adaptation, resolve=True),
        "data_name": str(cfg.data.name),
        "context_length": int(cfg.data.context_length),
        "prediction_length": int(cfg.data.prediction_length),
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "backbone_state_dict": wrapper.get_backbone().state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info("체크포인트 저장 완료: %s", checkpoint_path)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra 기반 시계열 적응 학습 엔트리포인트.

    Args:
        cfg: Hydra 실험 설정.

    Returns:
        None.

    Raises:
        Exception: 학습 과정에서 복구 불가능한 오류 발생 시.
    """

    _setup_logging()
    seed_everything(int(cfg.seed))
    device = get_device()

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Hydra 설정을 dict로 변환하지 못했습니다.")

    init_wandb(
        project=str(cfg.logging.wandb_project),
        entity=(
            None if cfg.logging.wandb_entity is None else str(cfg.logging.wandb_entity)
        ),
        config=cfg_dict,
        name=f"{cfg.model.name}_{cfg.adaptation.name}_{cfg.data.name}",
        tags=[str(cfg.model.name), str(cfg.adaptation.name), str(cfg.data.name)],
    )

    output_dir = Path(hydra.utils.get_original_cwd()) / str(cfg.output_dir)
    checkpoint_path = output_dir / "best.pt"

    try:
        train_loader, val_loader, test_loader = _build_ett_dataloaders(cfg)
        wrapper = _build_wrapper(cfg)
        _apply_adaptation(cfg, wrapper)
        _move_to_device(wrapper, device)

        method = str(cfg.adaptation.method)
        fp16 = bool(cfg.training.fp16)
        prediction_length = int(cfg.data.prediction_length)

        best_val_loss = float("inf")
        best_state_dict: dict[str, torch.Tensor] | None = None
        global_step = 0

        if method != "zero_shot":
            optimizer, scheduler, _ = _build_optimizer_scheduler(
                cfg=cfg,
                wrapper=wrapper,
                steps_per_epoch=len(train_loader),
            )
            scaler = torch.cuda.amp.GradScaler(enabled=fp16 and device.type == "cuda")
            accumulation_steps = max(1, int(cfg.training.gradient_accumulation_steps))
            patience = int(cfg.training.early_stopping_patience)
            not_improved = 0

            for epoch in range(int(cfg.training.epochs)):
                wrapper.train()
                optimizer.zero_grad(set_to_none=True)
                train_loss_sum = 0.0

                for batch_index, batch in enumerate(train_loader, start=1):
                    context = batch["context"].to(device)
                    target = batch["target"].to(device)

                    with torch.cuda.amp.autocast(
                        enabled=fp16 and device.type == "cuda"
                    ):
                        outputs = wrapper(context=context, target=target)
                        loss = outputs["loss"] / accumulation_steps

                    scaler.scale(loss).backward()
                    train_loss_sum += float(loss.detach().item() * accumulation_steps)

                    should_step = (
                        batch_index % accumulation_steps == 0
                        or batch_index == len(train_loader)
                    )
                    if should_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            wrapper.get_trainable_parameters(),
                            max_norm=float(cfg.training.max_grad_norm),
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        global_step += 1

                        if global_step % int(cfg.logging.log_every_n_steps) == 0:
                            current_lr = float(optimizer.param_groups[0]["lr"])
                            log_metrics(
                                {
                                    "loss": float(
                                        loss.detach().item() * accumulation_steps
                                    ),
                                    "lr": current_lr,
                                },
                                step=global_step,
                                prefix="train/",
                            )

                avg_train_loss = train_loss_sum / max(1, len(train_loader))
                val_loss, val_metrics = _evaluate(
                    wrapper=wrapper,
                    data_loader=val_loader,
                    device=device,
                    method=method,
                    fp16=fp16,
                    prediction_length=prediction_length,
                )
                log_metrics(
                    {"loss": avg_train_loss}, step=global_step, prefix="train_epoch/"
                )
                log_metrics(
                    {"loss": val_loss, **val_metrics}, step=global_step, prefix="val/"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in wrapper.get_backbone().state_dict().items()
                    }
                    not_improved = 0
                    logger.info(
                        "검증 개선: epoch=%d, val_loss=%.6f",
                        epoch + 1,
                        best_val_loss,
                    )
                else:
                    not_improved += 1
                    logger.info(
                        "검증 미개선: epoch=%d, patience=%d/%d",
                        epoch + 1,
                        not_improved,
                        patience,
                    )

                if not_improved >= patience:
                    logger.info("Early stopping 발동")
                    break

            if best_state_dict is not None:
                _ = wrapper.get_backbone().load_state_dict(
                    best_state_dict, strict=False
                )
        else:
            val_loss, val_metrics = _evaluate(
                wrapper=wrapper,
                data_loader=val_loader,
                device=device,
                method=method,
                fp16=False,
                prediction_length=prediction_length,
            )
            best_val_loss = val_loss
            log_metrics({"loss": val_loss, **val_metrics}, step=0, prefix="val/")

        test_loss, test_metrics = _evaluate(
            wrapper=wrapper,
            data_loader=test_loader,
            device=device,
            method=method,
            fp16=fp16,
            prediction_length=prediction_length,
        )
        log_metrics(
            {"loss": test_loss, **test_metrics}, step=global_step, prefix="test/"
        )

        gpu_metrics = log_gpu_memory(prefix="학습 종료 | ")
        if gpu_metrics:
            numeric_gpu_metrics = {
                key: float(value) for key, value in gpu_metrics.items()
            }
            log_metrics(numeric_gpu_metrics, step=global_step, prefix="system/")

        _save_checkpoint(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            wrapper=wrapper,
            best_val_loss=best_val_loss,
            test_metrics=test_metrics,
        )
    finally:
        finish_wandb()


if __name__ == "__main__":
    main()
