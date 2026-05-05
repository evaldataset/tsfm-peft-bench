"""Phase 1B 파일럿: Adaptation Locus 스윕 실험.

가설 B 검증 — 아키텍처 간 보편적 적응 로커스가 존재하는지 테스트.
2 models × 2 shifts × 6 loci × 2 seeds = 48 runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from src.adaptation.lora import LoRAAdaptationConfig, LoRALocus, apply_lora
from src.data.ett import ETTConfig, ETTDataset, load_ett
from src.data.shift import ShiftGenerator, ShiftSeverity, ShiftType
from src.data.shift_metrics import compute_shift_profile
from src.evaluation.metrics import compute_metrics
from src.models.chronos import ChronosWrapper
from src.models.moment import MOMENTWrapper
from src.models.moirai import MoiraiWrapper
from src.models.timesfm_wrapper import TimesFMWrapper
from src.utils.device import get_device, log_gpu_memory
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

# ─── 실험 행렬 상수 ──────────────────────────────────────────
DEFAULT_MODELS: list[str] = ["chronos", "moment"]
ALL_LOCI: list[LoRALocus] = [
    LoRALocus.ATTN_QV,
    LoRALocus.ATTN_ALL,
    LoRALocus.FFN,
    LoRALocus.ATTN_QV_FFN,
    LoRALocus.EARLY_LAYERS,
    LoRALocus.LATE_LAYERS,
]
SELECTED_SHIFTS: list[ShiftType] = [ShiftType.AMPLITUDE, ShiftType.SPECTRAL]
DEFAULT_SEVERITY: ShiftSeverity = ShiftSeverity.STRONG
DEFAULT_SEEDS: list[int] = [42, 123]
DEFAULT_LORA_RANK: int = 8
DEFAULT_LORA_ALPHA: int = 16
DEFAULT_LORA_DROPOUT: float = 0.05

# ─── 모델 설정 맵 ────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "chronos": {
        "name": "chronos",
        "hf_id": "amazon/chronos-t5-base",
        "architecture": "t5_encoder_decoder",
        "num_layers": 12,
        "hidden_size": 768,
        "context_length": 512,
        "prediction_length": 64,  # Chronos-T5-Base native limit
    },
    "moment": {
        "name": "moment",
        "hf_id": "AutonLab/MOMENT-1-large",
        "architecture": "t5_encoder",
        "num_layers": 24,
        "hidden_size": 1024,
        "context_length": 512,
        "prediction_length": 96,
        "freeze_encoder": False,
    },
    "moirai": {
        "name": "moirai",
        "hf_id": "Salesforce/moirai-1.1-R-base",
        "architecture": "moirai",
        "num_layers": 12,
        "hidden_size": 768,
        "context_length": 512,
        "prediction_length": 96,
        "patch_size": "auto",
        "num_samples": 20,
    },
    "timesfm": {
        "name": "timesfm",
        "hf_id": "google/timesfm-1.0-200m-pytorch",
        "architecture": "timesfm",
        "num_layers": 20,
        "hidden_size": 1280,
        "context_length": 512,
        "prediction_length": 96,
        "backend": "gpu",
        "freq": 0,
        "per_core_batch_size": 32,
    },
}


def _setup_logging() -> None:
    """로깅 기본 설정 초기화."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱.

    Returns:
        파싱된 argparse 네임스페이스.
    """
    parser = argparse.ArgumentParser(
        description="Phase 1B: Adaptation Locus 스윕 파일럿"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pilot_1b",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="chronos,moment",
        help="콤마 구분 모델 목록",
    )
    parser.add_argument("--seeds", type=str, default="42,123", help="콤마 구분 시드")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 디바이스 인덱스")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/ETT-small/ETTm1.csv",
        help="ETT CSV 경로",
    )
    parser.add_argument("--dry_run", action="store_true", help="실행 없이 조합만 출력")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=DEFAULT_LORA_RANK, help="LoRA rank"
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="평가 시 최대 배치 수 (0=전체)",
    )
    return parser.parse_args()


def _collate_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """배치 콜레이트 함수.

    Args:
        batch: 샘플 리스트.

    Returns:
        스택된 배치 딕셔너리.
    """
    context = torch.stack([s["context"] for s in batch], dim=0)
    target = torch.stack([s["target"] for s in batch], dim=0)
    return {"context": context, "target": target}


def _create_wrapper(
    model_name: str,
) -> ChronosWrapper | MOMENTWrapper | MoiraiWrapper | TimesFMWrapper:
    """모델 래퍼를 생성하고 로드.

    Args:
        model_name: 모델 이름 (chronos 또는 moment).

    Returns:
        로드 완료된 모델 래퍼.

    Raises:
        ValueError: 지원하지 않는 모델일 때.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

    model_cfg = OmegaConf.create(MODEL_CONFIGS[model_name])
    model_cfg_obj = cast(Any, model_cfg)

    if model_name == "chronos":
        wrapper: ChronosWrapper | MOMENTWrapper | MoiraiWrapper | TimesFMWrapper = (
            ChronosWrapper(model_cfg_obj)
        )
    elif model_name == "moment":
        wrapper = MOMENTWrapper(model_cfg_obj)
    elif model_name == "moirai":
        wrapper = MoiraiWrapper(model_cfg_obj)
    elif model_name == "timesfm":
        wrapper = TimesFMWrapper(model_cfg_obj)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

    wrapper.load()
    return wrapper


def _set_wrapper_backbone(
    wrapper: ChronosWrapper | MOMENTWrapper | MoiraiWrapper | TimesFMWrapper,
    model_name: str,
    adapted: torch.nn.Module,
) -> None:
    if model_name == "moirai":
        setattr(cast(Any, wrapper), "module", adapted)
    else:
        setattr(cast(Any, wrapper), "backbone", adapted)


def _evaluate_on_loader(
    wrapper: ChronosWrapper | MOMENTWrapper | MoiraiWrapper | TimesFMWrapper,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    prediction_length: int,
    max_batches: int = 0,
) -> tuple[float, dict[str, float]]:
    """데이터 로더에서 평가를 수행.

    Args:
        wrapper: 모델 래퍼.
        loader: 평가용 DataLoader.
        device: 실행 디바이스.
        prediction_length: 예측 길이.
        max_batches: 최대 평가 배치 수 (0=전체).

    Returns:
        (평균 손실, 메트릭 딕셔너리) 튜플.
    """
    wrapper.eval()
    losses: list[float] = []
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            context = batch["context"].to(device)
            target = batch["target"].to(device)

            outputs = wrapper(context=context, target=target)
            pred = outputs["pred"]
            loss = outputs["loss"]

            losses.append(float(loss.detach().item()))
            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(pred=preds, target=targets)
    mean_loss = float(sum(losses) / max(1, len(losses)))
    return mean_loss, metrics


def _run_single_experiment(
    model_name: str,
    locus: LoRALocus,
    shift_type: ShiftType,
    seed: int,
    data_path: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    lora_rank: int,
    max_eval_batches: int = 0,
) -> dict[str, Any]:
    """단일 로커스 실험을 실행.

    Args:
        model_name: 모델 이름.
        locus: LoRA 삽입 위치.
        shift_type: 분포 이동 유형.
        seed: 랜덤 시드.
        data_path: ETT CSV 경로.
        device: 실행 디바이스.
        epochs: 학습 에폭.
        batch_size: 배치 크기.
        lr: 학습률.
        patience: Early stopping patience.
        lora_rank: LoRA rank.

    Returns:
        실험 결과 딕셔너리.
    """
    experiment_id = f"{model_name}_lora_{locus.value}_{shift_type.value}_seed{seed}"
    logger.info("실험 시작: %s", experiment_id)
    start_time = time.time()

    seed_everything(seed)
    prediction_length = MODEL_CONFIGS[model_name]["prediction_length"]

    # ─── 데이터 로드 + 이동 적용 ───────────────────────────
    ett_cfg = ETTConfig(
        dataset="ETTm1",
        path=data_path,
        target_col="OT",
        context_length=MODEL_CONFIGS[model_name]["context_length"],
        prediction_length=prediction_length,
    )
    train_ds, val_ds, test_ds = load_ett(ett_cfg)

    # 학습 데이터에 분포 이동 적용
    shift_gen = ShiftGenerator(seed=seed)
    original_train_data = train_ds.data.copy()
    shifted_train_data = shift_gen.apply(
        original_train_data, shift_type=shift_type, severity=DEFAULT_SEVERITY
    )
    shift_profile = compute_shift_profile(original_train_data, shifted_train_data)
    train_ds.data = shifted_train_data

    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch
    )
    test_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch
    )

    # ─── 모델 로드 + LoRA 적용 ─────────────────────────────
    wrapper = _create_wrapper(model_name)
    backbone = wrapper.get_backbone()

    architecture = MODEL_CONFIGS[model_name].get("architecture", "t5_encoder_decoder")
    task_type_map = {
        "t5_encoder_decoder": "SEQ_2_SEQ_LM",
        "t5_encoder": "FEATURE_EXTRACTION",
        "moirai": "FEATURE_EXTRACTION",
        "timesfm": "FEATURE_EXTRACTION",
    }
    layers_pattern_map = {
        "t5_encoder_decoder": "block",
        "t5_encoder": "block",
        "moirai": "layers",
        "timesfm": "layers",
    }
    task_type = task_type_map.get(architecture, "SEQ_2_SEQ_LM")

    # EARLY/LATE loci 는 layers 필터 사용
    layers_filter = "all"
    effective_locus = locus
    if locus == LoRALocus.EARLY_LAYERS:
        effective_locus = LoRALocus.ATTN_ALL
        layers_filter = "early"
    elif locus == LoRALocus.LATE_LAYERS:
        effective_locus = LoRALocus.ATTN_ALL
        layers_filter = "late"

    lora_cfg = LoRAAdaptationConfig(
        rank=lora_rank,
        alpha=DEFAULT_LORA_ALPHA,
        dropout=DEFAULT_LORA_DROPOUT,
        locus=effective_locus,
        task_type=task_type,
        layers=layers_filter,
        num_layers=MODEL_CONFIGS[model_name]["num_layers"],
        layers_pattern=layers_pattern_map.get(architecture, "block"),
        architecture=architecture,
    )
    adapted = apply_lora(backbone, lora_cfg)
    _set_wrapper_backbone(wrapper, model_name, adapted)
    wrapper.to(device)

    # 파라미터 수 기록
    trainable_params = sum(
        p.numel() for p in wrapper.get_backbone().parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in wrapper.get_backbone().parameters())

    # ─── 학습 루프 ─────────────────────────────────────────
    param_list = wrapper.get_trainable_parameters()
    if len(param_list) == 0:
        logger.warning("학습 가능한 파라미터가 없습니다: %s", experiment_id)
        train_loss_final = 0.0
        val_loss_best = float("inf")
    else:
        optimizer = AdamW(param_list, lr=lr, weight_decay=0.01)

        # BFloat16 models don't need GradScaler — disable to avoid
        # "_amp_foreach_non_finite_check_and_unscale_cuda not implemented
        # for BFloat16" errors.
        model_dtype = next(wrapper.get_backbone().parameters()).dtype
        use_scaler = device.type == "cuda" and model_dtype != torch.bfloat16
        scaler = GradScaler(enabled=use_scaler)
        not_improved = 0
        val_loss_best = float("inf")
        train_loss_final = 0.0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(epochs):
            wrapper.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                context = batch["context"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=device.type == "cuda"):
                    outputs = wrapper(context=context, target=target)
                    loss = outputs["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(param_list, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += float(loss.detach().item())
                num_batches += 1

            train_loss_final = epoch_loss / max(1, num_batches)

            val_loss, _ = _evaluate_on_loader(
                wrapper,
                val_loader,
                device,
                prediction_length,
                max_batches=max_eval_batches,
            )

            if val_loss < val_loss_best:
                val_loss_best = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in wrapper.get_backbone().state_dict().items()
                }
                not_improved = 0
            else:
                not_improved += 1

            logger.info(
                "[%s] epoch=%d, train_loss=%.6f, val_loss=%.6f, patience=%d/%d",
                experiment_id,
                epoch + 1,
                train_loss_final,
                val_loss,
                not_improved,
                patience,
            )

            if not_improved >= patience:
                logger.info("Early stopping: %s", experiment_id)
                break

        if best_state is not None:
            wrapper.get_backbone().load_state_dict(best_state, strict=False)

    # ─── 테스트 평가 ───────────────────────────────────────
    test_loss, test_metrics = _evaluate_on_loader(
        wrapper,
        test_loader,
        device,
        prediction_length,
        max_batches=max_eval_batches,
    )

    gpu_mem = log_gpu_memory(prefix=f"{experiment_id} | ")
    gpu_memory_mb = float(gpu_mem.get("gpu_max_memory_mb", 0.0)) if gpu_mem else 0.0

    elapsed = time.time() - start_time
    logger.info(
        "실험 완료: %s, MAE=%.6f, elapsed=%.1fs",
        experiment_id,
        test_metrics.get("mae", -1.0),
        elapsed,
    )

    del wrapper
    torch.cuda.empty_cache()

    return {
        "experiment_id": experiment_id,
        "model": model_name,
        "locus": locus.value,
        "shift_type": shift_type.value,
        "severity": DEFAULT_SEVERITY.value,
        "seed": seed,
        "lora_rank": lora_rank,
        "shift_profile": asdict(shift_profile),
        "metrics": test_metrics,
        "train_loss_final": train_loss_final,
        "val_loss_best": val_loss_best,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "gpu_memory_mb": gpu_memory_mb,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    """Phase 1B 파일럿 실험 메인 함수.

    Raises:
        RuntimeError: 실험 실행 중 복구 불가능한 오류 발생 시.
    """
    _setup_logging()
    args = _parse_args()

    models = [m.strip() for m in args.models.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = get_device()

    total = len(models) * len(SELECTED_SHIFTS) * len(ALL_LOCI) * len(seeds)
    logger.info(
        "Phase 1B 파일럿: %d models × %d shifts × %d loci × %d seeds = %d 실험",
        len(models),
        len(SELECTED_SHIFTS),
        len(ALL_LOCI),
        len(seeds),
        total,
    )

    if args.dry_run:
        run_idx = 0
        for model_name in models:
            for shift_type in SELECTED_SHIFTS:
                for locus in ALL_LOCI:
                    for seed in seeds:
                        run_idx += 1
                        exp_id = f"{model_name}_lora_{locus.value}_{shift_type.value}_seed{seed}"
                        logger.info("[%d/%d] (dry run) %s", run_idx, total, exp_id)
        logger.info("Dry run 완료: %d 실험 조합", total)
        return

    all_results: list[dict[str, Any]] = []
    run_idx = 0
    failed = 0

    for model_name in models:
        for shift_type in SELECTED_SHIFTS:
            for locus in ALL_LOCI:
                for seed in seeds:
                    run_idx += 1
                    exp_id = (
                        f"{model_name}_lora_{locus.value}_{shift_type.value}_seed{seed}"
                    )
                    logger.info("[%d/%d] %s", run_idx, total, exp_id)

                    result_file = output_dir / f"{exp_id}.json"
                    if result_file.exists():
                        logger.info("이미 완료: %s", exp_id)
                        with open(result_file, "r") as f:
                            all_results.append(json.load(f))
                        continue

                    try:
                        result = _run_single_experiment(
                            model_name=model_name,
                            locus=locus,
                            shift_type=shift_type,
                            seed=seed,
                            data_path=args.data_path,
                            device=device,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            patience=args.patience,
                            lora_rank=args.lora_rank,
                            max_eval_batches=args.max_eval_batches,
                        )
                        all_results.append(result)

                        with open(result_file, "w") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)

                    except Exception as exc:
                        logger.error("실험 실패: %s — %s", exp_id, exc)
                        failed += 1

    summary_path = output_dir / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(
        "Phase 1B 완료: 성공=%d, 실패=%d, 결과=%s",
        len(all_results),
        failed,
        summary_path,
    )


if __name__ == "__main__":
    main()
