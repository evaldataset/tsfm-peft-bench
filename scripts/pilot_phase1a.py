"""Phase 1A 파일럿: Method × Shift 상호작용 실험.

가설 A 검증 — 분포 이동 유형이 최적 PEFT 방법을 결정하는지 테스트.
2 models × 4 shifts × 2 severities × 4 methods × 2 seeds = 128 runs.
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
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from src.adaptation.adapter import AdapterAdaptationConfig, apply_adapter
from src.adaptation.head import apply_head_only
from src.adaptation.lora import LoRAAdaptationConfig, LoRALocus, apply_lora
from src.adaptation.prefix import PrefixAdaptationConfig, apply_prefix_tuning
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
DEFAULT_METHODS: list[str] = [
    "zero_shot",
    "head_only",
    "lora",
    "prefix",
    "adapter",
    "full_fine_tuning",
]
ALL_SHIFT_TYPES: list[ShiftType] = [
    ShiftType.AMPLITUDE,
    ShiftType.SPECTRAL,
    ShiftType.IRREGULARITY,
    ShiftType.NONSTATIONARITY,
]
ALL_SEVERITIES: list[ShiftSeverity] = [ShiftSeverity.MILD, ShiftSeverity.STRONG]
DEFAULT_SEEDS: list[int] = [42, 123]

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
        description="Phase 1A: Method × Shift 상호작용 파일럿"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pilot_1a",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="chronos,moment",
        help="콤마 구분 모델 목록",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="zero_shot,head_only,lora,prefix,adapter,full_fine_tuning",
        help="콤마 구분 적응 방법 목록",
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
        "--max_eval_batches",
        type=int,
        default=0,
        help="평가 시 최대 배치 수 (0=전체)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="데이터 슬라이딩 윈도우 stride (0=prediction_length)",
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


def _apply_adaptation(
    wrapper: ChronosWrapper | MOMENTWrapper | MoiraiWrapper | TimesFMWrapper,
    method: str,
    model_name: str,
) -> None:
    """적응 기법을 적용.

    Args:
        wrapper: 모델 래퍼.
        method: 적응 방법 이름.
        model_name: 모델 이름 (LoRA task_type 결정용).

    Raises:
        ValueError: 지원하지 않는 적응 방법일 때.
    """
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

    if method == "zero_shot":
        return
    elif method == "head_only":
        apply_head_only(backbone)
    elif method == "lora":
        lora_cfg = LoRAAdaptationConfig(
            rank=8,
            alpha=16,
            dropout=0.05,
            locus=LoRALocus.ATTN_ALL,
            task_type=task_type,
            layers="all",
            num_layers=MODEL_CONFIGS[model_name]["num_layers"],
            layers_pattern=layers_pattern_map.get(architecture, "block"),
            architecture=architecture,
        )
        adapted = apply_lora(backbone, lora_cfg)
        _set_wrapper_backbone(wrapper, model_name, adapted)
    elif method == "prefix":
        prefix_cfg = PrefixAdaptationConfig(
            num_virtual_tokens=32,
            task_type=task_type,
        )
        adapted = apply_prefix_tuning(backbone, prefix_cfg)
        _set_wrapper_backbone(wrapper, model_name, adapted)
    elif method == "adapter":
        adapter_cfg = AdapterAdaptationConfig(
            bottleneck_size=64,
            task_type=task_type,
        )
        adapted = apply_adapter(backbone, adapter_cfg)
        _set_wrapper_backbone(wrapper, model_name, adapted)
    elif method == "full_fine_tuning":
        for param in backbone.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"지원하지 않는 적응 방법: {method}")


def _evaluate_on_loader(
    wrapper: ChronosWrapper | MOMENTWrapper | MoiraiWrapper | TimesFMWrapper,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    method: str,
    prediction_length: int,
    max_batches: int = 0,
) -> tuple[float, dict[str, float]]:
    """데이터 로더에서 평가를 수행.

    Args:
        wrapper: 모델 래퍼.
        loader: 평가용 DataLoader.
        device: 실행 디바이스.
        method: 적응 방법.
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

            if method == "zero_shot":
                pred = wrapper.predict(
                    context=context, prediction_length=prediction_length
                )
                loss = torch.mean((pred - target) ** 2)
            else:
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
    method: str,
    shift_type: ShiftType,
    severity: ShiftSeverity,
    seed: int,
    data_path: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    max_eval_batches: int = 0,
    stride: int = 0,
) -> dict[str, Any]:
    """단일 실험을 실행.

    Args:
        model_name: 모델 이름.
        method: 적응 방법.
        shift_type: 분포 이동 유형.
        severity: 이동 강도.
        seed: 랜덤 시드.
        data_path: ETT CSV 경로.
        device: 실행 디바이스.
        epochs: 학습 에폭.
        batch_size: 배치 크기.
        lr: 학습률.
        patience: Early stopping patience.
        max_eval_batches: 평가 시 최대 배치 수 (0=전체).
        stride: 데이터 슬라이딩 윈도우 stride (0=prediction_length).

    Returns:
        실험 결과 딕셔너리.
    """
    experiment_id = (
        f"{model_name}_{method}_{shift_type.value}_{severity.value}_seed{seed}"
    )
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
        original_train_data, shift_type=shift_type, severity=severity
    )

    # 이동 프로파일 계산
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

    # ─── 모델 로드 + 적응 적용 ─────────────────────────────
    wrapper = _create_wrapper(model_name)
    _apply_adaptation(wrapper, method, model_name)
    wrapper.to(device)

    train_loss_final = 0.0
    val_loss_best = float("inf")

    # ─── 학습 루프 ─────────────────────────────────────────
    if method != "zero_shot":
        trainable_params = wrapper.get_trainable_parameters()
        if len(trainable_params) == 0:
            logger.warning("학습 가능한 파라미터가 없습니다: %s", experiment_id)
        else:
            optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

            # BFloat16 models don't need GradScaler — disable to avoid
            # "_amp_foreach_non_finite_check_and_unscale_cuda not implemented
            # for BFloat16" errors.
            model_dtype = next(wrapper.get_backbone().parameters()).dtype
            use_scaler = device.type == "cuda" and model_dtype != torch.bfloat16
            scaler = GradScaler(enabled=use_scaler)
            not_improved = 0
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
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += float(loss.detach().item())
                    num_batches += 1

                train_loss_final = epoch_loss / max(1, num_batches)

                # 검증
                val_loss, _ = _evaluate_on_loader(
                    wrapper,
                    val_loader,
                    device,
                    method,
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

    # ─── 테스트 평가 (원본 데이터) ──────────────────────────
    test_loss, test_metrics = _evaluate_on_loader(
        wrapper,
        test_loader,
        device,
        method,
        prediction_length,
        max_batches=max_eval_batches,
    )

    # GPU 메모리
    gpu_mem = log_gpu_memory(prefix=f"{experiment_id} | ")
    gpu_memory_mb = float(gpu_mem.get("gpu_max_memory_mb", 0.0)) if gpu_mem else 0.0

    elapsed = time.time() - start_time
    logger.info(
        "실험 완료: %s, MAE=%.6f, elapsed=%.1fs",
        experiment_id,
        test_metrics.get("mae", -1.0),
        elapsed,
    )

    # 메모리 해제
    del wrapper
    torch.cuda.empty_cache()

    return {
        "experiment_id": experiment_id,
        "model": model_name,
        "method": method,
        "shift_type": shift_type.value,
        "severity": severity.value,
        "seed": seed,
        "shift_profile": asdict(shift_profile),
        "metrics": test_metrics,
        "train_loss_final": train_loss_final,
        "val_loss_best": val_loss_best,
        "gpu_memory_mb": gpu_memory_mb,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    """Phase 1A 파일럿 실험 메인 함수.

    Raises:
        RuntimeError: 실험 실행 중 복구 불가능한 오류 발생 시.
    """
    _setup_logging()
    args = _parse_args()

    models = [m.strip() for m in args.models.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = get_device()

    # 실험 조합 계산
    total = (
        len(models)
        * len(ALL_SHIFT_TYPES)
        * len(ALL_SEVERITIES)
        * len(methods)
        * len(seeds)
    )
    logger.info(
        "Phase 1A 파일럿: %d models × %d shifts × %d severities × %d methods × %d seeds = %d 실험",
        len(models),
        len(ALL_SHIFT_TYPES),
        len(ALL_SEVERITIES),
        len(methods),
        len(seeds),
        total,
    )

    if args.dry_run:
        run_idx = 0
        for model_name in models:
            for shift_type in ALL_SHIFT_TYPES:
                for severity in ALL_SEVERITIES:
                    for method in methods:
                        for seed in seeds:
                            run_idx += 1
                            exp_id = f"{model_name}_{method}_{shift_type.value}_{severity.value}_seed{seed}"
                            logger.info("[%d/%d] (dry run) %s", run_idx, total, exp_id)
        logger.info("Dry run 완료: %d 실험 조합", total)
        return

    all_results: list[dict[str, Any]] = []
    run_idx = 0
    failed = 0

    for model_name in models:
        for shift_type in ALL_SHIFT_TYPES:
            for severity in ALL_SEVERITIES:
                for method in methods:
                    for seed in seeds:
                        run_idx += 1
                        exp_id = f"{model_name}_{method}_{shift_type.value}_{severity.value}_seed{seed}"
                        logger.info("[%d/%d] %s", run_idx, total, exp_id)

                        # 이미 완료된 결과가 있으면 건너뛰기
                        result_file = output_dir / f"{exp_id}.json"
                        if result_file.exists():
                            logger.info("이미 완료: %s", exp_id)
                            with open(result_file, "r") as f:
                                all_results.append(json.load(f))
                            continue

                        try:
                            result = _run_single_experiment(
                                model_name=model_name,
                                method=method,
                                shift_type=shift_type,
                                severity=severity,
                                seed=seed,
                                data_path=args.data_path,
                                device=device,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                patience=args.patience,
                                max_eval_batches=args.max_eval_batches,
                                stride=args.stride,
                            )
                            all_results.append(result)

                            # 개별 결과 저장
                            with open(result_file, "w") as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)

                        except Exception as exc:
                            logger.error("실험 실패: %s — %s", exp_id, exc)
                            failed += 1

    # 전체 결과 통합 저장
    summary_path = output_dir / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(
        "Phase 1A 완료: 성공=%d, 실패=%d, 결과=%s",
        len(all_results),
        failed,
        summary_path,
    )


if __name__ == "__main__":
    main()
