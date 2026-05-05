"""Phase 2: Multi-domain PEFT expansion experiments.

파일럿 결과를 기반으로 다중 도메인에서 PEFT 방법을 검증하는 확장 실험.
3가지 모드를 지원:
- domain: 메소드 비교 (models × methods × domains × seeds)
- rank: LoRA rank sweep (models × ranks × domains × seeds)
- locus: 도메인별 locus 분석 (models × loci × domains × seeds)
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import gc
import importlib.util
import os
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Union, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset

from src.adaptation.adapter import AdapterAdaptationConfig, apply_adapter
from src.adaptation.head import apply_head_only
from src.adaptation.lora import LoRAAdaptationConfig, LoRALocus, apply_lora
from src.adaptation.prefix import PrefixAdaptationConfig, apply_prefix_tuning
from src.data.ett import ETTConfig, ETTDataset, load_ett
from src.data.finance import FinanceConfig, FinanceDataset, load_finance
from src.data.physionet import PhysioNetConfig, PhysioNetDataset, load_physionet
from src.data.smd import SMDConfig, SMDDataset, load_smd
from src.evaluation.metrics import compute_metrics
from src.models.chronos import ChronosWrapper
from src.models.moment import MOMENTWrapper
from src.models.moirai import MoiraiWrapper
from src.models.timesfm_wrapper import TimesFMWrapper
from src.utils.device import get_device, log_gpu_memory
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

# ─── 모델 설정 (pilot_phase1a.py에서 복사) ─────────────────────
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "chronos": {
        "name": "chronos",
        "hf_id": "amazon/chronos-t5-base",
        "architecture": "t5_encoder_decoder",
        "num_layers": 12,
        "hidden_size": 768,
        "context_length": 512,
        "prediction_length": 64,
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

# ─── 도메인 설정 ────────────────────────────────────────────────
DOMAIN_CONFIGS: dict[str, dict[str, Any]] = {
    "ett_m1": {
        "loader": "ett",
        "path": "data/ETT-small/ETTm1.csv",
        "target_col": "OT",
        "dataset": "ETTm1",
    },
    "ett_h1": {
        "loader": "ett",
        "path": "data/ETT-small/ETTh1.csv",
        "target_col": "OT",
        "dataset": "ETTh1",
    },
    "smd": {
        "loader": "smd",
        "path": "data/SMD",
        "dataset": "SMD",
        "target_col": 0,
    },
    "psm": {
        "loader": "smd",
        "path": "data/PSM",
        "dataset": "PSM",
        "target_col": 0,
    },
    "finance": {
        "loader": "finance",
        "path": "data/exchange_rate/exchange_rate.csv",
        "target_col": "AUD",
        "dataset": "ExchangeRate",
    },
    "physionet": {
        "loader": "physionet",
        "path": "data/physionet",
        "target_col": "HR",
        "dataset": "PhysioNet2012",
    },
    "physionet_sao2": {
        "loader": "physionet",
        "path": "data/physionet_sao2",
        "target_col": "SaO2",
        "dataset": "PhysioNet2012-SaO2",
    },
    "physionet_resprate": {
        "loader": "physionet",
        "path": "data/physionet_resprate",
        "target_col": "RespRate",
        "dataset": "PhysioNet2012-RespRate",
    },
}

# ─── LoRA Locus 매핑 ───────────────────────────────────────────
LOCUS_MAP: dict[str, LoRALocus] = {
    "attn_qv": LoRALocus.ATTN_QV,
    "attn_all": LoRALocus.ATTN_ALL,
    "ffn": LoRALocus.FFN,
    "attn_qv_ffn": LoRALocus.ATTN_QV_FFN,
    "early_layers": LoRALocus.EARLY_LAYERS,
    "late_layers": LoRALocus.LATE_LAYERS,
}

WrapperType = Union[ChronosWrapper, MOMENTWrapper, MoiraiWrapper, TimesFMWrapper]


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
        description="Phase 2: Multi-domain PEFT expansion experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["domain", "rank", "locus"],
        required=True,
        help="실험 모드: domain(메소드 비교), rank(LoRA rank sweep), locus(locus 분석)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/expansion", help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="chronos,moment,moirai,timesfm",
        help="콤마 구분 모델 목록",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="ett_m1,smd,finance",
        help="콤마 구분 도메인 목록",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="zero_shot,head_only,lora,dora,adapter,prefix,full_fine_tuning",
        help="콤마 구분 적응 방법 목록 (domain 모드용)",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="4,8,16,32",
        help="콤마 구분 LoRA rank 목록 (rank 모드용)",
    )
    parser.add_argument(
        "--loci",
        type=str,
        default="attn_qv,attn_all,ffn,attn_qv_ffn,early_layers,late_layers",
        help="콤마 구분 LoRA locus 목록 (locus 모드용)",
    )
    parser.add_argument("--seeds", type=str, default="42,123,7,2024,3407", help="콤마 구분 시드")
    parser.add_argument("--context_length", type=int, default=None, help="모델 context 길이 오버라이드 (기본값: 모델 설정 사용)")
    parser.add_argument("--prediction_length", type=int, default=None, help="prediction 길이 오버라이드 (기본값: 모델 설정 사용)")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 디바이스 인덱스")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument("--dry_run", action="store_true", help="실행 없이 조합만 출력")
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="SMD/PSM train set 서브샘플 상한 (예: 50000)"
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=None,
        help="SMD/PSM val/test set 서브샘플 상한 (예: 10000)"
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="각 실험의 백본 체크포인트(.pt)를 저장",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/expansion",
        help="체크포인트 저장 루트 디렉토리",
    )
    parser.add_argument(
        "--skip_unavailable_models",
        action="store_true",
        help="의존 패키지가 없는 모델을 자동으로 제외",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="이미 완료된 실험도 강제 재실행 (체크포인트 재생성 등에 사용)",
    )
    return parser.parse_args()


# ─── 데이터 로딩 ────────────────────────────────────────────────


def _load_domain_data(
    domain: str,
    context_length: int,
    prediction_length: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[
    Dataset[dict[str, torch.Tensor]],
    Dataset[dict[str, torch.Tensor]],
    Dataset[dict[str, torch.Tensor]],
]:
    """도메인별 데이터를 로드.

    Args:
        domain: 도메인 이름 (ett_m1, smd, psm, finance 등).
        context_length: 컨텍스트 길이.
        prediction_length: 예측 길이.
        max_train_samples: SMD/PSM train 서브샘플 상한.
        max_eval_samples: SMD/PSM val/test 서브샘플 상한.

    Returns:
        (train_ds, val_ds, test_ds) 튜플.

    Raises:
        ValueError: 지원하지 않는 도메인일 때.
    """
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(
            f"지원하지 않는 도메인: {domain}. 지원 목록: {list(DOMAIN_CONFIGS.keys())}"
        )

    cfg = DOMAIN_CONFIGS[domain]
    loader_type = cfg["loader"]

    if loader_type == "ett":
        ett_cfg = ETTConfig(
            dataset=cfg["dataset"],
            path=cfg["path"],
            target_col=cfg["target_col"],
            context_length=context_length,
            prediction_length=prediction_length,
        )
        return load_ett(ett_cfg)
    elif loader_type == "smd":
        smd_cfg = SMDConfig(
            dataset=cfg["dataset"],
            path=cfg["path"],
            target_col=cfg["target_col"],
            context_length=context_length,
            prediction_length=prediction_length,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
        )
        return load_smd(smd_cfg)
    elif loader_type == "finance":
        fin_cfg = FinanceConfig(
            dataset=cfg["dataset"],
            path=cfg["path"],
            target_col=cfg["target_col"],
            context_length=context_length,
            prediction_length=prediction_length,
        )
        return load_finance(fin_cfg)
    elif loader_type == "physionet":
        # PhysioNet 긴 context 실험을 위해 환경변수로 entity-boundary 완화 가능.
        enforce_boundary = os.environ.get("TSFM_PHYSIONET_BOUNDARY", "1") != "0"
        physio_cfg = PhysioNetConfig(
            dataset=cfg["dataset"],
            data_dir=cfg["path"],
            target_col=cfg["target_col"],
            context_length=context_length,
            prediction_length=prediction_length,
            enforce_entity_boundary=enforce_boundary,
        )
        return load_physionet(physio_cfg)
    else:
        raise ValueError(f"지원하지 않는 로더: {loader_type}")


# ─── 모델 헬퍼 ─────────────────────────────────────────────────


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


def _create_wrapper(model_name: str) -> WrapperType:
    """모델 래퍼를 생성하고 로드.

    Args:
        model_name: 모델 이름.

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
        wrapper: WrapperType = ChronosWrapper(model_cfg_obj)
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


def _is_model_available(model_name: str) -> bool:
    """모델 의존 패키지 설치 여부를 확인.

    Args:
        model_name: 모델 이름.

    Returns:
        실행 가능하면 True, 아니면 False.
    """
    required_module_map: dict[str, str] = {
        "chronos": "chronos",
        "moment": "momentfm",
        "moirai": "uni2ts.model.moirai",
        "timesfm": "timesfm",
    }
    module_name = required_module_map.get(model_name)
    if module_name is None:
        return False
    return importlib.util.find_spec(module_name) is not None


def _filter_available_models(
    models: list[str],
    skip_unavailable_models: bool,
) -> list[str]:
    """실행 가능한 모델 목록만 필터링.

    Args:
        models: 사용자 입력 모델 목록.
        skip_unavailable_models: 의존 패키지 누락 모델을 제외할지 여부.

    Returns:
        실행 대상 모델 목록.
    """
    if not skip_unavailable_models:
        return models

    available_models: list[str] = []
    skipped_models: list[str] = []
    for model in models:
        if _is_model_available(model):
            available_models.append(model)
        else:
            skipped_models.append(model)

    if skipped_models:
        logger.warning(
            "의존 패키지 누락으로 모델 제외: %s",
            ", ".join(skipped_models),
        )
    if not available_models:
        logger.error(
            "실행 가능한 모델이 없습니다. --models 또는 환경 의존성을 확인하세요."
        )
    return available_models


def _set_wrapper_backbone(
    wrapper: WrapperType,
    model_name: str,
    adapted: torch.nn.Module,
) -> None:
    """래퍼에 적응된 백본을 설정.

    Args:
        wrapper: 모델 래퍼.
        model_name: 모델 이름.
        adapted: 적응이 적용된 모듈.
    """
    if model_name == "moirai":
        setattr(cast(Any, wrapper), "module", adapted)
    else:
        setattr(cast(Any, wrapper), "backbone", adapted)


def _apply_adaptation(
    wrapper: WrapperType,
    method: str,
    model_name: str,
    rank: int = 8,
    locus_name: str = "attn_all",
) -> None:
    """적응 기법을 적용.

    Args:
        wrapper: 모델 래퍼.
        method: 적응 방법 이름.
        model_name: 모델 이름.
        rank: LoRA rank (lora 메소드일 때만 사용).
        locus_name: LoRA locus 이름 (lora 메소드일 때만 사용).

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
        locus = LOCUS_MAP.get(locus_name, LoRALocus.ATTN_ALL)
        # EARLY/LATE loci use layer filtering with ATTN_ALL modules
        layers_filter = "all"
        effective_locus = locus
        if locus == LoRALocus.EARLY_LAYERS:
            effective_locus = LoRALocus.ATTN_ALL
            layers_filter = "early"
        elif locus == LoRALocus.LATE_LAYERS:
            effective_locus = LoRALocus.ATTN_ALL
            layers_filter = "late"
        lora_cfg = LoRAAdaptationConfig(
            rank=rank,
            alpha=rank * 2,
            dropout=0.05,
            locus=effective_locus,
            task_type=task_type,
            layers=layers_filter,
            num_layers=MODEL_CONFIGS[model_name]["num_layers"],
            layers_pattern=layers_pattern_map.get(architecture, "block"),
            architecture=architecture,
        )
        adapted = apply_lora(backbone, lora_cfg)
        _set_wrapper_backbone(wrapper, model_name, adapted)
    elif method == "dora":
        locus = LOCUS_MAP.get(locus_name, LoRALocus.ATTN_ALL)
        layers_filter = "all"
        effective_locus = locus
        if locus == LoRALocus.EARLY_LAYERS:
            effective_locus = LoRALocus.ATTN_ALL
            layers_filter = "early"
        elif locus == LoRALocus.LATE_LAYERS:
            effective_locus = LoRALocus.ATTN_ALL
            layers_filter = "late"
        lora_cfg = LoRAAdaptationConfig(
            rank=rank,
            alpha=rank * 2,
            dropout=0.05,
            locus=effective_locus,
            task_type=task_type,
            layers=layers_filter,
            num_layers=MODEL_CONFIGS[model_name]["num_layers"],
            layers_pattern=layers_pattern_map.get(architecture, "block"),
            architecture=architecture,
            use_dora=True,
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


# ─── 평가 ──────────────────────────────────────────────────────


def _evaluate_on_loader(
    wrapper: WrapperType,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    method: str,
    prediction_length: int,
) -> tuple[float, dict[str, float]]:
    """데이터 로더에서 평가를 수행.

    Args:
        wrapper: 모델 래퍼.
        loader: 평가용 DataLoader.
        device: 실행 디바이스.
        method: 적응 방법.
        prediction_length: 예측 길이.

    Returns:
        (평균 손실, 메트릭 딕셔너리) 튜플.
    """
    wrapper.eval()
    losses: list[float] = []
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_contexts: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
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
            all_contexts.append(context.detach().cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    contexts = torch.cat(all_contexts, dim=0)
    metrics = compute_metrics(
        pred=preds, target=targets, insample=contexts, seasonality=1
    )
    mean_loss = float(sum(losses) / max(1, len(losses)))
    return mean_loss, metrics


# ─── 단일 실험 실행 ────────────────────────────────────────────


def _save_result_atomic(result: dict[str, Any], path: Path) -> None:
    """결과를 원자적으로 JSON 파일에 저장.

    저장 직전에 ``experiment_mode`` 필드를 부모 디렉터리 이름에서 자동 주입한다.
    분석 파이프라인이 heuristic 없이 모드를 식별할 수 있도록 한다.

    Args:
        result: 저장할 결과 딕셔너리.
        path: 저장 경로.
    """
    # 부모 디렉터리 이름이 mode (domain/rank/locus).
    if "experiment_mode" not in result:
        result["experiment_mode"] = path.parent.name

    tmp_dir = path.parent / "_expansion_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(tmp_dir), suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _run_single_experiment(
    model_name: str,
    method: str,
    domain: str,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    rank: int = 8,
    locus_name: str = "attn_all",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    save_checkpoints: bool = False,
    checkpoint_dir: Path | None = None,
    context_length_override: int | None = None,
    prediction_length_override: int | None = None,
) -> dict[str, Any]:
    """단일 실험을 실행.

    Args:
        model_name: 모델 이름.
        method: 적응 방법.
        domain: 도메인 이름.
        seed: 랜덤 시드.
        device: 실행 디바이스.
        epochs: 학습 에폭.
        batch_size: 배치 크기.
        lr: 학습률.
        patience: Early stopping patience.
        rank: LoRA rank.
        locus_name: LoRA locus 이름.
        max_train_samples: SMD/PSM train 서브샘플 상한.
        max_eval_samples: SMD/PSM val/test 서브샘플 상한.
        save_checkpoints: 체크포인트 저장 여부.
        checkpoint_dir: 체크포인트 저장 루트 경로.

    Returns:
        실험 결과 딕셔너리.
    """
    experiment_id = f"{model_name}_{method}_{domain}_seed{seed}"
    logger.info("실험 시작: %s", experiment_id)
    start_time = time.time()

    seed_everything(seed)
    prediction_length = prediction_length_override or MODEL_CONFIGS[model_name]["prediction_length"]
    context_length = context_length_override or MODEL_CONFIGS[model_name]["context_length"]

    # ─── 도메인 데이터 로드 ────────────────────────────────
    train_ds, val_ds, test_ds = _load_domain_data(
        domain, context_length, prediction_length,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

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
    # PEFT (특히 PrefixTuning)가 default device에 임베딩을 생성하는 문제 방지
    # backbone을 먼저 target device로 이동 후 적응 적용
    wrapper.to(device)
    _apply_adaptation(wrapper, method, model_name, rank=rank, locus_name=locus_name)
    wrapper.to(device)  # 적응 후 새로 생성된 파라미터도 동일 device로 이동
    log_gpu_memory(prefix=f"{experiment_id} | ")

    trainable_params_count = 0
    total_params_count = sum(p.numel() for p in wrapper.get_backbone().parameters())
    train_loss_final = 0.0
    val_loss_best = float("inf")
    epochs_trained = 0

    # ─── 학습 루프 ─────────────────────────────────────────
    if method != "zero_shot":
        trainable_params = wrapper.get_trainable_parameters()
        trainable_params_count = sum(p.numel() for p in trainable_params)
        if len(trainable_params) == 0:
            logger.warning("학습 가능한 파라미터가 없습니다: %s", experiment_id)
        else:
            optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

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
                epochs_trained = epoch + 1

                # 검증
                val_loss, _ = _evaluate_on_loader(
                    wrapper, val_loader, device, method, prediction_length
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
        wrapper, test_loader, device, method, prediction_length
    )

    if save_checkpoints:
        ckpt_root = checkpoint_dir or Path("checkpoints/expansion")
        ckpt_path = ckpt_root / model_name / f"{experiment_id}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        # evaluate.py와 호환되는 단일 SoT 스키마 (train.py와 동일 키 집합).
        adaptation_config = {
            "method": method,
            "rank": rank,
            "locus": locus_name,
        }
        checkpoint_payload = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "adaptation_method": method,
            "adaptation_config": adaptation_config,
            "data_name": domain,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "seed": seed,
            "rank": rank,
            "locus": locus_name,
            "test_metrics": test_metrics,
            "backbone_state_dict": wrapper.get_backbone().state_dict(),
        }
        torch.save(checkpoint_payload, ckpt_path)
        logger.info("체크포인트 저장: %s", ckpt_path)

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
    gc.collect()

    return {
        "experiment_id": experiment_id,
        "model": model_name,
        "method": method,
        "domain": domain,
        "seed": seed,
        "rank": rank,
        "locus": locus_name,
        "metrics": test_metrics,
        "trainable_params": trainable_params_count,
        "total_params": total_params_count,
        "train_loss_final": train_loss_final,
        "val_loss_best": val_loss_best,
        "epochs_trained": epochs_trained,
        "gpu_memory_mb": gpu_memory_mb,
        "train_time_seconds": elapsed,
    }


# ─── 실험 모드 ─────────────────────────────────────────────────


def _run_domain_mode(args: argparse.Namespace) -> None:
    """Mode 1: 다중 도메인 메소드 비교.

    models × methods × domains × seeds 조합 실행.

    Args:
        args: CLI 인자.
    """
    models = [m.strip() for m in args.models.split(",")]
    models = _filter_available_models(models, args.skip_unavailable_models)
    if not models:
        raise SystemExit(2)
    methods = [m.strip() for m in args.methods.split(",")]
    domains = [d.strip() for d in args.domains.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    output_dir = Path(args.output_dir) / "domain"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = get_device()

    total = len(models) * len(methods) * len(domains) * len(seeds)
    logger.info(
        "Domain mode: %d models × %d methods × %d domains × %d seeds = %d 실험",
        len(models),
        len(methods),
        len(domains),
        len(seeds),
        total,
    )

    if args.dry_run:
        run_idx = 0
        for model in models:
            for method in methods:
                for domain in domains:
                    for seed in seeds:
                        run_idx += 1
                        exp_id = f"{model}_{method}_{domain}_seed{seed}"
                        logger.info("[%d/%d] (dry run) %s", run_idx, total, exp_id)
        return

    run_idx = 0
    failed = 0

    for model in models:
        for method in methods:
            for domain in domains:
                for seed in seeds:
                    run_idx += 1
                    exp_id = f"{model}_{method}_{domain}_seed{seed}"
                    result_file = output_dir / f"{exp_id}.json"

                    if result_file.exists() and not args.force_rerun:
                        logger.info("[%d/%d] 이미 완료: %s", run_idx, total, exp_id)
                        continue

                    logger.info("[%d/%d] %s", run_idx, total, exp_id)
                    try:
                        result = _run_single_experiment(
                            model_name=model,
                            method=method,
                            domain=domain,
                            seed=seed,
                            device=device,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            patience=args.patience,
                            max_train_samples=args.max_train_samples,
                            max_eval_samples=args.max_eval_samples,
                            save_checkpoints=args.save_checkpoints,
                            checkpoint_dir=Path(args.checkpoint_dir),
                            context_length_override=args.context_length,
                            prediction_length_override=args.prediction_length,
                        )
                        _save_result_atomic(result, result_file)
                    except Exception as exc:
                        logger.error("실험 실패: %s — %s", exp_id, exc)
                        failed += 1

    logger.info("Domain mode 완료: 성공=%d, 실패=%d", run_idx - failed, failed)


def _run_rank_mode(args: argparse.Namespace) -> None:
    """Mode 2: LoRA rank sweep.

    models × ranks × domains × seeds 조합 실행 (method 고정: lora).

    Args:
        args: CLI 인자.
    """
    models = [m.strip() for m in args.models.split(",")]
    models = _filter_available_models(models, args.skip_unavailable_models)
    if not models:
        raise SystemExit(2)
    ranks = [int(r.strip()) for r in args.ranks.split(",")]
    domains = [d.strip() for d in args.domains.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    output_dir = Path(args.output_dir) / "rank"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = get_device()

    total = len(models) * len(ranks) * len(domains) * len(seeds)
    logger.info(
        "Rank mode: %d models × %d ranks × %d domains × %d seeds = %d 실험",
        len(models),
        len(ranks),
        len(domains),
        len(seeds),
        total,
    )

    if args.dry_run:
        run_idx = 0
        for model in models:
            for rank in ranks:
                for domain in domains:
                    for seed in seeds:
                        run_idx += 1
                        exp_id = f"{model}_lora_r{rank}_{domain}_seed{seed}"
                        logger.info("[%d/%d] (dry run) %s", run_idx, total, exp_id)
        return

    run_idx = 0
    failed = 0

    for model in models:
        for rank in ranks:
            for domain in domains:
                for seed in seeds:
                    run_idx += 1
                    exp_id = f"{model}_lora_r{rank}_{domain}_seed{seed}"
                    result_file = output_dir / f"{exp_id}.json"

                    if result_file.exists() and not args.force_rerun:
                        logger.info("[%d/%d] 이미 완료: %s", run_idx, total, exp_id)
                        continue

                    logger.info("[%d/%d] %s", run_idx, total, exp_id)
                    try:
                        result = _run_single_experiment(
                            model_name=model,
                            method="lora",
                            domain=domain,
                            seed=seed,
                            device=device,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            patience=args.patience,
                            rank=rank,
                            max_train_samples=args.max_train_samples,
                            max_eval_samples=args.max_eval_samples,
                            save_checkpoints=args.save_checkpoints,
                            checkpoint_dir=Path(args.checkpoint_dir),
                        )
                        _save_result_atomic(result, result_file)
                    except Exception as exc:
                        logger.error("실험 실패: %s — %s", exp_id, exc)
                        failed += 1

    logger.info("Rank mode 완료: 성공=%d, 실패=%d", run_idx - failed, failed)


def _run_locus_mode(args: argparse.Namespace) -> None:
    """Mode 3: 도메인별 locus 분석.

    models × loci × domains × seeds 조합 실행 (method 고정: lora).

    Args:
        args: CLI 인자.
    """
    models = [m.strip() for m in args.models.split(",")]
    models = _filter_available_models(models, args.skip_unavailable_models)
    if not models:
        raise SystemExit(2)
    loci = [l.strip() for l in args.loci.split(",")]
    domains = [d.strip() for d in args.domains.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    output_dir = Path(args.output_dir) / "locus"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = get_device()

    total = len(models) * len(loci) * len(domains) * len(seeds)
    logger.info(
        "Locus mode: %d models × %d loci × %d domains × %d seeds = %d 실험",
        len(models),
        len(loci),
        len(domains),
        len(seeds),
        total,
    )

    if args.dry_run:
        run_idx = 0
        for model in models:
            for locus_name in loci:
                for domain in domains:
                    for seed in seeds:
                        run_idx += 1
                        exp_id = f"{model}_lora_{locus_name}_{domain}_seed{seed}"
                        logger.info("[%d/%d] (dry run) %s", run_idx, total, exp_id)
        return

    run_idx = 0
    failed = 0

    for model in models:
        for locus_name in loci:
            for domain in domains:
                for seed in seeds:
                    run_idx += 1
                    exp_id = f"{model}_lora_{locus_name}_{domain}_seed{seed}"
                    result_file = output_dir / f"{exp_id}.json"

                    if result_file.exists() and not args.force_rerun:
                        logger.info("[%d/%d] 이미 완료: %s", run_idx, total, exp_id)
                        continue

                    logger.info("[%d/%d] %s", run_idx, total, exp_id)
                    try:
                        result = _run_single_experiment(
                            model_name=model,
                            method="lora",
                            domain=domain,
                            seed=seed,
                            device=device,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            patience=args.patience,
                            locus_name=locus_name,
                            max_train_samples=args.max_train_samples,
                            max_eval_samples=args.max_eval_samples,
                            save_checkpoints=args.save_checkpoints,
                            checkpoint_dir=Path(args.checkpoint_dir),
                        )
                        _save_result_atomic(result, result_file)
                    except Exception as exc:
                        logger.error("실험 실패: %s — %s", exp_id, exc)
                        failed += 1

    logger.info("Locus mode 완료: 성공=%d, 실패=%d", run_idx - failed, failed)


# ─── 메인 ──────────────────────────────────────────────────────


def main() -> None:
    """Phase 2 확장 실험 메인 함수."""
    _setup_logging()
    args = _parse_args()

    logger.info("Phase 2 확장 실험 시작: mode=%s", args.mode)

    if args.mode == "domain":
        _run_domain_mode(args)
    elif args.mode == "rank":
        _run_rank_mode(args)
    elif args.mode == "locus":
        _run_locus_mode(args)


if __name__ == "__main__":
    main()
