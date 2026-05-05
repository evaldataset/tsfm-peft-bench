"""Gradient probe: 레이어별 gradient L2 norm 수집 도구.

PEFT 방법이 특정 아키텍처-도메인 쌍에서 성공/실패하는 이유를
메카니즘적 증거로 제공하기 위해 학습 중 per-layer gradient L2 norm을 수집.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import gc
import importlib.util
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
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
from src.data.smd import SMDConfig, SMDDataset, load_smd
from src.models.chronos import ChronosWrapper
from src.models.moment import MOMENTWrapper
from src.models.moirai import MoiraiWrapper
from src.utils.device import get_device, log_gpu_memory
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

# ─── 모델 설정 ─────────────────────────────────────────────────
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
    "finance": {
        "loader": "finance",
        "path": "data/exchange_rate/exchange_rate.csv",
        "target_col": "AUD",
        "dataset": "ExchangeRate",
    },
}

# ─── LoRA locus 매핑 ───────────────────────────────────────────
LOCUS_MAP: dict[str, LoRALocus] = {
    "attn_qv": LoRALocus.ATTN_QV,
    "attn_all": LoRALocus.ATTN_ALL,
    "ffn": LoRALocus.FFN,
    "attn_qv_ffn": LoRALocus.ATTN_QV_FFN,
    "early_layers": LoRALocus.EARLY_LAYERS,
    "late_layers": LoRALocus.LATE_LAYERS,
}

WrapperType = Union[ChronosWrapper, MOMENTWrapper, MoiraiWrapper]


@dataclass
class CellSpec:
    """단일 probe 셀 명세.

    Attributes:
        model: 모델 이름.
        method: 적응 방법 이름.
        domain: 도메인 이름.
    """

    model: str
    method: str
    domain: str

    @classmethod
    def parse(cls, cell_str: str) -> CellSpec:
        """'model:method:domain' 형식 문자열에서 CellSpec을 생성.

        Args:
            cell_str: 'model:method:domain' 형식 문자열.

        Returns:
            CellSpec 인스턴스.

        Raises:
            ValueError: 형식이 올바르지 않을 때.
        """
        parts = cell_str.strip().split(":")
        if len(parts) != 3:
            raise ValueError(
                f"셀 형식 오류: '{cell_str}'. 'model:method:domain' 형식이어야 합니다."
            )
        model, method, domain = parts
        if model not in MODEL_CONFIGS:
            raise ValueError(
                f"지원하지 않는 모델: '{model}'. 지원 목록: {list(MODEL_CONFIGS.keys())}"
            )
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(
                f"지원하지 않는 도메인: '{domain}'. 지원 목록: {list(DOMAIN_CONFIGS.keys())}"
            )
        return cls(model=model, method=method, domain=domain)


@dataclass
class GradStats:
    """파라미터/레이어의 gradient 통계.

    Attributes:
        steps: 각 step의 gradient norm 값 리스트.
    """

    steps: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """통계 요약 딕셔너리를 반환.

        Returns:
            mean/std/max/min 키를 포함한 딕셔너리.
        """
        if not self.steps:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
        arr = np.array(self.steps, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "max": float(arr.max()),
            "min": float(arr.min()),
        }


# ─── 레이어 인덱스 추출 ────────────────────────────────────────

_LAYER_INDEX_RE = re.compile(r"(?:block|layers)\.(\d+)\.")


def _extract_layer_index(param_name: str) -> str:
    """파라미터 이름에서 레이어 인덱스를 추출.

    Args:
        param_name: 파라미터의 fully-qualified 이름.

    Returns:
        'layer_N' 형식 문자열. 레이어를 찾지 못하면 'layer_other'.
    """
    match = _LAYER_INDEX_RE.search(param_name)
    if match:
        return f"layer_{match.group(1)}"
    return "layer_other"


# ─── 데이터 로딩 ────────────────────────────────────────────────


def _load_domain_data(
    domain: str,
    context_length: int,
    prediction_length: int,
) -> tuple[
    Dataset[dict[str, torch.Tensor]],
    Dataset[dict[str, torch.Tensor]],
    Dataset[dict[str, torch.Tensor]],
]:
    """도메인별 데이터를 로드.

    Args:
        domain: 도메인 이름.
        context_length: 컨텍스트 길이.
        prediction_length: 예측 길이.

    Returns:
        (train_ds, val_ds, test_ds) 튜플.

    Raises:
        ValueError: 지원하지 않는 도메인 또는 로더일 때.
    """
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
    else:
        raise ValueError(f"지원하지 않는 로더: {loader_type}")


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


# ─── 모델 헬퍼 ─────────────────────────────────────────────────


def _create_wrapper(model_name: str) -> WrapperType:
    """모델 래퍼를 생성하고 로드.

    Args:
        model_name: 모델 이름.

    Returns:
        로드 완료된 모델 래퍼.

    Raises:
        ValueError: 지원하지 않는 모델일 때.
    """
    model_cfg = OmegaConf.create(MODEL_CONFIGS[model_name])
    model_cfg_obj = cast(Any, model_cfg)

    if model_name == "chronos":
        wrapper: WrapperType = ChronosWrapper(model_cfg_obj)
    elif model_name == "moment":
        wrapper = MOMENTWrapper(model_cfg_obj)
    elif model_name == "moirai":
        wrapper = MoiraiWrapper(model_cfg_obj)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

    wrapper.load()
    return wrapper


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
) -> None:
    """적응 기법을 적용.

    Args:
        wrapper: 모델 래퍼.
        method: 적응 방법 이름.
        model_name: 모델 이름.
        rank: LoRA rank (lora 메소드일 때 사용).

    Raises:
        ValueError: 지원하지 않는 적응 방법일 때.
    """
    backbone = wrapper.get_backbone()
    architecture = MODEL_CONFIGS[model_name].get("architecture", "t5_encoder_decoder")

    task_type_map: dict[str, str] = {
        "t5_encoder_decoder": "SEQ_2_SEQ_LM",
        "t5_encoder": "FEATURE_EXTRACTION",
        "moirai": "FEATURE_EXTRACTION",
    }
    layers_pattern_map: dict[str, str] = {
        "t5_encoder_decoder": "block",
        "t5_encoder": "block",
        "moirai": "layers",
    }
    task_type = task_type_map.get(architecture, "SEQ_2_SEQ_LM")

    if method == "zero_shot":
        return
    elif method == "head_only":
        apply_head_only(backbone)
    elif method == "lora":
        lora_cfg = LoRAAdaptationConfig(
            rank=rank,
            alpha=rank * 2,
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
    elif method == "ia3" or method == "adapter":
        # ia3는 adapter 모듈을 통해 IA³ 방식으로 적용
        adapter_cfg = AdapterAdaptationConfig(
            bottleneck_size=64,
            task_type=task_type,
        )
        adapted = apply_adapter(backbone, adapter_cfg)
        _set_wrapper_backbone(wrapper, model_name, adapted)
    elif method == "prefix":
        prefix_cfg = PrefixAdaptationConfig(
            num_virtual_tokens=32,
            task_type=task_type,
        )
        adapted = apply_prefix_tuning(backbone, prefix_cfg)
        _set_wrapper_backbone(wrapper, model_name, adapted)
    elif method == "full_fine_tuning" or method == "full_ft":
        for param in backbone.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"지원하지 않는 적응 방법: {method}")


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
    }
    module_name = required_module_map.get(model_name)
    if module_name is None:
        return False
    return importlib.util.find_spec(module_name) is not None


# ─── 결과 저장 ─────────────────────────────────────────────────


def _save_result_atomic(result: dict[str, Any], path: Path) -> None:
    """결과를 원자적으로 JSON 파일에 저장.

    Args:
        result: 저장할 결과 딕셔너리.
        path: 저장 경로.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = path.parent / "_probe_tmp"
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


# ─── 핵심 probe 실행 ──────────────────────────────────────────


def _run_probe(
    cell: CellSpec,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: Path,
) -> dict[str, Any]:
    """단일 셀×시드 gradient probe를 실행.

    Args:
        cell: 실험 셀 명세 (model, method, domain).
        seed: 랜덤 시드.
        device: 실행 디바이스.
        epochs: 학습 에폭 수.
        batch_size: 배치 크기.
        lr: 학습률.
        output_dir: 결과 저장 디렉토리.

    Returns:
        probe 결과 딕셔너리.
    """
    probe_id = f"{cell.model}_{cell.method}_{cell.domain}_seed{seed}"
    logger.info("Gradient probe 시작: %s", probe_id)
    start_time = time.time()

    seed_everything(seed)

    model_cfg = MODEL_CONFIGS[cell.model]
    context_length: int = model_cfg["context_length"]
    prediction_length: int = model_cfg["prediction_length"]

    # ─── 데이터 로드 ──────────────────────────────────────
    train_ds, _val_ds, _test_ds = _load_domain_data(
        cell.domain, context_length, prediction_length
    )
    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )

    # ─── 모델 로드 + 적응 적용 ────────────────────────────
    wrapper = _create_wrapper(cell.model)
    wrapper.to(device)
    _apply_adaptation(wrapper, cell.method, cell.model)
    wrapper.to(device)

    log_gpu_memory(prefix=f"{probe_id} | ")

    # ─── gradient 통계 누적 버퍼 ──────────────────────────
    # param_name → GradStats
    param_grad_buffers: dict[str, GradStats] = {}
    # layer_key → GradStats (레이어별 집계)
    layer_grad_buffers: dict[str, GradStats] = {}

    # zero_shot은 학습 없음 — 빈 통계 반환
    if cell.method == "zero_shot":
        logger.warning("zero_shot은 gradient를 수집하지 않습니다: %s", probe_id)
        result: dict[str, Any] = {
            "model": cell.model,
            "method": cell.method,
            "domain": cell.domain,
            "seed": seed,
            "layer_grad_norms": {},
            "param_grad_norms": {},
            "total_steps": 0,
            "epochs": epochs,
            "probe_id": probe_id,
            "elapsed_seconds": 0.0,
            "note": "zero_shot: gradient 없음",
        }
        out_path = output_dir / f"{probe_id}.json"
        _save_result_atomic(result, out_path)
        del wrapper
        torch.cuda.empty_cache()
        gc.collect()
        return result

    trainable_params = wrapper.get_trainable_parameters()
    if not trainable_params:
        logger.warning("학습 가능한 파라미터가 없습니다: %s", probe_id)

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    model_dtype = next(wrapper.get_backbone().parameters()).dtype
    use_scaler = device.type == "cuda" and model_dtype != torch.bfloat16
    scaler = GradScaler(enabled=use_scaler)

    total_steps = 0

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

            # ─── gradient norm 수집 ────────────────────────
            for name, param in wrapper.get_backbone().named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue

                grad_norm = float(param.grad.data.norm(2).item())
                layer_key = _extract_layer_index(name)

                # 파라미터별 누적
                if name not in param_grad_buffers:
                    param_grad_buffers[name] = GradStats()
                param_grad_buffers[name].steps.append(grad_norm)

                # 레이어별 누적 (step당 레이어 내 최대값 사용)
                if layer_key not in layer_grad_buffers:
                    layer_grad_buffers[layer_key] = GradStats()
                layer_grad_buffers[layer_key].steps.append(grad_norm)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().item())
            num_batches += 1
            total_steps += 1

        logger.info(
            "[%s] epoch=%d/%d, train_loss=%.6f, steps=%d",
            probe_id,
            epoch + 1,
            epochs,
            epoch_loss / max(1, num_batches),
            total_steps,
        )

    # ─── 통계 직렬화 ──────────────────────────────────────
    layer_grad_norms: dict[str, Any] = {}
    for layer_key in sorted(layer_grad_buffers.keys(), key=_sort_layer_key):
        stats = layer_grad_buffers[layer_key].summary()
        stats["steps"] = layer_grad_buffers[layer_key].steps
        layer_grad_norms[layer_key] = stats

    param_grad_norms: dict[str, Any] = {}
    for param_name, grad_stats in param_grad_buffers.items():
        param_grad_norms[param_name] = grad_stats.summary()

    elapsed = time.time() - start_time
    logger.info(
        "Gradient probe 완료: %s, elapsed=%.1fs, total_steps=%d",
        probe_id,
        elapsed,
        total_steps,
    )

    result = {
        "model": cell.model,
        "method": cell.method,
        "domain": cell.domain,
        "seed": seed,
        "layer_grad_norms": layer_grad_norms,
        "param_grad_norms": param_grad_norms,
        "total_steps": total_steps,
        "epochs": epochs,
        "probe_id": probe_id,
        "elapsed_seconds": elapsed,
    }

    out_path = output_dir / f"{probe_id}.json"
    _save_result_atomic(result, out_path)
    logger.info("결과 저장: %s", out_path)

    del wrapper
    torch.cuda.empty_cache()
    gc.collect()

    return result


def _sort_layer_key(key: str) -> tuple[int, str]:
    """레이어 키를 숫자 우선으로 정렬하기 위한 보조 함수.

    Args:
        key: 'layer_N' 또는 'layer_other' 형식 문자열.

    Returns:
        (숫자 우선순위, 원본 키) 튜플.
    """
    match = re.match(r"layer_(\d+)$", key)
    if match:
        return (int(match.group(1)), key)
    return (10**9, key)


# ─── CLI ───────────────────────────────────────────────────────


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
        description="Gradient probe: PEFT 방법별 레이어 gradient L2 norm 수집"
    )
    parser.add_argument(
        "--cells",
        type=str,
        required=True,
        help="콤마 구분 셀 목록. 각 셀은 'model:method:domain' 형식. "
        "예: chronos:lora:ett_m1,moirai:adapter:smd",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123",
        help="콤마 구분 랜덤 시드 목록 (기본값: 42,123)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="학습 에폭 수 (기본값: 3). gradient 분포 관찰용으로 3-5가 적당.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="배치 크기 (기본값: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="학습률 (기본값: 1e-4)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU 디바이스 인덱스 (기본값: 0). -1이면 CPU 사용.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gradient_analysis",
        help="결과 저장 디렉토리 (기본값: results/gradient_analysis)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="실행 없이 셀 목록만 출력",
    )
    return parser.parse_args()


def main() -> None:
    """Gradient probe 메인 엔트리포인트."""
    _setup_logging()
    args = _parse_args()

    # ─── 셀 파싱 ──────────────────────────────────────────
    cells: list[CellSpec] = []
    for raw in args.cells.split(","):
        raw = raw.strip()
        if not raw:
            continue
        cells.append(CellSpec.parse(raw))

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)

    total = len(cells) * len(seeds)
    logger.info(
        "Gradient probe: %d 셀 × %d 시드 = %d 실험",
        len(cells),
        len(seeds),
        total,
    )

    if args.dry_run:
        for idx, (cell, seed) in enumerate(
            (c, s) for c in cells for s in seeds
        ):
            logger.info(
                "[%d/%d] (dry run) %s:%s:%s seed=%d",
                idx + 1,
                total,
                cell.model,
                cell.method,
                cell.domain,
                seed,
            )
        return

    # ─── 가용 모델 확인 ───────────────────────────────────
    unavailable: set[str] = set()
    for cell in cells:
        if not _is_model_available(cell.model):
            unavailable.add(cell.model)
    if unavailable:
        logger.warning(
            "의존 패키지 누락 모델이 포함되어 있습니다: %s. "
            "해당 모델 셀은 건너뜁니다.",
            ", ".join(sorted(unavailable)),
        )

    # ─── 디바이스 설정 ────────────────────────────────────
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = get_device()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── probe 실행 ───────────────────────────────────────
    run_idx = 0
    failed = 0
    results: list[dict[str, Any]] = []

    for cell in cells:
        if cell.model in unavailable:
            logger.warning("모델 건너뜀: %s", cell.model)
            continue

        for seed in seeds:
            run_idx += 1
            probe_id = f"{cell.model}_{cell.method}_{cell.domain}_seed{seed}"
            out_path = output_dir / f"{probe_id}.json"

            if out_path.exists():
                logger.info(
                    "[%d/%d] 이미 완료된 probe 건너뜀: %s",
                    run_idx,
                    total,
                    probe_id,
                )
                continue

            logger.info("[%d/%d] 시작: %s", run_idx, total, probe_id)
            try:
                result = _run_probe(
                    cell=cell,
                    seed=seed,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    output_dir=output_dir,
                )
                results.append(result)
            except Exception:
                logger.exception("Probe 실패: %s", probe_id)
                failed += 1

    # ─── 요약 저장 ────────────────────────────────────────
    summary = {
        "total_probes": run_idx,
        "completed": len(results),
        "failed": failed,
        "cells": [
            {"model": c.model, "method": c.method, "domain": c.domain}
            for c in cells
        ],
        "seeds": seeds,
        "epochs": args.epochs,
    }
    summary_path = output_dir / "probe_summary.json"
    _save_result_atomic(summary, summary_path)
    logger.info(
        "Gradient probe 전체 완료. 완료=%d, 실패=%d. 요약: %s",
        len(results),
        failed,
        summary_path,
    )


if __name__ == "__main__":
    main()
