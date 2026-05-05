"""에폭별 CKA 궤적 모니터링 스크립트.

학습 중 레이어별 표현이 초기 동결 모델 대비 어떻게 변화하는지 추적.
발산 감지 및 조기 경고를 위한 CKA 시계열 데이터 수집.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Union, cast

import torch
import torch.nn as nn
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
from src.evaluation.cka import CKAAnalyzer, cka
from src.evaluation.metrics import compute_metrics
from src.models.chronos import ChronosWrapper
from src.models.moment import MOMENTWrapper
from src.models.moirai import MoiraiWrapper
from src.models.timesfm_wrapper import TimesFMWrapper
from src.utils.device import get_device, log_gpu_memory
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

WrapperType = Union[ChronosWrapper, MOMENTWrapper, MoiraiWrapper, TimesFMWrapper]

# ─── 모델 설정 (run_expansion.py에서 복사) ────────────────────────
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

# ─── 도메인 설정 ─────────────────────────────────────────────────
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
    "physionet": {
        "loader": "physionet",
        "path": "data/physionet",
        "target_col": "HR",
        "dataset": "PhysioNet2012",
    },
}

# ─── 아키텍처별 레이어 패턴 ──────────────────────────────────────
LAYER_PATTERNS: dict[str, list[str]] = {
    "t5_encoder_decoder": ["encoder.block.", "decoder.block.", "block."],
    "t5_encoder": ["encoder.block.", "block."],
    "moirai": ["layers.", "module.layers."],
    "timesfm": ["layers."],
}

# ─── LoRA Locus 매핑 ──────────────────────────────────────────────
LOCUS_MAP: dict[str, LoRALocus] = {
    "attn_qv": LoRALocus.ATTN_QV,
    "attn_all": LoRALocus.ATTN_ALL,
    "ffn": LoRALocus.FFN,
    "attn_qv_ffn": LoRALocus.ATTN_QV_FFN,
    "early_layers": LoRALocus.EARLY_LAYERS,
    "late_layers": LoRALocus.LATE_LAYERS,
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
        description="에폭별 CKA 궤적 모니터링"
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--domain", type=str, required=True, choices=list(DOMAIN_CONFIGS.keys()))
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["zero_shot", "head_only", "lora", "adapter", "prefix", "full_fine_tuning"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cka_every_n", type=int, default=2, help="N 에폭마다 CKA 계산")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--probe_size", type=int, default=50, help="CKA 프로브 샘플 수")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cka_trajectories",
        help="결과 저장 디렉토리",
    )
    return parser.parse_args()


def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """배치 콜레이트 함수.

    Args:
        batch: 샘플 리스트.

    Returns:
        스택된 배치 딕셔너리.
    """
    context = torch.stack([s["context"] for s in batch], dim=0)
    target = torch.stack([s["target"] for s in batch], dim=0)
    return {"context": context, "target": target}


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
        ValueError: 지원하지 않는 도메인일 때.
    """
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"지원하지 않는 도메인: {domain}")

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
    elif loader_type == "physionet":
        physio_cfg = PhysioNetConfig(
            dataset=cfg["dataset"],
            data_dir=cfg["path"],
            target_col=cfg["target_col"],
            context_length=context_length,
            prediction_length=prediction_length,
        )
        return load_physionet(physio_cfg)
    else:
        raise ValueError(f"지원하지 않는 로더: {loader_type}")


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
    elif model_name == "timesfm":
        wrapper = TimesFMWrapper(model_cfg_obj)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

    wrapper.load()
    return wrapper


def _set_wrapper_backbone(wrapper: WrapperType, model_name: str, adapted: nn.Module) -> None:
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
) -> None:
    """적응 기법을 적용.

    Args:
        wrapper: 모델 래퍼.
        method: 적응 방법 이름.
        model_name: 모델 이름.

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


def _build_probe_set(
    val_ds: Dataset[dict[str, torch.Tensor]],
    probe_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """검증 셋에서 고정된 프로브 샘플을 구성.

    Args:
        val_ds: 검증 데이터셋.
        probe_size: 프로브 샘플 수.
        device: 실행 디바이스.

    Returns:
        {"context": ..., "target": ...} 딕셔너리 (GPU 상주).
    """
    n = min(probe_size, len(val_ds))  # type: ignore[arg-type]
    contexts: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for i in range(n):
        sample = val_ds[i]
        contexts.append(sample["context"])
        targets.append(sample["target"])
    return {
        "context": torch.stack(contexts).to(device),
        "target": torch.stack(targets).to(device),
    }


def _get_layer_names_for_model(model_name: str, backbone: nn.Module) -> list[str]:
    """모델 아키텍처에 맞는 레이어 이름 목록 반환.

    transformer block 레벨의 레이어만 수집 (서브모듈 제외).

    Args:
        model_name: 모델 이름.
        backbone: 백본 모듈.

    Returns:
        CKA를 계산할 레이어 이름 목록.
    """
    architecture = MODEL_CONFIGS[model_name].get("architecture", "t5_encoder_decoder")
    patterns = LAYER_PATTERNS.get(architecture, ["block."])

    layer_names: list[str] = []
    for name, _ in backbone.named_modules():
        # 패턴에 맞고 직접 블록 레벨인지 확인 (예: encoder.block.0, layers.0)
        for pat in patterns:
            if pat in name:
                # 블록 바로 아래 레벨만 (예: encoder.block.0 OK, encoder.block.0.layer.0 NG)
                suffix = name.split(pat)[-1]
                # suffix가 숫자만 있으면 블록 직접 레벨
                if suffix.isdigit():
                    layer_names.append(name)
                break

    if not layer_names:
        logger.warning("레이어 이름을 찾지 못했습니다 (모델: %s). 모든 named_modules 패턴 확인 필요.", model_name)

    return layer_names


class _ActivationCollector:
    """포워드 훅으로 중간 활성화를 수집하는 유틸리티.

    Args:
        backbone: 대상 백본 모듈.
        layer_names: 훅을 등록할 레이어 이름 목록.
    """

    def __init__(self, backbone: nn.Module, layer_names: list[str]) -> None:
        self.backbone = backbone
        self.layer_names = layer_names
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._activations: dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str):
        def hook_fn(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            _ = module
            _ = inputs
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self._activations[name] = out.detach().cpu()

        return hook_fn

    def register(self) -> None:
        """레이어 이름에 해당하는 모듈에 훅 등록.

        PEFT 적용 후 모듈 이름이 변경될 수 있으므로 (예: block.0 → base_model.model.block.0),
        원래 이름의 suffix 매칭을 시도한다.
        """
        self.remove()
        name_to_module: dict[str, nn.Module] = {n: m for n, m in self.backbone.named_modules()}
        for layer_name in self.layer_names:
            if layer_name in name_to_module:
                handle = name_to_module[layer_name].register_forward_hook(
                    self._make_hook(layer_name)
                )
                self._hooks.append(handle)
            else:
                # PEFT 래핑 후 이름이 바뀌었을 수 있음 — suffix 매칭 시도
                matched = False
                for full_name, mod in name_to_module.items():
                    if full_name.endswith(layer_name) or full_name.endswith("." + layer_name):
                        handle = mod.register_forward_hook(self._make_hook(layer_name))
                        self._hooks.append(handle)
                        matched = True
                        break
                if not matched:
                    logger.warning("레이어를 찾을 수 없음: %s", layer_name)

    def remove(self) -> None:
        """등록된 훅 모두 제거."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def collect(self, wrapper: WrapperType, probe: dict[str, torch.Tensor], method: str) -> dict[str, torch.Tensor]:
        """프로브 셋을 모델에 통과시켜 활성화 수집.

        Args:
            wrapper: 모델 래퍼.
            probe: {"context": ..., "target": ...} 프로브 배치.
            method: 적응 방법 (zero_shot이면 predict 사용).

        Returns:
            레이어 이름 → 활성화 텐서 딕셔너리 (CPU).
        """
        self._activations.clear()
        wrapper.eval()
        with torch.no_grad():
            if method == "zero_shot":
                _ = wrapper.predict(
                    context=probe["context"],
                    prediction_length=probe["target"].shape[-1],
                )
            else:
                _ = wrapper(context=probe["context"], target=probe["target"])
        return dict(self._activations)

    def clear(self) -> None:
        """수집된 활성화 메모리 해제."""
        self._activations.clear()


def _compute_layerwise_cka(
    baseline_acts: dict[str, torch.Tensor],
    current_acts: dict[str, torch.Tensor],
) -> dict[str, float]:
    """레이어별 CKA 값 계산.

    Args:
        baseline_acts: 동결 기준 모델의 활성화.
        current_acts: 현재 모델의 활성화.

    Returns:
        레이어 이름 → CKA 값 딕셔너리.
    """
    results: dict[str, float] = {}
    common = set(baseline_acts.keys()) & set(current_acts.keys())

    for layer_name in sorted(common):
        base = baseline_acts[layer_name]
        curr = current_acts[layer_name]

        # (batch, seq, hidden) → (batch*seq, hidden)
        base_2d = base.reshape(-1, base.shape[-1])
        curr_2d = curr.reshape(-1, curr.shape[-1])

        # 샘플 수 상한 (메모리 절약)
        max_n = min(base_2d.shape[0], curr_2d.shape[0], 1000)
        base_2d = base_2d[:max_n].float()
        curr_2d = curr_2d[:max_n].float()

        cka_val = cka(base_2d, curr_2d, kernel="linear")
        results[layer_name] = cka_val

    return results


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
                pred = wrapper.predict(context=context, prediction_length=prediction_length)
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
    metrics = compute_metrics(pred=preds, target=targets, insample=contexts, seasonality=1)
    mean_loss = float(sum(losses) / max(1, len(losses)))
    return mean_loss, metrics


def run_cka_trajectory(
    model_name: str,
    method: str,
    domain: str,
    seed: int,
    device: torch.device,
    epochs: int,
    cka_every_n: int,
    batch_size: int,
    lr: float,
    patience: int,
    probe_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    """단일 CKA 궤적 실험 실행.

    초기 동결 모델 표현을 기준으로 학습 과정에서 레이어별 CKA를 추적.

    Args:
        model_name: 모델 이름.
        method: 적응 방법.
        domain: 도메인 이름.
        seed: 랜덤 시드.
        device: 실행 디바이스.
        epochs: 학습 에폭 수.
        cka_every_n: N 에폭마다 CKA 계산.
        batch_size: 배치 크기.
        lr: 학습률.
        patience: Early stopping patience.
        probe_size: CKA 프로브 샘플 수.
        output_dir: 결과 저장 디렉토리.

    Returns:
        실험 결과 딕셔너리.
    """
    experiment_id = f"{model_name}_{method}_{domain}_seed{seed}"
    logger.info("CKA 궤적 실험 시작: %s", experiment_id)
    start_time = time.time()

    seed_everything(seed)
    prediction_length = MODEL_CONFIGS[model_name]["prediction_length"]
    context_length = MODEL_CONFIGS[model_name]["context_length"]

    # ─── 데이터 로드 ───────────────────────────────────────────
    train_ds, val_ds, test_ds = _load_domain_data(domain, context_length, prediction_length)

    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch
    )
    test_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch
    )

    # ─── 모델 로드 ─────────────────────────────────────────────
    wrapper = _create_wrapper(model_name)
    wrapper.to(device)

    # ─── 프로브 셋 구성 (학습 전 고정) ────────────────────────
    probe = _build_probe_set(val_ds, probe_size, device)
    logger.info("프로브 셋 크기: %d 샘플", probe["context"].shape[0])

    # ─── 레이어 이름 수집 ──────────────────────────────────────
    backbone_for_layers = wrapper.get_backbone()
    # LoRA 등 적응 전에 레이어 이름 수집 (backbone 구조는 동일)
    layer_names = _get_layer_names_for_model(model_name, backbone_for_layers)
    logger.info("CKA 추적 레이어 수: %d", len(layer_names))

    # ─── 동결 기준 활성화 수집 ────────────────────────────────
    collector = _ActivationCollector(wrapper.get_backbone(), layer_names)
    collector.register()
    baseline_acts = collector.collect(wrapper, probe, method="zero_shot")
    collector.remove()
    logger.info("기준 활성화 수집 완료: %d 레이어", len(baseline_acts))

    # ─── 적응 적용 ─────────────────────────────────────────────
    _apply_adaptation(wrapper, method, model_name)
    wrapper.to(device)
    log_gpu_memory(prefix=f"{experiment_id} | ")

    # ─── CKA 추적 데이터 초기화 ───────────────────────────────
    epochs_tracked: list[int] = []
    cka_matrix: list[list[float]] = []  # [n_checkpoints, n_layers]
    mean_cka_per_epoch: list[float] = []
    min_cka_per_epoch: list[float] = []

    # 에폭 0 (학습 전) CKA 기록
    if method != "zero_shot":
        collector_train = _ActivationCollector(wrapper.get_backbone(), layer_names)
        collector_train.register()
        epoch0_acts = collector_train.collect(wrapper, probe, method)
        collector_train.remove()
        collector_train.clear()

        cka_dict_0 = _compute_layerwise_cka(baseline_acts, epoch0_acts)
        layer_cka_0 = [cka_dict_0.get(ln, 1.0) for ln in layer_names]
        epochs_tracked.append(0)
        cka_matrix.append(layer_cka_0)
        mean_cka_per_epoch.append(float(sum(layer_cka_0) / max(1, len(layer_cka_0))))
        min_cka_per_epoch.append(float(min(layer_cka_0)) if layer_cka_0 else 1.0)
        del epoch0_acts

    # ─── 학습 루프 ─────────────────────────────────────────────
    val_loss_best = float("inf")
    not_improved = 0
    best_state: dict[str, torch.Tensor] | None = None
    final_mae = float("nan")

    if method != "zero_shot":
        trainable_params = wrapper.get_trainable_parameters()
        if len(trainable_params) == 0:
            logger.warning("학습 가능한 파라미터가 없습니다: %s", experiment_id)
        else:
            optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
            model_dtype = next(wrapper.get_backbone().parameters()).dtype
            use_scaler = device.type == "cuda" and model_dtype != torch.bfloat16
            scaler = GradScaler(enabled=use_scaler)

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

                train_loss = epoch_loss / max(1, num_batches)

                # 검증
                val_loss, _ = _evaluate_on_loader(wrapper, val_loader, device, method, prediction_length)

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
                    "[%s] epoch=%d/%d, train_loss=%.6f, val_loss=%.6f, patience=%d/%d",
                    experiment_id, epoch + 1, epochs, train_loss, val_loss, not_improved, patience,
                )

                # CKA 계산 (매 cka_every_n 에폭)
                current_epoch = epoch + 1
                if current_epoch % cka_every_n == 0 or current_epoch == epochs:
                    cka_collector = _ActivationCollector(wrapper.get_backbone(), layer_names)
                    cka_collector.register()
                    curr_acts = cka_collector.collect(wrapper, probe, method)
                    cka_collector.remove()

                    cka_dict = _compute_layerwise_cka(baseline_acts, curr_acts)
                    layer_cka = [cka_dict.get(ln, float("nan")) for ln in layer_names]

                    epochs_tracked.append(current_epoch)
                    cka_matrix.append(layer_cka)
                    mean_val = float(sum(v for v in layer_cka if not (v != v)) / max(1, len(layer_cka)))
                    min_val = float(min((v for v in layer_cka if not (v != v)), default=float("nan")))
                    mean_cka_per_epoch.append(mean_val)
                    min_cka_per_epoch.append(min_val)

                    logger.info(
                        "[%s] epoch=%d CKA: mean=%.4f, min=%.4f",
                        experiment_id, current_epoch, mean_val, min_val,
                    )

                    # 메모리 해제
                    cka_collector.clear()
                    del curr_acts
                    torch.cuda.empty_cache()

                if not_improved >= patience:
                    logger.info("Early stopping: %s", experiment_id)
                    break

            if best_state is not None:
                wrapper.get_backbone().load_state_dict(best_state, strict=False)

    # ─── 최종 평가 ─────────────────────────────────────────────
    _, test_metrics = _evaluate_on_loader(wrapper, test_loader, device, method, prediction_length)
    final_mae = float(test_metrics.get("mae", float("nan")))
    diverged = final_mae >= 50.0

    logger.info(
        "실험 완료: %s | MAE=%.6f | diverged=%s | elapsed=%.1fs",
        experiment_id, final_mae, diverged, time.time() - start_time,
    )

    # ─── 결과 저장 ─────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{experiment_id}.json"

    result: dict[str, Any] = {
        "model": model_name,
        "method": method,
        "domain": domain,
        "seed": seed,
        "epochs_tracked": epochs_tracked,
        "layer_names": layer_names,
        "cka_matrix": cka_matrix,
        "mean_cka_per_epoch": mean_cka_per_epoch,
        "min_cka_per_epoch": min_cka_per_epoch,
        "final_mae": final_mae,
        "diverged": diverged,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("결과 저장: %s", out_path)

    # 메모리 해제
    del wrapper, probe, baseline_acts
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main() -> None:
    """메인 진입점."""
    _setup_logging()
    args = _parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = get_device()

    logger.info("디바이스: %s", device)

    run_cka_trajectory(
        model_name=args.model,
        method=args.method,
        domain=args.domain,
        seed=args.seed,
        device=device,
        epochs=args.epochs,
        cka_every_n=args.cka_every_n,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        probe_size=args.probe_size,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
