from __future__ import annotations

# pyright: reportMissingImports=false

# ============================================================================
# DEV/DEBUG ENTRY POINT — 단일 평가용. ETT 도메인 + Chronos/MOMENT 모델만 검증.
# 논문의 모든 결과는 ``scripts/run_expansion.py``로 생성됩니다.
# ============================================================================

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.adaptation.head import apply_head_only
from src.adaptation.lora import LoRAAdaptationConfig, LoRALocus, apply_lora
from src.adaptation.prefix import PrefixAdaptationConfig, apply_prefix_tuning
from src.data.ett import ETTConfig, load_ett
from src.evaluation.metrics import compute_metrics
from src.models.chronos import ChronosWrapper
from src.models.moment import MOMENTWrapper
from src.utils.device import get_device

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


def _parse_args() -> argparse.Namespace:
    """평가 스크립트 CLI 인자를 파싱.

    Args:
        None.

    Returns:
        파싱된 argparse 네임스페이스.

    Raises:
        SystemExit: 필수 인자 누락 등 파싱 실패 시.
    """

    parser = argparse.ArgumentParser(description="TSFM 체크포인트 평가 스크립트")
    parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
    parser.add_argument(
        "--model", type=str, required=True, help="모델 이름 (chronos, moment)"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="데이터 설정 이름 (예: ett_m1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="실행 디바이스 (auto, cpu, cuda:0 등)",
    )
    return parser.parse_args()


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


def _load_model_cfg(project_root: Path, model_name: str) -> DictConfig:
    """모델 설정 YAML을 로드.

    Args:
        project_root: 프로젝트 루트 경로.
        model_name: 모델 설정 파일 이름(확장자 제외).

    Returns:
        모델 DictConfig.

    Raises:
        FileNotFoundError: 설정 파일이 없을 때.
    """

    model_cfg_path = project_root / "configs" / "model" / f"{model_name}.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"모델 설정 파일을 찾을 수 없습니다: {model_cfg_path}")
    return OmegaConf.load(model_cfg_path)


def _load_data_cfg(project_root: Path, data_name: str) -> DictConfig:
    """데이터 설정 YAML을 로드.

    Args:
        project_root: 프로젝트 루트 경로.
        data_name: 데이터 설정 파일 이름(확장자 제외).

    Returns:
        데이터 DictConfig.

    Raises:
        FileNotFoundError: 설정 파일이 없을 때.
    """

    data_cfg_path = project_root / "configs" / "data" / f"{data_name}.yaml"
    if not data_cfg_path.exists():
        raise FileNotFoundError(f"데이터 설정 파일을 찾을 수 없습니다: {data_cfg_path}")
    return OmegaConf.load(data_cfg_path)


def _build_wrapper(model_cfg: DictConfig) -> ChronosWrapper | MOMENTWrapper:
    """모델 설정에 맞는 래퍼를 생성하고 로드.

    Args:
        model_cfg: 모델 DictConfig.

    Returns:
        로드 완료된 모델 래퍼.

    Raises:
        ValueError: 지원하지 않는 모델 이름일 때.
    """

    if str(model_cfg.name) == "chronos":
        wrapper: ChronosWrapper | MOMENTWrapper = ChronosWrapper(model_cfg)
    elif str(model_cfg.name) == "moment":
        wrapper = MOMENTWrapper(model_cfg)
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_cfg.name}")

    wrapper.load()
    return wrapper


def _apply_adaptation_from_checkpoint(
    wrapper: ChronosWrapper | MOMENTWrapper,
    adaptation_method: str,
    adaptation_config: dict[str, Any],
    model_cfg: DictConfig,
) -> None:
    """체크포인트 메타데이터 기반으로 적응 모듈 구조를 복원.

    Args:
        wrapper: 모델 래퍼.
        adaptation_method: 적응 방법.
        adaptation_config: 적응 설정 딕셔너리.
        model_cfg: 모델 설정.

    Returns:
        None.

    Raises:
        ValueError: 지원하지 않는 적응 방법일 때.
    """

    backbone = wrapper.get_backbone()

    if adaptation_method == "zero_shot":
        return

    if adaptation_method == "head_only":
        _ = apply_head_only(backbone)
        return

    if adaptation_method == "lora":
        locus = LoRALocus(str(adaptation_config.get("locus", "attn_all")))
        target_modules_obj = adaptation_config.get("target_modules")
        target_modules: list[str] | None = None
        if target_modules_obj is not None:
            target_modules = [str(value) for value in target_modules_obj]

        lora_cfg = LoRAAdaptationConfig(
            rank=int(adaptation_config.get("rank", 8)),
            alpha=int(adaptation_config.get("alpha", 16)),
            dropout=float(adaptation_config.get("dropout", 0.05)),
            locus=locus,
            target_modules=target_modules,
            task_type=str(adaptation_config.get("task_type", "SEQ_2_SEQ_LM")),
            layers=str(adaptation_config.get("layers", "all")),
            num_layers=int(model_cfg.get("num_layers", 12)),
        )
        adapted = apply_lora(backbone, lora_cfg)
        wrapper.backbone = adapted
        if hasattr(wrapper, "model") and getattr(wrapper, "model", None) is not None:
            model_obj = getattr(wrapper, "model")
            if hasattr(model_obj, "encoder"):
                setattr(model_obj, "encoder", adapted)
        return

    if adaptation_method == "prefix_tuning":
        prefix_cfg = PrefixAdaptationConfig(
            num_virtual_tokens=int(adaptation_config.get("num_virtual_tokens", 32)),
            task_type=str(adaptation_config.get("task_type", "SEQ_2_SEQ_LM")),
        )
        adapted = apply_prefix_tuning(backbone, prefix_cfg)
        wrapper.backbone = adapted
        if hasattr(wrapper, "model") and getattr(wrapper, "model", None) is not None:
            model_obj = getattr(wrapper, "model")
            if hasattr(model_obj, "encoder"):
                setattr(model_obj, "encoder", adapted)
        return

    if adaptation_method == "full_fine_tuning":
        for parameter in backbone.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"지원하지 않는 adaptation.method 입니다: {adaptation_method}")


def _build_test_loader(
    data_cfg: DictConfig, batch_size: int = 32
) -> DataLoader[dict[str, torch.Tensor]]:
    """데이터 설정으로 테스트 DataLoader를 생성.

    Args:
        data_cfg: 데이터 DictConfig.
        batch_size: 배치 크기.

    Returns:
        테스트 DataLoader.

    Raises:
        ValueError: 현재 스크립트에서 지원하지 않는 데이터셋일 때.
    """

    if not str(data_cfg.name).startswith("ett_"):
        raise ValueError(
            f"현재 평가 스크립트는 ETT 데이터셋만 지원합니다. 현재: {data_cfg.name}"
        )

    ett_cfg = ETTConfig(
        dataset=str(data_cfg.dataset),
        path=str(data_cfg.path),
        target_col=str(data_cfg.target_col),
        context_length=int(data_cfg.context_length),
        prediction_length=int(data_cfg.prediction_length),
        train_ratio=float(data_cfg.train_ratio),
        val_ratio=float(data_cfg.val_ratio),
        test_ratio=float(data_cfg.test_ratio),
    )
    _, _, test_dataset = load_ett(ett_cfg)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )


def _evaluate(
    wrapper: ChronosWrapper | MOMENTWrapper,
    test_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    adaptation_method: str,
    prediction_length: int,
) -> dict[str, float]:
    """테스트 로더 전체에 대해 메트릭을 계산.

    Args:
        wrapper: 모델 래퍼.
        test_loader: 테스트 DataLoader.
        device: 실행 디바이스.
        adaptation_method: 적응 방식.
        prediction_length: 예측 길이.

    Returns:
        계산된 메트릭 딕셔너리.

    Raises:
        ValueError: 테스트 로더가 비어 있을 때.
    """

    wrapper.eval()
    if len(test_loader) == 0:
        raise ValueError("테스트 DataLoader가 비어 있습니다.")

    preds_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in test_loader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)

            if adaptation_method == "zero_shot":
                pred = wrapper.predict(
                    context=context, prediction_length=prediction_length
                )
            else:
                outputs = wrapper(context=context, target=target)
                pred = outputs["pred"]

            preds_list.append(pred.detach().to("cpu"))
            targets_list.append(target.detach().to("cpu"))

    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return compute_metrics(pred=preds, target=targets)


def main() -> None:
    """체크포인트를 로드해 테스트셋 평가를 실행.

    Args:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: 체크포인트 파일을 찾지 못했을 때.
        RuntimeError: 체크포인트 로딩 실패 시.
    """

    _setup_logging()
    args = _parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}"
        )

    project_root = Path(__file__).resolve().parents[1]
    model_cfg = _load_model_cfg(project_root=project_root, model_name=str(args.model))
    data_cfg = _load_data_cfg(project_root=project_root, data_name=str(args.data))

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(str(args.device))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("체크포인트 형식이 올바르지 않습니다.")

    adaptation_method = str(checkpoint.get("adaptation_method", "zero_shot"))
    adaptation_cfg_obj = checkpoint.get("adaptation_config", {})
    adaptation_cfg = adaptation_cfg_obj if isinstance(adaptation_cfg_obj, dict) else {}

    wrapper = _build_wrapper(model_cfg)
    _apply_adaptation_from_checkpoint(
        wrapper=wrapper,
        adaptation_method=adaptation_method,
        adaptation_config=adaptation_cfg,
        model_cfg=model_cfg,
    )

    # B4: train.py/run_expansion.py 통일 후 backbone_state_dict가 표준이지만,
    # 구버전 run_expansion.py가 저장한 체크포인트는 state_dict 키만 가지므로
    # 두 키 모두 허용한다.
    backbone_state = checkpoint.get("backbone_state_dict")
    if not isinstance(backbone_state, dict):
        backbone_state = checkpoint.get("state_dict")
    if not isinstance(backbone_state, dict):
        raise RuntimeError(
            "체크포인트에 backbone_state_dict 또는 state_dict가 없습니다."
        )

    load_result = wrapper.get_backbone().load_state_dict(backbone_state, strict=False)
    logger.info(
        "체크포인트 로드 완료: missing_keys=%d, unexpected_keys=%d",
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )

    wrapper.to(device)
    test_loader = _build_test_loader(data_cfg=data_cfg, batch_size=32)

    prediction_length = int(
        checkpoint.get("prediction_length", data_cfg.prediction_length)
    )
    metrics = _evaluate(
        wrapper=wrapper,
        test_loader=test_loader,
        device=device,
        adaptation_method=adaptation_method,
        prediction_length=prediction_length,
    )
    logger.info("평가 결과 (%s): %s", checkpoint_path, metrics)


if __name__ == "__main__":
    main()
