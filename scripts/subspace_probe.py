from __future__ import annotations

import argparse
from importlib import import_module
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Protocol, TypeAlias, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class _WrapperProtocol(Protocol):
    def get_backbone(self) -> nn.Module: ...

    def predict(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor: ...

    def __call__(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...

    def eval(self) -> None: ...

    def to(self, device: torch.device) -> None: ...


class _CKAAnalyzerProtocol(Protocol):
    def __init__(self, model: nn.Module, layer_patterns: list[str] | None = None) -> None: ...

    def register_hooks(self) -> None: ...

    def remove_hooks(self) -> None: ...

    def get_activations(self) -> dict[str, torch.Tensor]: ...

    def clear_activations(self) -> None: ...

    def compare_representations(
        self,
        activations_before: dict[str, torch.Tensor],
        activations_after: dict[str, torch.Tensor],
        kernel: str = "linear",
    ) -> dict[str, float]: ...


CreateWrapperFn: TypeAlias = Callable[[str], _WrapperProtocol]
CollateFn: TypeAlias = Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]
LoadDomainDataFn: TypeAlias = Callable[
    [str, int, int],
    tuple[
        Dataset[dict[str, torch.Tensor]],
        Dataset[dict[str, torch.Tensor]],
        Dataset[dict[str, torch.Tensor]],
    ],
]
GetDeviceFn: TypeAlias = Callable[[], torch.device]
class _CKAFnProtocol(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel: str = "linear",
    ) -> float: ...

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

run_expansion_module = import_module("scripts.run_expansion")
cka_module = import_module("src.evaluation.cka")
device_module = import_module("src.utils.device")

MODEL_CONFIGS = cast(dict[str, dict[str, object]], getattr(run_expansion_module, "MODEL_CONFIGS"))
_collate_batch = cast(CollateFn, getattr(run_expansion_module, "_collate_batch"))
_create_wrapper = cast(CreateWrapperFn, getattr(run_expansion_module, "_create_wrapper"))
_load_domain_data = cast(LoadDomainDataFn, getattr(run_expansion_module, "_load_domain_data"))
CKAAnalyzer = cast(type[_CKAAnalyzerProtocol], getattr(cka_module, "CKAAnalyzer"))
cka = cast(_CKAFnProtocol, getattr(cka_module, "cka"))
get_device = cast(GetDeviceFn, getattr(device_module, "get_device"))

LAYER_PATTERNS: list[str] = ["encoder", "decoder", "layer", "block", "attention"]

PROBE_TYPE_MAP: dict[tuple[str, str, str], str] = {
    # 기존 success/failure 셀
    ("chronos", "head_only", "finance"): "success",
    ("chronos", "lora", "ett_m1"): "failure",
    ("moirai", "adapter", "smd"): "success",
    ("moirai", "zero_shot", "ett_m1"): "failure",
    ("moment", "adapter", "smd"): "success",
    ("moment", "zero_shot", "ett_m1"): "failure",
    ("timesfm", "lora", "smd"): "success",
    ("timesfm", "zero_shot", "ett_m1"): "failure",
    # PhysioNet 셀 (결과 확인 후 분류)
    ("chronos", "adapter", "physionet"): "success",
    ("chronos", "lora", "physionet"): "probe",
    ("moirai", "adapter", "physionet"): "probe",
    ("moirai", "full_fine_tuning", "physionet"): "probe",
    ("moment", "lora", "physionet"): "probe",
    ("moment", "adapter", "physionet"): "probe",
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapted vs frozen subspace probe")
    _ = parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/pivot_subspace",
        help="체크포인트 루트 디렉토리",
    )
    _ = parser.add_argument(
        "--output_dir",
        type=str,
        default="results/subspace_probe",
        help="분석 결과 저장 디렉토리",
    )
    _ = parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    _ = parser.add_argument(
        "--kernel",
        type=str,
        choices=["linear", "rbf"],
        default="linear",
        help="CKA 커널 종류",
    )
    return parser.parse_args()


def _scan_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        logger.info("체크포인트 디렉토리가 없습니다: %s", checkpoint_dir)
        return []

    paths = sorted(checkpoint_dir.rglob("*.pt"))
    if not paths:
        logger.info("체크포인트 파일이 없습니다: %s", checkpoint_dir)
    return paths


def _load_checkpoint(path: Path) -> dict[str, object]:
    payload_obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload_obj, dict):
        raise TypeError(f"체크포인트 형식이 dict가 아닙니다: {path}")

    payload: dict[str, object] = dict(payload_obj)
    required_keys = ["experiment_id", "model", "method", "domain", "state_dict"]
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(
            f"체크포인트 필수 키가 없습니다: {missing_keys} (path={path})"
        )
    return payload


def _build_probe_batch(
    model_name: str,
    domain: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    model_cfg = MODEL_CONFIGS.get(model_name)
    if model_cfg is None:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    context_length_obj = model_cfg.get("context_length")
    prediction_length_obj = model_cfg.get("prediction_length")
    if not isinstance(context_length_obj, int) or not isinstance(prediction_length_obj, int):
        raise ValueError(f"모델 길이 설정이 올바르지 않습니다: model={model_name}")

    _, _, test_ds = _load_domain_data(domain, context_length_obj, prediction_length_obj)
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError(f"배치를 생성할 수 없습니다: domain={domain}") from exc
    return batch


def _run_wrapper_for_hooks(
    wrapper: _WrapperProtocol,
    method: str,
    context: torch.Tensor,
    target: torch.Tensor,
    prediction_length: int,
) -> None:
    with torch.no_grad():
        if method == "zero_shot":
            _ = wrapper.predict(context=context, prediction_length=prediction_length)
        else:
            _ = wrapper(context=context, target=target)


def _capture_activations(
    wrapper: _WrapperProtocol,
    method: str,
    batch: dict[str, torch.Tensor],
    prediction_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    analyzer = CKAAnalyzer(wrapper.get_backbone(), layer_patterns=LAYER_PATTERNS)
    analyzer.register_hooks()

    wrapper.eval()
    wrapper.to(device)

    context = batch["context"].to(device)
    target = batch["target"].to(device)
    _run_wrapper_for_hooks(wrapper, method, context, target, prediction_length)

    activations = analyzer.get_activations()
    analyzer.remove_hooks()
    analyzer.clear_activations()
    return activations


def _compute_update_norm_per_layer(
    layer_names: list[str],
    frozen_state: dict[str, torch.Tensor],
    adapted_state: dict[str, torch.Tensor],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for layer_name in layer_names:
        sq_sum = 0.0
        prefix = f"{layer_name}."
        for key, adapted_param in adapted_state.items():
            if key != layer_name and not key.startswith(prefix):
                continue
            frozen_param = frozen_state.get(key)
            if frozen_param is None or frozen_param.shape != adapted_param.shape:
                continue
            delta = (adapted_param.detach().float() - frozen_param.detach().float()).reshape(-1)
            sq_sum += float(torch.dot(delta, delta).item())
        result[layer_name] = float(np.sqrt(sq_sum))
    return result


def _compute_update_norm_concentration(
    layer_names: list[str],
    update_norm_per_layer: dict[str, float],
) -> dict[str, float]:
    if not layer_names:
        return {"early_layers": 0.0, "late_layers": 0.0}

    split_idx = len(layer_names) // 2
    if split_idx == 0:
        split_idx = 1

    early_layers = layer_names[:split_idx]
    late_layers = layer_names[split_idx:]

    early_sum = float(sum(update_norm_per_layer.get(name, 0.0) for name in early_layers))
    late_sum = float(sum(update_norm_per_layer.get(name, 0.0) for name in late_layers))
    total = early_sum + late_sum
    if total <= 0.0:
        return {"early_layers": 0.0, "late_layers": 0.0}
    return {
        "early_layers": early_sum / total,
        "late_layers": late_sum / total,
    }


def _save_heatmap(
    experiment_id: str,
    layer_cka: dict[str, float],
    concentration: dict[str, float],
    output_dir: Path,
) -> None:
    if not layer_cka:
        logger.warning("CKA 레이어가 없어 히트맵을 건너뜁니다: %s", experiment_id)
        return

    layer_names = sorted(layer_cka.keys())
    values = np.array([layer_cka[name] for name in layer_names], dtype=np.float32)[None, :]

    fig_width = max(10.0, 0.28 * len(layer_names))
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))

    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks([0])
    ax.set_yticklabels(["CKA"])
    ax.set_xlabel("Layer")
    ax.set_title(f"Layer-wise CKA: {experiment_id}")

    split_idx = len(layer_names) // 2
    if len(layer_names) >= 2:
        ax.axvline(x=split_idx - 0.5, color="white", linestyle="--", linewidth=1.2)

    tick_step = max(1, len(layer_names) // 12)
    tick_positions = list(range(0, len(layer_names), tick_step))
    tick_labels = [layer_names[idx].split(".")[-1] for idx in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    early_ratio = concentration.get("early_layers", 0.0)
    late_ratio = concentration.get("late_layers", 0.0)
    ax.text(
        0.01,
        1.18,
        f"update concentration | early={early_ratio:.3f}, late={late_ratio:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
    )

    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02, label="CKA")
    fig.tight_layout()

    pdf_path = output_dir / f"cka_heatmap_{experiment_id}.pdf"
    png_path = output_dir / f"cka_heatmap_{experiment_id}.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _analyze_checkpoint(
    payload: dict[str, object],
    batch: dict[str, torch.Tensor],
    kernel: str,
    device: torch.device,
) -> dict[str, object]:
    experiment_id_obj = payload.get("experiment_id")
    model_name_obj = payload.get("model")
    method_obj = payload.get("method")
    domain_obj = payload.get("domain")
    state_dict_obj = payload.get("state_dict")

    if not isinstance(experiment_id_obj, str):
        raise ValueError("experiment_id가 문자열이 아닙니다.")
    if not isinstance(model_name_obj, str):
        raise ValueError("model이 문자열이 아닙니다.")
    if not isinstance(method_obj, str):
        raise ValueError("method가 문자열이 아닙니다.")
    if not isinstance(domain_obj, str):
        raise ValueError("domain이 문자열이 아닙니다.")
    if not isinstance(state_dict_obj, dict):
        raise ValueError("state_dict 형식이 dict가 아닙니다.")

    experiment_id = experiment_id_obj
    model_name = model_name_obj
    method = method_obj
    domain = domain_obj
    adapted_state_dict: dict[str, torch.Tensor] = {}
    for key_obj, value_obj in state_dict_obj.items():
        if not isinstance(key_obj, str) or not isinstance(value_obj, torch.Tensor):
            raise ValueError("state_dict 키/값 형식이 올바르지 않습니다.")
        adapted_state_dict[key_obj] = value_obj

    model_cfg = MODEL_CONFIGS.get(model_name)
    if model_cfg is None:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    prediction_length_obj = model_cfg.get("prediction_length")
    if not isinstance(prediction_length_obj, int):
        raise ValueError(f"prediction_length 설정이 유효하지 않습니다: {model_name}")

    adapted_wrapper = _create_wrapper(model_name)
    frozen_wrapper = _create_wrapper(model_name)

    load_result = adapted_wrapper.get_backbone().load_state_dict(adapted_state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning("state_dict missing keys: %s | %s", experiment_id, load_result.missing_keys)
    if load_result.unexpected_keys:
        logger.warning(
            "state_dict unexpected keys: %s | %s", experiment_id, load_result.unexpected_keys
        )

    frozen_activations = _capture_activations(
        wrapper=frozen_wrapper,
        method=method,
        batch=batch,
        prediction_length=prediction_length_obj,
        device=device,
    )
    adapted_activations = _capture_activations(
        wrapper=adapted_wrapper,
        method=method,
        batch=batch,
        prediction_length=prediction_length_obj,
        device=device,
    )

    cka_analyzer = CKAAnalyzer(adapted_wrapper.get_backbone(), layer_patterns=LAYER_PATTERNS)
    layer_cka = cka_analyzer.compare_representations(
        frozen_activations,
        adapted_activations,
        kernel=kernel,
    )

    common_layers = sorted(layer_cka.keys())
    if common_layers:
        first_layer = common_layers[0]
        before = frozen_activations[first_layer].reshape(-1, frozen_activations[first_layer].shape[-1])
        after = adapted_activations[first_layer].reshape(-1, adapted_activations[first_layer].shape[-1])
        min_n = min(before.shape[0], after.shape[0], 1000)
        _ = cka(before[:min_n], after[:min_n], kernel=kernel)

    frozen_state = {
        key: value.detach().cpu()
        for key, value in frozen_wrapper.get_backbone().state_dict().items()
    }
    adapted_state = {
        key: value.detach().cpu()
        for key, value in adapted_wrapper.get_backbone().state_dict().items()
    }

    update_norm_per_layer = _compute_update_norm_per_layer(
        layer_names=common_layers,
        frozen_state=frozen_state,
        adapted_state=adapted_state,
    )
    concentration = _compute_update_norm_concentration(common_layers, update_norm_per_layer)
    mean_cka = float(np.mean(list(layer_cka.values()))) if layer_cka else 0.0

    probe_key = (model_name, method, domain)
    probe_type = PROBE_TYPE_MAP.get(probe_key, "unknown")

    return {
        "experiment_id": experiment_id,
        "model": model_name,
        "method": method,
        "domain": domain,
        "probe_type": probe_type,
        "layer_cka": {name: float(layer_cka[name]) for name in common_layers},
        "mean_cka": mean_cka,
        "update_norm_per_layer": {
            name: float(update_norm_per_layer[name]) for name in common_layers
        },
        "update_norm_concentration": {
            "early_layers": float(concentration["early_layers"]),
            "late_layers": float(concentration["late_layers"]),
        },
    }


def main() -> None:
    _setup_logging()
    args = _parse_args()

    checkpoint_dir = Path(str(args.checkpoint_dir))
    output_dir = Path(str(args.output_dir))
    batch_size = int(args.batch_size)
    kernel = str(args.kernel)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = _scan_checkpoints(checkpoint_dir)
    if not checkpoint_paths:
        return

    device = get_device()
    logger.info("분석 디바이스: %s", device)

    cached_batches: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}
    results: list[dict[str, object]] = []

    for ckpt_path in checkpoint_paths:
        try:
            payload = _load_checkpoint(ckpt_path)
            model_name_obj = payload.get("model")
            domain_obj = payload.get("domain")
            if not isinstance(model_name_obj, str) or not isinstance(domain_obj, str):
                raise ValueError("체크포인트의 model/domain 정보가 문자열이 아닙니다.")

            batch_key = (model_name_obj, domain_obj, batch_size)
            if batch_key not in cached_batches:
                cached_batches[batch_key] = _build_probe_batch(
                    model_name=model_name_obj,
                    domain=domain_obj,
                    batch_size=batch_size,
                )

            result = _analyze_checkpoint(
                payload=payload,
                batch=cached_batches[batch_key],
                kernel=kernel,
                device=device,
            )
            results.append(result)

            experiment_id_obj = result.get("experiment_id")
            layer_cka_obj = result.get("layer_cka")
            concentration_obj = result.get("update_norm_concentration")
            if not isinstance(experiment_id_obj, str):
                raise ValueError("결과 experiment_id 형식이 잘못되었습니다.")
            if not isinstance(layer_cka_obj, dict):
                raise ValueError("결과 layer_cka 형식이 잘못되었습니다.")
            if not isinstance(concentration_obj, dict):
                raise ValueError("결과 update_norm_concentration 형식이 잘못되었습니다.")
            layer_cka: dict[str, float] = {
                str(name): float(value)
                for name, value in layer_cka_obj.items()
            }
            concentration: dict[str, float] = {
                str(name): float(value)
                for name, value in concentration_obj.items()
            }

            _save_heatmap(
                experiment_id=experiment_id_obj,
                layer_cka=layer_cka,
                concentration=concentration,
                output_dir=output_dir,
            )
            logger.info("분석 완료: %s", result["experiment_id"])
        except Exception as exc:
            logger.warning("체크포인트 분석 실패, 건너뜀: %s (%s)", ckpt_path, exc)

    json_path = output_dir / "cka_results.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    logger.info("결과 저장 완료: %s (count=%d)", json_path, len(results))


def generate_paper_figure(results_path: str, output_path: str) -> None:
    """논문용 combined CKA heatmap 생성.

    Args:
        results_path: cka_results.json 경로.
        output_path: 출력 PDF 경로.

    Returns:
        None.

    Raises:
        FileNotFoundError: 결과 파일이 없을 때.
    """
    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"결과 파일이 없습니다: {results_path}")

    with results_file.open("r", encoding="utf-8") as f:
        all_results: list[dict[str, object]] = json.load(f)

    successes = [r for r in all_results if r.get("probe_type") == "success"]
    failures = [r for r in all_results if r.get("probe_type") == "failure"]

    if not successes and not failures:
        logger.warning("success/failure 결과가 없습니다.")
        return

    n_rows = max(len(successes), len(failures))
    if n_rows == 0:
        return

    fig, axes = plt.subplots(
        n_rows, 2, figsize=(16, 2.0 * n_rows + 1.5),
        squeeze=False,
    )
    fig.suptitle("Layer-wise CKA: Success (left) vs Failure (right)", fontsize=14, y=0.98)

    for col_idx, (group, group_label) in enumerate(
        [(successes, "Success"), (failures, "Failure")]
    ):
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            if row_idx >= len(group):
                ax.set_visible(False)
                continue

            result = group[row_idx]
            layer_cka_obj = result.get("layer_cka", {})
            if not isinstance(layer_cka_obj, dict):
                ax.set_visible(False)
                continue

            layer_names = sorted(layer_cka_obj.keys())
            values = np.array(
                [float(layer_cka_obj[n]) for n in layer_names], dtype=np.float32
            )[None, :]

            im = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
            exp_id = str(result.get("experiment_id", ""))
            mean_cka_val = result.get("mean_cka", 0.0)
            ax.set_title(f"{exp_id} (mean CKA={float(mean_cka_val):.3f})", fontsize=9)
            ax.set_yticks([])

            tick_step = max(1, len(layer_names) // 8)
            tick_positions = list(range(0, len(layer_names), tick_step))
            tick_labels = [layer_names[i].split(".")[-1] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="CKA")
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("논문 figure 저장: %s", out)


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "--paper-figure":
        _setup_logging()
        generate_paper_figure(
            results_path=_sys.argv[2] if len(_sys.argv) > 2 else "results/mechanism_analysis/cka_results.json",
            output_path=_sys.argv[3] if len(_sys.argv) > 3 else "results/expansion_analysis/mechanism_cka_heatmap.pdf",
        )
    else:
        main()
