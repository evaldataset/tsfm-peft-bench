from __future__ import annotations

# pyright: reportMissingImports=false

"""FIM 지문 계산 스크립트.

3개 모델 × 4개 도메인 = 12 조합에 대해 FIM 대각선을 계산하고,
레이어별 프로파일과 도메인 간 거리 행렬을 저장한다.

출력:
    results/fim_fingerprints.json  — 지문 데이터 및 거리 행렬
    results/expansion_analysis/fig8_fim_heatmap.pdf  — 도메인 거리 히트맵 (3 패널)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.ett import ETTConfig, load_ett
from src.data.finance import FinanceConfig, load_finance
from src.data.physionet import PhysioNetConfig, load_physionet
from src.data.smd import SMDConfig, load_smd
from src.evaluation.fim_fingerprint import (
    compute_fim_diagonal,
    fim_distance,
    fim_to_layer_profile,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_MODELS: list[str] = ["chronos", "moment", "moirai"]
_DOMAINS: list[str] = ["ett_m1", "finance", "smd", "physionet"]
_N_SAMPLES: int = 50
_FIM_METRIC: str = "cosine"

_RESULTS_DIR = _PROJECT_ROOT / "results"
_FIM_OUTPUT_PATH = _RESULTS_DIR / "fim_fingerprints.json"
_HEATMAP_DIR = _RESULTS_DIR / "expansion_analysis"


# ---------------------------------------------------------------------------
# 모델 로더
# ---------------------------------------------------------------------------

class _SimpleNamespace:
    """getattr 기반 단순 설정 네임스페이스."""

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name: str) -> object:
        return None


def _load_model(model_name: str) -> nn.Module:
    """모델 이름으로 래퍼를 로드하고 백본 nn.Module을 반환.

    Args:
        model_name: ``"chronos"``, ``"moment"``, ``"moirai"`` 중 하나.

    Returns:
        로드된 nn.Module (백본).

    Raises:
        ValueError: 지원하지 않는 모델 이름일 때.
        ImportError: 필요한 패키지가 없을 때.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "chronos":
        from src.models.chronos import ChronosWrapper

        cfg = _SimpleNamespace(
            hf_id="amazon/chronos-t5-small",
            context_length=512,
            prediction_length=96,
            torch_dtype="bfloat16",
            device_map=device,
        )
        wrapper = ChronosWrapper(cfg)
        wrapper.load()
        return wrapper.get_backbone()

    if model_name == "moment":
        from src.models.moment import MOMENTWrapper

        cfg = _SimpleNamespace(
            hf_id="AutonLab/MOMENT-1-large",
            context_length=512,
            prediction_length=96,
            freeze_encoder=False,
            freeze_embedder=True,
            freeze_head=False,
            head_dropout=0.1,
        )
        wrapper = MOMENTWrapper(cfg)
        wrapper.load()
        return wrapper.get_backbone()

    if model_name == "moirai":
        from src.models.moirai import MoiraiWrapper

        cfg = _SimpleNamespace(
            hf_id="Salesforce/moirai-1.1-R-base",
            context_length=512,
            prediction_length=96,
            patch_size="auto",
            num_samples=20,
        )
        wrapper = MoiraiWrapper(cfg)
        wrapper.load()
        return wrapper.get_backbone()

    raise ValueError(f"지원하지 않는 모델: {model_name}")


def _load_wrapper(model_name: str) -> Any:
    """forward() 인터페이스를 가진 래퍼 객체를 반환.

    FIM 계산에는 wrapper.forward(context, target) -> {"loss": ..., "pred": ...}
    인터페이스가 필요하다.

    Args:
        model_name: 모델 이름.

    Returns:
        래퍼 인스턴스 (nn.Module로 취급 가능).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "chronos":
        from src.models.chronos import ChronosWrapper

        cfg = _SimpleNamespace(
            hf_id="amazon/chronos-t5-small",
            context_length=512,
            prediction_length=96,
            torch_dtype="bfloat16",
            device_map=device,
        )
        wrapper = ChronosWrapper(cfg)
        wrapper.load()
        return wrapper

    if model_name == "moment":
        from src.models.moment import MOMENTWrapper

        cfg = _SimpleNamespace(
            hf_id="AutonLab/MOMENT-1-large",
            context_length=512,
            prediction_length=96,
            freeze_encoder=False,
            freeze_embedder=True,
            freeze_head=False,
            head_dropout=0.1,
        )
        wrapper = MOMENTWrapper(cfg)
        wrapper.load()
        return wrapper

    if model_name == "moirai":
        from src.models.moirai import MoiraiWrapper

        cfg = _SimpleNamespace(
            hf_id="Salesforce/moirai-1.1-R-base",
            context_length=512,
            prediction_length=96,
            patch_size="auto",
            num_samples=20,
        )
        wrapper = MoiraiWrapper(cfg)
        wrapper.load()
        return wrapper

    raise ValueError(f"지원하지 않는 모델: {model_name}")


# ---------------------------------------------------------------------------
# 데이터셋 로더
# ---------------------------------------------------------------------------

def _load_train_dataset(domain_name: str) -> Any:
    """도메인 이름으로 학습 데이터셋을 로드.

    Args:
        domain_name: ``"ett_m1"``, ``"finance"``, ``"smd"``, ``"physionet"`` 중 하나.

    Returns:
        학습용 Dataset 인스턴스.

    Raises:
        ValueError: 지원하지 않는 도메인 이름일 때.
    """
    if domain_name == "ett_m1":
        cfg = ETTConfig(
            dataset="ETTm1",
            path=str(_PROJECT_ROOT / "data/ETT-small/ETTm1.csv"),
            context_length=512,
            prediction_length=96,
        )
        train_ds, _, _ = load_ett(cfg)
        return train_ds

    # FIM 계산에서는 모든 도메인에서 모델 wrapper와 동일한 context/prediction
    # length를 사용해야 한다 (그렇지 않으면 forward pass shape mismatch).
    if domain_name == "finance":
        cfg = FinanceConfig(
            dataset="ExchangeRate",
            path=str(_PROJECT_ROOT / "data/exchange_rate/exchange_rate.csv"),
            context_length=512,
            prediction_length=96,
        )
        train_ds, _, _ = load_finance(cfg)
        return train_ds

    if domain_name == "smd":
        cfg = SMDConfig(
            dataset="SMD",
            path=str(_PROJECT_ROOT / "data/SMD"),
            context_length=512,
            prediction_length=96,
        )
        train_ds, _, _ = load_smd(cfg)
        return train_ds

    if domain_name == "physionet":
        # PhysioNet patient files are typically too short for context=512+pred=96.
        # We try with the wrapper's context length and the script naturally
        # skips this cell if dataset is empty.
        cfg = PhysioNetConfig(
            dataset="PhysioNet",
            data_dir=str(_PROJECT_ROOT / "data/physionet"),
            context_length=512,
            prediction_length=96,
        )
        train_ds, _, _ = load_physionet(cfg)
        return train_ds

    raise ValueError(f"지원하지 않는 도메인: {domain_name}")


# ---------------------------------------------------------------------------
# FIM 지문 계산용 래퍼 어댑터 (nn.Module로 감싸기)
# ---------------------------------------------------------------------------

class _WrapperModule(nn.Module):
    """Wrapper 객체를 nn.Module로 감싸 FIM 계산 인터페이스에 맞춤.

    Args:
        wrapper: ChronosWrapper / MOMENTWrapper / MoiraiWrapper 인스턴스.

    Returns:
        _WrapperModule 인스턴스.
    """

    def __init__(self, wrapper: Any) -> None:
        super().__init__()
        # 백본 nn.Module을 서브모듈로 등록하여 parameters()가 올바르게 작동하도록 함
        if model_name := getattr(wrapper, "backbone", None):
            self._backbone = model_name
        elif hasattr(wrapper, "model") and isinstance(
            getattr(wrapper, "model", None), nn.Module
        ):
            self._backbone = getattr(wrapper, "model")
        elif hasattr(wrapper, "module") and isinstance(
            getattr(wrapper, "module", None), nn.Module
        ):
            self._backbone = getattr(wrapper, "module")
        else:
            # MoiraiWrapper: get_backbone() 사용
            try:
                self._backbone = wrapper.get_backbone()
            except Exception:
                self._backbone = nn.Linear(1, 1)  # 폴백

        self._wrapper = wrapper

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        """백본 파라미터 반환."""
        return self._backbone.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):  # type: ignore[override]
        """백본 named_parameters 반환."""
        return self._backbone.named_parameters(prefix=prefix, recurse=recurse)

    def forward(  # type: ignore[override]
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """래퍼의 forward를 호출."""
        return self._wrapper.forward(context=context, target=target)

    def train(self, mode: bool = True) -> _WrapperModule:
        """학습 모드 전환."""
        super().train(mode)
        self._backbone.train(mode)
        return self

    def to(self, *args: Any, **kwargs: Any) -> _WrapperModule:  # type: ignore[override]
        """디바이스/dtype 이동."""
        super().to(*args, **kwargs)
        return self


# ---------------------------------------------------------------------------
# 히트맵 생성
# ---------------------------------------------------------------------------

def _generate_heatmap(
    distance_data: dict[str, dict[str, float]],
    domains: list[str],
    output_path: Path,
) -> None:
    """도메인 간 FIM 거리 행렬 히트맵 생성 (모델별 3 패널).

    Args:
        distance_data: ``{model: {dom_a_vs_dom_b: dist}}`` 형태의 거리 데이터.
        domains: 도메인 이름 리스트.
        output_path: 저장할 PDF 경로.

    Returns:
        None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn 없음. 히트맵 생성 건너뜀.")
        return

    n_models = len(_MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    domain_labels = [d.replace("_", "\n") for d in domains]

    for ax, model_name in zip(axes, _MODELS):
        n_d = len(domains)
        mat = np.zeros((n_d, n_d), dtype=np.float32)
        model_dists = distance_data.get(model_name, {})

        for i, da in enumerate(domains):
            for j, db in enumerate(domains):
                if i == j:
                    mat[i, j] = 0.0
                else:
                    key = f"{da}_vs_{db}"
                    alt_key = f"{db}_vs_{da}"
                    mat[i, j] = model_dists.get(key, model_dists.get(alt_key, float("nan")))

        sns.heatmap(
            mat,
            ax=ax,
            xticklabels=domain_labels,
            yticklabels=domain_labels,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{model_name.capitalize()}\nFIM 도메인 거리 ({_FIM_METRIC})", fontsize=11)
        ax.set_xlabel("도메인")
        ax.set_ylabel("도메인")

    fig.suptitle(
        "Fig. 8: FIM 기반 도메인 유사도 행렬 (모델별)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("히트맵 저장: %s", output_path)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    """FIM 지문 계산 메인 함수."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("사용 디바이스: %s", device)

    fingerprints: dict[str, dict[str, Any]] = {}

    # 12 조합에 대해 FIM 계산
    for model_name in _MODELS:
        logger.info("===== 모델: %s =====", model_name)

        # 모델은 도메인 반복마다 새로 로드 (GPU 메모리 관리)
        for domain_name in _DOMAINS:
            key = f"{model_name}_{domain_name}"
            logger.info("--- %s ---", key)

            try:
                dataset = _load_train_dataset(domain_name)
            except FileNotFoundError as exc:
                logger.warning("데이터셋 로드 실패, 건너뜀: %s", exc)
                continue
            except Exception as exc:
                logger.warning("데이터셋 로드 오류, 건너뜀: %s", exc)
                continue

            try:
                wrapper = _load_wrapper(model_name)
            except Exception as exc:
                logger.warning("모델 로드 실패, 건너뜀: %s", exc)
                continue

            # 래퍼를 nn.Module로 감쌈
            wrapped_module = _WrapperModule(wrapper)

            try:
                fim_diag = compute_fim_diagonal(
                    model=wrapped_module,
                    dataset=dataset,
                    n_samples=_N_SAMPLES,
                    device=device,
                )
            except Exception as exc:
                logger.error("FIM 계산 실패 [%s]: %s", key, exc)
                continue

            # 레이어 프로파일
            layer_profile = fim_to_layer_profile(fim_diag, wrapped_module)

            total_norm = float(np.sqrt(np.sum(fim_diag ** 2)))
            n_params = int(len(fim_diag))

            fingerprints[key] = {
                "layer_profile": layer_profile,
                "total_fim_norm": total_norm,
                "n_params": n_params,
                # FIM 대각선 자체는 용량이 크므로 별도 npy 파일로 저장
            }

            # FIM 대각선 배열을 npy로 저장
            npy_path = _RESULTS_DIR / f"fim_{key}.npy"
            np.save(str(npy_path), fim_diag)
            logger.info(
                "FIM 저장: %s (총 노름=%.4f, 파라미터=%d)",
                npy_path,
                total_norm,
                n_params,
            )

            # GPU 메모리 해제
            del wrapper, wrapped_module
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # 도메인 간 거리 계산
    # ---------------------------------------------------------------------------
    distances: dict[str, dict[str, float]] = {}

    for model_name in _MODELS:
        distances[model_name] = {}
        for i, da in enumerate(_DOMAINS):
            for j, db in enumerate(_DOMAINS):
                if j <= i:
                    continue
                key_a = f"{model_name}_{da}"
                key_b = f"{model_name}_{db}"
                npy_a = _RESULTS_DIR / f"fim_{key_a}.npy"
                npy_b = _RESULTS_DIR / f"fim_{key_b}.npy"

                if not npy_a.exists() or not npy_b.exists():
                    logger.warning("FIM 파일 없음: %s 또는 %s", npy_a, npy_b)
                    continue

                fim_a = np.load(str(npy_a))
                fim_b = np.load(str(npy_b))

                dist = fim_distance(fim_a, fim_b, metric=_FIM_METRIC)
                pair_key = f"{da}_vs_{db}"
                distances[model_name][pair_key] = dist
                logger.info(
                    "거리 [%s] %s vs %s = %.4f",
                    model_name,
                    da,
                    db,
                    dist,
                )

    # ---------------------------------------------------------------------------
    # 결과 저장
    # ---------------------------------------------------------------------------
    output_data: dict[str, Any] = {
        "fingerprints": fingerprints,
        "distances": distances,
        "meta": {
            "n_samples": _N_SAMPLES,
            "metric": _FIM_METRIC,
            "models": _MODELS,
            "domains": _DOMAINS,
        },
    }

    with open(_FIM_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info("FIM 지문 결과 저장: %s", _FIM_OUTPUT_PATH)

    # ---------------------------------------------------------------------------
    # 히트맵 생성
    # ---------------------------------------------------------------------------
    heatmap_path = _HEATMAP_DIR / "fig8_fim_heatmap.pdf"
    _generate_heatmap(
        distance_data=distances,
        domains=_DOMAINS,
        output_path=heatmap_path,
    )

    logger.info("완료.")


if __name__ == "__main__":
    main()
