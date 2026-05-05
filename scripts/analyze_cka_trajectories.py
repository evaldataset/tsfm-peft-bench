"""CKA 궤적 분석 스크립트.

수집된 CKA 궤적 파일을 로드하여 발산 예측 임계값 분석 및 시각화 생성.
성공/실패 케이스 비교, 조기 경고 임계값 최적화.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# matplotlib/sklearn은 런타임에 임포트 (프로토콜 패턴)
_plt: Any = None
_sklearn_metrics: Any = None


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
    parser = argparse.ArgumentParser(description="CKA 궤적 분석")
    parser.add_argument(
        "--traj_dir",
        type=str,
        default="results/cka_trajectories",
        help="궤적 파일 디렉토리",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/expansion_analysis",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--analysis_out",
        type=str,
        default="results/cka_trajectory_analysis.json",
        help="분석 결과 JSON 저장 경로",
    )
    parser.add_argument(
        "--divergence_mae_threshold",
        type=float,
        default=50.0,
        help="발산 판정 MAE 임계값",
    )
    parser.add_argument(
        "--early_epoch",
        type=int,
        default=2,
        help="조기 경고 CKA를 측정할 에폭",
    )
    return parser.parse_args()


def _load_trajectories(traj_dir: Path) -> list[dict[str, Any]]:
    """궤적 디렉토리에서 JSON 파일 로드.

    Args:
        traj_dir: 궤적 파일 디렉토리.

    Returns:
        궤적 딕셔너리 리스트.
    """
    trajectories: list[dict[str, Any]] = []
    for json_path in sorted(traj_dir.glob("*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            trajectories.append(data)
            logger.info("로드: %s", json_path.name)
        except Exception as exc:
            logger.warning("파일 로드 실패 (%s): %s", json_path.name, exc)

    logger.info("총 %d 궤적 로드 완료.", len(trajectories))
    return trajectories


def _classify_trajectories(
    trajectories: list[dict[str, Any]],
    divergence_mae_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """궤적을 성공/실패로 분류.

    Args:
        trajectories: 궤적 딕셔너리 리스트.
        divergence_mae_threshold: 발산 판정 MAE 임계값.

    Returns:
        (success_list, failure_list) 튜플.
    """
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for traj in trajectories:
        mae = traj.get("final_mae", float("nan"))
        if mae != mae:  # NaN 체크
            logger.warning("NaN MAE 무시: %s", traj.get("model", "?"))
            continue
        if mae < divergence_mae_threshold:
            successes.append(traj)
        else:
            failures.append(traj)

    logger.info("성공: %d, 실패: %d", len(successes), len(failures))
    return successes, failures


def _get_cka_at_epoch(traj: dict[str, Any], target_epoch: int) -> float | None:
    """특정 에폭에서의 mean CKA 값을 반환.

    Args:
        traj: 궤적 딕셔너리.
        target_epoch: 조회할 에폭 번호.

    Returns:
        해당 에폭의 mean CKA 값, 없으면 None.
    """
    epochs_tracked = traj.get("epochs_tracked", [])
    mean_cka = traj.get("mean_cka_per_epoch", [])

    if not epochs_tracked or not mean_cka:
        return None

    # 가장 가까운 에폭 탐색
    closest_idx = None
    closest_diff = float("inf")
    for i, ep in enumerate(epochs_tracked):
        diff = abs(ep - target_epoch)
        if diff < closest_diff:
            closest_diff = diff
            closest_idx = i

    if closest_idx is None or closest_diff > 2:
        return None

    return float(mean_cka[closest_idx])


def _find_optimal_threshold(
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    early_epoch: int,
) -> dict[str, Any]:
    """CKA 임계값을 최적화 (F1 최대화).

    Args:
        successes: 성공 궤적 리스트.
        failures: 실패 궤적 리스트.

    Returns:
        임계값 분석 결과 딕셔너리.
    """
    # 성공=0, 실패=1로 레이블
    cka_values: list[float] = []
    labels: list[int] = []

    for traj in successes:
        val = _get_cka_at_epoch(traj, early_epoch)
        if val is not None:
            cka_values.append(val)
            labels.append(0)

    for traj in failures:
        val = _get_cka_at_epoch(traj, early_epoch)
        if val is not None:
            cka_values.append(val)
            labels.append(1)

    if len(cka_values) < 2:
        logger.warning("임계값 최적화에 충분한 데이터 없음.")
        return {
            "optimal_threshold": 0.5,
            "best_f1": float("nan"),
            "precision_at_threshold": float("nan"),
            "recall_at_threshold": float("nan"),
            "n_samples": len(cka_values),
            "mean_cka_success": float("nan"),
            "mean_cka_failure": float("nan"),
        }

    cka_arr = np.array(cka_values)
    label_arr = np.array(labels)

    success_cka = cka_arr[label_arr == 0]
    failure_cka = cka_arr[label_arr == 1]

    mean_cka_success = float(np.mean(success_cka)) if len(success_cka) > 0 else float("nan")
    mean_cka_failure = float(np.mean(failure_cka)) if len(failure_cka) > 0 else float("nan")

    # 임계값 그리드 탐색: CKA < threshold → 발산 예측 (label=1)
    thresholds = np.linspace(0.0, 1.0, 201)
    best_f1 = -1.0
    best_thresh = 0.5
    best_precision = float("nan")
    best_recall = float("nan")

    for thresh in thresholds:
        predicted = (cka_arr < thresh).astype(int)
        tp = int(np.sum((predicted == 1) & (label_arr == 1)))
        fp = int(np.sum((predicted == 1) & (label_arr == 0)))
        fn = int(np.sum((predicted == 0) & (label_arr == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            best_precision = precision
            best_recall = recall

    logger.info(
        "최적 임계값: %.3f, F1=%.4f, precision=%.4f, recall=%.4f",
        best_thresh, best_f1, best_precision, best_recall,
    )

    # CKA < 0.5 at early_epoch 기준 precision/recall
    fixed_thresh = 0.5
    predicted_fixed = (cka_arr < fixed_thresh).astype(int)
    tp_f = int(np.sum((predicted_fixed == 1) & (label_arr == 1)))
    fp_f = int(np.sum((predicted_fixed == 1) & (label_arr == 0)))
    fn_f = int(np.sum((predicted_fixed == 0) & (label_arr == 1)))
    prec_fixed = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
    rec_fixed = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0

    return {
        "optimal_threshold": best_thresh,
        "best_f1": float(best_f1),
        "precision_at_threshold": float(best_precision),
        "recall_at_threshold": float(best_recall),
        "precision_at_0_5": float(prec_fixed),
        "recall_at_0_5": float(rec_fixed),
        "n_samples": len(cka_values),
        "n_success": int(np.sum(label_arr == 0)),
        "n_failure": int(np.sum(label_arr == 1)),
        "mean_cka_success": mean_cka_success,
        "mean_cka_failure": mean_cka_failure,
    }


def _get_method_color(method: str) -> str:
    """적응 방법별 색상 매핑.

    Args:
        method: 적응 방법 이름.

    Returns:
        matplotlib 색상 문자열.
    """
    color_map: dict[str, str] = {
        "lora": "#e74c3c",
        "adapter": "#3498db",
        "prefix": "#2ecc71",
        "head_only": "#f39c12",
        "full_fine_tuning": "#9b59b6",
        "zero_shot": "#95a5a6",
    }
    return color_map.get(method, "#34495e")


def _plot_trajectories(
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    optimal_threshold: float,
    output_path: Path,
) -> None:
    """CKA 궤적 2-패널 시각화 생성.

    Args:
        successes: 성공 궤적 리스트.
        failures: 실패 궤적 리스트.
        optimal_threshold: 최적 CKA 임계값 (수평 점선으로 표시).
        output_path: 저장 경로 (.pdf).
    """
    import importlib

    plt_mod = importlib.import_module("matplotlib.pyplot")

    fig, axes = plt_mod.subplots(1, 2, figsize=(12, 5), sharey=True)

    panels = [
        (axes[0], successes, "Success (MAE < 50)"),
        (axes[1], failures, "Failure (MAE ≥ 50)"),
    ]

    for ax, traj_list, title in panels:
        plotted_methods: set[str] = set()

        for traj in traj_list:
            epochs = traj.get("epochs_tracked", [])
            mean_cka = traj.get("mean_cka_per_epoch", [])
            method = traj.get("method", "unknown")

            if not epochs or not mean_cka:
                continue

            color = _get_method_color(method)
            label = method if method not in plotted_methods else None
            plotted_methods.add(method)

            ax.plot(
                epochs,
                mean_cka,
                color=color,
                alpha=0.7,
                linewidth=1.5,
                label=label,
            )

        # 최적 임계값 수평 점선
        ax.axhline(
            y=optimal_threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Threshold={optimal_threshold:.2f}",
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Mean CKA (vs frozen baseline)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Layer-wise CKA Trajectory During Fine-tuning", fontsize=14, fontweight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight", dpi=150)
    plt_mod.close(fig)
    logger.info("그림 저장: %s", output_path)


def main() -> None:
    """메인 진입점."""
    _setup_logging()
    args = _parse_args()

    traj_dir = Path(args.traj_dir)
    output_dir = Path(args.output_dir)
    analysis_out = Path(args.analysis_out)

    if not traj_dir.exists():
        logger.error("궤적 디렉토리가 존재하지 않음: %s", traj_dir)
        raise SystemExit(1)

    # ─── 궤적 로드 및 분류 ────────────────────────────────────
    trajectories = _load_trajectories(traj_dir)
    if not trajectories:
        logger.error("분석할 궤적 파일이 없습니다.")
        raise SystemExit(1)

    successes, failures = _classify_trajectories(trajectories, args.divergence_mae_threshold)

    # ─── 임계값 최적화 ────────────────────────────────────────
    threshold_analysis = _find_optimal_threshold(successes, failures, args.early_epoch)
    optimal_threshold = threshold_analysis["optimal_threshold"]

    # ─── 결과 집계 ────────────────────────────────────────────
    analysis: dict[str, Any] = {
        "total_trajectories": len(trajectories),
        "n_success": len(successes),
        "n_failure": len(failures),
        "divergence_mae_threshold": args.divergence_mae_threshold,
        "early_epoch_analyzed": args.early_epoch,
        "threshold_analysis": threshold_analysis,
        "per_trajectory": [],
    }

    for traj in trajectories:
        cka_at_early = _get_cka_at_epoch(traj, args.early_epoch)
        analysis["per_trajectory"].append({
            "experiment_id": f"{traj.get('model')}_{traj.get('method')}_{traj.get('domain')}_seed{traj.get('seed')}",
            "model": traj.get("model"),
            "method": traj.get("method"),
            "domain": traj.get("domain"),
            "seed": traj.get("seed"),
            "final_mae": traj.get("final_mae"),
            "diverged": traj.get("diverged"),
            f"cka_at_epoch{args.early_epoch}": cka_at_early,
            "mean_cka_final": traj["mean_cka_per_epoch"][-1] if traj.get("mean_cka_per_epoch") else None,
        })

    # ─── JSON 저장 ────────────────────────────────────────────
    analysis_out.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info("분석 결과 저장: %s", analysis_out)

    # ─── 시각화 ───────────────────────────────────────────────
    fig_path = output_dir / "fig9_cka_trajectory.pdf"
    try:
        _plot_trajectories(successes, failures, optimal_threshold, fig_path)
    except ImportError:
        logger.warning("matplotlib 미설치. 그림 생성 생략.")
    except Exception as exc:
        logger.error("그림 생성 실패: %s", exc)

    # ─── 요약 출력 ────────────────────────────────────────────
    logger.info("=== CKA 궤적 분석 요약 ===")
    logger.info("총 궤적: %d (성공=%d, 실패=%d)", len(trajectories), len(successes), len(failures))
    logger.info(
        "에폭 %d 평균 CKA: 성공=%.4f, 실패=%.4f",
        args.early_epoch,
        threshold_analysis["mean_cka_success"],
        threshold_analysis["mean_cka_failure"],
    )
    logger.info(
        "최적 임계값: %.3f (F1=%.4f, precision=%.4f, recall=%.4f)",
        optimal_threshold,
        threshold_analysis["best_f1"],
        threshold_analysis["precision_at_threshold"],
        threshold_analysis["recall_at_threshold"],
    )
    logger.info(
        "CKA < 0.5 기준: precision=%.4f, recall=%.4f",
        threshold_analysis["precision_at_0_5"],
        threshold_analysis["recall_at_0_5"],
    )


if __name__ == "__main__":
    main()
