"""파일럿 실험 결과 통계 분석 및 시각화.

Phase 1A (method × shift 상호작용)와 Phase 1B (locus 스윕) 결과를 분석.
- Two-way ANOVA: method × shift_type
- Cliff's delta 효과 크기
- Kendall's tau locus 순위 상관
- Heatmap, Radar chart, Bar chart 생성
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(description="파일럿 실험 결과 분석")
    parser.add_argument(
        "--phase1a_dir",
        type=str,
        default="results/pilot_1a",
        help="Phase 1A 결과 디렉토리",
    )
    parser.add_argument(
        "--phase1b_dir",
        type=str,
        default="results/pilot_1b",
        help="Phase 1B 결과 디렉토리",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pilot_analysis",
        help="분석 결과 저장 디렉토리",
    )
    return parser.parse_args()


# ─── 통계 함수 ────────────────────────────────────────────────


def cliffs_delta(
    group1: NDArray[np.floating[Any]], group2: NDArray[np.floating[Any]]
) -> tuple[float, str]:
    """Cliff's delta 효과 크기를 계산.

    Args:
        group1: 첫 번째 그룹 관측값.
        group2: 두 번째 그룹 관측값.

    Returns:
        (delta 값, 해석 문자열) 튜플.
        해석은 negligible, small, medium, large 중 하나.

    Raises:
        ValueError: 빈 배열이 입력되었을 때.
    """
    if group1.size == 0 or group2.size == 0:
        raise ValueError("Cliff's delta 계산에 빈 배열이 입력되었습니다.")

    n1, n2 = len(group1), len(group2)
    dominance = 0.0

    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1.0
            elif x < y:
                dominance -= 1.0

    delta = dominance / (n1 * n2)

    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return delta, interpretation


def relative_improvement(baseline: float, improved: float) -> float:
    """기준선 대비 상대 개선율(%)을 계산.

    Args:
        baseline: 기준선 값.
        improved: 개선된 값 (낮을수록 좋은 메트릭 기준).

    Returns:
        상대 개선율 (%). 양수면 개선, 음수면 악화.
    """
    if abs(baseline) < 1e-10:
        return 0.0
    return (baseline - improved) / abs(baseline) * 100.0


# ─── 데이터 로딩 ──────────────────────────────────────────────


def _load_results(result_dir: Path) -> list[dict[str, Any]]:
    """결과 디렉토리에서 JSON 파일들을 로드.

    Args:
        result_dir: 결과 디렉토리 경로.

    Returns:
        결과 딕셔너리 리스트.
    """
    results: list[dict[str, Any]] = []

    # 통합 파일이 있으면 우선 사용
    all_results_path = result_dir / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                results = loaded
                logger.info(
                    "통합 결과 로드: %d 건 (%s)", len(results), all_results_path
                )
                return results

    # 개별 JSON 파일 로드
    for json_file in sorted(result_dir.glob("*.json")):
        if json_file.name == "all_results.json":
            continue
        with open(json_file, "r") as f:
            results.append(json.load(f))

    logger.info("개별 결과 로드: %d 건 (%s)", len(results), result_dir)
    return results


# ─── Phase 1A 분석 ────────────────────────────────────────────


def _analyze_phase1a(results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Phase 1A (method × shift) 결과를 분석.

    Args:
        results: 실험 결과 리스트.
        output_dir: 출력 디렉토리.

    Returns:
        분석 결과 딕셔너리.
    """
    if not results:
        logger.warning("Phase 1A 결과가 비어 있습니다.")
        return {}

    # 고유 값 추출
    models = sorted(set(r["model"] for r in results))
    methods = sorted(set(r["method"] for r in results))
    shift_types = sorted(set(r["shift_type"] for r in results))
    severities = sorted(set(r["severity"] for r in results))

    analysis: dict[str, Any] = {
        "models": models,
        "methods": methods,
        "shift_types": shift_types,
        "severities": severities,
        "n_results": len(results),
    }

    for model_name in models:
        model_results = [r for r in results if r["model"] == model_name]
        model_analysis: dict[str, Any] = {}

        # ─── Zero-shot 기준선 MAE 계산 ─────────────────────
        zs_maes: dict[str, list[float]] = {}
        for r in model_results:
            if r["method"] == "zero_shot":
                key = f"{r['shift_type']}_{r['severity']}"
                zs_maes.setdefault(key, []).append(r["metrics"]["mae"])

        zs_baseline: dict[str, float] = {
            k: float(np.mean(v)) for k, v in zs_maes.items()
        }
        model_analysis["zero_shot_baselines"] = zs_baseline

        # ─── 상대 개선율 계산 ──────────────────────────────
        improvements: dict[str, dict[str, list[float]]] = {}
        for r in model_results:
            if r["method"] == "zero_shot":
                continue
            key = f"{r['shift_type']}_{r['severity']}"
            baseline = zs_baseline.get(key, 0.0)
            rel_imp = relative_improvement(baseline, r["metrics"]["mae"])
            improvements.setdefault(r["method"], {}).setdefault(
                r["shift_type"], []
            ).append(rel_imp)

        model_analysis["relative_improvements"] = {
            method: {
                shift: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }
                for shift, vals in shifts.items()
            }
            for method, shifts in improvements.items()
        }

        # ─── Two-way ANOVA: method × shift_type ───────────
        # F 통계량 = 그룹 간 분산 / 그룹 내 분산
        groups_by_method: dict[str, list[float]] = {}
        groups_by_shift: dict[str, list[float]] = {}
        for r in model_results:
            mae_val = r["metrics"]["mae"]
            groups_by_method.setdefault(r["method"], []).append(mae_val)
            groups_by_shift.setdefault(r["shift_type"], []).append(mae_val)

        # Method 단방향 ANOVA
        method_groups = [np.array(v) for v in groups_by_method.values() if len(v) >= 2]
        if len(method_groups) >= 2:
            f_method, p_method = stats.f_oneway(*method_groups)
            model_analysis["anova_method"] = {
                "F": float(f_method),
                "p": float(p_method),
                "significant": bool(p_method < 0.05),
            }
        else:
            model_analysis["anova_method"] = {"F": 0.0, "p": 1.0, "significant": False}

        # Shift 단방향 ANOVA
        shift_groups = [np.array(v) for v in groups_by_shift.values() if len(v) >= 2]
        if len(shift_groups) >= 2:
            f_shift, p_shift = stats.f_oneway(*shift_groups)
            model_analysis["anova_shift"] = {
                "F": float(f_shift),
                "p": float(p_shift),
                "significant": bool(p_shift < 0.05),
            }
        else:
            model_analysis["anova_shift"] = {"F": 0.0, "p": 1.0, "significant": False}

        # ─── Cliff's delta: 각 method 쌍 ──────────────────
        cliff_results: dict[str, dict[str, float | str]] = {}
        non_zs_methods = [m for m in methods if m != "zero_shot"]
        for m1, m2 in combinations(non_zs_methods, 2):
            m1_maes = np.array(groups_by_method.get(m1, []))
            m2_maes = np.array(groups_by_method.get(m2, []))
            if m1_maes.size > 0 and m2_maes.size > 0:
                delta, interp = cliffs_delta(m1_maes, m2_maes)
                cliff_results[f"{m1}_vs_{m2}"] = {
                    "delta": delta,
                    "interpretation": interp,
                }
        model_analysis["cliffs_delta"] = cliff_results

        # ─── 성공 기준 판정 ────────────────────────────────
        max_imp = 0.0
        max_cliff = 0.0
        for method_data in improvements.values():
            for shift_data in method_data.values():
                imp_mean = abs(float(np.mean(shift_data)))
                if imp_mean > max_imp:
                    max_imp = imp_mean
        for cliff_data in cliff_results.values():
            if isinstance(cliff_data, dict) and "delta" in cliff_data:
                cliff_val = abs(float(cliff_data["delta"]))
                if cliff_val > max_cliff:
                    max_cliff = cliff_val

        model_analysis["success_criteria"] = {
            "max_relative_improvement_pct": max_imp,
            "max_cliffs_delta": max_cliff,
            "passes_improvement_threshold": max_imp >= 3.0,
            "passes_effect_size_threshold": max_cliff >= 0.3,
            "overall_pass": max_imp >= 3.0 or max_cliff >= 0.3,
        }

        analysis[model_name] = model_analysis

    # ─── 시각화 ────────────────────────────────────────────
    try:
        _plot_phase1a(results, analysis, output_dir)
    except ImportError as exc:
        logger.warning("matplotlib 미설치로 시각화 건너뜀: %s", exc)

    return analysis


def _plot_phase1a(
    results: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Phase 1A 시각화를 생성.

    Args:
        results: 실험 결과 리스트.
        analysis: 분석 결과 딕셔너리.
        output_dir: 출력 디렉토리.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = analysis.get("models", [])
    methods = analysis.get("methods", [])
    shift_types = analysis.get("shift_types", [])

    for model_name in models:
        model_data = analysis.get(model_name, {})
        improvements = model_data.get("relative_improvements", {})

        if not improvements:
            continue

        # ─── Heatmap: shift × method → 상대 개선율 ────────
        non_zs_methods = [m for m in methods if m != "zero_shot" and m in improvements]
        if non_zs_methods and shift_types:
            fig, ax = plt.subplots(figsize=(10, 6))
            data_matrix = np.zeros((len(shift_types), len(non_zs_methods)))

            for j, method in enumerate(non_zs_methods):
                for i, shift in enumerate(shift_types):
                    vals = improvements.get(method, {}).get(shift, {})
                    data_matrix[i, j] = (
                        vals.get("mean", 0.0) if isinstance(vals, dict) else 0.0
                    )

            im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(non_zs_methods)))
            ax.set_xticklabels(non_zs_methods, rotation=45, ha="right")
            ax.set_yticks(range(len(shift_types)))
            ax.set_yticklabels(shift_types)

            for i in range(len(shift_types)):
                for j in range(len(non_zs_methods)):
                    ax.text(
                        j,
                        i,
                        f"{data_matrix[i, j]:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black" if abs(data_matrix[i, j]) < 20 else "white",
                    )

            plt.colorbar(im, ax=ax, label="Relative MAE Improvement (%)")
            ax.set_title(f"Phase 1A: Method × Shift ({model_name})")
            ax.set_xlabel("Adaptation Method")
            ax.set_ylabel("Shift Type")
            fig.tight_layout()
            fig.savefig(output_dir / f"phase1a_heatmap_{model_name}.png", dpi=300)
            plt.close(fig)
            logger.info(
                "Heatmap 저장: %s", output_dir / f"phase1a_heatmap_{model_name}.png"
            )

        # ─── Radar chart: method별 shift 차원 성능 ─────────
        if non_zs_methods and len(shift_types) >= 3:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
            angles = np.linspace(0, 2 * np.pi, len(shift_types), endpoint=False)
            angles = np.concatenate([angles, [angles[0]]])

            for method in non_zs_methods:
                values = []
                for shift in shift_types:
                    val_data = improvements.get(method, {}).get(shift, {})
                    values.append(
                        val_data.get("mean", 0.0) if isinstance(val_data, dict) else 0.0
                    )
                values.append(values[0])
                ax.plot(angles, values, "o-", label=method, linewidth=2)
                ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(shift_types, fontsize=9)
            ax.set_title(f"Phase 1A: Radar ({model_name})")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            fig.tight_layout()
            fig.savefig(output_dir / f"phase1a_radar_{model_name}.png", dpi=300)
            plt.close(fig)
            logger.info(
                "Radar chart 저장: %s", output_dir / f"phase1a_radar_{model_name}.png"
            )

        # ─── Bar chart: method별 MAE (mean ± std) ─────────
        fig, ax = plt.subplots(figsize=(10, 6))
        method_mae_mean: list[float] = []
        method_mae_std: list[float] = []
        method_labels: list[str] = []
        for method in methods:
            maes = [
                r["metrics"]["mae"]
                for r in results
                if r["model"] == model_name and r["method"] == method
            ]
            if maes:
                method_labels.append(method)
                method_mae_mean.append(float(np.mean(maes)))
                method_mae_std.append(float(np.std(maes)))

        if method_labels:
            x = np.arange(len(method_labels))
            ax.bar(x, method_mae_mean, yerr=method_mae_std, capsize=5, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=45, ha="right")
            ax.set_ylabel("MAE")
            ax.set_title(f"Phase 1A: Method MAE ({model_name})")
            fig.tight_layout()
            fig.savefig(output_dir / f"phase1a_bar_{model_name}.png", dpi=300)
            plt.close(fig)
            logger.info(
                "Bar chart 저장: %s", output_dir / f"phase1a_bar_{model_name}.png"
            )


# ─── Phase 1B 분석 ────────────────────────────────────────────


def _analyze_phase1b(results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Phase 1B (locus 스윕) 결과를 분석.

    Args:
        results: 실험 결과 리스트.
        output_dir: 출력 디렉토리.

    Returns:
        분석 결과 딕셔너리.
    """
    if not results:
        logger.warning("Phase 1B 결과가 비어 있습니다.")
        return {}

    models = sorted(set(r["model"] for r in results))
    loci = sorted(set(r["locus"] for r in results))
    shift_types = sorted(set(r["shift_type"] for r in results))

    analysis: dict[str, Any] = {
        "models": models,
        "loci": loci,
        "shift_types": shift_types,
        "n_results": len(results),
    }

    # ─── 모델×shift별 locus MAE 순위 계산 ──────────────────
    rankings: dict[str, dict[str, list[str]]] = {}

    for model_name in models:
        rankings[model_name] = {}
        model_analysis: dict[str, Any] = {}

        for shift_type in shift_types:
            locus_maes: dict[str, float] = {}
            for locus in loci:
                maes = [
                    r["metrics"]["mae"]
                    for r in results
                    if r["model"] == model_name
                    and r["locus"] == locus
                    and r["shift_type"] == shift_type
                ]
                if maes:
                    locus_maes[locus] = float(np.mean(maes))

            # MAE 오름차순으로 순위 매기기 (낮을수록 좋음)
            ranked = sorted(locus_maes.keys(), key=lambda l: locus_maes[l])
            rankings[model_name][shift_type] = ranked
            model_analysis[f"ranking_{shift_type}"] = {
                "order": ranked,
                "maes": {l: locus_maes[l] for l in ranked},
            }

        # ─── 파라미터 효율성 ───────────────────────────────
        param_efficiency: dict[str, dict[str, float]] = {}
        for locus in loci:
            locus_results = [
                r for r in results if r["model"] == model_name and r["locus"] == locus
            ]
            if locus_results:
                avg_mae = float(np.mean([r["metrics"]["mae"] for r in locus_results]))
                avg_params = float(
                    np.mean([r.get("trainable_params", 0) for r in locus_results])
                )
                param_efficiency[locus] = {
                    "avg_mae": avg_mae,
                    "avg_trainable_params": avg_params,
                    "efficiency": avg_mae * avg_params
                    if avg_params > 0
                    else float("inf"),
                }
        model_analysis["param_efficiency"] = param_efficiency

        analysis[model_name] = model_analysis

    # ─── Kendall's tau: 모델 간 locus 순위 상관 ────────────
    if len(models) >= 2:
        tau_results: dict[str, dict[str, float]] = {}

        for shift_type in shift_types:
            common_loci = set(loci)
            ranks_by_model: dict[str, list[int]] = {}

            for model_name in models:
                ranked = rankings[model_name].get(shift_type, [])
                rank_map = {l: i for i, l in enumerate(ranked)}
                ranks_by_model[model_name] = [
                    rank_map.get(l, len(loci)) for l in sorted(common_loci)
                ]

            for m1, m2 in combinations(models, 2):
                r1 = np.array(ranks_by_model.get(m1, []))
                r2 = np.array(ranks_by_model.get(m2, []))
                if r1.size >= 3 and r2.size >= 3:
                    tau, p_value = stats.kendalltau(r1, r2)
                    key = f"{m1}_vs_{m2}_{shift_type}"
                    tau_results[key] = {
                        "tau": float(tau),
                        "p_value": float(p_value),
                        "significant": bool(p_value < 0.05),
                    }

        analysis["kendall_tau"] = tau_results

        # 전체 상관 (모든 shift 통합)
        all_r1: list[int] = []
        all_r2: list[int] = []
        for shift_type in shift_types:
            for model_name in models[:2]:  # 첫 두 모델
                ranked = rankings[model_name].get(shift_type, [])
                rank_map = {l: i for i, l in enumerate(ranked)}
                ranks = [rank_map.get(l, len(loci)) for l in sorted(set(loci))]
                if model_name == models[0]:
                    all_r1.extend(ranks)
                else:
                    all_r2.extend(ranks)

        if len(all_r1) >= 3 and len(all_r2) >= 3:
            tau_overall, p_overall = stats.kendalltau(
                np.array(all_r1), np.array(all_r2)
            )
            analysis["kendall_tau_overall"] = {
                "tau": float(tau_overall),
                "p_value": float(p_overall),
                "significant": bool(p_overall < 0.05),
            }

    # ─── 성공 기준 판정 ────────────────────────────────────
    tau_overall_data = analysis.get("kendall_tau_overall", {})
    analysis["success_criteria"] = {
        "kendall_tau": tau_overall_data.get("tau", 0.0),
        "kendall_p": tau_overall_data.get("p_value", 1.0),
        "significant_correlation": tau_overall_data.get("significant", False),
        "universal_locus_evidence": tau_overall_data.get("significant", False),
    }

    # ─── 시각화 ────────────────────────────────────────────
    try:
        _plot_phase1b(results, analysis, output_dir)
    except ImportError as exc:
        logger.warning("matplotlib 미설치로 시각화 건너뜀: %s", exc)

    return analysis


def _plot_phase1b(
    results: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Phase 1B 시각화를 생성.

    Args:
        results: 실험 결과 리스트.
        analysis: 분석 결과 딕셔너리.
        output_dir: 출력 디렉토리.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = analysis.get("models", [])
    loci = analysis.get("loci", [])
    shift_types = analysis.get("shift_types", [])

    # ─── Heatmap: locus × (model × shift) → MAE ──────────
    col_labels: list[str] = []
    for model_name in models:
        for shift_type in shift_types:
            col_labels.append(f"{model_name}\n{shift_type}")

    if loci and col_labels:
        data_matrix = np.zeros((len(loci), len(col_labels)))

        col_idx = 0
        for model_name in models:
            for shift_type in shift_types:
                for i, locus in enumerate(loci):
                    maes = [
                        r["metrics"]["mae"]
                        for r in results
                        if r["model"] == model_name
                        and r["locus"] == locus
                        and r["shift_type"] == shift_type
                    ]
                    data_matrix[i, col_idx] = float(np.mean(maes)) if maes else 0.0
                col_idx += 1

        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(data_matrix, cmap="YlOrRd_r", aspect="auto")

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(loci)))
        ax.set_yticklabels(loci)

        for i in range(len(loci)):
            for j in range(len(col_labels)):
                ax.text(
                    j,
                    i,
                    f"{data_matrix[i, j]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        plt.colorbar(im, ax=ax, label="MAE (lower is better)")
        ax.set_title("Phase 1B: Locus × Model×Shift MAE")
        ax.set_xlabel("Model × Shift Type")
        ax.set_ylabel("LoRA Locus")
        fig.tight_layout()
        fig.savefig(output_dir / "phase1b_heatmap.png", dpi=300)
        plt.close(fig)
        logger.info("Heatmap 저장: %s", output_dir / "phase1b_heatmap.png")

    # ─── Bar chart: locus별 MAE (model별) ─────────────────
    for model_name in models:
        fig, ax = plt.subplots(figsize=(10, 6))
        locus_means: list[float] = []
        locus_stds: list[float] = []
        locus_labels: list[str] = []

        for locus in loci:
            maes = [
                r["metrics"]["mae"]
                for r in results
                if r["model"] == model_name and r["locus"] == locus
            ]
            if maes:
                locus_labels.append(locus)
                locus_means.append(float(np.mean(maes)))
                locus_stds.append(float(np.std(maes)))

        if locus_labels:
            x = np.arange(len(locus_labels))
            ax.bar(x, locus_means, yerr=locus_stds, capsize=5, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(locus_labels, rotation=45, ha="right")
            ax.set_ylabel("MAE")
            ax.set_title(f"Phase 1B: Locus MAE ({model_name})")
            fig.tight_layout()
            fig.savefig(output_dir / f"phase1b_bar_{model_name}.png", dpi=300)
            plt.close(fig)
            logger.info(
                "Bar chart 저장: %s", output_dir / f"phase1b_bar_{model_name}.png"
            )


# ─── 메인 ─────────────────────────────────────────────────────


def main() -> None:
    """파일럿 분석 메인 함수."""
    _setup_logging()
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Phase 1A 분석 ─────────────────────────────────────
    phase1a_dir = Path(args.phase1a_dir)
    if phase1a_dir.exists():
        logger.info("Phase 1A 분석 시작: %s", phase1a_dir)
        results_1a = _load_results(phase1a_dir)
        analysis_1a = _analyze_phase1a(results_1a, output_dir)

        report_1a_path = output_dir / "phase1a_analysis.json"
        with open(report_1a_path, "w") as f:
            json.dump(analysis_1a, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Phase 1A 분석 완료: %s", report_1a_path)

        # 성공 기준 출력
        for model_name in analysis_1a.get("models", []):
            model_data = analysis_1a.get(model_name, {})
            criteria = model_data.get("success_criteria", {})
            logger.info(
                "Phase 1A [%s] 성공 기준: improvement=%.2f%% (≥3%%?%s), "
                "cliff_delta=%.3f (≥0.3?%s), 전체=%s",
                model_name,
                criteria.get("max_relative_improvement_pct", 0.0),
                "✓" if criteria.get("passes_improvement_threshold") else "✗",
                criteria.get("max_cliffs_delta", 0.0),
                "✓" if criteria.get("passes_effect_size_threshold") else "✗",
                "PASS" if criteria.get("overall_pass") else "FAIL",
            )
    else:
        logger.info("Phase 1A 디렉토리 없음, 건너뜀: %s", phase1a_dir)

    # ─── Phase 1B 분석 ─────────────────────────────────────
    phase1b_dir = Path(args.phase1b_dir)
    if phase1b_dir.exists():
        logger.info("Phase 1B 분석 시작: %s", phase1b_dir)
        results_1b = _load_results(phase1b_dir)
        analysis_1b = _analyze_phase1b(results_1b, output_dir)

        report_1b_path = output_dir / "phase1b_analysis.json"
        with open(report_1b_path, "w") as f:
            json.dump(analysis_1b, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Phase 1B 분석 완료: %s", report_1b_path)

        # 성공 기준 출력
        criteria = analysis_1b.get("success_criteria", {})
        logger.info(
            "Phase 1B 성공 기준: kendall_tau=%.3f (p=%.4f), "
            "유의미한 상관=%s, 보편 로커스 근거=%s",
            criteria.get("kendall_tau", 0.0),
            criteria.get("kendall_p", 1.0),
            "✓" if criteria.get("significant_correlation") else "✗",
            "YES" if criteria.get("universal_locus_evidence") else "NO",
        )
    else:
        logger.info("Phase 1B 디렉토리 없음, 건너뜀: %s", phase1b_dir)


if __name__ == "__main__":
    main()
