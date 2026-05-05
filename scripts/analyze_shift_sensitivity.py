"""Architecture Sensitivity Profile (ASP) 및 Adaptation Risk Score (ARS) 분석.

각 아키텍처가 도메인 시프트의 어느 차원에 민감한지 계산하고,
이를 기반으로 ARS를 정의하여 PEFT 방법 선택 실패를 예측한다.

출력:
  - results/expansion_analysis/fig6_sensitivity_profile.pdf
  - results/expansion_analysis/fig7_ars_validation.pdf
  - results/shift_sensitivity_analysis.json
  - results/asp_table.tex
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)

# ─── 상수 ──────────────────────────────────────────────────────────

DIVERGENCE_THRESHOLD = 50.0

MODELS = ["chronos", "moment", "moirai"]

DOMAINS = ["ett_m1", "finance", "smd", "physionet"]

METHODS = ["zero_shot", "head_only", "lora", "adapter", "full_fine_tuning"]

# 5차원 시프트 프로파일: 계산된 artifact에서 로드 (하드코딩 금지).
from scripts.build_selector import (  # noqa: E402
    _SHIFT_PROFILES as SHIFT_PROFILES,
    _SHIFT_DIM_NAMES as SHIFT_DIM_NAMES,
)

RESULT_DIR = Path("results/expansion/domain")
OUTPUT_DIR = Path("results/expansion_analysis")
SUMMARY_JSON = Path("results/shift_sensitivity_analysis.json")
ASP_TEX = Path("results/asp_table.tex")


# ─── 데이터 로딩 ───────────────────────────────────────────────────


def _load_domain_results(result_dir: Path) -> list[dict[str, Any]]:
    """도메인 모드 실험 결과를 로드.

    Args:
        result_dir: JSON 결과 파일들이 저장된 디렉토리.

    Returns:
        실험 결과 딕셔너리 리스트.
    """
    rows: list[dict[str, Any]] = []
    for json_file in sorted(result_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                row = json.load(f)
            if not isinstance(row, dict):
                continue
            rows.append(row)
        except (OSError, json.JSONDecodeError):
            logger.warning("파일 로드 실패: %s", json_file)
            continue

    logger.info("총 %d개 결과 로드 완료", len(rows))
    return rows


def _is_diverged(mae: float) -> bool:
    """학습 발산 여부 판단.

    Args:
        mae: Mean Absolute Error 값.

    Returns:
        발산으로 판단되면 True.
    """
    return mae > DIVERGENCE_THRESHOLD


# ─── Step 1: Mean MAE 계산 ─────────────────────────────────────────


def compute_mean_mae(
    results: list[dict[str, Any]],
) -> dict[tuple[str, str, str], float]:
    """(model, method, domain) 별 seed 평균 MAE 계산.

    발산한 결과(MAE > threshold)는 제외하고 평균을 낸다.
    유효한 결과가 없으면 해당 조합은 딕셔너리에 포함되지 않는다.

    Args:
        results: 실험 결과 리스트.

    Returns:
        (model, method, domain) → 평균 MAE 딕셔너리.
    """
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    diverged_count = 0

    for row in results:
        model = row.get("model")
        method = row.get("method")
        domain = row.get("domain")
        mae = row.get("metrics", {}).get("mae")

        if not (model and method and domain and isinstance(mae, (int, float))):
            continue
        if model not in MODELS or method not in METHODS or domain not in DOMAINS:
            continue

        mae_f = float(mae)
        if _is_diverged(mae_f):
            diverged_count += 1
            continue

        buckets[(str(model), str(method), str(domain))].append(mae_f)

    logger.info("발산 결과 %d개 제외", diverged_count)

    mean_mae: dict[tuple[str, str, str], float] = {}
    for key, vals in buckets.items():
        if vals:
            mean_mae[key] = float(np.mean(vals))

    return mean_mae


# ─── Step 2: Method Gap 계산 ──────────────────────────────────────


def compute_method_gap(
    mean_mae: dict[tuple[str, str, str], float],
) -> dict[tuple[str, str], float]:
    """(model, domain) 별 method gap = max_MAE - min_MAE 계산.

    Args:
        mean_mae: (model, method, domain) → 평균 MAE.

    Returns:
        (model, domain) → method gap 딕셔너리.
    """
    domain_method_mae: dict[tuple[str, str], list[float]] = defaultdict(list)

    for (model, method, domain), mae in mean_mae.items():
        domain_method_mae[(model, domain)].append(mae)

    gap: dict[tuple[str, str], float] = {}
    for (model, domain), maes in domain_method_mae.items():
        if len(maes) >= 2:
            gap[(model, domain)] = float(max(maes) - min(maes))
        else:
            gap[(model, domain)] = 0.0

    return gap


# ─── Step 3: Architecture Sensitivity Profile (ASP) ───────────────


def compute_asp(
    method_gap: dict[tuple[str, str], float],
) -> dict[str, NDArray[np.float64]]:
    """각 아키텍처의 시프트 차원별 Spearman 상관계수 계산.

    n=4 도메인으로 Spearman rank correlation을 사용한다.

    Args:
        method_gap: (model, domain) → method gap.

    Returns:
        model → 길이 5의 Spearman 상관계수 배열 (shift_dim 순서).
    """
    asp: dict[str, NDArray[np.float64]] = {}

    for model in MODELS:
        gaps = []
        shift_vecs: list[list[float]] = []

        for domain in DOMAINS:
            gap_val = method_gap.get((model, domain))
            if gap_val is None:
                logger.warning("gap 없음: model=%s domain=%s", model, domain)
                continue
            gaps.append(gap_val)
            shift_vecs.append(SHIFT_PROFILES[domain])

        if len(gaps) < 3:
            logger.warning(
                "model=%s: 유효 도메인 %d개 — ASP 계산 불가", model, len(gaps)
            )
            asp[model] = np.zeros(len(SHIFT_DIM_NAMES))
            continue

        gaps_arr = np.array(gaps)
        shift_arr = np.array(shift_vecs)  # (n_domains, n_dims)

        corrs = np.zeros(len(SHIFT_DIM_NAMES))
        for i in range(len(SHIFT_DIM_NAMES)):
            rho, _ = stats.spearmanr(shift_arr[:, i], gaps_arr)
            corrs[i] = float(rho) if not math.isnan(rho) else 0.0

        asp[model] = corrs
        logger.info(
            "ASP [%s]: %s",
            model,
            dict(zip(SHIFT_DIM_NAMES, corrs.round(3).tolist())),
        )

    return asp


# ─── Step 4: Adaptation Risk Score (ARS) ──────────────────────────


def compute_ars(
    asp: dict[str, NDArray[np.float64]],
) -> dict[str, dict[str, float]]:
    """ARS(architecture, domain) = sum_i(ASP_a[i] * shift_d[i]) 계산 후 [0,1] 정규화.

    Args:
        asp: model → 길이 5 Spearman 상관계수 배열.

    Returns:
        model → (domain → ARS 값) 딕셔너리. 값은 [0, 1] 범위로 정규화.
    """
    ars_raw: dict[str, dict[str, float]] = {}

    for model, weights in asp.items():
        domain_scores: dict[str, float] = {}
        for domain, shift_vec in SHIFT_PROFILES.items():
            score = float(np.dot(weights, np.array(shift_vec)))
            domain_scores[domain] = score
        ars_raw[model] = domain_scores

    # 각 모델별로 [0, 1] 정규화
    ars_norm: dict[str, dict[str, float]] = {}
    for model, domain_scores in ars_raw.items():
        vals = list(domain_scores.values())
        vmin, vmax = min(vals), max(vals)
        denom = vmax - vmin if vmax > vmin else 1.0
        ars_norm[model] = {
            d: (v - vmin) / denom for d, v in domain_scores.items()
        }

    return ars_norm


# ─── Step 5: ARS 검증 통계 ────────────────────────────────────────


def _divergence_rate(
    mean_mae: dict[tuple[str, str, str], float],
    results: list[dict[str, Any]],
) -> dict[tuple[str, str], float]:
    """(model, domain) 별 발산율 계산.

    Args:
        mean_mae: 정상 결과의 평균 MAE (참조용, 사용 안 함).
        results: 전체 실험 결과 리스트.

    Returns:
        (model, domain) → 발산율 [0, 1].
    """
    total: dict[tuple[str, str], int] = defaultdict(int)
    div: dict[tuple[str, str], int] = defaultdict(int)

    for row in results:
        model = row.get("model")
        domain = row.get("domain")
        mae = row.get("metrics", {}).get("mae")
        if not (model and domain and isinstance(mae, (int, float))):
            continue
        if model not in MODELS or domain not in DOMAINS:
            continue
        key = (str(model), str(domain))
        total[key] += 1
        if _is_diverged(float(mae)):
            div[key] += 1

    rate: dict[tuple[str, str], float] = {}
    for key, cnt in total.items():
        rate[key] = div.get(key, 0) / cnt if cnt > 0 else 0.0

    return rate


def validate_ars(
    ars: dict[str, dict[str, float]],
    mean_mae: dict[tuple[str, str, str], float],
    method_gap: dict[tuple[str, str], float],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """ARS와 (best-method MAE, method gap, divergence rate) 간의 Spearman 상관 계산.

    Args:
        ars: model → (domain → ARS 값).
        mean_mae: (model, method, domain) → 평균 MAE.
        method_gap: (model, domain) → method gap.
        results: 전체 실험 결과 리스트.

    Returns:
        검증 통계 딕셔너리.
    """
    div_rate = _divergence_rate(mean_mae, results)

    # 각 (model, domain) 쌍에 대해 best-method MAE 계산
    best_mae: dict[tuple[str, str], float] = {}
    for (model, method, domain), mae in mean_mae.items():
        key = (model, domain)
        if key not in best_mae or mae < best_mae[key]:
            best_mae[key] = mae

    ars_vals: list[float] = []
    best_mae_vals: list[float] = []
    gap_vals: list[float] = []
    div_vals: list[float] = []

    for model in MODELS:
        for domain in DOMAINS:
            ars_val = ars.get(model, {}).get(domain)
            if ars_val is None:
                continue
            key = (model, domain)
            bm = best_mae.get(key)
            gp = method_gap.get(key)
            dv = div_rate.get(key, 0.0)

            if bm is not None and gp is not None:
                ars_vals.append(ars_val)
                best_mae_vals.append(bm)
                gap_vals.append(gp)
                div_vals.append(dv)

    validation: dict[str, Any] = {}

    if len(ars_vals) >= 3:
        rho_bm, p_bm = stats.spearmanr(ars_vals, best_mae_vals)
        rho_gp, p_gp = stats.spearmanr(ars_vals, gap_vals)
        rho_dv, p_dv = stats.spearmanr(ars_vals, div_vals)
        validation = {
            "ars_vs_best_mae": {"rho": float(rho_bm), "p": float(p_bm)},
            "ars_vs_method_gap": {"rho": float(rho_gp), "p": float(p_gp)},
            "ars_vs_divergence_rate": {"rho": float(rho_dv), "p": float(p_dv)},
            "n_points": len(ars_vals),
        }
        logger.info(
            "ARS 검증 — best_mae: ρ=%.3f (p=%.3f), gap: ρ=%.3f (p=%.3f), "
            "div: ρ=%.3f (p=%.3f)",
            rho_bm, p_bm, rho_gp, p_gp, rho_dv, p_dv,
        )
    else:
        logger.warning("ARS 검증 데이터 부족: %d points", len(ars_vals))

    return validation


# ─── Step 6: LOOCV Regret Prediction ──────────────────────────────


def loocv_regret_prediction(
    mean_mae: dict[tuple[str, str, str], float],
    method_gap: dict[tuple[str, str], float],
    asp_full: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    """LOOCV로 ARS-gated selector vs pure best-method 비교.

    각 도메인을 순서대로 held-out하여:
    - 나머지 3개 도메인으로 ASP 재계산
    - held-out 도메인의 ARS 예측
    - ARS > 0.5 → 보수적 방법(head_only/zero_shot); else → 최적 방법

    Args:
        mean_mae: (model, method, domain) → 평균 MAE.
        method_gap: (model, domain) → method gap.
        asp_full: 전체 데이터로 계산된 ASP (초기화 참조용).

    Returns:
        LOOCV 결과 딕셔너리.
    """
    ARS_THRESHOLD = 0.5
    CONSERVATIVE_METHODS = {"head_only", "zero_shot"}

    loocv_results: list[dict[str, Any]] = []

    for held_out in DOMAINS:
        train_domains = [d for d in DOMAINS if d != held_out]

        # 훈련 도메인으로 gap 재계산
        train_gap: dict[tuple[str, str], float] = {
            k: v for k, v in method_gap.items() if k[1] in train_domains
        }

        # 훈련 도메인으로 ASP 재계산
        asp_train = compute_asp(train_gap)

        # held-out 도메인의 ARS 계산
        for model in MODELS:
            weights = asp_train.get(model, np.zeros(len(SHIFT_DIM_NAMES)))
            shift_vec = np.array(SHIFT_PROFILES[held_out])
            ars_raw = float(np.dot(weights, shift_vec))

            # 훈련 도메인 ARS로 min/max 계산 (정규화)
            train_scores = [
                float(np.dot(weights, np.array(SHIFT_PROFILES[d])))
                for d in train_domains
            ]
            vmin, vmax = min(train_scores), max(train_scores)
            denom = vmax - vmin if vmax > vmin else 1.0
            ars_norm = (ars_raw - vmin) / denom

            # 예측: ARS > threshold → conservative
            if ars_norm > ARS_THRESHOLD:
                predicted_methods = list(CONSERVATIVE_METHODS)
            else:
                # 훈련 도메인에서 평균적으로 가장 좋은 방법 선택
                method_train_mae: dict[str, list[float]] = defaultdict(list)
                for (m, meth, d), mae in mean_mae.items():
                    if m == model and d in train_domains:
                        method_train_mae[meth].append(mae)
                best_method = min(
                    method_train_mae,
                    key=lambda meth: float(np.mean(method_train_mae[meth]))
                    if method_train_mae[meth]
                    else float("inf"),
                    default="head_only",
                )
                predicted_methods = [best_method]

            # 실제 held-out 성능
            actual_maes = {
                meth: mean_mae.get((model, meth, held_out))
                for meth in METHODS
            }
            actual_maes_valid = {
                k: v for k, v in actual_maes.items() if v is not None
            }

            if not actual_maes_valid:
                continue

            oracle_mae = min(actual_maes_valid.values())
            pred_mae = min(
                (actual_maes_valid[m] for m in predicted_methods if m in actual_maes_valid),
                default=min(actual_maes_valid.values()),
            )
            conservative_mae = min(
                (actual_maes_valid[m] for m in CONSERVATIVE_METHODS if m in actual_maes_valid),
                default=min(actual_maes_valid.values()),
            )
            # pure best: 훈련 도메인 평균 가장 좋은 방법
            method_train_mae_all: dict[str, list[float]] = defaultdict(list)
            for (m, meth, d), mae in mean_mae.items():
                if m == model and d in train_domains:
                    method_train_mae_all[meth].append(mae)
            pure_best_method = min(
                method_train_mae_all,
                key=lambda meth: float(np.mean(method_train_mae_all[meth]))
                if method_train_mae_all[meth]
                else float("inf"),
                default="head_only",
            )
            pure_best_mae = actual_maes_valid.get(pure_best_method, min(actual_maes_valid.values()))

            regret_ars = pred_mae - oracle_mae
            regret_pure = pure_best_mae - oracle_mae

            loocv_results.append({
                "held_out": held_out,
                "model": model,
                "ars_norm": float(ars_norm),
                "predicted_methods": predicted_methods,
                "oracle_mae": float(oracle_mae),
                "ars_gated_mae": float(pred_mae),
                "pure_best_mae": float(pure_best_mae),
                "regret_ars": float(regret_ars),
                "regret_pure": float(regret_pure),
            })

    mean_regret_ars = float(np.mean([r["regret_ars"] for r in loocv_results])) if loocv_results else 0.0
    mean_regret_pure = float(np.mean([r["regret_pure"] for r in loocv_results])) if loocv_results else 0.0

    logger.info(
        "LOOCV 결과 — ARS-gated regret: %.4f, pure-best regret: %.4f",
        mean_regret_ars, mean_regret_pure,
    )

    return {
        "mean_regret_ars_gated": mean_regret_ars,
        "mean_regret_pure_best": mean_regret_pure,
        "improvement": mean_regret_pure - mean_regret_ars,
        "per_fold": loocv_results,
    }


# ─── Step 7: 시각화 ────────────────────────────────────────────────


def plot_asp_heatmap(
    asp: dict[str, NDArray[np.float64]],
    output_path: Path,
) -> None:
    """Architecture Sensitivity Profile 히트맵 생성.

    모델을 X축, 시프트 차원을 Y축으로 하는 단일 히트맵.

    Args:
        asp: model → 길이 5 Spearman 상관계수 배열.
        output_path: 저장 경로 (.pdf).
    """
    models_ordered = [m for m in MODELS if m in asp]
    n_models = len(models_ordered)
    n_dims = len(SHIFT_DIM_NAMES)

    matrix = np.zeros((n_dims, n_models))
    for j, model in enumerate(models_ordered):
        matrix[:, j] = asp[model]

    fig, ax = plt.subplots(figsize=(max(4, n_models * 1.8), n_dims * 1.0 + 1.5))

    vabs = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    im = ax.imshow(
        matrix,
        cmap="RdBu",
        aspect="auto",
        vmin=-vabs,
        vmax=vabs,
    )

    ax.set_xticks(range(n_models))
    ax.set_xticklabels([m.upper() for m in models_ordered], fontsize=11)
    ax.set_yticks(range(n_dims))
    ax.set_yticklabels(SHIFT_DIM_NAMES, fontsize=10)

    # 값 주석
    for i in range(n_dims):
        for j in range(n_models):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.5 * vabs else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman ρ (method gap vs shift dim)", fontsize=9)

    ax.set_title(
        "Architecture Sensitivity Profile:\nwhich shift dimensions drive method choice?",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("Architecture", fontsize=10)
    ax.set_ylabel("Shift Dimension", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    # PNG도 저장
    fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("fig6 저장: %s", output_path)


def plot_ars_validation(
    ars: dict[str, dict[str, float]],
    method_gap: dict[tuple[str, str], float],
    output_path: Path,
) -> None:
    """ARS vs method gap 산점도 생성.

    Args:
        ars: model → (domain → ARS 값).
        method_gap: (model, domain) → method gap.
        output_path: 저장 경로 (.pdf).
    """
    model_colors = {
        "chronos": "#E74C3C",
        "moment": "#3498DB",
        "moirai": "#2ECC71",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    all_ars: list[float] = []
    all_gaps: list[float] = []

    for model in MODELS:
        if model not in ars:
            continue
        xs: list[float] = []
        ys: list[float] = []
        labels: list[str] = []

        for domain in DOMAINS:
            ars_val = ars[model].get(domain)
            gap_val = method_gap.get((model, domain))
            if ars_val is None or gap_val is None:
                continue
            xs.append(ars_val)
            ys.append(gap_val)
            labels.append(domain)
            all_ars.append(ars_val)
            all_gaps.append(gap_val)

        color = model_colors.get(model, "gray")
        ax.scatter(xs, ys, color=color, s=80, zorder=3, label=model.upper())

        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(
                lbl.replace("_", "\n"),
                (x, y),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
                color=color,
            )

    # 트렌드 라인
    if len(all_ars) >= 3:
        slope, intercept, r, p, _ = stats.linregress(all_ars, all_gaps)
        x_line = np.linspace(min(all_ars), max(all_ars), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", linewidth=1.2, alpha=0.6,
                label=f"trend (r={r:.2f}, p={p:.3f})")

    ax.set_xlabel("Adaptation Risk Score (ARS, normalized)", fontsize=11)
    ax.set_ylabel("Method Gap (max MAE − min MAE)", fontsize=11)
    ax.set_title("ARS Validation: Risk Score vs Method Sensitivity", fontsize=11)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("fig7 저장: %s", output_path)


# ─── Step 8: LaTeX 테이블 ──────────────────────────────────────────


def save_asp_latex(
    asp: dict[str, NDArray[np.float64]],
    output_path: Path,
) -> None:
    """ASP 행렬의 LaTeX 테이블 저장.

    Args:
        asp: model → 길이 5 Spearman 상관계수 배열.
        output_path: .tex 파일 경로.
    """
    models_ordered = [m for m in MODELS if m in asp]
    col_spec = "l" + "r" * len(models_ordered)
    header_cols = " & ".join([m.capitalize() for m in models_ordered])

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Architecture Sensitivity Profile (ASP): Spearman $\rho$ between "
        r"shift dimension and method gap across domains ($n=4$).}",
        r"\label{tab:asp}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Shift Dimension & {header_cols} \\",
        r"\midrule",
    ]

    for i, dim in enumerate(SHIFT_DIM_NAMES):
        row_vals = " & ".join(
            f"{asp[m][i]:.2f}" if m in asp else "--"
            for m in models_ordered
        )
        lines.append(rf"{dim} & {row_vals} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("ASP LaTeX 테이블 저장: %s", output_path)


# ─── Step 9: JSON 요약 저장 ────────────────────────────────────────


def save_summary_json(
    asp: dict[str, NDArray[np.float64]],
    ars: dict[str, dict[str, float]],
    validation: dict[str, Any],
    loocv: dict[str, Any],
    method_gap: dict[tuple[str, str], float],
    output_path: Path,
) -> None:
    """분석 결과를 JSON으로 저장.

    Args:
        asp: model → Spearman 상관계수 배열.
        ars: model → (domain → ARS 값).
        validation: ARS 검증 통계.
        loocv: LOOCV 결과.
        method_gap: (model, domain) → method gap.
        output_path: 저장 경로.
    """
    summary: dict[str, Any] = {
        "asp": {
            model: dict(zip(SHIFT_DIM_NAMES, corrs.tolist()))
            for model, corrs in asp.items()
        },
        "ars": ars,
        "method_gap": {
            f"{model}_{domain}": gap
            for (model, domain), gap in method_gap.items()
        },
        "ars_validation": validation,
        "loocv_regret": loocv,
        "metadata": {
            "models": MODELS,
            "domains": DOMAINS,
            "methods": METHODS,
            "shift_dim_names": SHIFT_DIM_NAMES,
            "shift_profiles": SHIFT_PROFILES,
            "divergence_threshold": DIVERGENCE_THRESHOLD,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("요약 JSON 저장: %s", output_path)


# ─── 메인 ──────────────────────────────────────────────────────────


def _print_summary(
    asp: dict[str, NDArray[np.float64]],
    ars: dict[str, dict[str, float]],
    validation: dict[str, Any],
    loocv: dict[str, Any],
) -> None:
    """분석 결과 요약 출력.

    Args:
        asp: Architecture Sensitivity Profile.
        ars: Adaptation Risk Score.
        validation: ARS 검증 통계.
        loocv: LOOCV 결과.
    """
    print("\n" + "=" * 60)
    print("Architecture Sensitivity Profile (ASP)")
    print("=" * 60)
    header = f"{'Dimension':<18}" + "".join(f"{m.upper():>12}" for m in MODELS if m in asp)
    print(header)
    print("-" * len(header))
    for i, dim in enumerate(SHIFT_DIM_NAMES):
        row = f"{dim:<18}" + "".join(
            f"{asp[m][i]:>12.3f}" if m in asp else f"{'N/A':>12}"
            for m in MODELS
        )
        print(row)

    print("\n" + "=" * 60)
    print("Adaptation Risk Score (ARS, normalized [0,1])")
    print("=" * 60)
    for model in MODELS:
        if model not in ars:
            continue
        print(f"\n  {model.upper()}:")
        for domain, val in sorted(ars[model].items()):
            bar = "█" * int(val * 20)
            print(f"    {domain:<12} {val:.3f}  {bar}")

    print("\n" + "=" * 60)
    print("ARS Validation Statistics")
    print("=" * 60)
    if validation:
        for key, stat in validation.items():
            if isinstance(stat, dict):
                print(f"  {key}: ρ={stat['rho']:.3f}, p={stat['p']:.3f}")
            else:
                print(f"  {key}: {stat}")

    print("\n" + "=" * 60)
    print("LOOCV Regret Prediction")
    print("=" * 60)
    if loocv:
        print(f"  Mean regret (ARS-gated):  {loocv['mean_regret_ars_gated']:.4f}")
        print(f"  Mean regret (pure-best):  {loocv['mean_regret_pure_best']:.4f}")
        print(f"  Improvement:              {loocv['improvement']:.4f}")
    print()


def main() -> None:
    """메인 실행 함수."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("결과 로드 시작: %s", RESULT_DIR)
    results = _load_domain_results(RESULT_DIR)

    logger.info("평균 MAE 계산 중...")
    mean_mae = compute_mean_mae(results)
    logger.info("유효 (model, method, domain) 조합: %d개", len(mean_mae))

    logger.info("Method gap 계산 중...")
    method_gap = compute_method_gap(mean_mae)

    logger.info("ASP 계산 중...")
    asp = compute_asp(method_gap)

    logger.info("ARS 계산 중...")
    ars = compute_ars(asp)

    logger.info("ARS 검증 중...")
    validation = validate_ars(ars, mean_mae, method_gap, results)

    logger.info("LOOCV regret 예측 중...")
    loocv = loocv_regret_prediction(mean_mae, method_gap, asp)

    logger.info("시각화 생성 중...")
    plot_asp_heatmap(asp, OUTPUT_DIR / "fig6_sensitivity_profile.pdf")
    plot_ars_validation(ars, method_gap, OUTPUT_DIR / "fig7_ars_validation.pdf")

    logger.info("결과 저장 중...")
    save_asp_latex(asp, ASP_TEX)
    save_summary_json(asp, ars, validation, loocv, method_gap, SUMMARY_JSON)

    _print_summary(asp, ars, validation, loocv)
    logger.info("분석 완료.")


if __name__ == "__main__":
    main()
