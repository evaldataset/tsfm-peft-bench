from __future__ import annotations

"""보충 figure 생성 스크립트 (Task B2 / Task 3).

두 가지 보충 figure를 생성:
  - Figure S1: DoRA vs LoRA 산점도 — (model, domain) 셀별 MAE 비교
  - Figure S2: Per-domain η² 그리드 — 4개 패널(도메인별) bar chart

출력:
  results/expansion_analysis_v3/figures/dora_vs_lora.pdf
  results/expansion_analysis_v3/figures/per_domain_eta.pdf

Usage:
    PYTHONPATH=. python scripts/plot_supplementary_figures.py
    PYTHONPATH=. python scripts/plot_supplementary_figures.py \\
        --input_dir results/expansion/domain \\
        --robustness results/expansion_analysis_v3/robustness.json \\
        --output_dir results/expansion_analysis_v3/figures
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ─── NeurIPS 스타일 ──────────────────────────────────────────────

NEURIPS_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

MODEL_NAMES: dict[str, str] = {
    "chronos": "Chronos",
    "moment": "MOMENT",
    "moirai": "Moirai",
    "timesfm": "TimesFM",
}
DOMAIN_NAMES: dict[str, str] = {
    "ett_m1": "ETTm1",
    "finance": "Finance",
    "smd": "SMD",
    "physionet": "PhysioNet",
}
MODEL_COLORS: dict[str, str] = {
    "chronos": "#E24A33",
    "moment": "#348ABD",
    "moirai": "#988ED5",
    "timesfm": "#56B4E9",
}
MODEL_MARKERS: dict[str, str] = {
    "chronos": "o",
    "moment": "s",
    "moirai": "^",
    "timesfm": "D",
}


# ─── 데이터 로딩 ─────────────────────────────────────────────────


def _load_domain_results(input_dir: Path) -> list[dict[str, Any]]:
    """domain 모드 결과 JSON 파일을 로드.

    Args:
        input_dir: domain 결과 JSON이 담긴 디렉토리.

    Returns:
        유효한 결과 레코드 리스트.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("all_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        mode = rec.get("experiment_mode", "domain")
        if mode != "domain":
            continue
        mae = rec.get("metrics", {}).get("mae")
        if not isinstance(mae, (int, float)):
            continue
        records.append(rec)
    logger.info("로드된 레코드 수: %d", len(records))
    return records


def _filter_outliers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """analyze_expansion_v2.py 와 동일한 이상치 필터 적용.

    Args:
        records: 원시 레코드 리스트.

    Returns:
        이상치가 제거된 레코드 리스트.
    """
    zero_shot_mae: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        if r.get("method") != "zero_shot":
            continue
        mae = r.get("metrics", {}).get("mae")
        if isinstance(mae, (int, float)):
            key = (str(r.get("model", "")), str(r.get("domain", "")))
            zero_shot_mae[key].append(float(mae))

    baselines: dict[tuple[str, str], float] = {
        k: float(np.median(v)) for k, v in zero_shot_mae.items() if v
    }

    cell_mae: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        mae = r.get("metrics", {}).get("mae")
        if isinstance(mae, (int, float)):
            key = (str(r.get("model", "")), str(r.get("domain", "")))
            cell_mae[key].append(float(mae))

    normal: list[dict[str, Any]] = []
    n_out = 0
    for r in records:
        mae = r.get("metrics", {}).get("mae")
        key = (str(r.get("model", "")), str(r.get("domain", "")))
        baseline = baselines.get(key)
        if baseline is None and key in cell_mae:
            baseline = float(np.median(cell_mae[key]))
        if isinstance(mae, (int, float)) and baseline is not None and baseline > 0:
            if mae > 10.0 * baseline:
                n_out += 1
                continue
        normal.append(r)

    logger.info("이상치 %d개 제거, 정상 %d개 유지", n_out, len(normal))
    return normal


def _aggregate_by_cell(
    records: list[dict[str, Any]],
    method: str,
    keys: tuple[str, str],
) -> dict[tuple[str, str], float]:
    """특정 method에 대해 (model, domain) 셀별 mean MAE를 계산.

    Args:
        records: 필터링된 레코드 리스트.
        method: 집계할 method 이름.
        keys: 셀 키로 사용할 필드명 튜플 (예: ("model", "domain")).

    Returns:
        {(key1_val, key2_val): mean_mae} 딕셔너리.
    """
    cell_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        if r.get("method") != method:
            continue
        mae = r.get("metrics", {}).get("mae")
        if not isinstance(mae, (int, float)):
            continue
        cell_key = (str(r.get(keys[0], "")), str(r.get(keys[1], "")))
        cell_vals[cell_key].append(float(mae))
    return {k: float(np.mean(v)) for k, v in cell_vals.items() if v}


# ─── Figure S1: DoRA vs LoRA 산점도 ──────────────────────────────


def fig_s1_dora_vs_lora(records: list[dict[str, Any]], output_dir: Path) -> None:
    """DoRA vs LoRA 산점도 생성.

    각 (model, domain) 셀에서 LoRA MAE (x축) vs DoRA MAE (y축)를 점으로 표시.
    대각선(y=x)에서 벗어난 점이 DoRA 개선/악화를 나타냄.
    색상 = 모델, 마커 = 도메인.

    Args:
        records: 필터링된 레코드 리스트.
        output_dir: figure 저장 디렉토리.
    """
    lora_cell = _aggregate_by_cell(records, "lora", ("model", "domain"))
    dora_cell = _aggregate_by_cell(records, "dora", ("model", "domain"))

    # 두 method 모두 데이터가 있는 셀만 사용
    common_cells = sorted(set(lora_cell.keys()) & set(dora_cell.keys()))

    if len(common_cells) < 2:
        logger.warning(
            "DoRA/LoRA 공통 셀이 %d개뿐입니다. Figure S1 생성을 건너뜁니다.", len(common_cells)
        )
        return

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    x_all = [lora_cell[c] for c in common_cells]
    y_all = [dora_cell[c] for c in common_cells]

    # 도메인별 마커
    domain_markers: dict[str, str] = {
        "ett_m1": "o",
        "finance": "s",
        "smd": "^",
        "physionet": "D",
    }

    for (model, domain), x_val, y_val in zip(common_cells, x_all, y_all):
        color = MODEL_COLORS.get(model, "#888888")
        marker = domain_markers.get(domain, "o")
        ax.scatter(
            x_val,
            y_val,
            color=color,
            marker=marker,
            s=55,
            alpha=0.85,
            linewidths=0.5,
            edgecolors="white",
            zorder=3,
        )

    # y=x 기준선
    all_vals = x_all + y_all
    if all_vals:
        lo = min(all_vals) * 0.95
        hi = max(all_vals) * 1.05
        ax.plot([lo, hi], [lo, hi], color="#AAAAAA", linewidth=0.9,
                linestyle="--", zorder=1, label="y = x (no difference)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("LoRA MAE", fontsize=9)
    ax.set_ylabel("DoRA MAE", fontsize=9)
    ax.set_title(
        "DoRA vs. LoRA: per-(model, domain) MAE\n"
        "(above diagonal = DoRA worse; below = DoRA better)",
        fontsize=9,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 범례: 모델(색상)
    model_handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=MODEL_COLORS.get(m, "#888888"),
            markersize=7,
            label=MODEL_NAMES.get(m, m),
        )
        for m in sorted({c[0] for c in common_cells})
    ]
    # 범례: 도메인(마커)
    domain_handles = [
        plt.Line2D(
            [0], [0],
            marker=domain_markers.get(d, "o"), color="w",
            markerfacecolor="#555555",
            markersize=7,
            label=DOMAIN_NAMES.get(d, d),
        )
        for d in sorted({c[1] for c in common_cells})
    ]

    legend1 = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        fontsize=7,
        title_fontsize=7,
        framealpha=0.8,
        edgecolor="#CCCCCC",
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=domain_handles,
        title="Domain",
        loc="lower right",
        fontsize=7,
        title_fontsize=7,
        framealpha=0.8,
        edgecolor="#CCCCCC",
    )

    # n=X 표시
    ax.text(
        0.97, 0.03,
        f"n = {len(common_cells)} cells",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="#666666",
    )

    fig.tight_layout()
    _save_figure(fig, output_dir / "dora_vs_lora")
    logger.info("Figure S1 저장 완료: dora_vs_lora.pdf")


# ─── Figure S2: Per-domain η² 그리드 ─────────────────────────────


def fig_s2_per_domain_eta(robustness_path: Path, output_dir: Path) -> None:
    """도메인별 η² 그리드 (4-패널 bar chart).

    robustness.json 에서 per_domain_eta_squared를 읽어
    도메인마다 하나의 패널을 그림.
    각 패널: x축 = 모델, bar 색상 = 모델.

    Args:
        robustness_path: compute_robustness.py 출력 JSON 경로.
        output_dir: figure 저장 디렉토리.
    """
    if not robustness_path.exists():
        logger.error("robustness.json을 찾을 수 없습니다: %s", robustness_path)
        logger.error("먼저 compute_robustness.py를 실행하세요.")
        return

    with open(robustness_path, "r", encoding="utf-8") as f:
        robustness: dict[str, Any] = json.load(f)

    # 모델 및 도메인 목록 수집
    models = sorted(robustness.keys())
    domains: set[str] = set()
    for mdata in robustness.values():
        domains.update(mdata.get("per_domain_eta_squared", {}).keys())
    domains_sorted = sorted(domains)

    if not domains_sorted:
        logger.warning("per_domain_eta_squared 데이터가 없습니다.")
        return

    n_domains = len(domains_sorted)
    ncols = min(n_domains, 4)
    nrows = (n_domains + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.8 * ncols, 2.6 * nrows), squeeze=False
    )

    bar_width = 0.6
    x_pos = np.arange(len(models))

    for idx, domain in enumerate(domains_sorted):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        eta_vals = []
        bar_colors = []
        for m in models:
            eta = robustness.get(m, {}).get("per_domain_eta_squared", {}).get(domain, 0.0)
            eta_vals.append(float(eta))
            bar_colors.append(MODEL_COLORS.get(m, "#888888"))

        bars = ax.bar(
            x_pos, eta_vals,
            width=bar_width,
            color=bar_colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        # 수치 annotation
        for bar, val in zip(bars, eta_vals):
            if val > 0.005:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#333333",
                )

        ax.set_title(DOMAIN_NAMES.get(domain, domain), fontweight="bold", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [MODEL_NAMES.get(m, m) for m in models],
            rotation=20,
            ha="right",
            fontsize=7.5,
        )
        ax.set_ylabel("η² (method factor)" if col == 0 else "", fontsize=8)
        ax.set_ylim(0, max(max(eta_vals) * 1.18, 0.05))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2)

    # 빈 서브플롯 숨기기
    for idx in range(n_domains, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Per-domain η² for Method Factor (one-way ANOVA per domain)\n"
        "Higher = method choice matters more in this domain",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir / "per_domain_eta")
    logger.info("Figure S2 저장 완료: per_domain_eta.pdf")


# ─── 저장 헬퍼 ───────────────────────────────────────────────────


def _save_figure(fig: plt.Figure, base_path: Path) -> None:  # type: ignore[name-defined]
    """figure를 PDF 및 PNG로 저장.

    Args:
        fig: matplotlib Figure 객체.
        base_path: 확장자 없는 저장 경로.
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base_path) + ".pdf")
    fig.savefig(str(base_path) + ".png")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────


def _setup_logging() -> None:
    """로깅 설정 초기화."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="보충 Figure S1 (DoRA vs LoRA) 및 S2 (per-domain η²) 생성"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/expansion/domain",
        help="domain 결과 JSON 디렉토리 (기본값: results/expansion/domain)",
    )
    parser.add_argument(
        "--robustness",
        type=str,
        default="results/expansion_analysis_v3/robustness.json",
        help="compute_robustness.py 출력 JSON (기본값: results/expansion_analysis_v3/robustness.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/expansion_analysis_v3/figures",
        help="figure 저장 디렉토리 (기본값: results/expansion_analysis_v3/figures)",
    )
    return parser.parse_args()


def main() -> None:
    """보충 figure 생성 메인 엔트리포인트."""
    _setup_logging()
    args = _parse_args()

    input_dir = Path(args.input_dir)
    robustness_path = Path(args.robustness)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(NEURIPS_RC)

    if not input_dir.exists():
        logger.error("입력 디렉토리가 존재하지 않습니다: %s", input_dir)
        return

    # Figure S1: DoRA vs LoRA
    records = _load_domain_results(input_dir)
    records = _filter_outliers(records)
    fig_s1_dora_vs_lora(records, output_dir)

    # Figure S2: Per-domain η²
    fig_s2_per_domain_eta(robustness_path, output_dir)

    logger.info("모든 보충 figure 생성 완료: %s", output_dir)


if __name__ == "__main__":
    main()
