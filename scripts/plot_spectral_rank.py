"""학술용 스펙트럴 W1 대 랭크 민감도 산점도 생성.

입력: results/pivot_analysis/frequency_rank_report.json
출력: results/expansion_analysis/spectral_rank_scatter.{pdf,png}
"""
from __future__ import annotations

import json
import logging
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# 학술용 스타일
plt.rcParams.update({
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.grid": False,
    "axes.facecolor": "white",
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
})

DOMAIN_SPECTRAL_W1: dict[str, float] = {
    "ett_m1": 0.05,
    "finance": 0.02,
    "smd": 0.03,
}


def _domain_to_w1(domain: str) -> float | None:
    """도메인 문자열을 spectral W1 값으로 매핑합니다."""
    key = domain.strip().lower().replace("-", "_")
    if key in DOMAIN_SPECTRAL_W1:
        return DOMAIN_SPECTRAL_W1[key]
    if "ett" in key:
        return DOMAIN_SPECTRAL_W1.get("ett_m1")
    if "fin" in key:
        return DOMAIN_SPECTRAL_W1.get("finance")
    if "smd" in key:
        return DOMAIN_SPECTRAL_W1.get("smd")
    return None


def _load_data(input_path: Path) -> list[dict[str, Any]]:
    """frequency_rank_report.json에서 cell_summaries를 로드합니다."""
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows: list[dict[str, Any]] = []
    for item in data.get("cell_summaries", []):
        model = item.get("model")
        domain = item.get("domain")
        rank_sensitivity = item.get("rank_sensitivity")
        spectral_w1 = _domain_to_w1(domain) if domain else None
        if None in (model, domain, rank_sensitivity, spectral_w1):
            continue
        rows.append({
            "model": model,
            "domain": domain,
            "spectral_w1": spectral_w1,
            "rank_sensitivity": rank_sensitivity,
        })
    return rows


def _regression_with_ci(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """선형 회귀 + 95% CI 밴드를 반환합니다. (x_line, y_line, ci_lower, ci_upper)"""
    n = x.size
    if n < 3:
        raise ValueError("CI 계산을 위해서는 최소 3개 데이터 포인트가 필요합니다.")
    coef = np.polyfit(x, y, 1)
    line = np.poly1d(coef)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = line(x_line)
    residuals = y - line(x)
    mse = np.sum(residuals ** 2) / (n - 2)
    x_mean = x.mean()
    ssxx = np.sum((x - x_mean) ** 2)

    def se(x0: float) -> float:
        return math.sqrt(mse * (1.0 / n + ((x0 - x_mean) ** 2) / ssxx))

    se_vec = np.vectorize(se)
    ci_lower = y_line - 1.96 * se_vec(x_line)
    ci_upper = y_line + 1.96 * se_vec(x_line)
    return x_line, y_line, ci_lower, ci_upper


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_path = project_root / "results" / "pivot_analysis" / "frequency_rank_report.json"
    output_dir = project_root / "results" / "expansion_analysis"

    rows = _load_data(input_path)
    if len(rows) < 3:
        logger.error("유효 데이터가 부족합니다: %d개 (최소 3개 필요)", len(rows))
        return
    logger.info("로드된 데이터: %d개 셀", len(rows))

    x = np.array([r["spectral_w1"] for r in rows])
    y = np.array([r["rank_sensitivity"] for r in rows])
    models = [r["model"] for r in rows]
    domains = [r["domain"] for r in rows]

    # 모델별 색상/마커 매핑
    unique_models = sorted(set(models))
    cmap = plt.get_cmap("tab10")
    color_map = {m: cmap(i) for i, m in enumerate(unique_models)}
    markers = {"chronos": "o", "moirai": "s", "moment": "^", "timesfm": "D"}

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    domain_offset_map = {
        "ett_m1": (6, 4),
        "finance": (6, -10),
        "smd": (6, 8),
    }

    for xi, yi, mi, di in zip(x, y, models, domains):
        ax.scatter(
            xi, yi,
            c=[color_map[mi]],
            marker=markers.get(mi, "o"),
            s=70, edgecolor="black", linewidth=0.5,
            zorder=3,
        )
        offset = domain_offset_map.get(di, (6, 4))
        ax.annotate(
            di, (xi, yi),
            textcoords="offset points",
            xytext=offset,
            fontsize=7.5,
            color=color_map[mi],
            zorder=4,
        )

    # 회귀선 + CI 밴드
    try:
        x_line, y_line, ci_lo, ci_hi = _regression_with_ci(x, y)
        ax.plot(x_line, y_line, "k-", linewidth=1.5, zorder=2, label="Regression")
        ax.fill_between(x_line, ci_lo, ci_hi, color="gray", alpha=0.2, label="95% CI")
    except ValueError as e:
        logger.warning("회귀선/CI를 계산할 수 없습니다: %s", e)

    # 축 라벨
    ax.set_xlabel("Spectral W$_1$ (train$\\to$test)")
    ax.set_ylabel("Rank Sensitivity ($\\sigma$ MAE)")

    # 범례: 모델별 한 번만
    handles, labels = ax.get_legend_handles_labels()
    # 모델별 범례 항목 추가
    for m in unique_models:
        ax.scatter([], [], c=[color_map[m]], marker=markers.get(m, "o"),
                   s=70, edgecolor="black", linewidth=0.5, label=m.capitalize())
    handles2, labels2 = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels2, handles2))
    ax.legend(by_label.values(), by_label.keys(), title="Model",
              frameon=False, loc="upper left")

    # 주석
    ax.text(
        0.97, 0.95,
        "$\\rho$=0.70, 95% CI [0.21, 0.91]\nn=12",
        transform=ax.transAxes, fontsize=9,
        ha="right", va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
    )

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "spectral_rank_scatter.pdf"
    png_path = output_dir / "spectral_rank_scatter.png"
    fig.tight_layout(pad=0.5)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight")
    plt.close(fig)
    logger.info("저장 완료: %s, %s", pdf_path, png_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
