from __future__ import annotations

"""Gradient probe 결과 시각화 스크립트.

`results/gradient_analysis/` 디렉토리의 JSON 결과를 읽어
NeurIPS 논문 품질의 3종 figure를 생성:
  - gradient_heatmap.pdf  : 레이어 × 실험셀 heatmap
  - gradient_distribution.pdf : 모델별 레이어 gradient 분포 grouped bar
  - gradient_entropy.pdf  : gradient entropy vs. final MAE 산점도

Usage:
    PYTHONPATH=. python scripts/plot_gradient_analysis.py
    PYTHONPATH=. python scripts/plot_gradient_analysis.py --input_dir results/gradient_analysis --output_dir results/gradient_analysis/figures
"""

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)

# ─── NeurIPS 스타일 설정 ───────────────────────────────────────────

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

# 논문 표시용 이름 매핑
MODEL_NAMES: dict[str, str] = {
    "chronos": "Chronos",
    "moment": "MOMENT",
    "moirai": "Moirai",
    "timesfm": "TimesFM",
}
DOMAIN_NAMES: dict[str, str] = {
    "ett_m1": "ETTm1",
    "ett_h1": "ETTh1",
    "finance": "Finance",
    "smd": "SMD",
    "physionet": "PhysioNet",
}
METHOD_NAMES: dict[str, str] = {
    "zero_shot": "Zero-shot",
    "head_only": "Head-only",
    "lora": "LoRA",
    "adapter": "Adapter",
    "full_fine_tuning": "Full-FT",
    "full_ft": "Full-FT",
    "prefix": "Prefix",
}
METHOD_COLORS: dict[str, str] = {
    "Zero-shot": "#4C72B0",
    "Head-only": "#55A868",
    "LoRA": "#C44E52",
    "Adapter": "#8172B3",
    "Full-FT": "#CCB974",
    "Prefix": "#64B5CD",
}
MODEL_COLORS: dict[str, str] = {
    "chronos": "#E24A33",
    "moment": "#348ABD",
    "moirai": "#988ED5",
    "timesfm": "#56B4E9",
}
METHOD_MARKERS: dict[str, str] = {
    "Zero-shot": "o",
    "Head-only": "s",
    "LoRA": "^",
    "Adapter": "D",
    "Full-FT": "P",
    "Prefix": "X",
}

_LAYER_INDEX_RE = re.compile(r"layer_(\d+)$")


# ─── 유틸리티 ────────────────────────────────────────────────────


def _sort_layer_key(key: str) -> tuple[int, str]:
    """레이어 키를 숫자 순서로 정렬하기 위한 보조 함수.

    Args:
        key: 'layer_N' 또는 'layer_other' 형식 문자열.

    Returns:
        (숫자 우선순위, 원본 키) 튜플.
    """
    match = _LAYER_INDEX_RE.match(key)
    if match:
        return (int(match.group(1)), key)
    return (10**9, key)


def _display_method(raw: str) -> str:
    """원시 method 이름을 논문용 표시 이름으로 변환.

    Args:
        raw: 원시 method 문자열.

    Returns:
        표시용 이름.
    """
    return METHOD_NAMES.get(raw, raw)


def _display_model(raw: str) -> str:
    """원시 model 이름을 논문용 표시 이름으로 변환.

    Args:
        raw: 원시 model 문자열.

    Returns:
        표시용 이름.
    """
    return MODEL_NAMES.get(raw, raw)


def _display_domain(raw: str) -> str:
    """원시 domain 이름을 논문용 표시 이름으로 변환.

    Args:
        raw: 원시 domain 문자열.

    Returns:
        표시용 이름.
    """
    return DOMAIN_NAMES.get(raw, raw)


# ─── 데이터 로딩 ─────────────────────────────────────────────────


def _load_records(input_dir: Path) -> list[dict[str, Any]]:
    """input_dir 내 모든 JSON probe 결과 파일을 로드.

    summary 파일과 비어있는 결과는 건너뜁니다.

    Args:
        input_dir: JSON 파일이 담긴 디렉토리.

    Returns:
        유효한 결과 레코드 리스트.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("probe_summary") or path.name.startswith("all_"):
            continue
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except Exception:
            logger.warning("JSON 파싱 실패, 건너뜀: %s", path)
            continue

        # 필수 키 확인
        if not all(k in rec for k in ("model", "method", "domain")):
            logger.warning("필수 키 누락, 건너뜀: %s", path)
            continue
        if not rec.get("layer_grad_norms"):
            logger.debug("layer_grad_norms 없음, 건너뜀: %s", path)
            continue

        records.append(rec)

    logger.info("로드된 레코드 수: %d", len(records))
    return records


def _aggregate_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """동일 (model, method, domain) 셀의 레코드를 시드 평균으로 집계.

    Args:
        records: 원시 probe 결과 리스트.

    Returns:
        (model, method, domain) → 집계 통계 딕셔너리.
        각 값은 {layer_key: mean_norm, ...} 형태의 'layer_means'와
        'mae' (있으면) 를 포함.
    """
    # (model, method, domain) → list of per-seed layer_grad_norms
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    mae_map: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for rec in records:
        cell_key = (rec["model"], rec["method"], rec["domain"])
        groups[cell_key].append(rec["layer_grad_norms"])
        mae = rec.get("mae") or rec.get("metrics", {}).get("mae")
        if mae is not None:
            mae_map[cell_key].append(float(mae))

    aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell_key, norm_list in groups.items():
        # 모든 시드에 걸쳐 동일한 레이어 키 수집
        all_layer_keys: set[str] = set()
        for lgn in norm_list:
            all_layer_keys.update(lgn.keys())

        layer_means: dict[str, float] = {}
        for lk in all_layer_keys:
            seed_means: list[float] = []
            for lgn in norm_list:
                if lk not in lgn:
                    continue
                m = lgn[lk].get("mean")
                if m is None:
                    continue
                m_f = float(m)
                if not math.isfinite(m_f):
                    continue
                seed_means.append(m_f)
            if seed_means:
                layer_means[lk] = float(np.mean(seed_means))

        mae_vals = mae_map.get(cell_key)
        agg: dict[str, Any] = {"layer_means": layer_means}
        if mae_vals:
            agg["mae"] = float(np.mean(mae_vals))

        aggregated[cell_key] = agg

    return aggregated


def _compute_entropy(layer_means: dict[str, float]) -> float:
    """레이어별 평균 gradient norm으로부터 엔트로피를 계산.

    p_i = layer_i_mean_norm / sum(all_layer_means)
    entropy = -sum(p_i * log(p_i))

    Args:
        layer_means: {layer_key: mean_norm} 딕셔너리.

    Returns:
        Shannon entropy 값 (nats). 유효한 레이어가 없으면 0.0.
    """
    values = np.array(
        [v for v in layer_means.values() if v > 0.0], dtype=np.float64
    )
    if values.size == 0:
        return 0.0
    total = values.sum()
    if total <= 0.0:
        return 0.0
    probs = values / total
    # 부동소수점 안전 처리
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _get_sorted_layers(layer_means: dict[str, float]) -> list[str]:
    """레이어 키를 숫자 순서로 정렬하여 반환.

    Args:
        layer_means: {layer_key: mean_norm} 딕셔너리.

    Returns:
        정렬된 레이어 키 리스트.
    """
    return sorted(layer_means.keys(), key=_sort_layer_key)


# ─── Figure 1: Gradient Heatmap ──────────────────────────────────


def fig1_gradient_heatmap(
    aggregated: dict[tuple[str, str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    """레이어 × 실험셀 gradient norm heatmap.

    행 = 레이어, 열 = model:method:domain 실험 셀.
    색상 = mean gradient L2 norm (log scale, diverging colormap).

    Args:
        aggregated: _aggregate_records() 결과.
        output_dir: figure 저장 디렉토리.
    """
    if not aggregated:
        logger.warning("집계 데이터가 없어 Figure 1을 건너뜁니다.")
        return

    # 모든 레이어 키 수집 및 정렬
    all_layers: set[str] = set()
    for agg in aggregated.values():
        all_layers.update(agg["layer_means"].keys())
    sorted_layers = sorted(all_layers, key=_sort_layer_key)

    if not sorted_layers:
        logger.warning("레이어 데이터 없음, Figure 1 건너뜀.")
        return

    # 셀 정렬: success(mean norm > 0) 먼저, 그 다음 failure
    cell_keys = sorted(aggregated.keys())

    # 성공/실패 분리 기준: layer_means의 평균이 임계값 이상인지
    # (실제 MAE 정보가 없을 경우 그라디언트 norm이 매우 낮은 셀을 failure로 간주)
    def _cell_mean_norm(ck: tuple[str, str, str]) -> float:
        lm = aggregated[ck]["layer_means"]
        vals = list(lm.values())
        return float(np.mean(vals)) if vals else 0.0

    overall_median = np.median([_cell_mean_norm(ck) for ck in cell_keys])
    success_cells = [ck for ck in cell_keys if _cell_mean_norm(ck) >= overall_median]
    failure_cells = [ck for ck in cell_keys if _cell_mean_norm(ck) < overall_median]
    ordered_cells = success_cells + failure_cells
    n_success = len(success_cells)

    n_rows = len(sorted_layers)
    n_cols = len(ordered_cells)

    # 행렬 구성 (log scale)
    grid = np.full((n_rows, n_cols), np.nan)
    for j, ck in enumerate(ordered_cells):
        lm = aggregated[ck]["layer_means"]
        for i, lk in enumerate(sorted_layers):
            if lk in lm and lm[lk] > 0.0:
                grid[i, j] = math.log10(lm[lk] + 1e-12)

    # figure 크기: 가로로 길게 (\linewidth로 페이지 폭 차지하도록)
    fig_width = max(6.5, n_cols * 1.05 + 1.5)
    fig_height = max(2.6, n_rows * 0.18 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")

    if np.all(np.isnan(grid)):
        vmin, vmax = -6.0, 0.0
    else:
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
    if vmax - vmin < 0.1:  # 동일하면 vcenter 강제 분리
        vmin, vmax = vmin - 0.1, vmax + 0.1
    vcenter = (vmin + vmax) / 2.0

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(
        grid,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    # NaN 셀에 회색 hatch
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(grid[i, j]):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, facecolor="#F0F0F0",
                    edgecolor="white", linewidth=0.3,
                    hatch="///", alpha=0.5,
                ))

    # 열 레이블: model:method:domain → 줄바꿈 포맷
    col_labels = [
        f"{_display_model(ck[0])}\n{_display_method(ck[1])}\n{_display_domain(ck[2])}"
        for ck in ordered_cells
    ]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=0, ha="center")

    # 행 레이블: layer_0, layer_1, ... (간격 조정)
    row_labels = [lk.replace("layer_", "L") for lk in sorted_layers]
    step = max(1, n_rows // 12)
    tick_positions = list(range(0, n_rows, step))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([row_labels[i] for i in tick_positions], fontsize=7)

    ax.set_xlabel("Experiment Cell (model : method : domain)", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.set_title(
        "Layer-wise Gradient L2 Norm (log₁₀ scale)", fontsize=10, fontweight="bold"
    )

    # 성공/실패 경계선
    if 0 < n_success < n_cols:
        ax.axvline(x=n_success - 0.5, color="black", linewidth=1.2, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log₁₀(mean grad norm)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    _save_figure(fig, output_dir / "gradient_heatmap")
    logger.info("Figure 1 저장 완료: gradient_heatmap.pdf")


# ─── Figure 2: Gradient Distribution ─────────────────────────────


def fig2_gradient_distribution(
    aggregated: dict[tuple[str, str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    """모델별 레이어 gradient 분포 grouped bar chart.

    3개 서브플롯 (Chronos, MOMENT, Moirai).
    각 패널: x축 = 레이어, bar 색상 = method.

    Args:
        aggregated: _aggregate_records() 결과.
        output_dir: figure 저장 디렉토리.
    """
    target_models = ["chronos", "moment", "moirai"]
    present_models = [m for m in target_models if any(ck[0] == m for ck in aggregated)]

    if not present_models:
        logger.warning("대상 모델 데이터 없음, Figure 2 건너뜀.")
        return

    n_panels = len(present_models)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels / 3 * 1.8, 2.6))
    if n_panels == 1:
        axes = [axes]

    for ax, model in zip(axes, present_models):
        # 이 모델에 해당하는 (method, domain) 조합 수집
        model_cells = {ck: v for ck, v in aggregated.items() if ck[0] == model}
        if not model_cells:
            ax.set_visible(False)
            continue

        # 레이어 키 통합
        all_layers: set[str] = set()
        for agg in model_cells.values():
            all_layers.update(agg["layer_means"].keys())
        sorted_layers = sorted(all_layers, key=_sort_layer_key)

        if not sorted_layers:
            ax.set_visible(False)
            continue

        # 고유 method 수집
        methods_present = sorted({ck[1] for ck in model_cells})
        n_methods = len(methods_present)
        n_layers = len(sorted_layers)

        bar_width = 0.8 / max(n_methods, 1)
        x = np.arange(n_layers)

        for midx, method in enumerate(methods_present):
            # 이 model×method 에 해당하는 모든 domain 값을 평균
            layer_vals: dict[str, list[float]] = defaultdict(list)
            for ck, agg in model_cells.items():
                if ck[1] != method:
                    continue
                for lk, v in agg["layer_means"].items():
                    layer_vals[lk].append(v)

            bar_means = np.array(
                [
                    float(np.mean(layer_vals[lk])) if layer_vals.get(lk) else 0.0
                    for lk in sorted_layers
                ]
            )

            display_method = _display_method(method)
            color = METHOD_COLORS.get(display_method, "#999999")
            offset = (midx - n_methods / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                bar_means,
                width=bar_width * 0.92,
                color=color,
                label=display_method,
                alpha=0.85,
                edgecolor="none",
            )

        ax.set_title(_display_model(model), fontweight="bold", fontsize=10)
        ax.set_xlabel("Layer", fontsize=8)
        if model == present_models[0]:
            ax.set_ylabel("Mean Gradient L2 Norm", fontsize=8)

        # x축 레이블: 레이어 수가 많으면 간격을 둠
        step = max(1, n_layers // 8)
        tick_positions = list(range(0, n_layers, step))
        ax.set_xticks([x[i] for i in tick_positions])
        ax.set_xticklabels(
            [sorted_layers[i].replace("layer_", "L") for i in tick_positions],
            fontsize=6,
        )
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            fontsize=6.5,
            loc="upper right",
            framealpha=0.7,
            edgecolor="none",
            handlelength=1.2,
        )

    fig.suptitle(
        "Gradient Distribution by Layer and Method\n(even distribution = good, concentrated = potential issue)",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir / "gradient_distribution")
    logger.info("Figure 2 저장 완료: gradient_distribution.pdf")


# ─── Figure 3: Gradient Entropy vs. MAE ──────────────────────────


def fig3_gradient_entropy(
    aggregated: dict[tuple[str, str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    """Gradient entropy vs. final MAE 산점도.

    x = gradient entropy (레이어 분포 균일도)
    y = final MAE
    색상 = model, 마커 모양 = method
    Pearson 상관계수 annotation 포함.

    MAE 데이터가 없는 셀은 제외됩니다.

    Args:
        aggregated: _aggregate_records() 결과.
        output_dir: figure 저장 디렉토리.
    """
    # MAE 있는 데이터 포인트 수집
    points: list[dict[str, Any]] = []
    for (model, method, domain), agg in aggregated.items():
        if "mae" not in agg:
            continue
        entropy = _compute_entropy(agg["layer_means"])
        points.append(
            {
                "model": model,
                "method": method,
                "domain": domain,
                "entropy": entropy,
                "mae": agg["mae"],
            }
        )

    if len(points) < 2:
        logger.warning(
            "MAE 데이터를 가진 포인트가 부족합니다 (%d개). "
            "Figure 3을 건너뜁니다. gradient_probe 결과에 'mae' 키를 포함시키세요.",
            len(points),
        )
        _fig3_placeholder(aggregated, output_dir)
        return

    entropies = np.array([p["entropy"] for p in points])
    maes = np.array([p["mae"] for p in points])

    # Pearson 상관계수
    if entropies.std() > 1e-12 and maes.std() > 1e-12:
        corr = float(np.corrcoef(entropies, maes)[0, 1])
    else:
        corr = 0.0

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # 고유 model 및 method 목록
    all_models = sorted({p["model"] for p in points})
    all_methods = sorted({p["method"] for p in points})

    for p in points:
        model = p["model"]
        method = p["method"]
        display_method = _display_method(method)
        color = MODEL_COLORS.get(model, "#999999")
        marker = METHOD_MARKERS.get(display_method, "o")
        ax.scatter(
            p["entropy"],
            p["mae"],
            color=color,
            marker=marker,
            s=40,
            alpha=0.8,
            linewidths=0.4,
            edgecolors="white",
            zorder=3,
        )

    # 추세선 (엔트로피 분산이 충분할 때만)
    if len(points) >= 3 and entropies.std() > 1e-9:
        slope, intercept = np.polyfit(entropies, maes, 1)
        x_line = np.linspace(entropies.min(), entropies.max(), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#666666",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
            zorder=2,
        )

    # Pearson r annotation
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {corr:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"),
    )

    ax.set_xlabel("Gradient Entropy (higher = more uniform across layers)", fontsize=8)
    ax.set_ylabel("Final MAE", fontsize=8)
    ax.set_title("Gradient Entropy vs. Final MAE", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 범례: model (색상)
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MODEL_COLORS.get(m, "#999999"),
            markersize=6,
            label=_display_model(m),
        )
        for m in all_models
    ]
    # 범례: method (마커 모양)
    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS.get(_display_method(mt), "o"),
            color="w",
            markerfacecolor="#555555",
            markersize=6,
            label=_display_method(mt),
        )
        for mt in all_methods
    ]

    legend1 = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper right",
        fontsize=7,
        title_fontsize=7,
        framealpha=0.8,
        edgecolor="#CCCCCC",
        handlelength=0.8,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=method_handles,
        title="Method",
        loc="lower right",
        fontsize=7,
        title_fontsize=7,
        framealpha=0.8,
        edgecolor="#CCCCCC",
        handlelength=0.8,
    )

    fig.tight_layout()
    _save_figure(fig, output_dir / "gradient_entropy")
    logger.info("Figure 3 저장 완료: gradient_entropy.pdf")


def _fig3_placeholder(
    aggregated: dict[tuple[str, str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    """MAE 데이터 없을 때 entropy-only 산점도 placeholder 생성.

    Args:
        aggregated: _aggregate_records() 결과.
        output_dir: figure 저장 디렉토리.
    """
    if not aggregated:
        logger.warning("데이터가 없어 Figure 3 placeholder도 생성하지 않습니다.")
        return

    entropies = [
        _compute_entropy(agg["layer_means"]) for agg in aggregated.values()
    ]
    cell_labels = [
        f"{_display_model(ck[0])}\n{_display_method(ck[1])}"
        for ck in aggregated.keys()
    ]

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    x_pos = np.arange(len(entropies))
    colors_bar = [
        MODEL_COLORS.get(ck[0], "#999999") for ck in aggregated.keys()
    ]
    ax.bar(x_pos, entropies, color=colors_bar, alpha=0.8, edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cell_labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Gradient Entropy", fontsize=8)
    ax.set_title(
        "Gradient Entropy per Cell\n(MAE data unavailable — scatter plot requires 'mae' in probe JSON)",
        fontsize=9,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, output_dir / "gradient_entropy")
    logger.info("Figure 3 placeholder 저장 완료: gradient_entropy.pdf")


# ─── 저장 헬퍼 ───────────────────────────────────────────────────


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    """figure를 PDF 및 PNG로 저장.

    Args:
        fig: matplotlib Figure 객체.
        base_path: 확장자 없는 저장 경로.
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base_path) + ".pdf")
    fig.savefig(str(base_path) + ".png")
    plt.close(fig)


# ─── 초기화 ──────────────────────────────────────────────────────


def _setup() -> None:
    """로깅 및 matplotlib 스타일 초기화."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    plt.rcParams.update(NEURIPS_RC)


# ─── CLI ─────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """CLI 인자 파싱.

    Returns:
        파싱된 argparse 네임스페이스.
    """
    parser = argparse.ArgumentParser(
        description="Gradient probe JSON 결과를 읽어 NeurIPS 품질 figure를 생성합니다."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/gradient_analysis",
        help="gradient probe JSON 파일이 있는 디렉토리 (기본값: results/gradient_analysis)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gradient_analysis/figures",
        help="figure 저장 디렉토리 (기본값: results/gradient_analysis/figures)",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="1,2,3",
        help="생성할 figure 번호 (콤마 구분). 기본값: 1,2,3",
    )
    return parser.parse_args()


def main() -> None:
    """Gradient 분석 figure 생성 메인 엔트리포인트."""
    _setup()
    args = _parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_figures = {int(x.strip()) for x in args.figures.split(",") if x.strip()}

    if not input_dir.exists():
        logger.error("입력 디렉토리가 존재하지 않습니다: %s", input_dir)
        return

    records = _load_records(input_dir)

    if not records:
        logger.warning(
            "유효한 레코드가 없습니다. '%s' 에 gradient probe JSON 파일이 있는지 확인하세요.",
            input_dir,
        )
        return

    aggregated = _aggregate_records(records)
    logger.info("집계된 셀 수: %d", len(aggregated))

    if 1 in requested_figures:
        fig1_gradient_heatmap(aggregated, output_dir)
    if 2 in requested_figures:
        fig2_gradient_distribution(aggregated, output_dir)
    if 3 in requested_figures:
        fig3_gradient_entropy(aggregated, output_dir)

    logger.info("모든 figure 생성 완료: %s", output_dir)


if __name__ == "__main__":
    main()
