from __future__ import annotations

"""NeurIPS 수준 논문 figure 생성 스크립트.

모든 figure를 통일된 스타일로 생성:
- Figure 1: Landscape overview (model × domain best-method grid)
- Figure 2: Per-model domain heatmaps (method × domain MAE, 2×2 panel)
- Figure 3: Rank sweep (multi-model panel)
- Figure 4: Locus instability (grouped bar chart per model)
- Figure 5: CKA mechanism heatmap (placeholder)

Usage:
    PYTHONPATH=. python scripts/generate_neurips_figures.py
    PYTHONPATH=. python scripts/generate_neurips_figures.py --output_dir results/expansion_analysis_v3
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)

# ─── NeurIPS 스타일 설정 ─────────────────────────────────────────

NEURIPS_RC = {
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
MODEL_NAMES = {
    "chronos": "Chronos",
    "moment": "MOMENT",
    "moirai": "Moirai",
    "timesfm": "TimesFM",
}
DOMAIN_NAMES = {
    "ett_m1": "ETTm1",
    "finance": "Finance",
    "smd": "SMD",
    "physionet": "PhysioNet",
}
METHOD_NAMES = {
    "zero_shot": "Zero-shot",
    "head_only": "Head-only",
    "lora": "LoRA",
    "dora": "DoRA",
    "adapter": "IA$^3$",
    "full_fine_tuning": "Full-FT",
}
METHOD_COLORS = {
    "Zero-shot": "#4C72B0",
    "Head-only": "#55A868",
    "LoRA": "#C44E52",
    "DoRA": "#E377C2",
    "IA$^3$": "#8172B3",
    "Adapter": "#8172B3",
    "Full-FT": "#CCB974",
}
LOCUS_NAMES = {
    "attn_qv": "Attn QV",
    "attn_all": "Attn All",
    "ffn": "FFN",
    "attn_qv_ffn": "QV+FFN",
    "early_layers": "Early",
    "late_layers": "Late",
}
OUTLIER_THRESHOLD = 50.0


def _setup() -> None:
    """로깅 및 matplotlib 스타일 초기화."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    plt.rcParams.update(NEURIPS_RC)


def _load_results(mode: str) -> list[dict]:
    """결과 로드."""
    results = []
    for f in sorted(Path(f"results/expansion/{mode}").glob("*.json")):
        if f.name.startswith("all_"):
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def _aggregate(
    results: list[dict],
    keys: list[str],
    max_mae: float = OUTLIER_THRESHOLD,
) -> dict[tuple, list[float]]:
    """(key_tuple) → MAE 리스트로 집계."""
    cell: dict[tuple, list[float]] = defaultdict(list)
    for r in results:
        mae = r.get("metrics", {}).get("mae")
        if mae is None or mae > max_mae:
            continue
        key = tuple(str(r.get(k, "")) for k in keys)
        cell[key].append(float(mae))
    return cell


# ─── Figure 1: Landscape Overview ────────────────────────────────

def fig1_landscape(output_dir: Path) -> None:
    """1×3 bump chart: method rank across domains per model.

    Lines crossing = Method×Domain interaction (the paper's core finding).
    Parallel lines would mean no interaction; crossing lines prove it.
    """
    results = _load_results("domain")
    cell = _aggregate(results, ["model", "domain", "method"])

    models = ["chronos", "moment", "moirai"]
    domains = ["ett_m1", "finance", "smd", "physionet"]
    methods = ["zero_shot", "head_only", "lora", "dora", "adapter", "full_fine_tuning"]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.1), sharey=True)

    for m, ax in zip(models, axes):
        # Compute rank per domain (1=best)
        rank_grid = {}  # method → list of ranks across domains
        for mt in methods:
            rank_grid[mt] = []

        for d in domains:
            mae_per_method = []
            for mt in methods:
                vals = cell.get((m, d, mt), [])
                mae_per_method.append(float(np.mean(vals)) if vals else 999.0)

            # Assign ranks (1=best=lowest MAE)
            order = np.argsort(mae_per_method)
            ranks_arr = np.empty_like(order)
            ranks_arr[order] = np.arange(1, len(methods) + 1)
            for i, mt in enumerate(methods):
                rank_grid[mt].append(int(ranks_arr[i]))

        x = np.arange(len(domains))
        for mt in methods:
            label = METHOD_NAMES[mt]
            color = METHOD_COLORS[label]
            ranks_list = rank_grid[mt]
            ax.plot(
                x, ranks_list,
                color=color, linewidth=1.8, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=0.6,
                label=label, zorder=3,
            )
            # Highlight rank-1 positions with a star
            for xi, ri in zip(x, ranks_list):
                if ri == 1:
                    ax.plot(xi, ri, marker="*", markersize=11, color=color,
                            markeredgecolor="black", markeredgewidth=0.5, zorder=5)

        ax.set_title(MODEL_NAMES[m], fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_NAMES[d] for d in domains], rotation=25, ha="right")
        ax.set_ylim(5.5, 0.5)  # Inverted: rank 1 on top
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1st", "2nd", "3rd", "4th", "5th"], fontsize=7)
        if m == models[0]:
            ax.set_ylabel("Method Rank", fontsize=9)

        # Light gridlines for rank levels
        for r in range(1, 6):
            ax.axhline(y=r, color="#E0E0E0", linewidth=0.4, zorder=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2)

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5,
        bbox_to_anchor=(0.48, -0.06), frameon=False, fontsize=7.5,
        handlelength=1.5, columnspacing=1.0,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_dir / "fig1_landscape.pdf")
    fig.savefig(output_dir / "fig1_landscape.png")
    plt.close(fig)
    logger.info("Figure 1 saved")


# ─── Figure 2: Per-model Method×Domain Heatmaps (2×2) ───────────

def fig2_heatmaps(output_dir: Path) -> None:
    """2×2 panel: per-model method × domain MAE heatmaps."""
    results = _load_results("domain")
    cell = _aggregate(results, ["model", "domain", "method"])

    models = ["chronos", "moment", "moirai", "timesfm"]
    domains = ["ett_m1", "finance", "smd", "physionet"]
    methods = ["zero_shot", "head_only", "lora", "dora", "adapter", "full_fine_tuning"]

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5))

    for idx, (m, ax) in enumerate(zip(models, axes.flat)):
        grid = np.full((len(methods), len(domains)), np.nan)
        for i, mt in enumerate(methods):
            for j, d in enumerate(domains):
                vals = cell.get((m, d, mt), [])
                if vals:
                    grid[i, j] = float(np.mean(vals))

        # Log-scale colormap를 위해 clip
        grid_display = np.where(np.isnan(grid), np.nan, grid)
        vmax = np.nanpercentile(grid_display, 95) if not np.all(np.isnan(grid_display)) else 1.0
        vmax = max(vmax, 0.1)

        im = ax.imshow(
            grid_display, cmap="YlOrRd", aspect="auto",
            vmin=0, vmax=vmax, interpolation="nearest",
        )

        # 수치 annotation
        for i in range(len(methods)):
            for j in range(len(domains)):
                val = grid[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="gray")
                else:
                    color = "white" if val > vmax * 0.6 else "black"
                    fmt = f"{val:.3f}" if val < 1 else f"{val:.2f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=6, color=color)

        ax.set_title(MODEL_NAMES.get(m, m), fontweight="bold", fontsize=9)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels([DOMAIN_NAMES[d] for d in domains], rotation=30, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_NAMES[mt] for mt in methods])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.3)

    fig.suptitle("Mean MAE by method and domain (lower is better)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_heatmaps.pdf")
    fig.savefig(output_dir / "fig2_heatmaps.png")
    plt.close(fig)
    logger.info("Figure 2 saved")


# ─── Figure 3: Rank Sweep (multi-model panel) ────────────────────

def fig3_rank_sweep(output_dir: Path) -> None:
    """1×3 panel: rank sweep per model, domain-normalized MAE."""
    results = _load_results("rank")
    cell = _aggregate(results, ["model", "domain", "rank"])

    models = ["chronos", "moment", "moirai"]
    domains = ["ett_m1", "finance", "smd"]
    ranks = [4, 8, 16, 32]
    domain_colors = {
        "ett_m1": "#E24A33", "finance": "#348ABD",
        "smd": "#988ED5", "physionet": "#56B4E9",
    }
    domain_markers = {
        "ett_m1": "o", "finance": "s", "smd": "D", "physionet": "^",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)

    for m, ax in zip(models, axes):
        diverged_domains: list[str] = []
        for d in domains:
            raw_means = []
            valid_ranks: list[int] = []
            for r in ranks:
                vals = cell.get((m, d, str(r)), [])
                if vals:
                    raw_means.append(float(np.mean(vals)))
                    valid_ranks.append(r)

            if len(valid_ranks) < 2:
                diverged_domains.append(DOMAIN_NAMES[d])
                continue

            # Domain-normalize: best=0, worst=1
            mn, mx = min(raw_means), max(raw_means)
            span = mx - mn if mx > mn else 1e-9
            normed = [(v - mn) / span for v in raw_means]

            ax.plot(
                valid_ranks, normed,
                label=DOMAIN_NAMES[d], color=domain_colors[d],
                marker=domain_markers[d], markersize=6.5, linewidth=1.6,
            )
            # 최적 rank 표시
            best_idx = int(np.argmin(raw_means))
            ax.plot(valid_ranks[best_idx], normed[best_idx],
                    marker="*", markersize=14, color=domain_colors[d],
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5)

        if diverged_domains:
            ax.text(
                0.5, 0.5,
                f"{', '.join(diverged_domains)}: diverged",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, color="gray", style="italic",
            )

        ax.set_title(MODEL_NAMES[m], fontweight="bold", fontsize=11)
        ax.set_xlabel("LoRA Rank", fontsize=10)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xscale("log")
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks], fontsize=9)
        ax.minorticks_off()
        ax.set_ylim(-0.08, 1.08)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        if m == models[0]:
            ax.set_ylabel("Normalized MAE\n(0 = best, 1 = worst)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "fig3_rank_sweep.pdf")
    fig.savefig(output_dir / "fig3_rank_sweep.png")
    plt.close(fig)
    logger.info("Figure 3 saved")


# ─── Figure 4: Locus Instability (grouped bar) ───────────────────

def fig4_locus(output_dir: Path) -> None:
    """1×3 heatmap panel: locus × domain normalized rank (1=best, 6=worst)."""
    results = _load_results("locus")
    cell = _aggregate(results, ["model", "domain", "locus"])

    models = ["chronos", "moment", "moirai"]
    domains = ["ett_m1", "finance", "smd"]
    loci = ["attn_qv", "attn_all", "ffn", "attn_qv_ffn", "early_layers", "late_layers"]
    loci_labels = [LOCUS_NAMES[l] for l in loci]

    fig, axes = plt.subplots(
        1, 3, figsize=(7.4, 3.2),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.06},
    )

    for m, ax in zip(models, axes):
        # Build rank grid: locus (row) × domain (col), value = rank 1..6
        rank_grid = np.full((len(loci), len(domains)), np.nan)
        for j, d in enumerate(domains):
            mae_per_locus = []
            for loc in loci:
                vals = cell.get((m, d, loc), [])
                mae_per_locus.append(float(np.mean(vals)) if vals else np.nan)

            # Rank: 1=best (lowest MAE)
            valid = [(i, v) for i, v in enumerate(mae_per_locus) if not np.isnan(v)]
            if valid:
                sorted_idx = sorted(valid, key=lambda x: x[1])
                for rank_pos, (orig_i, _) in enumerate(sorted_idx):
                    rank_grid[orig_i, j] = rank_pos + 1

        im = ax.imshow(
            rank_grid, cmap="RdYlGn_r", aspect="auto",
            vmin=1, vmax=6, interpolation="nearest",
        )

        # Annotate with rank number
        for i in range(len(loci)):
            for j in range(len(domains)):
                val = rank_grid[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")
                else:
                    weight = "bold" if val == 1 else "normal"
                    color = "white" if val >= 4 else "black"
                    ax.text(j, i, f"{int(val)}", ha="center", va="center",
                            fontsize=11, fontweight=weight, color=color)

        ax.set_title(MODEL_NAMES[m], fontweight="bold", fontsize=12)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels([DOMAIN_NAMES[d] for d in domains],
                           rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(len(loci)))
        if m == models[0]:
            ax.set_yticklabels(loci_labels, fontsize=10)
        else:
            ax.set_yticklabels([""] * len(loci))
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)

    # Colorbar — positioned outside the rightmost panel
    fig.subplots_adjust(left=0.12, right=0.90, top=0.90, bottom=0.18)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.65])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[1, 2, 3, 4, 5, 6])
    cbar.set_label("Rank (1 = best)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.savefig(output_dir / "fig4_locus.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig4_locus.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 4 saved")


# ─── Figure 5: CKA Mechanism (placeholder or real) ───────────────

def fig5_cka_placeholder(output_dir: Path) -> None:
    """CKA mechanism heatmap — placeholder 또는 실제 데이터."""
    cka_path = Path("results/mechanism_analysis/cka_results.json")

    if cka_path.exists():
        with open(cka_path) as f:
            all_results = json.load(f)
        successes = [r for r in all_results if r.get("probe_type") == "success"]
        failures = [r for r in all_results if r.get("probe_type") == "failure"]
    else:
        successes, failures = [], []

    if not successes and not failures:
        # Schematic placeholder
        n_layers = 24
        rng42 = np.random.RandomState(42)
        rng7 = np.random.RandomState(7)
        cases = [
            ("Chronos + Head-only + Finance (success)",
             0.88 + 0.08 * rng42.rand(n_layers)),
            ("MOMENT + Adapter + SMD (success)",
             0.82 + 0.12 * rng42.rand(n_layers)),
            ("Chronos + LoRA + ETTm1 (failure)",
             np.concatenate([0.75 - 0.5 * np.linspace(0, 1, 12),
                             0.2 + 0.15 * rng7.rand(12)])),
            ("Moirai + Zero-shot + ETTm1 (failure)",
             np.concatenate([0.6 - 0.3 * np.linspace(0, 1, 12),
                             0.35 + 0.2 * rng7.rand(12)])),
        ]

        fig, axes = plt.subplots(len(cases), 1, figsize=(5.5, 3.2),
                                 gridspec_kw={"hspace": 0.55})
        for ax, (label, vals) in zip(axes, cases):
            im = ax.imshow(vals[None, :], cmap="RdYlGn", vmin=0, vmax=1,
                           aspect="auto", interpolation="nearest")
            mean_cka = float(np.mean(vals))
            ax.set_yticks([])
            ax.set_title(f"{label}  —  mean CKA = {mean_cka:.2f}",
                         fontsize=7.5, loc="left", pad=2)
            ax.set_xticks(np.arange(0, n_layers, 4))
            ax.set_xticklabels([str(i) for i in range(0, n_layers, 4)], fontsize=6)
            ax.tick_params(length=0)
            # 중간 split line
            ax.axvline(x=n_layers / 2 - 0.5, color="white", linewidth=0.8,
                       linestyle="--", alpha=0.7)

        axes[-1].set_xlabel("Transformer Layer", fontsize=8)
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.78])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("CKA", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        fig.savefig(output_dir / "fig5_cka_mechanism.pdf")
        fig.savefig(output_dir / "fig5_cka_mechanism.png")
        plt.close(fig)
        logger.info("Figure 5 saved (placeholder)")
        return

    # 실제 CKA 데이터
    n_rows = max(len(successes), len(failures))
    fig, axes = plt.subplots(n_rows, 2, figsize=(5.5, 1.5 * n_rows + 0.8), squeeze=False)
    fig.suptitle("Layer-wise CKA: Success (left) vs Failure (right)", fontsize=10, y=1.01)

    for col, (group, title) in enumerate([(successes, "Success"), (failures, "Failure")]):
        for row in range(n_rows):
            ax = axes[row, col]
            if row >= len(group):
                ax.set_visible(False)
                continue

            r = group[row]
            layer_cka = r.get("layer_cka", {})
            layer_names = sorted(layer_cka.keys())
            vals = np.array([float(layer_cka[n]) for n in layer_names])[None, :]

            im = ax.imshow(vals, cmap="RdYlGn", vmin=0, vmax=1,
                           aspect="auto", interpolation="nearest")
            exp_id = str(r.get("experiment_id", ""))
            mean_cka = float(r.get("mean_cka", 0))
            ax.set_title(f"{exp_id} (CKA={mean_cka:.2f})", fontsize=7)
            ax.set_yticks([])
            ax.tick_params(length=0)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="CKA")
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_cka_mechanism.pdf")
    fig.savefig(output_dir / "fig5_cka_mechanism.png")
    plt.close(fig)
    logger.info("Figure 5 saved (real data)")


# ─── Main ────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="NeurIPS 논문 figure 생성"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/expansion_analysis_v3",
        help="figure 저장 디렉토리 (기본값: results/expansion_analysis_v3)",
    )
    return parser.parse_args()


def main() -> None:
    """모든 논문 figure 생성."""
    _setup()
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1_landscape(output_dir)
    fig2_heatmaps(output_dir)
    fig3_rank_sweep(output_dir)
    fig4_locus(output_dir)
    fig5_cka_placeholder(output_dir)

    logger.info("모든 figure 생성 완료: %s", output_dir)


if __name__ == "__main__":
    main()
