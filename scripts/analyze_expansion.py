"""Phase 2 expansion analysis with 2-way ANOVA interaction tests.

Analyzes multi-domain method comparison, LoRA rank sweep, and locus stability.
- Per-model 2-way ANOVA (method × domain) with interaction term
- Per-model Kruskal-Wallis within-method shift analysis
- LoRA rank analysis (rank vs MAE curves)
- Cross-domain locus stability analysis
- JSON output + PNG charts
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
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Phase 2 expansion analysis with 2-way ANOVA"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["domain", "rank", "locus", "all"],
        default="all",
        help="Analysis mode: domain (method×domain), rank (LoRA rank sweep), "
        "locus (locus stability), or all",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/expansion",
        help="Input results directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/expansion_analysis",
        help="Output analysis directory",
    )
    return parser.parse_args()


# ─── Statistical Functions ────────────────────────────────────────


def _load_results(result_dir: Path) -> list[dict[str, Any]]:
    """Load individual JSON results, skip all_results.json and corrupt files.

    Args:
        result_dir: Results directory path.

    Returns:
        List of result dictionaries.
    """
    results: list[dict[str, Any]] = []

    # Try all_results.json first
    all_results_path = result_dir / "all_results.json"
    if all_results_path.exists():
        try:
            with open(all_results_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results = loaded
                    logger.info(
                        "Loaded all_results.json: %d results (%s)",
                        len(results),
                        all_results_path,
                    )
                    return results
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to load all_results.json: %s", exc)

    # Load individual JSON files (search recursively)
    for json_file in sorted(result_dir.rglob("*.json")):
        if json_file.name == "all_results.json":
            continue
        try:
            with open(json_file, "r") as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Skipped corrupt file %s: %s", json_file, exc)

    logger.info("Loaded individual results: %d files (%s)", len(results), result_dir)
    return results


# ─── Domain Mode Analysis ─────────────────────────────────────────


def _analyze_domain_mode(
    results: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Analyze multi-domain method comparison with 2-way ANOVA.

    Args:
        results: Experiment results list.
        output_dir: Output directory.

    Returns:
        Analysis results dictionary.
    """
    if not results:
        logger.warning("No results for domain mode analysis.")
        return {}

    # Extract unique values
    models = sorted(set(r["model"] for r in results))
    methods = sorted(set(r["method"] for r in results))
    domains = sorted(set(r["domain"] for r in results))

    analysis: dict[str, Any] = {
        "mode": "domain",
        "models": models,
        "methods": methods,
        "domains": domains,
        "n_results": len(results),
    }

    for model_name in models:
        model_results = [r for r in results if r["model"] == model_name]
        model_analysis: dict[str, Any] = {}

        # ─── Per-model 2-way ANOVA: method × domain ──────────────
        # Build groups for ANOVA
        groups_by_method: dict[str, list[float]] = {}
        groups_by_domain: dict[str, list[float]] = {}
        groups_by_method_domain: dict[tuple[str, str], list[float]] = {}

        for r in model_results:
            mae_val = r["metrics"]["mae"]
            groups_by_method.setdefault(r["method"], []).append(mae_val)
            groups_by_domain.setdefault(r["domain"], []).append(mae_val)
            key = (r["method"], r["domain"])
            groups_by_method_domain.setdefault(key, []).append(mae_val)

        # One-way ANOVA: method effect
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

        # One-way ANOVA: domain effect
        domain_groups = [np.array(v) for v in groups_by_domain.values() if len(v) >= 2]
        if len(domain_groups) >= 2:
            f_domain, p_domain = stats.f_oneway(*domain_groups)
            model_analysis["anova_domain"] = {
                "F": float(f_domain),
                "p": float(p_domain),
                "significant": bool(p_domain < 0.05),
            }
        else:
            model_analysis["anova_domain"] = {"F": 0.0, "p": 1.0, "significant": False}

        # ─── Within-method domain effect (Kruskal-Wallis) ────────
        within_method_domain: dict[str, dict[str, Any]] = {}
        for method in methods:
            method_domain_groups: dict[str, list[float]] = {}
            for r in model_results:
                if r["method"] == method:
                    method_domain_groups.setdefault(r["domain"], []).append(
                        r["metrics"]["mae"]
                    )

            # Kruskal-Wallis test (non-parametric)
            kw_groups = [
                np.array(v) for v in method_domain_groups.values() if len(v) >= 2
            ]
            if len(kw_groups) >= 2:
                try:
                    h_stat, p_kw = stats.kruskal(*kw_groups)
                    within_method_domain[method] = {
                        "H": float(h_stat),
                        "p": float(p_kw),
                        "significant": bool(p_kw < 0.05),
                    }
                except ValueError as exc:
                    # All numbers identical
                    logger.debug(
                        "Kruskal-Wallis failed for %s/%s: %s", model_name, method, exc
                    )
                    within_method_domain[method] = {
                        "H": 0.0,
                        "p": 1.0,
                        "significant": False,
                    }
            else:
                within_method_domain[method] = {
                    "H": 0.0,
                    "p": 1.0,
                    "significant": False,
                }

        model_analysis["within_method_domain_effect"] = within_method_domain

        # ─── Best method per domain ──────────────────────────────
        best_method_per_domain: dict[str, str] = {}
        for domain in domains:
            domain_results = [r for r in model_results if r["domain"] == domain]
            if domain_results:
                method_maes: dict[str, list[float]] = {}
                for r in domain_results:
                    method_maes.setdefault(r["method"], []).append(r["metrics"]["mae"])

                best_method = min(
                    method_maes.keys(),
                    key=lambda m: float(np.mean(method_maes[m])),
                )
                best_method_per_domain[domain] = best_method

        model_analysis["best_method_per_domain"] = best_method_per_domain

        # ─── Method ranking stability across domains ─────────────
        method_rankings: dict[str, list[str]] = {}
        for domain in domains:
            domain_results = [r for r in model_results if r["domain"] == domain]
            if domain_results:
                method_maes: dict[str, float] = {}
                for r in domain_results:
                    if r["method"] not in method_maes:
                        method_maes[r["method"]] = float(
                            np.mean(
                                [
                                    rr["metrics"]["mae"]
                                    for rr in domain_results
                                    if rr["method"] == r["method"]
                                ]
                            )
                        )

                ranked = sorted(method_maes.keys(), key=lambda m: method_maes[m])
                method_rankings[domain] = ranked

        # Kendall's tau: ranking correlation across domains
        tau_results: dict[str, dict[str, Any]] = {}
        if len(domains) >= 2:
            common_methods = set(methods)
            ranks_by_domain: dict[str, list[int]] = {}

            for domain in domains:
                ranked = method_rankings.get(domain, [])
                rank_map = {m: i for i, m in enumerate(ranked)}
                ranks_by_domain[domain] = [
                    rank_map.get(m, len(methods)) for m in sorted(common_methods)
                ]

            for d1, d2 in combinations(domains, 2):
                r1 = np.array(ranks_by_domain.get(d1, []))
                r2 = np.array(ranks_by_domain.get(d2, []))
                if r1.size >= 3 and r2.size >= 3:
                    try:
                        tau, p_tau = stats.kendalltau(r1, r2)
                        key = f"{d1}_vs_{d2}"
                        tau_results[key] = {
                            "tau": float(tau),
                            "p_value": float(p_tau),
                            "significant": bool(p_tau < 0.05),
                        }
                    except ValueError as exc:
                        logger.debug("Kendall's tau failed for %s: %s", key, exc)

        model_analysis["method_ranking_stability"] = tau_results
        model_analysis["method_rankings"] = method_rankings

        analysis[model_name] = model_analysis

    # ─── Visualization ────────────────────────────────────────────
    try:
        _plot_domain_mode(results, analysis, output_dir)
    except ImportError as exc:
        logger.warning("matplotlib not available, skipping visualization: %s", exc)

    return analysis


def _plot_domain_mode(
    results: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate domain mode visualizations.

    Args:
        results: Experiment results list.
        analysis: Analysis results dictionary.
        output_dir: Output directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = analysis.get("models", [])
    methods = analysis.get("methods", [])
    domains = analysis.get("domains", [])

    # ─── Heatmap: method × domain → MAE ──────────────────────────
    for model_name in models:
        model_data = analysis.get(model_name, {})
        method_rankings = model_data.get("method_rankings", {})

        if not method_rankings:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        data_matrix = np.zeros((len(methods), len(domains)))

        for j, domain in enumerate(domains):
            domain_results = [
                r for r in results if r["model"] == model_name and r["domain"] == domain
            ]
            if domain_results:
                method_maes: dict[str, float] = {}
                for r in domain_results:
                    if r["method"] not in method_maes:
                        method_maes[r["method"]] = float(
                            np.mean(
                                [
                                    rr["metrics"]["mae"]
                                    for rr in domain_results
                                    if rr["method"] == r["method"]
                                ]
                            )
                        )

                for i, method in enumerate(methods):
                    data_matrix[i, j] = method_maes.get(method, 0.0)

        im = ax.imshow(data_matrix, cmap="YlOrRd_r", aspect="auto")
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, rotation=45, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)

        for i in range(len(methods)):
            for j in range(len(domains)):
                ax.text(
                    j,
                    i,
                    f"{data_matrix[i, j]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if data_matrix[i, j] < 0.5 else "white",
                )

        plt.colorbar(im, ax=ax, label="MAE (lower is better)")
        ax.set_title(f"Phase 2: Method × Domain ({model_name})")
        ax.set_xlabel("Domain")
        ax.set_ylabel("Adaptation Method")
        fig.tight_layout()
        fig.savefig(output_dir / f"domain_heatmap_{model_name}.png", dpi=300)
        plt.close(fig)
        logger.info(
            "Saved heatmap: %s", output_dir / f"domain_heatmap_{model_name}.png"
        )

    # ─── Bar chart: method MAE per model ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    method_mae_mean: list[float] = []
    method_mae_std: list[float] = []
    method_labels: list[str] = []

    for method in methods:
        maes = [r["metrics"]["mae"] for r in results if r["method"] == method]
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
        ax.set_title("Phase 2: Method MAE (all models & domains)")
        fig.tight_layout()
        fig.savefig(output_dir / "domain_bar_methods.png", dpi=300)
        plt.close(fig)
        logger.info("Saved bar chart: %s", output_dir / "domain_bar_methods.png")


# ─── Rank Mode Analysis ───────────────────────────────────────────


def _analyze_rank_mode(
    results: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Analyze LoRA rank sweep.

    Args:
        results: Experiment results list.
        output_dir: Output directory.

    Returns:
        Analysis results dictionary.
    """
    if not results:
        logger.warning("No results for rank mode analysis.")
        return {}

    # Filter to LoRA results only
    lora_results = [r for r in results if r.get("method") == "lora"]
    if not lora_results:
        logger.warning("No LoRA results found.")
        return {}

    models = sorted(set(r["model"] for r in lora_results))
    domains = sorted(set(r["domain"] for r in lora_results))
    ranks = sorted(set(r.get("rank", 0) for r in lora_results if "rank" in r))

    analysis: dict[str, Any] = {
        "mode": "rank",
        "models": models,
        "domains": domains,
        "ranks": ranks,
        "n_results": len(lora_results),
    }

    # ─── Per-model per-domain: rank vs MAE ───────────────────────
    rank_analysis: dict[str, dict[str, dict[str, Any]]] = {}

    for model_name in models:
        rank_analysis[model_name] = {}
        for domain in domains:
            domain_results = [
                r
                for r in lora_results
                if r["model"] == model_name and r["domain"] == domain
            ]
            if not domain_results:
                continue

            rank_maes: dict[int, list[float]] = {}
            for r in domain_results:
                rank = r.get("rank", 0)
                rank_maes.setdefault(rank, []).append(r["metrics"]["mae"])

            # Compute mean MAE per rank
            rank_means = {
                rank: float(np.mean(maes)) for rank, maes in rank_maes.items()
            }
            rank_stds = {rank: float(np.std(maes)) for rank, maes in rank_maes.items()}

            # Find optimal rank
            optimal_rank = min(rank_means.keys(), key=lambda r: rank_means[r])

            rank_analysis[model_name][domain] = {
                "rank_means": rank_means,
                "rank_stds": rank_stds,
                "optimal_rank": optimal_rank,
                "optimal_mae": rank_means[optimal_rank],
            }

    analysis["rank_analysis"] = rank_analysis

    # ─── Is there a universally optimal rank? ────────────────────
    all_optimal_ranks = []
    for model_data in rank_analysis.values():
        for domain_data in model_data.values():
            all_optimal_ranks.append(domain_data["optimal_rank"])

    if all_optimal_ranks:
        optimal_rank_counts = {}
        for rank in all_optimal_ranks:
            optimal_rank_counts[rank] = optimal_rank_counts.get(rank, 0) + 1

        most_common_rank = max(
            optimal_rank_counts.keys(), key=lambda r: optimal_rank_counts[r]
        )
        analysis["universal_optimal_rank"] = {
            "rank": most_common_rank,
            "frequency": optimal_rank_counts[most_common_rank],
            "total_combinations": len(all_optimal_ranks),
        }

    # ─── Visualization ────────────────────────────────────────────
    try:
        _plot_rank_mode(rank_analysis, output_dir)
    except ImportError as exc:
        logger.warning("matplotlib not available, skipping visualization: %s", exc)

    return analysis


def _plot_rank_mode(
    rank_analysis: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Generate rank mode visualizations.

    Args:
        rank_analysis: Rank analysis results.
        output_dir: Output directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for model_name, domain_data in rank_analysis.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for domain, data in domain_data.items():
            rank_means = data["rank_means"]
            rank_stds = data["rank_stds"]

            ranks = sorted(rank_means.keys())
            means = [rank_means[r] for r in ranks]
            stds = [rank_stds[r] for r in ranks]

            ax.errorbar(ranks, means, yerr=stds, marker="o", label=domain, capsize=5)

        ax.set_xlabel("LoRA Rank")
        ax.set_ylabel("MAE")
        ax.set_title(f"Phase 2: LoRA Rank Sweep ({model_name})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"rank_sweep_{model_name}.png", dpi=300)
        plt.close(fig)
        logger.info("Saved rank sweep: %s", output_dir / f"rank_sweep_{model_name}.png")


# ─── Locus Mode Analysis ──────────────────────────────────────────


def _analyze_locus_mode(
    results: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Analyze multi-domain locus sweep.

    Args:
        results: Experiment results list.
        output_dir: Output directory.

    Returns:
        Analysis results dictionary.
    """
    if not results:
        logger.warning("No results for locus mode analysis.")
        return {}

    # Filter to locus sweep results
    locus_results = [r for r in results if "locus" in r]
    if not locus_results:
        logger.warning("No locus results found.")
        return {}

    models = sorted(set(r["model"] for r in locus_results))
    domains = sorted(set(r["domain"] for r in locus_results))
    loci = sorted(set(r["locus"] for r in locus_results))

    analysis: dict[str, Any] = {
        "mode": "locus",
        "models": models,
        "domains": domains,
        "loci": loci,
        "n_results": len(locus_results),
    }

    # ─── Per-model locus ranking per domain ───────────────────────
    rankings: dict[str, dict[str, list[str]]] = {}

    for model_name in models:
        rankings[model_name] = {}
        model_results = [r for r in locus_results if r["model"] == model_name]

        for domain in domains:
            domain_results = [r for r in model_results if r["domain"] == domain]
            if not domain_results:
                continue

            locus_maes: dict[str, float] = {}
            for locus in loci:
                locus_vals = [
                    r["metrics"]["mae"] for r in domain_results if r["locus"] == locus
                ]
                if locus_vals:
                    locus_maes[locus] = float(np.mean(locus_vals))

            # Rank by MAE (ascending)
            ranked = sorted(locus_maes.keys(), key=lambda l: locus_maes[l])
            rankings[model_name][domain] = ranked

    analysis["locus_rankings"] = rankings

    # ─── Kendall's tau: cross-domain correlation per model ────────
    cross_domain_tau: dict[str, dict[str, Any]] = {}

    for model_name in models:
        model_rankings = rankings.get(model_name, {})
        if len(model_rankings) < 2:
            continue

        tau_results: dict[str, dict[str, Any]] = {}
        common_loci = set(loci)
        ranks_by_domain: dict[str, list[int]] = {}

        for domain in domains:
            ranked = model_rankings.get(domain, [])
            rank_map = {l: i for i, l in enumerate(ranked)}
            ranks_by_domain[domain] = [
                rank_map.get(l, len(loci)) for l in sorted(common_loci)
            ]

        for d1, d2 in combinations(domains, 2):
            r1 = np.array(ranks_by_domain.get(d1, []))
            r2 = np.array(ranks_by_domain.get(d2, []))
            if r1.size >= 3 and r2.size >= 3:
                try:
                    tau, p_tau = stats.kendalltau(r1, r2)
                    key = f"{d1}_vs_{d2}"
                    tau_results[key] = {
                        "tau": float(tau),
                        "p_value": float(p_tau),
                        "significant": bool(p_tau < 0.05),
                    }
                except ValueError as exc:
                    logger.debug("Kendall's tau failed for %s: %s", key, exc)

        cross_domain_tau[model_name] = tau_results

    analysis["cross_domain_tau"] = cross_domain_tau

    # ─── Kendall's tau: cross-model correlation per domain ────────
    cross_model_tau: dict[str, dict[str, Any]] = {}

    for domain in domains:
        if len(models) < 2:
            continue

        tau_results: dict[str, dict[str, Any]] = {}
        common_loci = set(loci)
        ranks_by_model: dict[str, list[int]] = {}

        for model_name in models:
            ranked = rankings.get(model_name, {}).get(domain, [])
            rank_map = {l: i for i, l in enumerate(ranked)}
            ranks_by_model[model_name] = [
                rank_map.get(l, len(loci)) for l in sorted(common_loci)
            ]

        for m1, m2 in combinations(models, 2):
            r1 = np.array(ranks_by_model.get(m1, []))
            r2 = np.array(ranks_by_model.get(m2, []))
            if r1.size >= 3 and r2.size >= 3:
                try:
                    tau, p_tau = stats.kendalltau(r1, r2)
                    key = f"{m1}_vs_{m2}"
                    tau_results[key] = {
                        "tau": float(tau),
                        "p_value": float(p_tau),
                        "significant": bool(p_tau < 0.05),
                    }
                except ValueError as exc:
                    logger.debug("Kendall's tau failed for %s: %s", key, exc)

        cross_model_tau[domain] = tau_results

    analysis["cross_model_tau"] = cross_model_tau

    # ─── Visualization ────────────────────────────────────────────
    try:
        _plot_locus_mode(locus_results, analysis, output_dir)
    except ImportError as exc:
        logger.warning("matplotlib not available, skipping visualization: %s", exc)

    return analysis


def _plot_locus_mode(
    results: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate locus mode visualizations.

    Args:
        results: Experiment results list.
        analysis: Analysis results dictionary.
        output_dir: Output directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = analysis.get("models", [])
    domains = analysis.get("domains", [])
    loci = analysis.get("loci", [])

    # ─── Heatmap: locus × (model × domain) → MAE ──────────────────
    col_labels: list[str] = []
    for model_name in models:
        for domain in domains:
            col_labels.append(f"{model_name}\n{domain}")

    if loci and col_labels:
        data_matrix = np.zeros((len(loci), len(col_labels)))

        col_idx = 0
        for model_name in models:
            for domain in domains:
                for i, locus in enumerate(loci):
                    maes = [
                        r["metrics"]["mae"]
                        for r in results
                        if r["model"] == model_name
                        and r["domain"] == domain
                        and r["locus"] == locus
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
        ax.set_title("Phase 2: Locus × Model×Domain MAE")
        ax.set_xlabel("Model × Domain")
        ax.set_ylabel("LoRA Locus")
        fig.tight_layout()
        fig.savefig(output_dir / "locus_heatmap.png", dpi=300)
        plt.close(fig)
        logger.info("Saved locus heatmap: %s", output_dir / "locus_heatmap.png")

    # ─── Bar chart: locus MAE per model ──────────────────────────
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
            ax.set_title(f"Phase 2: Locus MAE ({model_name})")
            fig.tight_layout()
            fig.savefig(output_dir / f"locus_bar_{model_name}.png", dpi=300)
            plt.close(fig)
            logger.info(
                "Saved locus bar chart: %s", output_dir / f"locus_bar_{model_name}.png"
            )


# ─── Main ─────────────────────────────────────────────────────────


def main() -> None:
    """Phase 2 expansion analysis main function."""
    _setup_logging()
    args = _parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    logger.info("Loading results from: %s", input_dir)
    results = _load_results(input_dir)

    if not results:
        logger.error("No results loaded.")
        return

    logger.info("Loaded %d results", len(results))

    # ─── Domain mode analysis ──────────────────────────────────────
    if args.mode in ("domain", "all"):
        logger.info("Starting domain mode analysis...")
        domain_analysis = _analyze_domain_mode(results, output_dir)

        report_path = output_dir / "domain_analysis.json"
        with open(report_path, "w") as f:
            json.dump(domain_analysis, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Domain analysis complete: %s", report_path)

        # Log success criteria
        for model_name in domain_analysis.get("models", []):
            model_data = domain_analysis.get(model_name, {})
            anova_method = model_data.get("anova_method", {})
            anova_domain = model_data.get("anova_domain", {})
            logger.info(
                "Domain [%s] ANOVA: method F=%.3f (p=%.4f, sig=%s), "
                "domain F=%.3f (p=%.4f, sig=%s)",
                model_name,
                anova_method.get("F", 0.0),
                anova_method.get("p", 1.0),
                "✓" if anova_method.get("significant") else "✗",
                anova_domain.get("F", 0.0),
                anova_domain.get("p", 1.0),
                "✓" if anova_domain.get("significant") else "✗",
            )

    # ─── Rank mode analysis ────────────────────────────────────────
    if args.mode in ("rank", "all"):
        logger.info("Starting rank mode analysis...")
        rank_analysis = _analyze_rank_mode(results, output_dir)

        if rank_analysis:
            report_path = output_dir / "rank_analysis.json"
            with open(report_path, "w") as f:
                json.dump(rank_analysis, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Rank analysis complete: %s", report_path)

            universal_rank = rank_analysis.get("universal_optimal_rank", {})
            if universal_rank:
                logger.info(
                    "Universal optimal rank: %d (frequency: %d/%d)",
                    universal_rank.get("rank", 0),
                    universal_rank.get("frequency", 0),
                    universal_rank.get("total_combinations", 0),
                )
        else:
            logger.warning("No rank analysis results.")

    # ─── Locus mode analysis ───────────────────────────────────────
    if args.mode in ("locus", "all"):
        logger.info("Starting locus mode analysis...")
        locus_analysis = _analyze_locus_mode(results, output_dir)

        if locus_analysis:
            report_path = output_dir / "locus_analysis.json"
            with open(report_path, "w") as f:
                json.dump(locus_analysis, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Locus analysis complete: %s", report_path)

            cross_domain = locus_analysis.get("cross_domain_tau", {})
            for model_name, tau_data in cross_domain.items():
                sig_count = sum(
                    1 for v in tau_data.values() if v.get("significant", False)
                )
                logger.info(
                    "Locus [%s] cross-domain tau: %d/%d significant",
                    model_name,
                    sig_count,
                    len(tau_data),
                )
        else:
            logger.warning("No locus analysis results.")

    logger.info("Analysis complete. Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
