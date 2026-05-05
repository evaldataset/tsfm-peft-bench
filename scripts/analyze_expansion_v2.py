"""Phase 2 expansion analysis — corrected statistical pipeline (v2).

Fixes from v1:
- Mode-specific result loading (prevents LoRA data leakage)
- True two-way ANOVA with interaction term (statsmodels)
- Outlier detection for training divergence
- Effect sizes (η², Cliff's delta)
- Holm-Bonferroni multiple comparison correction
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.typing import NDArray
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """로깅 설정 초기화."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="Phase 2 expansion analysis — corrected (v2)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["domain", "rank", "locus", "all"],
        default="all",
    )
    parser.add_argument(
        "--input_dir", type=str, default="results/expansion",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/expansion_analysis_v2",
    )
    return parser.parse_args()


# ─── Statistical Helper Functions ─────────────────────────────────


def _load_results_by_mode(
    result_dir: Path, mode: str,
) -> list[dict[str, Any]]:
    """모드별 필터링된 실험 결과를 로드.

    Args:
        result_dir: 결과 디렉토리 경로.
        mode: 'domain', 'rank', 'locus' 중 하나.

    Returns:
        해당 모드의 결과 딕셔너리 리스트.

    Raises:
        ValueError: 지원하지 않는 mode일 때.
    """
    valid_modes = {"domain", "rank", "locus"}
    if mode not in valid_modes:
        raise ValueError(f"지원하지 않는 mode입니다: {mode}")

    default_locus = "attn_all"
    all_rows: list[dict[str, Any]] = []

    for json_file in sorted(result_dir.rglob("*.json")):
        if json_file.name == "all_results.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                row = json.load(f)
            if not isinstance(row, dict):
                continue
            row["_source_path"] = str(json_file)
            all_rows.append(row)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

    def _infer_mode(row: dict[str, Any]) -> str:
        # 명시적 ``experiment_mode`` 필드를 우선 사용 (run_expansion.py가 저장).
        explicit = row.get("experiment_mode")
        if isinstance(explicit, str) and explicit in {"domain", "rank", "locus"}:
            return explicit

        # Fallback: 디렉터리 경로 기반 추론.
        src = Path(str(row.get("_source_path", ""))).parts
        if "rank" in src:
            return "rank"
        if "locus" in src:
            return "locus"
        if "domain" in src:
            return "domain"

        # Last resort: 휴리스틱.
        rank = row.get("rank")
        locus = row.get("locus", default_locus)
        if isinstance(rank, int) and rank in {4, 16, 32}:
            return "rank"
        if isinstance(locus, str) and locus != default_locus:
            return "locus"
        return "domain"

    filtered = [r for r in all_rows if _infer_mode(r) == mode]

    method_counts = Counter(str(r.get("method", "unknown")) for r in filtered)
    logger.info(
        "[%s] 필터링 결과: %d개 (전체 %d개), 방법별: %s",
        mode, len(filtered), len(all_rows), dict(method_counts),
    )
    return filtered


def _detect_outliers(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """학습 발산 이상치를 탐지.

    Per-(model, domain) 기준으로 zero-shot MAE의 10배를 초과하는 결과를 이상치로
    간주한다. 이 정책은 도메인별 scale을 보존하므로, 큰 값을 가지는 도메인이
    통째로 outlier로 잡히는 문제를 방지한다.

    Args:
        results: 실험 결과 리스트.

    Returns:
        (이상치 리스트, 정상 결과 리스트) 튜플.
    """
    # (model, domain) → zero-shot MAE의 median을 기준선으로 사용.
    zero_shot_mae: dict[tuple[str, str], list[float]] = {}
    for r in results:
        if r.get("method") != "zero_shot":
            continue
        mae = r.get("metrics", {}).get("mae")
        if not isinstance(mae, (int, float)):
            continue
        key = (str(r.get("model")), str(r.get("domain")))
        zero_shot_mae.setdefault(key, []).append(float(mae))

    baselines = {
        k: float(np.median(v)) for k, v in zero_shot_mae.items() if v
    }

    # Fallback: (model, domain) zero-shot이 없으면 cell의 모든 method median 사용.
    cell_mae: dict[tuple[str, str], list[float]] = {}
    for r in results:
        mae = r.get("metrics", {}).get("mae")
        if not isinstance(mae, (int, float)):
            continue
        key = (str(r.get("model")), str(r.get("domain")))
        cell_mae.setdefault(key, []).append(float(mae))

    outliers: list[dict[str, Any]] = []
    normal: list[dict[str, Any]] = []

    for r in results:
        mae = r.get("metrics", {}).get("mae")
        key = (str(r.get("model")), str(r.get("domain")))
        baseline = baselines.get(key)
        if baseline is None and key in cell_mae:
            baseline = float(np.median(cell_mae[key]))
        if isinstance(mae, (int, float)) and baseline is not None and baseline > 0:
            # zero-shot의 10배 초과는 학습 발산으로 판정.
            if mae > 10.0 * baseline:
                outliers.append(r)
                continue
        normal.append(r)

    if outliers:
        combo = Counter(
            (str(r.get("model")), str(r.get("method")), str(r.get("domain")))
            for r in outliers
        )
        for (m, meth, d), cnt in sorted(combo.items(), key=lambda x: -x[1]):
            logger.warning("이상치: model=%s method=%s domain=%s count=%d", m, meth, d, cnt)

    logger.info("이상치 %d개 탐지, 정상 %d개", len(outliers), len(normal))
    return outliers, normal


def _two_way_anova(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Two-way ANOVA with interaction term.

    Args:
        results: 실험 결과 리스트.

    Returns:
        method, domain, interaction 각각의 F, p, eta_squared 딕셔너리.

    Raises:
        None.
    """
    rows = []
    for item in results:
        metrics = item.get("metrics", {})
        if "method" in item and "domain" in item and "mae" in metrics:
            rows.append({
                "method": str(item["method"]),
                "domain": str(item["domain"]),
                "mae": float(metrics["mae"]),
            })

    empty = {"F": 0.0, "p": 1.0, "eta_squared": 0.0}
    if len(rows) < 6:
        return {"method": dict(empty), "domain": dict(empty), "interaction": dict(empty)}

    df = pd.DataFrame(rows)
    model = smf.ols(
        "mae ~ C(method) + C(domain) + C(method):C(domain)", data=df,
    ).fit()
    table = sm.stats.anova_lm(model, typ=2)

    ss_total = float(table["sum_sq"].sum())
    if ss_total <= 0:
        ss_total = 1.0

    def _extract(name: str) -> dict[str, float]:
        if name not in table.index:
            return dict(empty)
        row = table.loc[name]
        f_val = row["F"]
        p_val = row["PR(>F)"]
        return {
            "F": float(0.0 if pd.isna(f_val) else f_val),
            "p": float(1.0 if pd.isna(p_val) else p_val),
            "eta_squared": float(row["sum_sq"] / ss_total),
        }

    return {
        "method": _extract("C(method)"),
        "domain": _extract("C(domain)"),
        "interaction": _extract("C(method):C(domain)"),
    }


def _cliffs_delta(
    group1: NDArray[np.floating[Any]], group2: NDArray[np.floating[Any]],
) -> tuple[float, str]:
    """Cliff's delta 효과 크기를 계산.

    Args:
        group1: 첫 번째 그룹 관측값.
        group2: 두 번째 그룹 관측값.

    Returns:
        (delta 값, 해석 문자열) 튜플.

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
    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.33:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"
    return float(delta), interp


def _holm_correction(pvalues: list[float]) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni 다중비교 보정.

    Args:
        pvalues: 원본 p-value 목록.

    Returns:
        (기각 여부 목록, 보정된 p-value 목록).

    Raises:
        ValueError: p-value가 유효하지 않을 때.
    """
    if not pvalues:
        return [], []
    arr = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("유한하지 않은 p-value가 포함되었습니다.")
    reject, corrected, _, _ = multipletests(arr, alpha=0.05, method="holm")
    return reject.tolist(), corrected.astype(float).tolist()


# ─── Domain Mode Analysis ─────────────────────────────────────────


def _analyze_domain_mode(
    results: list[dict[str, Any]], output_dir: Path,
) -> dict[str, Any]:
    """도메인 모드 분석 — corrected two-way ANOVA.

    Args:
        results: 도메인 모드 결과 리스트.
        output_dir: 출력 디렉토리.

    Returns:
        분석 결과 딕셔너리.

    Raises:
        None.
    """
    if not results:
        logger.warning("도메인 모드 분석용 결과가 없습니다.")
        return {}

    models = sorted(set(r["model"] for r in results))
    methods = sorted(set(r["method"] for r in results))
    domains = sorted(set(r["domain"] for r in results))

    analysis: dict[str, Any] = {
        "mode": "domain_v2",
        "models": models,
        "methods": methods,
        "domains": domains,
        "n_results": len(results),
        "corrections_applied": [
            "mode_specific_loading",
            "two_way_anova_with_interaction",
            "outlier_detection",
            "effect_sizes",
            "holm_correction",
        ],
    }

    for model_name in models:
        model_results = [r for r in results if r["model"] == model_name]
        model_analysis: dict[str, Any] = {"n_results": len(model_results)}

        # Two-way ANOVA
        anova = _two_way_anova(model_results)
        model_analysis["anova_two_way"] = anova

        # Per-method domain Kruskal-Wallis (within-method)
        kw_results: dict[str, Any] = {}
        for method in methods:
            method_results = [r for r in model_results if r["method"] == method]
            domain_groups = {}
            for r in method_results:
                domain_groups.setdefault(r["domain"], []).append(r["metrics"]["mae"])

            groups = [np.array(v) for v in domain_groups.values() if len(v) >= 1]
            if len(groups) >= 2:
                h_stat, p_val = stats.kruskal(*groups)
                n_total = sum(len(g) for g in groups)
                k = len(groups)
                epsilon_sq = (h_stat - k + 1) / (n_total - k) if n_total > k else 0.0
                kw_results[method] = {
                    "H": float(h_stat), "p": float(p_val),
                    "epsilon_squared": float(epsilon_sq),
                    "n_per_domain": {d: len(v) for d, v in domain_groups.items()},
                    "significant": bool(p_val < 0.05),
                }

        model_analysis["within_method_kw"] = kw_results

        # Best method per domain
        best: dict[str, Any] = {}
        for domain in domains:
            domain_results = [
                r for r in model_results if r["domain"] == domain
            ]
            if domain_results:
                best_r = min(domain_results, key=lambda r: r["metrics"]["mae"])
                best[domain] = {
                    "method": best_r["method"],
                    "mae": best_r["metrics"]["mae"],
                }
        model_analysis["best_method_per_domain"] = best

        analysis[model_name] = model_analysis

    return analysis


def _analyze_rank_mode(results: list[dict[str, Any]]) -> dict[str, Any]:
    """랭크 모드 결과 분석: 모델×도메인별 최적 rank, Spearman 상관.

    Args:
        results: 랭크 모드 결과 리스트.

    Returns:
        분석 결과 딕셔너리.

    Raises:
        None.
    """
    from scipy.stats import spearmanr

    models = sorted({str(r["model"]) for r in results})
    domains = sorted({str(r["domain"]) for r in results})
    ranks = sorted({int(r.get("rank", 8)) for r in results})

    analysis: dict[str, Any] = {
        "mode": "rank",
        "n_results": len(results),
        "models": models,
        "domains": domains,
        "ranks": ranks,
        "per_model": {},
    }

    for model in models:
        model_data: dict[str, Any] = {"optimal_rank": {}, "rank_sensitivity": {}}

        for domain in domains:
            rank_maes: dict[int, list[float]] = {}
            for r in results:
                if str(r["model"]) != model or str(r["domain"]) != domain:
                    continue
                rank_val = int(r.get("rank", 8))
                m_obj = r.get("metrics", {})
                if isinstance(m_obj, dict) and "mae" in m_obj:
                    rank_maes.setdefault(rank_val, []).append(float(m_obj["mae"]))

            if not rank_maes:
                continue

            mean_per_rank = {rk: float(np.mean(v)) for rk, v in rank_maes.items()}
            best_rank = min(mean_per_rank, key=lambda rk: mean_per_rank[rk])
            sensitivity = float(np.std(list(mean_per_rank.values())))

            model_data["optimal_rank"][domain] = best_rank
            model_data["rank_sensitivity"][domain] = round(sensitivity, 6)

        # Spearman: rank 값과 mean MAE의 상관 (전 도메인)
        all_ranks_flat: list[int] = []
        all_maes_flat: list[float] = []
        for r in results:
            if str(r["model"]) != model:
                continue
            m_obj = r.get("metrics", {})
            if isinstance(m_obj, dict) and "mae" in m_obj:
                all_ranks_flat.append(int(r.get("rank", 8)))
                all_maes_flat.append(float(m_obj["mae"]))

        if len(all_ranks_flat) >= 4:
            rho, p_val = spearmanr(all_ranks_flat, all_maes_flat)
            model_data["spearman_rank_mae"] = {
                "rho": round(float(rho), 4),
                "p": round(float(p_val), 6),
            }

        analysis["per_model"][model] = model_data

    return analysis


def _analyze_locus_mode(results: list[dict[str, Any]]) -> dict[str, Any]:
    """로커스 모드 결과 분석: Kendall tau cross-model/cross-domain.

    Args:
        results: 로커스 모드 결과 리스트.

    Returns:
        분석 결과 딕셔너리.

    Raises:
        None.
    """
    from scipy.stats import kendalltau

    models = sorted({str(r["model"]) for r in results})
    domains = sorted({str(r["domain"]) for r in results})
    loci = sorted({str(r.get("locus", "attn_all")) for r in results})

    analysis: dict[str, Any] = {
        "mode": "locus",
        "n_results": len(results),
        "models": models,
        "domains": domains,
        "loci": loci,
        "per_model": {},
        "cross_model_tau": {},
    }

    # 모델별 도메인별 locus ranking
    locus_rankings: dict[str, dict[str, list[tuple[str, float]]]] = {}

    for model in models:
        model_data: dict[str, Any] = {"top_locus": {}, "locus_mae": {}}

        for domain in domains:
            locus_maes: dict[str, list[float]] = {}
            for r in results:
                if str(r["model"]) != model or str(r["domain"]) != domain:
                    continue
                locus_val = str(r.get("locus", "attn_all"))
                m_obj = r.get("metrics", {})
                if isinstance(m_obj, dict) and "mae" in m_obj:
                    locus_maes.setdefault(locus_val, []).append(float(m_obj["mae"]))

            if not locus_maes:
                continue

            mean_per_locus = {loc: float(np.mean(v)) for loc, v in locus_maes.items()}
            ranked = sorted(mean_per_locus.items(), key=lambda x: x[1])
            model_data["top_locus"][domain] = ranked[0][0] if ranked else ""
            model_data["locus_mae"][domain] = {
                loc: round(mae, 6) for loc, mae in mean_per_locus.items()
            }

            locus_rankings.setdefault(model, {})[domain] = ranked

        analysis["per_model"][model] = model_data

    # Cross-model Kendall tau per domain
    for domain in domains:
        tau_matrix: dict[str, dict[str, Any]] = {}
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                r1 = locus_rankings.get(m1, {}).get(domain, [])
                r2 = locus_rankings.get(m2, {}).get(domain, [])
                if len(r1) < 3 or len(r2) < 3:
                    continue

                loci_m1 = [loc for loc, _ in r1]
                loci_m2 = [loc for loc, _ in r2]
                common = [loc for loc in loci_m1 if loc in loci_m2]
                if len(common) < 3:
                    continue

                ranks_m1 = [loci_m1.index(loc) for loc in common]
                ranks_m2 = [loci_m2.index(loc) for loc in common]
                tau, p_val = kendalltau(ranks_m1, ranks_m2)

                pair_key = f"{m1}_vs_{m2}"
                tau_matrix[pair_key] = {
                    "tau": round(float(tau), 4),
                    "p": round(float(p_val), 6),
                    "n_common": len(common),
                }

        analysis["cross_model_tau"][domain] = tau_matrix

    # Holm correction on all tau p-values
    all_p_values: list[float] = []
    all_p_keys: list[tuple[str, str]] = []
    for domain, pairs in analysis["cross_model_tau"].items():
        for pair_key, vals in pairs.items():
            all_p_values.append(vals["p"])
            all_p_keys.append((domain, pair_key))

    if all_p_values:
        significant, corrected = _holm_correction(all_p_values)
        for idx, (domain, pair_key) in enumerate(all_p_keys):
            analysis["cross_model_tau"][domain][pair_key]["p_corrected"] = round(
                corrected[idx], 6
            )
            analysis["cross_model_tau"][domain][pair_key]["significant"] = significant[idx]

    return analysis


# ─── Main ─────────────────────────────────────────────────────────


def main() -> None:
    """메인 실행 함수."""
    _setup_logging()
    args = _parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error("입력 디렉토리가 존재하지 않습니다: %s", input_dir)
        return

    # ─── Domain mode analysis ──────────────────────────────────
    if args.mode in ("domain", "all"):
        logger.info("=== 도메인 모드 분석 (corrected v2) ===")
        domain_results = _load_results_by_mode(input_dir, "domain")

        outliers, normal_results = _detect_outliers(domain_results)

        analysis = _analyze_domain_mode(normal_results, output_dir)
        analysis["outlier_summary"] = {
            "total_outliers": len(outliers),
            "total_normal": len(normal_results),
            "outlier_details": [
                {
                    "model": r.get("model"),
                    "method": r.get("method"),
                    "domain": r.get("domain"),
                    "mae": r.get("metrics", {}).get("mae"),
                }
                for r in outliers[:20]
            ],
        }

        report_path = output_dir / "domain_analysis_v2.json"
        with open(report_path, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        logger.info("도메인 분석 저장: %s", report_path)

        # Log key results
        for model_name in analysis.get("models", []):
            mdata = analysis.get(model_name, {})
            anova = mdata.get("anova_two_way", {})
            fm = anova.get("method", {})
            fd = anova.get("domain", {})
            fi = anova.get("interaction", {})
            logger.info(
                "[%s] method F=%.2f(p=%.4f,η²=%.3f) domain F=%.2f(p=%.4f,η²=%.3f)"
                " interaction F=%.2f(p=%.4f,η²=%.3f)",
                model_name,
                fm.get("F", 0), fm.get("p", 1), fm.get("eta_squared", 0),
                fd.get("F", 0), fd.get("p", 1), fd.get("eta_squared", 0),
                fi.get("F", 0), fi.get("p", 1), fi.get("eta_squared", 0),
            )

    # ─── Rank mode analysis ────────────────────────────────────
    if args.mode in ("rank", "all"):
        logger.info("=== 랭크 모드 분석 ===")
        rank_results = _load_results_by_mode(input_dir, "rank")
        if rank_results:
            rank_analysis = _analyze_rank_mode(rank_results)
            report_path = output_dir / "rank_analysis_v2.json"
            with open(report_path, "w") as f:
                json.dump(rank_analysis, f, indent=2, ensure_ascii=False, default=str)
            logger.info("랭크 분석 저장: %s", report_path)

    # ─── Locus mode analysis ───────────────────────────────────
    if args.mode in ("locus", "all"):
        logger.info("=== 로커스 모드 분석 ===")
        locus_results = _load_results_by_mode(input_dir, "locus")
        if locus_results:
            locus_analysis = _analyze_locus_mode(locus_results)
            report_path = output_dir / "locus_analysis_v2.json"
            with open(report_path, "w") as f:
                json.dump(locus_analysis, f, indent=2, ensure_ascii=False, default=str)
            logger.info("로커스 분석 저장: %s", report_path)

    logger.info("분석 완료. 결과: %s", output_dir)


if __name__ == "__main__":
    main()
