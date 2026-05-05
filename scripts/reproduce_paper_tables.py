"""논문 테이블/수치 단일 source-of-truth 재생성 스크립트.

NeurIPS audit blocker 1 해결:
- ``results/expansion/{domain,rank,locus}/*.json`` 원본 run을 직접 inventory.
- 동일한 outlier 필터 (per-cell scale-aware, MAE > 10x same-(model,domain) zero-shot)를
  모든 모드에 일관되게 적용.
- ``results/paper_manifest.json`` 을 단일 SoT로 재생성.
- 각 architecture별 two-way ANOVA를 재실행하여 paper에 인용된 n/F/eta_squared를
  새로 계산.
- 논문 본문/부록에 들어갈 핵심 숫자를 ``results/paper_numbers.json`` 으로 출력.
- LaTeX-ready snippet을 ``results/paper_numbers.tex`` 으로 출력하여 build에서 \\input 가능.

이 스크립트는 paper 수치의 단일 source of truth다. 논문 수치가 여기 출력과 다르면
논문이 틀렸다.

Usage:
    python scripts/reproduce_paper_tables.py
    python scripts/reproduce_paper_tables.py --input_dir results/expansion --output_dir results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)

# 논문 분석 대상 primary model 집합. TimesFM은 descriptive only.
PRIMARY_MODELS: tuple[str, ...] = ("chronos", "moirai", "moment")
DESCRIPTIVE_ONLY_MODELS: tuple[str, ...] = ("timesfm",)

# Six main methods for the per-model factorial design (paper Sec 3.3).
MAIN_METHODS: tuple[str, ...] = (
    "zero_shot",
    "head_only",
    "lora",
    "dora",
    "ia3",
    "adapter",
    "full_fine_tuning",
)
# IA^3 fallback이 adapter로 저장된 경우가 있으나 manifest에서는 adapter로 통합 노출.
DESCRIPTIVE_ONLY_METHODS: tuple[str, ...] = ("prefix",)


def _setup_logging() -> None:
    """로깅 설정 초기화."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=str, default="results/expansion")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument(
        "--analysis_output_dir",
        type=str,
        default="results/expansion_analysis_canonical",
        help="Per-architecture ANOVA artifact 저장 디렉토리",
    )
    return p.parse_args()


def _load_runs(input_dir: Path) -> list[dict[str, Any]]:
    """모든 run JSON을 로드하고 mode/source path를 부착.

    Args:
        input_dir: ``results/expansion`` 같은 root 디렉토리.

    Returns:
        run 딕셔너리 리스트 (각 항목에 ``_mode``, ``_source_path`` 포함).
    """
    rows: list[dict[str, Any]] = []
    for mode in ("domain", "rank", "locus"):
        d = input_dir / mode
        if not d.exists():
            logger.warning("디렉토리를 찾지 못했습니다: %s", d)
            continue
        for fp in sorted(d.glob("*.json")):
            if fp.name == "all_results.json":
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    row = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("로드 실패 %s: %s", fp, e)
                continue
            if not isinstance(row, dict):
                continue
            row["_mode"] = mode
            row["_source_path"] = str(fp)
            rows.append(row)
    logger.info("총 %d개 run 로드", len(rows))
    return rows


def _is_primary(row: dict[str, Any]) -> bool:
    """primary model (Chronos/MOMENT/Moirai) 여부."""
    return str(row.get("model", "")).lower() in PRIMARY_MODELS


def _is_main_method(row: dict[str, Any]) -> bool:
    """factorial 설계에 포함되는 main method 여부 (prefix 제외)."""
    return str(row.get("method", "")).lower() in MAIN_METHODS


def _detect_outliers_per_cell(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-cell scale-aware outlier 필터.

    Same-(model, domain) zero-shot MAE의 median × 10을 임계로 사용.
    zero-shot이 없는 cell은 cell median으로 fallback.

    Args:
        rows: 후보 run 리스트.

    Returns:
        (outlier 리스트, 정상 리스트)
    """
    zero_shot_mae: dict[tuple[str, str], list[float]] = {}
    cell_mae: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        mae = r.get("metrics", {}).get("mae")
        if not isinstance(mae, (int, float)):
            continue
        key = (str(r.get("model")), str(r.get("domain")))
        cell_mae.setdefault(key, []).append(float(mae))
        if r.get("method") == "zero_shot":
            zero_shot_mae.setdefault(key, []).append(float(mae))

    baselines: dict[tuple[str, str], float] = {
        k: float(np.median(v)) for k, v in zero_shot_mae.items() if v
    }

    outliers: list[dict[str, Any]] = []
    normal: list[dict[str, Any]] = []
    for r in rows:
        mae = r.get("metrics", {}).get("mae")
        if not isinstance(mae, (int, float)):
            normal.append(r)
            continue
        key = (str(r.get("model")), str(r.get("domain")))
        baseline = baselines.get(key)
        if baseline is None and key in cell_mae and cell_mae[key]:
            baseline = float(np.median(cell_mae[key]))
        if baseline is not None and baseline > 0 and mae > 10.0 * baseline:
            outliers.append(r)
        else:
            normal.append(r)
    return outliers, normal


def _two_way_anova(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """method, domain, interaction에 대한 two-way ANOVA (Type II SS).

    Args:
        rows: 단일 architecture에 해당하는 run 리스트 (main_methods only).

    Returns:
        n / per-effect F·p·eta·SS 딕셔너리.
    """
    data = []
    for r in rows:
        m = r.get("metrics", {})
        if "method" in r and "domain" in r and isinstance(m, dict) and "mae" in m:
            data.append({
                "method": str(r["method"]),
                "domain": str(r["domain"]),
                "mae": float(m["mae"]),
            })
    if len(data) < 6:
        return {"n": len(data), "method": {}, "domain": {}, "interaction": {}}
    df = pd.DataFrame(data)
    if df["method"].nunique() < 2 or df["domain"].nunique() < 2:
        return {"n": len(data), "method": {}, "domain": {}, "interaction": {}}
    fit = smf.ols(
        "mae ~ C(method) + C(domain) + C(method):C(domain)",
        data=df,
    ).fit()
    table = sm.stats.anova_lm(fit, typ=2)
    ss_total = float(table["sum_sq"].sum()) or 1.0

    def _ext(name: str) -> dict[str, float]:
        if name not in table.index:
            return {}
        row = table.loc[name]
        return {
            "F": float(row["F"]) if not pd.isna(row["F"]) else 0.0,
            "p": float(row["PR(>F)"]) if not pd.isna(row["PR(>F)"]) else 1.0,
            "sum_sq": float(row["sum_sq"]),
            "df": float(row["df"]),
            "eta_squared": float(row["sum_sq"] / ss_total),
        }

    return {
        "n": len(data),
        "method": _ext("C(method)"),
        "domain": _ext("C(domain)"),
        "interaction": _ext("C(method):C(domain)"),
    }


def _build_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """단일 source-of-truth manifest 구성."""
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_mode[r["_mode"]].append(r)

    manifest: dict[str, Any] = {
        "description": "Single source-of-truth experiment manifest "
                       "(reproduce_paper_tables.py 출력).",
        "inclusion_criteria": {
            "primary_models": list(PRIMARY_MODELS),
            "descriptive_only_models": list(DESCRIPTIVE_ONLY_MODELS),
            "main_methods": list(MAIN_METHODS),
            "descriptive_only_methods": list(DESCRIPTIVE_ONLY_METHODS),
            "outlier_rule": "MAE > 10 x same-(model,domain) zero-shot median",
        },
        "counts": {},
        "totals": {},
    }

    grand_paper_included = 0
    grand_primary_main = 0
    grand_primary_main_kept = 0
    grand_primary_main_outliers = 0

    for mode in ("domain", "rank", "locus"):
        mode_rows = by_mode.get(mode, [])
        primary_rows = [r for r in mode_rows if _is_primary(r)]
        main_rows = [r for r in primary_rows if _is_main_method(r)]
        prefix_rows = [r for r in primary_rows if not _is_main_method(r)]
        outliers, kept = _detect_outliers_per_cell(main_rows)

        per_model_counts = Counter(str(r.get("model")) for r in primary_rows)
        per_model_kept = Counter(str(r.get("model")) for r in kept)
        per_model_outliers = Counter(str(r.get("model")) for r in outliers)

        manifest["counts"][mode] = {
            "all_runs": len(mode_rows),
            "timesfm_descriptive": sum(
                1 for r in mode_rows
                if str(r.get("model")) in DESCRIPTIVE_ONLY_MODELS
            ),
            "primary_included": len(primary_rows),
            "primary_main_methods": len(main_rows),
            "primary_prefix_descriptive": len(prefix_rows),
            "primary_main_outliers": len(outliers),
            "primary_main_kept": len(kept),
            "per_model_primary_included": dict(per_model_counts),
            "per_model_main_kept": dict(per_model_kept),
            "per_model_main_outliers": dict(per_model_outliers),
        }
        grand_paper_included += len(primary_rows)
        grand_primary_main += len(main_rows)
        grand_primary_main_kept += len(kept)
        grand_primary_main_outliers += len(outliers)

    manifest["totals"] = {
        "paper_included_primary_runs": grand_paper_included,
        "primary_main_method_runs": grand_primary_main,
        "primary_main_kept_after_outliers": grand_primary_main_kept,
        "primary_main_outliers": grand_primary_main_outliers,
    }
    return manifest


def _per_arch_anova(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Architecture별 two-way ANOVA를 main-method outlier-removed domain 데이터에서 수행.

    Args:
        rows: 모든 run.

    Returns:
        {model: anova_result}
    """
    domain_rows = [r for r in rows if r["_mode"] == "domain" and _is_primary(r)]
    main_rows = [r for r in domain_rows if _is_main_method(r)]
    _, kept = _detect_outliers_per_cell(main_rows)

    per_arch: dict[str, dict[str, Any]] = {}
    for model in PRIMARY_MODELS:
        sub = [r for r in kept if str(r.get("model")).lower() == model]
        per_arch[model] = _two_way_anova(sub)
    return per_arch


def _emit_latex(numbers: dict[str, Any], out_path: Path) -> None:
    """논문 \\input용 LaTeX snippet 생성.

    Args:
        numbers: ``paper_numbers.json``에 들어가는 dict.
        out_path: 출력 ``.tex`` 경로.
    """
    totals = numbers["totals"]
    counts = numbers["counts"]
    arch = numbers["anova_per_arch"]

    def _fmt(x: float, digits: int = 3) -> str:
        return f"{x:.{digits}f}"

    lines: list[str] = [
        "% Auto-generated by scripts/reproduce_paper_tables.py.",
        "% Do not edit by hand. Re-run the script to refresh.",
        f"\\newcommand{{\\PaperIncludedRuns}}{{{totals['paper_included_primary_runs']}}}",
        f"\\newcommand{{\\PaperMainKept}}{{{totals['primary_main_kept_after_outliers']}}}",
        f"\\newcommand{{\\PaperMainOutliers}}{{{totals['primary_main_outliers']}}}",
        f"\\newcommand{{\\PaperPrimaryMain}}{{{totals['primary_main_method_runs']}}}",
        f"\\newcommand{{\\DomainModeAll}}{{{counts['domain']['primary_included']}}}",
        f"\\newcommand{{\\DomainModeKept}}{{{counts['domain']['primary_main_kept']}}}",
        f"\\newcommand{{\\DomainModeOutliers}}{{{counts['domain']['primary_main_outliers']}}}",
        f"\\newcommand{{\\RankModeRuns}}{{{counts['rank']['primary_included']}}}",
        f"\\newcommand{{\\LocusModeRuns}}{{{counts['locus']['primary_included']}}}",
    ]
    for m in PRIMARY_MODELS:
        a = arch.get(m, {})
        if not a or "n" not in a:
            continue
        n = a["n"]
        meth = a.get("method", {})
        dom = a.get("domain", {})
        inter = a.get("interaction", {})
        Cap = m.capitalize()
        if Cap == "Moment":
            Cap = "MOMENT"
        lines += [
            f"\\newcommand{{\\N{Cap}}}{{{n}}}",
            f"\\newcommand{{\\FmethodOf{Cap}}}{{{_fmt(meth.get('F', 0.0), 1)}}}",
            f"\\newcommand{{\\EtaMethodOf{Cap}}}{{{_fmt(meth.get('eta_squared', 0.0))}}}",
            f"\\newcommand{{\\FdomainOf{Cap}}}{{{_fmt(dom.get('F', 0.0), 1)}}}",
            f"\\newcommand{{\\EtaDomainOf{Cap}}}{{{_fmt(dom.get('eta_squared', 0.0))}}}",
            f"\\newcommand{{\\FinterOf{Cap}}}{{{_fmt(inter.get('F', 0.0), 1)}}}",
            f"\\newcommand{{\\EtaInterOf{Cap}}}{{{_fmt(inter.get('eta_squared', 0.0))}}}",
        ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """진입점: manifest + ANOVA + LaTeX snippet 재생성."""
    _setup_logging()
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_runs(input_dir)
    if not rows:
        raise SystemExit("로드된 run이 없습니다. --input_dir을 확인하세요.")

    manifest = _build_manifest(rows)
    arch_anova = _per_arch_anova(rows)

    paper_numbers: dict[str, Any] = {
        **manifest,
        "anova_per_arch": arch_anova,
    }

    manifest_path = output_dir / "paper_manifest.json"
    numbers_path = output_dir / "paper_numbers.json"
    tex_path = output_dir / "paper_numbers.tex"
    arch_anova_path = analysis_dir / "anova_primary_main_kept.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    numbers_path.write_text(
        json.dumps(paper_numbers, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    arch_anova_path.write_text(
        json.dumps(arch_anova, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    _emit_latex(paper_numbers, tex_path)

    logger.info("manifest=%s", manifest_path)
    logger.info("paper_numbers.json=%s", numbers_path)
    logger.info("paper_numbers.tex=%s", tex_path)
    logger.info("per-arch ANOVA=%s", arch_anova_path)
    logger.info("totals=%s", manifest["totals"])
    for m in PRIMARY_MODELS:
        a = arch_anova.get(m, {})
        if not a or "n" not in a:
            continue
        logger.info(
            "%s: n=%d eta_method=%.3f eta_domain=%.3f eta_inter=%.3f",
            m, a["n"],
            a["method"].get("eta_squared", 0.0),
            a["domain"].get("eta_squared", 0.0),
            a["interaction"].get("eta_squared", 0.0),
        )


if __name__ == "__main__":
    main()
