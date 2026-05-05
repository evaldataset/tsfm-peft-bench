from __future__ import annotations

"""통계적 견고성 분석 스크립트 (Task B1).

398개 expansion/domain 실험 결과에 대해:
  1. Kruskal-Wallis 비모수 검정 (모델별, method 요인, MAE)
  2. Cook's distance — 각 셀의 영향력 점수
  3. 도메인별 η² (도메인 단위 분리 ANOVA)

출력: results/expansion_analysis_v3/robustness.json

Usage:
    PYTHONPATH=. python scripts/compute_robustness.py
    PYTHONPATH=. python scripts/compute_robustness.py --input_dir results/expansion/domain --output_dir results/expansion_analysis_v3
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

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
        description="통계적 견고성 분석 — Kruskal-Wallis, Cook's distance, per-domain η²"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/expansion/domain",
        help="domain 모드 결과 JSON 디렉토리 (기본값: results/expansion/domain)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/expansion_analysis_v3",
        help="출력 디렉토리 (기본값: results/expansion_analysis_v3)",
    )
    return parser.parse_args()


# ─── 데이터 로딩 ─────────────────────────────────────────────────


def _load_domain_results(input_dir: Path) -> list[dict[str, Any]]:
    """domain 모드 결과 JSON 파일을 로드.

    Args:
        input_dir: JSON 파일이 담긴 디렉토리.

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
            logger.warning("JSON 파싱 실패, 건너뜀: %s", path)
            continue
        if not isinstance(rec, dict):
            continue
        # experiment_mode 필터: 없거나 domain이면 포함
        mode = rec.get("experiment_mode", "domain")
        if mode != "domain":
            continue
        mae = rec.get("metrics", {}).get("mae")
        if not isinstance(mae, (int, float)):
            continue
        records.append(rec)

    logger.info("로드된 레코드 수: %d", len(records))
    return records


# ─── 이상치 필터 ─────────────────────────────────────────────────


def _filter_outliers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """analyze_expansion_v2.py 와 동일한 이상치 필터 적용.

    (model, domain) 기준으로 zero-shot MAE의 10배 초과를 제거.

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

    # Fallback: cell median
    cell_mae: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        mae = r.get("metrics", {}).get("mae")
        if isinstance(mae, (int, float)):
            key = (str(r.get("model", "")), str(r.get("domain", "")))
            cell_mae[key].append(float(mae))

    normal: list[dict[str, Any]] = []
    n_outliers = 0
    for r in records:
        mae = r.get("metrics", {}).get("mae")
        key = (str(r.get("model", "")), str(r.get("domain", "")))
        baseline = baselines.get(key)
        if baseline is None and key in cell_mae:
            baseline = float(np.median(cell_mae[key]))
        if isinstance(mae, (int, float)) and baseline is not None and baseline > 0:
            if mae > 10.0 * baseline:
                n_outliers += 1
                continue
        normal.append(r)

    logger.info("이상치 %d개 제거, 정상 %d개 유지", n_outliers, len(normal))
    return normal


# ─── Kruskal-Wallis ───────────────────────────────────────────────


def _kruskal_wallis_per_model(
    df: pd.DataFrame, model_name: str
) -> dict[str, Any]:
    """모델별 Kruskal-Wallis 검정 (method 요인, 반응변수: MAE).

    Args:
        df: 필터링된 전체 데이터프레임 (model, method, domain, mae 열 포함).
        model_name: 분석할 모델 이름.

    Returns:
        H 통계량, p-value, 자유도, 그룹별 n을 담은 딕셔너리.
    """
    sub = df[df["model"] == model_name]
    groups = []
    group_ns: dict[str, int] = {}
    for method, grp in sub.groupby("method"):
        vals = grp["mae"].dropna().values
        if len(vals) >= 2:
            groups.append(vals)
            group_ns[str(method)] = len(vals)

    if len(groups) < 2:
        logger.warning("[%s] Kruskal-Wallis: 그룹 부족 (%d개)", model_name, len(groups))
        return {"H": 0.0, "p": 1.0, "dof": 0, "group_ns": group_ns}

    h_stat, p_val = stats.kruskal(*groups)
    dof = len(groups) - 1
    logger.info(
        "[%s] Kruskal-Wallis: H=%.4f, p=%.6f, dof=%d", model_name, h_stat, p_val, dof
    )
    return {
        "H": float(h_stat),
        "p": float(p_val),
        "dof": dof,
        "group_ns": group_ns,
        "significant": bool(p_val < 0.05),
    }


# ─── Cook's Distance ─────────────────────────────────────────────


def _cooks_distance_per_model(
    df: pd.DataFrame, model_name: str
) -> list[dict[str, Any]]:
    """모델별 OLS 회귀의 Cook's distance를 계산하여 상위 10개 반환.

    OLS 모델: mae ~ C(method) + C(domain) (interaction 없이 주효과만 사용,
    셀 단위 영향력을 안정적으로 추정하기 위함).

    Args:
        df: 필터링된 전체 데이터프레임.
        model_name: 분석할 모델 이름.

    Returns:
        상위 10개 Cook's distance 항목 리스트.
        각 항목: {"cell": "method|domain", "distance": float, "row_index": int}.
    """
    sub = df[df["model"] == model_name].copy().reset_index(drop=True)

    if len(sub) < 10:
        logger.warning("[%s] Cook's distance: 데이터 부족 (%d행)", model_name, len(sub))
        return []

    # 방법/도메인 레벨이 하나뿐이면 OLS 적합 불가
    if sub["method"].nunique() < 2 or sub["domain"].nunique() < 2:
        logger.warning("[%s] Cook's distance: 범주 다양성 부족", model_name)
        return []

    try:
        ols_model = smf.ols("mae ~ C(method) + C(domain)", data=sub).fit()
        influence = ols_model.get_influence()
        cooks_d = influence.cooks_distance[0]  # shape (n,)
    except Exception as exc:
        logger.warning("[%s] Cook's distance 계산 실패: %s", model_name, exc)
        return []

    sub["cooks_d"] = cooks_d
    sub["cell"] = sub["method"].astype(str) + "|" + sub["domain"].astype(str)

    # 상위 10개 (내림차순)
    top10 = sub.nlargest(10, "cooks_d")[["cell", "cooks_d"]].copy()
    result = [
        {"cell": row["cell"], "distance": float(row["cooks_d"])}
        for _, row in top10.iterrows()
    ]
    logger.info(
        "[%s] Cook's distance top-1: %s (%.4f)",
        model_name,
        result[0]["cell"] if result else "N/A",
        result[0]["distance"] if result else 0.0,
    )
    return result


# ─── Per-domain η² ───────────────────────────────────────────────


def _per_domain_eta_squared(
    df: pd.DataFrame, model_name: str, domains: list[str]
) -> dict[str, float]:
    """도메인별 분리 one-way ANOVA로 method 요인의 η²을 계산.

    각 도메인에 대해 독립적으로 ANOVA를 수행하므로, 어느 도메인에서
    interaction이 집중되는지 확인 가능.

    Args:
        df: 필터링된 전체 데이터프레임.
        model_name: 분석할 모델 이름.
        domains: 분석할 도메인 목록.

    Returns:
        {domain_name: eta_squared_float} 딕셔너리.
    """
    sub = df[df["model"] == model_name]
    result: dict[str, float] = {}

    for domain in domains:
        sub_d = sub[sub["domain"] == domain].copy()
        if len(sub_d) < 4 or sub_d["method"].nunique() < 2:
            result[domain] = 0.0
            continue

        try:
            ols_model = smf.ols("mae ~ C(method)", data=sub_d).fit()
            table = sm.stats.anova_lm(ols_model, typ=1)
            ss_method = float(table.loc["C(method)", "sum_sq"])
            ss_total = float(table["sum_sq"].sum())
            eta_sq = ss_method / ss_total if ss_total > 0 else 0.0
        except Exception as exc:
            logger.warning(
                "[%s/%s] per-domain η² 계산 실패: %s", model_name, domain, exc
            )
            eta_sq = 0.0

        result[domain] = float(eta_sq)
        logger.info(
            "[%s/%s] per-domain η²=%.4f (n=%d)", model_name, domain, eta_sq, len(sub_d)
        )

    return result


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

    # 데이터 로드 및 필터링
    records = _load_domain_results(input_dir)
    records = _filter_outliers(records)

    # 데이터프레임 구성
    rows = []
    for r in records:
        mae = r.get("metrics", {}).get("mae")
        if not isinstance(mae, (int, float)):
            continue
        rows.append({
            "model": str(r.get("model", "")),
            "method": str(r.get("method", "")),
            "domain": str(r.get("domain", "")),
            "mae": float(mae),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("유효한 데이터가 없습니다.")
        return

    models = sorted(df["model"].unique())
    domains = sorted(df["domain"].unique())

    logger.info(
        "분석 시작: 모델 %s, 도메인 %s, 총 %d행", models, domains, len(df)
    )

    output: dict[str, Any] = {}

    for model_name in models:
        logger.info("=== 모델: %s ===", model_name)

        kw = _kruskal_wallis_per_model(df, model_name)
        cook_top10 = _cooks_distance_per_model(df, model_name)
        per_domain_eta = _per_domain_eta_squared(df, model_name, domains)

        output[model_name] = {
            "kruskal_wallis": {
                "H": kw["H"],
                "p": kw["p"],
                "dof": kw["dof"],
            },
            "cook_distance_top10": cook_top10,
            "per_domain_eta_squared": per_domain_eta,
        }

    out_path = output_dir / "robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info("견고성 분석 저장 완료: %s", out_path)

    # 요약 출력
    for model_name, mdata in output.items():
        kw = mdata["kruskal_wallis"]
        top_cook = mdata["cook_distance_top10"]
        best_domain = max(
            mdata["per_domain_eta_squared"].items(), key=lambda x: x[1], default=("N/A", 0.0)
        )
        logger.info(
            "[%s] KW: H=%.2f p=%.4f | top Cook: %s (%.4f) | max domain η²: %s=%.4f",
            model_name,
            kw["H"],
            kw["p"],
            top_cook[0]["cell"] if top_cook else "N/A",
            top_cook[0]["distance"] if top_cook else 0.0,
            best_domain[0],
            best_domain[1],
        )


if __name__ == "__main__":
    main()
