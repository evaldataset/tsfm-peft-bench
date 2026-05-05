from __future__ import annotations

# pyright: reportMissingImports=false

"""주파수 이동량과 LoRA rank 선택의 연관성을 탐색하는 스크립트.

기존 rank sweep 결과와 domain shift profile을 결합해,
"스펙트럼 이동이 큰 도메인일수록 높은 rank가 유리한가" 가설의
사전 근거를 정량화한다.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr, pearsonr

import numpy as np

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """로깅 설정 초기화.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    """CLI 인자 파싱.

    Args:
        None.

    Returns:
        파싱된 인자.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description="주파수-랭크 가설 사전 분석")
    parser.add_argument(
        "--rank_results_dir",
        type=str,
        default="results/expansion/rank",
        help="rank 모드 결과 디렉토리",
    )
    parser.add_argument(
        "--shift_profile_path",
        type=str,
        default="results/expansion_analysis_v2/domain_shift_profiles.json",
        help="도메인 shift profile JSON 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pivot_analysis",
        help="출력 디렉토리",
    )
    return parser.parse_args()


def _load_rank_rows(rank_results_dir: Path) -> list[dict[str, Any]]:
    """rank 모드 결과를 로드한다.

    Args:
        rank_results_dir: rank 모드 결과 디렉토리.

    Returns:
        유효 결과 딕셔너리 리스트.

    Raises:
        ValueError: 디렉토리가 없거나 유효 결과가 없을 때.
    """
    if not rank_results_dir.exists():
        raise ValueError(f"rank 결과 디렉토리가 없습니다: {rank_results_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(rank_results_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics", {})
            mae = metrics.get("mae") if isinstance(metrics, dict) else None
            rank = item.get("rank")
            if not isinstance(rank, int):
                continue
            if not isinstance(mae, (int, float)):
                continue
            if not all(k in item for k in ("model", "domain", "method")):
                continue
            if str(item.get("method")) != "lora":
                continue
            rows.append(item)
        except (OSError, json.JSONDecodeError):
            continue

    if not rows:
        raise ValueError("유효한 rank 결과를 찾지 못했습니다.")
    return rows


def _load_shift(path: Path) -> dict[str, dict[str, float]]:
    """도메인 shift profile을 로드한다.

    Args:
        path: shift profile JSON 파일 경로.

    Returns:
        domain -> feature dict 맵.

    Raises:
        ValueError: 파일이 없거나 형식이 잘못됐을 때.
    """
    if not path.exists():
        raise ValueError(f"shift profile 파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("shift profile 형식이 dict가 아닙니다.")
    return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}


def _build_rank_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """모델-도메인 단위 rank 민감도 요약을 생성한다.

    Args:
        rows: rank 실험 결과 목록.

    Returns:
        모델-도메인 요약 리스트.

    Raises:
        None.
    """
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["domain"]), int(row["rank"]))
        mae = float(row["metrics"]["mae"])
        grouped.setdefault(key, []).append(mae)

    by_cell: dict[tuple[str, str], dict[int, float]] = {}
    for (model, domain, rank), maes in grouped.items():
        by_cell.setdefault((model, domain), {})[rank] = float(np.mean(maes))

    summary: list[dict[str, Any]] = []
    for (model, domain), rank_map in sorted(by_cell.items()):
        sorted_rank_items = sorted(rank_map.items(), key=lambda kv: kv[0])
        ranks = [r for r, _ in sorted_rank_items]
        maes = [m for _, m in sorted_rank_items]
        best_rank = int(min(sorted_rank_items, key=lambda kv: kv[1])[0])
        sensitivity = float(max(maes) - min(maes)) if maes else 0.0

        baseline_r8 = rank_map.get(8)
        high_rank = rank_map.get(32)
        high_rank_gain = None
        if baseline_r8 is not None and high_rank is not None and baseline_r8 > 0:
            high_rank_gain = float((baseline_r8 - high_rank) / baseline_r8)

        summary.append(
            {
                "model": model,
                "domain": domain,
                "best_rank": best_rank,
                "rank_to_mae": {str(k): v for k, v in sorted_rank_items},
                "rank_sensitivity": sensitivity,
                "high_rank_gain_vs_r8": high_rank_gain,
                "available_ranks": ranks,
            }
        )
    return summary


def _safe_corr(x: list[float], y: list[float]) -> float:
    """피어슨 상관계수를 안전하게 계산한다.

    Args:
        x: 첫 번째 샘플.
        y: 두 번째 샘플.

    Returns:
        상관계수. 계산 불가능하면 0.0.

    Raises:
        None.
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman_corr(x: list[float], y: list[float]) -> dict[str, float]:
    """스피어만 순위 상관계수와 p-value를 계산한다.

    Args:
        x: 첫 번째 샘플.
        y: 두 번째 샘플.

    Returns:
        rho와 p-value를 담은 딕셔너리.

    Raises:
        None.
    """
    if len(x) < 2 or len(y) < 2:
        return {"rho": 0.0, "p_value": 1.0}
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return {"rho": 0.0, "p_value": 1.0}
    rho, p_val = spearmanr(x_arr, y_arr)
    return {"rho": float(rho), "p_value": float(p_val)}


def _permutation_test(
    x: list[float],
    y: list[float],
    n_permutations: int = 10000,
    seed: int = 42,
    metric: str = "pearson",
) -> dict[str, float]:
    """순열 검정(permutation test)으로 상관계수의 p-value를 계산한다.

    Args:
        x: 첫 번째 샘플.
        y: 두 번째 샘플.
        n_permutations: 순열 횟수. 기본 10000.
        seed: 난수 시드.
        metric: "pearson" 또는 "spearman".

    Returns:
        observed, p_value, n_permutations를 담은 딕셔너리.

    Raises:
        None.
    """
    rng = np.random.default_rng(seed)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)

    if n < 3:
        return {"observed": 0.0, "p_value": 1.0, "n_permutations": 0}

    # 관측된 상관계수
    if metric == "spearman":
        observed = float(spearmanr(x_arr, y_arr)[0])
    else:
        observed = float(np.corrcoef(x_arr, y_arr)[0, 1])

    # 순열 분포 계산
    count_extreme = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_arr)
        if metric == "spearman":
            perm_corr = float(spearmanr(x_arr, y_perm)[0])
        else:
            perm_corr = float(np.corrcoef(x_arr, y_perm)[0, 1])
        if abs(perm_corr) >= abs(observed):
            count_extreme += 1

    p_value = float(count_extreme / n_permutations)

    return {
        "observed": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def main() -> None:
    """메인 실행 함수.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    _setup_logging()
    args = _parse_args()

    rank_rows = _load_rank_rows(Path(args.rank_results_dir))
    shift_profiles = _load_shift(Path(args.shift_profile_path))
    summary = _build_rank_summary(rank_rows)

    corr_x: list[float] = []
    corr_best_rank: list[float] = []
    corr_sensitivity: list[float] = []
    corr_gain: list[float] = []
    targeted_cells: list[dict[str, Any]] = []

    for item in summary:
        domain = str(item["domain"])
        profile = shift_profiles.get(domain)
        if profile is None:
            continue
        spectral_w1 = float(profile.get("spectral_w1", 0.0))
        corr_x.append(spectral_w1)
        corr_best_rank.append(float(item["best_rank"]))
        corr_sensitivity.append(float(item["rank_sensitivity"]))

        gain = item.get("high_rank_gain_vs_r8")
        if isinstance(gain, float):
            corr_gain.append(gain)

        targeted_cells.append(
            {
                "model": item["model"],
                "domain": domain,
                "spectral_w1": spectral_w1,
                "best_rank": item["best_rank"],
                "rank_sensitivity": item["rank_sensitivity"],
                "high_rank_gain_vs_r8": gain,
                "proposed_budgeted_comparison": ["r4", "r8", "r16", "r32"],
            }
        )

    targeted_cells = sorted(
        targeted_cells,
        key=lambda x: (float(x["spectral_w1"]), float(x["rank_sensitivity"])),
        reverse=True,
    )

    spearman_sensitivity = _spearman_corr(corr_x, corr_sensitivity)
    perm_pearson = _permutation_test(
        corr_x, corr_sensitivity, n_permutations=10000, seed=42, metric="pearson"
    )
    perm_spearman = _permutation_test(
        corr_x, corr_sensitivity, n_permutations=10000, seed=42, metric="spearman"
    )

    report: dict[str, Any] = {
        "task": "frequency_rank_hypothesis_precheck",
        "n_rank_rows": len(rank_rows),
        "n_cells": len(summary),
        "correlations": {
            "spectral_w1_vs_best_rank": _safe_corr(corr_x, corr_best_rank),
            "spectral_w1_vs_rank_sensitivity": _safe_corr(corr_x, corr_sensitivity),
            "spectral_w1_vs_high_rank_gain": _safe_corr(corr_x[: len(corr_gain)], corr_gain)
            if corr_gain
            else 0.0,
        },
        "spearman": {
            "spectral_w1_vs_rank_sensitivity": spearman_sensitivity,
        },
        "permutation_tests": {
            "pearson_spectral_vs_sensitivity": perm_pearson,
            "spearman_spectral_vs_sensitivity": perm_spearman,
        },
        "cell_summaries": summary,
        "priority_cells_for_frequency_targeted_tests": targeted_cells[:8],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frequency_rank_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("주파수-랭크 가설 분석 저장 완료: %s", out_path)
    correlations = report["correlations"]
    if isinstance(correlations, dict):
        logger.info(
            "corr(spectral,best_rank)=%.4f corr(spectral,sensitivity)=%.4f",
            float(correlations.get("spectral_w1_vs_best_rank", 0.0)),
            float(correlations.get("spectral_w1_vs_rank_sensitivity", 0.0)),
        )
    spearman_data = report.get("spearman", {}).get("spectral_w1_vs_rank_sensitivity", {})
    if isinstance(spearman_data, dict):
        logger.info(
            "spearman(spectral,sensitivity): rho=%.4f p=%.4f",
            float(spearman_data.get("rho", 0.0)),
            float(spearman_data.get("p_value", 1.0)),
        )
    perm_data = report.get("permutation_tests", {}).get("spearman_spectral_vs_sensitivity", {})
    if isinstance(perm_data, dict):
        logger.info(
            "permutation(spearman): observed=%.4f p=%.4f (n_perm=%d)",
            float(perm_data.get("observed", 0.0)),
            float(perm_data.get("p_value", 1.0)),
            int(perm_data.get("n_permutations", 0)),
        )


if __name__ == "__main__":
    main()
