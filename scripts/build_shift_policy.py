from __future__ import annotations

# pyright: reportMissingImports=false

"""도메인 이동량 기반 PEFT 추천 정책(Shift-to-PEFT Policy) 분석 스크립트.

기존 expansion domain 결과와 도메인 이동 프로파일을 결합하여,
"주어진 (모델, 도메인 이동 특성)에서 어떤 적응 방법이 가장 유리한가"를
정책 문제로 재정의해 평가한다.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


@dataclass
class _MethodRow:
    """모델-도메인-방법 단위 집계 행.

    Args:
        model: 모델 이름.
        domain: 도메인 이름.
        method: 적응 방법 이름.
        mean_mae: seed 평균 MAE.
        n: 관측치 수.

    Returns:
        _MethodRow 인스턴스.

    Raises:
        None.
    """

    model: str
    domain: str
    method: str
    mean_mae: float
    n: int


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
    parser = argparse.ArgumentParser(description="Shift-to-PEFT policy 분석")
    parser.add_argument(
        "--domain_results_dir",
        type=str,
        default="results/expansion/domain",
        help="domain 모드 json 디렉토리",
    )
    parser.add_argument(
        "--shift_profile_path",
        type=str,
        default="results/expansion_analysis_v2/domain_shift_profiles.json",
        help="도메인 이동 프로파일 JSON 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pivot_analysis",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="정책 모델 random seed",
    )
    return parser.parse_args()


def _load_domain_results(domain_results_dir: Path) -> list[dict[str, Any]]:
    """domain 모드 결과를 로드한다.

    Args:
        domain_results_dir: domain 결과 디렉토리.

    Returns:
        결과 딕셔너리 리스트.

    Raises:
        ValueError: 디렉토리가 없거나 유효 결과가 없을 때.
    """
    if not domain_results_dir.exists():
        raise ValueError(
            f"domain 결과 디렉토리가 존재하지 않습니다: {domain_results_dir}"
        )

    rows: list[dict[str, Any]] = []
    for path in sorted(domain_results_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics", {})
            mae = metrics.get("mae") if isinstance(metrics, dict) else None
            if not isinstance(mae, (int, float)):
                continue
            if not all(k in item for k in ("model", "domain", "method")):
                continue
            rows.append(item)
        except (OSError, json.JSONDecodeError):
            continue

    if not rows:
        raise ValueError("유효한 domain 결과를 찾지 못했습니다.")
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[_MethodRow]:
    """seed 축 결과를 모델-도메인-방법 단위로 집계한다.

    Args:
        rows: 원본 실험 결과.

    Returns:
        집계 결과 목록.

    Raises:
        None.
    """
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["domain"]), str(row["method"]))
        mae = float(row["metrics"]["mae"])
        grouped.setdefault(key, []).append(mae)

    out: list[_MethodRow] = []
    for (model, domain, method), values in sorted(grouped.items()):
        out.append(
            _MethodRow(
                model=model,
                domain=domain,
                method=method,
                mean_mae=float(np.mean(values)),
                n=len(values),
            )
        )
    return out


def _best_method_map(agg: list[_MethodRow]) -> dict[tuple[str, str], str]:
    """모델-도메인별 best method를 계산한다.

    Args:
        agg: 집계 결과 목록.

    Returns:
        (model, domain) -> best_method 맵.

    Raises:
        None.
    """
    best: dict[tuple[str, str], tuple[str, float]] = {}
    for row in agg:
        key = (row.model, row.domain)
        cur = best.get(key)
        if cur is None or row.mean_mae < cur[1]:
            best[key] = (row.method, row.mean_mae)
    return {k: v[0] for k, v in best.items()}


def _load_shift_profiles(path: Path) -> dict[str, dict[str, float]]:
    """도메인 이동 프로파일을 로드한다.

    Args:
        path: shift profile JSON 경로.

    Returns:
        domain -> feature dict.

    Raises:
        ValueError: 파일이 없거나 형식이 잘못됐을 때.
    """
    if not path.exists():
        raise ValueError(f"shift profile 파일이 존재하지 않습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("shift profile 형식이 dict가 아닙니다.")
    return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}


def _build_candidate_frame(
    agg: list[_MethodRow],
    shift_profiles: dict[str, dict[str, float]],
    best_map: dict[tuple[str, str], str],
) -> pd.DataFrame:
    """정책 학습용 후보 프레임을 생성한다.

    Args:
        agg: 집계 결과 목록.
        shift_profiles: 도메인 이동 프로파일.
        best_map: 모델-도메인 best method 맵.

    Returns:
        학습/평가용 데이터프레임.

    Raises:
        ValueError: 프로파일이 누락된 도메인이 있을 때.
    """
    records: list[dict[str, Any]] = []
    for row in agg:
        if row.domain not in shift_profiles:
            raise ValueError(
                f"도메인 {row.domain}에 대한 shift profile이 없습니다."
            )
        profile = shift_profiles[row.domain]
        rec: dict[str, Any] = {
            "model": row.model,
            "domain": row.domain,
            "method": row.method,
            "mean_mae": row.mean_mae,
            "n": row.n,
            "is_best": int(best_map[(row.model, row.domain)] == row.method),
        }
        for key, value in profile.items():
            if isinstance(value, (int, float)):
                rec[f"shift_{key}"] = float(value)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _evaluate_leave_one_domain_out(
    frame: pd.DataFrame,
    random_seed: int,
) -> dict[str, Any]:
    """Leave-One-Domain-Out 방식으로 정책 성능을 평가한다.

    Args:
        frame: 후보 프레임.
        random_seed: 랜덤 시드.

    Returns:
        fold 및 전체 요약 딕셔너리.

    Raises:
        ValueError: 평가 불가능한 입력일 때.
    """
    domains = sorted(frame["domain"].unique().tolist())
    if len(domains) < 2:
        raise ValueError("LODO 평가를 위해 최소 2개 도메인이 필요합니다.")

    policy_methods = sorted(frame["method"].unique().tolist())
    folds: list[dict[str, Any]] = []

    for heldout in domains:
        train_df = frame[frame["domain"] != heldout].copy()
        test_df = frame[frame["domain"] == heldout].copy()
        if train_df.empty or test_df.empty:
            continue

        x_train = pd.get_dummies(
            train_df.drop(columns=["is_best", "mean_mae", "n"]),
            columns=["model", "domain", "method"],
        )
        x_test = pd.get_dummies(
            test_df.drop(columns=["is_best", "mean_mae", "n"]),
            columns=["model", "domain", "method"],
        )
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

        y_train = np.asarray(train_df["is_best"], dtype=int)

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            class_weight="balanced",
            max_depth=6,
            min_samples_leaf=2,
        )
        model.fit(x_train, y_train)
        prob_matrix = np.asarray(model.predict_proba(x_test), dtype=float)
        if prob_matrix.ndim == 2 and prob_matrix.shape[1] > 1:
            prob = prob_matrix[:, 1]
        else:
            prob = np.zeros(len(x_test), dtype=float)
        test_df = test_df.assign(pred_best_prob=prob)

        method_acc: dict[str, list[float]] = {}
        model_method_acc: dict[tuple[str, str], list[float]] = {}
        model_vals = [str(v) for v in train_df["model"].tolist()]
        method_vals = [str(v) for v in train_df["method"].tolist()]
        mae_vals = [float(v) for v in train_df["mean_mae"].tolist()]

        for model_name, method_name, mae_val in zip(
            model_vals, method_vals, mae_vals
        ):
            method_acc.setdefault(method_name, []).append(mae_val)
            model_method_acc.setdefault((model_name, method_name), []).append(mae_val)

        global_best_method = min(
            method_acc.items(), key=lambda kv: float(np.mean(kv[1]))
        )[0]

        model_best_method: dict[str, str] = {}
        model_names = sorted({k[0] for k in model_method_acc.keys()})
        for model_name in model_names:
            candidates: list[tuple[str, float]] = []
            for (m_name, method_name), values in model_method_acc.items():
                if m_name == model_name:
                    candidates.append((method_name, float(np.mean(values))))
            if candidates:
                best_method = min(candidates, key=lambda x: x[1])[0]
                model_best_method[model_name] = best_method

        top1_correct = 0
        regret_list: list[float] = []
        baseline_global_top1 = 0
        baseline_global_regret: list[float] = []
        baseline_model_top1 = 0
        baseline_model_regret: list[float] = []
        per_cell: list[dict[str, Any]] = []
        for key, group in test_df.groupby(["model", "domain"]):
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            model_name = str(key[0])
            domain_name = str(key[1])
            ordered = group.sort_values("pred_best_prob", ascending=False)
            rec_method = str(ordered.iloc[0]["method"])
            oracle_row = group.loc[group["is_best"] == 1]
            if oracle_row.empty:
                continue
            oracle_method = str(oracle_row.iloc[0]["method"])
            oracle_mae = float(oracle_row.iloc[0]["mean_mae"])
            rec_mae = float(group.loc[group["method"] == rec_method].iloc[0]["mean_mae"])

            global_mae = float(
                group.loc[group["method"] == global_best_method].iloc[0]["mean_mae"]
            )
            model_default = model_best_method.get(model_name, global_best_method)
            model_mae = float(
                group.loc[group["method"] == model_default].iloc[0]["mean_mae"]
            )

            top1_correct += int(rec_method == oracle_method)
            regret = 0.0 if oracle_mae <= 0 else (rec_mae - oracle_mae) / oracle_mae
            regret_list.append(float(regret))

            baseline_global_top1 += int(global_best_method == oracle_method)
            baseline_model_top1 += int(model_default == oracle_method)
            baseline_global_regret.append(
                0.0 if oracle_mae <= 0 else (global_mae - oracle_mae) / oracle_mae
            )
            baseline_model_regret.append(
                0.0 if oracle_mae <= 0 else (model_mae - oracle_mae) / oracle_mae
            )
            per_cell.append(
                {
                    "model": model_name,
                    "domain": domain_name,
                    "recommended_method": rec_method,
                    "oracle_method": oracle_method,
                    "oracle_mae": oracle_mae,
                    "recommended_mae": rec_mae,
                    "relative_regret": float(regret),
                    "baseline_global_method": global_best_method,
                    "baseline_global_mae": global_mae,
                    "baseline_model_method": model_default,
                    "baseline_model_mae": model_mae,
                }
            )

        n_cells = len(per_cell)
        fold_report = {
            "heldout_domain": heldout,
            "n_models": len({str(v) for v in test_df["model"].tolist()}),
            "n_methods": len(policy_methods),
            "n_cells": n_cells,
            "global_default_method": global_best_method,
            "model_defaults": model_best_method,
            "top1_accuracy": float(top1_correct / n_cells) if n_cells > 0 else 0.0,
            "mean_relative_regret": float(np.mean(regret_list)) if regret_list else 0.0,
            "baseline_global_top1_accuracy": float(baseline_global_top1 / n_cells)
            if n_cells > 0
            else 0.0,
            "baseline_global_mean_relative_regret": float(np.mean(baseline_global_regret))
            if baseline_global_regret
            else 0.0,
            "baseline_model_top1_accuracy": float(baseline_model_top1 / n_cells)
            if n_cells > 0
            else 0.0,
            "baseline_model_mean_relative_regret": float(np.mean(baseline_model_regret))
            if baseline_model_regret
            else 0.0,
            "cell_reports": per_cell,
        }
        folds.append(fold_report)

    top1_vals = [f["top1_accuracy"] for f in folds]
    regret_vals = [f["mean_relative_regret"] for f in folds]
    base_global_top1_vals = [f["baseline_global_top1_accuracy"] for f in folds]
    base_global_regret_vals = [f["baseline_global_mean_relative_regret"] for f in folds]
    base_model_top1_vals = [f["baseline_model_top1_accuracy"] for f in folds]
    base_model_regret_vals = [f["baseline_model_mean_relative_regret"] for f in folds]
    return {
        "cv": "leave_one_domain_out",
        "n_folds": len(folds),
        "folds": folds,
        "summary": {
            "mean_top1_accuracy": float(np.mean(top1_vals)) if top1_vals else 0.0,
            "std_top1_accuracy": float(np.std(top1_vals)) if top1_vals else 0.0,
            "mean_relative_regret": float(np.mean(regret_vals)) if regret_vals else 0.0,
            "baseline_global_mean_top1_accuracy": float(np.mean(base_global_top1_vals))
            if base_global_top1_vals
            else 0.0,
            "baseline_global_mean_relative_regret": float(np.mean(base_global_regret_vals))
            if base_global_regret_vals
            else 0.0,
            "baseline_model_mean_top1_accuracy": float(np.mean(base_model_top1_vals))
            if base_model_top1_vals
            else 0.0,
            "baseline_model_mean_relative_regret": float(np.mean(base_model_regret_vals))
            if base_model_regret_vals
            else 0.0,
        },
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

    domain_results_dir = Path(args.domain_results_dir)
    shift_profile_path = Path(args.shift_profile_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_domain_results(domain_results_dir)
    agg = _aggregate(rows)
    best_map = _best_method_map(agg)
    shift_profiles = _load_shift_profiles(shift_profile_path)
    frame = _build_candidate_frame(agg, shift_profiles, best_map)
    report = _evaluate_leave_one_domain_out(frame, random_seed=int(args.seed))

    grouped_best: dict[str, dict[str, str]] = {}
    for (model, domain), method in sorted(best_map.items()):
        grouped_best.setdefault(model, {})[domain] = method

    payload: dict[str, Any] = {
        "task": "shift_to_peft_policy",
        "n_raw_results": len(rows),
        "n_aggregated_rows": len(agg),
        "n_models": len({str(v) for v in frame["model"].tolist()}),
        "n_domains": len({str(v) for v in frame["domain"].tolist()}),
        "n_methods": len({str(v) for v in frame["method"].tolist()}),
        "oracle_best_method": grouped_best,
        "policy_evaluation": report,
    }

    out_path = output_dir / "shift_policy_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Shift policy 분석 저장 완료: %s", out_path)
    logger.info(
        "LODO top1=%.3f±%.3f, regret=%.4f",
        payload["policy_evaluation"]["summary"]["mean_top1_accuracy"],
        payload["policy_evaluation"]["summary"]["std_top1_accuracy"],
        payload["policy_evaluation"]["summary"]["mean_relative_regret"],
    )


if __name__ == "__main__":
    main()
