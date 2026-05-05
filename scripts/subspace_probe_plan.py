from __future__ import annotations

# pyright: reportMissingImports=false

"""적응 서브스페이스 메커니즘 분석을 위한 실행 계획 생성 스크립트.

체크포인트 존재 여부를 점검하고, 모델별 성공/실패 셀을 자동 선택해
layer-wise 표현 변화(예: CKA) 분석을 위한 최소 실험 계획을 생성한다.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

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
    parser = argparse.ArgumentParser(description="적응 서브스페이스 probe 계획 생성")
    parser.add_argument(
        "--domain_results_dir",
        type=str,
        default="results/expansion/domain",
        help="domain 모드 결과 디렉토리",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="checkpoints",
        help="체크포인트 루트 디렉토리",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pivot_analysis",
        help="출력 디렉토리",
    )
    return parser.parse_args()


def _load_domain_rows(path: Path) -> list[dict[str, Any]]:
    """domain 모드 결과 로드.

    Args:
        path: domain 결과 디렉토리.

    Returns:
        유효한 결과 리스트.

    Raises:
        ValueError: 디렉토리가 없거나 결과가 비어 있을 때.
    """
    if not path.exists():
        raise ValueError(f"domain 결과 디렉토리가 없습니다: {path}")

    rows: list[dict[str, Any]] = []
    for p in sorted(path.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                item = json.load(f)
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics", {})
            mae = metrics.get("mae") if isinstance(metrics, dict) else None
            if not isinstance(mae, (int, float)):
                continue
            if not all(k in item for k in ("model", "method", "domain")):
                continue
            rows.append(item)
        except (OSError, json.JSONDecodeError):
            continue

    if not rows:
        raise ValueError("유효한 domain 결과를 찾지 못했습니다.")
    return rows


def _aggregate_seed_mean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """seed 축 평균 MAE로 집계.

    Args:
        rows: 원본 결과 리스트.

    Returns:
        모델-방법-도메인 집계 리스트.

    Raises:
        None.
    """
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["method"]), str(row["domain"]))
        grouped.setdefault(key, []).append(float(row["metrics"]["mae"]))

    out: list[dict[str, Any]] = []
    for (model, method, domain), maes in sorted(grouped.items()):
        out.append(
            {
                "model": model,
                "method": method,
                "domain": domain,
                "mean_mae": float(sum(maes) / len(maes)),
                "n": len(maes),
            }
        )
    return out


def _select_success_failure_cells(agg: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """모델별 성공/실패 셀을 선택한다.

    Args:
        agg: 집계 결과 리스트.

    Returns:
        model -> {success, failure} 매핑.

    Raises:
        None.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in agg:
        by_model.setdefault(str(row["model"]), []).append(row)

    picks: dict[str, dict[str, Any]] = {}
    for model, rows in sorted(by_model.items()):
        success = min(rows, key=lambda r: float(r["mean_mae"]))
        failure = max(rows, key=lambda r: float(r["mean_mae"]))
        picks[model] = {"success": success, "failure": failure}
    return picks


def _count_checkpoints(root: Path) -> dict[str, Any]:
    """체크포인트 존재 현황을 집계한다.

    Args:
        root: 체크포인트 루트 경로.

    Returns:
        체크포인트 수와 샘플 경로 정보를 담은 딕셔너리.

    Raises:
        None.
    """
    if not root.exists():
        return {
            "exists": False,
            "n_pt": 0,
            "n_ckpt": 0,
            "sample_paths": [],
        }

    pt_files = sorted(root.rglob("*.pt"))
    ckpt_files = sorted(root.rglob("*.ckpt"))
    sample_paths = [str(p) for p in (pt_files[:3] + ckpt_files[:3])]
    return {
        "exists": True,
        "n_pt": len(pt_files),
        "n_ckpt": len(ckpt_files),
        "sample_paths": sample_paths,
    }


def _to_train_command(model: str, method: str, domain: str) -> str:
    """Hydra 학습 커맨드 템플릿을 생성한다.

    Args:
        model: 모델 이름.
        method: 방법 이름.
        domain: 도메인 이름.

    Returns:
        실행 가능한 학습 커맨드 문자열.

    Raises:
        ValueError: 지원하지 않는 이름이 입력될 때.
    """
    model_map = {
        "chronos": "chronos",
        "moment": "moment",
        "moirai": "moirai",
        "timesfm": "timesfm",
    }
    method_map = {
        "zero_shot": "zero_shot",
        "head_only": "head",
        "lora": "lora",
        "adapter": "adapter",
        "full_fine_tuning": "full_ft",
    }
    domain_map = {
        "ett_m1": "ett_m1",
        "finance": "finance",
        "smd": "smd",
        "physionet": "physionet",
    }

    if model not in model_map:
        raise ValueError(f"지원하지 않는 model입니다: {model}")
    if method not in method_map:
        raise ValueError(f"지원하지 않는 method입니다: {method}")
    if domain not in domain_map:
        raise ValueError(f"지원하지 않는 domain입니다: {domain}")

    return (
        "python scripts/train.py "
        f"model={model_map[model]} adaptation={method_map[method]} data={domain_map[domain]}"
    )


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

    rows = _load_domain_rows(Path(args.domain_results_dir))
    agg = _aggregate_seed_mean(rows)
    picks = _select_success_failure_cells(agg)
    ckpt_status = _count_checkpoints(Path(args.checkpoint_root))

    probes: list[dict[str, Any]] = []
    for model, pair in sorted(picks.items()):
        for tag in ("success", "failure"):
            cell = pair[tag]
            probes.append(
                {
                    "probe_type": tag,
                    "model": model,
                    "method": cell["method"],
                    "domain": cell["domain"],
                    "mean_mae": cell["mean_mae"],
                    "n": cell["n"],
                    "train_command_template": _to_train_command(
                        model=model,
                        method=str(cell["method"]),
                        domain=str(cell["domain"]),
                    ),
                    "analysis_notes": [
                        "동일 seed로 pre/post activation 저장",
                        "layer-wise CKA/feature drift 비교",
                        "update norm 집중도(late vs early) 계산",
                    ],
                }
            )

    out: dict[str, Any] = {
        "task": "adaptation_subspace_probe_plan",
        "checkpoint_status": ckpt_status,
        "n_raw_rows": len(rows),
        "n_aggregated_cells": len(agg),
        "probe_cells": probes,
        "execution_guideline": {
            "phase_1": "성공/실패 셀 각각 1회 재학습 + activation dump",
            "phase_2": "CKA 및 layer drift 분석",
            "phase_3": "공통 적응축(top-k) 압축 실험",
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subspace_probe_plan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    logger.info("서브스페이스 probe 계획 저장 완료: %s", out_path)
    ckpt_status = out.get("checkpoint_status", {})
    if isinstance(ckpt_status, dict):
        logger.info(
            "checkpoint exists=%s, pt=%d, ckpt=%d",
            bool(ckpt_status.get("exists", False)),
            int(ckpt_status.get("n_pt", 0)),
            int(ckpt_status.get("n_ckpt", 0)),
        )


if __name__ == "__main__":
    main()
