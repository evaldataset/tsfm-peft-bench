from __future__ import annotations

# pyright: reportMissingImports=false

"""NeurIPS 피벗 실행용 추가 실험 매니페스트 생성 스크립트.

기존 domain 결과 셀을 기준으로 seed 확장 실행 커맨드를 자동 생성한다.
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
    parser = argparse.ArgumentParser(description="피벗 실험 실행 매니페스트 생성")
    parser.add_argument(
        "--domain_results_dir",
        type=str,
        default="results/expansion/domain",
        help="기존 domain 결과 경로",
    )
    parser.add_argument(
        "--new_seeds",
        type=int,
        nargs="+",
        default=[7, 2024, 3407],
        help="추가 실행할 seed 목록",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pivot_analysis",
        help="출력 디렉토리",
    )
    return parser.parse_args()


def _load_cells(path: Path) -> list[tuple[str, str, str]]:
    """기존 결과에서 유니크 (model, method, domain) 셀을 추출한다.

    Args:
        path: domain 결과 디렉토리.

    Returns:
        유니크 셀 목록.

    Raises:
        ValueError: 입력 디렉토리가 없거나 결과가 비어 있을 때.
    """
    if not path.exists():
        raise ValueError(f"domain 결과 디렉토리가 없습니다: {path}")

    cells: set[tuple[str, str, str]] = set()
    for p in sorted(path.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                item = json.load(f)
            if not isinstance(item, dict):
                continue
            model = item.get("model")
            method = item.get("method")
            domain = item.get("domain")
            if all(isinstance(x, str) for x in (model, method, domain)):
                cells.add((str(model), str(method), str(domain)))
        except (OSError, json.JSONDecodeError):
            continue

    if not cells:
        raise ValueError("유효한 결과 셀을 찾지 못했습니다.")
    return sorted(cells)


def _to_hydra(model: str, method: str, domain: str, seed: int) -> str:
    """Hydra 실행 커맨드 변환.

    Args:
        model: 모델 이름.
        method: 방법 이름.
        domain: 도메인 이름.
        seed: 시드 값.

    Returns:
        실행 커맨드 문자열.

    Raises:
        ValueError: 지원하지 않는 값이 있을 때.
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
    }

    if model not in model_map:
        raise ValueError(f"지원하지 않는 model입니다: {model}")
    if method not in method_map:
        raise ValueError(f"지원하지 않는 method입니다: {method}")
    if domain not in domain_map:
        raise ValueError(f"지원하지 않는 domain입니다: {domain}")

    return (
        "python scripts/train.py "
        f"model={model_map[model]} adaptation={method_map[method]} data={domain_map[domain]} "
        f"seed={seed}"
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

    cells = _load_cells(Path(args.domain_results_dir))
    seeds = [int(s) for s in args.new_seeds]

    commands: list[str] = []
    manifests: list[dict[str, Any]] = []
    for model, method, domain in cells:
        for seed in seeds:
            cmd = _to_hydra(model=model, method=method, domain=domain, seed=seed)
            commands.append(cmd)
            manifests.append(
                {
                    "model": model,
                    "method": method,
                    "domain": domain,
                    "seed": seed,
                    "command": cmd,
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "pivot_seed_expansion_manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "pivot_seed_expansion",
                "n_unique_cells": len(cells),
                "new_seeds": seeds,
                "n_commands": len(commands),
                "commands": manifests,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    sh_path = out_dir / "pivot_seed_expansion_manifest.sh"
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        for cmd in commands:
            f.write(f"{cmd}\n")

    logger.info("피벗 seed 확장 매니페스트 저장 완료: %s", json_path)
    logger.info("실행 스크립트 저장 완료: %s", sh_path)
    logger.info("추가 실행 커맨드 수: %d", len(commands))


if __name__ == "__main__":
    main()
