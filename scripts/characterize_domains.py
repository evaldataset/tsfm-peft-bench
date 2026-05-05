from __future__ import annotations

"""도메인별 분포 이동 프로파일 특성화 스크립트.

ETTm1, Finance(exchange_rate), SMD 데이터셋의 train→test 분포 이동을 정량화한다.
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.data.ett import ETTConfig, load_ett
from src.data.finance import FinanceConfig, load_finance
from src.data.physionet import PhysioNetConfig, load_physionet
from src.data.shift_metrics import compute_shift_profile
from src.data.smd import SMDConfig, load_smd

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
    parser = argparse.ArgumentParser(
        description="도메인별 분포 이동 프로파일 특성화"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/expansion_analysis_v2",
        help="출력 디렉토리",
    )
    return parser.parse_args()


def _sample_if_large(data: np.ndarray, max_samples: int = 10000) -> np.ndarray:
    """대용량 데이터 샘플링.

    Args:
        data: 입력 데이터 배열.
        max_samples: 최대 샘플 수.

    Returns:
        샘플링된 데이터 (원본이 max_samples보다 작으면 원본 반환).

    Raises:
        None.
    """
    if len(data) <= max_samples:
        return data
    # 균등 샘플링
    indices = np.linspace(0, len(data) - 1, max_samples, dtype=int)
    return data[indices]


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles: dict[str, dict[str, float]] = {}

    # ─── ETTm1 ────────────────────────────────────────────────────
    logger.info("ETTm1 분포 이동 프로파일 계산 중...")
    try:
        ett_config = ETTConfig(
            dataset="ETTm1",
            path="data/ETT-small/ETTm1.csv",
            target_col="OT",
        )
        train_ds, _, test_ds = load_ett(ett_config)
        train_vals = _sample_if_large(train_ds.data.flatten())
        test_vals = _sample_if_large(test_ds.data.flatten())
        profile = compute_shift_profile(train_vals, test_vals)
        profiles["ett_m1"] = asdict(profile)
        logger.info("ETTm1 완료: amplitude_w1=%.4f", profile.amplitude_w1)
    except Exception as exc:
        logger.error("ETTm1 처리 실패: %s", exc)

    # ─── Finance (Exchange Rate) ──────────────────────────────────
    logger.info("Finance 분포 이동 프로파일 계산 중...")
    try:
        finance_config = FinanceConfig(
            dataset="exchange_rate",
            path="data/exchange_rate/exchange_rate.csv",
            target_col=0,
        )
        train_ds, _, test_ds = load_finance(finance_config)
        train_vals = _sample_if_large(train_ds.data.flatten())
        test_vals = _sample_if_large(test_ds.data.flatten())
        profile = compute_shift_profile(train_vals, test_vals)
        profiles["finance"] = asdict(profile)
        logger.info("Finance 완료: amplitude_w1=%.4f", profile.amplitude_w1)
    except Exception as exc:
        logger.error("Finance 처리 실패: %s", exc)

    # ─── SMD ──────────────────────────────────────────────────────
    logger.info("SMD 분포 이동 프로파일 계산 중...")
    try:
        smd_config = SMDConfig(
            dataset="SMD",
            path="data/SMD",
            target_col=0,
        )
        train_ds, _, test_ds = load_smd(smd_config)
        train_vals = _sample_if_large(train_ds.data.flatten())
        test_vals = _sample_if_large(test_ds.data.flatten())
        profile = compute_shift_profile(train_vals, test_vals)
        profiles["smd"] = asdict(profile)
        logger.info("SMD 완료: amplitude_w1=%.4f", profile.amplitude_w1)
    except Exception as exc:
        logger.error("SMD 처리 실패: %s", exc)

    # ─── PhysioNet ────────────────────────────────────────────────
    logger.info("PhysioNet 분포 이동 프로파일 계산 중...")
    try:
        physio_config = PhysioNetConfig(
            dataset="PhysioNet2012",
            data_dir="data/physionet",
            target_col="HR",
        )
        train_ds, _, test_ds = load_physionet(physio_config)
        train_vals = _sample_if_large(train_ds.data.flatten())
        test_vals = _sample_if_large(test_ds.data.flatten())
        profile = compute_shift_profile(train_vals, test_vals)
        profiles["physionet"] = asdict(profile)
        logger.info("PhysioNet 완료: amplitude_w1=%.4f", profile.amplitude_w1)
    except Exception as exc:
        logger.error("PhysioNet 처리 실패: %s", exc)

    # ─── 저장 ─────────────────────────────────────────────────────
    output_path = output_dir / "domain_shift_profiles.json"
    with open(output_path, "w") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    logger.info("도메인 분포 이동 프로파일 저장 완료: %s", output_path)
    logger.info("총 %d개 도메인 처리 완료", len(profiles))


if __name__ == "__main__":
    main()
