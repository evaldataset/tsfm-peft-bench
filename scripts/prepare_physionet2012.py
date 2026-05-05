from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversionStats:
    total_records: int
    converted_records: int
    skipped_missing_target: int
    skipped_empty_target: int
    total_points: int


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhysioNet Challenge 2012 set-a를 patient-level CSV로 변환"
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/physionet_raw"),
        help="set-a.zip 또는 set-a/가 있는 원시 데이터 디렉토리",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/physionet"),
        help="변환된 CSV 출력 디렉토리",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="HR",
        help="추출할 PhysioNet 변수명 (기본값: HR)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 CSV가 있어도 덮어쓴다",
    )
    return parser.parse_args()


def _time_to_minutes(value: str) -> int:
    hour_text, minute_text = value.split(":", maxsplit=1)
    return int(hour_text) * 60 + int(minute_text)


def _ensure_raw_records(raw_dir: Path) -> Path:
    set_a_dir = raw_dir / "set-a"
    if set_a_dir.exists():
        return set_a_dir

    archive_path = raw_dir / "set-a.zip"
    if not archive_path.exists():
        raise FileNotFoundError(
            f"PhysioNet archive not found: {archive_path}. "
            "set-a.zip 또는 set-a/ 디렉토리가 필요합니다."
        )

    logger.info("원시 archive 압축 해제: %s", archive_path)
    with ZipFile(archive_path) as archive:
        archive.extractall(raw_dir)

    if not set_a_dir.exists():
        raise FileNotFoundError(
            f"압축 해제 후에도 set-a 디렉토리를 찾을 수 없습니다: {set_a_dir}"
        )
    return set_a_dir


def _convert_record(record_path: Path, output_dir: Path, target_col: str) -> tuple[str, int]:
    df = pd.read_csv(record_path)
    expected_cols = {"Time", "Parameter", "Value"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"예상 컬럼 {sorted(expected_cols)} 가 없습니다: {record_path}"
        )

    target_df = df.loc[df["Parameter"] == target_col, ["Time", "Value"]].copy()
    if target_df.empty:
        return "missing_target", 0

    target_df["Value"] = pd.to_numeric(target_df["Value"], errors="coerce")
    target_df = target_df.dropna(subset=["Value"])
    target_df = target_df.loc[target_df["Value"] >= 0.0]
    if target_df.empty:
        return "empty_target", 0

    target_df["minutes"] = target_df["Time"].map(_time_to_minutes)
    target_df = target_df.sort_values("minutes", kind="stable")

    output_df = pd.DataFrame(
        {target_col: target_df["Value"].astype("float32").to_numpy()}
    )
    output_path = output_dir / f"{record_path.stem}.csv"
    output_df.to_csv(output_path, index=False)
    return "converted", len(output_df)


def _convert_directory(
    records_dir: Path,
    output_dir: Path,
    target_col: str,
    overwrite: bool,
) -> ConversionStats:
    record_paths = sorted(records_dir.glob("*.txt"))
    if not record_paths:
        raise FileNotFoundError(f"원시 레코드가 없습니다: {records_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    converted_records = 0
    skipped_missing_target = 0
    skipped_empty_target = 0
    total_points = 0

    for record_path in record_paths:
        output_path = output_dir / f"{record_path.stem}.csv"
        if output_path.exists() and not overwrite:
            converted_records += 1
            total_points += len(pd.read_csv(output_path))
            continue

        status, num_points = _convert_record(record_path, output_dir, target_col)
        if status == "converted":
            converted_records += 1
            total_points += num_points
        elif status == "missing_target":
            skipped_missing_target += 1
        else:
            skipped_empty_target += 1

    return ConversionStats(
        total_records=len(record_paths),
        converted_records=converted_records,
        skipped_missing_target=skipped_missing_target,
        skipped_empty_target=skipped_empty_target,
        total_points=total_points,
    )


def main() -> None:
    _setup_logging()
    args = _parse_args()

    records_dir = _ensure_raw_records(args.raw_dir)
    stats = _convert_directory(
        records_dir=records_dir,
        output_dir=args.output_dir,
        target_col=args.target_col,
        overwrite=args.overwrite,
    )

    logger.info(
        (
            "PhysioNet 변환 완료: total=%d, converted=%d, missing_target=%d, "
            "empty_target=%d, total_points=%d, output=%s"
        ),
        stats.total_records,
        stats.converted_records,
        stats.skipped_missing_target,
        stats.skipped_empty_target,
        stats.total_points,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
