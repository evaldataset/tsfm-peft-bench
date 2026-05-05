from __future__ import annotations

# pyright: reportMissingImports=false

"""FIM-NN 적응 방법 셀렉터 평가 스크립트.

FIM 지문 기반 최근접 이웃(FIM-NN) 셀렉터를 구현하고,
Leave-One-Domain-Out(LOOCV) 프로토콜로 Shift-NN 및 베이스라인과 비교한다.

평가 기준:
    - Top-1 accuracy: 추천 방법 == 오라클 방법 비율
    - Top-2 accuracy: 오라클 방법이 상위 2 추천 안에 포함되는 비율
    - Mean relative regret: (추천 MAE - 오라클 MAE) / 오라클 MAE 의 평균
    - Median relative regret
    - Max regret
    - High-regret miss rate: regret > 50% 셀 비율
    - Mean rank: 추천 방법의 평균 순위 (1=최적)

출력:
    results/fim_selector_evaluation.json
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_DIVERGENCE_THRESHOLD: float = 50.0
_HIGH_REGRET_THRESHOLD: float = 0.50
_BOOTSTRAP_N: int = 10_000
_BOOTSTRAP_SEED: int = 0
_FIM_METRIC: str = "cosine"

_PRIMARY_MODELS: list[str] = ["chronos", "moment", "moirai"]
_DOMAINS: list[str] = ["ett_m1", "finance", "smd", "physionet"]

_RESULTS_DIR = _PROJECT_ROOT / "results"
_DOMAIN_RESULTS_DIR = _RESULTS_DIR / "expansion" / "domain"
_FIM_FINGERPRINTS_PATH = _RESULTS_DIR / "fim_fingerprints.json"
_OUTPUT_PATH = _RESULTS_DIR / "fim_selector_evaluation.json"

# 5차원 이동 프로파일 (Shift-NN 비교용): 계산된 artifact에서 로드.
from scripts.build_selector import _SHIFT_PROFILES  # noqa: E402

# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------

@dataclass
class _RunRecord:
    """개별 실험 실행 레코드.

    Args:
        model: 모델 이름.
        domain: 도메인 이름.
        method: 적응 방법 이름.
        seed: 랜덤 시드.
        mae: Mean Absolute Error.

    Returns:
        _RunRecord 인스턴스.
    """

    model: str
    domain: str
    method: str
    seed: int
    mae: float


@dataclass
class _CellResult:
    """(모델, 도메인) 셀 단위 집계 결과.

    Args:
        model: 모델 이름.
        domain: 도메인 이름.
        method_maes: 방법별 평균 MAE.
        oracle_method: 최적 방법 이름.
        oracle_mae: 최적 방법의 평균 MAE.

    Returns:
        _CellResult 인스턴스.
    """

    model: str
    domain: str
    method_maes: dict[str, float]
    oracle_method: str
    oracle_mae: float

    def method_rank(self, method: str) -> int:
        """주어진 방법의 MAE 기준 순위 (1=최적)."""
        sorted_methods = sorted(self.method_maes, key=lambda m: self.method_maes[m])
        if method in sorted_methods:
            return sorted_methods.index(method) + 1
        return len(sorted_methods)

    def top2_methods(self) -> list[str]:
        """MAE 기준 상위 2개 방법 반환."""
        return sorted(self.method_maes, key=lambda m: self.method_maes[m])[:2]


@dataclass
class _SelectorResult:
    """셀렉터 전략 평가 결과.

    Args:
        strategy: 전략 이름.
        top1_accuracy: Top-1 정확도.
        top2_accuracy: Top-2 정확도.
        mean_regret: 평균 상대 regret.
        median_regret: 중앙값 regret.
        max_regret: 최대 regret.
        high_regret_rate: regret > 50% 셀 비율.
        mean_rank: 추천 방법의 평균 순위.
        ci_top1: Top-1의 95% 부트스트랩 CI.
        ci_mean_regret: mean_regret의 95% 부트스트랩 CI.
        per_cell: 셀별 상세 결과.

    Returns:
        _SelectorResult 인스턴스.
    """

    strategy: str
    top1_accuracy: float
    top2_accuracy: float
    mean_regret: float
    median_regret: float
    max_regret: float
    high_regret_rate: float
    mean_rank: float
    ci_top1: tuple[float, float] = (0.0, 0.0)
    ci_mean_regret: tuple[float, float] = (0.0, 0.0)
    per_cell: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """직렬화 가능한 딕셔너리로 변환."""
        return {
            "strategy": self.strategy,
            "top1_accuracy": self.top1_accuracy,
            "top2_accuracy": self.top2_accuracy,
            "mean_regret": self.mean_regret,
            "median_regret": self.median_regret,
            "max_regret": self.max_regret,
            "high_regret_rate": self.high_regret_rate,
            "mean_rank": self.mean_rank,
            "ci_top1_95": list(self.ci_top1),
            "ci_mean_regret_95": list(self.ci_mean_regret),
            "per_cell": self.per_cell,
        }


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------

def _load_domain_results() -> list[_RunRecord]:
    """domain 모드 JSON 결과 파일을 로드.

    Args:
        None.

    Returns:
        유효한 실행 레코드 목록.

    Raises:
        ValueError: 유효 결과가 없을 때.
    """
    if not _DOMAIN_RESULTS_DIR.exists():
        raise ValueError(
            f"domain 결과 디렉토리가 없습니다: {_DOMAIN_RESULTS_DIR}"
        )

    records: list[_RunRecord] = []
    skipped = 0

    for path in sorted(_DOMAIN_RESULTS_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception as exc:
            logger.warning("JSON 파싱 실패 [%s]: %s", path.name, exc)
            continue

        model = item.get("model", "")
        domain = item.get("domain", "")
        method = item.get("method", "")
        seed = int(item.get("seed", 0))
        metrics = item.get("metrics", {})
        mae = float(metrics.get("mae", float("inf")))

        if model not in _PRIMARY_MODELS:
            continue
        if mae > _DIVERGENCE_THRESHOLD or not np.isfinite(mae):
            skipped += 1
            continue

        records.append(_RunRecord(model=model, domain=domain, method=method, seed=seed, mae=mae))

    logger.info(
        "레코드 로드: %d개 유효, %d개 발산 제외",
        len(records),
        skipped,
    )
    if not records:
        raise ValueError("유효한 실험 결과가 없습니다.")
    return records


def _build_cells(records: list[_RunRecord]) -> list[_CellResult]:
    """실행 레코드를 (모델, 도메인) 셀로 집계.

    Args:
        records: 실행 레코드 목록.

    Returns:
        셀 결과 목록.
    """
    # (model, domain, method) → [mae, ...]
    cell_method_maes: dict[tuple[str, str, str], list[float]] = {}
    for rec in records:
        k = (rec.model, rec.domain, rec.method)
        cell_method_maes.setdefault(k, []).append(rec.mae)

    # (model, domain) → {method: mean_mae}
    cell_agg: dict[tuple[str, str], dict[str, float]] = {}
    for (model, domain, method), maes in cell_method_maes.items():
        cell_agg.setdefault((model, domain), {})[method] = float(np.mean(maes))

    cells: list[_CellResult] = []
    for (model, domain), method_maes in cell_agg.items():
        if not method_maes:
            continue
        oracle_method = min(method_maes, key=lambda m: method_maes[m])
        oracle_mae = method_maes[oracle_method]
        cells.append(
            _CellResult(
                model=model,
                domain=domain,
                method_maes=method_maes,
                oracle_method=oracle_method,
                oracle_mae=oracle_mae,
            )
        )

    logger.info("셀 집계: %d개", len(cells))
    return cells


# ---------------------------------------------------------------------------
# FIM 지문 로딩
# ---------------------------------------------------------------------------

def _load_fim_fingerprints() -> dict[str, dict[str, Any]]:
    """저장된 FIM 지문 데이터 로드.

    Args:
        None.

    Returns:
        ``{key: {"layer_profile": ..., "total_fim_norm": ..., "n_params": ...}}`` 딕셔너리.
        key는 ``"model_domain"`` 형식.

    Raises:
        FileNotFoundError: fim_fingerprints.json 파일이 없을 때.
    """
    if not _FIM_FINGERPRINTS_PATH.exists():
        raise FileNotFoundError(
            f"FIM 지문 파일을 찾을 수 없습니다: {_FIM_FINGERPRINTS_PATH}\n"
            "먼저 scripts/compute_fim_fingerprints.py 를 실행하세요."
        )

    with open(_FIM_FINGERPRINTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fingerprints: dict[str, dict[str, Any]] = data.get("fingerprints", {})
    logger.info("FIM 지문 로드: %d개 항목", len(fingerprints))
    return fingerprints


def _load_fim_npy(model_name: str, domain_name: str) -> np.ndarray | None:
    """저장된 FIM 대각선 배열 로드.

    Args:
        model_name: 모델 이름.
        domain_name: 도메인 이름.

    Returns:
        FIM 대각선 배열, 없으면 None.
    """
    npy_path = _RESULTS_DIR / f"fim_{model_name}_{domain_name}.npy"
    if not npy_path.exists():
        return None
    return np.load(str(npy_path))


# ---------------------------------------------------------------------------
# 거리 함수
# ---------------------------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 거리 계산 (1 - 코사인 유사도).

    Args:
        a: 첫 번째 벡터.
        b: 두 번째 벡터.

    Returns:
        코사인 거리 [0, 2].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return 1.0 - float(np.clip(sim, -1.0, 1.0))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """유클리드 거리 계산.

    Args:
        a: 첫 번째 벡터.
        b: 두 번째 벡터.

    Returns:
        유클리드 거리.
    """
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# 셀렉터 전략 구현
# ---------------------------------------------------------------------------

def _evaluate_strategy(
    cells: list[_CellResult],
    recommendations: dict[tuple[str, str], str],
    strategy_name: str,
) -> _SelectorResult:
    """추천 딕셔너리로부터 셀렉터 성능 지표를 계산.

    Args:
        cells: 전체 셀 결과 목록.
        recommendations: ``{(model, domain): recommended_method}`` 딕셔너리.
        strategy_name: 전략 이름.

    Returns:
        _SelectorResult 인스턴스.
    """
    top1_hits: list[int] = []
    top2_hits: list[int] = []
    regrets: list[float] = []
    ranks: list[int] = []
    per_cell: list[dict[str, Any]] = []

    for cell in cells:
        key = (cell.model, cell.domain)
        rec_method = recommendations.get(key)
        if rec_method is None:
            continue

        # 추천 방법이 해당 셀에 없으면 최악으로 처리
        if rec_method not in cell.method_maes:
            rec_mae = max(cell.method_maes.values())
        else:
            rec_mae = cell.method_maes[rec_method]

        top1 = int(rec_method == cell.oracle_method)
        top2 = int(rec_method in cell.top2_methods())
        regret = (rec_mae - cell.oracle_mae) / (cell.oracle_mae + 1e-8)
        rank = cell.method_rank(rec_method)

        top1_hits.append(top1)
        top2_hits.append(top2)
        regrets.append(regret)
        ranks.append(rank)

        per_cell.append({
            "model": cell.model,
            "domain": cell.domain,
            "oracle_method": cell.oracle_method,
            "oracle_mae": cell.oracle_mae,
            "recommended_method": rec_method,
            "recommended_mae": rec_mae,
            "top1": bool(top1),
            "top2": bool(top2),
            "regret": regret,
            "rank": rank,
        })

    n = len(top1_hits)
    if n == 0:
        return _SelectorResult(
            strategy=strategy_name,
            top1_accuracy=0.0,
            top2_accuracy=0.0,
            mean_regret=float("inf"),
            median_regret=float("inf"),
            max_regret=float("inf"),
            high_regret_rate=1.0,
            mean_rank=float("inf"),
        )

    top1_acc = float(np.mean(top1_hits))
    top2_acc = float(np.mean(top2_hits))
    mean_regret = float(np.mean(regrets))
    median_regret = float(np.median(regrets))
    max_regret = float(np.max(regrets))
    high_regret_rate = float(np.mean([r > _HIGH_REGRET_THRESHOLD for r in regrets]))
    mean_rank = float(np.mean(ranks))

    # 부트스트랩 CI
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    boot_top1 = np.array([
        np.mean(rng.choice(top1_hits, size=n, replace=True))
        for _ in range(_BOOTSTRAP_N)
    ])
    boot_regret = np.array([
        np.mean(rng.choice(regrets, size=n, replace=True))
        for _ in range(_BOOTSTRAP_N)
    ])
    ci_top1 = (float(np.percentile(boot_top1, 2.5)), float(np.percentile(boot_top1, 97.5)))
    ci_regret = (float(np.percentile(boot_regret, 2.5)), float(np.percentile(boot_regret, 97.5)))

    return _SelectorResult(
        strategy=strategy_name,
        top1_accuracy=top1_acc,
        top2_accuracy=top2_acc,
        mean_regret=mean_regret,
        median_regret=median_regret,
        max_regret=max_regret,
        high_regret_rate=high_regret_rate,
        mean_rank=mean_rank,
        ci_top1=ci_top1,
        ci_mean_regret=ci_regret,
        per_cell=per_cell,
    )


# ---------------------------------------------------------------------------
# LOOCV 평가 루프
# ---------------------------------------------------------------------------

def _loocv_fim_nn(
    cells: list[_CellResult],
) -> _SelectorResult:
    """FIM-NN 셀렉터의 LOOCV 평가.

    각 (모델, 도메인) 쌍을 테스트 셀로 삼고,
    같은 모델의 나머지 도메인 중 FIM 거리가 가장 가까운 도메인의
    오라클 방법을 추천한다.

    FIM 대각선 배열을 직접 로드하여 코사인 거리 기반 최근접 이웃을 찾는다.
    FIM 배열이 없으면 layer_profile의 L2 노름 벡터로 폴백한다.

    Args:
        cells: 전체 셀 결과 목록.

    Returns:
        FIM-NN 전략의 _SelectorResult.
    """
    fingerprints = _load_fim_fingerprints()

    recommendations: dict[tuple[str, str], str] = {}

    for test_cell in cells:
        test_model = test_cell.model
        test_domain = test_cell.domain

        # 같은 모델의 다른 도메인 셀
        train_cells = [
            c for c in cells
            if c.model == test_model and c.domain != test_domain
        ]
        if not train_cells:
            logger.warning(
                "FIM-NN: 학습 셀이 없음 [%s/%s]. 건너뜀.",
                test_model, test_domain,
            )
            continue

        # 테스트 FIM 로드
        test_fim = _load_fim_npy(test_model, test_domain)

        if test_fim is not None:
            # FIM 배열 기반 거리
            best_cell = None
            best_dist = float("inf")
            for tc in train_cells:
                train_fim = _load_fim_npy(tc.model, tc.domain)
                if train_fim is None or len(train_fim) != len(test_fim):
                    continue
                dist = _cosine_distance(test_fim.astype(np.float64), train_fim.astype(np.float64))
                if dist < best_dist:
                    best_dist = dist
                    best_cell = tc
        else:
            # 폴백: layer_profile의 노름 벡터 사용
            logger.debug(
                "FIM-NN: FIM 배열 없음, layer_profile 폴백 [%s/%s]",
                test_model, test_domain,
            )
            test_key = f"{test_model}_{test_domain}"
            test_fp = fingerprints.get(test_key, {})
            test_profile = test_fp.get("layer_profile", {})

            best_cell = None
            best_dist = float("inf")
            for tc in train_cells:
                train_key = f"{tc.model}_{tc.domain}"
                train_fp = fingerprints.get(train_key, {})
                train_profile = train_fp.get("layer_profile", {})

                common_layers = set(test_profile.keys()) & set(train_profile.keys())
                if not common_layers:
                    continue

                a = np.array([test_profile[l] for l in sorted(common_layers)], dtype=np.float64)
                b = np.array([train_profile[l] for l in sorted(common_layers)], dtype=np.float64)
                dist = _cosine_distance(a, b)
                if dist < best_dist:
                    best_dist = dist
                    best_cell = tc

        if best_cell is None:
            # 최후 폴백: 첫 번째 학습 셀의 오라클 사용
            best_cell = train_cells[0]
            logger.warning(
                "FIM-NN: 거리 계산 실패, 첫 번째 학습 셀 사용 [%s/%s]",
                test_model, test_domain,
            )

        recommendations[(test_model, test_domain)] = best_cell.oracle_method
        logger.debug(
            "FIM-NN [%s/%s] → 가장 유사한 도메인: %s → 추천: %s (거리=%.4f)",
            test_model, test_domain,
            best_cell.domain,
            best_cell.oracle_method,
            best_dist,
        )

    return _evaluate_strategy(cells, recommendations, strategy_name="fim_nn")


def _loocv_shift_nn(cells: list[_CellResult]) -> _SelectorResult:
    """Shift-NN 셀렉터의 LOOCV 평가 (기존 5차원 이동 프로파일 기반).

    Args:
        cells: 전체 셀 결과 목록.

    Returns:
        Shift-NN 전략의 _SelectorResult.
    """
    recommendations: dict[tuple[str, str], str] = {}

    for test_cell in cells:
        test_model = test_cell.model
        test_domain = test_cell.domain

        train_cells = [
            c for c in cells
            if c.model == test_model and c.domain != test_domain
        ]
        if not train_cells:
            continue

        test_profile = np.array(_SHIFT_PROFILES.get(test_domain, [0.0] * 5), dtype=np.float64)

        best_cell = None
        best_dist = float("inf")
        for tc in train_cells:
            train_profile = np.array(
                _SHIFT_PROFILES.get(tc.domain, [0.0] * 5), dtype=np.float64
            )
            dist = _euclidean_distance(test_profile, train_profile)
            if dist < best_dist:
                best_dist = dist
                best_cell = tc

        if best_cell is None:
            best_cell = train_cells[0]

        recommendations[(test_model, test_domain)] = best_cell.oracle_method

    return _evaluate_strategy(cells, recommendations, strategy_name="shift_nn")


def _loocv_always_lora(cells: list[_CellResult]) -> _SelectorResult:
    """항상 LoRA를 추천하는 베이스라인.

    Args:
        cells: 전체 셀 결과 목록.

    Returns:
        Always-LoRA 전략의 _SelectorResult.
    """
    recommendations = {(c.model, c.domain): "lora" for c in cells}
    return _evaluate_strategy(cells, recommendations, strategy_name="always_lora")


def _loocv_always_zero_shot(cells: list[_CellResult]) -> _SelectorResult:
    """항상 zero_shot을 추천하는 베이스라인.

    Args:
        cells: 전체 셀 결과 목록.

    Returns:
        Always-ZeroShot 전략의 _SelectorResult.
    """
    recommendations = {(c.model, c.domain): "zero_shot" for c in cells}
    return _evaluate_strategy(cells, recommendations, strategy_name="always_zero_shot")


def _loocv_random(cells: list[_CellResult]) -> _SelectorResult:
    """무작위 방법을 추천하는 베이스라인.

    Args:
        cells: 전체 셀 결과 목록.

    Returns:
        Random 전략의 _SelectorResult.
    """
    rng = np.random.default_rng(42)
    recommendations: dict[tuple[str, str], str] = {}
    for cell in cells:
        methods = list(cell.method_maes.keys())
        recommendations[(cell.model, cell.domain)] = str(rng.choice(methods))
    return _evaluate_strategy(cells, recommendations, strategy_name="random")


def _loocv_oracle(cells: list[_CellResult]) -> _SelectorResult:
    """오라클 상한선 (항상 최적 방법 선택).

    Args:
        cells: 전체 셀 결과 목록.

    Returns:
        Oracle 전략의 _SelectorResult.
    """
    recommendations = {(c.model, c.domain): c.oracle_method for c in cells}
    return _evaluate_strategy(cells, recommendations, strategy_name="oracle")


# ---------------------------------------------------------------------------
# 결과 비교 출력
# ---------------------------------------------------------------------------

def _print_comparison_table(results: list[_SelectorResult]) -> None:
    """전략별 성능 비교 테이블을 로그로 출력.

    Args:
        results: 전략별 _SelectorResult 목록.

    Returns:
        None.
    """
    header = (
        f"{'전략':<22} {'Top-1':>7} {'Top-2':>7} "
        f"{'Regret(mean)':>13} {'Regret(med)':>12} {'MaxReg':>8} "
        f"{'HiReg%':>8} {'Rank':>6}"
    )
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        logger.info(
            "%-22s %7.3f %7.3f %13.4f %12.4f %8.4f %8.3f %6.2f",
            r.strategy,
            r.top1_accuracy,
            r.top2_accuracy,
            r.mean_regret,
            r.median_regret,
            r.max_regret,
            r.high_regret_rate,
            r.mean_rank,
        )
    logger.info("=" * len(header))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    """FIM-NN 셀렉터 평가 메인 함수."""
    logger.info("FIM 셀렉터 평가 시작")

    # 실험 결과 로드
    records = _load_domain_results()
    cells = _build_cells(records)

    if not cells:
        logger.error("셀 데이터가 없습니다. 종료.")
        return

    # 각 전략 평가
    logger.info("전략 평가 중...")

    results: list[_SelectorResult] = []

    # Oracle (상한선)
    oracle_result = _loocv_oracle(cells)
    results.append(oracle_result)
    logger.info("Oracle 완료: Top-1=%.3f", oracle_result.top1_accuracy)

    # FIM-NN
    try:
        fim_nn_result = _loocv_fim_nn(cells)
        results.append(fim_nn_result)
        logger.info("FIM-NN 완료: Top-1=%.3f", fim_nn_result.top1_accuracy)
    except FileNotFoundError as exc:
        logger.warning("FIM-NN 건너뜀 (지문 파일 없음): %s", exc)

    # Shift-NN (기존 베이스라인)
    shift_nn_result = _loocv_shift_nn(cells)
    results.append(shift_nn_result)
    logger.info("Shift-NN 완료: Top-1=%.3f", shift_nn_result.top1_accuracy)

    # 베이스라인들
    always_lora = _loocv_always_lora(cells)
    results.append(always_lora)

    always_zs = _loocv_always_zero_shot(cells)
    results.append(always_zs)

    random_result = _loocv_random(cells)
    results.append(random_result)

    # 비교 테이블 출력
    _print_comparison_table(results)

    # ---------------------------------------------------------------------------
    # 결과 저장
    # ---------------------------------------------------------------------------
    output: dict[str, Any] = {
        "task": "fim_selector_evaluation",
        "divergence_threshold": _DIVERGENCE_THRESHOLD,
        "high_regret_threshold": _HIGH_REGRET_THRESHOLD,
        "bootstrap_n": _BOOTSTRAP_N,
        "n_records_loaded": len(records),
        "n_cells": len(cells),
        "domains": sorted(set(c.domain for c in cells)),
        "models": sorted(set(c.model for c in cells)),
        "fim_metric": _FIM_METRIC,
        "strategies": [r.to_dict() for r in results],
        "summary": {
            r.strategy: {
                "top1_accuracy": r.top1_accuracy,
                "mean_regret": r.mean_regret,
                "mean_rank": r.mean_rank,
            }
            for r in results
        },
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("FIM 셀렉터 평가 결과 저장: %s", _OUTPUT_PATH)
    logger.info("완료.")


if __name__ == "__main__":
    main()
