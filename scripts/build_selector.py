from __future__ import annotations

# pyright: reportMissingImports=false

"""Shift-Aware Adaptation Selector 구축 및 평가 스크립트.

도메인 이동 프로파일(5차원)과 아키텍처 메타데이터(4차원 one-hot)를 결합한
9차원 특징 벡터를 이용해 최적 적응 방법을 추천하는 셀렉터를 구축하고,
Leave-One-Domain-Out(LOOCV) 방식으로 10가지 전략의 성능을 비교한다.

평가 기준:
    - Top-1 accuracy: 추천 방법 == 오라클 방법 비율
    - Top-2 accuracy: 오라클 방법이 상위 2 추천 안에 포함되는 비율
    - Mean relative regret: (추천 MAE - 오라클 MAE) / 오라클 MAE 의 평균
    - Median relative regret: 중앙값 regret
    - Max regret: 최악 케이스 regret
    - High-regret miss rate: regret > 50% 인 셀 비율
    - Mean rank: 추천 방법의 평균 순위 (1=최적, 5=최악)
    - Bootstrap 95% CI: Top-1 및 mean regret의 신뢰구간 (10000 반복)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수 정의
# ---------------------------------------------------------------------------

# 계산된 shift profile JSON에서 사용할 5개 핵심 dimension.
# 하드코딩된 값 대신 이 키들을 domain_shift_profiles.json에서 읽어온다.
_SHIFT_PROFILE_PATH = Path("results/expansion_analysis_v2/domain_shift_profiles.json")

_SHIFT_DIM_KEYS: list[str] = [
    "amplitude_w1",
    "spectral_w1",
    "acf_distance",
    "nonstationarity_kpss_diff",
    "irregularity_perm_entropy_diff",
]

_SHIFT_DIM_NAMES: list[str] = [
    "amplitude",
    "spectral",
    "acf",
    "nonstationarity",
    "irregularity",
]


def _load_shift_profiles() -> dict[str, list[float]]:
    """계산된 shift profile artifact에서 5차원 프로파일 로드.

    Returns:
        도메인명 → 5차원 shift 벡터 매핑.

    Raises:
        FileNotFoundError: artifact 파일이 없을 때.
        KeyError: 필수 dimension key가 누락됐을 때.
    """
    if not _SHIFT_PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Shift profile artifact 없음: {_SHIFT_PROFILE_PATH}. "
            f"먼저 'python scripts/characterize_domains.py'를 실행하세요."
        )
    with open(_SHIFT_PROFILE_PATH) as fh:
        raw = json.load(fh)
    profiles: dict[str, list[float]] = {}
    for domain, dims in raw.items():
        if not isinstance(dims, dict):
            continue
        try:
            profiles[domain] = [float(dims[k]) for k in _SHIFT_DIM_KEYS]
        except KeyError as exc:
            raise KeyError(
                f"도메인 '{domain}'에서 누락된 shift dimension: {exc}"
            ) from exc
    return profiles


_SHIFT_PROFILES: dict[str, list[float]] = _load_shift_profiles()

# 아키텍처 유형 정의 (one-hot 인코딩 순서: enc_dec, enc_only, any_variate)
_ARCH_MAP: dict[str, str] = {
    "chronos": "enc_dec",
    "moment":  "enc_only",
    "moirai":  "any_variate",
    "timesfm": "dec_only",
}

_ARCH_TYPES: list[str] = ["enc_dec", "enc_only", "any_variate", "dec_only"]

# Divergence filter for the selector LOOCV pipeline.
#
# Pipeline-level note: the per-architecture ANOVA in
# scripts/reproduce_paper_tables.py uses the per-cell scale-aware rule
# 'MAE > 10 * same-(model, domain) zero-shot baseline'. This selector script
# uses an absolute threshold (50.0) for two reasons: (i) the selector pool
# already excludes prefix-tuning and is restricted to five primary methods,
# so the cell-level outliers are concentrated on Chronos+LoRA+ETTm1
# (mean 105+) and Chronos+Full-FT+PhysioNet (1400+), both of which exceed
# any reasonable absolute or scale-aware cutoff; (ii) Appendix
# sec:filter-sens shows that the selector plurality and LODO top-1 are
# invariant for k in {5, 10, 20, infinity} under the scale-aware rule, so
# the two rules are operationally equivalent at the (model, domain, method)
# aggregation level used by the selector.
_DIVERGENCE_THRESHOLD: float = 50.0

# Regret 경고 임계값
_HIGH_REGRET_THRESHOLD: float = 0.50

# 주요 3개 모델 (paper 범위)
_PRIMARY_MODELS: set[str] = {"chronos", "moment", "moirai"}

# Bootstrap 반복 횟수
_BOOTSTRAP_N: int = 10_000

# Bootstrap 랜덤 시드
_BOOTSTRAP_SEED: int = 0


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

    Raises:
        None.
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
        method_maes: 방법별 평균 MAE 딕셔너리.
        oracle_method: 최적 방법 이름.
        oracle_mae: 최적 방법의 평균 MAE.
        shift_features: 5차원 이동 프로파일 벡터.
        arch_onehot: 아키텍처 one-hot 벡터 (len == len(_ARCH_TYPES)).

    Returns:
        _CellResult 인스턴스.

    Raises:
        None.
    """

    model: str
    domain: str
    method_maes: dict[str, float]
    oracle_method: str
    oracle_mae: float
    shift_features: list[float]
    arch_onehot: list[float]

    @property
    def feature_vector(self) -> list[float]:
        """9차원 특징 벡터를 반환한다.

        Returns:
            [5 shift dims] + [4 arch one-hot] = 9 features.
        """
        return self.shift_features + self.arch_onehot

    def method_rank(self, method: str) -> int:
        """주어진 방법의 MAE 기준 순위를 반환한다 (1=최적, n=최악).

        Args:
            method: 방법 이름.

        Returns:
            순위 정수. 방법이 없으면 마지막 순위 반환.
        """
        sorted_methods = sorted(self.method_maes, key=lambda m: self.method_maes[m])
        if method in sorted_methods:
            return sorted_methods.index(method) + 1
        return len(sorted_methods)


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
        per_cell: 셀별 상세 결과.

    Returns:
        _SelectorResult 인스턴스.

    Raises:
        None.
    """

    strategy: str
    top1_accuracy: float
    top2_accuracy: float
    mean_regret: float
    median_regret: float
    max_regret: float
    high_regret_rate: float
    mean_rank: float
    per_cell: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    """로깅 설정을 초기화한다.

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


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------

def _load_domain_results(domain_results_dir: Path) -> list[_RunRecord]:
    """domain 모드 JSON 결과 파일을 로드한다.

    발산된 실행(MAE > _DIVERGENCE_THRESHOLD)은 제외한다.

    Args:
        domain_results_dir: JSON 결과 파일 디렉토리.

    Returns:
        유효한 실행 레코드 목록.

    Raises:
        ValueError: 디렉토리가 없거나 유효 결과가 없을 때.
    """
    if not domain_results_dir.exists():
        raise ValueError(
            f"domain 결과 디렉토리가 존재하지 않습니다: {domain_results_dir}"
        )

    records: list[_RunRecord] = []
    skipped_diverged = 0
    skipped_invalid = 0

    for path in sorted(domain_results_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("JSON 로드 실패 (%s): %s", path.name, exc)
            skipped_invalid += 1
            continue

        if not isinstance(item, dict):
            skipped_invalid += 1
            continue

        required_keys = ("model", "domain", "method")
        if not all(k in item for k in required_keys):
            skipped_invalid += 1
            continue

        metrics = item.get("metrics", {})
        mae = metrics.get("mae") if isinstance(metrics, dict) else None
        if not isinstance(mae, (int, float)) or np.isnan(mae):
            skipped_invalid += 1
            continue

        mae_f = float(mae)
        if mae_f > _DIVERGENCE_THRESHOLD:
            skipped_diverged += 1
            logger.debug(
                "발산 실행 제외: %s (MAE=%.2f)", path.name, mae_f
            )
            continue

        records.append(
            _RunRecord(
                model=str(item["model"]),
                domain=str(item["domain"]),
                method=str(item["method"]),
                seed=int(item.get("seed", 0)),
                mae=mae_f,
            )
        )

    logger.info(
        "로드 완료: %d 레코드 (발산 제외=%d, 형식 오류=%d)",
        len(records),
        skipped_diverged,
        skipped_invalid,
    )

    if not records:
        raise ValueError("유효한 domain 결과를 찾지 못했습니다.")

    return records


# ---------------------------------------------------------------------------
# 집계 및 셀 구축
# ---------------------------------------------------------------------------

def _build_cells(
    records: list[_RunRecord],
    shift_profiles: dict[str, list[float]],
    primary_only: bool = True,
) -> list[_CellResult]:
    """레코드를 (모델, 도메인) 셀 단위로 집계하고 특징 벡터를 구성한다.

    Args:
        records: 유효한 실행 레코드 목록.
        shift_profiles: 도메인별 5차원 이동 프로파일.
        primary_only: True이면 _PRIMARY_MODELS 소속 모델만 포함.

    Returns:
        셀 단위 결과 목록.

    Raises:
        None.
    """
    # (model, domain, method) -> [mae values]
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for rec in records:
        if primary_only and rec.model not in _PRIMARY_MODELS:
            continue
        if rec.domain not in shift_profiles:
            logger.debug(
                "shift profile 없음 - 셀 제외: model=%s domain=%s",
                rec.model,
                rec.domain,
            )
            continue
        key = (rec.model, rec.domain, rec.method)
        grouped.setdefault(key, []).append(rec.mae)

    # (model, domain) -> {method: mean_mae}
    cell_methods: dict[tuple[str, str], dict[str, float]] = {}
    for (model, domain, method), maes in grouped.items():
        cell_key = (model, domain)
        cell_methods.setdefault(cell_key, {})[method] = float(np.mean(maes))

    cells: list[_CellResult] = []
    for (model, domain), method_maes in sorted(cell_methods.items()):
        if not method_maes:
            continue

        oracle_method = min(method_maes, key=lambda m: method_maes[m])
        oracle_mae = method_maes[oracle_method]

        shift_vec = shift_profiles[domain]

        arch_type = _ARCH_MAP.get(model, "enc_dec")
        arch_onehot = [
            1.0 if a == arch_type else 0.0 for a in _ARCH_TYPES
        ]

        cells.append(
            _CellResult(
                model=model,
                domain=domain,
                method_maes=method_maes,
                oracle_method=oracle_method,
                oracle_mae=oracle_mae,
                shift_features=list(shift_vec),
                arch_onehot=arch_onehot,
            )
        )

    logger.info("셀 구축 완료: %d 셀", len(cells))
    return cells


# ---------------------------------------------------------------------------
# 평가 유틸리티
# ---------------------------------------------------------------------------

def _compute_regret(
    recommended_mae: float,
    oracle_mae: float,
) -> float:
    """상대 regret을 계산한다.

    Args:
        recommended_mae: 추천 방법의 MAE.
        oracle_mae: 오라클 최적 MAE.

    Returns:
        상대 regret 값. oracle_mae == 0 이면 0.0 반환.
    """
    if oracle_mae <= 0.0:
        return 0.0
    return (recommended_mae - oracle_mae) / oracle_mae


def _evaluate_strategy(
    strategy_name: str,
    recommendations: list[str],
    test_cells: list[_CellResult],
    top2_recommendations: list[list[str]] | None = None,
) -> _SelectorResult:
    """주어진 추천 목록에 대해 평가 지표를 계산한다.

    Args:
        strategy_name: 전략 이름.
        recommendations: test_cells 각 셀에 대한 1순위 추천 방법 목록.
        test_cells: 평가 대상 셀 목록.
        top2_recommendations: 각 셀에 대한 상위 2개 추천 목록 (None이면 top2=0).

    Returns:
        _SelectorResult 평가 결과.

    Raises:
        ValueError: 길이 불일치 시.
    """
    if len(recommendations) != len(test_cells):
        raise ValueError(
            f"추천 목록({len(recommendations)})과 셀 수({len(test_cells)}) 불일치"
        )

    top1_hits = 0
    top2_hits = 0
    regrets: list[float] = []
    ranks: list[int] = []
    per_cell: list[dict[str, Any]] = []

    for i, (cell, rec_method) in enumerate(zip(test_cells, recommendations)):
        # 추천 방법이 해당 셀에 없으면 가장 나쁜 방법의 MAE 사용
        if rec_method in cell.method_maes:
            rec_mae = cell.method_maes[rec_method]
        else:
            rec_mae = max(cell.method_maes.values())
            rec_method = f"{rec_method}(fallback)"

        top1_hit = int(rec_method == cell.oracle_method)
        top1_hits += top1_hit

        # Top-2 정확도
        if top2_recommendations is not None:
            top2_set = set(top2_recommendations[i])
            top2_hits += int(cell.oracle_method in top2_set)
        else:
            top2_hits += top1_hit  # top-2 = top-1 when only 1 recommendation

        regret = _compute_regret(rec_mae, cell.oracle_mae)
        regrets.append(regret)
        ranks.append(cell.method_rank(rec_method.replace("(fallback)", "")))

        per_cell.append(
            {
                "model": cell.model,
                "domain": cell.domain,
                "recommended_method": rec_method,
                "oracle_method": cell.oracle_method,
                "recommended_mae": rec_mae,
                "oracle_mae": cell.oracle_mae,
                "relative_regret": regret,
                "rank": ranks[-1],
            }
        )

    n = len(test_cells)
    regrets_arr = np.array(regrets, dtype=float)
    ranks_arr = np.array(ranks, dtype=float)

    return _SelectorResult(
        strategy=strategy_name,
        top1_accuracy=top1_hits / n if n > 0 else 0.0,
        top2_accuracy=top2_hits / n if n > 0 else 0.0,
        mean_regret=float(np.mean(regrets_arr)) if len(regrets_arr) > 0 else 0.0,
        median_regret=float(np.median(regrets_arr)) if len(regrets_arr) > 0 else 0.0,
        max_regret=float(np.max(regrets_arr)) if len(regrets_arr) > 0 else 0.0,
        high_regret_rate=float(np.mean(regrets_arr > _HIGH_REGRET_THRESHOLD))
        if len(regrets_arr) > 0
        else 0.0,
        mean_rank=float(np.mean(ranks_arr)) if len(ranks_arr) > 0 else 0.0,
        per_cell=per_cell,
    )


# ---------------------------------------------------------------------------
# 셀렉터 전략 구현
# ---------------------------------------------------------------------------

def _strategy_global_best(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
) -> list[str]:
    """전략 A: 훈련 도메인에서 가장 자주 오라클인 방법을 일괄 추천한다.

    Args:
        train_cells: 훈련 셀 목록.
        test_cells: 테스트 셀 목록 (미사용, 균일 추천).

    Returns:
        test_cells 각 셀에 대한 추천 방법 목록.
    """
    counts: dict[str, int] = {}
    for cell in train_cells:
        counts[cell.oracle_method] = counts.get(cell.oracle_method, 0) + 1
    best = max(counts, key=lambda m: counts[m])
    return [best] * len(test_cells)


def _strategy_lodo_majority(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
) -> list[str]:
    """전략 B: 훈련 도메인 오라클의 다수결로 추천한다 (LODO baseline).

    Args:
        train_cells: 훈련 셀 목록.
        test_cells: 테스트 셀 목록.

    Returns:
        test_cells 각 셀에 대한 추천 방법 목록.
    """
    # 방법별 오라클 빈도 집계
    counts: dict[str, int] = {}
    for cell in train_cells:
        counts[cell.oracle_method] = counts.get(cell.oracle_method, 0) + 1
    majority = max(counts, key=lambda m: counts[m])
    return [majority] * len(test_cells)


def _strategy_arch_default(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
) -> list[str]:
    """전략 C: 아키텍처별 최빈 오라클 방법을 추천한다.

    Args:
        train_cells: 훈련 셀 목록.
        test_cells: 테스트 셀 목록.

    Returns:
        test_cells 각 셀에 대한 추천 방법 목록.
    """
    arch_counts: dict[str, dict[str, int]] = {}
    for cell in train_cells:
        arch = _ARCH_MAP.get(cell.model, "enc_dec")
        if arch not in arch_counts:
            arch_counts[arch] = {}
        arch_counts[arch][cell.oracle_method] = (
            arch_counts[arch].get(cell.oracle_method, 0) + 1
        )

    # 아키텍처별 최빈 방법
    arch_best: dict[str, str] = {}
    for arch, counts in arch_counts.items():
        arch_best[arch] = max(counts, key=lambda m: counts[m])

    # 전체 훈련 데이터 기반 글로벌 최빈 (fallback용)
    global_counts: dict[str, int] = {}
    for cell in train_cells:
        global_counts[cell.oracle_method] = (
            global_counts.get(cell.oracle_method, 0) + 1
        )
    global_best = max(global_counts, key=lambda m: global_counts[m])

    recs: list[str] = []
    for cell in test_cells:
        arch = _ARCH_MAP.get(cell.model, "enc_dec")
        recs.append(arch_best.get(arch, global_best))
    return recs


def _strategy_nearest_neighbor(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
) -> list[str]:
    """전략 D: 이동 프로파일 유클리드 거리 기반 최근접 이웃 방법 추천.

    훈련 셀 중 shift_features 거리가 가장 가까운 셀의 오라클 방법을 사용한다.

    Args:
        train_cells: 훈련 셀 목록.
        test_cells: 테스트 셀 목록.

    Returns:
        test_cells 각 셀에 대한 추천 방법 목록.
    """
    if not train_cells:
        return ["zero_shot"] * len(test_cells)

    train_shift = np.array(
        [c.shift_features for c in train_cells], dtype=float
    )
    train_oracles = [c.oracle_method for c in train_cells]

    recs: list[str] = []
    for cell in test_cells:
        query = np.array(cell.shift_features, dtype=float)
        dists = np.linalg.norm(train_shift - query, axis=1)
        nearest_idx = int(np.argmin(dists))
        recs.append(train_oracles[nearest_idx])
    return recs


def _strategy_regret_weighted_nn(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
    epsilon: float = 1e-6,
) -> list[str]:
    """전략 E: Regret 가중 최근접 이웃.

    각 훈련 셀을 거리의 역수로 가중한 후, 방법별 기대 regret을 계산하여
    기대 regret이 가장 낮은 방법을 추천한다.

    Args:
        train_cells: 훈련 셀 목록.
        test_cells: 테스트 셀 목록.
        epsilon: 거리 0 방지용 작은 상수.

    Returns:
        test_cells 각 셀에 대한 추천 방법 목록.
    """
    if not train_cells:
        return ["zero_shot"] * len(test_cells)

    train_shift = np.array(
        [c.shift_features for c in train_cells], dtype=float
    )

    # 전체 훈련 셀에서 관측된 방법 집합
    all_methods: set[str] = set()
    for cell in train_cells:
        all_methods.update(cell.method_maes.keys())

    recs: list[str] = []
    for test_cell in test_cells:
        query = np.array(test_cell.shift_features, dtype=float)
        dists = np.linalg.norm(train_shift - query, axis=1)
        weights = 1.0 / (dists + epsilon)

        # 방법별 가중 regret 계산
        method_weighted_regret: dict[str, float] = {}
        method_weight_sum: dict[str, float] = {}

        for w, train_cell in zip(weights, train_cells):
            train_oracle_mae = train_cell.oracle_mae
            for method, method_mae in train_cell.method_maes.items():
                regret = _compute_regret(method_mae, train_oracle_mae)
                method_weighted_regret[method] = (
                    method_weighted_regret.get(method, 0.0) + w * regret
                )
                method_weight_sum[method] = (
                    method_weight_sum.get(method, 0.0) + w
                )

        # 가중 평균 regret 계산
        method_avg_regret: dict[str, float] = {}
        for method in method_weighted_regret:
            total_w = method_weight_sum.get(method, 0.0)
            if total_w > 0.0:
                method_avg_regret[method] = (
                    method_weighted_regret[method] / total_w
                )

        if method_avg_regret:
            best_method = min(method_avg_regret, key=lambda m: method_avg_regret[m])
        else:
            best_method = "zero_shot"

        recs.append(best_method)
    return recs


def _strategy_logistic_meta(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
    random_seed: int = 42,
) -> list[str]:
    """전략 F: 다항 로지스틱 회귀 메타 학습기.

    특징: [5 shift dims] + [4 arch one-hot] = 9 features
    레이블: oracle_method
    C 파라미터: GridSearchCV로 교차 검증 선택 (훈련 셀 내부 CV).

    셀이 3개 미만이면 고정 C=1.0으로 fitting한다.

    Args:
        train_cells: 훈련 셀 목록.
        test_cells: 테스트 셀 목록.
        random_seed: LogisticRegression 랜덤 시드.

    Returns:
        test_cells 각 셀에 대한 추천 방법 목록.
    """
    if not train_cells:
        return ["zero_shot"] * len(test_cells)

    x_train = np.array([c.feature_vector for c in train_cells], dtype=float)
    y_train = [c.oracle_method for c in train_cells]
    x_test = np.array([c.feature_vector for c in test_cells], dtype=float)

    # 훈련 셀 수가 너무 적으면 CV 불가
    if len(train_cells) < 4:
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=random_seed,
        )
        clf.fit(x_train, y_train)
    else:
        base_clf = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            random_state=random_seed,
        )
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
        # StratifiedKFold requires n_splits <= min class size
        from collections import Counter
        min_class = min(Counter(y_train).values())
        cv_folds = min(3, len(train_cells), max(2, min_class))
        gs = GridSearchCV(base_clf, param_grid, cv=cv_folds, scoring="accuracy")
        gs.fit(x_train, y_train)
        clf = gs.best_estimator_
        logger.debug("로지스틱 메타 학습기 최적 C=%.4f", gs.best_params_["C"])

    preds = clf.predict(x_test)
    return [str(p) for p in preds]


def _strategy_random_forest(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
    random_seed: int = 42,
) -> list[str]:
    """전략 G: RandomForestClassifier를 이용해 최적 방법을 예측한다.

    특징: [5 shift dims] + [4 arch one-hot] = 9 features
    레이블: oracle_method
    설정: n_estimators=50, max_depth=3

    Args:
        train_cells: 훈련 셀 목록.
        test_cells: 테스트 셀 목록.
        random_seed: RandomForestClassifier 랜덤 시드.

    Returns:
        test_cells 각 셀에 대한 추천 방법 목록.
    """
    if not train_cells:
        return ["zero_shot"] * len(test_cells)

    x_train = np.array([c.feature_vector for c in train_cells], dtype=float)
    y_train = [c.oracle_method for c in train_cells]
    x_test = np.array([c.feature_vector for c in test_cells], dtype=float)

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=random_seed,
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return [str(p) for p in preds]


def _strategy_decision_tree(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
    random_seed: int = 42,
) -> list[str]:
    """전략 H: DecisionTreeClassifier를 이용해 최적 방법을 예측한다.

    특징: [5 shift dims] + [4 arch one-hot] = 9 features
    레이블: oracle_method

    Args:
        train_cells: 훈련 셀 목록.
        test_cells: 테스트 셀 목록.
        random_seed: DecisionTreeClassifier 랜덤 시드.

    Returns:
        test_cells 각 셀에 대한 추천 방법 목록.
    """
    if not train_cells:
        return ["zero_shot"] * len(test_cells)

    x_train = np.array([c.feature_vector for c in train_cells], dtype=float)
    y_train = [c.oracle_method for c in train_cells]

    x_test = np.array([c.feature_vector for c in test_cells], dtype=float)

    clf = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=1,
        random_state=random_seed,
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return [str(p) for p in preds]


def _strategy_oracle(
    test_cells: list[_CellResult],
) -> list[str]:
    """전략 I: 오라클 상한 (항상 최적 방법 선택).

    Args:
        test_cells: 테스트 셀 목록.

    Returns:
        test_cells 각 셀에 대한 oracle 방법 목록.
    """
    return [cell.oracle_method for cell in test_cells]


def _strategy_random(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
    random_seed: int = 42,
) -> list[str]:
    """전략 J: 무작위 방법 선택 (하한 baseline).

    훈련 셀에서 관측된 모든 방법 중 무작위 선택.

    Note:
        이 함수는 단일 무작위 추출만 수행한다. 전체 평가에서는
        ``_strategy_random_multi_seed``가 더 안정적인 평균 regret을 산출한다.

    Args:
        train_cells: 훈련 셀 목록.
        test_cells: 테스트 셀 목록.
        random_seed: 난수 시드.

    Returns:
        test_cells 각 셀에 대한 추천 방법 목록.
    """
    all_methods: list[str] = sorted(
        {m for cell in train_cells for m in cell.method_maes}
    )
    if not all_methods:
        return ["zero_shot"] * len(test_cells)

    rng = np.random.default_rng(random_seed)
    choices = rng.choice(all_methods, size=len(test_cells), replace=True)
    return [str(c) for c in choices]


def _evaluate_random_marginal(
    train_cells: list[_CellResult],
    test_cells: list[_CellResult],
    n_seeds: int = 1000,
) -> _SelectorResult:
    """무작위 추천의 marginal 기대값을 다수 seed로 평균낸 평가 결과.

    한 셀에서 무작위 선택의 기대 regret은 해당 셀에서 관측된 모든
    method의 regret 평균과 같다 (균등 추출 가정). 이 함수는 그
    closed-form 평균값을 반환하기 위해 각 셀에 대해 n_seeds 만큼
    무작위 추출을 수행하고 모든 결과를 평균낸다.

    Args:
        train_cells: 훈련 셀 목록 (방법 후보 집합 결정).
        test_cells: 테스트 셀 목록.
        n_seeds: 무작위 추출 반복 수.

    Returns:
        n_seeds 평균에 기반한 _SelectorResult.

    Raises:
        ValueError: train_cells가 비었을 때.
    """
    all_methods: list[str] = sorted(
        {m for cell in train_cells for m in cell.method_maes}
    )
    if not all_methods:
        raise ValueError("훈련 셀에 사용 가능한 방법이 없습니다.")

    n = len(test_cells)
    if n == 0:
        return _SelectorResult(
            strategy="random_marginal",
            top1_accuracy=0.0,
            top2_accuracy=0.0,
            mean_regret=0.0,
            median_regret=0.0,
            max_regret=0.0,
            high_regret_rate=0.0,
            mean_rank=0.0,
        )

    # 각 cell의 method-wise regret을 미리 계산
    cell_method_regrets: list[dict[str, float]] = []
    for cell in test_cells:
        method_regrets: dict[str, float] = {}
        for method in all_methods:
            if method in cell.method_maes:
                method_regrets[method] = _compute_regret(
                    cell.method_maes[method], cell.oracle_mae
                )
            else:
                method_regrets[method] = _compute_regret(
                    max(cell.method_maes.values()), cell.oracle_mae
                )
        cell_method_regrets.append(method_regrets)

    # n_seeds 시드 × n cells 평균 regret/top1 계산
    accumulated_top1 = 0.0
    accumulated_regret = 0.0
    accumulated_max_regret = 0.0
    cell_avg_regret = [0.0] * n
    cell_avg_top1 = [0.0] * n

    for seed_idx in range(n_seeds):
        rng = np.random.default_rng(seed_idx)
        choices = rng.choice(all_methods, size=n, replace=True)

        seed_max = 0.0
        for i, (cell, choice) in enumerate(zip(test_cells, choices)):
            r = cell_method_regrets[i].get(str(choice), 0.0)
            top1 = float(str(choice) == cell.oracle_method)
            cell_avg_regret[i] += r / n_seeds
            cell_avg_top1[i] += top1 / n_seeds
            accumulated_regret += r
            accumulated_top1 += top1
            seed_max = max(seed_max, r)
        accumulated_max_regret += seed_max

    total_draws = n_seeds * n
    return _SelectorResult(
        strategy="random_marginal",
        top1_accuracy=accumulated_top1 / total_draws,
        top2_accuracy=accumulated_top1 / total_draws,
        mean_regret=accumulated_regret / total_draws,
        median_regret=float(np.median(cell_avg_regret)),
        max_regret=accumulated_max_regret / n_seeds,
        high_regret_rate=float(
            np.mean([r > _HIGH_REGRET_THRESHOLD for r in cell_avg_regret])
        ),
        mean_rank=0.0,
        per_cell=[
            {
                "model": cell.model,
                "domain": cell.domain,
                "recommended_method": "random(marginal)",
                "oracle_method": cell.oracle_method,
                "recommended_mae": float("nan"),
                "oracle_mae": cell.oracle_mae,
                "relative_regret": cell_avg_regret[i],
                "rank": 0,
            }
            for i, cell in enumerate(test_cells)
        ],
    )


# ---------------------------------------------------------------------------
# Bootstrap 신뢰구간
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = _BOOTSTRAP_N,
    ci_level: float = 0.95,
    rng_seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap 방식으로 95% 신뢰구간을 계산한다.

    Args:
        values: 원본 값 목록 (셀 단위).
        n_bootstrap: Bootstrap 반복 횟수.
        ci_level: 신뢰 수준.
        rng_seed: 난수 시드.

    Returns:
        (lower, upper) 신뢰구간 튜플.
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(rng_seed)
    boot_means = np.array(
        [
            np.mean(rng.choice(arr, size=len(arr), replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return (lower, upper)


# ---------------------------------------------------------------------------
# LOOCV 평가
# ---------------------------------------------------------------------------

def _run_loocv(
    cells: list[_CellResult],
    random_seed: int = 42,
) -> dict[str, list[_SelectorResult]]:
    """Leave-One-Domain-Out 교차 검증을 실행한다.

    10가지 전략 각각에 대해 도메인별 fold를 수행하고 결과를 수집한다.

    Args:
        cells: 전체 셀 목록.
        random_seed: 결정 트리 및 RF 시드.

    Returns:
        전략 이름 -> fold별 _SelectorResult 목록.

    Raises:
        ValueError: 도메인이 2개 미만일 때.
    """
    domains = sorted({c.domain for c in cells})
    if len(domains) < 2:
        raise ValueError(
            f"LOOCV에 최소 2개 도메인 필요, 현재: {len(domains)}"
        )

    strategy_keys = [
        "global_best",
        "lodo_majority",
        "arch_default",
        "nearest_neighbor",
        "regret_weighted_nn",
        "logistic_meta",
        "random_forest",
        "decision_tree",
        "oracle",
        "random",
    ]
    strategy_results: dict[str, list[_SelectorResult]] = {k: [] for k in strategy_keys}

    for heldout in domains:
        train_cells = [c for c in cells if c.domain != heldout]
        test_cells = [c for c in cells if c.domain == heldout]

        logger.info(
            "LOOCV fold: heldout=%s  train=%d 셀  test=%d 셀",
            heldout,
            len(train_cells),
            len(test_cells),
        )

        if not train_cells or not test_cells:
            logger.warning("fold 건너뜀: 빈 train 또는 test (%s)", heldout)
            continue

        recs_gb = _strategy_global_best(train_cells, test_cells)
        recs_lm = _strategy_lodo_majority(train_cells, test_cells)
        recs_ad = _strategy_arch_default(train_cells, test_cells)
        recs_nn = _strategy_nearest_neighbor(train_cells, test_cells)
        recs_rwnn = _strategy_regret_weighted_nn(train_cells, test_cells)
        recs_lr = _strategy_logistic_meta(train_cells, test_cells, random_seed)
        recs_rf = _strategy_random_forest(train_cells, test_cells, random_seed)
        recs_dt = _strategy_decision_tree(train_cells, test_cells, random_seed)
        recs_oracle = _strategy_oracle(test_cells)
        recs_rand = _strategy_random(train_cells, test_cells, random_seed)

        strategy_results["global_best"].append(
            _evaluate_strategy(f"global_best[{heldout}]", recs_gb, test_cells)
        )
        strategy_results["lodo_majority"].append(
            _evaluate_strategy(f"lodo_majority[{heldout}]", recs_lm, test_cells)
        )
        strategy_results["arch_default"].append(
            _evaluate_strategy(f"arch_default[{heldout}]", recs_ad, test_cells)
        )
        strategy_results["nearest_neighbor"].append(
            _evaluate_strategy(f"nearest_neighbor[{heldout}]", recs_nn, test_cells)
        )
        strategy_results["regret_weighted_nn"].append(
            _evaluate_strategy(f"regret_weighted_nn[{heldout}]", recs_rwnn, test_cells)
        )
        strategy_results["logistic_meta"].append(
            _evaluate_strategy(f"logistic_meta[{heldout}]", recs_lr, test_cells)
        )
        strategy_results["random_forest"].append(
            _evaluate_strategy(f"random_forest[{heldout}]", recs_rf, test_cells)
        )
        strategy_results["decision_tree"].append(
            _evaluate_strategy(f"decision_tree[{heldout}]", recs_dt, test_cells)
        )
        strategy_results["oracle"].append(
            _evaluate_strategy(f"oracle[{heldout}]", recs_oracle, test_cells)
        )
        strategy_results["random"].append(
            _evaluate_strategy(f"random[{heldout}]", recs_rand, test_cells)
        )

    return strategy_results


# ---------------------------------------------------------------------------
# 결과 집계 및 출력
# ---------------------------------------------------------------------------

def _aggregate_strategy(
    fold_results: list[_SelectorResult],
    strategy_name: str,
) -> dict[str, Any]:
    """폴드별 결과를 집계하고 Bootstrap CI를 계산한다.

    Args:
        fold_results: 폴드별 _SelectorResult 목록.
        strategy_name: 전략 이름.

    Returns:
        집계 딕셔너리.

    Raises:
        None.
    """
    top1_vals = [r.top1_accuracy for r in fold_results]
    top2_vals = [r.top2_accuracy for r in fold_results]
    regret_vals = [r.mean_regret for r in fold_results]
    median_regret_vals = [r.median_regret for r in fold_results]
    max_regret_vals = [r.max_regret for r in fold_results]
    hr_vals = [r.high_regret_rate for r in fold_results]
    rank_vals = [r.mean_rank for r in fold_results]
    all_per_cell = [cell for r in fold_results for cell in r.per_cell]

    # 셀 단위 값으로 Bootstrap CI 계산 (fold 수가 4뿐이므로 셀 단위가 더 정직함)
    all_top1_hits = [
        float(cell["recommended_method"] == cell["oracle_method"])
        for cell in all_per_cell
    ]
    all_regrets = [float(cell["relative_regret"]) for cell in all_per_cell]

    top1_ci = _bootstrap_ci(all_top1_hits)
    regret_ci = _bootstrap_ci(all_regrets)

    return {
        "strategy": strategy_name,
        "n_folds": len(fold_results),
        "n_cells_total": len(all_per_cell),
        # Top-1
        "mean_top1_accuracy": float(np.mean(top1_vals)) if top1_vals else 0.0,
        "std_top1_accuracy": float(np.std(top1_vals)) if top1_vals else 0.0,
        "top1_ci_lower": top1_ci[0],
        "top1_ci_upper": top1_ci[1],
        # Top-2
        "mean_top2_accuracy": float(np.mean(top2_vals)) if top2_vals else 0.0,
        # Regret
        "mean_relative_regret": float(np.mean(regret_vals)) if regret_vals else 0.0,
        "std_relative_regret": float(np.std(regret_vals)) if regret_vals else 0.0,
        "regret_ci_lower": regret_ci[0],
        "regret_ci_upper": regret_ci[1],
        "median_relative_regret": float(np.mean(median_regret_vals)) if median_regret_vals else 0.0,
        "max_regret": float(np.max(max_regret_vals)) if max_regret_vals else 0.0,
        "high_regret_miss_rate": float(np.mean(hr_vals)) if hr_vals else 0.0,
        # Rank
        "mean_rank": float(np.mean(rank_vals)) if rank_vals else 0.0,
        "per_cell": all_per_cell,
    }


def _print_comparison_table(summaries: list[dict[str, Any]]) -> None:
    """비교 테이블을 로그로 출력한다.

    Args:
        summaries: 전략별 집계 딕셔너리 목록.

    Returns:
        None.

    Raises:
        None.
    """
    header = (
        f"{'Strategy':<24} "
        f"{'Top-1':>7} "
        f"{'95%CI':>14} "
        f"{'Top-2':>7} "
        f"{'MnRgrt':>8} "
        f"{'MdRgrt':>8} "
        f"{'MaxRgrt':>8} "
        f"{'HiRg%':>7} "
        f"{'MnRank':>7}"
    )
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for s in summaries:
        ci_str = f"[{s['top1_ci_lower']:.3f},{s['top1_ci_upper']:.3f}]"
        logger.info(
            "%-24s %6.3f  %-14s %6.3f  %7.4f  %7.4f  %7.4f  %6.1f%%  %6.2f",
            s["strategy"],
            s["mean_top1_accuracy"],
            ci_str,
            s["mean_top2_accuracy"],
            s["mean_relative_regret"],
            s["median_relative_regret"],
            s["max_regret"],
            s["high_regret_miss_rate"] * 100.0,
            s["mean_rank"],
        )
    logger.info(sep)


def _build_latex_table(summaries: list[dict[str, Any]]) -> str:
    """LaTeX 테이블 프래그먼트를 생성한다.

    전략별 Top-1 (95% CI), Top-2, Mean Regret (95% CI), Median Regret,
    Max Regret, Hi-Regret%, Mean Rank을 포함한다.

    Args:
        summaries: 전략별 집계 딕셔너리 목록.

    Returns:
        LaTeX 문자열.

    Raises:
        None.
    """
    display_names: dict[str, str] = {
        "global_best": r"Global-Best",
        "lodo_majority": r"LODO Majority",
        "arch_default": r"Arch-Default",
        "nearest_neighbor": r"Shift-NN",
        "regret_weighted_nn": r"Regret-Weighted NN (ours)",
        "logistic_meta": r"Logistic Meta (ours)",
        "random_forest": r"Random Forest (ours)",
        "decision_tree": r"Decision-Tree",
        "oracle": r"\textit{Oracle (upper bound)}",
        "random": r"\textit{Random single-seed}",
        "random_marginal": r"\textit{Random (1000-seed mean)}",
    }

    # 가장 높은 top-1 (oracle 제외) 강조
    non_oracle = [
        s for s in summaries
        if s["strategy"] not in ("oracle", "random", "random_marginal")
    ]
    best_top1 = (
        max(s["mean_top1_accuracy"] for s in non_oracle) if non_oracle else -1.0
    )
    best_regret = (
        min(s["mean_relative_regret"] for s in non_oracle) if non_oracle else 1e9
    )

    lines: list[str] = [
        r"% Auto-generated by build_selector.py",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        (
            r"Strategy & Top-1 (\%) [95\% CI] & Top-2 (\%) "
            r"& Mean Regret [95\% CI] & Median Regret & Max Regret "
            r"& Hi-Regret\% & Mean Rank \\"
        ),
        r"\midrule",
    ]

    # oracle과 random은 구분선 위에
    main_strategies = [
        s for s in summaries
        if s["strategy"] not in ("oracle", "random", "random_marginal")
    ]
    bound_strategies = [
        s for s in summaries
        if s["strategy"] in ("oracle", "random", "random_marginal")
    ]

    def _fmt_row(s: dict[str, Any]) -> str:
        """단일 행 LaTeX 문자열을 생성한다."""
        name = display_names.get(s["strategy"], s["strategy"])
        top1_pct = s["mean_top1_accuracy"] * 100.0
        top1_lo = s["top1_ci_lower"] * 100.0
        top1_hi = s["top1_ci_upper"] * 100.0
        top2_pct = s["mean_top2_accuracy"] * 100.0
        mean_r = s["mean_relative_regret"]
        rci_lo = s["regret_ci_lower"]
        rci_hi = s["regret_ci_upper"]
        med_r = s["median_relative_regret"]
        max_r = s["max_regret"]
        hr = s["high_regret_miss_rate"] * 100.0
        rank = s["mean_rank"]

        top1_str = rf"{top1_pct:.1f} [{top1_lo:.1f}, {top1_hi:.1f}]"
        regret_str = rf"{mean_r:.4f} [{rci_lo:.4f}, {rci_hi:.4f}]"

        is_best_top1 = (
            abs(s["mean_top1_accuracy"] - best_top1) < 1e-9
            and s["strategy"] not in ("oracle", "random")
        )
        is_best_regret = (
            abs(s["mean_relative_regret"] - best_regret) < 1e-9
            and s["strategy"] not in ("oracle", "random")
        )

        if is_best_top1:
            top1_str = rf"\textbf{{{top1_str}}}"
        if is_best_regret:
            regret_str = rf"\textbf{{{regret_str}}}"

        return (
            rf"{name} & {top1_str} & {top2_pct:.1f} & "
            rf"{regret_str} & {med_r:.4f} & {max_r:.4f} & "
            rf"{hr:.1f} & {rank:.2f} \\"
        )

    for s in main_strategies:
        lines.append(_fmt_row(s))

    if bound_strategies:
        lines.append(r"\midrule")
        for s in bound_strategies:
            lines.append(_fmt_row(s))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _save_per_cell_detail(
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """셀별 상세 결과를 JSON으로 저장한다.

    도메인, 모델, 각 전략의 추천 방법, 오라클, 각 전략의 regret을 포함한다.

    Args:
        summaries: 전략별 집계 딕셔너리 목록.
        output_path: 저장 경로.

    Returns:
        None.

    Raises:
        None.
    """
    # (model, domain) -> {strategy: cell_info}
    cell_index: dict[tuple[str, str], dict[str, Any]] = {}

    for s in summaries:
        strategy = s["strategy"]
        for cell in s["per_cell"]:
            key = (cell["model"], cell["domain"])
            if key not in cell_index:
                cell_index[key] = {
                    "model": cell["model"],
                    "domain": cell["domain"],
                    "oracle_method": cell["oracle_method"],
                    "oracle_mae": cell["oracle_mae"],
                }
            cell_index[key][f"{strategy}_rec"] = cell["recommended_method"]
            cell_index[key][f"{strategy}_regret"] = cell["relative_regret"]
            cell_index[key][f"{strategy}_rank"] = cell.get("rank", None)

    rows = sorted(cell_index.values(), key=lambda r: (r["model"], r["domain"]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    logger.info("셀별 상세 결과 저장: %s", output_path)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

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

    repo_root = Path(__file__).parent.parent
    domain_results_dir = repo_root / "results" / "expansion" / "domain"
    output_dir = repo_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Shift-Aware Adaptation Selector 구축 시작 ===")
    logger.info("도메인 결과 디렉토리: %s", domain_results_dir)

    # 1. 결과 로드
    records = _load_domain_results(domain_results_dir)

    # Selector 평가는 기존 5개 canonical methods 기준으로 유지
    # (DoRA는 별도 supplementary comparison; selector 성능 해석 보호)
    _SELECTOR_METHODS = {"zero_shot", "head_only", "lora", "adapter", "full_fine_tuning"}
    n_before = len(records)
    records = [r for r in records if r.method in _SELECTOR_METHODS]
    logger.info("Selector 메서드 필터: %d → %d (제외: DoRA/prefix)", n_before, len(records))

    # 2. 셀 구축 (primary 3 models + hardcoded shift profiles)
    cells = _build_cells(records, _SHIFT_PROFILES, primary_only=True)

    domains_found = sorted({c.domain for c in cells})
    models_found = sorted({c.model for c in cells})
    logger.info("도메인: %s", domains_found)
    logger.info("모델: %s", models_found)

    # 3. 셀별 오라클 방법 요약 로그
    logger.info("=== 셀별 오라클 방법 ===")
    for cell in sorted(cells, key=lambda c: (c.model, c.domain)):
        logger.info(
            "  model=%-10s  domain=%-12s  oracle=%-20s  oracle_mae=%.4f",
            cell.model,
            cell.domain,
            cell.oracle_method,
            cell.oracle_mae,
        )

    # 4. LOOCV 실행
    logger.info("=== LOOCV 평가 시작 (Bootstrap CI: n=%d) ===", _BOOTSTRAP_N)
    strategy_fold_results = _run_loocv(cells, random_seed=42)

    # 5. 집계
    strategy_order = [
        "global_best",
        "lodo_majority",
        "arch_default",
        "nearest_neighbor",
        "regret_weighted_nn",
        "logistic_meta",
        "random_forest",
        "decision_tree",
        "oracle",
        "random",
    ]
    summaries = [
        _aggregate_strategy(strategy_fold_results[s], s)
        for s in strategy_order
        if s in strategy_fold_results
    ]

    # 6. 콘솔 출력
    logger.info("=== 전략별 성능 비교 (LOOCV) ===")
    _print_comparison_table(summaries)

    # 7. JSON 저장
    json_path = output_dir / "selector_evaluation.json"
    payload: dict[str, Any] = {
        "task": "shift_aware_adaptation_selector",
        "divergence_threshold": _DIVERGENCE_THRESHOLD,
        "high_regret_threshold": _HIGH_REGRET_THRESHOLD,
        "bootstrap_n": _BOOTSTRAP_N,
        "n_records_loaded": len(records),
        "n_cells": len(cells),
        "domains": domains_found,
        "models": models_found,
        "shift_profiles": _SHIFT_PROFILES,
        "shift_dim_names": _SHIFT_DIM_NAMES,
        "arch_types": _ARCH_TYPES,
        "strategies": summaries,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("JSON 결과 저장: %s", json_path)

    # 8. LaTeX 테이블 저장
    tex_path = output_dir / "selector_table.tex"
    latex_str = _build_latex_table(summaries)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    logger.info("LaTeX 테이블 저장: %s", tex_path)

    # 9. 셀별 상세 결과 저장
    detail_path = output_dir / "selector_per_cell_detail.json"
    _save_per_cell_detail(summaries, detail_path)

    # 10. 결과 요약 로그
    oracle_s = next((s for s in summaries if s["strategy"] == "oracle"), None)
    random_s = next((s for s in summaries if s["strategy"] == "random"), None)
    nn_s = next((s for s in summaries if s["strategy"] == "nearest_neighbor"), None)
    rwnn_s = next((s for s in summaries if s["strategy"] == "regret_weighted_nn"), None)
    lr_s = next((s for s in summaries if s["strategy"] == "logistic_meta"), None)
    rf_s = next((s for s in summaries if s["strategy"] == "random_forest"), None)

    logger.info("=== 최종 요약 ===")
    if oracle_s:
        logger.info(
            "Oracle upper bound: top1=%.3f  mean_regret=%.4f",
            oracle_s["mean_top1_accuracy"],
            oracle_s["mean_relative_regret"],
        )
    if random_s:
        logger.info(
            "Random lower bound: top1=%.3f  mean_regret=%.4f",
            random_s["mean_top1_accuracy"],
            random_s["mean_relative_regret"],
        )
    if nn_s:
        logger.info(
            "Shift-NN:          top1=%.3f [%.3f,%.3f]  mean_regret=%.4f [%.4f,%.4f]",
            nn_s["mean_top1_accuracy"],
            nn_s["top1_ci_lower"],
            nn_s["top1_ci_upper"],
            nn_s["mean_relative_regret"],
            nn_s["regret_ci_lower"],
            nn_s["regret_ci_upper"],
        )
    if rwnn_s:
        logger.info(
            "Regret-W NN:       top1=%.3f [%.3f,%.3f]  mean_regret=%.4f [%.4f,%.4f]",
            rwnn_s["mean_top1_accuracy"],
            rwnn_s["top1_ci_lower"],
            rwnn_s["top1_ci_upper"],
            rwnn_s["mean_relative_regret"],
            rwnn_s["regret_ci_lower"],
            rwnn_s["regret_ci_upper"],
        )
    if lr_s:
        logger.info(
            "Logistic Meta:     top1=%.3f [%.3f,%.3f]  mean_regret=%.4f [%.4f,%.4f]",
            lr_s["mean_top1_accuracy"],
            lr_s["top1_ci_lower"],
            lr_s["top1_ci_upper"],
            lr_s["mean_relative_regret"],
            lr_s["regret_ci_lower"],
            lr_s["regret_ci_upper"],
        )
    if rf_s:
        logger.info(
            "Random Forest:     top1=%.3f [%.3f,%.3f]  mean_regret=%.4f [%.4f,%.4f]",
            rf_s["mean_top1_accuracy"],
            rf_s["top1_ci_lower"],
            rf_s["top1_ci_upper"],
            rf_s["mean_relative_regret"],
            rf_s["regret_ci_lower"],
            rf_s["regret_ci_upper"],
        )

    logger.info("=== 완료 ===")


if __name__ == "__main__":
    main()
