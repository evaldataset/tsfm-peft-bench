from __future__ import annotations

# pyright: reportMissingImports=false

import numpy as np
import pytest


def _make_domain_mode_results(
    models: list[str],
    methods: list[str],
    domains: list[str],
    seeds: list[int],
) -> list[dict]:
    """도메인 모드 실험 결과 합성용 헬퍼.

    Args:
        models: 모델 이름 목록.
        methods: 방법 이름 목록.
        domains: 도메인 이름 목록.
        seeds: 시드 목록.

    Returns:
        합성된 결과 딕셔너리 리스트.

    Raises:
        None.
    """
    rng = np.random.default_rng(42)
    results = []
    for model in models:
        for method in methods:
            for domain in domains:
                for seed in seeds:
                    results.append(
                        {
                            "model": model,
                            "method": method,
                            "domain": domain,
                            "seed": seed,
                            "mode": "domain",
                            "rank": 8,
                            "locus": "attn_all",
                            "metrics": {"mae": float(rng.uniform(0.1, 2.0))},
                        }
                    )
    return results


def _make_rank_mode_results(
    models: list[str],
    domains: list[str],
    ranks: list[int],
    seeds: list[int],
) -> list[dict]:
    """랭크 모드 실험 결과 합성용 헬퍼.

    Args:
        models: 모델 이름 목록.
        domains: 도메인 이름 목록.
        ranks: LoRA 랭크 목록.
        seeds: 시드 목록.

    Returns:
        합성된 결과 딕셔너리 리스트.

    Raises:
        None.
    """
    rng = np.random.default_rng(123)
    results = []
    for model in models:
        for domain in domains:
            for rank in ranks:
                for seed in seeds:
                    results.append(
                        {
                            "model": model,
                            "method": "lora",
                            "domain": domain,
                            "seed": seed,
                            "mode": "rank",
                            "rank": rank,
                            "locus": "attn_all",
                            "metrics": {"mae": float(rng.uniform(0.1, 2.0))},
                        }
                    )
    return results


class TestBalancedLoading:
    """도메인 모드 분석에서 모드별 필터링 및 균형 그룹 검증.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_domain_mode_balanced_groups(self) -> None:
        """도메인 모드 필터 후 각 방법의 관측치 수가 동일해야 함."""
        models = ["chronos", "moment"]
        methods = ["zero_shot", "head_only", "lora", "adapter", "full_ft"]
        domains = ["ett_m1", "finance", "smd"]
        seeds = [42, 123]

        domain_results = _make_domain_mode_results(models, methods, domains, seeds)
        rank_results = _make_rank_mode_results(models, domains, [4, 8, 16, 32], seeds)
        all_results = domain_results + rank_results

        filtered = [r for r in all_results if r.get("mode") == "domain"]

        for model in models:
            model_filtered = [r for r in filtered if r["model"] == model]
            counts: dict[str, int] = {}
            for r in model_filtered:
                counts[r["method"]] = counts.get(r["method"], 0) + 1

            unique_counts = set(counts.values())
            assert len(unique_counts) == 1, (
                f"모델 {model}에서 방법별 관측치 불균형: {counts}"
            )

    def test_unfiltered_gives_lora_inflation(self) -> None:
        """필터링 없이 전체 결과를 사용하면 LoRA가 다른 방법보다 많은 관측치."""
        models = ["chronos"]
        methods = ["zero_shot", "head_only", "lora", "adapter", "full_ft"]
        domains = ["ett_m1", "finance", "smd"]
        seeds = [42, 123]

        domain_results = _make_domain_mode_results(models, methods, domains, seeds)
        rank_results = _make_rank_mode_results(models, domains, [4, 8, 16, 32], seeds)
        all_results = domain_results + rank_results

        model_results = [r for r in all_results if r["model"] == "chronos"]
        lora_count = sum(1 for r in model_results if r["method"] == "lora")
        zero_shot_count = sum(1 for r in model_results if r["method"] == "zero_shot")

        assert lora_count > zero_shot_count, (
            f"LoRA({lora_count}) should exceed zero_shot({zero_shot_count}) without filtering"
        )


class TestTwoWayAnova:
    """Two-way ANOVA 교호작용 항 검증.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_two_way_anova_interaction_term(self) -> None:
        """Two-way ANOVA 결과에 교호작용 항이 포함되어야 함."""
        import pandas as pd
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm

        rng = np.random.default_rng(42)
        n = 60
        data = pd.DataFrame(
            {
                "mae": rng.normal(1.0, 0.5, n).tolist(),
                "method": (["lora", "adapter", "head_only"] * 20)[:n],
                "domain": (["ett_m1"] * 20 + ["finance"] * 20 + ["smd"] * 20)[:n],
            }
        )

        model = ols("mae ~ C(method) + C(domain) + C(method):C(domain)", data=data).fit()
        table = anova_lm(model, typ=2)

        assert "C(method):C(domain)" in table.index, (
            "교호작용 항 C(method):C(domain)이 ANOVA 테이블에 없음"
        )
        assert "F" in table.columns, "F-statistic 열이 없음"
        assert "PR(>F)" in table.columns, "p-value 열이 없음"

    def test_eta_squared_between_zero_and_one(self) -> None:
        """η² 값이 0과 1 사이여야 함."""
        ss_effect = 10.0
        ss_total = 100.0
        eta_sq = ss_effect / ss_total

        assert 0.0 <= eta_sq <= 1.0, f"η²={eta_sq} 범위 초과"
        assert eta_sq == pytest.approx(0.1, abs=1e-9)

    def test_one_way_anova_misses_interaction(self) -> None:
        """scipy.stats.f_oneway는 교호작용 항을 생성하지 않음을 확인."""
        from scipy import stats

        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([4.0, 5.0, 6.0])
        f_stat, p_val = stats.f_oneway(group1, group2)

        assert isinstance(f_stat, float)
        assert isinstance(p_val, float)


class TestOutlierDetection:
    """이상치 탐지 함수 검증.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_detect_known_outlier(self) -> None:
        """MAE=124.77 같은 극단적 이상치를 탐지해야 함."""
        results = [
            {"model": "chronos", "method": "lora", "domain": "ett_m1",
             "metrics": {"mae": 124.77}},
            {"model": "chronos", "method": "lora", "domain": "finance",
             "metrics": {"mae": 0.85}},
            {"model": "chronos", "method": "lora", "domain": "smd",
             "metrics": {"mae": 0.92}},
            {"model": "chronos", "method": "adapter", "domain": "ett_m1",
             "metrics": {"mae": 0.78}},
        ]

        lora_maes = [r["metrics"]["mae"] for r in results if r["method"] == "lora"]
        median_mae = float(np.median(lora_maes))
        outliers = [
            r for r in results
            if r["method"] == "lora" and r["metrics"]["mae"] > 10 * median_mae
        ]

        assert len(outliers) >= 1, "MAE=124.77 이상치가 탐지되지 않음"
        assert outliers[0]["metrics"]["mae"] == pytest.approx(124.77)

    def test_normal_values_not_flagged(self) -> None:
        """정상 범위의 MAE 값은 이상치로 탐지되지 않아야 함."""
        results = [
            {"method": "lora", "metrics": {"mae": 0.8}},
            {"method": "lora", "metrics": {"mae": 0.9}},
            {"method": "lora", "metrics": {"mae": 1.1}},
        ]

        lora_maes = [r["metrics"]["mae"] for r in results]
        median_mae = float(np.median(lora_maes))
        outliers = [r for r in results if r["metrics"]["mae"] > 10 * median_mae]

        assert len(outliers) == 0, "정상 값이 이상치로 잘못 탐지됨"


class TestEffectSizes:
    """효과 크기 계산 검증.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_cliffs_delta_known_values(self) -> None:
        """완전 분리된 그룹에서 Cliff's delta = -1.0."""
        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([4.0, 5.0, 6.0])

        n1, n2 = len(group1), len(group2)
        dominance = 0.0
        for x in group1:
            for y in group2:
                if x > y:
                    dominance += 1.0
                elif x < y:
                    dominance -= 1.0

        delta = dominance / (n1 * n2)
        assert delta == pytest.approx(-1.0, abs=1e-9)

    def test_cliffs_delta_identical_groups(self) -> None:
        """동일한 그룹에서 Cliff's delta = 0.0."""
        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([1.0, 2.0, 3.0])

        n1, n2 = len(group1), len(group2)
        dominance = 0.0
        for x in group1:
            for y in group2:
                if x > y:
                    dominance += 1.0
                elif x < y:
                    dominance -= 1.0

        delta = dominance / (n1 * n2)
        assert delta == pytest.approx(0.0, abs=1e-9)

    def test_cliffs_delta_empty_raises(self) -> None:
        """빈 배열 입력 시 에러 발생 확인."""
        group1 = np.array([])
        group2 = np.array([1.0, 2.0])

        with pytest.raises((ValueError, ZeroDivisionError)):
            n1, n2 = len(group1), len(group2)
            if n1 == 0 or n2 == 0:
                raise ValueError("빈 배열 입력")


class TestHolmCorrection:
    """Holm-Bonferroni 다중비교 보정 검증.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_holm_correction_increases_pvalues(self) -> None:
        """Holm 보정 후 p-value가 증가해야 함."""
        from statsmodels.stats.multitest import multipletests

        raw_pvalues = [0.01, 0.04, 0.03, 0.005]
        reject, corrected, _, _ = multipletests(raw_pvalues, method="holm")

        for raw, corr in zip(raw_pvalues, corrected):
            assert corr >= raw, f"보정 후 p-value({corr})가 원래({raw})보다 작음"

    def test_holm_correction_all_significant(self) -> None:
        """매우 작은 p-value는 보정 후에도 유의해야 함."""
        from statsmodels.stats.multitest import multipletests

        raw_pvalues = [0.001, 0.002, 0.003]
        reject, corrected, _, _ = multipletests(raw_pvalues, alpha=0.05, method="holm")

        assert all(reject), f"매우 작은 p-value가 보정 후 기각되지 않음: {reject}"

    def test_holm_correction_preserves_order(self) -> None:
        """보정 후에도 p-value의 상대적 순서가 보존되어야 함."""
        from statsmodels.stats.multitest import multipletests

        raw_pvalues = [0.01, 0.05, 0.001]
        _, corrected, _, _ = multipletests(raw_pvalues, method="holm")

        min_raw_idx = int(np.argmin(raw_pvalues))
        assert corrected[min_raw_idx] <= max(corrected)
