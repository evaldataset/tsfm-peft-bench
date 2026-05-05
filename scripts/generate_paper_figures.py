from __future__ import annotations

"""논문용 주요 figure 생성 스크립트.

Figure 1: 전체 adaptation landscape 요약 (model × domain × method heatmap)
Figure 5: Prescriptive decision flowchart (tikz 기반, LaTeX 코드 생성)

사용법:
    PYTHONPATH=. python scripts/generate_paper_figures.py \
        --results_dir results/expansion/domain \
        --output_dir results/expansion_analysis
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """로깅 초기화.

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


def _load_domain_results(results_dir: Path) -> list[dict[str, object]]:
    """도메인 모드 결과 JSON 로드.

    Args:
        results_dir: 결과 디렉토리 경로.

    Returns:
        결과 딕셔너리 리스트.

    Raises:
        FileNotFoundError: 디렉토리가 없을 때.
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"결과 디렉토리가 없습니다: {results_dir}")

    results: list[dict[str, object]] = []
    for json_path in sorted(results_dir.glob("*.json")):
        if json_path.name.startswith("all_results"):
            continue
        with json_path.open("r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def generate_landscape_heatmap(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Figure 1: 전체 adaptation landscape heatmap.

    Args:
        results: 도메인 모드 결과 리스트.
        output_path: 출력 경로.

    Returns:
        None.

    Raises:
        None.
    """
    # model × domain별로 best method 및 MAE 집계
    from collections import defaultdict

    # (model, domain, method) → list of MAE
    cell_mae: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in results:
        model = str(r.get("model", ""))
        domain = str(r.get("domain", ""))
        method = str(r.get("method", ""))
        metrics = r.get("metrics", {})
        if isinstance(metrics, dict):
            mae_val = metrics.get("mae")
            if mae_val is not None:
                cell_mae[(model, domain, method)].append(float(mae_val))

    models = sorted({k[0] for k in cell_mae.keys()})
    domains = sorted({k[1] for k in cell_mae.keys()})
    methods = sorted({k[2] for k in cell_mae.keys()})

    if not models or not domains or not methods:
        logger.warning("데이터가 부족하여 heatmap을 생성할 수 없습니다.")
        return

    # 각 (model, domain)에서 best method 결정
    best_method_grid: dict[tuple[str, str], str] = {}
    best_mae_grid: dict[tuple[str, str], float] = {}

    for model in models:
        for domain in domains:
            best_method = ""
            best_mae = float("inf")
            for method in methods:
                key = (model, domain, method)
                if key in cell_mae:
                    mean_mae = float(np.mean(cell_mae[key]))
                    if mean_mae < best_mae:
                        best_mae = mean_mae
                        best_method = method
            best_method_grid[(model, domain)] = best_method
            best_mae_grid[(model, domain)] = best_mae

    # 시각화: model (row) × domain (col), 색상 = method
    method_colors = {
        m: plt.cm.Set2(i / max(1, len(methods) - 1))  # type: ignore[attr-defined]
        for i, m in enumerate(methods)
    }

    fig, ax = plt.subplots(figsize=(2.5 * len(domains) + 2, 1.5 * len(models) + 1))

    method_to_idx = {m: i for i, m in enumerate(methods)}
    grid = np.zeros((len(models), len(domains)))

    for i, model in enumerate(models):
        for j, domain in enumerate(domains):
            bm = best_method_grid.get((model, domain), "")
            grid[i, j] = method_to_idx.get(bm, 0)
            mae_val = best_mae_grid.get((model, domain), 0.0)
            short_method = bm.replace("full_fine_tuning", "Full-FT").replace(
                "head_only", "Head"
            ).replace("zero_shot", "Zero").replace("_", " ").title()
            ax.text(
                j, i, f"{short_method}\n({mae_val:.3f})",
                ha="center", va="center", fontsize=8, fontweight="bold",
            )

    im = ax.imshow(
        grid, cmap="Set2", aspect="auto",
        vmin=0, vmax=max(1, len(methods) - 1),
    )
    ax.set_xticks(range(len(domains)))
    domain_labels = [d.replace("ett_m1", "ETTm1").replace("finance", "Finance").replace(
        "smd", "SMD").replace("physionet", "PhysioNet") for d in domains]
    ax.set_xticklabels(domain_labels, fontsize=11)
    ax.set_yticks(range(len(models)))
    model_labels = [m.replace("chronos", "Chronos").replace("moment", "MOMENT").replace(
        "moirai", "Moirai").replace("timesfm", "TimesFM") for m in models]
    ax.set_yticklabels(model_labels, fontsize=11)
    ax.set_xlabel("Target Domain", fontsize=12)
    ax.set_ylabel("Architecture", fontsize=12)
    ax.set_title(
        "Best PEFT method varies by architecture--domain pair",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Landscape heatmap 저장: %s", output_path)


def generate_flowchart_tikz(output_path: Path) -> None:
    """Figure 5: Prescriptive decision flowchart (TikZ LaTeX 코드).

    Args:
        output_path: 출력 .tex 파일 경로.

    Returns:
        None.

    Raises:
        None.
    """
    tikz_code = r"""\begin{figure}[t]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[
  node distance=1.2cm and 2cm,
  startstop/.style={rectangle, rounded corners, minimum width=3cm, minimum height=0.8cm, text centered, draw=black, fill=blue!10, font=\small},
  process/.style={rectangle, minimum width=3cm, minimum height=0.8cm, text centered, draw=black, fill=orange!10, font=\small},
  decision/.style={diamond, minimum width=2.5cm, minimum height=0.8cm, text centered, draw=black, fill=green!10, font=\small, aspect=2.5},
  arrow/.style={thick,->,>=stealth}
]
\node (start) [startstop] {New target domain $\mathcal{D}$};
\node (zs) [process, below=of start] {Evaluate zero-shot \& head-only};
\node (dec1) [decision, below=of zs] {MAE acceptable?};
\node (done1) [startstop, right=of dec1] {Deploy frozen/head-only};
\node (sweep) [process, below=of dec1] {Sweep: LoRA (r=8, early\_layers), Adapter};
\node (dec2) [decision, below=of sweep] {PEFT $<$ baseline?};
\node (lora) [process, below left=of dec2] {Tune rank \& locus};
\node (done2) [startstop, below right=of dec2] {Use best baseline};
\node (cka) [decision, below=of lora] {CKA $> 0.5$?};
\node (deploy) [startstop, below left=of cka] {Deploy PEFT};
\node (abort) [startstop, below right=of cka] {Abort: diffuse drift};

\draw [arrow] (start) -- (zs);
\draw [arrow] (zs) -- (dec1);
\draw [arrow] (dec1) -- node[anchor=south] {yes} (done1);
\draw [arrow] (dec1) -- node[anchor=east] {no} (sweep);
\draw [arrow] (sweep) -- (dec2);
\draw [arrow] (dec2) -- node[anchor=east] {yes} (lora);
\draw [arrow] (dec2) -- node[anchor=west] {no} (done2);
\draw [arrow] (lora) -- (cka);
\draw [arrow] (cka) -- node[anchor=east] {yes} (deploy);
\draw [arrow] (cka) -- node[anchor=west] {no} (abort);
\end{tikzpicture}%
}
\caption{Recommended TSFM adaptation protocol. Start with free baselines, then sweep PEFT methods with a small validation budget. Monitor CKA to detect representation drift before deployment.}
\label{fig:flowchart}
\end{figure}"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(tikz_code)
    logger.info("Flowchart TikZ 저장: %s", output_path)


def main() -> None:
    """메인 함수.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    _setup_logging()

    parser = argparse.ArgumentParser(description="논문용 figure 생성")
    parser.add_argument(
        "--results_dir", type=str, default="results/expansion/domain",
        help="도메인 모드 결과 디렉토리",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/expansion_analysis",
        help="출력 디렉토리",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_domain_results(results_dir)
    logger.info("결과 로드 완료: %d 건", len(results))

    generate_landscape_heatmap(
        results=results,
        output_path=output_dir / "landscape_heatmap.pdf",
    )

    generate_flowchart_tikz(
        output_path=output_dir / "flowchart.tex",
    )


if __name__ == "__main__":
    main()
