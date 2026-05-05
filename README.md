# TSFM-PEFT-Bench

A cross-architecture benchmark for evaluating Parameter-Efficient Fine-Tuning
(PEFT) recommendation reliability in Time Series Foundation Models (TSFMs).
Companion code and artifacts for the paper "TSFM-PEFT-Bench: A
Cross-Architecture Benchmark for PEFT Selection in Time Series Foundation
Models" (under double-blind review at NeurIPS 2026 Datasets and Benchmarks
Track).

**Quick metadata:**
- **License:** Apache-2.0 (`LICENSE`)
- **Croissant manifest:** `tsfm_peft_bench.croissant.json`
  (MLCommons Croissant 1.0 with mandatory RAI fields)
- **Headline scale:** 882 paper-included primary-model runs
  (3 architectures × 4 domains × 6 main methods + rank/locus sweeps)
- **Reproduction in one command:**
  `python scripts/reproduce_paper_tables.py`
- **Anonymous review URL (review-only):**
  `https://anonymous.4open.science/r/tsfm-peft-bench-anon`

---

## Hosting and accessibility (NeurIPS 2026 D&B Track)

Per the NeurIPS 2026 Evaluations & Datasets hosting policy, this artifact will
be hosted at:

| Asset | Platform | Notes |
|---|---|---|
| Code (frozen at submission) | Anonymous-4-Open-Science (review) → Hugging Face Spaces (camera-ready) | All scripts, configs, src/ |
| Run manifest + headline numbers | Hugging Face Datasets `tsfm-peft-bench/runs` | `paper_manifest.json`, `paper_numbers.{json,tex}`, `selector_evaluation.json` |
| Per-run JSON files (full grid) | Hugging Face Datasets `tsfm-peft-bench/runs` | `results/expansion/{domain,rank,locus}/*.json` (~50–200 MB total) |
| Domain shift profiles | Hugging Face Datasets `tsfm-peft-bench/runs` | `domain_shift_profiles.json` |
| Croissant metadata | top-level `tsfm_peft_bench.croissant.json` | Auto-validated against MLCommons spec 1.0 |

The Croissant file is the canonical machine-readable description of the
benchmark. RAI fields (limitations, biases, sensitive information, intended
use, social impact, sources, preprocessing, release plan) are populated.

---

## What's in here

- `src/` — model wrappers (Chronos, MOMENT, Moirai, TimesFM), PEFT
  adaptations (LoRA / DoRA / IA³ / Adapter / Prefix / Head-only / Full-FT),
  dataset loaders, and evaluation utilities.
- `scripts/` — Hydra-based single-run trainer (`train.py`), full benchmark
  driver (`run_expansion.py`), analysis pipeline (`analyze_expansion_v2.py`,
  `reproduce_paper_tables.py`), selector (`build_selector.py`), and
  mechanism probes (`subspace_probe.py`, `gradient_probe.py`).
- `configs/` — Hydra YAML configs for models, adaptations, and data; zero
  hard-coded hyperparameters.
- `tests/` — pytest suite (100 tests) covering data loaders, adaptations,
  metrics, shift profiles, and analysis utilities.
- `results/` — paper artifacts (manifest, ANOVA, selector tables, paper
  numbers). `paper_manifest.json` is the single source of truth for
  paper-included runs.
- `paper_submission.tex`, `paper_appendix.tex`, `paper_supplementary.tex`,
  `neurips_checklist.tex` — the manuscript and supplementary materials.

---

## Setup

The repository targets **Python 3.10–3.12**.

```bash
# Recommended (exact reproduction): pin every transitive dep
python -m pip install -r requirements-lock.txt
python -m pip install -e .

# Lighter (looser bounds): top-level constraints only
python -m pip install -e ".[dev]"
```

`requirements-lock.txt` is captured by `pip freeze` from the environment that
produced the released results (PyTorch 2.11, chronos-forecasting 2.2.2,
uni2ts 2.0.0, momentfm @ upstream commit `38f7310a`).

TimesFM is optional and conflicts with Python 3.12 (paxml/lingvo
dependencies). Install only on Python 3.10/3.11:

```bash
python -m pip install -e ".[timesfm]"
```

GPU: experiments were run on 4× RTX 3090 (cluster) and 1–4× RTX 3060 nodes.
Mixed-precision (FP16/BF16) is always enabled; FP32 is unsupported.

---

## Reproducing the paper

### Headline tables

```bash
# Re-derive paper_manifest.json + paper_numbers.{json,tex} from raw runs
python scripts/reproduce_paper_tables.py \
    --input_dir results/expansion --output_dir results

# ANOVA / outlier filtering / per-architecture statistics (v2)
python scripts/analyze_expansion_v2.py

# Selector evaluation (LOOCV, 12 held-out cells)
python scripts/build_selector.py
```

These three commands regenerate every numerical claim in
`paper_submission.tex` from raw run JSON. Latex macros for the headline
numbers are emitted to `results/paper_numbers.tex`.

### Single-run training

```bash
python scripts/train.py model=chronos adaptation=lora data=ett_m1
python scripts/train.py model=chronos adaptation=lora data=ett_h1 \
    adaptation.rank=16 training.lr=1e-4 training.epochs=50
```

### Full benchmark grid

```bash
# 3 primary models × 4 domains × 7 main methods × 5 seeds = 882 paper-included runs
python scripts/run_expansion.py \
    --models chronos,moment,moirai \
    --mode domain \
    --seeds 42,123,7,2024,3407 \
    --save_checkpoints \
    --checkpoint_dir checkpoints/expansion
```

`scripts/run_benchmark.sh` wraps the full sweep across the three modes
(domain / rank / locus).

### Evaluation from a checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/expansion/chronos/<experiment_id>.pt \
    --model chronos --data ett_m1
```

Both `train.py` and `run_expansion.py` save checkpoints in a unified schema
(`backbone_state_dict` + `adaptation_method` + `adaptation_config` +
`prediction_length` / `context_length`). `evaluate.py` accepts either
`backbone_state_dict` or the legacy `state_dict` key for backward
compatibility with pre-2026-04-27 checkpoints.

---

## Repository conventions

- **Docstrings and error messages are written in Korean**; identifiers and
  paper-facing artifacts are in English. See `CLAUDE.md` for full coding
  standards.
- All hyperparameters live in `configs/`; do not hard-code values in scripts.
- External libraries (chronos, peft, transformers, wandb) are imported
  dynamically via `importlib` and accessed through `Protocol` types in
  `src/`. Direct imports in `scripts/` and `tests/` are fine.
- `from __future__ import annotations` at the top of every module.

---

## Data and checkpoints

`data/` is `.gitignore`d. Public datasets used:

- **ETTm1**: standard ETT benchmark (Zhou et al., 2021).
- **Exchange-rate ("Finance")**: Lai et al., 2017.
- **SMD**: Server Machine Dataset (Su et al., 2019), entity boundaries
  preserved during splitting (`src/data/smd.py`).
- **PhysioNet**: subset processed in `src/data/physionet.py`, subject IDs
  preserved across splits to prevent leakage.

`checkpoints/` is a local symlink to NAS storage on the maintainer's
machine. Downstream users should either remove the symlink and create a
local directory, or override the path with `checkpoint_dir=...` /
`--checkpoint_dir <path>` on the command line.

---

## Layout

```
src/
├── adaptation/    # LoRA / DoRA / IA3 / Adapter / Prefix / Head / Full
├── data/          # ETT, finance, SMD, PhysioNet, shift metrics
├── evaluation/    # MAE / MSE / MASE / CRPS / CKA
├── models/        # Chronos / MOMENT / Moirai / TimesFM wrappers
└── utils/         # Seeds, device, logging
scripts/
├── train.py                       # single-run Hydra entry point
├── run_expansion.py               # full benchmark grid
├── reproduce_paper_tables.py      # SoT manifest + paper_numbers regenerator
├── analyze_expansion_v2.py        # ANOVA / outlier policy
├── build_selector.py              # selector LOOCV evaluation
├── subspace_probe.py              # mechanism probe (representation)
└── gradient_probe.py              # mechanism probe (gradient flow)
configs/
├── model/        chronos.yaml | moment.yaml | moirai.yaml | timesfm.yaml
├── adaptation/   lora | dora | ia3 | adapter | prefix | head_only | full_ft
└── data/         ett_m1 | ett_h1 | finance | smd | physionet | ...
results/
├── paper_manifest.json            # SoT: 882 paper-included primary-model runs
├── paper_numbers.{json,tex}       # auto-generated latex macros
├── selector_evaluation.json       # selector LOOCV results
├── expansion_analysis_canonical/  # canonical ANOVA outputs
└── expansion_analysis_v3/         # v3 (current) analysis outputs
```

---

## Quality gates

```bash
# Tests (100 tests, ~3s on CPU)
pytest tests/ -v

# Lint / format
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
isort --check-only src/ scripts/ tests/
mypy src/
```

---

## Citation

Anonymous under review. Citation will be added on publication.

## License

Apache 2.0 (see `LICENSE` once added).
