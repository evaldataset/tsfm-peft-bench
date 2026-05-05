from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.pilot_phase1a import (
    ShiftSeverity,
    ShiftType,
    _run_single_experiment,
)


def main() -> None:
    out = Path("results/pilot_1a")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tasks = [
        ("chronos", "lora", ShiftType.IRREGULARITY, ShiftSeverity.MILD, 123),
        ("timesfm", "adapter", ShiftType.AMPLITUDE, ShiftSeverity.STRONG, 123),
    ]

    print(f"[1a-backfill] start device={device}")
    for model, method, shift, severity, seed in tasks:
        exp_id = f"{model}_{method}_{shift.value}_{severity.value}_seed{seed}"
        result_file = out / f"{exp_id}.json"
        try:
            print(f"[1a-backfill] run {exp_id}")
            result = _run_single_experiment(
                model_name=model,
                method=method,
                shift_type=shift,
                severity=severity,
                seed=seed,
                data_path="data/ETT-small/ETTm1.csv",
                device=device,
                epochs=10,
                batch_size=32,
                lr=1e-4,
                patience=5,
                max_eval_batches=0,
                stride=0,
            )
            result_file.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[1a-backfill] done {exp_id}")
        except Exception as exc:
            print(f"[1a-backfill] fail {exp_id}: {exc}")


if __name__ == "__main__":
    main()
