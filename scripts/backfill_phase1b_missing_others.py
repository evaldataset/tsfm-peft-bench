from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.pilot_phase1b import LoRALocus, ShiftType, _run_single_experiment


def main() -> None:
    out = Path("results/pilot_1b")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tasks = [
        ("chronos", LoRALocus.EARLY_LAYERS, ShiftType.SPECTRAL, 123),
        ("timesfm", LoRALocus.ATTN_ALL, ShiftType.SPECTRAL, 123),
        ("timesfm", LoRALocus.FFN, ShiftType.SPECTRAL, 42),
    ]

    done = 0
    failed = 0
    print(f"[backfill] start device={device}")
    for model, locus, shift, seed in tasks:
        exp_id = f"{model}_lora_{locus.value}_{shift.value}_seed{seed}"
        result_file = out / f"{exp_id}.json"
        if result_file.exists():
            print(f"[backfill] exists {exp_id}")
            continue

        try:
            print(f"[backfill] run {exp_id}")
            result = _run_single_experiment(
                model_name=model,
                locus=locus,
                shift_type=shift,
                seed=seed,
                data_path="data/ETT-small/ETTm1.csv",
                device=device,
                epochs=10,
                batch_size=32,
                lr=1e-4,
                patience=5,
                lora_rank=8,
                max_eval_batches=0,
            )
            result_file.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            done += 1
            print(f"[backfill] done {exp_id}")
        except Exception as exc:
            failed += 1
            print(f"[backfill] fail {exp_id}: {exc}")

    print(f"[backfill] finished done={done} failed={failed}")


if __name__ == "__main__":
    main()
