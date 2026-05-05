from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.pilot_phase1b import LoRALocus, ShiftType, _run_single_experiment


def main() -> None:
    output_dir = Path("results/pilot_1b")
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_backfill_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tasks = [
        ("moirai", LoRALocus.ATTN_ALL, ShiftType.AMPLITUDE, 42),
        ("moirai", LoRALocus.EARLY_LAYERS, ShiftType.SPECTRAL, 123),
        ("moirai", LoRALocus.LATE_LAYERS, ShiftType.SPECTRAL, 123),
        ("moirai", LoRALocus.LATE_LAYERS, ShiftType.SPECTRAL, 42),
    ]

    done = 0
    failed = 0
    print(f"[backfill1b-moirai] start device={device}")
    for model, locus, shift, seed in tasks:
        exp_id = f"{model}_lora_{locus.value}_{shift.value}_seed{seed}"
        final_path = output_dir / f"{exp_id}.json"
        if final_path.exists():
            print(f"[backfill1b-moirai] exists {exp_id}")
            continue
        try:
            print(f"[backfill1b-moirai] run {exp_id}")
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
            tmp_path = tmp_dir / f"{exp_id}.json"
            tmp_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if not final_path.exists():
                tmp_path.replace(final_path)
            done += 1
            print(f"[backfill1b-moirai] done {exp_id}")
        except Exception as exc:
            failed += 1
            print(f"[backfill1b-moirai] fail {exp_id}: {exc}")

    print(f"[backfill1b-moirai] finished done={done} failed={failed}")


if __name__ == "__main__":
    main()
