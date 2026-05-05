from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.pilot_phase1b import LoRALocus, ShiftType, _run_single_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing Phase 1B results")
    parser.add_argument("--models", type=str, default="chronos,moment,moirai,timesfm")
    parser.add_argument("--output_dir", type=str, default="results/pilot_1b")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="data/ETT-small/ETTm1.csv")
    parser.add_argument("--max_eval_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_backfill_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    shifts = [ShiftType.AMPLITUDE, ShiftType.SPECTRAL]
    loci = [
        LoRALocus.ATTN_QV,
        LoRALocus.ATTN_ALL,
        LoRALocus.FFN,
        LoRALocus.ATTN_QV_FFN,
        LoRALocus.EARLY_LAYERS,
        LoRALocus.LATE_LAYERS,
    ]
    seeds = [42, 123]

    tasks: list[tuple[str, LoRALocus, ShiftType, int, str]] = []
    for model in selected_models:
        for shift in shifts:
            for locus in loci:
                for seed in seeds:
                    exp_id = f"{model}_lora_{locus.value}_{shift.value}_seed{seed}"
                    file_name = f"{exp_id}.json"
                    if not (output_dir / file_name).exists():
                        tasks.append((model, locus, shift, seed, file_name))

    print(
        f"[backfill1b] start models={selected_models} device={device} missing={len(tasks)}"
    )

    done = 0
    failed = 0
    for model, locus, shift, seed, file_name in tasks:
        final_path = output_dir / file_name
        if final_path.exists():
            continue
        exp_id = file_name.removesuffix(".json")
        try:
            print(f"[backfill1b] run {exp_id}")
            result = _run_single_experiment(
                model_name=model,
                locus=locus,
                shift_type=shift,
                seed=seed,
                data_path=args.data_path,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                lora_rank=args.lora_rank,
                max_eval_batches=args.max_eval_batches,
            )
            tmp_path = tmp_dir / file_name
            tmp_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if not final_path.exists():
                tmp_path.replace(final_path)
            done += 1
            print(f"[backfill1b] done {exp_id}")
        except Exception as exc:
            failed += 1
            print(f"[backfill1b] fail {exp_id}: {exc}")

    print(f"[backfill1b] finished done={done} failed={failed}")


if __name__ == "__main__":
    main()
