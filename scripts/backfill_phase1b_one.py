from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.pilot_phase1b import LoRALocus, ShiftType, _run_single_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Phase1B missing experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--locus", type=str, required=True)
    parser.add_argument("--shift", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_backfill_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    locus = LoRALocus(args.locus)
    shift = ShiftType(args.shift)
    exp_id = f"{args.model}_lora_{locus.value}_{shift.value}_seed{args.seed}"
    final_path = output_dir / f"{exp_id}.json"
    if final_path.exists():
        print(f"[backfill1b-one] exists {exp_id}")
        return

    print(f"[backfill1b-one] run {exp_id} on {device}")
    result = _run_single_experiment(
        model_name=args.model,
        locus=locus,
        shift_type=shift,
        seed=args.seed,
        data_path=args.data_path,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        lora_rank=args.lora_rank,
        max_eval_batches=args.max_eval_batches,
    )
    tmp_path = tmp_dir / f"{exp_id}.json"
    tmp_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not final_path.exists():
        tmp_path.replace(final_path)
    print(f"[backfill1b-one] done {exp_id}")


if __name__ == "__main__":
    main()
