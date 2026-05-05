#!/bin/bash
# 메커니즘 분석: checkpoint 생성 → CKA probe → paper figure
# 모든 domain mode 실험 완료 후 실행
set -e
export PYTHONPATH=.

echo "=== Phase 3A: Mechanism checkpoint 생성 ==="
# 모델별 success/failure 셀 (key methods만) — seed 42, 123
for MODEL in chronos moment moirai; do
    echo "[$MODEL] checkpoint 생성 중..."
    .venv/bin/python scripts/run_expansion.py --mode domain --models "$MODEL" \
      --domains ett_m1,smd,finance,physionet --seeds 42,123 --gpu 0 \
      --force_rerun --save_checkpoints --checkpoint_dir checkpoints/mechanism \
      --methods head_only,lora,adapter --skip_unavailable_models \
      2>&1 | tail -5
done

echo "=== Phase 3B: CKA Subspace Probe ==="
.venv/bin/python scripts/subspace_probe.py \
  --checkpoint_dir checkpoints/mechanism \
  --output_dir results/mechanism_analysis \
  --kernel linear

echo "=== Phase 3C: Paper figure 생성 ==="
.venv/bin/python scripts/subspace_probe.py --paper-figure \
  results/mechanism_analysis/cka_results.json \
  mechanism_cka_heatmap.pdf

echo "=== Phase 3D: 분석 재실행 ==="
.venv/bin/python scripts/analyze_expansion_v2.py --mode all \
  --input_dir results/expansion --output_dir results/expansion_analysis_v3

echo "=== Phase 3E: LODO 정책 업데이트 ==="
.venv/bin/python scripts/build_shift_policy.py \
  --domain_results_dir results/expansion/domain \
  --shift_profile_path results/expansion_analysis_v2/domain_shift_profiles.json \
  --output_dir results/pivot_analysis_v2

echo "=== Phase 3F: Landscape heatmap 업데이트 ==="
.venv/bin/python scripts/generate_paper_figures.py \
  --results_dir results/expansion/domain --output_dir results/expansion_analysis

echo "=== 모든 메커니즘 분석 완료! ==="
echo "paper_submission.tex 컴파일:"
echo "  pdflatex paper_submission.tex && bibtex paper_submission && pdflatex paper_submission.tex && pdflatex paper_submission.tex"
