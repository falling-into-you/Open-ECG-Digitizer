#!/bin/bash
set -e

# Activate conda environment (try multiple methods)
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate open_dig
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/root/miniconda3/etc/profile.d/conda.sh"
    conda activate open_dig
else
    echo "[INFO] conda not found, assuming environment already active"
fi

echo "=========================================="
echo "  ECG Digitization: Evaluate vs GT"
echo "=========================================="

# Step 1: Run baseline (skip if output already exists)
if [ -d "sandbox/inference_output_baseline" ] && [ "$(ls sandbox/inference_output_baseline/*_timeseries_canonical.csv 2>/dev/null | wc -l)" -gt 0 ]; then
    echo ""
    echo "[Step 1/3] Baseline output exists, skipping..."
else
    echo ""
    echo "[Step 1/3] Running baseline inference..."
    python3 -m src.digitize \
        --config src/config/inference_wrapper.yml \
        DATA.output_path=sandbox/inference_output_baseline
fi

# Step 2: Run improved version
echo ""
echo "[Step 2/3] Running improved inference..."
python3 -m src.digitize_pmcardio \
    --config src/config/inference_wrapper_pmcardio.yml

# Step 3: Evaluate both against ground truth
echo ""
echo "[Step 3/3] Evaluating against ground truth..."
python3 -m src.metrics.evaluate_vs_gt \
    --gt ecg_timeseries \
    --baseline sandbox/inference_output_baseline \
    --improved sandbox/inference_output_pmcardio \
    --images ecg_images \
    --output sandbox/evaluation_results

echo ""
echo "=========================================="
echo "  Done! Check results:"
echo "  - Baseline output:   sandbox/inference_output_baseline/"
echo "  - Improved output:   sandbox/inference_output_pmcardio/"
echo "  - Evaluation:        sandbox/evaluation_results/"
echo "=========================================="
