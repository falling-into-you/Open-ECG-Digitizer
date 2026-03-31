#!/bin/bash
set -e

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate open_dig

echo "=========================================="
echo "  ECG Digitization: Baseline vs PMcardio"
echo "=========================================="

# Step 1: Run baseline
echo ""
echo "[Step 1/3] Running baseline inference..."
python3 -m src.digitize \
    --config src/config/inference_wrapper.yml \
    DATA.output_path=sandbox/inference_output_baseline

# Step 2: Run PMcardio-improved version
echo ""
echo "[Step 2/3] Running PMcardio-improved inference..."
python3 -m src.digitize_pmcardio \
    --config src/config/inference_wrapper_pmcardio.yml

# Step 3: Compare results
echo ""
echo "[Step 3/3] Comparing results..."
python3 -m src.metrics.comparison \
    --baseline sandbox/inference_output_baseline \
    --improved sandbox/inference_output_pmcardio \
    --output sandbox/comparison_results

echo ""
echo "=========================================="
echo "  Done! Check results:"
echo "  - Baseline PNG:    sandbox/inference_output_baseline/"
echo "  - PMcardio PNG:    sandbox/inference_output_pmcardio/"
echo "  - Comparison:      sandbox/comparison_results/"
echo "=========================================="
