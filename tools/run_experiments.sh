#!/bin/bash
# =============================================================================
# Yellow River PINN - Multi-Model Training Experiments
# =============================================================================
# This script runs three model variants for comparative analysis:
#   1. PINN_SOTA: Full SOTA implementation with L-BFGS fine-tuning
#   2. Baseline: Pure data-driven (no physics constraints)
#   3. Sensitive_n: Testing Manning roughness sensitivity (n=0.05)
#
# Usage: chmod +x run_experiments.sh && ./run_experiments.sh
# =============================================================================

set -e  # Exit on error

# Navigate to project root (script is in tools/)
cd "$(dirname "$0")/.."

echo "========================================================================"
echo "Yellow River PINN - Multi-Model Training Pipeline"
echo "========================================================================"
echo ""

# Create timestamp for this batch of experiments
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Experiment Batch ID: $TIMESTAMP"
echo ""

# Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[Info] Activating virtual environment..."
    source .venv/bin/activate
fi

PYTHON=".venv/bin/python"

# =============================================================================
# Experiment 1: PINN_SOTA (State-of-the-Art)
# =============================================================================
echo "========================================================================"
echo "[Experiment 1/3] PINN_SOTA - Full Physics + L-BFGS Fine-tuning"
echo "========================================================================"
$PYTHON scripts/training/train_v2.py \
    --epochs 200 \
    --lbfgs-epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --hidden-dim 128 \
    --num-res-blocks 4 \
    --use-attention True \
    --g 9.8 \
    --n 0.03 \
    --S0 0.001 \
    --B 50.0 \
    --max-grad-norm 1.0 \
    --save-dir "models/pinn/PINN_SOTA_${TIMESTAMP}" \
    --early-stop 30 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "[✓] PINN_SOTA training completed successfully"
else
    echo "[✗] PINN_SOTA training failed"
    exit 1
fi

echo ""
echo ""

# =============================================================================
# Experiment 2: Baseline (Pure Data-Driven)
# =============================================================================
echo "========================================================================"
echo "[Experiment 2/3] Baseline - Pure Data-Driven (No Physics)"
echo "========================================================================"
echo "[Note] Using v1.0 architecture without physics loss for pure comparison"
$PYTHON scripts/training/train.py \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.001 \
    --hidden-dim 64 \
    --lambda-phys 0.0 \
    --save-dir "models/baseline/Baseline_${TIMESTAMP}" \
    --early-stop 30 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "[✓] Baseline training completed successfully"
else
    echo "[✗] Baseline training failed"
    exit 1
fi

echo ""
echo ""

# =============================================================================
# Experiment 3: Sensitive_n (Manning Roughness Sensitivity)
# =============================================================================
echo "========================================================================"
echo "[Experiment 3/3] Sensitive_n - Manning n=0.05 (Friction Sensitivity)"
echo "========================================================================"
$PYTHON scripts/training/train_v2.py \
    --epochs 200 \
    --lbfgs-epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --hidden-dim 128 \
    --num-res-blocks 4 \
    --use-attention True \
    --g 9.8 \
    --n 0.05 \
    --S0 0.001 \
    --B 50.0 \
    --max-grad-norm 1.0 \
    --save-dir "models/pinn/Sensitive_n_${TIMESTAMP}" \
    --early-stop 30 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "[✓] Sensitive_n training completed successfully"
else
    echo "[✗] Sensitive_n training failed"
    exit 1
fi

echo ""
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "========================================================================"
echo "All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved in:"
echo "  - models/pinn/PINN_SOTA_${TIMESTAMP}/"
echo "  - models/baseline/Baseline_${TIMESTAMP}/"
echo "  - models/pinn/Sensitive_n_${TIMESTAMP}/"
echo ""
echo "Next steps:"
echo "  1. Run consensus verification:"
echo "     python scripts/verification/verify_consensus.py --model-dir models/pinn/"
echo "  2. Run sensitivity analysis:"
echo "     python scripts/training/train_v2.py --sensitivity-analysis --S0-range 0.0005,0.002"
echo ""
echo "========================================================================"
