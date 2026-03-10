#!/bin/bash
#SBATCH -p i64m512u           # Partition: CPU
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -J geoaux_eval
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# ---- Parameters ----
INPUT_FILE="${INPUT_FILE:?Please set INPUT_FILE}"
STEP="${STEP:-both}"         # extract / score / both
USE_JUDGE="${USE_JUDGE:-}"   # set to --use_judge to enable
SMART="${SMART:-}"           # set to --smart to enable

# ---- Setup ----
source ~/.bashrc
conda activate vlmeval
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs eval_results

echo "=== Geoaux Evaluation ==="
echo "Input: $INPUT_FILE"
echo "Step:  $STEP"

python scripts/run_eval.py \
    --input_file "$INPUT_FILE" \
    --step "$STEP" \
    $SMART $USE_JUDGE \
    --num_workers 16

echo "Done."
