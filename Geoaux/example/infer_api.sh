#!/bin/bash
#SBATCH -p i64m512u           # Partition: CPU
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -J geoaux_api
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# ---- Parameters ----
MODEL="${MODEL:?Please set MODEL (e.g. qwen-plus)}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# ---- Setup ----
source ~/.bashrc
conda activate vlmeval
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs output

echo "=== Geoaux API Inference ==="
echo "Model: $MODEL"
echo "Workers: $NUM_WORKERS"

python scripts/run_infer.py \
    --model_key api \
    --model_path "$MODEL" \
    --num_workers "$NUM_WORKERS"

echo "Done."
