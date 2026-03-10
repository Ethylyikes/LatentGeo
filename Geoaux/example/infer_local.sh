#!/bin/bash
#SBATCH -p i64m1tga800u       # Partition: A800
#SBATCH --gres=gpu:4          # GPUs
#SBATCH --cpus-per-task=32    # CPUs
#SBATCH -t 03:00:00           # Time limit
#SBATCH -J geoaux_infer       # Job name
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# ---- Parameters (override via sbatch --export) ----
MODEL_KEY="${MODEL_KEY:?Please set MODEL_KEY (e.g. qwen2.5-vl)}"
MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH (e.g. Qwen/Qwen2.5-VL-7B-Instruct)}"
CONDA_ENV="${CONDA_ENV:-vlmeval}"
WORLD_SIZE="${WORLD_SIZE:-4}"

# ---- Setup ----
source ~/.bashrc
conda activate "$CONDA_ENV"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs output

echo "=== Geoaux Local Inference ==="
echo "Model key:  $MODEL_KEY"
echo "Model path: $MODEL_PATH"
echo "Conda env:  $CONDA_ENV"
echo "World size: $WORLD_SIZE"

python scripts/run_infer.py \
    --model_key "$MODEL_KEY" \
    --model_path "$MODEL_PATH" \
    --world_size "$WORLD_SIZE"

echo "Done."
