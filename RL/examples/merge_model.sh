cd "$(dirname "$0")/.."
conda activate easyr1
CKPT_PATH=latent/RL/run_name/global_step_xxx/actor
python3 -m scripts.model_merger --local_dir=${CKPT_PATH}
