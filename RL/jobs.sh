export OPENAI_API_KEY=
export OPENAI_BASE=

# The latent patch is not part of our model's logic; our latent logic is already defined in LatentGeo/RL/transformers-4.56.0.
export DISABLE_LATENT_MASK=0
export LATENT_LOGIT_BIAS=10
export LATENT_LOGIT_BIAS_DECAY=1.0
export LATENT_LOGIT_BIAS_MIN=0
export LATENT_REWARD_EMA=0.9  

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RL_ROOT="${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export LATENT_PATCH_DISABLE=1

export LATENT_PATCH_DISABLE="${LATENT_PATCH_DISABLE:-1}"
export LATENT_SIZE="${LATENT_SIZE:-10}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE="${OPENAI_BASE:-${OPENAI_API_BASE:-}}"
export LATENT_IMG_ROOT="${LATENT_IMG_ROOT:-/hpc2hdd/home/yuxuanzhao/Xuhaiying}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-gdpo}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"
export RAY_DISABLE_DASHBOARD="${RAY_DISABLE_DASHBOARD:-1}"
export RAY_DASHBOARD_ENABLED="${RAY_DASHBOARD_ENABLED:-0}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export RAY_NUM_GPUS="${RAY_NUM_GPUS:-4}"
export USE_RAY_LOCAL="${USE_RAY_LOCAL:-1}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-0}"
export RAY_LOG_TO_STDERR="${RAY_LOG_TO_STDERR:-0}"
export RAY_LOCAL_MODE="${RAY_LOCAL_MODE:-0}"
export RAY_task_exit_on_oom="${RAY_task_exit_on_oom:-1}"
export RAY_SPILL_DIR="${RAY_SPILL_DIR:-/tmp/ray_spill}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_tmp}"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-134217728}"
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS="${RAY_WORKER_REGISTER_TIMEOUT_SECONDS:-300}"
unset LD_PRELOAD
unset NCCL_TOPO_FILE
mkdir -p "${RAY_SPILL_DIR}" "${RAY_TMPDIR}"

MODEL_PATH="${MODEL_PATH:-}"
TRAIN_FILE="${TRAIN_FILE:-}"
VAL_FILE="${VAL_FILE:-}"

if [[ "${LATENT_PATCH_DISABLE}" != "1" ]]; then
  export PYTHONPATH="${RL_ROOT}:${REPO_ROOT}/transformers-4.56.0/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="${RL_ROOT}:${PYTHONPATH:-}"
fi
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"

cd "${RL_ROOT}"

run_job() {
  local exp_suffix="$1"; shift
  local exp_name="${EXPERIMENT_NAME:-latent_rl_run}_${exp_suffix}"
  python -m verl.trainer.main \
      config=examples/config.yaml \
      ${TRAIN_FILE:+data.train_files="${TRAIN_FILE}"} \
      ${VAL_FILE:+data.val_files="${VAL_FILE}"} \
      ${MODEL_PATH:+worker.actor.model.model_path="${MODEL_PATH}"} \
      algorithm.adv_estimator="${ADV_ESTIMATOR}" \
      trainer.experiment_name="${exp_name}" \
      trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-4}" \
      "$@"
}
run_job "final" worker.rollout.temperature=0.9 algorithm.kl_coef=0.03

