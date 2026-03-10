
set -euo pipefail
cat >/tmp/tmp_cfg.json <<EOF
{
  "model": {
    "final": {
      "class": "Qwen2VLChat",
      "model_path": "/hpc2hdd/home/yuxuanzhao/zihanwang/latent/RL/checkpoints/easy_r1/latent_rl_run_temp0.1kl0.01/global_step_27/infer",
      "min_pixels": 1003520,
      "max_pixels": 12845056,
      "use_custom_prompt": false,
      "use_latent_prompt": false,
      "latent_slots": 10,
      "latent_debug": true,
      "prefix_question": false,
      "vqa_prompt_suffix": null,
      "post_process": false,
      "log_input_prompt": true,
      "temperature": 0.01,
      "top_p": 0.001,
      "fixed_user_prefix": "\\nYou are a helpful assistant.\\nFirst, analyze the problem with the image and describe all necessary geometric constructions or spatial operations using concise and standard language.\\nThen, provide a reasoning to solve the math problem.\\nFinally, give the final answer. The final answer must be boxed, e.g., \\\\boxed{answer}."
    }
  },
  "data": {
    "MathVerse_MINI": {"class": "MathVerse", "dataset": "MathVerse_MINI"}
  }
}
EOF

# The latent patch is not part of our model's logic; our latent logic is already defined in LatentGeo/RL/transformers-4.56.0.
export LATENT_PATCH_DISABLE=1
export LATENT_START_ID=151666
export LATENT_END_ID=151667
export LATENT_SIZE=10
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CUDA_VISIBLE_DEVICES=0 \
python run.py \
  --config /tmp/tmp_cfg.json \
  --verbose \
  --use-vllm

rm -f /tmp/tmp_cfg.json

