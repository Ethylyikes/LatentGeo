torchrun --nproc_per_node=4 tools/export_fsdp_actor_to_hf.py \
  --local_dir /hpc2hdd/home/yuxuanzhao/zihanwang/latent/RL/checkpoints/easy_r1/latent_rl_run_temp0.1kl0.01/global_step_16/actor \
  --out       /hpc2hdd/home/yuxuanzhao/zihanwang/latent/RL/checkpoints/easy_r1/latent_rl_run_temp0.1kl0.01/global_step_27/infer \
  --dtype fp16 \
  --trust_remote_code
