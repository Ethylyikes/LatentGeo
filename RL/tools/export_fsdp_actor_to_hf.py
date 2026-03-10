"""
Merge FSDP-sharded actor checkpoints into a single HuggingFace-style folder.

Usage (run with the same world_size as training, e.g., 4):
  torchrun --nproc_per_node=4 tools/export_fsdp_actor_to_hf.py \
    --ckpt checkpoints/easy_r1/qwen2_5_7b_math_grpo/global_step_19/actor \
    --out  checkpoints/easy_r1/qwen2_5_7b_math_grpo/global_step_19/merged_actor \
    --dtype bf16  # or fp16/fp32

Aliases: --ckpt == --local_dir; --out == --hf_upload_path

The script:
  - loads FSDP shards (model_world_size_*_rank_*.pt) from --ckpt
  - reconstructs full weights
  - saves a standard HuggingFace folder at --out
Tokenizer/processor/config are copied from the saved "huggingface" subfolder in the checkpoint.
"""

import argparse
import os
import shutil
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardedStateDictConfig
from torch.distributed.checkpoint.state_dict import set_state_dict, StateDictOptions
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, AutoConfig

# Ensure repo root is on sys.path for utils imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from verl.utils.fsdp_utils import get_fsdp_wrap_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        "--local_dir",
        dest="ckpt",
        type=str,
        required=True,
        help="Path to actor checkpoint folder (contains model_world_size_*_rank_*.pt); alias: --local_dir",
    )
    parser.add_argument(
        "--out",
        "--hf_upload_path",
        dest="out",
        type=str,
        required=True,
        help="Output directory for merged HuggingFace weights; alias kept for backward compatibility: --hf_upload_path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Load dtype for the merge process",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass through to AutoModelForVision2Seq for custom architectures",
    )
    return parser.parse_args()


def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def build_fsdp_model(base_model: torch.nn.Module, dtype: torch.dtype, device: torch.device, world_size: int):
    # Recreate the training-time FSDP wrapping so checkpoint keys match.
    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    auto_wrap_policy = get_fsdp_wrap_policy(base_model)
    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    sharding_strategy = ShardingStrategy.FULL_SHARD
    return FSDP(
        base_model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        cpu_offload=None,
        forward_prefetch=False,
        use_orig_params=False,
        device_mesh=device_mesh,
        device_id=device,
    )


def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    hf_src = os.path.join(args.ckpt, "huggingface")
    if not os.path.isdir(hf_src):
        if rank == 0:
            print(f"[merge] Cannot find huggingface folder at {hf_src}.")
        dist.barrier()
        dist.destroy_process_group()
        sys.exit(1)

    dtype = get_dtype(args.dtype)
    if rank == 0:
        print(f"[merge] Loading config from {hf_src} with dtype={dtype}")

    cfg = AutoConfig.from_pretrained(hf_src, trust_remote_code=args.trust_remote_code)
    # from_config 在老版本 transformers 里不接受 trust_remote_code；做一次兼容处理
    try:
        base_model = AutoModelForVision2Seq.from_config(cfg, trust_remote_code=args.trust_remote_code)
    except TypeError:
        if args.trust_remote_code:
            setattr(cfg, "trust_remote_code", True)
        base_model = AutoModelForVision2Seq.from_config(cfg)
    base_model = base_model.to(dtype=dtype, device=device)

    # Wrap with FSDP using the same policy as training so checkpoint keys match.
    fsdp_model = build_fsdp_model(base_model, dtype=dtype, device=device, world_size=world_size)

    # Load local shard
    model_path = os.path.join(args.ckpt, f"model_world_size_{world_size}_rank_{rank}.pt")
    if not os.path.isfile(model_path):
        if rank == 0:
            print(f"[merge] Missing shard: {model_path}")
        dist.barrier()
        dist.destroy_process_group()
        sys.exit(1)

    model_state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if rank == 0:
        print(f"[merge] Loading shard {model_path}")

    # torch.distributed.checkpoint.set_state_dict in some PyTorch versions does not
    # accept None for optimizers; use empty tuples for compatibility.
    sharded_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
        set_state_dict(
            model=fsdp_model,
            optimizers=(),
            model_state_dict=model_state_dict,
            optim_state_dict=(),
            options=StateDictOptions(cpu_offload=True),
        )
    dist.barrier()

    # Gather full state dict on rank 0
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_cfg):
        full_state = fsdp_model.state_dict()

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        # Load a fresh base model on CPU and set weights
        cpu_model = None
        try:
            cpu_model = AutoModelForVision2Seq.from_pretrained(
                hf_src,
                torch_dtype=torch.float32,
                trust_remote_code=args.trust_remote_code,
            )
        except OSError as e:
            print(f"[merge] HF weights not found in {hf_src}, falling back to from_config ({e})")
        if cpu_model is None:
            try:
                cpu_model = AutoModelForVision2Seq.from_config(cfg, trust_remote_code=args.trust_remote_code)
            except TypeError:
                if args.trust_remote_code:
                    setattr(cfg, "trust_remote_code", True)
                cpu_model = AutoModelForVision2Seq.from_config(cfg)
        cpu_model.load_state_dict(full_state, strict=True)
        cpu_model.save_pretrained(args.out)

        # Copy tokenizer/processor
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_src, trust_remote_code=args.trust_remote_code)
            tokenizer.save_pretrained(args.out)
        except Exception:
            pass
        try:
            processor = AutoProcessor.from_pretrained(hf_src, trust_remote_code=args.trust_remote_code)
            processor.save_pretrained(args.out)
        except Exception:
            pass

        print(f"[merge] Merged weights saved to {os.path.abspath(args.out)}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
