#!/usr/bin/env python
"""Unified inference CLI.

Examples:
    # Local model
    python scripts/run_infer.py --model_key qwen2.5-vl --model_path Qwen/Qwen2.5-VL-7B-Instruct

    # API model
    python scripts/run_infer.py --model_key api --model_path qwen-plus
"""

import argparse
import sys
import os

# Ensure project root is on sys.path so `geoaux` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geoaux.models import get_runner, MODEL_REGISTRY
from geoaux.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Geoaux unified inference")
    parser.add_argument(
        "--model_key", type=str, required=True,
        help=f"Model key. Available: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument("--model_path", type=str, required=True, help="HF model ID or local path")
    parser.add_argument("--data_path", type=str, default=None, help="Path to benchmark JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--world_size", type=int, default=None, help="Number of GPUs (local models)")
    parser.add_argument("--num_workers", type=int, default=None, help="Thread workers (API models)")
    parser.add_argument("--api_key", type=str, default=None, help="API key (API models)")
    parser.add_argument("--base_url", type=str, default=None, help="API base URL (API models)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")

    args = parser.parse_args()

    cfg = load_config(args.config)

    runner_cls = get_runner(args.model_key)

    if args.model_key == "api":
        # APIRunner
        runner = runner_cls(
            model_name=args.model_path,
            config=cfg,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        runner.run(
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )
    else:
        # BaseRunner subclass
        runner = runner_cls(model_path=args.model_path, config=cfg)
        runner.run(
            data_path=args.data_path,
            output_dir=args.output_dir,
            world_size=args.world_size,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
