#!/usr/bin/env python
"""Unified evaluation CLI.

Examples:
    # Extract + score in one go
    python scripts/run_eval.py --input_file ./output/model_results.json --smart --use_judge

    # Extract only
    python scripts/run_eval.py --input_file ./output/model_results.json --step extract --smart

    # Score only (from previously extracted file)
    python scripts/run_eval.py --input_file ./eval_results/extracted_answers.json --step score --use_judge
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geoaux.eval.extract_answer import run_extraction
from geoaux.eval.score_answer import run_scoring
from geoaux.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Geoaux unified evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--step", type=str, default="both", choices=["extract", "score", "both"],
        help="Which step to run (default: both)",
    )
    # Extraction args
    parser.add_argument("--smart", action="store_true", help="Force LLM extraction for all samples")
    parser.add_argument("--no_llm", action="store_true", help="Regex-only extraction")
    # Scoring args
    parser.add_argument("--use_judge", action="store_true", help="Use LLM judge for non-exact matches")
    parser.add_argument("--benchmark_file", type=str, default=None, help="Benchmark metadata JSON")
    # Shared
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--base_url", type=str, default=None, help="API base URL")
    parser.add_argument("--model", type=str, default=None, help="LLM model for extraction/judge")
    parser.add_argument("--num_workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (debug)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")

    args = parser.parse_args()
    cfg = load_config(args.config)

    output_dir = args.output_dir or cfg.get("eval_output_dir", "./eval_results")

    if args.step in ("extract", "both"):
        extracted_file = run_extraction(
            input_file=args.input_file,
            output_dir=output_dir,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            smart=args.smart,
            no_llm=args.no_llm,
            limit=args.limit,
            num_workers=args.num_workers,
        )
    else:
        extracted_file = args.input_file

    if args.step in ("score", "both"):
        run_scoring(
            input_file=extracted_file,
            output_dir=output_dir,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            benchmark_file=args.benchmark_file,
            use_judge=args.use_judge,
            limit=args.limit,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
