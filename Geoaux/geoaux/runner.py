"""BaseRunner — multiprocessing skeleton for local-model inference."""

import json
import math
import multiprocessing
import os
import traceback
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from geoaux.prompt import PromptTemplate
from geoaux.utils import load_config, resolve_image_path


class BaseRunner(ABC):
    """
    Subclass this for each local HuggingFace model.
    Override ``load_model`` and ``run_single_item``.
    """

    # Subclasses should set this to the prompt key used in PromptTemplate.get_prompt
    prompt_key: str = "default"

    def __init__(self, model_path, config=None):
        self.model_path = model_path
        self.config = config or load_config()
        self.model_name_short = os.path.basename(model_path.rstrip("/"))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self, device):
        """Return (model, processor) loaded onto *device*."""

    @abstractmethod
    def run_single_item(self, model, processor, item, device):
        """Return a response string for one benchmark item.

        *item* is a dict with at least:
          - text_input, image_input (resolved abs path), question_type
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_shard(self, rank, world_size, data, output_file, benchmark_root, debug=False):
        device = f"cuda:{rank}"
        print(f"Process {rank}/{world_size} started on {device}")

        try:
            model, processor = self.load_model(device)
        except Exception as e:
            print(f"Rank {rank}: Error loading model: {e}")
            traceback.print_exc()
            return

        chunk_size = math.ceil(len(data) / world_size)
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, len(data))
        my_data = data[start_idx:end_idx]

        results = []

        for i, item in enumerate(tqdm(my_data, desc=f"Rank {rank}", position=rank)):
            item_id = item.get("id")
            text_input = item.get("text_input")
            image_path = resolve_image_path(item.get("image_input"), benchmark_root)

            if not image_path:
                print(f"Warning: Image path is None, skipping item {item_id}")
                continue
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}, skipping item {item_id}")
                continue

            q_type = item.get("question_type", "free-form")
            prompt_text = PromptTemplate.get_prompt(self.prompt_key, text_input, q_type)

            # Build enriched item for the adapter
            enriched = {
                **item,
                "image_path": image_path,
                "prompt_text": prompt_text,
                "question_type": q_type,
            }

            try:
                output_text = self.run_single_item(model, processor, enriched, device)

                if debug:
                    print(f"\n[Rank {rank} Debug] ID: {item_id}")
                    print(f"Prompt (truncated): {prompt_text[:100]}...")
                    print(f"Response: {output_text}\n{'-' * 50}")

                results.append({
                    "id": item_id,
                    "model": self.model_name_short,
                    "prompt": prompt_text,
                    "response": output_text,
                    "correct_answer": item.get("correct_answer"),
                    "question_type": q_type,
                })

                # Checkpoint every 50 items
                if (i + 1) % 50 == 0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"Rank {rank}: Error processing item {item_id}: {e}")
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                continue

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Rank {rank} finished. Saved to {output_file}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, data_path=None, output_dir=None, world_size=None, debug=False):
        cfg = self.config
        data_path = data_path or cfg["data_path"]
        output_dir = output_dir or cfg["output_dir"]
        world_size = world_size or cfg.get("world_size", 4)
        benchmark_root = cfg["benchmark_root"]

        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Total samples: {len(data)}. World size: {world_size}")

        # Use spawn for CUDA compatibility
        multiprocessing.set_start_method("spawn", force=True)

        processes = []
        temp_files = []
        tag = self.model_name_short.lower().replace("/", "_")

        for rank in range(world_size):
            output_file = os.path.join(output_dir, f"{tag}_part_{rank}.json")
            temp_files.append(output_file)
            p = multiprocessing.Process(
                target=self._run_shard,
                args=(rank, world_size, data, output_file, benchmark_root, debug),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge shards
        print("Merging results...")
        all_results = []
        for tf in temp_files:
            if os.path.exists(tf):
                with open(tf, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))
                os.remove(tf)

        final_file = os.path.join(output_dir, f"{self.model_name_short}_results.json")
        print(f"Saving combined results to {final_file}...")
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print("Done.")
        return final_file
