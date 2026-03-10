"""APIRunner — ThreadPoolExecutor skeleton for OpenAI-compatible API models."""

import base64
import json
import mimetypes
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from geoaux.prompt import PromptTemplate
from geoaux.utils import get_api_key, get_base_url, load_config, resolve_image_path


class APIRunner:
    """Run inference against an OpenAI-compatible API endpoint."""

    def __init__(self, model_name, config=None, api_key=None, base_url=None):
        self.config = config or load_config()
        api_cfg = self.config.get("api", {})

        self.model_name = model_name or api_cfg.get("default_model", "qwen-plus")
        self.api_key = api_key or api_cfg.get("api_key") or get_api_key()
        self.base_url = base_url or api_cfg.get("base_url", "") or get_base_url("")
        self.num_workers = api_cfg.get("num_workers", 16)
        if not self.api_key:
            raise ValueError(
                "Missing API key. Set DASHSCOPE_API_KEY (or GEOAUX_API_KEY) "
                "or pass --api_key."
            )
        # DashScope compatible-mode enforces max_tokens <= 32768.
        configured_max_tokens = api_cfg.get("max_tokens", 32768)
        try:
            configured_max_tokens = int(configured_max_tokens)
        except Exception:
            configured_max_tokens = 32768
        self.max_tokens = max(1, min(configured_max_tokens, 32768))
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image_path):
        if not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _process_item(self, item, benchmark_root, max_retries=5):
        item_id = item.get("id")
        text_input = item.get("text_input")
        image_path_rel = item.get("image_input")

        prompt_text = PromptTemplate.get_prompt(self.model_name, text_input)

        messages = [{"role": "user", "content": []}]

        # Image
        if image_path_rel:
            image_path = resolve_image_path(image_path_rel, benchmark_root)
            b64 = self._encode_image(image_path) if image_path else None
            if b64:
                mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                })
            else:
                print(f"Warning: Image not found at {image_path} for item {item_id}")

        messages[0]["content"].append({"type": "text", "text": prompt_text})

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                message = response.choices[0].message
                output_text = message.content

                # Capture reasoning_content for Thinking models
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    reasoning = f"<reasoning>\n{message.reasoning_content}\n</reasoning>"
                    output_text = f"{reasoning}\n\n{output_text}" if output_text else reasoning

                finish_reason = response.choices[0].finish_reason

                if finish_reason == "length":
                    print(f"Warning: Response truncated for item {item_id}.")

                if not output_text:
                    if finish_reason == "content_filter":
                        print(f"Warning: Content filtered for item {item_id}")
                    raise ValueError(f"Empty response (finish_reason={finish_reason})")

                return {
                    "id": item_id,
                    "model": self.model_name,
                    "prompt": prompt_text,
                    "response": output_text,
                    "correct_answer": item.get("correct_answer"),
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                else:
                    print(f"Failed item {item_id} after {max_retries} attempts: {e}")
                    return {
                        "id": item_id,
                        "model": self.model_name,
                        "prompt": prompt_text,
                        "response": "",
                        "correct_answer": item.get("correct_answer"),
                        "error": str(e),
                    }
        return None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, data_path=None, output_dir=None, num_workers=None):
        cfg = self.config
        data_path = data_path or cfg["data_path"]
        output_dir = output_dir or cfg["output_dir"]
        num_workers = num_workers or self.num_workers
        benchmark_root = cfg["benchmark_root"]

        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Total samples: {len(data)}")
        print(f"Model: {self.model_name}, Workers: {num_workers}")

        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {
                executor.submit(self._process_item, item, benchmark_root): item
                for item in data
            }
            for future in tqdm(as_completed(future_to_item), total=len(data), desc="Processing"):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"Critical error for item {item.get('id')}: {e}")

        # Sort by id
        try:
            results.sort(key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else x["id"])
        except Exception:
            pass

        safe_name = self.model_name.replace("/", "_")
        final_file = os.path.join(output_dir, f"{safe_name}_results.json")
        print(f"Saving {len(results)} results to {final_file}...")
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("Done.")
        return final_file
