"""Adapter for Qwen3-VL."""

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from geoaux.runner import BaseRunner


class Qwen3VLRunner(BaseRunner):
    prompt_key = "qwen3"

    def load_model(self, device):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=None,
            attn_implementation="flash_attention_2",
        )
        model.to(device).eval()
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        return model, processor

    def run_single_item(self, model, processor, item, device):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image_path"]},
                    {"type": "text", "text": item["prompt_text"]},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
