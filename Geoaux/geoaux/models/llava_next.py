"""Adapter for LLaVA-Next (LLaVA-1.6)."""

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from geoaux.runner import BaseRunner


def _load_processor(model_path):
    """Load LlavaNextProcessor, with fallback for older transformers."""
    try:
        return LlavaNextProcessor.from_pretrained(model_path)
    except TypeError:
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return LlavaNextProcessor(image_processor=image_processor, tokenizer=tokenizer)


class LLaVANextRunner(BaseRunner):
    prompt_key = "llava"

    def load_model(self, device):
        processor = _load_processor(self.model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        return model, processor

    def run_single_item(self, model, processor, item, device):
        image = Image.open(item["image_path"]).convert("RGB")

        # Build Vicuna-format prompt directly to avoid apply_chat_template issues
        prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            f"USER: <image>\n{item['prompt_text']} ASSISTANT:"
        )

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        # Decode only newly generated tokens
        input_len = inputs.input_ids.shape[1]
        return processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
