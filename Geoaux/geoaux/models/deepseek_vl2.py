"""Adapter for DeepSeek-VL2."""

import os
import sys

import torch
from transformers import AutoModelForCausalLM

from geoaux.runner import BaseRunner

# Ensure the vendored DeepSeek-VL2 package is importable
_DEEPSEEK_VL2_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "DeepSeek-VL2"
)
if os.path.isdir(_DEEPSEEK_VL2_DIR) and _DEEPSEEK_VL2_DIR not in sys.path:
    sys.path.insert(0, _DEEPSEEK_VL2_DIR)

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images


class DeepSeekVL2Runner(BaseRunner):
    prompt_key = "deepseek"

    def load_model(self, device):
        vl_processor = DeepseekVLV2Processor.from_pretrained(self.model_path)
        vl_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        vl_model = vl_model.to(torch.bfloat16).to(device).eval()
        return vl_model, vl_processor

    def run_single_item(self, model, processor, item, device):
        tokenizer = processor.tokenizer

        conversation = [
            {
                "role": "<|User|>",
                "content": item["prompt_text"],
                "images": [item["image_path"]],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(model.device)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
        )
        return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
