"""Adapter for Mirage (Qwen2.5-VL based with latent tokens)."""

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from geoaux.runner import BaseRunner


def _place_input_image(
    text,
    image_pad="<|vision_start|><|image_pad|><|vision_end|>",
    image_placeholder="<image>",
    sep_token=None,
):
    """Replace <image> placeholder with the Qwen VL vision padding tokens."""
    if sep_token is not None:
        assert sep_token in text
        t1, t2 = text.split(sep_token)
        if image_placeholder in t1:
            t1 = t1.replace(image_pad, "")
            t1 = t1.replace(image_placeholder, image_pad)
        return t1 + sep_token + t2
    return text.replace(image_pad, "").replace(image_placeholder, image_pad)


class MirageRunner(BaseRunner):
    prompt_key = "mirage"

    def load_model(self, device):
        processor_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
        model.to(device)

        # Add Mirage special tokens
        processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
        model.eval()

        return model, processor

    def run_single_item(self, model, processor, item, device):
        image = Image.open(item["image_path"]).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": item["prompt_text"]},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_prompt = _place_input_image(text_prompt, sep_token=None)

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
