"""Adapter for InternVL2 / InternVL2.5 / InternVL3.5."""

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from geoaux.runner import BaseRunner

# ---------- InternVL image preprocessing ----------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tar = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    tw, th = image_size * tar[0], image_size * tar[1]
    resized = image.resize((tw, th))
    blocks = tar[0] * tar[1]
    processed = []
    for idx in range(blocks):
        box = (
            (idx % (tw // image_size)) * image_size,
            (idx // (tw // image_size)) * image_size,
            ((idx % (tw // image_size)) + 1) * image_size,
            ((idx // (tw // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) > 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _load_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert("RGB")
    transform = _build_transform(input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


# ---------- Runner ----------

class InternVLRunner(BaseRunner):
    prompt_key = "internvl"

    def load_model(self, device):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().to(device)
        return model, tokenizer

    def run_single_item(self, model, processor, item, device):
        # processor is actually tokenizer for InternVL
        tokenizer = processor
        pixel_values = _load_image(item["image_path"], max_num=12).to(torch.bfloat16).to(device)

        generation_config = dict(max_new_tokens=1024, do_sample=False)
        response, _ = model.chat(
            tokenizer, pixel_values, item["prompt_text"],
            generation_config, history=None, return_history=True,
        )
        return response
