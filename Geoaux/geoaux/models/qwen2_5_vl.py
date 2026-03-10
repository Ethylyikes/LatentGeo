"""Adapter for Qwen2-VL / Qwen2.5-VL / GeoInt-R1."""

import torch
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    _USE_QWEN25_CLASS = True
except ImportError:
    from transformers import AutoModelForVision2Seq as Qwen2_5_VLForConditionalGeneration
    _USE_QWEN25_CLASS = False
from transformers import AutoConfig, AutoProcessor
from qwen_vl_utils import process_vision_info

from geoaux.runner import BaseRunner


class Qwen25VLRunner(BaseRunner):
    prompt_key = "qwen"

    def _load_processor(self):
        # 1) Normal load from target repo
        try:
            return AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception as e:
            err_msg = str(e)
            if "Unrecognized image processor" not in err_msg:
                raise

        # 2) Retry with force_download to avoid stale/corrupted HF cache metadata.
        try:
            print(
                f"Warning: processor auto-load failed for {self.model_path}. "
                "Retrying with force_download=True."
            )
            return AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
                force_download=True,
            )
        except Exception as e2:
            if "Unrecognized image processor" not in str(e2):
                raise

        # 3) Try explicit class loader before fallback.
        try:
            from transformers import Qwen2_5_VLProcessor

            print(
                f"Warning: AutoProcessor still failed for {self.model_path}. "
                "Trying Qwen2_5_VLProcessor.from_pretrained(...)."
            )
            return Qwen2_5_VLProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception as e3:
            if "Unrecognized image processor" not in str(e3):
                raise

        # 4) Last-resort fallback: only for qwen2_5_vl-family checkpoints.
        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", None)
        if model_type != "qwen2_5_vl":
            raise RuntimeError(
                f"Processor load failed for {self.model_path}, and model_type={model_type} "
                "is not qwen2_5_vl, so fallback is unsafe."
            )

        fallback_repo = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(
            f"Warning: processor loading failed for {self.model_path}. "
            f"Falling back to processor from {fallback_repo} as last resort."
        )
        return AutoProcessor.from_pretrained(
            fallback_repo,
            trust_remote_code=True,
            use_fast=False,
        )

    def load_model(self, device):
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        for attr in ("tp_plan", "_tp_plan", "base_model_tp_plan"):
            if hasattr(config, attr):
                setattr(config, attr, None)
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            for attr in ("tp_plan", "_tp_plan", "base_model_tp_plan"):
                if hasattr(text_cfg, attr):
                    setattr(text_cfg, attr, None)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        if hasattr(model, "generation_config"):
            for key in ("temperature", "top_p", "top_k"):
                if hasattr(model.generation_config, key):
                    setattr(model.generation_config, key, None)
        processor = self._load_processor()
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

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        model_device = next(model.parameters()).device
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        else:
            inputs = {
                k: (v.to(model_device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }

        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
