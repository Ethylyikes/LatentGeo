"""Adapter for Qwen2-VL (and Qwen2-VL-based models like R1-VL)."""

import torch
from transformers import AutoConfig, AutoProcessor, PretrainedConfig
from qwen_vl_utils import process_vision_info

from geoaux.runner import BaseRunner

# transformers 4.57+ may not re-export Qwen2VLForConditionalGeneration at the top level.
# Try submodule path first, then fall back to AutoModelForCausalLM with trust_remote_code.
try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    _MODEL_CLS = Qwen2VLForConditionalGeneration
except (ImportError, ModuleNotFoundError):
    try:
        from transformers import Qwen2VLForConditionalGeneration
        _MODEL_CLS = Qwen2VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForCausalLM
        _MODEL_CLS = AutoModelForCausalLM


class Qwen2VLRunner(BaseRunner):
    prompt_key = "qwen"

    @staticmethod
    def _is_valid_vl_processor(proc):
        # Some repos resolve AutoProcessor to tokenizer-only objects.
        # For VLM inference we need both tokenizer and image_processor.
        return hasattr(proc, "image_processor") and hasattr(proc, "tokenizer")

    def _load_processor(self):
        try:
            proc = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
            )
            if not self._is_valid_vl_processor(proc):
                raise ValueError(
                    f"Loaded processor is tokenizer-only ({type(proc).__name__}); "
                    "multimodal processor required."
                )
            return proc
        except Exception as e:
            if (
                "Unrecognized image processor" not in str(e)
                and "tokenizer-only" not in str(e)
            ):
                raise

        try:
            print(
                f"Warning: processor auto-load failed for {self.model_path}. "
                "Retrying with force_download=True."
            )
            proc = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
                force_download=True,
            )
            if not self._is_valid_vl_processor(proc):
                raise ValueError(
                    f"Loaded processor is tokenizer-only ({type(proc).__name__}); "
                    "multimodal processor required."
                )
            return proc
        except Exception as e2:
            if (
                "Unrecognized image processor" not in str(e2)
                and "tokenizer-only" not in str(e2)
            ):
                raise

        try:
            from transformers import Qwen2VLProcessor

            print(
                f"Warning: AutoProcessor still failed for {self.model_path}. "
                "Trying Qwen2VLProcessor.from_pretrained(...)."
            )
            proc = Qwen2VLProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
            )
            if not self._is_valid_vl_processor(proc):
                raise ValueError(
                    f"Loaded processor is tokenizer-only ({type(proc).__name__}); "
                    "multimodal processor required."
                )
            return proc
        except Exception as e3:
            if (
                "Unrecognized image processor" not in str(e3)
                and "tokenizer-only" not in str(e3)
            ):
                raise

        config_dict, _ = PretrainedConfig.get_config_dict(
            self.model_path,
            trust_remote_code=True,
        )
        model_type = config_dict.get("model_type")
        if model_type not in ("qwen2_vl", "qwen2vlm"):
            raise RuntimeError(
                f"Processor load failed for {self.model_path}, and model_type={model_type} "
                "is not qwen2_vl/qwen2vlm, so fallback is unsafe."
            )

        fallback_repo = "Qwen/Qwen2-VL-7B-Instruct"
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
        model_cls = _MODEL_CLS
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }

        # Some checkpoints (e.g. qwen2vlm) use custom model_type that AutoConfig
        # cannot parse in older transformers.
        try:
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            for attr in ("tp_plan", "_tp_plan", "base_model_tp_plan"):
                if hasattr(config, attr):
                    setattr(config, attr, None)
            text_cfg = getattr(config, "text_config", None)
            if text_cfg is not None:
                for attr in ("tp_plan", "_tp_plan", "base_model_tp_plan"):
                    if hasattr(text_cfg, attr):
                        setattr(text_cfg, attr, None)
            model_kwargs["config"] = config
        except Exception as e:
            msg = str(e)
            if "model type `qwen2vlm`" in msg:
                # Avoid AutoConfig re-parse inside AutoModel by constructing a
                # compatible Qwen2VLConfig from raw config dict.
                config_dict, _ = PretrainedConfig.get_config_dict(
                    self.model_path,
                    trust_remote_code=True,
                )
                config_dict["model_type"] = "qwen2_vl"
                config_dict["architectures"] = ["Qwen2VLForConditionalGeneration"]

                try:
                    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
                except Exception:
                    from transformers import Qwen2VLConfig

                patched_config = Qwen2VLConfig.from_dict(config_dict)
                # Math-PUMA style configs keep LM dims only in text_config.
                # Qwen2VL modeling code expects some of them at top-level.
                tc = getattr(patched_config, "text_config", None)
                if tc is not None:
                    for attr in (
                        "hidden_size",
                        "vocab_size",
                        "intermediate_size",
                        "num_attention_heads",
                        "num_hidden_layers",
                        "num_key_value_heads",
                        "max_position_embeddings",
                        "bos_token_id",
                        "eos_token_id",
                    ):
                        top_val = getattr(patched_config, attr, None) if hasattr(patched_config, attr) else None
                        text_val = getattr(tc, attr, None) if hasattr(tc, attr) else None
                        if (top_val is None) and (text_val is not None):
                            setattr(patched_config, attr, text_val)

                for attr in ("tp_plan", "_tp_plan", "base_model_tp_plan"):
                    if hasattr(patched_config, attr):
                        setattr(patched_config, attr, None)
                text_cfg = getattr(patched_config, "text_config", None)
                if text_cfg is not None:
                    for attr in ("tp_plan", "_tp_plan", "base_model_tp_plan"):
                        if hasattr(text_cfg, attr):
                            setattr(text_cfg, attr, None)

                model_kwargs["config"] = patched_config
                # Force native Qwen2VL class when available.
                try:
                    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                        Qwen2VLForConditionalGeneration as _NativeQwen2VL,
                    )
                    model_cls = _NativeQwen2VL
                except Exception:
                    pass
                print(
                    f"Warning: AutoConfig failed for {self.model_path}: {e}\n"
                    "Patched model_type qwen2vlm -> qwen2_vl and retrying load."
                )
            elif "does not recognize this architecture" in msg:
                from transformers import AutoModelForCausalLM
                model_cls = AutoModelForCausalLM
                print(
                    f"Warning: AutoConfig failed for {self.model_path}: {e}\n"
                    "Falling back to AutoModelForCausalLM with trust_remote_code=True."
                )
            else:
                raise

        model = model_cls.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        model.to(device)
        model.eval()
        if hasattr(model, "generation_config"):
            for key in ("temperature", "top_p", "top_k"):
                if hasattr(model.generation_config, key):
                    setattr(model.generation_config, key, None)
        processor = self._load_processor()
        print(f"Loaded processor: {type(processor).__name__}")
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
