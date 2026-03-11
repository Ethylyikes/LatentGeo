from __future__ import annotations

import importlib
import logging
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Prefer the local transformers-4.56.0 (matches RL training/latent logic) when available.
try:
    _REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    _REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_TRANSFORMERS = _REPO_ROOT / "transformers-4.56.0" / "src"
if _LOCAL_TRANSFORMERS.exists() and str(_LOCAL_TRANSFORMERS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_TRANSFORMERS))

import torch

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...dataset import DATASET_MODALITY
from ...smp import get_gpu_memory, listinstr

VLLM_MAX_IMAGE_INPUT_NUM = 24
LATENT_PROMPT_TEMPLATE = """{question}
{latent_placeholder}"""
_LATENT_VLLM_PATCHED = False


@dataclass
class LatentSpec:
    pad_token: str
    start_token: str
    end_token: str
    slots: int
    pad_id: int | None = None
    start_id: int | None = None
    end_id: int | None = None


def _env_int(*keys: str) -> int | None:
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _normalize_dtype(value: Any) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"fp16", "float16", "torch.float16"}:
            return torch.float16
        if key in {"bf16", "bfloat16", "torch.bfloat16"}:
            return torch.bfloat16
        if key in {"fp32", "float32", "torch.float32"}:
            return torch.float32
    return None


def _strip_latent_segments(text: str, latent: LatentSpec | None) -> str:
    if not latent or not text:
        return text
    start = re.escape(latent.start_token)
    end = re.escape(latent.end_token)
    pattern = re.compile(f"{start}.*?{end}", flags=re.DOTALL)
    cleaned = pattern.sub("", text)
    cleaned = cleaned.replace(latent.pad_token, "")
    return cleaned.strip()


def _patch_vllm_latent(repo_root: Path, latent: LatentSpec | None) -> bool:
    global _LATENT_VLLM_PATCHED
    if _LATENT_VLLM_PATCHED:
        return True

    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    if latent:
        if latent.start_id is not None:
            os.environ.setdefault("LATENT_START_ID", str(latent.start_id))
        if latent.end_id is not None:
            os.environ.setdefault("LATENT_END_ID", str(latent.end_id))
        if latent.slots and latent.slots > 0:
            existing = _env_int("LATENT_SIZE")
            if existing is None or existing <= 0:
                os.environ["LATENT_SIZE"] = str(latent.slots)

    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    patched = None
    import_candidates = [
        "inference.vllm.latent_gpu_model_runner",
    ]
    for module_name in import_candidates:
        try:
            patched = importlib.import_module(module_name)
            break
        except Exception:
            continue

    if patched is None:
        rl_candidates = [
            repo_root / "RL",
            repo_root / "LatentGeo" / "RL",
        ]
        for rl_path in rl_candidates:
            if rl_path.exists() and str(rl_path) not in sys.path:
                sys.path.insert(0, str(rl_path))

        for module_name in (
            "latent_models.vllm.latent_gpu_model_runner",
        ):
            try:
                patched = importlib.import_module(module_name)
                break
            except Exception:
                continue

    if patched is None:
        logging.warning("[Latent vLLM] Failed to import patched GPU model runner.")
        return False

    for key in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.worker.gpu_model_runner",
        "vllm.worker.model_runner",
    ):
        sys.modules[key] = patched

    _LATENT_VLLM_PATCHED = True
    logging.info("[Latent vLLM] Patched vLLM GPU model runner.")
    return True


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    import tempfile

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sampling_gap_maxframe = (1 if not num_frames else math.ceil(frame_count / num_frames))
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_number % sampling_gap == 0:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(f"Sampled {len(images)}/{frame_number} frames from video {video_path}")
    cap.release()
    return images


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.99,
        top_k=1,
        temperature=0.9,
        repetition_penalty=1.0,
        do_sample: bool = True,
        use_custom_prompt: bool = False,
        use_latent_prompt: bool = False,
        fixed_user_prefix: str | None = None,
        log_input_prompt: bool = False,
        latent_slots: int = 10,
        latent_pad_token: str = "<|latent_pad|>",
        latent_start_token: str = "<|latent_start|>",
        latent_end_token: str = "<|latent_end|>",
        prefix_question: bool = False,
        system_prompt: str | None = None,
        vqa_prompt_suffix: str | None = None,
        latent_prompt_template: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = False,
        latent_debug: bool = False,
        use_latent_vllm: bool | None = None,
        strip_latent_output: bool = True,
        return_raw_response: bool = False,
        log_latent_tokens: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt, vqa_prompt_suffix=vqa_prompt_suffix)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print('The total number of video tokens might become too large, resulting in an overly long input sequence. We recommend lowering **total_pixels** to below **24576 × 28 × 28**.')  # noqa: E501
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.use_latent_prompt = use_latent_prompt
        self.fixed_user_prefix = fixed_user_prefix
        self.log_input_prompt = log_input_prompt
        self.strip_latent_output = strip_latent_output
        self.return_raw_response = return_raw_response
        self.log_latent_tokens = log_latent_tokens
        env_latent_pad = os.environ.get("LATENT_PAD_TOKEN") or os.environ.get("ABS_VIS_PAD_TOKEN")
        env_latent_start = os.environ.get("LATENT_START_TOKEN") or os.environ.get("ABS_VIS_START_TOKEN")
        env_latent_end = os.environ.get("LATENT_END_TOKEN") or os.environ.get("ABS_VIS_END_TOKEN")
        if env_latent_pad:
            latent_pad_token = env_latent_pad
        if env_latent_start:
            latent_start_token = env_latent_start
        if env_latent_end:
            latent_end_token = env_latent_end
        self.latent = LatentSpec(
            pad_token=latent_pad_token,
            start_token=latent_start_token,
            end_token=latent_end_token,
            slots=latent_slots,
        )
        self.prefix_question = prefix_question
        self.latent_prompt_template = latent_prompt_template or LATENT_PROMPT_TEMPLATE
        self.latent_debug = latent_debug
        if use_latent_vllm is None:
            use_latent_vllm = os.environ.get("LATENT_VLLM_PATCH", "0") == "1"
        self.use_latent_vllm = use_latent_vllm
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        if self.fps is None and self.nframe is None:
            print("Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted")
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path
        print(f"[Qwen2VLChat] model_path = {self.model_path}")

        MODEL_CLS = None
        self._is_qwen25_vl = False

        model_dtype = kwargs.pop("dtype", None)
        model_torch_dtype = kwargs.pop("torch_dtype", None)
        dtype_from_kwargs = model_dtype is not None or model_torch_dtype is not None
        if model_dtype is None:
            model_dtype = model_torch_dtype
        model_dtype = _normalize_dtype(model_dtype)

        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
                raise err
            MODEL_CLS = Qwen2_5OmniForConditionalGeneration
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        else:
            from transformers import AutoConfig, AutoProcessor
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if model_dtype is None:
                model_dtype = _normalize_dtype(getattr(cfg, "torch_dtype", None))
            if (not dtype_from_kwargs) and "fp16" in model_path.lower():
                if model_dtype is None or model_dtype == torch.float32:
                    model_dtype = torch.float16
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            if getattr(cfg, "model_type", "") == "qwen2_5_vl":
                from transformers import Qwen2_5_VLForConditionalGeneration
                MODEL_CLS = Qwen2_5_VLForConditionalGeneration
                self._is_qwen25_vl = True
            else:
                from transformers import Qwen2VLForConditionalGeneration
                MODEL_CLS = Qwen2VLForConditionalGeneration
                self._is_qwen25_vl = False

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        if self.use_latent_vllm:
            self.use_vllm = True
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag between `use_vllm` and `use_lmdeploy` to True"  # noqa: E501

        if self.use_vllm:
            if self.use_latent_vllm:
                self._setup_latent_tokens(model=None)
                _patch_vllm_latent(_REPO_ROOT, self.latent)
            from vllm import LLM
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
                try:
                    import multiprocessing as mp
                    mp.set_start_method("spawn", force=True)
                except RuntimeError:
                    pass
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})')
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning("VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn' to avoid potential multi-process issues")  # noqa: E501
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=5,
                max_model_len=32768,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )
        elif self.use_lmdeploy:
            from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig
            num_gpus = torch.cuda.device_count()
            self.model = pipeline(
                model_path,
                backend_config=TurbomindEngineConfig(session_len=32768, cache_max_entry_count=0.1, tp=num_gpus),
                chat_template_config=ChatTemplateConfig(model_name='qwen2d5-vl'))
            torch.cuda.set_device(0)
            self.device = 'cuda'
            self._setup_latent_tokens()
        else:
            if getattr(self, "_is_qwen25_vl", False):
                self.model = MODEL_CLS.from_pretrained(
                    model_path,
                    device_map="auto",
                    dtype=model_dtype or torch.bfloat16,
                    attn_implementation='flash_attention_2',
                )
            else:
                self.model = MODEL_CLS.from_pretrained(
                    model_path,
                    device_map="auto",
                    dtype=model_dtype or torch.bfloat16,
                    attn_implementation='flash_attention_2',
                )
            self.model.eval()
            self._setup_latent_tokens()

        if self.latent_debug:
            os.environ.setdefault("LATENT_DEBUG", "1")
            logging.getLogger("latent_debug").setLevel(logging.INFO)

        torch.cuda.empty_cache()

    def _apply_latent_prompt(self, question: str) -> str:
        latent_placeholder = (
            f"{self.latent.start_token}"
            f"{self.latent.pad_token * self.latent.slots}"
            f"{self.latent.end_token}"
        )
        return self.latent_prompt_template.format(
            latent_placeholder=latent_placeholder,
            question=question.strip(),
        )

    def _setup_latent_tokens(self, model: Any = None):
        tokenizer = getattr(self.processor, "tokenizer", None)
        model = model or getattr(self, "model", None)
        if tokenizer is None:
            return

        added = tokenizer.add_tokens(
            [self.latent.pad_token, self.latent.start_token, self.latent.end_token],
            special_tokens=True,
        )
        if added and model is not None and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
            logging.info(f"[latent] Resized embeddings to {len(tokenizer)} after adding latent tokens.")

        try:
            self.latent.pad_id = int(tokenizer(self.latent.pad_token, return_tensors="pt")["input_ids"][0])
            self.latent.start_id = int(tokenizer(self.latent.start_token, return_tensors="pt")["input_ids"][0])
            self.latent.end_id = int(tokenizer(self.latent.end_token, return_tensors="pt")["input_ids"][0])
        except Exception as exc:
            logging.warning(f"[latent] Failed to fetch latent token ids: {exc}")
            return

        cfg = getattr(model, "config", None)
        if cfg is not None:
            cfg.latent_token_id = self.latent.pad_id
            cfg.latent_start_id = self.latent.start_id
            cfg.latent_end_id = self.latent.end_id
            if getattr(cfg, "latent_size", None) is None:
                cfg.latent_size = self.latent.slots
            else:
                self.latent.slots = cfg.latent_size
            if getattr(cfg, "stage", None) is None:
                cfg.stage = "stage2"
        if model is not None and hasattr(model, "latent_model"):
            model.latent_model = True

        os.environ.setdefault("LATENT_PAD_TOKEN", self.latent.pad_token)
        os.environ.setdefault("LATENT_START_TOKEN", self.latent.start_token)
        os.environ.setdefault("LATENT_END_TOKEN", self.latent.end_token)
        os.environ.setdefault("LATENT_SIZE", str(self.latent.slots))

        if self.latent_debug or self.log_latent_tokens:
            logging.info(
                f"[latent] ids start={self.latent.start_id}, pad={self.latent.pad_id}, end={self.latent.end_id}, slots={self.latent.slots}"
            )

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                text_value = s['value']
                if self.fixed_user_prefix:
                    text_value = f"{self.fixed_user_prefix}\n\n{text_value}"
                if self.prefix_question and not text_value.lstrip().lower().startswith("question:"):
                    text_value = f"Question: {text_value}"
                if self.use_latent_prompt:
                    text_value = self._apply_latent_prompt(text_value)
                item = {'type': 'text', 'text': text_value}
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        video_inputs = [s for s in inputs if s['type'] == 'video']
        video_count = len(video_inputs)
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                if video_count > 1:
                    logging.warning("Multiple videos detected. Using video frames for each video")
                    if dataset == 'OCRBench':
                        min_pixels = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={min_pixels}")
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    else:
                        if self.min_pixels is not None:
                            min_pixels = self.min_pixels
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()

                    frames_per_video = max(1, self.limit_mm_per_prompt // video_count)
                    content.append({"type": "text", "text": "<video frames start>"})
                    content.extend(process_video(s['value'], frames_per_video, min_pixels, max_pixels))
                    content.append({"type": "text", "text": "<video frames end>"})

                else:
                    item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                    if self.total_pixels is not None:
                        item['total_pixels'] = self.total_pixels
                    if self.fps is not None:
                        item['fps'] = self.fps
                    elif self.nframe is not None:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
                    content.append(item)
            elif s['type'] == 'text':
                text_value = s['value']
                if self.fixed_user_prefix:
                    text_value = f"{self.fixed_user_prefix}\n\n{text_value}"
                if self.prefix_question and not text_value.lstrip().lower().startswith("question:"):
                    text_value = f"Question: {text_value}"
                if self.use_latent_prompt:
                    text_value = self._apply_latent_prompt(text_value)
                item = {'type': 'text', 'text': text_value}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content

    def _build_messages(self, message, dataset=None, for_vllm: bool = False):
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        if for_vllm:
            content = self._prepare_content_vllm(message, dataset=dataset)
        else:
            content = self._prepare_content(message, dataset=dataset)
        messages.append({'role': 'user', 'content': content})
        return messages

    def _log_prompt(self, text: str):
        if self.log_input_prompt:
            logging.info(f"[prompt] {text}")
            print(f"[prompt] {text}")
        if self.latent_debug or self.log_latent_tokens:
            tokenizer = getattr(self.processor, "tokenizer", None)
            try:
                ids = tokenizer(text, add_special_tokens=False).input_ids
                start_positions = [i for i, t in enumerate(ids) if t == self.latent.start_id]
                pad_positions = [i for i, t in enumerate(ids) if t == self.latent.pad_id]
                end_positions = [i for i, t in enumerate(ids) if t == self.latent.end_id]
                logging.info(f"[latent] prompt token positions start={start_positions}, pad_count={len(pad_positions)}, end={end_positions}")
            except Exception as exc:
                logging.warning(f"[latent] Failed to tokenize prompt for debug: {exc}")

    def _decode_response(self, generated_ids, inputs):
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response_raw = out[0]
        response_clean = _strip_latent_segments(response_raw, self.latent) if self.strip_latent_output else response_raw
        if self.return_raw_response and response_clean != response_raw:
            logging.info(f"[latent] raw response kept for debug: {response_raw}")
        response = response_clean
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner_transformers(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = self._build_messages(message, dataset=dataset, for_vllm=False)
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self._log_prompt(text)

        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info([messages], use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(text=text, images=images, audio=audios, videos=videos, padding=True, return_tensors='pt', use_audio_in_video=self.use_audio_in_video)  # noqa: E501
            self.generate_kwargs['use_audio_in_video'] = self.use_audio_in_video
            self.generate_kwargs['return_audio'] = False
        else:
            images, videos = process_vision_info(messages)
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        return self._decode_response(generated_ids, inputs)

    def generate_inner_lmdeploy(self, message, dataset=None):
        from lmdeploy import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            top_p=self.generate_kwargs['top_p'],
            top_k=self.generate_kwargs['top_k'],
            temperature=self.generate_kwargs['temperature'],
            repetition_penalty=self.generate_kwargs['repetition_penalty'],
        )
        gen_config.random_seed = None
        messages_list = self.message_to_lmdeploy(message, system_prompt=self.system_prompt)
        assert len(messages_list) == 1
        response = self.model(messages_list, gen_config=gen_config)[0]
        response = response.text
        if self.strip_latent_output:
            response = _strip_latent_segments(response, self.latent)
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams

        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = self._build_messages(message, dataset=dataset, for_vllm=True)
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self._log_prompt(text)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            images, videos = process_vision_info(messages)
        print('finishing process vision info in vllm.')

        sampling_params = SamplingParams(
            temperature=self.generate_kwargs['temperature'],
            top_p=self.generate_kwargs['top_p'],
            top_k=self.generate_kwargs['top_k'],
            repetition_penalty=self.generate_kwargs['repetition_penalty'],
            max_tokens=self.max_new_tokens,
            stop_token_ids=None,
        )

        if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            assert len(videos) == 1
            videos_nd = [videos[0].detach().cpu().numpy().transpose(0, 2, 3, 1)]
            video_inputs = {
                "prompt": text[0] if isinstance(text, list) else text,
                "multi_modal_data": {"video": videos_nd[0]},
                "mm_processor_kwargs": {}
            }
            if self.use_audio_in_video:
                import vllm
                assert not vllm.envs.VLLM_USE_V1, ("V1 does not support use_audio_in_video. Please launch this example with `VLLM_USE_V1=0`.")  # noqa: E501
                video_inputs["multi_modal_data"]["audio"] = audios[0]
                video_inputs['mm_processor_kwargs']['use_audio_in_video'] = True
            if videos_nd[0].shape[0] > VLLM_MAX_IMAGE_INPUT_NUM:
                print('video input sequence may be too long for vllm, Maybe cannot generate response for VLLM')

        if images:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                    "multi_modal_data": {"image": images},
                },
                sampling_params=sampling_params,
            )
        elif DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            outputs = self.llm.generate(video_inputs, sampling_params=sampling_params)
        else:
            outputs = self.llm.generate({"prompt": text}, sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text
        if self.strip_latent_output:
            generated_text = _strip_latent_segments(generated_text, self.latent)
        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        elif self.use_lmdeploy:
            return self.generate_inner_lmdeploy(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)


class Qwen2VLChatAguvis(Qwen2VLChat):
    def __init__(self, mode=None, **kwargs):
        self.mode = mode
        super().__init__(**kwargs)
        self.processor.max_pixels = self.max_pixels
        self.processor.min_pixels = self.min_pixels

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        messages = []
        user_message = []
        for item in message:
            if "role" in item.keys():
                if item["role"] == "system":
                    self.system_prompt = item["value"]
                else:
                    item.pop("role")
                    user_message.append(item)
            else:
                user_message.append(item)
        message = user_message

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"  # noqa: E501

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=chat_template,
        )
        if self.mode == "force-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
        elif self.mode == "force-plan-l1":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nAction: "
        elif self.mode == "force-plan-l3":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nObservation: "
        elif self.mode == "grounding":
            recipient_text = "<|im_start|>assistant<|recipient|>os\n"
        elif self.mode == "force-plan-free":
            recipient_text = "<|im_start|>assistant<|recipient|>all\n"
        elif self.mode == "self-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        text += recipient_text

        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]

        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response
