import os
from PIL import Image

from utils import get_args

_RUNTIME_ARGS = None


def _normalize_relative_path(path: str) -> str:
    if path.startswith("./data/"):
        return path.replace("./data/", "", 1)
    if path.startswith("data/"):
        return path.replace("data/", "", 1)
    return path


def _resolve_from_data_dir(data_path: str, rel_path: str) -> str:
    data_dir = os.path.dirname(os.path.abspath(data_path))
    return os.path.join(data_dir, _normalize_relative_path(rel_path))


def latentgeo_preprocess_function(sample):
    """
    Build one conversation sample for LatentGeo training.

    Stage1/2: user image + assistant latent target image + text output
    Stage3:   user image + assistant latent placeholder + text output
    """
    global _RUNTIME_ARGS
    if _RUNTIME_ARGS is None:
        _RUNTIME_ARGS = get_args()
    args = _RUNTIME_ARGS

    image_input_path = _resolve_from_data_dir(args.data_path, sample["image_input"])
    image_input = Image.open(image_input_path).convert("RGB")

    # Avoid OOM from oversized inputs.
    max_size = 1024
    if max(image_input.size) > max_size:
        image_input.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    user_content = [
        {"type": "image", "image": image_input},
        {"type": "text", "text": sample["text_input"]},
    ]

    use_latent = sample.get("use_latent", True)
    is_stage3 = args.stage == "stage3"

    if not use_latent:
        assistant_content = [{"type": "text", "text": sample["text_output"]}]
    elif is_stage3:
        assistant_content = [
            {"type": "text", "text": sample["aux_think"] + "\n"},
            {"type": "text", "text": "<|latent_start|><|latent_pad|><|latent_end|>"},
            {"type": "text", "text": "\n" + sample["text_output"]},
        ]
    else:
        image_output_path = _resolve_from_data_dir(args.data_path, sample["image_output"])
        image_output = Image.open(image_output_path).convert("RGB")
        assistant_content = [
            {"type": "text", "text": sample["aux_think"] + "\n"},
            {"type": "image", "image": image_output},
            {"type": "text", "text": "\n" + sample["text_output"]},
        ]

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


task_preprocess_config = {
    "latentgeo": latentgeo_preprocess_function,
}

# Backward compatibility with old typo name.
task_preporcess_config = task_preprocess_config
