import importlib

MODEL_REGISTRY = {
    "qwen2.5-vl":   "geoaux.models.qwen2_5_vl.Qwen25VLRunner",
    "qwen2-vl":     "geoaux.models.qwen2_vl.Qwen2VLRunner",
    "qwen3-vl":     "geoaux.models.qwen3_vl.Qwen3VLRunner",
    "internvl":     "geoaux.models.internvl.InternVLRunner",
    "deepseek-vl2": "geoaux.models.deepseek_vl2.DeepSeekVL2Runner",
    "llava-next":   "geoaux.models.llava_next.LLaVANextRunner",
    "mirage":       "geoaux.models.mirage.MirageRunner",
    "api":          "geoaux.models.openai_api.OpenAIAPIRunner",
}


def get_runner(model_key):
    """Lazy-import and return the runner class for a given model key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    dotted = MODEL_REGISTRY[model_key]
    module_path, cls_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)
