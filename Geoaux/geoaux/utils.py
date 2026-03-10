import json
import os

import yaml


def get_api_key():
    """Resolve API key with DashScope-first fallback."""
    return (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("GEOAUX_API_KEY", "").strip()
    )


def get_base_url(default=""):
    """Resolve API base URL with DashScope-first fallback."""
    return (
        os.environ.get("DASHSCOPE_BASE_URL", "").strip()
        or os.environ.get("GEOAUX_BASE_URL", "").strip()
        or default
    )


def get_default_model(default="qwen-plus"):
    """Resolve default model from env vars."""
    return (
        os.environ.get("DASHSCOPE_MODEL", "").strip()
        or os.environ.get("GEOAUX_MODEL", "").strip()
        or default
    )


def load_config(path=None):
    """Load config from defaults.yaml, with env-var overrides."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config", "defaults.yaml")
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_dir = os.path.dirname(path)
    project_root = os.path.abspath(os.path.join(config_dir, ".."))

    benchmark_root = cfg.get("benchmark_root", ".")
    if not os.path.isabs(benchmark_root):
        benchmark_root = os.path.abspath(os.path.join(project_root, benchmark_root))
    cfg["benchmark_root"] = benchmark_root

    for key in ("data_path", "output_dir", "eval_output_dir"):
        value = cfg.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value):
            cfg[key] = os.path.abspath(os.path.join(project_root, value))

    api_cfg = cfg.setdefault("api", {})

    api_key = get_api_key()
    if api_key:
        api_cfg["api_key"] = api_key
    else:
        api_cfg.setdefault("api_key", "")

    base_url = get_base_url(api_cfg.get("base_url", ""))
    if base_url:
        api_cfg["base_url"] = base_url

    api_cfg["default_model"] = get_default_model(api_cfg.get("default_model", "qwen-plus"))

    return cfg


def resolve_image_path(rel_path, benchmark_root):
    """Resolve a possibly-relative image path against the benchmark root."""
    if not rel_path:
        return None
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(benchmark_root, rel_path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
