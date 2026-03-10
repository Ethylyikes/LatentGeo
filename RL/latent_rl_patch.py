# Custom sitecustomize patch for RL (vllm==0.8.5).
# This is only for RL codes, not for SFT and inference. Keep disabled with
# LATENT_PATCH_DISABLE=1 when you do not want monkey patches applied.

import os, sys, importlib, importlib.util, importlib.machinery, inspect
# Backward-compatible imports (some versions used relative imports here)
try:
    from verl.models.transformers.flash_attention_utils import flash_attention_forward  # noqa: F401
    from verl.models.transformers.qwen2_vl import qwen2_vl_attn_forward  # noqa: F401
except Exception:
    # Not fatal for sitecustomize; only used by older patches
    pass

# Allow disabling this monkey patch when users want to use their own transformers impl.
DISABLE_PATCH = os.environ.get("LATENT_PATCH_DISABLE", "0") == "1"

print(f"[sitecustomize] imported from {__file__}", file=sys.stderr)


def _load_custom_qwen2_5_module(path: str):
    """Load a user-provided Qwen2.5-VL modeling file and register it as the HF module.

    Set env `LATENT_QWEN2_5_CUSTOM_PATH=/abs/path/to/modeling_qwen2_5_vl.py`.
    This keeps the custom patch pipeline intact (vLLM hooks, etc.) while swapping
    the modeling implementation to your own file, avoiding partial code copies.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LATENT_QWEN2_5_CUSTOM_PATH not found: {path}")

    hf_key = "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
    module_name = hf_key  # ensure relative imports (e.g., ...activations) resolve

    # Make sure parent package exists so relative imports work.
    try:
        import transformers.models.qwen2_5_vl  # noqa: F401
    except Exception:
        # fallback to importing transformers to populate subpackages
        import transformers  # noqa: F401

    loader = importlib.machinery.SourceFileLoader(module_name, path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)

    print(f"[Latent RL patch] Using custom Qwen2.5-VL module from {path}", file=sys.stderr)
    return module

def patch_qwen_latent():
    custom_path = os.environ.get("LATENT_QWEN2_5_CUSTOM_PATH")
    if custom_path:
        _load_custom_qwen2_5_module(custom_path)
        print("[Latent RL patch] Skipping bundled override; custom module will be used.", file=sys.stderr)
        return

    # Import official and in-repo implementations
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q_official
    import latent_models.transformers.latent_modeling_qwen2_5_vl as q_latent

    off_cls = q_official.Qwen2_5_VLForConditionalGeneration
    mon_cls = q_latent.Qwen2_5_VLForConditionalGeneration

    # Debug: check signatures before patch
    try:
        print(
            "[Latent RL patch] official forward sig before:",
            inspect.signature(off_cls.forward),
            file=sys.stderr,
        )
        print(
            "[Latent RL patch] bundled forward sig:",
            inspect.signature(mon_cls.forward),
            file=sys.stderr,
        )
    except Exception:
        pass

    # In-place monkey patch: copy methods from the bundled class to the official class
    off_cls.forward = mon_cls.forward
    #off_cls.__init__ = mon_cls.__init__

    # Debug: check signature after patch
    try:
        print(
            "[Latent RL patch] official forward sig after:",
            inspect.signature(off_cls.forward),
            file=sys.stderr,
        )
    except Exception:
        pass

    print(
        "[Latent RL patch] Patched methods of Qwen2_5_VLForConditionalGeneration in-place",
        file=sys.stderr,
    )


def patch():
    if DISABLE_PATCH:
        print("[Latent RL patch] disabled by LATENT_PATCH_DISABLE=1", file=sys.stderr)
        return
    print("[sitecustomize] patch() called", file=sys.stderr)

    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"

    workspace = os.path.abspath(".")
    old_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
    os.environ["LATENT_START_ID"] = "151666"
    os.environ["LATENT_END_ID"] = "151667"
    os.environ["AVT_LATENT_HOOK_BIN"] = "1"

    sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module(
        "latent_models.vllm.latent_gpu_model_runner"
    )

    patch_qwen_latent()
    '''sys.modules[
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
    ] = importlib.import_module(
        "latent_models.transformers.latent_modeling_qwen2_5_vl"
    )'''

    print("[Latent RL patch] vllm & transformers patched", file=sys.stderr)


if not DISABLE_PATCH:
    patch()
else:
    print("[Latent RL patch] skipping patch()", file=sys.stderr)
