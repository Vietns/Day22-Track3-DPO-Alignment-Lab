"""Colab / T4 GPU compatibility helpers (sm_75 vs sm_80+)."""
from __future__ import annotations

import os
import subprocess
import sys

import torch


def is_t4_or_older() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8


def setup_colab_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if is_t4_or_older():
        os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "0"


def uninstall_xformers_on_t4() -> None:
    """xformers FA2 backward needs sm_80+; T4 is sm_75."""
    if not is_t4_or_older():
        return
    import importlib.util

    if importlib.util.find_spec("xformers") is None:
        return
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "xformers"], check=False)
    print("T4: uninstalled xformers (use PyTorch SDPA instead)")


def force_sdpa_attn(model) -> None:
    if not is_t4_or_older():
        return
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "sdpa"
    print("T4/V100: attn_implementation → sdpa")


def grad_checkpointing_mode():
    return True if is_t4_or_older() else "unsloth"


def ensure_qwen_chat_template(tokenizer, *, base_model: str | None = None, compute_tier: str = "T4") -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    from transformers import AutoTokenizer

    if base_model and "3B" in base_model:
        ref = "Qwen/Qwen2.5-3B-Instruct"
    elif base_model and "7B" in base_model:
        ref = "Qwen/Qwen2.5-7B-Instruct"
    else:
        ref = "Qwen/Qwen2.5-3B-Instruct" if compute_tier == "T4" else "Qwen/Qwen2.5-7B-Instruct"
    tokenizer.chat_template = AutoTokenizer.from_pretrained(ref).chat_template
    print(f"Patched chat_template from {ref}")


def preflight_colab(*, require_deps_flag: bool = True) -> None:
    """Fail fast before NB1 if Colab setup incomplete."""
    from pathlib import Path
    import importlib.util

    if require_deps_flag:
        flag = Path("/content/.lab22_deps_installed")
        if not flag.exists():
            raise RuntimeError(
                "Deps chưa sẵn sàng.\n"
                "  1) Chạy cell Install (section A)\n"
                "  2) Runtime → Restart session\n"
                "  3) Run all lại từ cell COMPUTE_TIER\n"
                "(Bước Restart bắt buộc sau pip lần đầu.)"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy GPU. Runtime → Change runtime type → T4 GPU → Factory reset.")

    for mod in ("unsloth", "trl", "peft", "bitsandbytes", "datasets"):
        if importlib.util.find_spec(mod) is None:
            raise RuntimeError(
                f"Thiếu package `{mod}`. Restart session sau Install, rồi Run all lại."
            )

    try:
        import unsloth  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "unsloth import failed — thường do chưa Restart sau cell Install.\n"
            "  → Runtime → Restart session → Run all lại từ COMPUTE_TIER"
        ) from exc

    uninstall_xformers_on_t4()
    setup_colab_env()
    gpu = torch.cuda.get_device_properties(0)
    print(f"✓ Preflight OK — {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)")
