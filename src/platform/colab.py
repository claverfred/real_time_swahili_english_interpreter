"""Thin Google Colab helpers (expanded in Phase 9)."""

from __future__ import annotations

import os

from .detect import is_colab
from .secrets import get_secret


def mount_drive(mount_point: str = "/content/drive") -> bool:
    if not is_colab():
        return False
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mount_point)
    return True


def hf_login_from_secrets() -> bool:
    """Log in to Hugging Face with the HF_TOKEN Colab Secret, without printing it."""
    token = get_secret("HF_TOKEN")
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except ImportError:
        pass
    return True
