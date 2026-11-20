"""Secret lookup: environment first, then Colab Secrets. Values are never logged."""

from __future__ import annotations

import os

from .detect import is_colab


def get_secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if is_colab():
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            return userdata.get(name)
        except Exception:
            return None
    return None


def has_secret(name: str) -> bool:
    return bool(get_secret(name))


def load_dotenv(path: str = ".env") -> int:
    """Minimal .env reader (no extra dependency). Existing env vars win. Returns count loaded."""
    if not os.path.isfile(path):
        return 0
    loaded = 0
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded += 1
    return loaded
