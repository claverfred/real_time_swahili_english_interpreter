"""Structured logging with secret redaction."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# Token-like patterns that must never appear in logs.
_TOKEN_PATTERNS = [re.compile(r"hf_[A-Za-z0-9]{20,}"), re.compile(r"sk-[A-Za-z0-9_-]{20,}")]
_KEY_VALUE = re.compile(r"(?i)\b(api[_-]?key|token|secret|password|authorization)(\s*[=:]\s*)\S+")
_SENSITIVE_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "AZURE_SPEECH_KEY", "OPENAI_API_KEY")


def redact(text: str) -> str:
    for name in _SENSITIVE_ENV:
        value = os.environ.get(name)
        if value and len(value) >= 6:
            text = text.replace(value, "***")
    text = _KEY_VALUE.sub(r"\1\2***", text)
    for pat in _TOKEN_PATTERNS:
        text = pat.sub("***", text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.getMessage()))
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", json_format: bool = False, log_file: Path | None = None) -> None:
    root = logging.getLogger("swi")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False
    fmt: logging.Formatter = (
        JsonFormatter() if json_format else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(RedactingFilter())
        root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"swi.{name}")
