"""Model families and their language-code conventions.

Config uses FLORES-200 codes (``swh_Latn``/``eng_Latn``); each family gets its own form.
"""

from __future__ import annotations

FAMILIES = ("nllb", "marian", "m2m")

_TO_M2M = {"swh_Latn": "sw", "eng_Latn": "en"}


def family_of(model_id: str, default: str = "nllb") -> str:
    low = model_id.lower()
    if "nllb" in low:
        return "nllb"
    if "opus-mt" in low or "marian" in low:
        return "marian"
    if "m2m100" in low or "m2m_100" in low:
        return "m2m"
    return default


def lang_code(family: str, flores_code: str) -> str | None:
    if family == "nllb":
        return flores_code
    if family == "m2m":
        try:
            return _TO_M2M[flores_code]
        except KeyError as exc:
            raise ValueError(f"no M2M-100 code known for {flores_code}") from exc
    return None  # Marian models are single-direction; no language token


def tokenizer_files(family: str) -> tuple[str, ...]:
    common = ("tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json")
    return common + {
        "nllb": ("tokenizer.json", "sentencepiece.bpe.model"),
        "marian": ("source.spm", "target.spm", "vocab.json"),
        "m2m": ("sentencepiece.bpe.model", "vocab.json"),
    }[family]
