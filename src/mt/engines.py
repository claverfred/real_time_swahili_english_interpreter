"""Seq2seq MT engines: CTranslate2 (fast, int8 on CPU) and transformers (reference).

Both expose ``translate_batch(sources, prefixes, beam_size, max_tokens) -> list[str]``.
A *prefix* is English text forced as the start of the output (used for context: the previous
translation). The returned text excludes the prefix.
"""

from __future__ import annotations

import time
from pathlib import Path

from .convert import MtError, ensure_ct2_model, hf_snapshot
from .langs import lang_code


class Seq2SeqEngine:
    engine_name = "base"

    def __init__(self, model_id: str, family: str, *, src_lang: str, tgt_lang: str,
                 revision: str = "main", device: str = "cpu", compute_type: str = "int8",
                 threads: int = 0, gpu_index: int = 0, cache_root: str | Path = "cache",
                 local_files_only: bool = False) -> None:
        self.model_id, self.family, self.revision = model_id, family, revision
        self.src_code = lang_code(family, src_lang)
        self.tgt_code = lang_code(family, tgt_lang)
        self.device, self.compute_type, self.threads, self.gpu_index = device, compute_type, threads, gpu_index
        self.cache_root, self.local_files_only = Path(cache_root), local_files_only
        self.tokenizer = None
        self.load_info: dict[str, object] = {}

    # -- tokenizer helpers -------------------------------------------------------------------
    def _load_tokenizer(self, path: Path) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise MtError("the MT tokenizer needs `transformers` (pip install -e .[mt])") from exc
        kwargs = {"src_lang": self.src_code} if self.src_code else {}
        if self.tgt_code:
            kwargs["tgt_lang"] = self.tgt_code
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(path), **kwargs)
        except Exception as exc:
            raise MtError(f"could not load tokenizer for {self.model_id}: {exc}") from exc

    def _tgt_lang_token(self) -> list[str]:
        if not self.tgt_code:
            return []
        if self.family == "m2m":
            return [self.tokenizer.lang_code_to_token[self.tgt_code]]
        return [self.tgt_code]

    def _prefix_tokens(self, prefix: str | None) -> list[str]:
        if not prefix:
            return []
        ids = self.tokenizer(text_target=prefix, add_special_tokens=False)["input_ids"]
        return self.tokenizer.convert_ids_to_tokens(ids)

    def _decode_tokens(self, tokens: list[str]) -> str:
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    # -- interface ---------------------------------------------------------------------------
    def load(self) -> dict[str, object]:
        raise NotImplementedError

    def translate_batch(self, sources: list[str], prefixes: list[str | None] | None = None, *,
                        beam_size: int = 4, max_tokens: int = 256) -> list[str]:
        raise NotImplementedError

    def warmup(self) -> float:
        t0 = time.perf_counter()
        self.translate_batch(["Habari ya leo."], beam_size=1, max_tokens=16)
        return (time.perf_counter() - t0) * 1000


class Ct2Engine(Seq2SeqEngine):
    engine_name = "ctranslate2"

    def load(self) -> dict[str, object]:
        try:
            import ctranslate2
        except ImportError as exc:
            raise MtError("ctranslate2 is not installed") from exc
        path, prov = ensure_ct2_model(self.model_id, revision=self.revision, family=self.family,
                                      compute_type=self.compute_type, cache_root=self.cache_root,
                                      local_files_only=self.local_files_only)
        t0 = time.perf_counter()
        try:
            self.translator = ctranslate2.Translator(
                str(path), device=self.device, device_index=self.gpu_index,
                compute_type=self.compute_type, intra_threads=self.threads, inter_threads=1)
        except Exception as exc:
            raise MtError(f"CTranslate2 could not load {self.model_id}: {exc}") from exc
        self._load_tokenizer(path / "hf_tokenizer")
        self.load_info = {
            "engine": self.engine_name, "model": self.model_id, "family": self.family,
            "path": str(path), "resolved_revision": prov.get("resolved_revision"),
            "stored_quantization": prov.get("quantization"), "device": self.device,
            "compute_type": self.compute_type, "threads": self.threads,
            "fetch_ms": prov.get("fetch_ms"), "convert_ms": prov.get("convert_ms"),
            "reused_conversion": prov.get("reused"), "load_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        return self.load_info

    def translate_batch(self, sources, prefixes=None, *, beam_size=4, max_tokens=256):
        if not sources:
            return []
        prefixes = prefixes or [None] * len(sources)
        src_tokens = [self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(s)) for s in sources]
        tgt_prefix = [self._tgt_lang_token() + self._prefix_tokens(p) for p in prefixes]
        use_prefix = any(tgt_prefix)
        results = self.translator.translate_batch(
            src_tokens, target_prefix=tgt_prefix if use_prefix else None, beam_size=beam_size,
            max_decoding_length=max_tokens + max(len(p) for p in tgt_prefix), max_input_length=1024)
        return [self._decode_tokens(r.hypotheses[0][len(p):]) for r, p in zip(results, tgt_prefix, strict=True)]


class HfEngine(Seq2SeqEngine):
    engine_name = "transformers"

    def load(self) -> dict[str, object]:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM
        except ImportError as exc:
            raise MtError("mt.engine=transformers needs transformers and torch") from exc
        t0 = time.perf_counter()
        snap, sha = hf_snapshot(self.model_id, self.revision, self.cache_root / "huggingface" / "hub",
                                self.local_files_only)
        fetch_ms = (time.perf_counter() - t0) * 1000
        dtype = {"float16": torch.float16, "int8_float16": torch.float16,
                 "bfloat16": torch.bfloat16, "int8_bfloat16": torch.bfloat16}.get(self.compute_type, torch.float32)
        if self.device == "cpu":
            dtype = torch.float32
            if self.threads:
                torch.set_num_threads(self.threads)
        t1 = time.perf_counter()
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(snap), torch_dtype=dtype)
            self.model.to(f"cuda:{self.gpu_index}" if self.device == "cuda" else "cpu").eval()
        except Exception as exc:
            raise MtError(f"transformers could not load {self.model_id}: {exc}") from exc
        self._load_tokenizer(snap)
        self._torch = torch
        self.load_info = {
            "engine": self.engine_name, "model": self.model_id, "family": self.family, "path": str(snap),
            "resolved_revision": sha, "device": self.device, "dtype": str(dtype).replace("torch.", ""),
            "threads": self.threads, "fetch_ms": round(fetch_ms, 1),
            "load_ms": round((time.perf_counter() - t1) * 1000, 1),
        }
        return self.load_info

    def _generate(self, enc, beam_size, max_tokens, decoder_input_ids=None):
        kwargs = {"num_beams": beam_size, "max_new_tokens": max_tokens, "do_sample": False}
        if decoder_input_ids is not None:
            kwargs["decoder_input_ids"] = decoder_input_ids
        elif self.tgt_code:
            kwargs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(self._tgt_lang_token()[0])
        with self._torch.inference_mode():
            return self.model.generate(**enc, **kwargs)

    def translate_batch(self, sources, prefixes=None, *, beam_size=4, max_tokens=256):
        if not sources:
            return []
        dev = self.model.device
        if not prefixes or not any(prefixes):
            enc = self.tokenizer(sources, return_tensors="pt", padding=True, truncation=True).to(dev)
            out = self._generate(enc, beam_size, max_tokens)
            return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]
        texts = []
        for src, prefix in zip(sources, prefixes, strict=True):
            enc = self.tokenizer([src], return_tensors="pt", truncation=True).to(dev)
            start = [self.model.config.decoder_start_token_id]
            start += self.tokenizer.convert_tokens_to_ids(self._tgt_lang_token() + self._prefix_tokens(prefix))
            dec = self._torch.tensor([start], device=dev)
            out = self._generate(enc, beam_size, max_tokens, decoder_input_ids=dec)
            texts.append(self.tokenizer.decode(out[0][len(start):], skip_special_tokens=True).strip())
        return texts


def make_engine(engine: str, **kwargs) -> Seq2SeqEngine:
    if engine == "ctranslate2":
        return Ct2Engine(**kwargs)
    if engine == "transformers":
        return HfEngine(**kwargs)
    raise MtError(f"unknown MT engine {engine!r}")
