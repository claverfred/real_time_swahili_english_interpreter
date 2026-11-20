"""High-level translation: caching, context, masking, glossary and output guards.

Guards never invent content. They flag, retry without the risky option (context or masking),
or collapse runaway repeats. Every flag is reported on the result:

- ``empty_output``: the model returned nothing for non-empty input
- ``length_ratio``: the output is far longer than the input (possible hallucinated completion)
- ``repeat_collapsed``: a runaway n-gram loop in the output was collapsed
- ``context_retry``: the context-prefixed decode looked wrong, so it was redone without context
- ``mask_failed``: a placeholder was dropped by the model, so it was redone unmasked
- ``number_missing`` / ``acronym_missing`` / ``glossary_missing``: preservation checks (name
  preservation is reported in ``preservation`` only)
- ``incomplete_source``: the source was not a finished sentence
"""

from __future__ import annotations

import time
from collections import OrderedDict

from ..asr.filters import collapse_repeats
from ..config.schema import MtConfig
from ..text.entities import check_preservation, mask, number_spans, unmask
from ..text.glossary import Glossary
from ..text.sentences import is_complete
from .engines import Seq2SeqEngine
from .types import MtResult


class Translator:
    def __init__(self, engine: Seq2SeqEngine, cfg: MtConfig, glossary: Glossary | None = None) -> None:
        self.engine, self.cfg = engine, cfg
        self.glossary = glossary or Glossary()
        self._cache: OrderedDict[tuple, tuple[str, list[str]]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def model_name(self) -> str:
        return self.engine.model_id

    @property
    def load_info(self) -> dict[str, object]:
        return self.engine.load_info

    def warmup(self) -> float:
        return self.engine.warmup()

    # -- internals ---------------------------------------------------------------------------
    def _decode(self, sources: list[str], prefixes: list[str | None] | None, beam: int) -> list[str]:
        return self.engine.translate_batch(sources, prefixes, beam_size=beam, max_tokens=self.cfg.max_tokens)

    def _ratio_bad(self, src: str, out: str) -> bool:
        n_src, n_out = len(src.split()), len(out.split())
        return n_out > 6 and n_out > self.cfg.max_length_ratio * max(n_src, 1)

    def _mask_spans(self, text: str):
        spans = []
        if self.cfg.entity_strategy == "mask":
            spans += number_spans(text)
            spans += [(s, e, entry.target) for s, e, entry in self.glossary.find(text)]
        return spans

    def _translate_one(self, text: str, context: list[tuple[str, str]], beam: int) -> tuple[str, list[str]]:
        flags: list[str] = []
        spans = self._mask_spans(text)
        masked = mask(text, spans) if spans else None
        src = masked.text if masked else text

        out = None
        if context:
            ctx_src = " ".join(s for s, _ in context)
            ctx_tgt = " ".join(t for _, t in context)
            out = self._decode([f"{ctx_src} {src}"], [ctx_tgt], beam)[0]
            if not out or self._ratio_bad(src, out):
                flags.append("context_retry")
                out = None
        if out is None:
            out = self._decode([src], None, beam)[0]

        if masked:
            restored, missing = unmask(out, masked.mapping)
            if missing:
                flags.append("mask_failed")
                out = self._decode([text], None, beam)[0]
            else:
                out = restored

        collapsed = collapse_repeats(out)
        if collapsed != out:
            flags.append("repeat_collapsed")
            out = collapsed
        if not out:
            flags.append("empty_output")
        elif self._ratio_bad(text, out):
            flags.append("length_ratio")
        return out, flags

    # -- public ------------------------------------------------------------------------------
    def translate(self, text: str, *, context: list[tuple[str, str]] | None = None,
                  provisional: bool = False, uncertain: bool = False,
                  beam_size: int | None = None) -> MtResult:
        """Translate one Swahili unit. ``context`` is [(swahili, english), ...] oldest first."""
        t0 = time.perf_counter()
        text = " ".join(text.split())
        beam = 1 if provisional else (beam_size or self.cfg.beam_size)
        ctx = list(context or [])[-self.cfg.context_sentences:] if self.cfg.context_sentences else []
        if self.cfg.context_mode == "none" or provisional:
            ctx = []
        base = {"source": text, "model": self.model_name, "engine": self.engine.engine_name,
                "provisional": provisional, "uncertain": uncertain, "context_used": len(ctx)}
        if not text:
            return MtResult(text="", latency_ms=0.0, flags=["empty_source"], **base)

        key = (text, tuple(ctx), beam, self.cfg.entity_strategy)
        cached = key in self._cache
        if cached:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            out, flags = self._cache[key]
        else:
            self.cache_misses += 1
            out, flags = self._translate_one(text, ctx, beam)
            if self.cfg.cache_size:
                self._cache[key] = (out, flags)
                while len(self._cache) > self.cfg.cache_size:
                    self._cache.popitem(last=False)
        flags = list(flags)

        pres = check_preservation(text, out)
        # Names stay report-only: Swahili capitalizes common nouns that are correctly translated
        # ("Kamati" -> "Committee"), so a name flag fired on 44% of FLORES sentences (noise).
        for kind in ("numbers", "acronyms"):
            if pres[kind]["missing"]:
                flags.append(f"{kind[:-1]}_missing")
        gloss_missing = self.glossary.missing_targets(text, out) if len(self.glossary) else []
        if gloss_missing:
            pres["glossary"] = {"missing": gloss_missing}
            flags.append("glossary_missing")
        if not is_complete(text):
            flags.append("incomplete_source")
        return MtResult(text=out, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                        cached=cached, flags=flags, preservation=pres, **base)

    def translate_batch(self, texts: list[str], *, batch_size: int = 16,
                        beam_size: int | None = None) -> list[str]:
        """Context-free batched translation for evaluation (no masking, no cache)."""
        beam = beam_size or self.cfg.beam_size
        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            out += self._decode(texts[i:i + batch_size], None, beam)
        return out
