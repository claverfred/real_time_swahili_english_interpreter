"""Build the configured translator, walking engines and ``mt.fallback_models`` (always logged)."""

from __future__ import annotations

import gc
import importlib.util

from ..config.schema import InterpreterConfig
from ..platform.resolve import FallbackRefused
from ..text.glossary import Glossary
from ..utils.manifest import FallbackLog
from .convert import MtError
from .engines import make_engine
from .langs import family_of
from .translator import Translator


def _engine_order(requested: str) -> list[str]:
    if requested != "auto":
        return [requested]
    order = []
    if importlib.util.find_spec("ctranslate2"):
        order.append("ctranslate2")
    if importlib.util.find_spec("torch"):
        order.append("transformers")
    return order or ["ctranslate2"]


def create_translator(cfg: InterpreterConfig, fallbacks: FallbackLog | None = None,
                      local_files_only: bool = False) -> Translator:
    """``cfg`` must already be runtime-resolved (concrete device and compute type)."""
    mt, env = cfg.mt, cfg.environment
    if mt.backend not in {"nllb", "marian", "m2m"}:
        raise MtError(f"MT backend {mt.backend!r} is not implemented in this release")
    glossary = Glossary.load(mt.glossary_path) if mt.glossary_path else None
    models = [mt.model, *[m for m in mt.fallback_models if m != mt.model]]
    engines = _engine_order(mt.engine)
    errors: list[str] = []
    first = True
    for model in models:
        family = family_of(model, default=mt.backend) if model != mt.model else mt.backend
        revision = mt.revision if model == mt.model else "main"
        for engine_name in engines:
            engine = make_engine(engine_name, model_id=model, family=family, src_lang=mt.src_lang,
                                 tgt_lang=mt.tgt_lang, revision=revision, device=env.device,
                                 compute_type=mt.compute_type, threads=env.cpu_threads,
                                 gpu_index=env.gpu_index, cache_root=cfg.storage.cache_dir,
                                 local_files_only=local_files_only)
            try:
                engine.load()
            except MtError as exc:
                errors.append(f"{model} [{engine_name}]: {exc}")
                gc.collect()
                if cfg.fallback.policy == "fail":
                    raise FallbackRefused(f"MT {model} [{engine_name}] failed and fallback.policy=fail: {exc}") from exc
                first = False
                continue
            if not first and fallbacks is not None:
                fallbacks.add("mt", f"{mt.model} [{engines[0]}]", f"{model} [{engine_name}]", " | ".join(errors))
            return Translator(engine, mt, glossary)
    raise MtError("no MT model could be loaded: " + " | ".join(errors))
