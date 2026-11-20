"""Profile loading with inheritance, environment overrides, and CLI overrides.

Precedence (highest first), per the system spec section 8:
    CLI ``--set key=value``  ->  environment ``SWI__SECTION__KEY``  ->  profile YAML chain
Device/platform "auto" values are resolved afterwards by :mod:`src.platform.resolve`.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from .schema import InterpreterConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("SWI_CONFIG_DIR", REPO_ROOT / "configs"))
ENV_PREFIX = "SWI__"
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised for unknown profiles, inheritance cycles, or malformed overrides."""


def available_profiles(config_dir: Path = CONFIG_DIR) -> list[str]:
    return sorted(p.stem for p in config_dir.glob("*.yaml"))


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at top level")
    return data


def load_profile_dict(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load a profile and its ``inherits`` chain into one merged dict."""
    chain: list[dict[str, Any]] = []
    seen: list[str] = []
    current: str | None = name
    while current:
        if current in seen:
            raise ConfigError(f"Inheritance cycle: {' -> '.join([*seen, current])}")
        path = config_dir / f"{current}.yaml"
        if not path.is_file():
            raise ConfigError(
                f"Unknown profile {current!r}. Available: {', '.join(available_profiles(config_dir))}"
            )
        seen.append(current)
        data = _load_yaml(path)
        chain.append(data)
        current = data.pop("inherits", None)
    merged: dict[str, Any] = {}
    for layer in reversed(chain):
        merged = deep_merge(merged, layer)
    return merged


def _parse_scalar(raw: str) -> Any:
    # YAML parsing gives ints, floats, bools, null, and [lists] for free.
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = [k for k in dotted.split(".") if k]
    if not keys:
        raise ConfigError(f"Empty override key in {dotted!r}")
    node = data
    for key in keys[:-1]:
        nxt = node.setdefault(key, {})
        if not isinstance(nxt, dict):
            raise ConfigError(f"Override {dotted!r}: {key!r} is not a section")
        node = nxt
    node[keys[-1]] = value


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """``SWI__ASR__BEAM_SIZE=3`` -> ``{"asr": {"beam_size": 3}}``.

    Also honours the short forms ``SWI_DEVICE`` and ``SWI_PLATFORM``.
    """
    environ = os.environ if environ is None else environ
    out: dict[str, Any] = {}
    for key, raw in environ.items():
        if key.startswith(ENV_PREFIX):
            dotted = key[len(ENV_PREFIX):].lower().replace("__", ".")
            _set_path(out, dotted, _parse_scalar(raw))
    for short, dotted in (("SWI_DEVICE", "environment.device"), ("SWI_PLATFORM", "environment.platform")):
        if environ.get(short):
            _set_path(out, dotted, environ[short])
    return out


def cli_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"Override {pair!r} must look like section.key=value")
        key, raw = pair.split("=", 1)
        _set_path(out, key.strip(), _parse_scalar(raw.strip()))
    return out


def _expand(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in environ and environ[name] != "":
                return environ[name]
            if default is not None:
                return default
            raise ConfigError(f"Environment variable ${{{name}}} is required by the config but unset")
        return _VAR.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v, environ) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, environ) for v in value]
    return value


def load_config(
    profile: str | None = None,
    overrides: Iterable[str] = (),
    environ: Mapping[str, str] | None = None,
    config_dir: Path = CONFIG_DIR,
) -> InterpreterConfig:
    """Load and validate a configuration. Device "auto" is left for platform resolution."""
    environ = os.environ if environ is None else environ
    name = profile or environ.get("SWI_PROFILE") or "balanced"
    data = load_profile_dict(name, config_dir)
    data = deep_merge(data, env_overrides(environ))
    data = deep_merge(data, cli_overrides(overrides))
    data = _expand(data, environ)
    try:
        return InterpreterConfig.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError -> clearer top-level message
        raise ConfigError(f"Invalid configuration for profile {name!r}:\n{exc}") from exc
