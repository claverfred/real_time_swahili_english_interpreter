import pytest

from src.config import ConfigError, available_profiles, load_config
from src.config.loader import cli_overrides, deep_merge, env_overrides

REQUIRED_PROFILES = {
    "accuracy", "balanced", "realtime", "low_vram", "cpu", "offline", "colab",
    "local_windows", "hpc", "hpc_finetune", "hpc_evaluate", "cloud_fallback",
}


def test_all_required_profiles_exist():
    assert REQUIRED_PROFILES <= set(available_profiles())


@pytest.mark.parametrize("name", sorted(REQUIRED_PROFILES | {"default"}))
def test_every_profile_validates(name):
    cfg = load_config(name, environ={})
    assert cfg.asr.model and cfg.mt.model and cfg.tts.voice
    assert cfg.asr.revision and cfg.mt.revision and cfg.tts.revision
    assert cfg.asr.license and cfg.mt.license


def test_inheritance_chain_local_windows_inherits_cpu():
    cfg = load_config("local_windows", environ={})
    assert cfg.asr.compute_type == "int8"      # from cpu
    assert cfg.tts.backend == "edge"           # own override
    assert cfg.vad.min_speech_ms == 250        # from default


def test_precedence_cli_beats_env_beats_profile():
    env = {"SWI__ASR__BEAM_SIZE": "3"}
    assert load_config("balanced", environ=env).asr.beam_size == 3
    assert load_config("balanced", ["asr.beam_size=7"], environ=env).asr.beam_size == 7


def test_env_short_forms_and_profile_env():
    cfg = load_config(None, environ={"SWI_PROFILE": "cpu", "SWI_DEVICE": "cpu"})
    assert cfg.profile == "cpu" and cfg.environment.device == "cpu"


def test_override_parsing_types():
    out = cli_overrides(["asr.beam_size=2", "mt.translate_partials=true",
                         "asr.fallback_models=[base, tiny]", "mt.glossary_path=null"])
    assert out["asr"]["beam_size"] == 2
    assert out["mt"]["translate_partials"] is True
    assert out["asr"]["fallback_models"] == ["base", "tiny"]
    assert out["mt"]["glossary_path"] is None
    assert env_overrides({"SWI__VAD__THRESHOLD": "0.3"}) == {"vad": {"threshold": 0.3}}


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}} and base["a"]["b"] == 1


def test_unknown_profile_and_bad_values_rejected():
    with pytest.raises(ConfigError, match="Unknown profile"):
        load_config("nope", environ={})
    with pytest.raises(ConfigError):
        load_config("balanced", ["asr.beam_size=0"], environ={})
    with pytest.raises(ConfigError):
        load_config("balanced", ["asr.not_a_field=1"], environ={})
    with pytest.raises(ConfigError, match="overlap_s"):
        load_config("balanced", ["asr.overlap_s=2.0"], environ={})
    with pytest.raises(ConfigError, match="must look like"):
        load_config("balanced", ["asr.beam_size"], environ={})


def test_privacy_guard_blocks_cloud_for_restricted_data():
    with pytest.raises(ConfigError, match="forbids cloud"):
        load_config("offline", ["tts.backend=edge", "fallback.allow_cloud=true"], environ={})


def test_cloud_backend_requires_explicit_opt_in():
    with pytest.raises(ConfigError, match="allow_cloud"):
        load_config("balanced", ["tts.backend=edge"], environ={})
    cfg = load_config("cloud_fallback", environ={})
    assert cfg.uses_cloud() == ["edge"]


def test_voice_cloning_refused():
    with pytest.raises(ConfigError, match="voice_cloning"):
        load_config("balanced", ["tts.voice_cloning=true"], environ={})


def test_env_var_expansion_in_paths():
    cfg = load_config("hpc", environ={"SCRATCH_DIR": "/scratch/u1"})
    assert cfg.storage.cache_dir == "/scratch/u1/swahili_interpreter/cache"
    cfg = load_config("hpc", environ={})
    assert cfg.storage.cache_dir == "./swahili_interpreter/cache"
