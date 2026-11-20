import pytest

from src.config import load_config
from src.platform.cuda import CudaStatus, GpuInfo
from src.platform.detect import detect_platform
from src.platform.resolve import FallbackRefused, resolve_runtime

NO_GPU = CudaStatus(False, "none", notes=["no NVIDIA GPU detected"])
T4 = CudaStatus(True, "torch", gpus=[GpuInfo(0, "Tesla T4", 15360, "7.5")])
A100 = CudaStatus(True, "torch", gpus=[GpuInfo(0, "A100", 81920, "8.0")])


def test_auto_on_cpu_machine_picks_cpu_int8_without_fallback():
    cfg, info, fb = resolve_runtime(load_config("balanced", environ={}), cuda=NO_GPU)
    assert (info.device, info.precision, info.asr_compute_type) == ("cpu", "int8", "int8")
    assert fb.records == []
    assert info.cpu_threads >= 1


def test_requested_cuda_falls_back_to_cpu_and_is_logged():
    cfg, info, fb = resolve_runtime(load_config("accuracy", environ={}), cuda=NO_GPU)
    assert info.device == "cpu" and info.precision == "int8"
    stages = [r.stage for r in fb.records]
    assert stages == ["device", "precision"]
    assert fb.records[0].requested == "cuda" and fb.records[0].selected == "cpu"


def test_fail_policy_refuses_fallback():
    cfg = load_config("accuracy", ["fallback.policy=fail"], environ={})
    with pytest.raises(FallbackRefused):
        resolve_runtime(cfg, cuda=NO_GPU)


def test_bf16_downgraded_on_t4_but_kept_on_a100():
    cfg, info, fb = resolve_runtime(load_config("hpc", environ={}), cuda=T4)
    assert info.precision == "fp16" and fb.records[0].stage == "precision"
    cfg, info, fb = resolve_runtime(load_config("hpc", environ={}), cuda=A100)
    assert info.precision == "bf16" and info.asr_compute_type == "bfloat16" and not fb.records


def test_auto_precision_on_gpu():
    _, info, _ = resolve_runtime(load_config("balanced", environ={}), cuda=T4)
    assert (info.device, info.precision, info.gpu_name) == ("cuda", "fp16", "Tesla T4")


def test_original_config_not_mutated():
    cfg = load_config("balanced", environ={})
    resolve_runtime(cfg, cuda=NO_GPU)
    assert cfg.environment.device == "auto"


def test_platform_detection():
    assert detect_platform({"SLURM_JOB_ID": "1"}) == "hpc"
    assert detect_platform({"COLAB_RELEASE_TAG": "x"}) == "colab"
