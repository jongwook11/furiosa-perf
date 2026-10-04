"""Tests for the device names used in result paths."""

import pytest

from furiosa_perf.runner.env import device_label, gpu_product_name, npu_arch_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NVIDIA H100 80GB HBM3", "H100-80GB-HBM3"),
        ("NVIDIA H100 NVL", "H100-NVL"),
        ("NVIDIA H100 PCIe", "H100-PCIe"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX-PRO-6000-Blackwell-Server-Edition"),
        ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "RTX-PRO-6000-Blackwell-MaxQ-Workstation-Edition"),
    ],
)
def test_gpu_product_name_keeps_the_words_that_tell_variants_apart(raw: str, expected: str) -> None:
    """SXM/NVL/PCIe H100s and the RTX PRO 6000 editions get distinct names."""
    assert gpu_product_name(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [("Arch.Rngd", "RNGD"), ("Rngd", "RNGD"), ("Arch.RngdMax", "RNGD-MAX")])
def test_npu_arch_name(raw: str, expected: str) -> None:
    """RNGD variants are named after their arch, whether or not the enum prefix is present."""
    assert npu_arch_name(raw) == expected


def test_device_label_marks_only_non_default_power_settings() -> None:
    """Default limits and the Performance governor add nothing; a lowered limit or PowerSave is shown."""
    default = {"power_limit_w": 700.0, "default_power_limit_w": 700.0}
    lowered = {"power_limit_w": 500.0, "default_power_limit_w": 700.0}

    assert device_label("H100-80GB-HBM3", [default, default]) == "H100-80GB-HBM3"
    assert device_label("H100-80GB-HBM3", [default, lowered]) == "H100-80GB-HBM3@500W"
    assert device_label("RNGD", [{"governor": "Performance"}]) == "RNGD"
    assert device_label("RNGD", [{"governor": "PowerSave"}]) == "RNGD@powersave"
