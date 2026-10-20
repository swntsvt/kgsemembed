"""Tests that the lock file pins the environment results are produced in (#141)."""

from importlib import metadata
from pathlib import Path

import pytest

from kgsemembed.pipeline.run_experiment import _PROVENANCE_DISTRIBUTIONS

_LOCK_FILE = Path(__file__).resolve().parents[1] / "requirements-lock.txt"


def _locked_versions() -> dict[str, str]:
    lines = _LOCK_FILE.read_text(encoding="utf-8").splitlines()
    pins = [line.split("==", 1) for line in lines if "==" in line and not line.startswith("#")]
    return {name.lower(): version for name, version in pins}


def test_lock_file_pins_every_requirement_exactly() -> None:
    lines = _LOCK_FILE.read_text(encoding="utf-8").splitlines()
    requirements = [line for line in lines if line and not line.startswith("#")]
    assert requirements and all("==" in line for line in requirements)


@pytest.mark.parametrize("distribution", sorted(_PROVENANCE_DISTRIBUTIONS.values()))
def test_installed_versions_match_the_lock_file(distribution: str) -> None:
    assert _locked_versions()[distribution] == metadata.version(distribution)
