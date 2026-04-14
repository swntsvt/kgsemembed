import subprocess
import os
import sys


def _env_with_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src:{existing}" if existing else "src"
    return env


def test_smoke_pipeline_main_default() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kgsemembed.pipeline.run_experiment"],
        env=_env_with_pythonpath(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_smoke_pipeline_main_with_overrides() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kgsemembed.pipeline.run_experiment",
            "model=all-minilm-l6-v2",
            "verbalisation=v1",
            "experiment.seed=7",
        ],
        env=_env_with_pythonpath(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
