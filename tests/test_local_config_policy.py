from pathlib import Path


def test_runtime_example_exists() -> None:
    assert Path("configs/local/runtime.example.yaml").exists()


def test_gitignore_ignores_local_runtime_configs() -> None:
    content = Path(".gitignore").read_text()
    assert "configs/local/*.yaml" in content
    assert "!configs/local/*.example.yaml" in content
