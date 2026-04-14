from kgsemembed.pipeline.run_experiment import main


def test_smoke_pipeline_main() -> None:
    assert main() == 0
