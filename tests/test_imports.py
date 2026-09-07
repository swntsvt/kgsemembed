def test_imports() -> None:
    import kgsemembed
    import kgsemembed.pipeline
    import kgsemembed.embeddings
    import kgsemembed.verbalisation
    import kgsemembed.evaluation
    import kgsemembed.utils
    import kgsemembed.datasets
    import kgsemembed.candidates

    assert kgsemembed is not None
