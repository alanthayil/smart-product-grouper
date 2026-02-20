"""Smoke test: pipeline modules are importable."""


def test_import_stubs():
    """All pipeline modules can be imported."""
    from src import ingest, normalize, extract, cluster, canonicalize, evaluate
    assert callable(ingest.ingest)
    assert callable(normalize.normalize)
    assert callable(extract.extract)
    assert callable(cluster.cluster)
    assert callable(canonicalize.canonicalize)
    assert callable(evaluate.evaluate)
