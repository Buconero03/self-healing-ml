def test_smoke_imports():
    """Controlla che i principali moduli vengano importati senza errori."""
    import src.evaluate as ev
    import src.train as tr
    assert hasattr(ev, "evaluate_model")
    assert hasattr(tr, "main")

