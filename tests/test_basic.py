from importlib.metadata import version


def test_distribution_has_version():
    assert version("bacpredict") is not None


def test_top_level_packages_importable():
    import bacpredict.apps.kleb
    import bacpredict.apps.tb
    import bacpredict.engine.download
    import bacpredict.engine.embedding
    import bacpredict.engine.finetune  # noqa: F401
    import kleb_iso_source  # noqa: F401
