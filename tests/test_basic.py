from importlib.metadata import version


def test_distribution_has_version():
    assert version("bacpredict") is not None


def test_top_level_packages_importable():
    import bacpredict.apps.kleb  # noqa: F401
    import bacpredict.apps.tb  # noqa: F401
    import bacpredict.engine.download  # noqa: F401
    import bacpredict.engine.embedding  # noqa: F401
    import bacpredict.engine.finetune  # noqa: F401
    import kleb_iso_source  # noqa: F401
