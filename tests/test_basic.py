from importlib.metadata import version


def test_distribution_has_version():
    assert version("bacpredict") is not None


def test_top_level_packages_importable():
    import bacpredict.engine.download  # noqa: F401
    import bacpredict.engine.embedding  # noqa: F401
    import bacpredict.engine.finetune  # noqa: F401
    import kleb_ast  # noqa: F401
    import kleb_iso_source  # noqa: F401
    import pangena_predict  # noqa: F401
    import tb_ast  # noqa: F401
