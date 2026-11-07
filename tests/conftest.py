"""Keep generated files and visualization configuration local to each test."""

import pytest

from symergetics.visualization import reset_config


@pytest.fixture(autouse=True)
def isolated_test_environment(tmp_path, monkeypatch):
    """Contain even default outputs after a test calls reset_config()."""
    monkeypatch.chdir(tmp_path)
    reset_config()
    try:
        yield
    finally:
        reset_config()
