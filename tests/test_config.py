"""Tests for configuration and auth discovery."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from zylora import config as config_mod
from zylora.config import _extract_token, resolve_config
from zylora.exceptions import AuthenticationError


@pytest.fixture(autouse=True)
def _clear_overrides() -> None:
    """Reset module-level config overrides between tests."""
    config_mod._overrides.clear()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZYLORA_API_KEY env var should be discovered."""
    monkeypatch.setenv("ZYLORA_API_KEY", "zy_test_abc123")
    monkeypatch.delenv("ZYLORA_API_URL", raising=False)
    cfg = resolve_config()
    assert cfg.api_key == "zy_test_abc123"
    assert cfg.api_url == "https://api.zylora.dev"


def test_api_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZYLORA_API_URL env var should override the default endpoint."""
    monkeypatch.setenv("ZYLORA_API_KEY", "zy_test_abc123")
    monkeypatch.setenv("ZYLORA_API_URL", "http://localhost:8080")
    cfg = resolve_config()
    assert cfg.api_url == "http://localhost:8080"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should raise AuthenticationError when no key is found."""
    monkeypatch.delenv("ZYLORA_API_KEY", raising=False)
    # Mock _read_token_from_config to return None
    with (
        patch("zylora.config._read_token_from_config", return_value=None),
        pytest.raises(AuthenticationError, match="No API key found"),
    ):
        resolve_config()


def test_extract_token_from_toml(tmp_path: Path) -> None:
    """Should read auth.token from a TOML config file."""
    config = tmp_path / "config.toml"
    config.write_text('[auth]\ntoken = "zy_live_fromfile"\n')
    assert _extract_token(config) == "zy_live_fromfile"


def test_extract_token_missing_file(tmp_path: Path) -> None:
    """Non-existent file should return None, not raise."""
    assert _extract_token(tmp_path / "nope.toml") is None


def test_extract_token_no_auth_section(tmp_path: Path) -> None:
    """TOML without auth section should return None."""
    config = tmp_path / "config.toml"
    config.write_text("[other]\nkey = 1\n")
    assert _extract_token(config) is None
