"""Configuration and auth discovery for the Zylora SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_API_URL = "https://api.zylora.dev"

# Module-level overrides set by ``zylora.configure()``.
_overrides: dict[str, Any] = {}


@dataclass(frozen=True)
class Config:
    """Resolved SDK configuration."""

    api_key: str
    api_url: str = _DEFAULT_API_URL


def configure(
    *,
    api_key: str | None = None,
    api_url: str | None = None,
) -> None:
    """Override default configuration values.

    Call this early in your program if you don't want to rely on env vars /
    config files::

        import zylora
        zylora.configure(api_key="zy_live_abc123")
    """
    if api_key is not None:
        _overrides["api_key"] = api_key
    if api_url is not None:
        _overrides["api_url"] = api_url


def resolve_config() -> Config:
    """Build a :class:`Config` using the standard discovery chain.

    Resolution order for ``api_key``:
      1. ``zylora.configure(api_key=...)`` (code override)
      2. ``ZYLORA_API_KEY`` environment variable
      3. ``~/.zylora/config.toml`` written by ``zy login``
      4. ``.zylora/config.toml`` in project directory

    Resolution order for ``api_url``:
      1. ``zylora.configure(api_url=...)``
      2. ``ZYLORA_API_URL`` environment variable
      3. Default ``https://api.zylora.dev``
    """
    api_key = (
        _overrides.get("api_key")
        or os.environ.get("ZYLORA_API_KEY")
        or _read_token_from_config()
    )
    if not api_key:
        from zylora.exceptions import AuthenticationError

        raise AuthenticationError(
            "No API key found. Set ZYLORA_API_KEY or run `zy login`."
        )

    api_url = (
        _overrides.get("api_url")
        or os.environ.get("ZYLORA_API_URL")
        or _DEFAULT_API_URL
    )

    return Config(api_key=api_key, api_url=api_url)


def _read_token_from_config() -> str | None:
    """Try to read the auth token from config files on disk."""
    # Project-local config first, then global.
    candidates = [
        Path.cwd() / ".zylora" / "config.toml",
        Path.home() / ".zylora" / "config.toml",
    ]

    for path in candidates:
        token = _extract_token(path)
        if token:
            return token
    return None


def _extract_token(path: Path) -> str | None:
    """Read ``auth.token`` from a TOML config file."""
    if not path.is_file():
        return None
    try:
        # Use tomllib (3.11+) or tomli fallback.
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomllib  # type: ignore[import-not-found]
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[import-not-found,no-redef]

        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data.get("auth", {}).get("token") or None  # type: ignore[no-any-return]
    except Exception:
        return None
