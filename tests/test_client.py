"""Tests for the HTTP client — error mapping and invocation."""

from __future__ import annotations

import httpx
import pytest
import respx

from zylora.client import Zylora, _map_error
from zylora.exceptions import (
    AuthenticationError,
    FunctionNotFoundError,
    InsufficientCreditsError,
    RateLimitError,
    ZyloraError,
)


def test_map_error_401() -> None:
    resp = httpx.Response(
        401,
        json={"error": {"code": "unauthorized", "message": "Bad key"}},
    )
    err = _map_error(resp)
    assert isinstance(err, AuthenticationError)
    assert "Bad key" in str(err)


def test_map_error_402() -> None:
    resp = httpx.Response(
        402,
        json={
            "error": {
                "code": "insufficient_credits",
                "message": "Not enough credits",
                "request_id": "abc-123",
            }
        },
    )
    err = _map_error(resp)
    assert isinstance(err, InsufficientCreditsError)
    assert err.request_id == "abc-123"


def test_map_error_404() -> None:
    resp = httpx.Response(
        404,
        json={"error": {"code": "not_found", "message": "Function not found"}},
    )
    err = _map_error(resp)
    assert isinstance(err, FunctionNotFoundError)


def test_map_error_429_retry_after() -> None:
    resp = httpx.Response(
        429,
        json={"error": {"code": "rate_limit_exceeded", "message": "Slow down"}},
        headers={"retry-after": "30"},
    )
    err = _map_error(resp)
    assert isinstance(err, RateLimitError)
    assert err.retry_after == 30


def test_map_error_unknown_status() -> None:
    resp = httpx.Response(418, json={"error": {"code": "teapot", "message": "I'm a teapot"}})
    err = _map_error(resp)
    assert isinstance(err, ZyloraError)
    assert "teapot" in str(err)


@respx.mock
def test_invoke_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful invocation should return the deserialized result."""
    monkeypatch.setenv("ZYLORA_API_KEY", "zy_test_key")
    monkeypatch.setenv("ZYLORA_API_URL", "https://mock.api")

    respx.post("https://mock.api/v1/functions/embed/invoke").mock(
        return_value=httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]}),
    )

    zy = Zylora(api_key="zy_test_key", api_url="https://mock.api")
    result = zy.invoke("embed", {"text": "hello"})
    assert result == {"embedding": [0.1, 0.2, 0.3]}


@respx.mock
def test_invoke_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """401 should raise AuthenticationError."""
    monkeypatch.setenv("ZYLORA_API_KEY", "zy_bad_key")
    monkeypatch.setenv("ZYLORA_API_URL", "https://mock.api")

    respx.post("https://mock.api/v1/functions/embed/invoke").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"code": "unauthorized", "message": "Invalid API key"}},
        ),
    )

    zy = Zylora(api_key="zy_bad_key", api_url="https://mock.api")
    with pytest.raises(AuthenticationError, match="Invalid API key"):
        zy.invoke("embed", {"text": "hello"})
