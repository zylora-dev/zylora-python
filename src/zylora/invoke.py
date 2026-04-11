"""Standalone invocation helpers (without the decorator).

These are thin wrappers around :class:`zylora.client.Zylora` for quick
one-off calls::

    from zylora.invoke import invoke, stream

    result = invoke("embed", {"text": "hello"})

    for token in stream("generate", {"prompt": "Hi"}):
        print(token, end="")
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from zylora.client import AsyncJob, Zylora
from zylora.types import BatchResponse

_default_client: Zylora | None = None


def _client() -> Zylora:
    global _default_client
    if _default_client is None:
        _default_client = Zylora()
    return _default_client


def invoke(function: str, input: Any = None) -> Any:
    """Synchronously invoke a deployed function by name."""
    return _client().invoke(function, input)


def batch(
    function: str,
    inputs: list[Any],
    *,
    concurrency: int = 10,
) -> BatchResponse:
    """Batch-invoke a function with multiple inputs."""
    return _client().batch(function, inputs, concurrency=concurrency)


def stream(function: str, input: Any = None) -> Iterator[str]:
    """Stream tokens from a deployed function (SSE)."""
    return _client().stream(function, input)


def invoke_async(function: str, input: Any = None) -> AsyncJob:
    """Start an async invocation — returns a job handle."""
    return _client().invoke_async(function, input)
