"""HTTP client for the Zylora API.

Wraps ``httpx`` with authentication, retries, error mapping, and
serialization. Both sync and async interfaces are provided.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

import zylora
from zylora.config import Config, resolve_config
from zylora.exceptions import (
    AuthenticationError,
    FunctionNotFoundError,
    InsufficientCreditsError,
    NoCapacityError,
    RateLimitError,
    ValidationError,
    ZyloraError,
    ZyloraTimeoutError,
)
from zylora.serialization import deserialize_output, serialize_input
from zylora.types import (
    AsyncJobResponse,
    AsyncJobResult,
    BatchResponse,
    ErrorResponse,
    InvocationStatus,
)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5  # seconds


class Zylora:
    """High-level Zylora API client.

    Example::

        from zylora import Zylora

        zy = Zylora()                       # auto-discover API key
        result = zy.invoke("embed", {"text": "hello"})
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        if api_key or api_url:
            from zylora.config import configure

            configure(api_key=api_key, api_url=api_url)

        self._timeout = timeout
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Synchronous helpers
    # ------------------------------------------------------------------

    def _get_sync(self, config: Config) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=config.api_url,
                headers=self._headers(config),
                timeout=self._timeout,
            )
        return self._sync_client

    def _get_async(self, config: Config) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=config.api_url,
                headers=self._headers(config),
                timeout=self._timeout,
            )
        return self._async_client

    @staticmethod
    def _headers(config: Config) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {config.api_key}",
            "User-Agent": f"zylora-python/{zylora.__version__}",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def invoke(self, function: str, input: Any = None) -> Any:
        """Synchronously invoke a deployed function.

        Args:
            function: Function name or ID.
            input: JSON-serializable input passed to the handler.

        Returns:
            The deserialized function result.
        """
        config = resolve_config()
        payload, _ = serialize_input(input)
        resp = self._request_with_retry(
            config,
            "POST",
            f"/v1/functions/{function}/invoke",
            json=payload,
        )
        return deserialize_output(resp.json())

    def batch(
        self,
        function: str,
        inputs: list[Any],
        *,
        concurrency: int = 10,
    ) -> BatchResponse:
        """Invoke a function once per input in parallel.

        Args:
            function: Function name or ID.
            inputs: List of inputs (max 1000).
            concurrency: Max parallel invocations (1–100).
        """
        config = resolve_config()
        serialized = [serialize_input(i)[0] for i in inputs]
        resp = self._request_with_retry(
            config,
            "POST",
            f"/v1/functions/{function}/map",
            json={"inputs": serialized, "concurrency": concurrency},
        )
        return BatchResponse.model_validate(resp.json())

    def stream(self, function: str, input: Any = None) -> Iterator[str]:
        """Stream response tokens via SSE.

        Yields string chunks as they arrive from the function.
        """
        config = resolve_config()
        payload, _ = serialize_input(input)
        client = self._get_sync(config)
        with client.stream(
            "POST",
            f"/v1/functions/{function}/invoke/stream",
            json=payload,
        ) as resp:
            _check_response(resp)
            yield from _parse_sse_sync(resp)

    def invoke_async(self, function: str, input: Any = None) -> AsyncJob:
        """Fire-and-forget invocation. Returns a job handle.

        Use ``job.result()`` to block until the result is ready.
        """
        config = resolve_config()
        payload, _ = serialize_input(input)
        resp = self._request_with_retry(
            config,
            "POST",
            f"/v1/functions/{function}/invoke/async",
            json=payload,
        )
        data = AsyncJobResponse.model_validate(resp.json())
        return AsyncJob(function=function, job_id=data.job_id, client=self)

    def get_job_result(self, function: str, job_id: str) -> AsyncJobResult:
        """Poll for an async job result."""
        config = resolve_config()
        resp = self._request_with_retry(
            config,
            "GET",
            f"/v1/functions/{function}/invoke/{job_id}",
        )
        return AsyncJobResult.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def ainvoke(self, function: str, input: Any = None) -> Any:
        """Async version of :meth:`invoke`."""
        config = resolve_config()
        payload, _ = serialize_input(input)
        resp = await self._arequest_with_retry(
            config,
            "POST",
            f"/v1/functions/{function}/invoke",
            json=payload,
        )
        return deserialize_output(resp.json())

    async def abatch(
        self,
        function: str,
        inputs: list[Any],
        *,
        concurrency: int = 10,
    ) -> BatchResponse:
        """Async version of :meth:`batch`."""
        config = resolve_config()
        serialized = [serialize_input(i)[0] for i in inputs]
        resp = await self._arequest_with_retry(
            config,
            "POST",
            f"/v1/functions/{function}/map",
            json={"inputs": serialized, "concurrency": concurrency},
        )
        return BatchResponse.model_validate(resp.json())

    async def astream(
        self, function: str, input: Any = None
    ) -> AsyncIterator[str]:
        """Async version of :meth:`stream`."""
        config = resolve_config()
        payload, _ = serialize_input(input)
        client = self._get_async(config)
        async with client.stream(
            "POST",
            f"/v1/functions/{function}/invoke/stream",
            json=payload,
        ) as resp:
            _check_response(resp)
            async for chunk in _parse_sse_async(resp):
                yield chunk

    async def ainvoke_async(
        self, function: str, input: Any = None
    ) -> AsyncJob:
        """Async version of :meth:`invoke_async`."""
        config = resolve_config()
        payload, _ = serialize_input(input)
        resp = await self._arequest_with_retry(
            config,
            "POST",
            f"/v1/functions/{function}/invoke/async",
            json=payload,
        )
        data = AsyncJobResponse.model_validate(resp.json())
        return AsyncJob(function=function, job_id=data.job_id, client=self)

    async def aget_job_result(self, function: str, job_id: str) -> AsyncJobResult:
        """Async poll for a job result."""
        config = resolve_config()
        resp = await self._arequest_with_retry(
            config,
            "GET",
            f"/v1/functions/{function}/invoke/{job_id}",
        )
        return AsyncJobResult.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _request_with_retry(
        self,
        config: Config,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        client = self._get_sync(config)
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.request(method, path, **kwargs)
                if resp.status_code < 500:
                    _check_response(resp)
                    return resp
                last_exc = _map_error(resp)
            except httpx.TransportError as exc:
                last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
        raise last_exc or ZyloraError("Request failed after retries")

    async def _arequest_with_retry(
        self,
        config: Config,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        client = self._get_async(config)
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.request(method, path, **kwargs)
                if resp.status_code < 500:
                    _check_response(resp)
                    return resp
                last_exc = _map_error(resp)
            except httpx.TransportError as exc:
                last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))
        raise last_exc or ZyloraError("Request failed after retries")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close underlying HTTP connections."""
        if self._sync_client:
            self._sync_client.close()
        if self._async_client:
            # Fire-and-forget close in async context
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_client.aclose())
            except RuntimeError:
                asyncio.run(self._async_client.aclose())

    def __enter__(self) -> Zylora:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ======================================================================
# Async job handle
# ======================================================================


class AsyncJob:
    """Handle for an async invocation.

    Use :meth:`result` to block until the result is ready, or :meth:`aresult`
    for the async variant.
    """

    def __init__(self, *, function: str, job_id: str, client: Zylora) -> None:
        self.function = function
        self.job_id = job_id
        self._client = client

    def result(self, *, timeout: float = 300.0, poll_interval: float = 1.0) -> Any:
        """Block until the job completes and return the result."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self._client.get_job_result(self.function, self.job_id)
            if job.status == InvocationStatus.COMPLETED:
                return deserialize_output(job.result)
            if job.status in (
                InvocationStatus.FAILED,
                InvocationStatus.TIMEOUT,
                InvocationStatus.CANCELLED,
            ):
                msg = job.error.message if job.error else f"Job {job.status.value}"
                raise ZyloraError(msg)
            time.sleep(poll_interval)
        raise ZyloraTimeoutError(f"Job {self.job_id} did not complete within {timeout}s")

    async def aresult(
        self, *, timeout: float = 300.0, poll_interval: float = 1.0
    ) -> Any:
        """Async variant of :meth:`result`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = await self._client.aget_job_result(self.function, self.job_id)
            if job.status == InvocationStatus.COMPLETED:
                return deserialize_output(job.result)
            if job.status in (
                InvocationStatus.FAILED,
                InvocationStatus.TIMEOUT,
                InvocationStatus.CANCELLED,
            ):
                msg = job.error.message if job.error else f"Job {job.status.value}"
                raise ZyloraError(msg)
            await asyncio.sleep(poll_interval)
        raise ZyloraTimeoutError(f"Job {self.job_id} did not complete within {timeout}s")


# ======================================================================
# Helpers
# ======================================================================

_ERROR_MAP: dict[int, type[ZyloraError]] = {
    401: AuthenticationError,
    402: InsufficientCreditsError,
    404: FunctionNotFoundError,
    408: ZyloraTimeoutError,
    422: ValidationError,
    429: RateLimitError,
    503: NoCapacityError,
}


def _map_error(resp: httpx.Response) -> ZyloraError:
    """Convert an HTTP error response into the appropriate exception."""
    request_id: str | None = resp.headers.get("x-request-id")

    try:
        body = ErrorResponse.model_validate(resp.json())
        message = body.error.message
        request_id = body.error.request_id or request_id
    except Exception:
        message = resp.text or f"HTTP {resp.status_code}"

    exc_cls = _ERROR_MAP.get(resp.status_code, ZyloraError)

    if exc_cls is RateLimitError:
        retry_after = int(resp.headers.get("retry-after", "0"))
        return RateLimitError(message, request_id=request_id, retry_after=retry_after)

    return exc_cls(message, request_id=request_id)


def _check_response(resp: httpx.Response) -> None:
    """Raise if the response indicates an error."""
    if resp.status_code >= 400:
        raise _map_error(resp)


def _parse_sse_sync(resp: httpx.Response) -> Iterator[str]:
    """Parse SSE events from a sync streaming response."""
    import json as json_mod

    for line in resp.iter_lines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                return
            try:
                data = json_mod.loads(payload)
                if "chunk" in data:
                    yield data["chunk"]
            except json_mod.JSONDecodeError:
                yield payload
        elif line.startswith("event: done"):
            return
        elif line.startswith("event: error"):
            # Next data line will have the error details
            continue


async def _parse_sse_async(resp: httpx.Response) -> AsyncIterator[str]:
    """Parse SSE events from an async streaming response."""
    import json as json_mod

    async for line in resp.aiter_lines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                return
            try:
                data = json_mod.loads(payload)
                if "chunk" in data:
                    yield data["chunk"]
            except json_mod.JSONDecodeError:
                yield payload
        elif line.startswith("event: done"):
            return
        elif line.startswith("event: error"):
            continue
