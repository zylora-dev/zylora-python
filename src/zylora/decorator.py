"""The ``@zylora.fn()`` decorator.

Attaches GPU function metadata to a Python callable.  The decorated function
retains its original behavior when called locally; use ``.remote()``,
``.map()``, or ``.stream()`` to invoke the deployed version.
"""

from __future__ import annotations

import functools
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, TypeVar, overload

from zylora.client import AsyncJob, Zylora
from zylora.types import (
    BatchResponse,
    FunctionConfig,
    GpuType,
    RoutingStrategy,
    Runtime,
    Visibility,
)

F = TypeVar("F", bound=Callable[..., Any])


class ZyloraFunction:
    """Wrapper around a user function decorated with ``@zylora.fn()``.

    The wrapper preserves local call semantics and adds remote invocation
    methods: ``.remote()``, ``.map()``, ``.stream()``, ``.remote_async()``.
    """

    def __init__(self, func: Callable[..., Any], config: FunctionConfig) -> None:
        self._func = func
        self._config = config
        self._client: Zylora | None = None

        # Preserve the original function's metadata.
        functools.update_wrapper(self, func)

        # Attach config so the CLI / build system can introspect it.
        self.__zylora_config__ = config  # type: ignore[attr-defined]

    @property
    def function_name(self) -> str:
        """The name used to identify this function on the platform."""
        return self._config.name or self._func.__name__

    def _get_client(self) -> Zylora:
        if self._client is None:
            self._client = Zylora()
        return self._client

    # ------------------------------------------------------------------
    # Local execution — just call the original function
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the function locally (for development)."""
        return self._func(*args, **kwargs)

    # ------------------------------------------------------------------
    # Remote invocation (sync)
    # ------------------------------------------------------------------

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the deployed function synchronously.

        Arguments are serialized and sent to the platform.  Returns the
        deserialized result.
        """
        payload = _build_payload(args, kwargs)
        return self._get_client().invoke(self.function_name, payload)

    def map(self, inputs: list[Any], *, concurrency: int = 10) -> BatchResponse:
        """Invoke the function once per input in parallel (batch).

        Args:
            inputs: Up to 1 000 inputs.
            concurrency: Max parallel invocations (1-100).
        """
        payloads = [_build_payload((inp,), {}) for inp in inputs]
        return self._get_client().batch(
            self.function_name, payloads, concurrency=concurrency
        )

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[str]:
        """Stream response tokens from the deployed function (SSE)."""
        payload = _build_payload(args, kwargs)
        return self._get_client().stream(self.function_name, payload)

    def remote_async(self, *args: Any, **kwargs: Any) -> AsyncJob:
        """Fire-and-forget invocation.  Returns a job handle.

        Use ``job.result()`` to block until the result is ready.
        """
        payload = _build_payload(args, kwargs)
        return self._get_client().invoke_async(self.function_name, payload)

    # ------------------------------------------------------------------
    # Remote invocation (async)
    # ------------------------------------------------------------------

    async def aremote(self, *args: Any, **kwargs: Any) -> Any:
        """Async version of :meth:`remote`."""
        payload = _build_payload(args, kwargs)
        return await self._get_client().ainvoke(self.function_name, payload)

    async def amap(
        self, inputs: list[Any], *, concurrency: int = 10
    ) -> BatchResponse:
        """Async version of :meth:`map`."""
        payloads = [_build_payload((inp,), {}) for inp in inputs]
        return await self._get_client().abatch(
            self.function_name, payloads, concurrency=concurrency
        )

    def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        """Async version of :meth:`stream`."""
        payload = _build_payload(args, kwargs)
        return self._get_client().astream(self.function_name, payload)

    async def aremote_async(self, *args: Any, **kwargs: Any) -> AsyncJob:
        """Async version of :meth:`remote_async`."""
        payload = _build_payload(args, kwargs)
        return await self._get_client().ainvoke_async(self.function_name, payload)


# ======================================================================
# The decorator factory
# ======================================================================


@overload
def fn(
    *,
    gpu: str,
    name: str | None = ...,
    packages: list[str] | None = ...,
    model: str | None = ...,
    timeout: int = ...,
    min_instances: int = ...,
    max_instances: int = ...,
    concurrency: int = ...,
    image: str | None = ...,
    runtime: str = ...,
    routing: str = ...,
    visibility: str = ...,
) -> Callable[[F], ZyloraFunction]: ...


@overload
def fn(*, gpu: str) -> Callable[[F], ZyloraFunction]: ...


def fn(
    *,
    gpu: str,
    name: str | None = None,
    packages: list[str] | None = None,
    model: str | None = None,
    timeout: int = 300,
    min_instances: int = 0,
    max_instances: int = 10,
    concurrency: int = 1,
    image: str | None = None,
    runtime: str = "python312",
    routing: str = "cost_optimized",
    visibility: str = "private",
) -> Callable[[F], ZyloraFunction]:
    """Decorator that marks a function as a Zylora GPU function.

    Example::

        @zylora.fn(gpu="H100")
        def embed(text: str) -> list[float]:
            ...

    The decorated function behaves identically when called locally.
    Use ``.remote()`` to invoke the deployed version.
    """

    # Validate GPU type eagerly so typos are caught at import time.
    try:
        gpu_type = GpuType(gpu.lower())
    except ValueError:
        valid = ", ".join(g.value for g in GpuType)
        raise ValueError(f"Invalid GPU type '{gpu}'. Valid options: {valid}") from None

    config = FunctionConfig(
        gpu=gpu_type,
        name=name,
        packages=packages or [],
        model=model,
        timeout=timeout,
        min_instances=min_instances,
        max_instances=max_instances,
        concurrency=concurrency,
        image=image,
        runtime=Runtime(runtime),
        routing=RoutingStrategy(routing),
        visibility=Visibility(visibility),
    )

    def decorator(func: F) -> ZyloraFunction:
        return ZyloraFunction(func, config)

    return decorator


# ======================================================================
# Helpers
# ======================================================================


def _build_payload(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Convert positional + keyword args into a JSON-friendly payload."""
    if args and kwargs:
        return {"args": list(args), "kwargs": kwargs}
    if kwargs:
        return kwargs
    if len(args) == 1:
        return args[0]
    if args:
        return list(args)
    return None
