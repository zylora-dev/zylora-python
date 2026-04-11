"""Tests for the @zylora.fn() decorator."""

from __future__ import annotations

import pytest

import zylora
from zylora.decorator import ZyloraFunction, _build_payload
from zylora.types import FunctionConfig, GpuType


def test_decorator_preserves_local_call() -> None:
    """Calling a decorated function locally should just run the original."""

    @zylora.fn(gpu="H100")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_decorator_attaches_config() -> None:
    """The decorator should attach __zylora_config__ with correct values."""

    @zylora.fn(gpu="a100_80gb", timeout=60, min_instances=1)
    def predict(x: float) -> float:
        return x * 2.0

    assert isinstance(predict, ZyloraFunction)
    cfg: FunctionConfig = predict.__zylora_config__  # type: ignore[attr-defined]
    assert cfg.gpu == GpuType.A100_80GB
    assert cfg.timeout == 60
    assert cfg.min_instances == 1
    assert cfg.max_instances == 10  # default


def test_decorator_uses_function_name_as_default() -> None:
    @zylora.fn(gpu="t4")
    def my_cool_function() -> None:
        pass

    assert isinstance(my_cool_function, ZyloraFunction)
    assert my_cool_function.function_name == "my_cool_function"


def test_decorator_custom_name() -> None:
    @zylora.fn(gpu="t4", name="custom-embed")
    def embed() -> None:
        pass

    assert embed.function_name == "custom-embed"


def test_decorator_rejects_invalid_gpu() -> None:
    with pytest.raises(ValueError, match="Invalid GPU type"):

        @zylora.fn(gpu="invalid_gpu_999")
        def oops() -> None:
            pass


def test_decorator_case_insensitive_gpu() -> None:
    """GPU type should be case-insensitive."""

    @zylora.fn(gpu="H100")
    def func() -> None:
        pass

    assert func._config.gpu == GpuType.H100


def test_build_payload_single_arg() -> None:
    assert _build_payload(("hello",), {}) == "hello"


def test_build_payload_multiple_args() -> None:
    assert _build_payload((1, 2, 3), {}) == [1, 2, 3]


def test_build_payload_kwargs_only() -> None:
    assert _build_payload((), {"text": "hi"}) == {"text": "hi"}


def test_build_payload_mixed() -> None:
    result = _build_payload(("a",), {"b": 1})
    assert result == {"args": ["a"], "kwargs": {"b": 1}}


def test_build_payload_empty() -> None:
    assert _build_payload((), {}) is None
