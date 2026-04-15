"""Pydantic models matching the Zylora OpenAPI spec."""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class GpuType(str, enum.Enum):
    """Available GPU types."""

    T4 = "t4"
    L4 = "l4"
    RTX4090 = "rtx4090"
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    L40S = "l40s"
    H100 = "h100"
    H200 = "h200"
    B200 = "b200"
    MI300X = "mi300x"


class Runtime(str, enum.Enum):
    """Supported runtimes."""

    PYTHON310 = "python310"
    PYTHON311 = "python311"
    PYTHON312 = "python312"
    PYTHON313 = "python313"
    NODE20 = "node20"
    NODE22 = "node22"
    CUSTOM = "custom"


class InvocationStatus(str, enum.Enum):
    """Status of an invocation or async job."""

    QUEUED = "queued"
    COLD_STARTING = "cold_starting"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Visibility(str, enum.Enum):
    """Function visibility."""

    PUBLIC = "public"
    PRIVATE = "private"
    UNLISTED = "unlisted"


class RoutingStrategy(str, enum.Enum):
    """GPU routing strategy."""

    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    RELIABILITY_OPTIMIZED = "reliability_optimized"


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Structured error from the API."""

    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Top-level error wrapper."""

    error: ErrorDetail


class AsyncJobResponse(BaseModel):
    """Returned by POST /v1/functions/{id}/invoke/async."""

    job_id: str
    status: InvocationStatus
    created_at: datetime | None = None


class AsyncJobResult(BaseModel):
    """Returned by GET /v1/functions/{id}/invoke/{job_id}."""

    job_id: str
    status: InvocationStatus
    output: object | None = None
    error: ErrorDetail | None = None
    duration_ms: int | None = None
    cost_cents: int | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class BatchResultItem(BaseModel):
    """Single item in a batch (/map) response."""

    index: int
    status: InvocationStatus
    output: object | None = None
    error: str | None = None
    duration_ms: int | None = None
    cost_cents: int | None = None


class BatchResponse(BaseModel):
    """Returned by POST /v1/functions/{id}/map."""

    results: list[BatchResultItem]
    total_cost_cents: int = 0
    # Computed from results when not supplied by the server.
    total: int = 0
    succeeded: int = 0
    failed: int = 0

    @model_validator(mode="after")
    def _compute_stats(self) -> "BatchResponse":
        if not self.total and self.results:
            self.total = len(self.results)
            self.succeeded = sum(
                1 for r in self.results if r.status == InvocationStatus.COMPLETED
            )
            self.failed = self.total - self.succeeded
        return self


class FunctionInfo(BaseModel):
    """Minimal function metadata."""

    id: str
    name: str
    slug: str
    gpu_type: GpuType
    runtime: Runtime
    entry_point: str
    min_instances: int = 0
    max_instances: int = 10
    concurrency: int = 1
    timeout_seconds: int = 300
    visibility: Visibility = Visibility.PRIVATE
    active_deployment_id: str | None = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Decorator config model (validated at decoration time)
# ---------------------------------------------------------------------------


class FunctionConfig(BaseModel):
    """Configuration attached to a decorated function via ``@zylora.fn()``."""

    gpu: GpuType
    name: str | None = None
    packages: list[str] = Field(default_factory=list)
    model: str | None = None
    timeout: int = Field(default=300, ge=1, le=3600)
    min_instances: int = Field(default=0, ge=0)
    max_instances: int = Field(default=10, ge=1)
    concurrency: int = Field(default=1, ge=1)
    image: str | None = None
    runtime: Runtime = Runtime.PYTHON312
    routing: RoutingStrategy = RoutingStrategy.COST_OPTIMIZED
    visibility: Visibility = Visibility.PRIVATE
