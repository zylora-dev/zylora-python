"""Zylora exception hierarchy.

Every error includes a ``request_id`` for support lookup when available.
"""

from __future__ import annotations


class ZyloraError(Exception):
    """Base exception for all Zylora SDK errors."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        self.message = message
        self.request_id = request_id
        super().__init__(message)

    def __str__(self) -> str:
        base = self.message
        if self.request_id:
            base += f" [request_id: {self.request_id}]"
        return base


class AuthenticationError(ZyloraError):
    """API key is missing, invalid, or expired."""


class InsufficientCreditsError(ZyloraError):
    """Account does not have enough credits to complete the invocation."""


class FunctionNotFoundError(ZyloraError):
    """The requested function does not exist or has been deleted."""


class RateLimitError(ZyloraError):
    """Too many requests — slow down.

    Attributes:
        retry_after: Seconds to wait before retrying.
    """

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        retry_after: int = 0,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.retry_after = retry_after


class ProviderError(ZyloraError):
    """The upstream GPU provider returned an error."""


class ZyloraTimeoutError(ZyloraError):
    """Function execution exceeded the configured timeout."""


class BuildError(ZyloraError):
    """Container build failed during deployment."""


class ValidationError(ZyloraError):
    """Invalid input — failed schema or parameter validation."""


class NoCapacityError(ZyloraError):
    """No GPU capacity available to serve the request."""
