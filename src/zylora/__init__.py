"""Zylora — Serverless GPU functions.

Deploy ML models with a decorator, scale to zero, pay per GPU-second.

Usage::

    import zylora

    @zylora.fn(gpu="H100")
    def embed(text: str) -> list[float]:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        return model.encode(text).tolist()

    # Local execution (development)
    result = embed("hello")

    # Remote execution (deployed)
    result = embed.remote("hello")
"""

from zylora.client import Zylora
from zylora.config import configure
from zylora.decorator import fn
from zylora.exceptions import (
    AuthenticationError,
    BuildError,
    FunctionNotFoundError,
    InsufficientCreditsError,
    ProviderError,
    RateLimitError,
    ValidationError,
    ZyloraError,
    ZyloraTimeoutError,
)

__version__ = "0.1.0"

__all__ = [
    "AuthenticationError",
    "BuildError",
    "FunctionNotFoundError",
    "InsufficientCreditsError",
    "ProviderError",
    "RateLimitError",
    "ValidationError",
    "Zylora",
    "ZyloraError",
    "ZyloraTimeoutError",
    "__version__",
    "configure",
    "fn",
]
