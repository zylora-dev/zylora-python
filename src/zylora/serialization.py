"""Input / output serialization helpers.

Simple JSON-serializable inputs go as-is.  Complex Python objects (numpy arrays,
torch tensors, etc.) are serialized with ``cloudpickle`` and sent as base64.
"""

from __future__ import annotations

import base64
import json
from typing import Any


def serialize_input(obj: Any) -> tuple[Any, str]:
    """Serialize *obj* for transport.

    Returns ``(payload, content_type)`` where *payload* is JSON-safe.
    """
    try:
        # Fast path: if it round-trips through JSON it's already fine.
        json.dumps(obj)
        return obj, "application/json"
    except (TypeError, ValueError, OverflowError):
        pass

    # Fallback to cloudpickle for non-JSON types.
    import cloudpickle

    pickled = cloudpickle.dumps(obj)
    encoded = base64.b64encode(pickled).decode("ascii")
    return {"__zylora_pickle__": encoded}, "application/json"


def deserialize_output(data: Any) -> Any:
    """Deserialize a response payload.

    If the payload contains a ``__zylora_pickle__`` key it's decoded with
    ``cloudpickle``; otherwise returned verbatim.
    """
    if isinstance(data, dict) and "__zylora_pickle__" in data:
        import cloudpickle

        raw = base64.b64decode(data["__zylora_pickle__"])
        return cloudpickle.loads(raw)
    return data
