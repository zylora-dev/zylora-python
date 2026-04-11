"""Tests for serialization helpers."""

from __future__ import annotations

from zylora.serialization import deserialize_output, serialize_input


def test_serialize_json_passthrough() -> None:
    """JSON-serializable inputs should pass through unchanged."""
    payload, ct = serialize_input({"text": "hello", "count": 42})
    assert ct == "application/json"
    assert payload == {"text": "hello", "count": 42}


def test_serialize_string() -> None:
    payload, ct = serialize_input("hello")
    assert ct == "application/json"
    assert payload == "hello"


def test_serialize_non_json_uses_pickle() -> None:
    """Non-JSON objects should be cloudpickled and base64-encoded."""
    data = {1, 2, 3}  # sets aren't JSON-serializable
    payload, ct = serialize_input(data)
    assert ct == "application/json"
    assert "__zylora_pickle__" in payload


def test_roundtrip_pickle() -> None:
    """Pickle serialization should round-trip correctly."""
    original = {1, 2, 3}
    payload, _ = serialize_input(original)
    result = deserialize_output(payload)
    assert result == original


def test_deserialize_plain_json() -> None:
    """Plain JSON should be returned as-is."""
    data = {"embedding": [0.1, 0.2]}
    assert deserialize_output(data) == data


def test_deserialize_non_dict() -> None:
    """Non-dict data should pass through."""
    assert deserialize_output(42) == 42
    assert deserialize_output("hello") == "hello"
