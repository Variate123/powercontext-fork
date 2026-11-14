"""Canonical wire format for incremental prepared-context transfer."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any


class ContextPackError(ValueError):
    """Raised when a context pack or delta violates the wire contract."""


_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_for_items(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": _PACK_SCHEMA, "items": [_encode(item) for item in items]}
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _encode(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise ContextPackError("pack items must be bytes")
    return base64.b64encode(value).decode("ascii")


def _decode(value: object) -> bytes:
    if not isinstance(value, str):
        raise ContextPackError("wire item must be a base64 string")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise ContextPackError("wire item is not valid base64") from error


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ContextPackError("invalid context digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ContextPackError("invalid context digest") from error
    return value


@dataclass(frozen=True, slots=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes]) -> ContextPack:
        try:
            normalized = tuple(items)
        except TypeError as error:
            raise ContextPackError("items must be a bytes sequence") from error
        for item in normalized:
            if not isinstance(item, bytes):
                raise ContextPackError("pack items must be bytes")
        digest = _digest_for_items(normalized)
        payload = {"schema": _PACK_SCHEMA, "items": [_encode(item) for item in normalized], "digest": digest, "byte_count": 0}
        count = 0
        for _ in range(8):
            payload["byte_count"] = count
            encoded = _canonical(payload)
            next_count = len(encoded)
            if next_count == count:
                return cls(normalized, digest, count)
            count = next_count
        raise ContextPackError("could not determine pack byte count")

    @classmethod
    def from_content(cls, content: str | bytes, *, chunk_bytes: int = 512) -> ContextPack:
        if isinstance(content, str):
            raw = content.encode("utf-8")
        elif isinstance(content, bytes):
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ContextPackError("content must be valid UTF-8") from error
        else:
            raise ContextPackError("content must be text or bytes")
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be positive")
        items: list[bytes] = []
        offset = 0
        while offset < len(raw):
            end = min(offset + chunk_bytes, len(raw))
            while end > offset and end < len(raw) and (raw[end] & 0xC0) == 0x80:
                end -= 1
            if end == offset:
                raise ContextPackError("chunk_bytes cannot contain one UTF-8 code point")
            items.append(raw[offset:end])
            offset = end
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        payload = {
            "schema": _PACK_SCHEMA,
            "items": [_encode(item) for item in self.items],
            "digest": self.digest,
            "byte_count": self.byte_count,
        }
        encoded = _canonical(payload)
        if len(encoded) != self.byte_count:
            raise ContextPackError("pack byte_count does not match wire bytes")
        return encoded

    @classmethod
    def from_bytes(cls, value: bytes) -> ContextPack:
        payload = _load_object(value)
        if set(payload) != {"schema", "items", "digest", "byte_count"} or payload["schema"] != _PACK_SCHEMA:
            raise ContextPackError("invalid context pack fields")
        if type(payload["byte_count"]) is not int or payload["byte_count"] < 0:
            raise ContextPackError("invalid pack byte_count")
        raw_items = payload["items"]
        if not isinstance(raw_items, list):
            raise ContextPackError("pack items must be an array")
        items = tuple(_decode(item) for item in raw_items)
        digest = _digest(payload["digest"])
        expected = _digest_for_items(items)
        if digest != expected:
            raise ContextPackError("pack digest mismatch")
        pack = cls(items, digest, payload["byte_count"])
        if pack.byte_count != len(value):
            raise ContextPackError("pack byte_count mismatch")
        if pack.to_bytes() != value:
            raise ContextPackError("pack is not canonical")
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("between requires ContextPack values")
        prefix = 0
        for left, right in zip(base.items, target.items):
            if left != right:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    def to_bytes(self) -> bytes:
        payload = {
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_encode(item) for item in self.appended],
        }
        return _canonical(payload)

    @classmethod
    def from_bytes(cls, value: bytes) -> ContextPackDelta:
        payload = _load_object(value)
        if set(payload) != {"schema", "base_digest", "target_digest", "prefix_items", "appended"}:
            raise ContextPackError("invalid context delta fields")
        if payload["schema"] != _DELTA_SCHEMA:
            raise ContextPackError("invalid context delta schema")
        prefix = payload["prefix_items"]
        if type(prefix) is not int or prefix < 0:
            raise ContextPackError("invalid context delta prefix")
        appended = payload["appended"]
        if not isinstance(appended, list):
            raise ContextPackError("delta appended must be an array")
        return cls(_digest(payload["base_digest"]), _digest(payload["target_digest"]), prefix, tuple(_decode(item) for item in appended))


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta application requires pack and delta")
    if base.digest != delta.base_digest:
        raise ContextPackError("base digest mismatch")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds base items")
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("target digest mismatch")
    return target


def _load_object(value: bytes) -> dict[str, Any]:
    if not isinstance(value, bytes):
        raise ContextPackError("wire value must be bytes")
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("wire value is not canonical JSON") from error
    if not isinstance(decoded, dict):
        raise ContextPackError("wire value must be a JSON object")
    return decoded


__all__ = ["ContextPack", "ContextPackDelta", "ContextPackError", "apply_context_pack_delta"]
