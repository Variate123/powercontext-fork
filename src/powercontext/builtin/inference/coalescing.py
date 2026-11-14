# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Cross-call batching and bounded caching for embedding models."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel, QueryEmbeddingModel, embed_query


_EmbeddingKind = Literal["document", "query"]
_Key = tuple[_EmbeddingKind, str]


@dataclass(slots=True)
class _Entry:
    future: asyncio.Future["_ItemResult"]
    pending: bool = True


@dataclass(frozen=True, slots=True)
class _ItemResult:
    vector: EmbeddingVector
    batch_id: int
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Share in-flight text embeddings, batch them, and cache successful vectors.

    Each invocation keeps its own ordering and usage accounting. Shared work is
    shielded from waiter cancellation so cancelling one caller cannot cancel a
    provider request needed by another caller.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)):
            raise ValueError("coalesce_ms must be a non-negative finite number")
        if not math.isfinite(float(coalesce_ms)) or float(coalesce_ms) < 0:
            raise ValueError("coalesce_ms must be a non-negative finite number")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._model = model
        self.profile = model.profile
        self.max_batch_size = max_batch_size
        self.coalesce_ms = float(coalesce_ms)
        self.cache_size = cache_size
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[_Key, EmbeddingVector] = OrderedDict()
        self._entries: dict[_Key, _Entry] = {}
        self._pending: list[_Key] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._next_batch_id = 0

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return await self._embed_kind(texts, "document")

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return await self._embed_kind(texts, "query")

    async def _embed_kind(self, texts: tuple[str, ...], kind: _EmbeddingKind) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=())
        unique = tuple(dict.fromkeys(texts))
        cached: dict[str, EmbeddingVector] = {}
        waiting: dict[_Key, _Entry] = {}
        async with self._lock:
            for text in unique:
                key = (kind, text)
                if self.cache_size:
                    vector = self._cache.get(key)
                    if vector is not None:
                        self._cache.move_to_end(key)
                        cached[text] = vector
                        continue
                entry = self._entries.get(key)
                if entry is None:
                    entry = _Entry(asyncio.get_running_loop().create_future())
                    self._entries[key] = entry
                    self._pending.append(key)
                waiting[key] = entry
            if self._pending and (self._flush_task is None or self._flush_task.done()):
                self._flush_task = asyncio.create_task(self._flush_after_window())

        results: dict[_Key, _ItemResult] = {}
        batches: dict[int, InferenceUsage] = {}
        for key, entry in waiting.items():
            item = await asyncio.shield(entry.future)
            results[key] = item
            batches.setdefault(item.batch_id, item.usage)
        vectors = tuple(
            cached[text] if text in cached else results[(kind, text)].vector
            for text in texts
        )
        return EmbeddingResult(vectors=vectors, usage=_sum_usage(tuple(batches.values())))

    async def _flush_after_window(self) -> None:
        await asyncio.sleep(self.coalesce_ms / 1000.0)
        async with self._lock:
            keys = tuple(self._pending)
            self._pending.clear()
            self._flush_task = None
            if not keys:
                return
            for start in range(0, len(keys), self.max_batch_size):
                batch = keys[start : start + self.max_batch_size]
                self._next_batch_id += 1
                batch_id = self._next_batch_id
                entries = tuple((key, self._entries[key]) for key in batch)
                for _, entry in entries:
                    entry.pending = False
                asyncio.create_task(self._run_batch(batch_id, entries))

    async def _run_batch(self, batch_id: int, entries: tuple[tuple[_Key, _Entry], ...]) -> None:
        keys = tuple(key for key, _ in entries)
        texts = tuple(key[1] for key in keys)
        kind = keys[0][0]
        try:
            result = await (embed_query(self._model, texts) if kind == "query" else self._model.embed(texts))
            if len(result.vectors) != len(texts):
                raise ValueError("embedding provider returned the wrong vector count")
            items = tuple(_ItemResult(tuple(vector), batch_id, result.usage) for vector in result.vectors)
        except BaseException as error:
            async with self._lock:
                for key, entry in entries:
                    if self._entries.get(key) is entry:
                        self._entries.pop(key, None)
                    if not entry.future.done():
                        entry.future.set_exception(error)
            return

        async with self._lock:
            if self.cache_size:
                for (key, entry), item in zip(entries, items, strict=True):
                    self._cache[key] = item.vector
                    self._cache.move_to_end(key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            for (key, entry), item in zip(entries, items, strict=True):
                if self._entries.get(key) is entry:
                    self._entries.pop(key, None)
                if not entry.future.done():
                    entry.future.set_result(item)


def _sum_usage(usages: tuple[InferenceUsage, ...]) -> InferenceUsage:
    if not usages:
        return InferenceUsage(requests=0)
    return InferenceUsage(
        requests=sum(item.requests for item in usages),
        input_tokens=_sum_optional(item.input_tokens for item in usages),
        output_tokens=_sum_optional(item.output_tokens for item in usages),
    )


def _sum_optional(values: Any) -> int | None:
    concrete = tuple(value for value in values if value is not None)
    return None if not concrete else sum(concrete)


__all__ = ["CoalescingEmbeddingModel"]
