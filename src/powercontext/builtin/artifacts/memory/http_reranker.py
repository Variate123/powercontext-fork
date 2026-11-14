"""HTTP adapter for deterministic, batched Memory reranking services."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision, MemoryReranker
from powercontext.builtin.inference.models import InferenceUsage


class HTTPMemoryReranker:
    """Score a complete candidate pool through an offline HTTP endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")
        if not isinstance(client, httpx.AsyncClient):
            raise ValueError("client must be an httpx.AsyncClient")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        self.endpoint = endpoint
        self._client = client
        self.batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        if not candidates:
            raise ValueError("rerank requires at least one candidate")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            payload = {
                "query": query,
                "candidates": [
                    {"rank": start + offset + 1, "text": candidate.text}
                    for offset, candidate in enumerate(batch)
                ],
            }
            response = await self._client.post(self.endpoint, json=payload)
            response.raise_for_status()
            requests += 1
            try:
                decoded = response.json()
            except (ValueError, TypeError) as error:
                raise ValueError("reranker response must be valid JSON") from error
            batch_scores = _validate_response(decoded, {start + offset + 1 for offset in range(len(batch))})
            scores.update(batch_scores)

        selected = tuple(
            rank
            for rank, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        )
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
        )


def _validate_response(value: object, expected_ranks: set[int]) -> dict[int, float]:
    if not isinstance(value, dict) or set(value) != {"scores"}:
        raise ValueError("reranker response must contain exactly scores")
    scores = value["scores"]
    if not isinstance(scores, list) or len(scores) != len(expected_ranks):
        raise ValueError("reranker scores must exactly cover the request batch")
    result: dict[int, float] = {}
    for item in scores:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("reranker score records have an invalid schema")
        rank = item["rank"]
        score = item["score"]
        if type(rank) is not int or rank not in expected_ranks or rank in result:
            raise ValueError("reranker scores contain invalid or duplicate ranks")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("reranker scores must be finite numbers")
        result[rank] = float(score)
    if set(result) != expected_ranks:
        raise ValueError("reranker scores do not cover the request batch")
    return result


__all__ = ["HTTPMemoryReranker"]
