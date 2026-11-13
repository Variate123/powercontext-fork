"""Pi v3 session history parsing and incremental capture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"


class PiSessionInput(BaseModel):
    """A complete Pi v3 JSONL archive and optional selected leaf."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: bytes
    leaf_id: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def require_bytes(cls, value: object) -> bytes:
        if not isinstance(value, bytes):
            raise ValueError("content must be bytes")
        return value

    @field_validator("leaf_id")
    @classmethod
    def validate_leaf(cls, value: str | None) -> str | None:
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("leaf_id must be a non-empty string")
        return value


class PiEventInput(BaseModel):
    """One visible event on a selected Pi history path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured, immutable representation of one Pi event."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_identity(self) -> PiEventSource:
        if self.name != self.source_id:
            raise ValueError("name and source_id must match")
        if self.source_id != pi_event_source_id(self.session_id, self.event_id):
            raise ValueError("source_id does not match event identity")
        return self


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build a delimiter-safe identity, hashing only unreasonably long components."""

    if not isinstance(session_id, str) or not session_id or not isinstance(event_id, str) or not event_id:
        raise ValueError("session_id and event_id must be non-empty strings")
    encoded = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(encoded) <= MAX_SOURCE_ID_LENGTH:
        return encoded
    digest = hashlib.sha256((session_id + "\0" + event_id).encode("utf-8")).hexdigest()
    return f"pi:sha256:{digest}"


def _json_load(line: bytes) -> object:
    try:
        return json.loads(
            line.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        raise ValueError("invalid Pi session JSONL") from error


def _parse_archive(value: PiSessionInput) -> tuple[str, dict[str, dict[str, JsonValue]], str, tuple[str, ...]]:
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session must be UTF-8") from error
    records: list[dict[str, JsonValue]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parsed = _json_load(line.encode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("each Pi session record must be an object")
        records.append(parsed)
    if not records:
        raise ValueError("Pi session is empty")
    header = records[0]
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("first Pi record must be a v3 session header")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session id must be a non-empty string")
    by_id: dict[str, dict[str, JsonValue]] = {session_id: header}
    ordered_ids: list[str] = []
    parent_by_id: dict[str, str | None] = {session_id: None}
    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id or event_id in by_id:
            raise ValueError("event ids must be unique non-empty strings")
        if "parentId" not in record:
            raise ValueError("event parentId is required")
        raw_parent = record.get("parentId")
        if raw_parent is None or raw_parent == "":
            parent_id = None
        elif isinstance(raw_parent, str) and raw_parent in by_id:
            parent_id = raw_parent
        else:
            raise ValueError("event parentId must refer to an earlier record")
        by_id[event_id] = record
        parent_by_id[event_id] = parent_id
        ordered_ids.append(event_id)
    leaf_id = value.leaf_id
    if leaf_id is None:
        leaf_id = ordered_ids[-1] if ordered_ids else session_id
    if leaf_id not in by_id:
        raise ValueError("selected leaf does not exist")
    # Walk the selected ancestry and defensively detect malformed cycles.
    chain: list[str] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            raise ValueError("Pi session graph contains a cycle")
        seen.add(current)
        chain.append(current)
        current = parent_by_id[current]
    chain.reverse()
    return session_id, by_id, leaf_id, tuple(chain)


def _visible_event(session_id: str, record: dict[str, JsonValue], parent_id: str | None) -> PiEventInput | None:
    event_type = record.get("type")
    if not isinstance(event_type, str):
        return None
    role: str | None = None
    body: str | None = None
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        raw_role = message.get("role")
        if not isinstance(raw_role, str) or not raw_role:
            return None
        role = raw_role
        if raw_role == "toolResult":
            tool_name = message.get("toolName", "")
            tool_call_id = message.get("toolCallId", "")
            text = _visible_content(message.get("content"))
            if not text:
                return None
            body = f"[tool_result {tool_name} {tool_call_id}] {text}"
        elif raw_role == "bashExecution":
            command = message.get("command", "")
            output = message.get("output", "")
            if not isinstance(command, str):
                command = str(command)
            if not isinstance(output, str):
                output = str(output)
            body = f"$ {command}\n{output}"
        else:
            body = _visible_content(message.get("content"))
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else None
    elif event_type == "custom_message":
        role = "custom"
        body = _visible_content(record.get("content"))
    if role is None or body is None or body == "":
        return None
    event_id = record.get("id")
    if not isinstance(event_id, str):
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _visible_content(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for block in value:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block_type == "toolCall":
            name = block.get("name", "")
            call_id = block.get("id", "")
            arguments = block.get("arguments", {})
            try:
                arguments_json = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError) as error:
                raise ValueError("toolCall arguments must be JSON serializable") from error
            parts.append(f"[tool_call {name} {call_id}] {arguments_json}")
    return "\n".join(parts) if parts else None


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse and validate an archive atomically, returning the selected path."""

    if not isinstance(value, PiSessionInput):
        raise ValueError("value must be a PiSessionInput")
    session_id, by_id, _leaf_id, chain = _parse_archive(value)
    parents: dict[str, str | None] = {session_id: None}
    for event_id, record in by_id.items():
        if event_id != session_id:
            raw_parent = record.get("parentId")
            parents[event_id] = raw_parent if isinstance(raw_parent, str) and raw_parent else None
    events: list[PiEventInput] = []
    for event_id in chain:
        if event_id == session_id:
            continue
        event = _visible_event(session_id, by_id[event_id], parents[event_id])
        if event is not None:
            events.append(event)
    return tuple(events)


class PiEventSourceAdapter:
    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class: type[PiEventSource]

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PiEventSourceAdapter.source_class = PiEventSource
PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER, version="1", projections=(PiEventTextEvidenceProjection(),)
)


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PiSessionConnector:
    """Incrementally submit visible events while retaining immutable history."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = self.path.read_bytes()
        parsed_input = PiSessionInput(content=raw, leaf_id=self.leaf_id)
        session_id, by_id, _leaf_id, chain = _parse_archive(parsed_input)
        previous = _checkpoint_events(session.checkpoint)

        # Verify every previously imported event still exists byte-for-byte in the archive,
        # including events that are currently on an unselected branch.
        for identity, entry in previous.items():
            event_id = entry.get("event_id")
            if not isinstance(event_id, str) or event_id not in by_id:
                raise ValueError("an imported Pi event disappeared from the archive")
            current = by_id[event_id]
            if entry.get("session_id") != session_id or entry.get("digest") != _record_digest(current):
                raise ValueError("an imported Pi event was changed")
            if identity != pi_event_source_id(session_id, event_id):
                raise ValueError("checkpoint event identity is invalid")

        current_events: list[PiEventInput] = []
        parents: dict[str, str | None] = {session_id: None}
        for event_id, record in by_id.items():
            if event_id != session_id:
                parent = record.get("parentId")
                parents[event_id] = parent if isinstance(parent, str) and parent else None
        for event_id in chain:
            if event_id == session_id:
                continue
            event = _visible_event(session_id, by_id[event_id], parents[event_id])
            if event is not None:
                current_events.append(event)

        next_events: dict[str, dict[str, JsonValue]] = dict(previous)
        for event in current_events:
            identity = pi_event_source_id(event.session_id, event.event_id)
            if identity in previous:
                continue
            outcome = await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            if outcome.status.value == "accepted":
                next_events[identity] = {
                    "session_id": event.session_id,
                    "event_id": event.event_id,
                    "digest": _record_digest(event.record),
                }
        checkpoint: dict[str, JsonValue] = {"version": 1, "events": next_events}
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


def _checkpoint_events(checkpoint: JsonValue | None) -> dict[str, dict[str, JsonValue]]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
        raise ValueError("invalid Pi connector checkpoint")
    raw = checkpoint.get("events", {})
    if not isinstance(raw, dict):
        raise ValueError("invalid Pi connector checkpoint events")
    result: dict[str, dict[str, JsonValue]] = {}
    for identity, entry in raw.items():
        if not isinstance(identity, str) or not isinstance(entry, dict):
            raise ValueError("invalid Pi connector checkpoint event")
        result[identity] = dict(entry)
    return result


__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
