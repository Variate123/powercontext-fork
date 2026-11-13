# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CSV documents captured as stable, independently addressable rows."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"


def _validate_document_id(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("document_id must be non-empty and must not contain leading or trailing whitespace")
    return value


def _validate_primary_key(value: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError("primary_key must be non-empty")
    return value


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document and the column naming its rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    primary_key: str
    content: bytes

    _document_id = field_validator("document_id")(_validate_document_id)
    _primary_key = field_validator("primary_key")(_validate_primary_key)


class CSVRowInput(BaseModel):
    """The lossless header and values for one parsed CSV record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    _document_id = field_validator("document_id")(_validate_document_id)
    _primary_key = field_validator("primary_key")(_validate_primary_key)

    @model_validator(mode="after")
    def validate_row(self) -> CSVRowInput:
        if not self.columns:
            raise ValueError("columns must not be empty")
        if len(self.columns) != len(self.values):
            raise ValueError("values must have the same width as columns")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be unique")
        try:
            primary_key_index = self.columns.index(self.primary_key)
        except ValueError:
            raise ValueError("primary_key must name one of the columns") from None
        if not self.primary_key_value or not self.primary_key_value.strip():
            raise ValueError("primary_key_value must not be empty or whitespace")
        if self.values[primary_key_index] != self.primary_key_value:
            raise ValueError("primary_key_value must match the primary-key column value")
        return self


class CSVRowSource(Source):
    """A captured CSV row whose name and source id are the row identity."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    _document_id = field_validator("document_id")(_validate_document_id)
    _primary_key = field_validator("primary_key")(_validate_primary_key)

    @model_validator(mode="after")
    def validate_source(self) -> CSVRowSource:
        if self.name != self.source_id:
            raise ValueError("name and source_id must match")
        expected = csv_row_source_id(self.document_id, self.primary_key_value)
        if self.source_id != expected:
            raise ValueError("source_id does not match the row identity")
        if not self.columns or len(self.columns) != len(self.values):
            raise ValueError("columns and values must have the same non-zero width")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be unique")
        try:
            primary_key_index = self.columns.index(self.primary_key)
        except ValueError:
            raise ValueError("primary_key must name one of the columns") from None
        if not self.primary_key_value or not self.primary_key_value.strip():
            raise ValueError("primary_key_value must not be empty or whitespace")
        if self.values[primary_key_index] != self.primary_key_value:
            raise ValueError("primary_key_value must match the primary-key column value")
        return self


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return a delimiter-safe stable identity for one document row."""

    _validate_document_id(document_id)
    if not isinstance(primary_key_value, str) or not primary_key_value or not primary_key_value.strip():
        raise ValueError("primary_key_value must not be empty or whitespace")
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate a complete CSV document atomically."""

    if not isinstance(value, CSVDocumentInput):
        raise ValueError("value must be a CSVDocumentInput")
    try:
        text = value.content.decode("utf-8-sig")
        records = csv.reader(io.StringIO(text, newline=""), strict=True)
        rows = list(records)
    except (UnicodeDecodeError, csv.Error, TypeError) as error:
        raise ValueError("invalid UTF-8 or CSV document") from error

    if not rows or not rows[0]:
        raise ValueError("CSV document must contain a non-empty header")
    columns = tuple(rows[0])
    if any(not column or not column.strip() for column in columns):
        raise ValueError("CSV header fields must be non-empty and non-blank")
    if len(set(columns)) != len(columns):
        raise ValueError("CSV header fields must be unique")
    if value.primary_key not in columns:
        raise ValueError("primary_key must exactly match a CSV header field")
    primary_key_index = columns.index(value.primary_key)

    parsed: list[CSVRowInput] = []
    primary_key_values: set[str] = set()
    width = len(columns)
    for record in rows[1:]:
        # csv.reader represents a physically blank record as an empty list.
        if not record:
            continue
        if len(record) != width:
            raise ValueError("CSV data record has the wrong field count")
        primary_key_value = record[primary_key_index]
        if not primary_key_value or not primary_key_value.strip():
            raise ValueError("CSV primary-key values must not be empty or whitespace")
        if primary_key_value in primary_key_values:
            raise ValueError("CSV primary-key values must be unique")
        primary_key_values.add(primary_key_value)
        parsed.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(parsed)


class CSVRowSourceAdapter:
    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row: dict[str, str] = dict(zip(source.columns, source.values, strict=True))
        content = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        metadata: dict[str, JsonValue] = {
            "document_id": source.document_id,
            "primary_key": source.primary_key,
            "primary_key_value": source.primary_key_value,
            "columns": list(source.columns),
            "row": row,
        }
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=content,
            metadata=metadata,
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version="1",
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
