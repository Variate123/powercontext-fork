"""Page-preserving PDF Sources and lexical page retrieval."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_SOURCE_NAME = "pdf-page"


class PDFDocumentInput(BaseModel):
    """One complete PDF byte stream identified by the caller."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value or not value.strip():
            raise ValueError("document_id must not be empty or whitespace")
        return value

    @field_validator("content", mode="before")
    @classmethod
    def require_bytes(cls, value: object) -> bytes:
        if not isinstance(value, bytes):
            raise ValueError("content must be bytes")
        return value


class PDFPageInput(BaseModel):
    """The complete extracted text and identity of one physical PDF page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value or not value.strip():
            raise ValueError("document_id must not be empty or whitespace")
        return value

    @field_validator("document_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
            raise ValueError("document_sha256 must be a lowercase SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("document_sha256 must be hexadecimal") from error
        return value

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("page_number must be positive")
        return value


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return a collision-safe identity for one immutable document page."""

    if not isinstance(document_id, str) or not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty or whitespace")
    if not isinstance(document_sha256, str) or len(document_sha256) != 64:
        raise ValueError("document_sha256 must be a SHA-256 digest")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be positive")
    encoded = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(encoded) <= MAX_SOURCE_ID_LENGTH:
        return encoded
    digest = hashlib.sha256(f"{document_id}\0{document_sha256}\0{page_number}".encode("utf-8")).hexdigest()
    return f"pdf-hash:{digest}"


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Read every physical page from a complete PDF atomically."""

    if not isinstance(value, PDFDocumentInput):
        raise ValueError("value must be a PDFDocumentInput")
    if not value.content:
        raise ValueError("PDF content must not be empty")
    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted:
            result = reader.decrypt("")
            if not result:
                raise ValueError("PDF requires a password")
        pages: list[PDFPageInput] = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text is None:
                text = ""
            if not isinstance(text, str):
                raise ValueError("PDF page text is not text")
            pages.append(
                PDFPageInput(
                    document_id=value.document_id,
                    document_sha256=digest,
                    page_number=number,
                    text=text,
                )
            )
        if not pages:
            raise ValueError("PDF contains no pages")
        return tuple(pages)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("unable to parse complete PDF") from error


class PDFPageSource(Source):
    """Captured page text retaining the original document revision identity."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @model_validator(mode="after")
    def validate_source_identity(self) -> PDFPageSource:
        PDFPageInput(
            document_id=self.document_id,
            document_sha256=self.document_sha256,
            page_number=self.page_number,
            text=self.text,
        )
        if self.name != self.source_id:
            raise ValueError("name and source_id must match")
        if self.source_id != pdf_page_source_id(self.document_id, self.document_sha256, self.page_number):
            raise ValueError("source_id does not match page identity")
        return self


class PDFPageSourceAdapter:
    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version="1",
    projections=(PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple["MemoryCitation", ...]:
    """Search a scoped Memory and return distinct, exact page citations."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        return ()
    # Imported lazily because ``powercontext.sources`` is itself used while the
    # artifact model package is being initialized.
    from powercontext.artifacts import MemoryCitation
    from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest

    # Retrieve a broad lexical pool, then inspect each entry's Source references to
    # deduplicate pages. The entry and citation remain entirely Memory-owned.
    page = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=max(limit * 10, 100)))
    results: list[MemoryCitation] = []
    seen_pages: set[tuple[str, str]] = set()
    for hit in page.hits:
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        entry = await memory.get(GetMemoryEntryRequest(citation=citation))
        references = tuple(reference for reference in entry.entry.sources if reference.source_type == PDF_PAGE_SOURCE_NAME)
        for reference in references:
            identity = (reference.source_type, reference.source_id)
            if identity in seen_pages:
                continue
            seen_pages.add(identity)
            results.append(citation)
            break
        if len(results) >= limit:
            break
    return tuple(results)


__all__ = [
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
