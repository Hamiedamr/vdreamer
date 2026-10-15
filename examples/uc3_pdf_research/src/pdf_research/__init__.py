"""UC3 — Multi-PDF Research & Comparison, the vdreamer reference agents (PRD §7.3)."""

from pdf_research.agents import (
    CitationMiddleware,
    IngestInput,
    PdfIngestAgent,
    PdfResearchAgent,
    ingest_agent,
    research_agent,
)

__all__ = [
    "CitationMiddleware",
    "IngestInput",
    "PdfIngestAgent",
    "PdfResearchAgent",
    "ingest_agent",
    "research_agent",
]
