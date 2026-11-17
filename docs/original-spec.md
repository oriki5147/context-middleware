> **Note.** This is the design spec the project started from, kept for the record. The values in it were illustrative and most were replaced after measurement. The design as built is described in the [project report](PROJECT_REPORT.md).

# High-Performance CPU-First Context Optimization Middleware

## 1. Executive Summary & Vision
We are building a production-ready Python Middleware for B2B RAG applications. 
The middleware sits between raw document repositories/queries and target LLMs.
Its purpose is to cut context token counts by 90%–95%, reduce prompt latency, and prevent hallucinations by enforcing 100% extractive faithfulness in CPU execution.

## 2. Theoretical Foundations & Architecture

### Phase 1: Ingestion & Pre-processing (MarkItDown Engine)
- Convert raw documents (PDFs, DOCX, XLSX, etc.) into clean Markdown.
- Apply Fixed-length Chunking (100–120 tokens) with a sliding window overlap (25% / ~30 tokens) to ensure zero answer fragmentation.

### Phase 2: Dense Passage Retrieval (DPR Paradigm)
- Architecture: Dual-Encoder (Bi-Encoder) optimized for CPU.
- Model Format: ONNX Runtime with INT8 Quantization (e.g., `bge-small-en-v1.5` or `all-MiniLM-L6-v2`).
- Search: In-memory MIPS / Cosine Similarity scoring.
- Output: Top-K candidate chunks ($K=10$).

### Phase 3: Extractive & Selective Compression (RECOMP Paradigm)
- Deconstruct top-K chunks into individual sentences.
- Compute sentence-level semantic similarity against the user query $x$.
- Apply dynamic similarity thresholding ($\ge 0.65$).
- **Selective Augmentation:** If all sentence scores fall below the absolute quality threshold ($\le 0.45$), return an empty string (`is_empty=True`, 0 tokens) to allow the target LLM to answer using its parametric knowledge without noise.

### Phase 4: Context Reordering (Lost in the Middle / LongLLMLingua)
- Reorder extracted sentences in a U-Shape (Sandwich) structure.
- Top-scoring sentences placed at the very top and very bottom of the context prompt.

## 3. Data Structures Specification
```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TextSegment:
    segment_id: str
    content: str
    source_doc: str
    embedding: Optional[List[float]] = None
    score: float = 0.0

@dataclass
class CompressionResult:
    compressed_prompt: str
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    is_empty: bool
