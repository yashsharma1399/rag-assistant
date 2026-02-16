"""
chunking.py

Takes page-wise text and splits it into smaller overlapping chunks.
Each chunk carries metadata for citations.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import List

from rag_papers.pdf_loader import PDFPage


@dataclass
class TextChunk:
    
    #One chunk of text + metadata for citations and debugging.
    chunk_id: str
    pdf_name: str
    page_number: int
    text: str
    char_start: int
    char_end: int


# Split a single PDFPage into overlapping character chunks.
#     chunk_size: number of characters in each chunk
#     overlap: how many characters overlap between consecutive chunks
def chunk_page(page: PDFPage, chunk_size: int = 800, overlap: int = 150) -> List[TextChunk]:
    # get the page text, handle empty
    text = page.text
    if not text:
        return []

    chunks: List[TextChunk] = []
    start = 0 #where the current chunk begins in the page text.
    chunk_index = 0 #used to create chunk_id

    #loop until we reach end of text
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        # create the chunk object if there’s any text
        if chunk_text:
            # Stable chunk id: pdf__p{page}__c{chunk_index}
            chunk_id = f"{page.pdf_name}__p{page.page_number}__c{chunk_index}"

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    pdf_name=page.pdf_name,
                    page_number=page.page_number,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                )
            )

        # Move forward, keeping overlap
        if end == len(text):
            break
        start = max(0, end - overlap)
        chunk_index += 1

    return chunks

# Chunk a list of PDF pages.
def chunk_pages(pages: List[PDFPage], chunk_size: int = 800, overlap: int = 150) -> List[TextChunk]:
    all_chunks: List[TextChunk] = []
    for page in pages:
        all_chunks.extend(chunk_page(page, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


# Save chunks into a JSONL file (1 chunk per line).
def save_chunks_jsonl(chunks: List[TextChunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


# Load chunks back from JSONL.
def load_chunks_jsonl(path: Path) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(TextChunk(**obj))
    return chunks
