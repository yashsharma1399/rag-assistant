"""
pdf_loader.py

Responsible for reading PDF files and extracting text page-by-page.
We keep page numbers 1-based so citations look natural (p.1, p.2, ...).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


@dataclass
class PDFPage:
    """
    Represents one page of a PDF.

    We store:
    - pdf_name: file name (for citations)
    - page_number: 1-based page index (for citations)
    - text: extracted text from that page
    """

    pdf_name: str
    page_number: int
    text: str


# Given a directory, return all PDFs inside it.
def list_pdfs(pdf_dir: Path) -> List[Path]:
    # Return all PDF files in a directory (non-recursive) sorted by name.
    # Step 1: Check folder exists
    if not pdf_dir.exists():
        return []
    # Step 2: Iterate through the folder, and Filter only PDFs --> return sorted result list
    return sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])


# Open one PDF and return a list where each element is:
    # page text
    # plus citation metadata (filename + page number)
def load_pdf_pages(pdf_path: Path) -> List[PDFPage]:
    # Extract text from a PDF file page-by-page. Returns a list of PDFPage objects.
    # doc is now a “PDF document object”
    doc = fitz.open(str(pdf_path))
    # output list: contains each page individually (PDFPage = object)
    pages: List[PDFPage] = []

    for i in range(len(doc)):
        # loads actual page from pdf
        page = doc[i]

        # "text" mode return plain extracted text
        text = page.get_text("text")

        pages.append(
            PDFPage(
                pdf_name=pdf_path.name, #file name
                page_number=i + 1,  # 1-based
                text=text.strip(), #removes whitespaces
            )
        )

    doc.close() #clode the PDF
    return pages #return list of pages
