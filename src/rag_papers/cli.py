# argparse = reads commands from the terminal (ingest, ask, --pdf-dir, etc.)
# argparse is a Python built-in library to read terminal commands
import argparse
# Path = helps handle file paths (like data/pdfs) safely
from pathlib import Path

from rag_papers.pdf_loader import list_pdfs, load_pdf_pages


def main():
    parser = argparse.ArgumentParser(
        prog="rag_papers",
        description="RAG Research Paper Assistant (PDF RAG + citations)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest command (now we add arguments)
    ingest_parser = sub.add_parser("ingest", help="Ingest PDFs into the local index (step 1: load text)")
    ingest_parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data/pdfs",
        help="Folder containing PDF files (default: data/pdfs)",
    )

    # ask command (still placeholder for now)
    sub.add_parser("ask", help="Ask a question over ingested PDFs (coming soon)")

# Parse what the user typed in terminal. This creates an object = args
    args = parser.parse_args()

    if args.command == "ingest":
        pdf_dir = Path(args.pdf_dir)
        pdfs = list_pdfs(pdf_dir)

        # Print how many PDFs were found
        print(f"Found {len(pdfs)} PDF(s) in: {pdf_dir}")

        # Handle empty folder nicely
        if not pdfs:
            print("No PDFs found. Put some PDFs inside data/pdfs and try again.")
            return

        # Load the first PDF as a quick test
        first_pdf = pdfs[0]
        pages = load_pdf_pages(first_pdf)

        print(f"Loaded {len(pages)} page(s) from: {first_pdf.name}")

        # Print a small preview + citation format
        for page in pages[:2]:  # just first 2 pages preview
            preview = page.text[:200].replace("\n", " ")
            print(f"[{page.pdf_name} p.{page.page_number}] {preview}...")

    elif args.command == "ask":
        print("ask: not implemented yet")


if __name__ == "__main__":
    main()
