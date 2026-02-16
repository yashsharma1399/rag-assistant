# cli.py
# This file defines the command-line interface (CLI) for our project.
# We will run it using: python -m rag_papers <command>

# argparse = reads commands from the terminal (ingest, ask, --pdf-dir, etc.)
# argparse is a Python built-in library to read terminal commands
import argparse
# Path = helps handle file paths (like data/pdfs) safely
from pathlib import Path

from rag_papers.pdf_loader import list_pdfs, load_pdf_pages

from rag_papers.chunking import chunk_pages, save_chunks_jsonl


def main():
    # Create the main CLI parser
    parser = argparse.ArgumentParser(
        prog="rag_papers",
        description="RAG Research Paper Assistant (PDF RAG + citations)",
    )
    # Sub-commands: ingest, ask
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest command (now we add arguments)
    ingest_parser = sub.add_parser("ingest", help="Ingest PDFs into the local index (step 1: load text + chunk)")
    ingest_parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data/pdfs",
        help="Folder containing PDF files (default: data/pdfs)",
    )

    ingest_parser.add_argument(
        "--chunk_size",
        type=int,
        default=800,
        help="Chunk size in characters (default: 800)",
    )

    ingest_parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Overlap between chunks in characters (default: 150)",
    )

    ingest_parser.add_argument(
        "--out",
        type=str,
        default="data/cache/chunks.jsonl",
        help="Where to save chunk cache (default: data/cache/chunks.jsonl)",
    )

    # ask command (still placeholder for now)
    ask_parser = sub.add_parser(
        "ask",
        help="Ask a question over ingested PDFs (coming soon)",
    )
    ask_parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default="",
        help="Question to ask (coming soon)",
    )


    # Parse what the user typed in terminal into an object = 'args'
    args = parser.parse_args()

    # Handle ingest
    if args.command == "ingest":
        pdf_dir = Path(args.pdf_dir)
        pdfs = list_pdfs(pdf_dir)

        # Print how many PDFs were found
        print(f"Found {len(pdfs)} PDF(s) in: {pdf_dir}")

        # Handle empty folder nicely
        # If no PDFs (empty folder), exit early with a helpful message
        if not pdfs:
            print("No PDFs found. Put some PDFs inside data/pdfs and try again.")
            return

        # Load pages from ALL PDFs
        all_pages = []
        for pdf_path in pdfs:
            pages = load_pdf_pages(pdf_path)
            all_pages.extend(pages)

        print(f"Loaded {len(all_pages)} total page(s) from {len(pdfs)} PDF(s)")

        # Chunk all pages
        chunks = chunk_pages(all_pages, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"Created {len(chunks)} chunk(s) (chunk_size={args.chunk_size}, overlap={args.overlap})")

        # Save chunks to JSONL cache
        out_path = Path(args.out)
        save_chunks_jsonl(chunks, out_path)
        print(f"Saved chunks to: {out_path}")

        # Print a few chunk previews with citation-style metadata
        print("\nPreview (first 3 chunks):")
        for c in chunks[:3]:
            preview = c.text[:200].replace("\n", " ")
            print(f"[{c.pdf_name} p.{c.page_number} | {c.chunk_id}] {preview}...")

        return

    # Handle ask (placeholder)
    if args.command == "ask":
        print("ask: not implemented yet")
        if args.question:
            print(f"Question was: {args.question}")
        return


if __name__ == "__main__":
    main()
