"""
cli.py

This file defines the Command-line interface (CLI) for our RAG project.

We run it like:
  python -m rag_papers ingest --pdf_dir data/pdfs
  python -m rag_papers ask "your question here" --top_k 5

What each command does:
1) ingest
   - Finds PDFs in a folder
   - Extracts text page-by-page
   - Splits pages into overlapping chunks
   - Saves chunks to a JSONL cache file
   - Builds embeddings for chunks (SentenceTransformers)
   - Stores embeddings + chunk text + metadata in Chroma (local persistent vector DB)

2) ask
   - Embeds the user question
   - Queries Chroma for top-k most similar chunks
   - Prints results with citations (pdf name + page number + chunk id)
"""

# argparse = reads commands from the terminal (ingest, ask, --pdf-dir, etc.)
# argparse is a Python built-in library to read terminal commands
import argparse
from pathlib import Path

# Our modules for PDF reading + chunking
from rag_papers.pdf_loader import list_pdfs, load_pdf_pages
from rag_papers.chunking import chunk_pages, save_chunks_jsonl

# Our modules for embeddings + vector DB
from rag_papers.embeddings import load_embedding_model, embed_texts
from rag_papers.vector_store import get_chroma_client, get_or_create_collection

def main():
    # -----------------------------
    # 1) Create the main CLI parser
    # -----------------------------
    parser = argparse.ArgumentParser(
        prog="rag_papers",
        description="RAG Research Paper Assistant (PDF RAG + citations)",
    )
    # Sub-commands: ingest, ask
    sub = parser.add_subparsers(dest="command", required=True)

    # -----------------------------
    # 2) ingest command (now we add arguments)
    # -----------------------------
    ingest_parser = sub.add_parser(
        "ingest",
        help="Ingest PDFs (load text -> chunk -> embed -> store in Chroma)",
    )

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

    ingest_parser.add_argument(
        "--persist_dir",
        type=str,
        default="data/index",
        help="Chroma persistent directory (default: data/index)",
    )

    ingest_parser.add_argument(
        "--collection",
        type=str,
        default="rag_papers",
        help="Chroma collection name (default: rag_papers)",
    )

    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        help="If set, clears the Chroma collection before inserting (avoids duplicate IDs).",
    )

    ingest_parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model name (default: all-MiniLM-L6-v2)",
    )

    # -----------------------------
    # 3) ask command
    # -----------------------------
    ask_parser = sub.add_parser(
        "ask",
        help="Ask a question over ingested PDFs (retrieves top chunks with citations)",
    )

    # For ask, we want the question as a required positional argument
    ask_parser.add_argument(
        "question",
        type=str,
        help="Question to ask",
    )

    ask_parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many chunks to retrieve (default: 5)",
    )

    ask_parser.add_argument(
        "--persist_dir",
        type=str,
        default="data/index",
        help="Chroma persistent directory (default: data/index)",
    )

    ask_parser.add_argument(
        "--collection",
        type=str,
        default="rag_papers",
        help="Chroma collection name (default: rag_papers)",
    )

    ask_parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model name (default: all-MiniLM-L6-v2)",
    )


    # Parse what the user typed in terminal into an object = 'args'
    args = parser.parse_args()


    # =========================================================
    # INGEST
    # =========================================================
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

        # A) Load pages from ALL PDFs
        all_pages = []
        for pdf_path in pdfs:
            pages = load_pdf_pages(pdf_path)
            all_pages.extend(pages)

        print(f"Loaded {len(all_pages)} total page(s) from {len(pdfs)} PDF(s)")

        # B) Chunk all pages
        chunks = chunk_pages(all_pages, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"Created {len(chunks)} chunk(s) (chunk_size={args.chunk_size}, overlap={args.overlap})")

        # C) Save chunks to JSONL cache
        out_path = Path(args.out)
        save_chunks_jsonl(chunks, out_path)
        print(f"Saved chunks to: {out_path}")
        
        # Preview chunks so we can visually confirm citations look correct
        print("\nPreview (first 3 chunks):")
        for c in chunks[:3]:
            preview = c.text[:200].replace("\n", " ")
            print(f"[{c.pdf_name} p.{c.page_number} | {c.chunk_id}] {preview}...")
        
        # ------------------------------------------
        # D) Build embeddings + store in Chroma
        # ------------------------------------------
        print("\nBuilding embeddings and storing in Chroma...")

        # Load local embedding model (downloads once, then cached)
        model = load_embedding_model(args.model)

        # Convert chunk texts to embeddings
        texts = [c.text for c in chunks]
        embeddings = embed_texts(model, texts)  # List[List[float]]

        # Create/load persistent Chroma client + collection
        client = get_chroma_client(persist_dir=args.persist_dir)
        collection = get_or_create_collection(client, name=args.collection)

        # Optional: reset collection to avoid duplicate IDs
        if args.reset:
            # Delete everything in the collection
            # (Chroma doesn't have "truncate", so we delete all ids if present)
            existing = collection.get(include=[])
            existing_ids = existing.get("ids", [])
            if existing_ids:
                collection.delete(ids=existing_ids)
            print(f"Reset collection '{args.collection}' (deleted {len(existing_ids)} existing items).")

        # Use chunk_id as stable unique id in Chroma
        ids = [c.chunk_id for c in chunks]

        # Store metadata for citations (and future debugging)
        metadatas = [{"pdf_name": c.pdf_name, "page_number": c.page_number} for c in chunks]

        # Add documents + embeddings + metadata to Chroma
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

        print(f"Stored {len(ids)} chunk embeddings in Chroma collection: '{args.collection}'")
        print("Ingest complete ✅")

        return


    # =========================================================
    # ASK
    # =========================================================
    if args.command == "ask":
        question = args.question.strip()
        if not question:
            print("Please provide a non-empty question.")
            return

        # Load the embedding model
        model = load_embedding_model(args.model)

        # Embed the question (query embedding)
        query_embedding = embed_texts(model, [question])[0]  # single vector

        # Load Chroma
        # Opens your saved Chroma DB from disk
        client = get_chroma_client(persist_dir=args.persist_dir)
        # Selects the collection where chunks were stored during ingest
        collection = get_or_create_collection(client, name=args.collection)

        # Query top-k similar chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=args.top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        print("\nTop results (with citations):")
        for i in range(len(docs)):
            pdf_name = metas[i].get("pdf_name", "unknown.pdf")
            page_number = metas[i].get("page_number", "?")
            preview = docs[i][:200].replace("\n", " ")
            print(f"{i+1}) [{pdf_name} p.{page_number}] (distance={dists[i]:.4f}) {preview}...")

        return


if __name__ == "__main__":
    main()
