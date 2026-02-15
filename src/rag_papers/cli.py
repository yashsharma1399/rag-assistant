import argparse

def main():
    parser = argparse.ArgumentParser(
        prog="rag_papers",
        description="RAG Research Paper Assistant (PDF RAG + citations)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Ingest PDFs into the local index (coming soon)")
    sub.add_parser("ask", help="Ask a question over ingested PDFs (coming soon)")

    args = parser.parse_args()

    if args.command == "ingest":
        print("ingest: not implemented yet")
    elif args.command == "ask":
        print("ask: not implemented yet")

if __name__ == "__main__":
    main()
