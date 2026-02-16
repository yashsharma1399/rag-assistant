# vector_store.py
# Stores and retrieves chunk embeddings using ChromaDB (persistent local vector DB).


from pathlib import Path
from typing import List, Dict, Any

# chromadb: the vector database library
# Settings: configuration for Chroma (we’ll use it to disable telemetry)
import chromadb
from chromadb.config import Settings


#Creates a persistent Chroma client (connection to DB) that stores data on disk in persist_dir.
# It’s like opening a local database connection.
# Default location where Chroma will store its files on your disk = data/index/
# -> chromadb.ClientAPI: “this function returns a Chroma client object.”
def get_chroma_client(persist_dir: str = "data/index") -> chromadb.ClientAPI:
    # Make sure the folder exists
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    # creates a persistent Chroma client located at data/index
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client
# Result: every time you run your project, it reconnects to the same local DB on disk.


# Creates (or loads if already exists) a Chroma collection.
# A collection is like a table in a normal database.
def get_or_create_collection(client: chromadb.ClientAPI, name: str = "rag_papers"):
    return client.get_or_create_collection(name=name)
