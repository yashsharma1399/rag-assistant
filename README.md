# RAG Research Paper Navigator (With Citations)

A **PDF RAG** project in Python that:
- Ingests research PDFs (page-wise extraction)
- Splits text into overlapping chunks
- Builds **local embeddings** (SentenceTransformers)
- Stores vectors in **ChromaDB** (persistent local vector DB)
- Retrieves top-K relevant chunks for a question
- Produces a **free extractive final answer** with **page-level citations**
- Includes a clean **Streamlit UI** + CLI

---


