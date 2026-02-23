"""
app.py (Streamlit UI)

This UI is a front-end for our existing RAG pipeline.
It reuses the same steps:
1) Ingest PDFs: PDF -> pages -> chunks -> embeddings -> Chroma
2) Ask: query -> retrieve top-k -> show cleaned previews -> show free extractive answer

Run:
  streamlit run app.py
"""

import streamlit as st
from pathlib import Path

# Reuse your project functions
from rag_papers.pdf_loader import list_pdfs, load_pdf_pages
from rag_papers.chunking import chunk_pages, save_chunks_jsonl
from rag_papers.embeddings import load_embedding_model, embed_texts
from rag_papers.vector_store import get_chroma_client, get_or_create_collection
from rag_papers.answering import RetrievedChunk, build_extractive_answer


# ---------------------------
# Helper functions (UI logic)
# ---------------------------

def ingest_pdfs(
    pdf_dir: str,
    chunk_size: int,
    overlap: int,
    persist_dir: str,
    collection_name: str,
    reset: bool,
    model_name: str,
    out_path: str,
) -> str:
    """
    Runs the same ingest pipeline your CLI does, but returns a status message for the UI.
    """

    pdf_dir_path = Path(pdf_dir)
    pdfs = list_pdfs(pdf_dir_path)

    if not pdfs:
        return f"No PDFs found in {pdf_dir}. Put PDFs inside that folder and try again."

    # Load pages from all PDFs
    all_pages = []
    for pdf_path in pdfs:
        all_pages.extend(load_pdf_pages(pdf_path))

    # Chunk pages
    chunks = chunk_pages(all_pages, chunk_size=chunk_size, overlap=overlap)

    # Save chunk cache
    out_file = Path(out_path)
    save_chunks_jsonl(chunks, out_file)

    # Embed + store in Chroma
    model = load_embedding_model(model_name)
    texts = [c.text for c in chunks]
    embeddings = embed_texts(model, texts)

    client = get_chroma_client(persist_dir=persist_dir)
    collection = get_or_create_collection(client, name=collection_name)

    # Reset collection to avoid duplicate IDs (optional)
    if reset:
        existing = collection.get(include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)

    ids = [c.chunk_id for c in chunks]
    metadatas = [{"pdf_name": c.pdf_name, "page_number": c.page_number} for c in chunks]

    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    return (
        f"✅ Ingest complete!\n"
        f"- PDFs: {len(pdfs)}\n"
        f"- Pages: {len(all_pages)}\n"
        f"- Chunks: {len(chunks)}\n"
        f"- Stored in Chroma: {persist_dir} / {collection_name}\n"
        f"- Cache saved: {out_path}"
    )


def ask_question(
    question: str,
    top_k: int,
    persist_dir: str,
    collection_name: str,
    model_name: str,
):
    """
    Runs the same ask pipeline your CLI does:
    - embed query
    - query Chroma
    - build cleaned previews
    - build free extractive final answer
    """
    question = question.strip()
    if not question:
        return None

    model = load_embedding_model(model_name)
    query_embedding = embed_texts(model, [question])[0]

    client = get_chroma_client(persist_dir=persist_dir)
    collection = get_or_create_collection(client, name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    retrieved_chunks = []
    for i in range(len(docs)):
        retrieved_chunks.append(
            RetrievedChunk(
                text=docs[i],
                metadata=metas[i],
                distance=dists[i],
            )
        )

    # Build cleaned previews (1 best sentence per chunk)
    cleaned_previews = []
    for ch in retrieved_chunks:
        one_bullet = build_extractive_answer(question, [ch], max_bullets=1)[0]
        # remove "- " prefix
        if one_bullet.startswith("- "):
            one_bullet = one_bullet[2:].strip()
        # remove trailing citation like "[... p.X]"
        one_bullet = one_bullet.rsplit("[", 1)[0].strip()

        pdf_name = ch.metadata.get("pdf_name", "unknown.pdf")
        page_number = ch.metadata.get("page_number", "?")

        cleaned_previews.append(
            {
                "pdf_name": pdf_name,
                "page_number": page_number,
                "distance": ch.distance,
                "preview": one_bullet,
            }
        )

    # Final free answer
    bullets = build_extractive_answer(question, retrieved_chunks, max_bullets=5)

    return cleaned_previews, bullets


# ---------------------------
# Streamlit Page Layout
# ---------------------------

st.set_page_config(page_title="RAG Paper Assistant", layout="wide")

# ---------------------------
# Modern Brown + Beige Theme (CSS)
# ---------------------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(180deg, #f6f0e6 0%, #efe3d2 100%);
        color: #2b1f16;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: #2b1f16;
    }

    /* Sidebar general text (keep light) */
    section[data-testid="stSidebar"] * {
        color: #f6f0e6 !important;
    }

    /* Main area inputs */
    .stApp input, .stApp textarea {
        background-color: #fffaf2 !important;
        color: #2b1f16 !important;
        border: 1px solid #d6c2aa !important;
        border-radius: 10px !important;
    }

    /* ✅ Sidebar inputs: force BLACK text so it stays visible */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] [data-baseweb="input"] input {
        background-color: #fffaf2 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        border: 1px solid #d6c2aa !important;
        border-radius: 10px !important;
    }

    /* Sidebar input placeholder color (optional) */
    section[data-testid="stSidebar"] input::placeholder,
    section[data-testid="stSidebar"] textarea::placeholder {
        color: #333333 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #6b4f3a;
        color: #f6f0e6;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #5a3f2d;
        color: #f6f0e6;
    }

    /* Headings */
    h1, h2, h3 {
        color: #2b1f16 !important;
    }

    /* ✅ Caption text (st.caption) */
    div[data-testid="stCaptionContainer"] {
        color: #2b1f16 !important;
    }
    div[data-testid="stCaptionContainer"] * {
        color: #2b1f16 !important;
    }

    /* ✅ Make widget labels (like "Enter your question") dark in the main page */
    .stApp label, 
    .stApp label span,
    .stApp .stTextInput label {
    color: #2b1f16 !important;   /* dark brown/black */
    }

    /* Sometimes Streamlit wraps labels inside this container */
    .stApp [data-testid="stWidgetLabel"] {
    color: #2b1f16 !important;
    }
    .stApp [data-testid="stWidgetLabel"] * {
    color: #2b1f16 !important;
    }

    /* ✅ Sidebar labels should stay WHITE (override main label rule) */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label span,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
    color: #f6f0e6 !important;   /* light beige/white */
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.title("📄 Research Paper Navigator (With Citations)")
st.caption("PDF RAG pipeline with Chroma retrieval + extractive answering + citations.")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

pdf_dir = st.sidebar.text_input("PDF folder", value="data/pdfs")
persist_dir = st.sidebar.text_input("Chroma persist dir", value="data/index")
collection_name = st.sidebar.text_input("Chroma collection", value="rag_papers")

chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=3000, value=800, step=50)
overlap = st.sidebar.number_input("Overlap (chars)", min_value=0, max_value=1000, value=150, step=10)

top_k = st.sidebar.number_input("Top-K retrieval", min_value=1, max_value=20, value=5, step=1)

model_name = st.sidebar.text_input("Embedding model", value="all-MiniLM-L6-v2")

reset = st.sidebar.checkbox("Reset Chroma collection before ingest", value=False)

out_path = st.sidebar.text_input("Chunks cache path", value="data/cache/chunks.jsonl")


# Main layout: two columns
col1, col2 = st.columns(2)

# ---------------------------
# Left Column: Ingest
# ---------------------------
with col1:
    st.subheader("1) Ingest PDFs")
    st.write(
        "This step reads PDFs from the folder, splits them into chunks, generates embeddings, "
        "and stores them in Chroma for retrieval."
    )

    # Show what PDFs exist
    if st.button("🔍 List PDFs in folder"):
        pdfs = list_pdfs(Path(pdf_dir))
        if not pdfs:
            st.warning(f"No PDFs found in {pdf_dir}")
        else:
            st.success(f"Found {len(pdfs)} PDF(s)")
            for p in pdfs:
                st.write(f"- {p.name}")

    if st.button("📥 Run Ingest"):
        with st.spinner("Ingesting PDFs... (may take a minute on first run)"):
            msg = ingest_pdfs(
                pdf_dir=pdf_dir,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
                persist_dir=persist_dir,
                collection_name=collection_name,
                reset=reset,
                model_name=model_name,
                out_path=out_path,
            )
        st.text(msg)

# ---------------------------
# Right Column: Ask
# ---------------------------
with col2:
    st.subheader("2) Ask a Question")
    st.write(
        "This step embeds your question, retrieves top-k relevant chunks from Chroma, "
        "shows cleaned previews, and generates a free extractive answer with citations."
    )

    question = st.text_input("Enter your question", value="What is this paper about?")

    if st.button("❓ Ask"):
        with st.spinner("Retrieving and building answer..."):
            result = ask_question(
                question=question,
                top_k=int(top_k),
                persist_dir=persist_dir,
                collection_name=collection_name,
                model_name=model_name,
            )

        if result is None:
            st.warning("Please type a question.")
        else:
            cleaned_previews, bullets = result

            st.markdown("### 🔎 Top Results (Cleaned Preview)")
            for i, item in enumerate(cleaned_previews, start=1):
                st.write(
                    f"**{i}) [{item['pdf_name']} p.{item['page_number']}]** "
                    f"(distance={item['distance']:.4f}) — {item['preview']}"
                )

            st.markdown("### ✅ Final Answer (Free, Extractive)")
            for b in bullets:
                st.write(b)