import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
import os
import PyPDF2

# ---- Page Config ----
st.set_page_config(
    page_title="AI StudyMate",
    page_icon="📘",
    layout="wide"
)

# ---- Title ----
st.markdown(
    """
    <div style='text-align:center'>
        <h1 style='color:#4B0082;'> AI StudyMate</h1>
        <p style='font-size:18px; color:#555;'>Your personal AI-powered study assistant</p>
    </div>
    """, 
    unsafe_allow_html=True
)

DB_PATH = "endee_db.json"
HISTORY_PATH = "search_history.json"

# ---- Sidebar Navigation ----
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Search Questions", "Upload Notes", "Summarize Text"])

# ---- Load Embedding Model ----
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---- Utility Functions ----
def load_vector_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        texts = [r["text"] for r in records]
        embeddings = np.array([r["embedding"] for r in records])
        metadata = [r.get("metadata", {}) for r in records]
        return texts, embeddings, metadata
    return [], np.array([]), []

def save_vector_db(texts, embeddings, metadata):
    records = []
    for i, text in enumerate(texts):
        records.append({
            "id": i,
            "text": text,
            "embedding": embeddings[i].tolist(),
            "metadata": metadata[i] if i < len(metadata) else {}
        })
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

def semantic_search(query, texts, embeddings, top_k=3):
    if len(texts) == 0:
        return []
    query_embedding = model.encode([query])[0]
    scores = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [{"text": texts[idx], "score": float(scores[idx])} for idx in top_indices]

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ---- Summarization Function ----
def summarize_text(text, top_k=3):
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return "No valid sentences to summarize."
    
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    query_embedding = model.encode("Summarize this text", convert_to_tensor=True)
    
    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k].tolist()
    
    summary = ". ".join([sentences[i] for i in top_indices])
    if not summary.endswith("."):
        summary += "."
    return summary

# ---- Load / Save Search History ----
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# ---- Pages ----
if page == "Search Questions":
    st.subheader(" Ask Your Question")
    texts, embeddings, metadata = load_vector_db()

    # Load search history
    search_history = load_history()

    query = st.text_input(
        " Type your question here...",
        placeholder="e.g. Explain Newton's third law"
    )

    if st.button("Search Answer", type="primary"):
        if not texts:
            st.warning(" No notes uploaded yet. Please upload notes first!")
        elif query.strip():
            # Save query to history
            search_history.append(query)
            save_history(search_history)

            results = semantic_search(query, texts, embeddings)
            st.subheader(" Top Matching Answers:")

            for i, res in enumerate(results, 1):
                source = metadata[i-1].get("source", "Unknown") if metadata else "Unknown"
                st.markdown(
                    f"""
                    <div style='background-color:#F5F5F5; padding:15px; border-radius:10px; margin-bottom:10px;'>
                        <h4 style='color:#4B0082;'>Answer {i}</h4>
                        <p style='font-size:16px; color:#333;'>{res['text']}</p>
                        <p style='font-size:13px; color:#888;'>Source: {source} | Score: {res['score']:.3f}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.warning(" Please enter a question.")

    # Display search history
    if search_history:
        st.subheader("Your Search History:")
        for i, q in enumerate(reversed(search_history), 1):
            st.write(f"{i}. {q}")

elif page == "Upload Notes":
    st.subheader(" Upload Your Study Notes")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Upload study notes in PDF or TXT format"
    )

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = str(uploaded_file.read(), "utf-8")
        else:
            text = ""

        if text.strip():
            st.success(" File uploaded successfully! Preview below:")
            st.text_area(
                "Preview",
                text,
                height=300,
                max_chars=None
            )

            chunks = chunk_text(text, chunk_size=300, overlap=50)
            texts, embeddings, metadata = load_vector_db()

            for chunk in chunks:
                new_embedding = model.encode([chunk])[0]
                texts.append(chunk)
                embeddings = np.vstack([embeddings, new_embedding]) if embeddings.size else np.array([new_embedding])
                metadata.append({"source": uploaded_file.name})

            save_vector_db(texts, embeddings, metadata)
            st.success(f" Notes added as {len(chunks)} chunks to the database!")

elif page == "Summarize Text":
    st.subheader("Text Summarizer")
    input_text = st.text_area("Paste your text here :", height=200)

    if st.button("Summarize"):
        if input_text.strip():
            summary = summarize_text(input_text, top_k=3)
            st.success("Summary:")
            st.write(summary)
        else:
            st.warning(" Please enter some text to summarize!")
