"""AI Medical Assistant - minimal RAG system (educational use only).

Run:  streamlit run app.py
Put your knowledge files (.txt / .md) in the ./data folder.
"""
import os
import glob
import numpy as np
import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
MODEL = "claude-sonnet-5-5"
TOP_K = 4
MIN_SCORE = 0.30  # below this similarity -> "insufficient evidence"

DISCLAIMER = (
    "Educational decision-support tool only. Not a diagnosis or treatment "
    "advice. In an emergency, call your local emergency number."
)
EMERGENCY_WORDS = ["chest pain", "can't breathe", "cannot breathe", "stroke",
                   "suicide", "overdose", "severe bleeding", "unconscious"]

SYSTEM_PROMPT = """You are a clinical information assistant for education.
Answer ONLY using the numbered context passages provided.
- Cite sources inline like [1], [2].
- If the context does not contain the answer, reply exactly:
  "I don't have enough evidence in my knowledge base to answer this."
- Never invent doses, drugs, or statistics.
- Be concise and structured. Recommend consulting a clinician where relevant."""


def chunk_text(text, size=120, overlap=25):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)
            if words[i:i + size]]


@st.cache_resource(show_spinner="Building knowledge base...")
def build_index():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks, sources = [], []
    for path in glob.glob(os.path.join(DATA_DIR, "*.*")):
        if not path.endswith((".txt", ".md")):
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
            pieces = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in pieces:
                for c in chunk_text(p, size=200, overlap=30):
                    chunks.append(c)
                    sources.append(os.path.basename(path))
    if not chunks:
        return embedder, [], [], None
    vecs = embedder.encode(chunks, normalize_embeddings=True)
    return embedder, chunks, sources, vecs


def retrieve(query, embedder, chunks, sources, vecs):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    scores = vecs @ q
    idx = np.argsort(-scores)[:TOP_K]
    return [(chunks[i], sources[i], float(scores[i])) for i in idx]


def answer(query, hits):
    context = "\n\n".join(f"[{i+1}] (source: {s})\n{c}"
                          for i, (c, s, _) in enumerate(hits))
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=MODEL,
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return resp.content[0].text


@st.cache_resource(show_spinner=False)
def load_local_llm():
    from transformers import pipeline
    return pipeline("text2text-generation", model="google/flan-t5-base")


def answer_local(query, hits):
    """Generate an answer from retrieved text with a free local model."""
    gen = load_local_llm()
    context = " ".join(c for c, _, _ in hits[:2])
    prompt = (
        "Answer the question in a few clear sentences using only the context.\n\n"
        f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    )
    out = gen(prompt, max_new_tokens=150, num_beams=4, truncation=True)
    return out[0]["generated_text"].strip()


# ---------------- UI ----------------
st.set_page_config(page_title="AI Medical Assistant", page_icon="🩺")
st.title("🩺 AI Medical Assistant (RAG)")
st.warning(DISCLAIMER)

embedder, chunks, sources, vecs = build_index()
if not chunks:
    st.error(f"No documents found. Add .txt/.md files to the '{DATA_DIR}' folder and restart.")
    st.stop()

st.sidebar.write(f"Knowledge base: {len(set(sources))} files, {len(chunks)} chunks")

query = st.text_input("Ask a clinical question:")
if query:
    if any(w in query.lower() for w in EMERGENCY_WORDS):
        st.error("This may be an emergency. Seek immediate medical care.")
    hits = retrieve(query, embedder, chunks, sources, vecs)
    if hits[0][2] < MIN_SCORE:
        st.info("I don't have enough evidence in my knowledge base to answer this.")
    else:
        if os.getenv("ANTHROPIC_API_KEY"):
            with st.spinner("Thinking..."):
                st.markdown(answer(query, hits))
        else:
            local = None
            try:
                with st.spinner("Generating answer with local model (first run downloads it, please wait)..."):
                    local = answer_local(query, hits)
            except Exception as e:
                st.caption(f"Local model unavailable, showing extracted answer instead. ({e})")
            if local:
                st.success("Answer (generated locally from retrieved documents):")
                st.markdown(f"{local}\n\n*Sources: {', '.join(sorted(set(h[1] for h in hits[:2])))}*")
            else:
                st.success("Answer (extracted from the knowledge base, retrieval-only mode):")
                sentences = []
                for c, s, sc in hits[:2]:
                    for sent in c.replace("\n", " ").split(". "):
                        sent = sent.strip().rstrip(".")
                        if len(sent) > 25:
                            sentences.append((sent + ".", s))
                sv = embedder.encode([x[0] for x in sentences], normalize_embeddings=True)
                qv = embedder.encode([query], normalize_embeddings=True)[0]
                top = np.argsort(-(sv @ qv))[:3]
                for i in sorted(top):
                    st.markdown(f"- {sentences[i][0]} *[Source: {sentences[i][1]}]*")
    with st.expander("Retrieved sources"):
        for i, (c, s, sc) in enumerate(hits):
            st.markdown(f"**[{i+1}] {s}** (similarity {sc:.2f})")
            st.caption(c)
