"""Core RAG pipeline for the AI Medical Assistant (educational use only).

Chunking -> embeddings -> FAISS vector index -> optional cross-encoder
reranking -> evidence threshold -> answer generation.
"""
import os
import re
import glob
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CLAUDE_MODEL = "claude-sonnet-5-5"
LOCAL_MODEL = "google/flan-t5-base"
DEFAULT_EMBED = "all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DISCLAIMER = (
    "Educational decision-support tool only. Not a diagnosis or treatment "
    "advice. In an emergency, call your local emergency number."
)
EMERGENCY_WORDS = ["chest pain", "can't breathe", "cannot breathe", "stroke",
                   "suicide", "overdose", "severe bleeding", "unconscious"]
NO_EVIDENCE = "I don't have enough evidence in my knowledge base to answer this."

SYSTEM_PROMPT = f"""You are a clinical information assistant for education.
Answer ONLY using the numbered context passages provided.
- Cite sources inline like [1], [2].
- If the context does not contain the answer, reply exactly:
  "{NO_EVIDENCE}"
- Never invent doses, drugs, or statistics.
- Be concise and structured. Recommend consulting a clinician where relevant."""


# ---------------------------------------------------------------- chunking
def chunk_sentences(text, size=60, overlap=1):
    """Pack whole sentences into chunks of about `size` words.
    `overlap` = number of sentences repeated at the start of the next chunk."""
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
    chunks, cur, count = [], [], 0
    for sent in sents:
        w = len(sent.split())
        if cur and count + w > size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            count = sum(len(x.split()) for x in cur)
        cur.append(sent)
        count += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def load_chunks(data_dir=DATA_DIR, size=60, overlap=1):
    """Read .txt/.md files; split by blank lines (topics), then into windows.
    An ALL-CAPS first line is treated as a heading and added to every chunk."""
    chunks, sources = [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not path.endswith((".txt", ".md")):
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            blocks = [b.strip() for b in f.read().split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.split("\n")
            title = ""
            if len(lines) > 1 and lines[0].isupper() and len(lines[0]) < 60:
                title, block = lines[0].title(), " ".join(lines[1:])
            for piece in chunk_sentences(block, size, overlap):
                chunks.append(f"{title}: {piece}" if title else piece)
                sources.append(os.path.basename(path))
    return chunks, sources


# --------------------------------------------------------------- retrieval
class RAG:
    def __init__(self, embed_model=DEFAULT_EMBED, reranker=None,
                 data_dir=DATA_DIR, chunk_size=60, overlap=1,
                 embedder_obj=None, reranker_obj=None):
        """reranker: cross-encoder model name or None.
        embedder_obj / reranker_obj let tests inject stand-in models."""
        import faiss
        if embedder_obj is not None:
            self.embedder = embedder_obj
        else:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embed_model)
        if reranker_obj is not None:
            self.reranker = reranker_obj
        elif reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker)
        else:
            self.reranker = None

        self.chunks, self.sources = load_chunks(data_dir, chunk_size, overlap)
        self.index = None
        if self.chunks:
            vecs = self._embed(self.chunks)
            self.index = faiss.IndexFlatIP(vecs.shape[1])  # cosine (normalized)
            self.index.add(vecs)

    def _embed(self, texts):
        v = self.embedder.encode(texts, normalize_embeddings=True)
        return np.ascontiguousarray(np.asarray(v, dtype="float32"))

    def retrieve(self, query, k=4, candidates=10):
        """Return (hits, top_sim). hits: dicts with text/source/sim/score.
        top_sim is the best vector-search cosine similarity, used as the
        evidence gate (independent of the reranker's score scale)."""
        if self.index is None:
            return [], 0.0
        n = max(k, candidates) if self.reranker else k
        n = min(n, len(self.chunks))
        sims, idx = self.index.search(self._embed([query]), n)
        hits = [{"text": self.chunks[i], "source": self.sources[i],
                 "sim": float(s), "score": float(s)}
                for s, i in zip(sims[0], idx[0]) if i >= 0]
        top_sim = max((h["sim"] for h in hits), default=0.0)
        if self.reranker and hits:
            scores = self.reranker.predict([(query, h["text"]) for h in hits])
            for h, sc in zip(hits, scores):
                h["score"] = float(sc)
            hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k], top_sim

    def extract(self, query, hits, n=3):
        """Extractive answer: the n sentences most similar to the query."""
        sents = []
        for h in hits[:2]:
            body = re.sub(r"^[^:.]{1,60}: ", "", h["text"])
            for s in re.split(r"(?<=[.!?])\s+", body):
                s = s.strip()
                if len(s) > 25:
                    sents.append((s, h["source"]))
        if not sents:
            return []
        sv = self._embed([s for s, _ in sents])
        qv = self._embed([query])[0]
        top = np.argsort(-(sv @ qv))[:n]
        return [sents[i] for i in sorted(top)]


# -------------------------------------------------------------- generation
_LOCAL = None


def _local_llm():
    global _LOCAL
    if _LOCAL is None:
        from transformers import pipeline
        _LOCAL = pipeline("text2text-generation", model=LOCAL_MODEL)
    return _LOCAL


def generate(rag, query, hits, api_key=None, use_local=True):
    """Return (answer_markdown, mode, error). mode: claude | local | extractive."""
    error = None
    if api_key:
        try:
            import anthropic
            context = "\n\n".join(f"[{i+1}] (source: {h['source']})\n{h['text']}"
                                  for i, h in enumerate(hits))
            resp = anthropic.Anthropic(api_key=api_key).messages.create(
                model=CLAUDE_MODEL, max_tokens=700, system=SYSTEM_PROMPT,
                messages=[{"role": "user",
                           "content": f"Context:\n{context}\n\nQuestion: {query}"}])
            return resp.content[0].text, "claude", None
        except Exception as e:
            error = f"Claude API failed ({e}); using fallback."
    if use_local:
        try:
            context = " ".join(h["text"] for h in hits[:2])
            prompt = ("Answer the question in a few clear sentences using only "
                      f"the context.\n\nQuestion: {query}\n\nContext: {context}\n\nAnswer:")
            out = _local_llm()(prompt, max_new_tokens=150, num_beams=4, truncation=True)
            srcs = ", ".join(sorted({h["source"] for h in hits[:2]}))
            return f"{out[0]['generated_text'].strip()}\n\n*Sources: {srcs}*", "local", error
        except Exception as e:
            error = (error + " " if error else "") + f"Local model unavailable ({e})."
    lines = [f"- {s} *[Source: {src}]*" for s, src in rag.extract(query, hits)]
    return "\n".join(lines), "extractive", error
