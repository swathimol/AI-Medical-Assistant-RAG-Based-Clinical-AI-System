"""Evaluate the RAG retrieval pipeline across four configurations.

Run:  python evaluate.py
Output: results.md (table) and results.csv

Configurations (2 x 2 design):
  A  all-MiniLM-L6-v2            no reranker
  B  all-MiniLM-L6-v2            + cross-encoder reranker
  C  BAAI/bge-small-en-v1.5      no reranker
  D  BAAI/bge-small-en-v1.5      + cross-encoder reranker

Metrics (retrieval is judged at chunk level: a hit means the retrieved chunk
contains at least one of the expected answer keywords):
  Hit@1, Hit@K, MRR        - did retrieval find the evidence?
  Out-of-scope refusal     - share of unanswerable questions correctly refused
  False-answer rate        - share of unanswerable questions that were answered
                             anyway (a proxy for hallucination risk; this is
                             NOT a measurement of LLM-generated text)

The evidence threshold is tuned on this same small test set, so the refusal
numbers are indicative only. Add your own questions to make it more reliable.
"""
import csv
import numpy as np
from rag_core import RAG, DEFAULT_RERANKER

K = 4

# (question, [answer keywords - any one must appear in the retrieved chunk])
IN_SCOPE = [
    ("What HbA1c level indicates diabetes?", ["6.5%"]),
    ("Which blood tests diagnose type 2 diabetes?", ["fasting plasma glucose", "oral glucose tolerance"]),
    ("What are the risk factors for type 2 diabetes?", ["family history", "physical inactivity"]),
    ("What are the long-term complications of diabetes?", ["kidney damage", "nerve damage"]),
    ("What blood pressure reading is considered high?", ["130/80", "140/90"]),
    ("Why is hypertension called the silent killer?", ["silent killer"]),
    ("What lifestyle changes help lower blood pressure?", ["salt intake", "limiting alcohol"]),
    ("What are the risks of untreated high blood pressure?", ["heart failure"]),
    ("What triggers asthma?", ["pollen", "dust mites"]),
    ("How is asthma diagnosed?", ["spirometry"]),
    ("What medicines are used for persistent asthma?", ["inhaled corticosteroids"]),
    ("When is an asthma episode an emergency?", ["blue lips"]),
    ("What are the symptoms of iron deficiency anemia?", ["pale skin", "brittle nails"]),
    ("What causes iron deficiency anemia?", ["blood loss", "inadequate dietary iron"]),
    ("Which foods are rich in iron?", ["red meat", "lentils"]),
    ("What affects iron absorption?", ["vitamin c", "tea and coffee"]),
    ("How long does a common cold last?", ["7 to 10 days"]),
    ("How does influenza differ from a cold?", ["start suddenly", "chills"]),
    ("Do antibiotics work against the flu?", ["antibiotics do not work"]),
    ("Who is at risk of flu complications?", ["older adults", "pregnant"]),
    ("How is dengue spread?", ["aedes"]),
    ("What are the warning signs of severe dengue?", ["severe abdominal pain", "bleeding gums"]),
    ("Which painkillers should be avoided in dengue?", ["aspirin", "ibuprofen"]),
    ("How can dengue be prevented?", ["standing water", "mosquito repellent"]),
    ("How long after a mosquito bite do dengue symptoms start?", ["4 to 10 days"]),
]

# Questions the knowledge base cannot answer - the system should refuse.
OUT_OF_SCOPE = [
    "How do I treat a broken leg?",
    "What is the capital of France?",
    "How do I treat a snake bite?",
    "What is the treatment for breast cancer?",
    "How do I bake a chocolate cake?",
    "What is the amoxicillin dose for a child?",
    "How is tuberculosis treated?",
    "Who won the cricket world cup?",
]

CONFIGS = [
    {"name": "A", "embed": "all-MiniLM-L6-v2", "rerank": None},
    {"name": "B", "embed": "all-MiniLM-L6-v2", "rerank": DEFAULT_RERANKER},
    {"name": "C", "embed": "BAAI/bge-small-en-v1.5", "rerank": None},
    {"name": "D", "embed": "BAAI/bge-small-en-v1.5", "rerank": DEFAULT_RERANKER},
]


def evaluate_config(rag, k=K):
    """Return a metrics dict for one RAG object."""
    hit1 = hitk = rr = 0.0
    in_sims = []
    for q, kws in IN_SCOPE:
        hits, top_sim = rag.retrieve(q, k=k)
        in_sims.append(top_sim)
        flags = [any(kw in h["text"].lower() for kw in kws) for h in hits]
        hit1 += bool(flags and flags[0])
        hitk += any(flags)
        rr += next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0)
    out_sims = [rag.retrieve(q, k=k)[1] for q in OUT_OF_SCOPE]

    n = len(IN_SCOPE)
    best = None
    for t in np.linspace(0.05, 0.95, 91):
        answered = float(np.mean([s >= t for s in in_sims]))
        refused = float(np.mean([s < t for s in out_sims]))
        bal = (answered + refused) / 2
        if best is None or bal > best[0] + 1e-9:
            best = (bal, float(t), answered, refused)
    _, t, answered, refused = best
    return {
        "hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n,
        "threshold": round(t, 2), "in_scope_answered": answered,
        "out_of_scope_refused": refused, "false_answer_rate": 1 - refused,
        "chunks": len(rag.chunks),
    }


def main():
    rows = []
    for cfg in CONFIGS:
        print(f"Running config {cfg['name']}: {cfg['embed']} | rerank={bool(cfg['rerank'])}")
        rag = RAG(embed_model=cfg["embed"], reranker=cfg["rerank"])
        m = evaluate_config(rag)
        m.update(config=cfg["name"], embedding=cfg["embed"],
                 reranker="yes" if cfg["rerank"] else "no")
        rows.append(m)

    cols = ["config", "embedding", "reranker", "hit@1", f"hit@{K}", "mrr",
            "threshold", "in_scope_answered", "out_of_scope_refused", "false_answer_rate"]
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    pct = lambda x: f"{x * 100:.0f}%"
    lines = [
        f"Test set: {len(IN_SCOPE)} in-scope and {len(OUT_OF_SCOPE)} out-of-scope questions, "
        f"{rows[0]['chunks']} chunks, K={K}.\n",
        "| Config | Embedding | Reranker | Hit@1 | Hit@%d | MRR | Threshold | In-scope answered | Out-of-scope refused | False-answer rate |" % K,
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['config']} | {r['embedding']} | {r['reranker']} | {pct(r['hit@1'])} | "
                     f"{pct(r[f'hit@{K}'])} | {r['mrr']:.2f} | {r['threshold']} | "
                     f"{pct(r['in_scope_answered'])} | {pct(r['out_of_scope_refused'])} | "
                     f"{pct(r['false_answer_rate'])} |")
    table = "\n".join(lines)
    with open("results.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    print("\nSaved results.md and results.csv")


if __name__ == "__main__":
    main()
