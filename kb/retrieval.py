"""Step 3: BM25 + dense (bge-small) + RRF over paragraphs, and the recall@k report."""
import json
import time

import bm25s
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from kb.common import INDEX, RESULTS, ROOT, read_jsonl, write_jsonl

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
DEPTH = 100  # per-list depth fed into RRF
KS = (5, 10, 20)
METHODS = ("bm25", "dense", "rrf")


def passages() -> list[str]:
    return [f"{p['title']}. {p['text']}" for p in read_jsonl(INDEX / "corpus.jsonl")]


def build() -> None:
    texts = passages()
    t0 = time.time()
    bm = bm25s.BM25()
    bm.index(bm25s.tokenize(texts, stopwords="en"))
    bm.save(INDEX / "bm25")
    t_bm = time.time() - t0

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)
    t0 = time.time()
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    np.save(INDEX / "dense.npy", emb.astype(np.float32))
    t_dense = time.time() - t0
    print(f"bm25 build {t_bm:.1f}s; dense build {t_dense:.1f}s on {device}; emb shape {emb.shape}")


class Retriever:
    def __init__(self) -> None:
        self.para_ids = [p["para_id"] for p in read_jsonl(INDEX / "corpus.jsonl")]
        self.bm = bm25s.BM25.load(INDEX / "bm25")
        self.emb = np.load(INDEX / "dense.npy")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL, device=device)

    def search(self, queries: list[str]) -> dict[str, list[list[int]]]:
        """Return top-DEPTH corpus row indices per query for bm25, dense and rrf."""
        idx, _ = self.bm.retrieve(bm25s.tokenize(queries, stopwords="en", show_progress=False),
                                  k=DEPTH, show_progress=False)
        q = self.model.encode([QUERY_PREFIX + s for s in queries], normalize_embeddings=True)
        sims = q @ self.emb.T  # 400 x 21100 matmul, no ANN needed
        dense = np.argsort(-sims, axis=1)[:, :DEPTH]
        bm_l, de_l = idx.tolist(), dense.tolist()
        return {"bm25": bm_l, "dense": de_l, "rrf": [rrf(a, b) for a, b in zip(bm_l, de_l)]}


def rrf(*rankings: list[int]) -> list[int]:
    score: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(score, key=lambda i: (-score[i], i))[:DEPTH]


def recall() -> None:
    subset = set((ROOT / "data" / "subset_ids.txt").read_text().split())
    qs = sorted((q for q in read_jsonl(INDEX / "questions.jsonl") if q["id"] in subset),
                key=lambda q: q["id"])
    assert len(qs) == 400
    r = Retriever()
    t0 = time.time()
    ranked = r.search([q["question"] for q in qs])
    print(f"retrieval for {len(qs)} questions: {time.time() - t0:.1f}s")

    summary: dict = {}
    for m in METHODS:
        rows, acc = [], {}
        for q, idxs in zip(qs, ranked[m]):
            pids = [r.para_ids[i] for i in idxs[: max(KS)]]
            rows.append({"id": q["id"], "hop": q["hop"], "retrieved_para_ids": pids})
            gold_d, gold_p = set(q["gold_doc_ids"]), set(q["gold_para_ids"])
            for k in KS:
                top = pids[:k]
                docs = {p.split(":")[0] for p in top}
                for hop in ("all", str(q["hop"])):
                    a = acc.setdefault((hop, k), [[], [], []])
                    a[0].append(len(gold_d & docs) / len(gold_d))
                    a[1].append(float(gold_d <= docs))
                    a[2].append(len(gold_p & set(top)) / len(gold_p))
        write_jsonl(RESULTS / "retrieval" / f"{m}.jsonl", rows)
        summary[m] = {
            f"hop_{hop}" if hop != "all" else "all": {
                f"k{k}": {"n": len(v[0]), "gold_doc_recall": round(float(np.mean(v[0])), 4),
                          "all_gold_docs": round(float(np.mean(v[1])), 4),
                          "gold_para_recall": round(float(np.mean(v[2])), 4)}
                for (h, k), v in sorted(acc.items()) if h == hop}
            for hop in ("all", "2", "3", "4")}
    (RESULTS / "retrieval" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{'method':6} {'hop':4} {'k':>2} | doc-recall all-gold para-recall")
    for m in METHODS:
        for hop, d in summary[m].items():
            for k, v in d.items():
                print(f"{m:6} {hop.removeprefix('hop_'):4} {k[1:]:>2} | "
                      f"{v['gold_doc_recall']:.3f}      {v['all_gold_docs']:.3f}    {v['gold_para_recall']:.3f}")
