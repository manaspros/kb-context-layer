"""Step 1: pinned MuSiQue-Ans dev download, corpus, questions, seeded subset."""
import hashlib
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from kb.common import INDEX, RAW, ROOT, read_jsonl, write_jsonl

URL = (
    "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/"
    "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321/musique_ans_v1.0_dev.jsonl"
)
RAW_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
SEED = 1337
PER_HOP = {2: 150, 3: 150, 4: 100}
SUBSET = ROOT / "data" / "subset_ids.txt"
SUBSET_SHA = ROOT / "data" / "subset_ids.sha256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_id(title: str) -> str:
    return hashlib.sha1(title.encode()).hexdigest()[:10]


def download() -> Path:
    path = RAW / "musique_ans_v1.0_dev.jsonl"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    if sha256(path) != RAW_SHA256:
        sys.exit(f"FAIL: {path} sha256 mismatch (expected {RAW_SHA256})")
    return path


def run() -> None:
    rows = read_jsonl(download())

    # Corpus: every unique (title, text) paragraph, grouped by title.
    by_title: dict[str, set[str]] = {}
    for r in rows:
        for p in r["paragraphs"]:
            by_title.setdefault(p["title"], set()).add(p["paragraph_text"])
    corpus, pid = [], {}
    for title in sorted(by_title):
        d = doc_id(title)
        for i, text in enumerate(sorted(by_title[title])):
            pid[(title, text)] = f"{d}:{i}"
            corpus.append({"para_id": f"{d}:{i}", "doc_id": d, "title": title, "text": text})
    assert len({c["doc_id"] for c in corpus}) == len(by_title), "doc_id collision"
    write_jsonl(INDEX / "corpus.jsonl", corpus)

    questions, single_doc = [], 0
    for r in rows:
        gold = [pid[(p["title"], p["paragraph_text"])] for p in r["paragraphs"] if p["is_supporting"]]
        gold_docs = sorted({g.split(":")[0] for g in gold})
        if len(gold_docs) < 2:
            single_doc += 1
            continue
        questions.append({
            "id": r["id"], "hop": int(r["id"][0]), "question": r["question"],
            "answer": r["answer"], "aliases": r["answer_aliases"],
            "gold_para_ids": sorted(set(gold)), "gold_doc_ids": gold_docs,
            "decomposition": r["question_decomposition"],
        })
    write_jsonl(INDEX / "questions.jsonl", questions)

    # Subset: seeded sample from sorted ids per hop stratum.
    rng = random.Random(SEED)
    chosen = []
    for hop, n in PER_HOP.items():
        ids = sorted(q["id"] for q in questions if q["hop"] == hop)
        chosen += rng.sample(ids, n)
    text = "\n".join(sorted(chosen)) + "\n"
    if SUBSET.exists():
        if SUBSET.read_text() != text:
            sys.exit("FAIL: regenerated subset differs from committed data/subset_ids.txt")
    else:
        SUBSET.write_text(text)
        SUBSET_SHA.write_text(sha256(SUBSET) + "\n")
    if sha256(SUBSET) != SUBSET_SHA.read_text().strip():
        sys.exit("FAIL: subset_ids.txt sha256 does not match data/subset_ids.sha256")

    sub = set(chosen)
    docs_per_q = [len(q["gold_doc_ids"]) for q in questions]
    print(f"rows={len(rows)} multi-doc questions={len(questions)} excluded single-doc={single_doc}")
    print(f"documents={len(by_title)} paragraphs={len(corpus)}")
    print("hop counts (multi-doc):", dict(sorted(Counter(q["hop"] for q in questions).items())))
    print(f"gold docs/question: mean={sum(docs_per_q)/len(docs_per_q):.2f} max={max(docs_per_q)}")
    print(f"subset={len(sub)} sha256={SUBSET_SHA.read_text().strip()} hops=",
          dict(sorted(Counter(int(i[0]) for i in sub).items())))
