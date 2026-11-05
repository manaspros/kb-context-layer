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
TUNE_SEED = 7
TUNE_PER_HOP = {2: 80, 3: 70, 4: 50}  # disjoint from the eval subset; every hyperparameter is chosen here
TUNE = ROOT / "data" / "tune_ids.txt"
TUNE_SHA = ROOT / "data" / "tune_ids.sha256"
UNANS_SEED = 11
UNANS_PER_HOP = 20
UNANS = ROOT / "data" / "unanswerable.jsonl"
UNANS_SHA = ROOT / "data" / "unanswerable.sha256"


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

    # Subset and tune split: seeded samples from sorted ids per hop stratum (tune excludes the subset).
    def sample(seed: int, per_hop: dict[int, int], exclude: set[str], path: Path, sha: Path) -> list[str]:
        rng = random.Random(seed)
        chosen = []
        for hop, n in per_hop.items():
            ids = sorted(q["id"] for q in questions if q["hop"] == hop and q["id"] not in exclude)
            chosen += rng.sample(ids, n)
        text = "\n".join(sorted(chosen)) + "\n"
        if path.exists():
            if path.read_text() != text:
                sys.exit(f"FAIL: regenerated ids differ from committed {path.name}")
        else:
            path.write_text(text)
            sha.write_text(sha256(path) + "\n")
        if sha256(path) != sha.read_text().strip():
            sys.exit(f"FAIL: {path.name} sha256 does not match {sha.name}")
        return chosen

    chosen = sample(SEED, PER_HOP, set(), SUBSET, SUBSET_SHA)
    tune = sample(TUNE_SEED, TUNE_PER_HOP, set(chosen), TUNE, TUNE_SHA)
    assert not set(tune) & set(chosen)

    # Unanswerable probe: dev questions outside eval and tune; mask every paragraph of the document that
    # supports the LAST decomposition step (how MuSiQue-Full builds its unanswerable contrasts).
    last_doc = {r["id"]: doc_id(r["paragraphs"][r["question_decomposition"][-1]["paragraph_support_idx"]]["title"]) for r in rows}
    paras_of: dict[str, list[str]] = {}
    for c in corpus:
        paras_of.setdefault(c["doc_id"], []).append(c["para_id"])
    rng, pool, unans = random.Random(UNANS_SEED), set(chosen) | set(tune), []
    for hop in PER_HOP:
        ids = sorted(q["id"] for q in questions if q["hop"] == hop and q["id"] not in pool)
        unans += [{"id": i, "hop": hop, "masked_doc_id": last_doc[i], "masked_para_ids": paras_of[last_doc[i]]}
                  for i in rng.sample(ids, UNANS_PER_HOP)]
    write_jsonl(UNANS, unans)
    if sha256(UNANS) != UNANS_SHA.read_text().strip():
        sys.exit(f"FAIL: regenerated {UNANS.name} does not match {UNANS_SHA.name}")

    sub = set(chosen)
    docs_per_q = [len(q["gold_doc_ids"]) for q in questions]
    print(f"rows={len(rows)} multi-doc questions={len(questions)} excluded single-doc={single_doc}")
    print(f"documents={len(by_title)} paragraphs={len(corpus)}")
    print("hop counts (multi-doc):", dict(sorted(Counter(q["hop"] for q in questions).items())))
    print(f"gold docs/question: mean={sum(docs_per_q)/len(docs_per_q):.2f} max={max(docs_per_q)}")
    print(f"subset={len(sub)} sha256={SUBSET_SHA.read_text().strip()} hops=",
          dict(sorted(Counter(int(i[0]) for i in sub).items())))
    print(f"unanswerable probe={len(unans)} masked paragraphs={sum(len(u['masked_para_ids']) for u in unans)} -> {UNANS.name}")
    print(f"tune={len(tune)} sha256={TUNE_SHA.read_text().strip()} hops=",
          dict(sorted(Counter(int(i[0]) for i in tune).items())), "overlap with subset=0")
