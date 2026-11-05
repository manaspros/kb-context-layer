"""Step 5: paragraph -> document links from title mentions, and the graph-expansion retrieval test."""
import json
import re
from collections import Counter, defaultdict

import numpy as np

from kb.answer import BUDGET, load_questions, subset_questions
from kb.common import INDEX, RESULTS, read_jsonl
from kb.retrieval import QUERY_PREFIX, Retriever

TOKEN = re.compile(r"\w+")
MIN_LEN = 4
COMMON_DF = 20  # single-word title is dropped if its lowercase form is a word in >= this many paragraphs
SEED_COUNTS = (3, 5, 7, 9)
GRAPH_SEEDS = 7  # chosen on the tune split by `kb graph-eval`


def aliases(title: str) -> list[str]:
    stripped = re.sub(r"\s*\([^)]*\)$", "", title)
    return [title] + ([stripped] if stripped != title else [])


def build() -> None:
    corpus = read_jsonl(INDEX / "corpus.jsonl")
    toks = [[(m.group(), m.start(), m.end()) for m in TOKEN.finditer(p["text"])] for p in corpus]
    df = Counter(w for ts in toks for w in {t for t, _, _ in ts if t.islower()})
    df_any = Counter(w for ts in toks for w in {t for t, _, _ in ts})
    # ponytail: exact case-sensitive title match with a corpus-frequency stoplist for one-word titles.
    # Ceiling: misses surface variants ("the Beatles" vs "The Beatles", "Obama" vs "Barack Obama") and
    # still links ambiguous short titles; upgrade to an alias table from redirects + a disambiguator.
    full = {p["title"]: p["doc_id"] for p in corpus}
    table: dict[str, str] = {}
    strip_docs: dict[str, set[str]] = defaultdict(set)
    for title, d in full.items():
        table[title] = d
        for a in aliases(title)[1:]:
            strip_docs[a].add(d)
    for a, ds in strip_docs.items():  # parenthetical-stripped alias only when unambiguous and not itself a title
        one_word = len(TOKEN.findall(a)) == 1  # "John (song)" -> "John" would link every John
        if a not in full and len(ds) == 1 and not (one_word and df_any[a] >= COMMON_DF):
            table[a] = next(iter(ds))
    keyed: dict[tuple, dict[str, str]] = {}
    for a, d in table.items():
        k = tuple(TOKEN.findall(a))
        bad = (len(a) < MIN_LEN or re.fullmatch(r"[\d\s.,-]+", a) or not k
               or (len(k) == 1 and df[a.lower()] >= COMMON_DF))
        if not bad:
            keyed.setdefault(k, {})[a] = d
    lens: dict[str, list[int]] = defaultdict(list)  # first token -> candidate title lengths, longest first
    for k in keyed:
        lens[k[0]].append(len(k))
    for v in lens.values():
        v.sort(reverse=True)

    graph: dict[str, list[str]] = {}
    for p, ts in zip(corpus, toks):
        found: set[str] = set()
        i = 0
        while i < len(ts):
            hit = 0
            for n in lens.get(ts[i][0], ()):
                if i + n <= len(ts):
                    cand = keyed.get(tuple(t for t, _, _ in ts[i:i + n]))
                    surface = p["text"][ts[i][1]:ts[i + n - 1][2]]
                    if cand and surface in cand:  # word-boundary by token, exact case by string
                        found.add(cand[surface])
                        hit = n
                        break
            i += hit or 1
        found.discard(p["doc_id"])
        graph[p["para_id"]] = sorted(found)
    (INDEX / "graph.json").write_text(json.dumps(graph))
    edges = sum(map(len, graph.values()))
    print(f"paragraphs={len(graph)} aliases kept={sum(len(v) for v in keyed.values())} edges={edges} "
          f"mean out-degree={edges / len(graph):.2f} "
          f"paragraphs with >=1 link={sum(bool(v) for v in graph.values()) / len(graph):.1%}")


class Graph:
    """Graph expansion: RRF top-s seeds + paragraphs of linked docs ranked by dense sim to the query."""

    def __init__(self, r: Retriever) -> None:
        corpus = read_jsonl(INDEX / "corpus.jsonl")
        self.r, self.links = r, json.loads((INDEX / "graph.json").read_text())
        self.row_of = {p["para_id"]: i for i, p in enumerate(corpus)}
        self.paras_of: dict[str, list[str]] = defaultdict(list)
        for p in corpus:
            self.paras_of[p["doc_id"]].append(p["para_id"])

    def expand(self, top: list[str], qv: np.ndarray, s: int) -> list[str]:
        seeds = top[:s]
        cands = sorted({c for p in seeds for d in self.links[p] for c in self.paras_of[d]} - set(seeds))
        sims = self.r.emb[[self.row_of[c] for c in cands]] @ qv if cands else []
        ranked = [c for _, c in sorted(zip(-np.asarray(sims), cands))]
        return list(dict.fromkeys(seeds + ranked + top[s:]))


def measure(r: Retriever, g: Graph, qs: list[dict], seeds: tuple[int, ...]) -> dict:
    """Retrieval-only test of graph expansion against plain RRF, same 10-paragraph budget."""
    rrf = r.search([q["question"] for q in qs])["rrf"]
    qv = r.model.encode([QUERY_PREFIX + q["question"] for q in qs], normalize_embeddings=True)
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q, v, rk in zip(qs, qv, rrf):
        gold, top = set(q["gold_doc_ids"]), [r.para_ids[i] for i in rk]
        variants = {"rrf": top[:BUDGET]}
        for s in seeds:
            variants[f"graph_s{s}"] = g.expand(top, v, s)[:BUDGET]
        reach = {p.split(":")[0] for p in top[:BUDGET]} | {d for p in top[:BUDGET] for d in g.links[p]}
        variants["reachable_ceiling"] = [f"{d}:0" for d in reach]  # every doc one link from top-10
        for name, ids in variants.items():
            docs = {p.split(":")[0] for p in ids}
            for h in ("all", str(q["hop"])):
                acc[name][f"recall_{h}"].append(len(gold & docs) / len(gold))
                acc[name][f"allgold_{h}"].append(float(gold <= docs))
    return {n: {k: round(float(np.mean(v)), 4) for k, v in sorted(m.items())} for n, m in acc.items()}


def evaluate() -> None:
    """Pick the seed count on the tune split only, then report eval at that value (and at rrf)."""
    r = Retriever()
    g = Graph(r)
    tune = measure(r, g, load_questions("tune_ids.txt", 200), SEED_COUNTS)
    best = max(SEED_COUNTS, key=lambda s: (tune[f"graph_s{s}"]["allgold_all"], tune[f"graph_s{s}"]["recall_all"]))
    ev = measure(r, g, subset_questions(), (best,))
    summary = {"chosen_seeds_on_tune": best, "tune": tune, "eval": ev}
    (RESULTS / "graph").mkdir(parents=True, exist_ok=True)
    (RESULTS / "graph" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for split, res in (("tune (n=200)", tune), ("eval (n=400)", ev)):
        print(f"\n{split}: {'variant':18} " + " ".join(f"{m}[{h}]".ljust(10) for m in ("recall", "allgold") for h in ("all", "2", "3", "4")))
        for n, m in res.items():
            print(f"{'':14} {n:18} " + "".join(f"{m[f'{k}_{h}']:.3f}".ljust(11) for k in ("recall", "allgold") for h in ("all", "2", "3", "4")))
    print(f"\nchosen graph seeds (best tune all-gold@10): {best}")
