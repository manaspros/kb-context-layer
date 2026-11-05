"""Step 8: claim-level support check with a local NLI model (MiniCheck-RoBERTa-Large); no LLM tokens.

Per claim: premise = the cited spans' text joined by a space, hypothesis = the claim text; the model returns
P(supported). ALCE-style metrics (Gao et al. 2023), claim = statement:
- citation recall: share of claims whose full cited set supports the claim (P >= 0.5).
- citation precision: share of citations that are relevant. For a claim whose full set is supported, citation i is
  relevant if it supports the claim alone or if the claim is no longer supported without it; for a claim whose full
  set is not supported, every citation counts as irrelevant.
- unsupported claim rate = 1 - citation recall (a claim without any valid cited span is unsupported).
Scores are cached per system in results/verify/<system>.jsonl, keyed by (claim text, cited span ids); `kb score` only
reads those files.
"""
import sys
import time

from kb.answer import KB
from kb.common import RESULTS, read_jsonl, write_jsonl

MODEL = "lytang/MiniCheck-RoBERTa-Large"
REVISION = "74c8919647e61ed0f71bc177d94f10930f090068"  # pinned HF commit
THRESHOLD = 0.5
CHUNK_TOKENS = 400  # premise longer than this is split and scored per chunk, max taken (MiniCheck's own convention)
BATCH = 16
VERIFY_SYSTEMS = ("bm25", "dense", "rrf", "kb", "kb_hop0", "kb_no_lookup", "kb_no_graph")
DIR = RESULTS / "verify"


def claim_jobs(rec: dict, kb: KB) -> list[tuple[int, str, tuple[str, ...]]]:
    """(claim index, claim text, cited valid span ids) for every claim; spans must be in the answer's context."""
    ctx = {s["span_id"] for p in rec["context_para_ids"] for s in kb.spans_of[p]}
    out = []
    for i, c in enumerate((rec["parsed"] or {}).get("claims", [])):
        if not isinstance(c, dict):
            continue
        ids = tuple(dict.fromkeys(x for x in (c.get("span_ids") or []) if isinstance(x, str) and x in ctx))
        out.append((i, str(c.get("text") or ""), ids))
    return out


def subsets(ids: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Span sets that must be scored: the full set, and for 2+ citations each single span and each leave-one-out."""
    if not ids:
        return []
    out = [ids]
    if len(ids) > 1:
        out += [(i,) for i in ids] + [tuple(j for j in ids if j != i) for i in ids]
    return list(dict.fromkeys(out))


class Verifier:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).to(self.device).eval()

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(supported) per (premise, claim); long premises are chunked and the max chunk score is kept."""
        items: list[tuple[int, str, str]] = []
        for n, (prem, claim) in enumerate(pairs):
            ids = self.tok(prem, add_special_tokens=False)["input_ids"]
            for a in range(0, max(len(ids), 1), CHUNK_TOKENS):
                items.append((n, self.tok.decode(ids[a:a + CHUNK_TOKENS]), claim))
        items.sort(key=lambda x: len(x[1]) + len(x[2]))
        best = [0.0] * len(pairs)
        for a in range(0, len(items), BATCH):
            b = items[a:a + BATCH]
            enc = self.tok([x[1] for x in b], [x[2] for x in b], truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                p = self.model(**enc).logits.float().softmax(-1)[:, 1].cpu().tolist()
            for (n, _, _), v in zip(b, p):
                best[n] = max(best[n], v)
        return best


def run() -> None:
    """python -m kb verify [system ...]: score every missing (claim, cited spans) pair for the given systems
    (default: all seven RAG systems) and write results/verify/<system>.jsonl."""
    systems = sys.argv[2:] or list(VERIFY_SYSTEMS)
    kb, ver = KB(), None
    for s in systems:
        t0 = time.time()
        path = DIR / f"{s}.jsonl"
        have = {(r["id"], r["claim_idx"], tuple(r["spans"])): r for r in read_jsonl(path)} if path.exists() else {}
        need = [(rec["id"], i, text, sub) for rec in read_jsonl(RESULTS / "answers" / f"{s}.jsonl")
                for i, text, ids in claim_jobs(rec, kb) for sub in subsets(ids)]
        have = {k: v for k, v in have.items() if k in {(a, b, d) for a, b, _, d in need}}  # drop stale rows
        todo = [x for x in need if (x[0], x[1], x[3]) not in have]
        if todo:
            ver = ver or Verifier()
            ps = ver.score([(" ".join(kb.span[i]["text"] for i in sub), text) for _, _, text, sub in todo])
            for (qid, i, text, sub), p in zip(todo, ps):
                have[(qid, i, sub)] = {"id": qid, "claim_idx": i, "claim": text, "spans": list(sub), "p": round(p, 5)}
        write_jsonl(path, [have[k] for k in sorted(have)])
        print(f"{s}: {len(todo)} new scores, {len(have)} cached, {time.time() - t0:.0f}s on "
              f"{ver.device if ver else 'cache'}")
