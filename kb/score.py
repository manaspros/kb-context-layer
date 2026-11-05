"""Recompute every answer metric from results/answers/*.jsonl alone (no LLM calls)."""
import json
import re
import string
from collections import Counter

import numpy as np

from kb.answer import KB, subset_questions
from kb.common import RESULTS, read_jsonl

SYSTEMS = ("closed", "bm25", "dense", "rrf", "kb_hop0", "kb_no_lookup", "kb")
HOPS = ("all", "2", "3", "4")
BOOT_N, BOOT_SEED = 10_000, 0


def norm(s: str) -> str:
    """Official MuSiQue / SQuAD answer normalisation."""
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g:
        return float(p == g)
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def question_metrics(rec: dict, q: dict, kb: KB) -> dict:
    golds = [q["answer"], *q["aliases"]]
    ans = (rec["parsed"] or {}).get("answer", "")
    ctx = set(rec["context_para_ids"])
    ctx_spans = {s["span_id"] for p in ctx for s in kb.spans_of[p]}
    gold_d = set(q["gold_doc_ids"])
    ctx_docs = {p.split(":")[0] for p in ctx}
    claims = (rec["parsed"] or {}).get("claims", [])
    ids_ok = quote_ok = 0
    cited: set[str] = set()
    for c in claims:
        ids = [i for i in (c.get("span_ids") or []) if isinstance(i, str)]
        valid = [i for i in ids if i in ctx_spans]
        ids_ok += bool(ids) and len(valid) == len(ids)
        quote = c.get("quote")
        quote_ok += bool(isinstance(quote, str) and quote and any(quote in kb.span[i]["text"] for i in valid))
        cited |= {i.split(":")[0] for i in valid}
    n_calls = len(rec["usage"])
    return {
        "em": float(any(norm(ans) == norm(g) for g in golds)),
        "f1": max(f1(ans, g) for g in golds),
        "abstain": float(ans.strip().upper() == "NOT ANSWERABLE"),
        "parse_err": float(rec["parse_error"] is not None),
        "all_gold_docs": float(gold_d <= ctx_docs),
        "gold_doc_recall": len(gold_d & ctx_docs) / len(gold_d),
        "cited2": float(len(gold_d & cited) >= 2),
        "cited_gold_recall": len(gold_d & cited) / len(gold_d),
        "claims": len(claims), "ids_ok": ids_ok, "quote_ok": quote_ok,
        "calls": n_calls, "tokens": sum(u["prompt_tokens"] + u["completion_tokens"] for u in rec["usage"]),
        "answer_prompt_tokens": rec["usage"][-1]["prompt_tokens"],
    }


def aggregate(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in ("em", "f1", "abstain", "parse_err", "all_gold_docs", "gold_doc_recall", "cited2",
              "cited_gold_recall", "calls", "tokens", "answer_prompt_tokens"):
        out[k] = round(float(np.mean([r[k] for r in rows])), 4)
    claims = sum(r["claims"] for r in rows)  # citation checks are per claim, micro-averaged
    out["claims"] = claims
    out["cite_ids_ok"] = round(sum(r["ids_ok"] for r in rows) / claims, 4) if claims else None
    out["cite_quote_ok"] = round(sum(r["quote_ok"] for r in rows) / claims, 4) if claims else None
    return out


def bootstrap(a: dict[str, dict], b: dict[str, dict], key: str) -> dict:
    """Paired bootstrap of mean(a[key] - b[key]) over questions."""
    ids = sorted(a)
    d = np.array([a[i][key] - b[i][key] for i in ids])
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(d), (BOOT_N, len(d)))
    m = d[idx].mean(axis=1)
    return {"delta": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(m, 2.5)), 4),
                                                         round(float(np.percentile(m, 97.5)), 4)], "n": len(d)}


def run() -> None:
    kb = KB()
    qs = {q["id"]: q for q in subset_questions()}
    per_q: dict[str, dict[str, dict]] = {}
    for s in SYSTEMS:
        path = RESULTS / "answers" / f"{s}.jsonl"
        if path.exists():
            per_q[s] = {r["id"]: {**question_metrics(r, qs[r["id"]], kb), "hop": r["hop"]}
                        for r in read_jsonl(path)}
    summary: dict = {"systems": {}, "bootstrap": {}}
    for s, rows in per_q.items():
        summary["systems"][s] = {h: aggregate([r for r in rows.values() if h == "all" or str(r["hop"]) == h])
                                 for h in HOPS}
    if "kb" in per_q and "rrf" in per_q:
        for h in HOPS:
            sub = lambda s: {i: r for i, r in per_q[s].items() if h == "all" or str(r["hop"]) == h}
            summary["bootstrap"][h] = {k: bootstrap(sub("kb"), sub("rrf"), k)
                                       for k in ("em", "f1", "all_gold_docs", "cited2")}
    (RESULTS / "answers" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for title, keys in (("answers", ("em", "f1", "abstain", "parse_err")),
                        ("coverage and multi-doc citation",
                         ("all_gold_docs", "gold_doc_recall", "cited2", "cited_gold_recall")),
                        ("citation checks (per claim) and cost per question",
                         ("cite_ids_ok", "cite_quote_ok", "calls", "tokens"))):
        print(f"\n{title}\n{'system':7} {'metric':17} " + " ".join(f"hop={h:>3}" for h in HOPS))
        for s, byhop in summary["systems"].items():
            for k in keys:
                vals = " ".join(f"{byhop[h][k]:7.3f}" if byhop[h][k] is not None else "     --" for h in HOPS)
                print(f"{s:7} {k:17} {vals}")
    for h, d in summary["bootstrap"].items():
        print(f"\nkb - rrf, hop={h}: " + "; ".join(f"{k} {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                                                  for k, v in d.items()) + f" (n={d['em']['n']})")
