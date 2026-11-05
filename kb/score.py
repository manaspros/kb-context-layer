"""Recompute every answer metric from results/answers/*.jsonl alone (no LLM calls)."""
import json
import re
import string
from collections import Counter

import numpy as np

from kb.answer import KB, subset_questions
from kb.common import RESULTS, read_jsonl

SYSTEMS = ("closed", "bm25", "dense", "rrf", "kb_hop0", "kb_no_lookup", "kb")
STAGES = ("retrieval_miss", "hop_miss", "assembly_drop", "generation_wrong", "correct_ok", "correct_citation_gap")
BOOT_KEYS = ("em", "f1", "all_gold_docs", "cited2")
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
    em = float(any(norm(ans) == norm(g) for g in golds))
    best_f1 = max(f1(ans, g) for g in golds)
    docs = lambda ids: {p.split(":")[0] for p in ids}
    tr = rec.get("trace")  # baselines: initial retrieval == pool == final context
    initial = docs(tr[0]["added"]) if tr else ctx_docs
    pool = docs(p for p, *_ in tr[-1]["pool"]) if tr else ctx_docs
    valid_cited = {i for c in claims for i in (c.get("span_ids") or []) if isinstance(i, str) and i in ctx_spans}
    if em == 0 and best_f1 < 0.5:
        stage = ("retrieval_miss" if not gold_d & initial else "hop_miss" if not gold_d <= pool
                 else "assembly_drop" if not gold_d <= ctx_docs else "generation_wrong")
    else:
        stage = "correct_citation_gap" if not gold_d <= {i.split(":")[0] for i in valid_cited} else "correct_ok"
    return {"stage": stage, "util": len(valid_cited) / len(ctx_spans) if ctx_spans else None,
        "em": em, "f1": best_f1,
        "answered": float(rec["parsed"] is not None and bool(ans.strip()) and ans.strip().upper() != "NOT ANSWERABLE"),
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
    ans = [r for r in rows if r["answered"]]  # precision when the system commits to an answer
    out["answered"] = round(len(ans) / len(rows), 4)
    for k in ("em", "f1"):
        out[f"{k}_answered"] = round(float(np.mean([r[k] for r in ans])), 4) if ans else None
    out["tokens_total"] = sum(r["tokens"] for r in rows)
    util = [r["util"] for r in rows if r["util"] is not None]  # cited valid spans / spans in context
    out["utilization"] = round(float(np.mean(util)), 4) if util else None
    out["stages"] = {k: sum(r["stage"] == k for r in rows) for k in STAGES}
    claims = sum(r["claims"] for r in rows)  # citation checks are per claim, micro-averaged
    out["claims"] = claims
    out["cite_ids_ok"] = round(sum(r["ids_ok"] for r in rows) / claims, 4) if claims else None
    out["cite_quote_ok"] = round(sum(r["quote_ok"] for r in rows) / claims, 4) if claims else None
    return out


def bootstrap(a: dict[str, dict], b: dict[str, dict], key: str) -> dict:
    """Paired bootstrap of mean(a[key] - b[key]) over questions."""
    ids = sorted(set(a) & set(b))  # ablations cover a stratified slice of the subset
    d = np.array([a[i][key] - b[i][key] for i in ids])
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(d), (BOOT_N, len(d)))
    m = d[idx].mean(axis=1)
    return {"delta": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(m, 2.5)), 4),
                                                         round(float(np.percentile(m, 97.5)), 4)], "n": len(d)}


def lookup_stats(rows: list[dict], qs: dict[str, dict]) -> dict:
    """Entity-lookup vs search path over every next-item in the hop traces."""
    st = {p: {"items": 0, "doc_hit": 0, "paras": 0, "gold_paras": 0} for p in ("lookup", "search")}
    hops = mixed = 0
    for r in rows:
        gold = set(qs[r["id"]]["gold_doc_ids"])
        for t in r.get("trace", []):
            if "items" not in t:
                continue
            hops += 1
            mixed += len({i["path"] for i in t["items"]}) > 1
            for i in t["items"]:
                a = st[i["path"]]
                a["items"] += 1
                a["doc_hit"] += i["doc_id"] in gold if i["path"] == "lookup" else any(p.split(":")[0] in gold for p in i["added"])
                a["paras"] += len(i["added"])
                a["gold_paras"] += sum(p.split(":")[0] in gold for p in i["added"])
    n = sum(a["items"] for a in st.values())
    return {"hops": hops, "hops_mixed": mixed, "items": n,
            "lookup_item_share": round(st["lookup"]["items"] / n, 4) if n else None,
            **{p: {**a, "item_precision": round(a["doc_hit"] / a["items"], 4) if a["items"] else None,
                   "para_gold_precision": round(a["gold_paras"] / a["paras"], 4) if a["paras"] else None}
               for p, a in st.items()}}


def run() -> None:
    kb = KB()
    qs = {q["id"]: q for q in subset_questions()}
    per_q: dict[str, dict[str, dict]] = {}
    raw: dict[str, list[dict]] = {}
    for s in SYSTEMS:
        path = RESULTS / "answers" / f"{s}.jsonl"
        if path.exists():
            raw[s] = read_jsonl(path)
            per_q[s] = {r["id"]: {**question_metrics(r, qs[r["id"]], kb), "hop": r["hop"]} for r in raw[s]}
    summary: dict = {"systems": {}, "bootstrap": {}, "lookup": {}}
    for s, rows in per_q.items():
        summary["systems"][s] = {h: aggregate([r for r in rows.values() if h == "all" or str(r["hop"]) == h])
                                 for h in HOPS}
    for other in ("rrf", "kb_hop0", "kb_no_lookup"):
        if "kb" in per_q and other in per_q:
            for h in HOPS:
                sub = lambda s: {i: r for i, r in per_q[s].items() if h == "all" or str(r["hop"]) == h}
                summary["bootstrap"].setdefault(f"kb-{other}", {})[h] = {
                    k: bootstrap(sub("kb"), sub(other), k) for k in BOOT_KEYS}
    abl = set(per_q.get("kb_hop0", {})) | set(per_q.get("kb_no_lookup", {}))  # same ids for every row
    if abl:
        summary["ablation_subset"] = {s: aggregate([r for i, r in per_q[s].items() if i in abl])
                                      for s in ("rrf", "kb_hop0", "kb_no_lookup", "kb") if s in per_q}
    for s in ("kb", "kb_no_lookup"):
        if s in raw:
            summary["lookup"][s] = lookup_stats(raw[s], qs)
    (RESULTS / "answers" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for title, keys in (("answers", ("n", "em", "f1", "answered", "em_answered", "f1_answered", "abstain", "parse_err")),
                        ("coverage and multi-doc citation",
                         ("all_gold_docs", "gold_doc_recall", "cited2", "cited_gold_recall")),
                        ("citation checks (per claim), context utilization, cost per question",
                         ("cite_ids_ok", "cite_quote_ok", "utilization", "calls", "tokens"))):
        print(f"\n{title}\n{'system':13} {'metric':17} " + " ".join(f"hop={h:>3}" for h in HOPS))
        for s, byhop in summary["systems"].items():
            for k in keys:
                vals = " ".join((f"{byhop[h][k]:7.0f}" if k == "n" else f"{byhop[h][k]:7.3f}") if byhop[h][k] is not None else "     --" for h in HOPS)
                print(f"{s:13} {k:17} {vals}")
    for s, a in summary.get("ablation_subset", {}).items():
        print(f"ablation slice n={a['n']}: {s:13} em {a['em']:.3f} f1 {a['f1']:.3f} all_gold {a['all_gold_docs']:.3f} "
              f"cited2 {a['cited2']:.3f} tokens/q {a['tokens']:.0f}")
    print("\nfailure stage counts (EM=0 and F1<0.5 is wrong; closed-book has no retrieval so it is n/a)")
    print(f"{'system':13} {'n':>4} " + " ".join(f"{k:>20}" for k in STAGES))
    for s, byhop in summary["systems"].items():
        if s != "closed":
            print(f"{s:13} {byhop['all']['n']:4d} " + " ".join(f"{byhop['all']['stages'][k]:20d}" for k in STAGES))
    for pair, byh in summary["bootstrap"].items():
        for h, d in byh.items():
            print(f"{pair}, hop={h}: " + "; ".join(f"{k} {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                                                    for k, v in d.items()) + f" (n={d['em']['n']})")
    for s, d in summary["lookup"].items():
        print(f"\nlookup stats {s}: " + json.dumps(d))
    print(f"\ntotal tokens, all systems in results/answers: {sum(v['all']['tokens_total'] for v in summary['systems'].values())}")
