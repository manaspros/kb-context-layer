"""`kb ask "question"` (full kb pipeline on a free-form question) and `kb show <question_id>` (stored eval result).
Both print the same record: answer, claims with resolved citations, hop trace, distinct source documents."""
import argparse
import json
import sys

from kb.answer import KB, answer_one
from kb.common import INDEX, RESULTS, read_jsonl


def cite(kb: KB, span_id: str) -> dict:
    """span id -> where it lives: document title, paragraph index in the document, sentence index, text."""
    s = kb.span[span_id]
    para = kb.paras[s["para_id"]]
    return {"span_id": span_id, "doc": para["title"], "paragraph": int(s["para_id"].split(":")[1]),
            "sentence": int(span_id.split(":")[2]), "text": s["text"]}


def view(kb: KB, rec: dict, p: dict | None = None) -> dict:
    """Display record from an answer record {context_para_ids, parsed, trace?, usage}; `p` = cached NLI scores."""
    ctx_spans = {s["span_id"] for pid in rec["context_para_ids"] for s in kb.spans_of[pid]}
    found = next((t["final"] for t in rec.get("trace") or [] if "final" in t), {})  # paragraph id -> hop that found it
    claims = []
    for i, c in enumerate((rec["parsed"] or {}).get("claims", [])):
        ids = list(dict.fromkeys(x for x in (c.get("span_ids") or []) if isinstance(x, str) and x in ctx_spans))
        quote = c.get("quote")
        claims.append({"text": c.get("text"), "citations": [{**cite(kb, x), "hop": found.get(kb.span[x]["para_id"])} for x in ids],
                       "quote_verbatim": bool(isinstance(quote, str) and quote and any(quote in kb.span[x]["text"] for x in ids)),
                       "nli_p_supported": (p or {}).get((i, tuple(ids)))})
    title = lambda pid: kb.paras[pid]["title"]
    hops = []
    for t in rec.get("trace") or []:
        if "final" in t:
            continue
        if t["hop"] == 0:
            h = {"hop": 0, "queries": t["queries"],
                 "added": [{"item": "question", "path": "search", "docs": sorted({title(x) for x in t["added"]})}]}
        else:
            h = {"hop": t["hop"], "queries": [f"{i['entity']}: {i['query']}" for i in t["items"]],
                 "added": [{"item": i["entity"], "path": i["path"], "docs": sorted({title(x) for x in i["added"]})}
                           for i in t["items"]]}
        # the sufficiency check that ran on the context after this hop (it decides whether another hop follows)
        h.update(sufficient=t.get("sufficient"), known=[f["fact"] for f in t.get("known", [])])
        hops.append(h)
    cited_docs = sorted({x["doc"] for c in claims for x in c["citations"]})
    return {"answer": (rec["parsed"] or {}).get("answer"), "claims": claims, "hops": hops,
            "docs_cited": cited_docs, "docs_in_context": sorted({title(x) for x in rec["context_para_ids"]}),
            "calls": len(rec["usage"]), "tokens": sum(u["prompt_tokens"] + u["completion_tokens"] for u in rec["usage"])}


def pretty(v: dict) -> str:
    out = [f"Q: {v['question']}"]
    if "gold_answer" in v:
        out.append(f"Gold answer: {v['gold_answer']}   ({v['hop_count']}-hop; EM {v['em']:.0f}, F1 {v['f1']:.2f}; "
                   f"failure stage: {v['stage']})")
    out.append(f"\nAnswer: {v['answer']}")
    for n, c in enumerate(v["claims"], 1):
        nli = f"  [NLI supported p={c['nli_p_supported']:.2f}]" if c["nli_p_supported"] is not None else ""
        out.append(f"\nClaim {n}: {c['text']}{nli}")
        out += [f"  - {x['doc']} — paragraph {x['paragraph']}, sentence {x['sentence']}: \"{x['text']}\"  ({x['span_id']}{', found at hop ' + str(x['hop']) if x['hop'] is not None else ''})"
                for x in c["citations"]] or ["  - (no valid citation)"]
    out.append("\nHop trace:" if v["hops"] else "\nHop trace: none (single-shot retrieval, no extra hops)")
    for h in v["hops"]:
        out.append(f"  hop {h['hop']}: " + " | ".join(h["queries"]))
        if h.get("sufficient") is not None:
            out.append(f"    sufficiency check after this hop: {'sufficient' if h['sufficient'] else 'not sufficient'}"
                       + (f"; facts so far: {h['known']}" if h.get("known") else ""))
        out += [f"    {a['item']} [{a.get('path', 'search')}] -> {', '.join(a['docs']) or '(nothing new)'}" for a in h["added"]]
    out.append(f"\nDistinct source documents cited ({len(v['docs_cited'])}): " + "; ".join(v["docs_cited"]))
    out.append(f"Documents in the final context ({len(v['docs_in_context'])}): " + "; ".join(v["docs_in_context"]))
    if "gold_docs" in v:
        out.append("Gold documents: " + "; ".join(f"{d}{' [in context]' if d in v['docs_in_context'] else ''}"
                                                    f"{' [cited]' if d in v['docs_cited'] else ''}" for d in v["gold_docs"]))
    out.append(f"LLM calls: {v['calls']}, tokens: {v['tokens']}")
    return "\n".join(out)


def ask() -> None:
    """python -m kb ask "question" [--json]: needs LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (see README)."""
    ap = argparse.ArgumentParser(prog="kb ask")
    ap.add_argument("question")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(sys.argv[2:])
    from kb.assemble import Assembler, Config
    from kb.llm import model_tag
    kb = KB()
    asm = Assembler(kb)
    q = {"id": "adhoc", "question": a.question}
    cfg = Config()
    ctx = asm.run(q, asm.first([q], cfg)["adhoc"], cfg)
    rec = answer_one(kb, a.question, ctx["para_ids"])
    rec = {**rec, "trace": ctx["trace"], "usage": ctx["usage"] + rec["usage"]}
    v = {"question": a.question, "model": model_tag(), **view(kb, rec)}
    print(json.dumps(v, indent=2, ensure_ascii=False) if a.json else pretty(v))


def show() -> None:
    """python -m kb show <question_id> [--system kb] [--json]: a stored eval result; no LLM call."""
    ap = argparse.ArgumentParser(prog="kb show")
    ap.add_argument("question_id")
    ap.add_argument("--system", default="kb")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(sys.argv[2:])
    from kb.score import question_metrics
    kb = KB()
    recs = {r["id"]: r for r in read_jsonl(RESULTS / "answers" / f"{a.system}.jsonl")}
    if a.question_id not in recs:
        sys.exit(f"{a.question_id} is not in results/answers/{a.system}.jsonl")
    rec = recs[a.question_id]
    q = next(x for x in read_jsonl(INDEX / "questions.jsonl") if x["id"] == a.question_id)
    nli_path = RESULTS / "verify" / f"{a.system}.jsonl"
    nli = {(r["claim_idx"], tuple(r["spans"])): r["p"] for r in read_jsonl(nli_path) if r["id"] == a.question_id} if nli_path.exists() else {}
    m = question_metrics(rec, q, kb)
    v = {"question_id": a.question_id, "system": a.system, "question": q["question"], "gold_answer": q["answer"],
         "hop_count": q["hop"], "em": m["em"], "f1": m["f1"], "stage": m["stage"],
         "gold_docs": [kb.paras[f"{d}:0"]["title"] for d in q["gold_doc_ids"]],
         **view(kb, rec, {k: p for k, p in nli.items()})}
    print(json.dumps(v, indent=2, ensure_ascii=False) if a.json else pretty(v))
