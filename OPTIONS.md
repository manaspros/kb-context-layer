# OPTIONS: approaches considered and why

Each decision lists the options, what evidence we had, the pick, and what would change our mind.
Numbers marked *(published)* come from papers and are not reproduced here; numbers from our own
runs are added as steps land.

## 1. Benchmark

| Option | For | Against |
|---|---|---|
| **MuSiQue-Ans** | 2-4 hop, built to defeat single-hop shortcuts; gold supporting paragraphs and sub-question decomposition; CC BY 4.0 | Wikipedia-derived, so some contamination |
| HotpotQA | Sentence-level supporting facts | Many questions solvable from one paragraph; heavy contamination |
| 2WikiMultiHopQA | Evidence triples | Templated questions, shortcut-prone |
| MultiHop-RAG (news) | Real 609-doc KB, comparison/temporal types | Snippet-level evidence only; no sentence locations |
| FRAMES | Recent, 2-5+ articles | Doc-level labels only; needs full Wikipedia fetches |
| BrowseComp-Plus | Low contamination | Very hard; baselines near floor, so no signal at our budget |
| ALCE | Defines citation metrics | Not multi-hop |

**Pick: MuSiQue-Ans dev**, with ALCE-style citation metrics layered on. No public text benchmark
we found has both multi-document gold evidence *and* claim-level citation labels, so we compose
them. **Check before trusting:** a no-retrieval baseline measures contamination directly.

Verified against data (HF `dgslibisey/MuSiQue` @ `c8f4f8c`): hop count is the id prefix
(2hop 1252, 3hop 760, 4hop 405); 54 questions have all gold paragraphs under one title and are
excluded as not multi-document.

## 2. Corpus construction

| Option | For | Against |
|---|---|---|
| Per-question 20 candidates | Matches original setup | Toy retrieval; 18 distractors only |
| Pool of sampled questions only | Small | Fewer distractors than reality |
| **Pool of all 21,100 dev paragraphs** | Real KB, cross-question distractors, still laptop-sized | Not comparable to full-Wikipedia numbers |

**Pick: full dev pool.** A *document* is a Wikipedia title; paragraphs are its sections. This is
what makes "distinct source document" well-defined for the hard requirement.

## 3. Ingestion unit and provenance

| Option | For | Against |
|---|---|---|
| Fixed-size token chunks | Standard | Cuts sentences; citations point at arbitrary windows |
| **Paragraph for retrieval, sentence for citation** | MuSiQue paragraphs are already coherent units; sentences give precise locations | Sentence splitting errors on abbreviations |
| Propositions (Dense X) | +recall at small budgets *(published)* | One LLM pass over the corpus; modest gain at our budget |

**Pick:** retrieve paragraphs, cite sentences. Every span id is `doc_id:para_idx:sent_idx` and
maps back to the exact raw text, so provenance is a lookup, not a model claim.

## 4. Retrieval

| Option | Evidence | Cost |
|---|---|---|
| BM25 | MuSiQue F1 28.8, R@5 43.5 *(published, HippoRAG 2 tables)* | trivial |
| Dense (small, bge) | Dense 7B: F1 45.7, R@5 69.7 *(published)*; small models lower | laptop |
| **Hybrid RRF** | Beats either alone on 2Wiki R@10 *(published, secondary)* | trivial |
| Cross-encoder rerank | Mixed for multi-hop; ~5x latency | ablation only |
| Contextual chunk prefix | -49% retrieval failures, non-multi-hop data *(published)* | 1 LLM call / chunk |

**Pick:** hybrid RRF as the base retriever, everything else ablated. Lexical-first retrieval is often hard
to beat in practice, so BM25 stays a serious baseline, not a straw man.

## 5. Multi-hop: how later hops get found

| Option | Evidence | Cost |
|---|---|---|
| Single-shot top-k | Misses bridge docs by construction | 1 call |
| Query decomposition, always on | Can hurt: MuSiQue MRR 0.469 -> 0.102 in one study *(published)* | 2-4 calls |
| IRCoT (retrieve per reasoning step) | +11-21 recall points *(published)* | 3-8 calls |
| GraphRAG / LightRAG | GraphRAG F1 38.5, below dense *(published)*; 115M indexing tokens | very high |
| HippoRAG 2 (entity graph + PPR) | F1 48.6, R@5 74.7 *(published)* | 9.2M indexing tokens |
| **Entity graph + gated bridge hop** | Combines the two mechanisms with evidence | 0-2 extra calls |

**Pick:** (a) a cheap entity graph: link a paragraph to every document whose title (or alias) it
mentions, which is a free, deterministic stand-in for hyperlinks; (b) a sufficiency check, and
only when evidence is missing, a follow-up query that carries the bridge entity found so far,
capped at 2 hops. Comparison questions get one retrieval per entity. Ablate graph on/off and hop
cap 0/1/2. **Would change our mind:** if the graph adds recall but no F1, drop it.

## 6. Context assembly

Fixed token budget for every system (fairness). Per-document cap so one article cannot fill the
window. Evidence grouped by document, ordered by hop. Conflicting spans are shown side by side,
not silently resolved.

## 7. Synthesis and citation

| Option | For | Against |
|---|---|---|
| Post-hoc citation | Simple | ALCE: closed-book + post-hoc cite has much lower citation recall *(published)* |
| **Cite during generation, with verbatim quotes** | Quotes are checkable by string match; span ids are checkable by lookup | Longer output |
| + NLI verifier (MiniCheck) | GPT-4-level fact-check accuracy at small size *(published)* | Extra local model |

**Pick:** structured output `{claim, span_ids, quote}`, then mechanical checks (span exists, quote
is a substring), then NLI support check. An explicit "not answerable from the corpus" path.

## 8. Metrics

- Retrieval: gold-doc recall@k; all-gold-docs-in-context rate (the real coverage metric).
- Answer: EM / F1 with the official MuSiQue scorer; aliases counted.
- Citation: claim-level precision and recall via NLI (ALCE definitions).
- Multi-doc: share of correct answers citing >= 2 distinct gold documents.
- Per-stage failure attribution: each wrong answer labelled retrieval-miss, hop-miss,
  assembly-drop, generation-unfaithful, or citation-wrong, computable from gold paragraph labels.
  (Layered evaluation: score discovery, evidence use and task completion separately, so a gain
  or loss is attributed to the stage that caused it.)
- Cost: tokens and USD per question.
- Statistics: paired bootstrap, 10k resamples, 95% CI on deltas; n always reported.

LLM-as-judge is avoided for headline numbers (EM/F1 and NLI are deterministic); if used, judge
family differs from the generator and 50 judgments are hand-audited.

## 9. Baselines (same generator, prompt, corpus, context budget)

Closed-book; BM25 top-k; dense top-k; hybrid RRF top-k; IRCoT-lite on the same retriever.
Published HippoRAG 2 / PropRAG numbers cited for context only, not as head-to-head rows
(different embedder and generator).

## 10. Generator LLM (open)

One model for every system. Options: a cheap hosted model via an OpenAI-compatible client;
Sarvam-M (OpenAI-compatible API); a local model on Apple Silicon (free, slow). Decision pending
on which API key is available. Cost target: under $50 for all runs.

## Log of experiments

(Filled in as steps land: what we tried, the number, keep or drop.)

Retrieval, 400-question subset, 21,100-paragraph pool, gold-doc recall@10 / all-gold-docs@10:
- BM25 (bm25s, title-prefixed, stopwords, no stemmer): 0.518 / 0.188. Keep as baseline.
- Dense (bge-small-en-v1.5, query prefix): 0.574 / 0.220. Keep as baseline.
- Hybrid RRF (k=60, depth 100): 0.596 / 0.250. Best single-shot; base retriever. 4-hop all-gold@10 is 0.020, so single-shot cannot cover later hops.
