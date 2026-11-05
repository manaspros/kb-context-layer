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
Sarvam-M (OpenAI-compatible API); a local model on Apple Silicon (free, slow). **Decision: `deepseek/deepseek-v4.1-flash-1017`** through an OpenAI-compatible gateway (stdlib client,
disk-cached by request hash, temperature 0, max_tokens 4000). Cost target: under $50 for all runs.

## Log of experiments

(Filled in as steps land: what we tried, the number, keep or drop.)

Retrieval, 400-question subset, 21,100-paragraph pool, gold-doc recall@10 / all-gold-docs@10:
- BM25 (bm25s, title-prefixed, stopwords, no stemmer): 0.518 / 0.188. Keep as baseline.
- Dense (bge-small-en-v1.5, query prefix): 0.574 / 0.220. Keep as baseline.
- Hybrid RRF (k=60, depth 100): 0.596 / 0.250. Best single-shot; base retriever. 4-hop all-gold@10 is 0.020, so single-shot cannot cover later hops.

Answers, 400 questions, same prompt and 10-paragraph budget, max_tokens 4000 (1200 gave 19-28% empty outputs from the reasoning model):
- Closed-book: EM 0.247 / F1 0.345, 8.7% empty outputs. Contamination is real: a quarter of questions are answered with no retrieval. Keep as baseline.
- BM25 0.125 / 0.182, dense 0.185 / 0.256, RRF 0.188 / 0.254; cited >= 2 gold docs 0.255 / 0.318 / 0.347. Keep.
- All three RAG baselines score below closed-book: the model answers NOT ANSWERABLE on 52-65% of questions because a hop is missing from the context, where closed-book guesses from memory. Coverage, not generation, is the bottleneck.

Graph (title/alias mentions: 24,647 edges over 21,100 paragraphs, 63.7% of paragraphs have >= 1 link; expansion = 7 RRF seeds + paragraphs of linked documents ranked by dense sim to the query, same 10-paragraph budget):
- Seed count chosen on the 200-question tune split only: all-gold@10 for 3/5/7/9 seeds = 0.200/0.215/0.220/0.200 vs RRF 0.145, so 7. (The earlier pick of 7 was made on the eval set; the tune split confirms it.)
- Eval at 7 seeds: all-gold@10 0.250 -> 0.302 (2-hop 0.440 -> 0.547, 3-hop 0.213 -> 0.240, 4-hop 0.020 -> 0.030); gold-doc recall 0.596 -> 0.625, but 3-hop recall is flat (0.618 -> 0.616). Ceiling if every doc one link from the top-10 were kept: all-gold 0.388. Modest: helps 2-hop, barely 3/4-hop. Keep as the seed stage; later hops need the LLM bridge step.

Assembly (tune split, n=200; every value below was chosen without touching the eval subset):
- Lookup threshold (min dense sim of the matched document's best paragraph to accept a title lookup) 0 / 0.5 / 0.6 / 0.7 vs search only: all-gold in the 10-paragraph context after hop 1 = 0.390 / 0.390 / 0.390 / 0.395 vs 0.395. Pick 0.7, a tie with search-only: lookup does not beat search on this split.
- Hop cap 0 -> 1: all-gold 0.220 -> 0.395. Per-document cap 2 vs 3: 0.395 vs 0.390, pick 2. Hop 2 is implemented but not run: each extra hop is another ~3.9k-token check per question and the 12M-token cap left no room, so the cap is tuned over {0, 1} only.
- 17.5% of hop-0 sufficiency checks on eval come back empty (the reasoning model spends all 4000 tokens), so those questions get no extra hop; 26% say "sufficient" and stop; 56% (225/400) take the extra hop.

kb = graph-expanded RRF + sufficiency check + one entity-first hop + per-document cap 2, 400 eval questions, same prompt and 10-paragraph budget:
- EM 0.188 -> 0.338, F1 0.254 -> 0.428, all-gold-in-context 0.250 -> 0.440, cited >= 2 gold docs 0.347 -> 0.517 (kb minus rrf, paired bootstrap: EM +0.150 [+0.110, +0.190], all-gold +0.190 [+0.150, +0.233]). 4-hop all-gold only 0.020 -> 0.070. Cost 2 calls (1 check + 1 answer) and 7.1k tokens per question vs 1 and 3.1k. Keep.
- Ablations on a 134-question stratified slice (every 3rd question per hop; the full 400 did not fit the token cap): rrf EM 0.172, kb hop cap 0 (graph-expanded RRF only) 0.246, kb search-only (no entity lookup) 0.366, kb 0.366. The hop step is worth +0.119 EM [+0.052, +0.187] over hop cap 0; entity-first lookup adds nothing measurable over search-only (EM +0.000 [-0.022, +0.022]) although lookup adds purer paragraphs (81% in gold documents vs 21% for search). Kept as specified; search-only is the simpler equivalent.

Answer-quality diagnostics (scorer additions): every one of the 7.5-8.7% unparseable outputs (all systems) is a reasoning-model truncation at max_tokens 4000 (empty content, completion_tokens = 4000; 3 of 168 are cut-off JSON), so no parser fix exists and the shared prompt was left unchanged. Answered-only precision: closed-book answers 81% of questions (EM 0.306 when it answers), rrf 37.5% (EM 0.500), kb 59.3% (EM 0.570). Closed-book F1 beats the RAG baselines only because they abstain; kb answers more and is more precise when it does. Failure stages for kb (n=400): 14 retrieval_miss, 138 hop_miss, 12 assembly_drop, 53 generation_wrong; 84 of 183 correct answers fail to cite a gold document (citation_gap). Context utilization (cited spans / spans in context) is 3-5% for every system.

Round 2, empty outputs (tune split, 68 questions = every 3rd per hop; each config answers RRF top-10 and runs the first sufficiency check on graph-expanded top-10; picked by empty rate, then EM):
- `reasoning: {effort: low}` at max_tokens 4000: answer empty 5.9%, check unparsed 14.7%, EM 0.176. The endpoint accepts it but barely changes reasoning (answer completion 999 vs 1081 tokens), so it is dropped. max_tokens 8000: 0.0% / 2.9%, EM 0.191. Both: 1.5% / 2.9%, EM 0.206. Pick max_tokens 8000 alone (lowest pooled empty rate, 2.9% vs 4.4%), applied to every system and to the sufficiency check; the 0.015 EM gap to "both" is 1 question.
- Re-run on the 400 eval (calls that stopped by themselves under the old 4000 cap were carried over, only truncated or empty ones re-issued; 3.0M new tokens): answer empty closed 8.7% -> 3.3%, bm25 5.2% -> 0.0%, dense 9.2% -> 1.8%, rrf 7.8% -> 0.5%, kb 7.5% -> 1.8%; kb hop-0 sufficiency check empty 17.5% -> 6.0%. EM / F1: closed 0.247/0.345 -> 0.250/0.350, bm25 0.125/0.182 -> 0.130/0.191, dense 0.185/0.256 -> 0.188/0.261, rrf 0.188/0.254 -> 0.195/0.263, kb 0.338/0.428 -> 0.352/0.448. Keep; the remaining empties still hit the 8000 cap.

Round 2, assembly caps (tune split n=200, all-gold in the final 10-paragraph context, max_tokens 8000, tau 0.7 kept): hop cap 0 / 1 / 2 at per-doc cap 2 = 0.220 / 0.415 / 0.425; at cap 3 = 0.220 / 0.410 / 0.420. Pick hop cap 2, per-doc cap 2 (the +0.010 over one hop is 2 questions, but it is the pre-set rule: highest tune coverage, ties to fewer hops). Eval, final kb: EM 0.352 -> 0.383, F1 0.448 -> 0.475, all-gold 0.445 -> 0.475, cited >= 2 gold 0.540 -> 0.560; kb minus rrf EM +0.188 [+0.145, +0.230]. Cost 2.66 calls and 9.9k tokens per question (was 2 and 7.4k). Keep.

Round 2, ablations with the final config (hop cap 2, per-doc cap 2, max_tokens 8000; the old 134-question slice files are replaced, nothing stale is scored). kb_hop0 and kb_no_lookup on the full 400, kb_no_graph on 267 of 400 (two of every three questions per hop; the budget did not allow the rest). EM kb 0.383, kb_hop0 0.228, kb_no_lookup 0.380, rrf 0.195; kb_no_graph 0.382 vs kb 0.363 on the same 267. Paired bootstrap, kb minus: hop0 EM +0.155 [+0.115, +0.198] (the hop step is the gain); no_lookup +0.003 [-0.015, +0.020] (entity lookup adds nothing over search-only); no_graph -0.019 [-0.064, +0.026] (graph expansion adds nothing once hops are on, though lookup paragraphs are purer: 75% vs 16% in gold docs). Failure stage hop_miss: rrf 236, kb_hop0 216, kb 114. Keep graph and lookup as shipped; search-only without graph is the simplest equivalent.
