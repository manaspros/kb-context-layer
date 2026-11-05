# DESIGN

A context layer over a multi-document corpus: it decides which evidence an answer model reads, and records
where every answered claim came from. Built and measured on MuSiQue-Ans (see [EVAL.md](EVAL.md) for every number;
all figures below are read from `results/summary.md`). A diagram of the same flow is in
[docs/architecture.html](docs/architecture.html).

## 1. The problem: covering the evidence set

A multi-hop question needs an *evidence set*: one paragraph per step of the reasoning chain, from at least two
distinct documents (in our 400-question subset: 2.84 gold documents per question on average). Answering is easy
once the set is in front of the model; the hard part is getting it there inside a fixed context budget.

Top-k retrieval ranks by similarity to the **question**. Hop 1 is similar to the question. Hop 2 usually is not:
it is about a *bridge entity* that only appears once hop 1 has been read ("the screenwriter of WarGames" is in the
question; "Lawrence Lasker's mother" is not). So single-shot retrieval misses later hops by construction. Measured:
hybrid RRF puts all gold documents in a 10-paragraph context for 25.0% of questions (2-hop 44.0%, 3-hop 21.3%,
4-hop 2.0%), and the answer model, told to answer only from evidence, abstains on 58% of questions.

Three properties must hold, and each has its own metric: **coverage** (all gold documents in the context),
**grounding** (each claim is entailed by the spans it cites), **attributability** (each failure belongs to one
stage, so a gain is earned, not guessed).

## 2. Architecture and data flow

```
 raw dev set ─► corpus: 21,100 paragraphs, 17,629 documents (a document = a Wikipedia title)
                   │
   L1 INDEX        ├─ sentence spans, id = doc:para:sent, exact char offsets   (72,934 spans)
   (structure)     ├─ BM25 + bge-small dense + RRF over paragraphs
                   └─ title/alias entity graph: paragraph ─► documents whose title it mentions (24,647 edges)
                   │
   L2 ASSEMBLY     question ─► RRF top-10 + graph expansion (7 seeds)             hop 0
   (gated hops,         │
   facts ledger)        ├─ sufficiency check (LLM): {sufficient, known facts+span ids, next entity+query}
                        │     sufficient? ─► stop
                        └─ else: resolve entity title ─► lookup that document's best paragraphs   (hop 1..2)
                                 no title match ─► RRF + graph search with "entity + query"
                        ─► select: per-document cap 2, reserved slots per next-item, 10 paragraphs total
                   │
   L3 SYNTHESIS    answer JSON {answer, claims:[{text, span_ids, quote}]}; "NOT ANSWERABLE" allowed
                   │
   L4 VERIFY       span exists in context · quote is a substring · local NLI (MiniCheck) support check per claim
                   │
   PROVENANCE      answer ─► claim ─► span (doc:para:sent) ─► paragraph ─► document ─► hop that found it
```

**Indexing with structure.** Retrieval works on paragraphs; citations point at sentences. Every span id maps to
exact character offsets in the raw paragraph (ingestion asserts the round trip for all 72,934 spans), so
"where did this come from" is a lookup, not a model output. The entity graph is deliberately cheap: a paragraph
links to every document whose title (or its parenthetical-stripped alias) it mentions. It is a deterministic stand-in
for hyperlinks and costs no LLM tokens. Single-word titles that are common words are dropped.

**Gated multi-hop assembly.** After the first retrieval, one LLM call reads the context and returns JSON: whether the
evidence is sufficient, the facts found so far (each with the span ids that state it, kept only if the span id exists),
and, if not, which entity to read next (as a likely Wikipedia title) and what to find about it. The facts ledger is
passed to the next check so a bridge learned at hop 1 is not forgotten at hop 2. The next hop is *entity-first*: if the
entity's title resolves and its best paragraph clears a similarity floor (0.7, chosen on the tune split), the document
is pulled directly (2 paragraphs); otherwise the system searches with the entity and query. At most 2 extra hops; comparison
questions may list two entities. Always-on decomposition was rejected because it can hurt (OPTIONS.md §5); gating means a
question the first retrieval already covers pays nothing: 28% of hop-0 checks said "sufficient" and 66% of questions took at least one extra hop.

**Cited synthesis.** The answerer sees evidence as `[span_id] sentence`, grouped by document title, and must return the
shortest possible answer plus claims, each with span ids and a verbatim quote. The evidence is declared untrusted data in
the prompt. Closed-book is allowed only when no evidence is given (that is the baseline).

**Verification.** Mechanical: the span id must be in the context, the quote must be a substring of a cited span. Semantic: a
local NLI model (MiniCheck-RoBERTa-Large, no LLM tokens) scores whether the concatenated cited spans support each claim. The
verifier is a *measurement* here, not a gate inside `kb ask`.

## 3. Provenance

Each answer is stored in `results/answers/<system>.jsonl` with: `parsed.claims[*].span_ids` (the lineage edge to a
sentence), the context paragraph ids, and `trace` (per hop: queries, lookup-or-search path, paragraphs added, the
sufficiency verdict and the facts ledger; the last entry maps every context paragraph to the hop that found it).
`python -m kb show <id>` and `kb ask` resolve this to `Document title — paragraph p, sentence s: "text"` plus the hop.
Measured on kb's 785 claims: 100.0% cite span ids that exist in the context, 99.8% quote verbatim, and the NLI verifier
supports 94.0% of claims from the cited spans (precision 94.0%). Only 5.2% of the spans in the context are ever cited.

## 4. Tradeoffs

- **Paragraph retrieval, sentence citation.** Precise locations at the price of sentence-splitting errors (regex plus
  abbreviation list; no trained splitter).
- **Free title graph over an LLM-built one.** No indexing tokens (HippoRAG 2 reports 9.2M, OPTIONS.md); but it misses surface
  variants and cannot link a paragraph that mentions an entity by a pronoun.
- **LLM gate over fixed hops.** kb costs 2.66 calls and 9.9k tokens per question against 1 call and 3.1k for RRF. That is an
  unequal compute budget: no equal-token baseline (for example a larger RRF context) was run, so part of the gain may be
  attributable to reading more tokens. The context budget itself is equal (10 paragraphs for every system).
- **Pooled 21,100-paragraph corpus.** A real retrieval problem with cross-question distractors, but not comparable to
  full-Wikipedia numbers.
- **One generator for every system.** A reasoning model whose outputs are sometimes truncated; `max_tokens` 8000 brought
  empty outputs from 5-9% to 0-3%.

## 5. Failure modes (measured, kb, n=400)

| stage | count | what it means |
|---|---|---|
| hop_miss | 114 | a gold document never entered the pool (rrf: 236) |
| generation_wrong | 55 | all gold documents in context, answer wrong |
| assembly_drop | 19 | gold in the pool but cut by the 10-paragraph budget (rrf: 0, by design) |
| retrieval_miss | 13 | no gold document in the first retrieval |
| correct but no gold doc cited (citation_gap) | 91 | answer right, provenance does not cover the evidence set |
| correct and cited | 108 | |

- **Confident multi-hop on the wrong bridge.** On the unanswerable probe (60 questions, last-step document masked) kb
  abstains less than RRF (0.617 vs 0.733) and gives a *different* answer more often (0.200 vs 0.133). The extra hop pulls in
  topically close paragraphs the answerer treats as enough. n=60, so a 0.1 gap is within noise, but the direction is
  the opposite of what we want and no abstain threshold was tuned.
- **4-hop is weak:** EM 0.200 (n=100), all-gold-in-context 0.12.
- **Closed-book contamination.** The model answers 25.0% of questions with no evidence (EM 0.297 when it answers), so
  correct answers are not all earned by retrieval; `citation_gap` is where that shows.
- **Verifier limits.** Cited sentences that start with a pronoun lack their subject, so the NLI model flags true claims as
  unsupported (we hand-checked: only about half of sampled flags were real, see EVAL.md). The unsupported rate is an upper bound.
- **Cost of a wrong hop.** Each hop is another ~4.0k-token check; a wrong bridge also displaces useful paragraphs.

## 6. What the measurements say about the design

The gated bridge hop is the entire measured gain: removing hops costs 0.155 EM [0.115, 0.198]; removing entity lookup
changes EM by -0.003 [-0.020, +0.015]; removing graph expansion changes it by +0.019 [-0.026, +0.064] (n=267, so
a graph benefit is not excluded). Graph expansion matters only without hops (kb with no hops 0.228 EM vs RRF 0.195).
Entity lookup returns purer paragraphs (75% in gold documents vs 16% for search) without improving answers. The
simplest equivalent system is search-only hops; we kept lookup and graph as shipped, and say so rather than claim they help.

## 7. Next steps (planned, not run)

1. Cross-encoder reranker on the final pool, and contextual chunk prefixes: both were in the plan and are not
   implemented or measured here.
2. Use the verifier as a gate: drop or flag unsupported claims, and feed unsupported-claim rate into abstention (the probe
   shows abstention is the weak spot). Prepend the document title to premises to remove pronoun false flags.
3. Tune an abstain-vs-answer threshold on the probe; re-tune the hop cap (the cap of 2 won by 2 tune questions).
4. An equal-token baseline (larger RRF context, or iterative retrieval) to separate "more tokens" from "better evidence".
5. 4-hop: a better bridge-entity resolver (aliases/redirects instead of exact titles).
