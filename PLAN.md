# PLAN

## Problem, from first principles

A question whose answer spans several documents needs a specific *evidence set*: every paragraph
on the reasoning chain, from at least two distinct sources. The context layer's job is to
**cover that evidence set inside a fixed context budget**, and then let a model answer with every
claim tied to a span it actually read.

Standard top-k RAG ranks by similarity to the question. Hop 1 is similar to the question; hop 2
usually is not (it is about a *bridge entity* that only appears once hop 1 is read). So the gap
we attack is coverage of later hops, not answer generation.

Three things must hold for the system to be trustworthy:

1. **Coverage**: all gold evidence documents end up in the assembled context.
2. **Grounding**: each answer claim is entailed by the span(s) it cites.
3. **Measurability**: each failure is attributable to one stage (retrieve, hop, assemble,
   generate, cite), so improvements are earned, not guessed.

## Scope

- Corpus: MuSiQue-Ans v1.0 dev, all 21,100 unique Wikipedia paragraphs pooled into one KB,
  grouped into documents by title. One KB for every question, not 20 candidates per question.
- Questions: seeded, hop-stratified subset of dev (see EVAL.md once written). Questions whose
  gold paragraphs come from a single title (54 in dev) are excluded: they are not multi-document.
- Text only, English, offline corpus. No UI; a CLI and a small Python API.

## Build steps (one or more commits each)

| # | Step | Done when |
|---|------|-----------|
| 0 | PLAN.md, OPTIONS.md, repo skeleton | committed |
| 1 ✓ | Data: pinned download, corpus build, seeded subset with committed id list + SHA-256 | `kb data` rebuilds identical ids and hash |
| 2 ✓ | Ingestion: paragraph -> sentence spans with stable ids `doc_id:para:sent` | every span round-trips to raw text |
| 3 ✓ | Indexes: BM25 + small dense (bge) over paragraphs | recall@k printed for both on subset |
| 4 ✓ | Baselines: closed-book, BM25, dense, hybrid RRF top-k | baseline answers + metrics saved as raw outputs |
| 5 ✓ | Entity graph: title-mention links between paragraphs (+ alias table) | graph stats; neighbour expansion recall delta |
| 6 ✓ | Context assembly: sufficiency check, bridge-entity follow-up hop (cap 2), per-doc cap, token budget | hop-2 recall delta vs hybrid |
| 7 | Cited synthesis: claims with span ids + verbatim quotes, abstain path | every claim's cited span exists and quote matches |
| 8 | Verifier: NLI check per claim (MiniCheck), drop/flag unsupported | citation precision delta |
| 9 | Eval harness: `make eval` reruns everything from raw outputs; bootstrap CIs; per-stage failure attribution | one command, results tables regenerated |
| 10 | Ablations: graph, hop cap 0/1/2, contextual prefix, reranker, verifier | ablation table in EVAL.md |
| 11 | DESIGN.md, EVAL.md, README.md, failure analysis, walkthrough script | docs complete |

Each step's commit message states what changed and the number it moved (or that it didn't).
Experiments that fail stay in history and get a line in OPTIONS.md.

## Metrics (detail in OPTIONS.md)

Retrieval: gold-doc recall@k, all-gold-docs-in-context rate. Answer: EM / F1 (MuSiQue scorer).
Citation: claim-level citation precision / recall (NLI). Multi-doc: share of correct answers that
cite >= 2 distinct gold documents. Cost: tokens and USD per question. Paired bootstrap 95% CIs.

## Open decisions

- Generator LLM and provider (needs an API key you can use for this). Fixed across all systems.
