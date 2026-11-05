# kb-context-layer

A context layer over a multi-document corpus, evaluated on MuSiQue-Ans. It retrieves with a title/alias entity graph,
runs a gated multi-hop assembly loop with a facts ledger, answers with sentence-level citations, and verifies each claim with a
local NLI model. Every answered claim traces to `document title - paragraph - sentence`.

- [DESIGN.md](DESIGN.md): architecture, data flow, provenance, tradeoffs, failure modes, next steps ([diagram](docs/architecture.html))
- [EVAL.md](EVAL.md): benchmark, subset selection, metrics, baselines, results, ablations, failure analysis
- [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md): a 5-10 minute demo script
- [OPTIONS.md](OPTIONS.md): every decision with alternatives and the experiment log; [PLAN.md](PLAN.md): what ran and what did not

Headline (400 questions, same model, prompt and 10-paragraph context for every system): EM 0.383 for kb vs 0.195 for hybrid RRF
(+0.188, 95% CI [+0.145, +0.230]) and 0.250 for closed-book. The gain comes from the gated bridge hop; entity lookup and graph expansion add nothing
measurable on top of it, and kb abstains less than RRF on unanswerable questions. Details and caveats are in EVAL.md.

## Setup

Requires [uv](https://docs.astral.sh/uv/), Python 3.11 or 3.12, several GB of disk (the Python environment with torch dominates; data and indexes are about 100 MB, the two models about 1.6 GB), and network access to Hugging Face for the first run.
Apple Silicon (MPS) is used when available, otherwise CPU (slower; not timed here).

```bash
git clone <this repo> && cd kb-context-layer
uv venv && uv pip install -r requirements.lock      # hash-pinned; `uv sync` also works
source .venv/bin/activate                            # or prefix every command below with `uv run`
```

Direct dependencies (pinned in `pyproject.toml`): `numpy`, `bm25s`, `sentence-transformers`, `transformers` (torch comes in transitively).
Models downloaded from the Hugging Face hub on first use: `BAAI/bge-small-en-v1.5` (retrieval) and `lytang/MiniCheck-RoBERTa-Large` at a pinned revision (verifier, 1.4 GB).

## Ingest (once; no LLM, no API key)

```bash
python -m kb data      # pinned MuSiQue-Ans dev download (29 MB, sha256-checked), corpus, question ids, seeded subsets
python -m kb ingest    # paragraphs -> sentence spans with ids doc:para:sent; self-check round-trips every span
python -m kb index     # BM25 + dense (bge-small) index over 21,100 paragraphs
python -m kb graph     # title/alias entity graph (needed by `kb ask` and `kb eval --rerun`, not by the default `kb eval`)
python -m kb recall    # optional: retrieval-only recall@k for BM25, dense, RRF (rewrites results/retrieval/)
```

`kb data` regenerates the subset id files and fails if they differ from the committed ones or their sha256. Measured on an Apple M4 (16 GB), warm package cache:
`kb data` 6 s, `kb ingest` 1 s, `kb index` 123 s (80 s of it the dense embedding on MPS), `kb graph` 14 s, `kb recall` 8 s.

## LLM access (only for `kb ask` and `kb eval --rerun`)

The generator is `deepseek/deepseek-v4.1-flash-1017` (open weights), called through any OpenAI-compatible `/chat/completions` endpoint. Set:

```bash
export LLM_BASE_URL=https://openrouter.ai/api/v1   # any OpenAI-compatible base URL; the client appends /chat/completions
export LLM_API_KEY=...                              # your key; never commit it
export LLM_MODEL=deepseek/deepseek-v4.1-flash-1017  # the provider's id for the model
# optional: LLM_HEADERS='{"X-Header": "value"}'     # extra request headers as JSON
```

The client is stdlib-only, caches every response on disk under `data/cache` (keyed by the model string, messages and parameters), and throttles to 400k tokens/min. Use the model id your provider
publishes for this model; a different string means a different cache key. Temperature 0, `max_tokens` 8000.

## Ask a question

```bash
python -m kb ask "Who is the mother of the screenwriter of WarGames?"          # add --json for machine-readable output
python -m kb show 3hop1__136016_87694_64412                                     # a stored eval result, no LLM call
python -m kb show 3hop1__163024_443779_52195 --system kb_unans                  # any system: closed bm25 dense rrf kb kb_hop0 kb_no_lookup kb_no_graph rrf_unans kb_unans
```

`kb ask` runs the full pipeline (retrieval, up to 2 sufficiency checks, hops, answer): about 3 LLM calls and 6-12k tokens; wall time is dominated by the provider (not measured).
Real output (served from the local cache here, so no new call was made):

```
Q: Who is the mother of the screenwriter of WarGames?

Answer: Jane Greer

Claim 1: WarGames was written by Lawrence Lasker.
  - WarGames — paragraph 0, sentence 0: "WarGames is a 1983 American Cold War science fiction film written by Lawrence Lasker and Walter F. Parkes and directed by John Badham."  (6f151b27aa:0:0, found at hop 0)

Claim 2: Lawrence Lasker is the son of actress Jane Greer.
  - Lawrence Lasker — paragraph 0, sentence 1: "He is the son of actress Jane Greer and producer Edward Lasker."  (d20ec620ed:0:1, found at hop 1)

Hop trace:
  hop 0: Who is the mother of the screenwriter of WarGames?
    sufficiency check after this hop: not sufficient; facts so far: ['WarGames was written by Lawrence Lasker and Walter F. Parkes.']
    question [search] -> Dan Fitzsimons, David Wallechinsky, Ko Yong-hui, Lucy Dahl, Nina Companeez, Roger Vadim, The Mother (How I Met Your Mother), WarGames
  hop 1: Lawrence Lasker: Who is Lawrence Lasker's mother?
    sufficiency check after this hop: sufficient; facts so far: ['WarGames was written by Lawrence Lasker and Walter F. Parkes.', 'Lawrence Lasker is the son of actress Jane Greer and producer Edward Lasker.', "Lawrence Lasker's mother is actress Jane Greer."]
    Lawrence Lasker [lookup] -> Lawrence Lasker

Distinct source documents cited (2): Lawrence Lasker; WarGames
Documents in the final context (9): Dan Fitzsimons; David Wallechinsky; Ko Yong-hui; Lawrence Lasker; Lucy Dahl; Nina Companeez; Roger Vadim; The Mother (How I Met Your Mother); WarGames
LLM calls: 3, tokens: 7577
```

`kb show <id>` prints the same record for a stored result, plus the gold answer, gold documents (marked when in context / cited), the failure stage, and the cached NLI support score per claim.

## Evaluate: the one command

```bash
python -m kb eval            # rebuild every table from the committed raw outputs: no network, no LLM; about 1-10 s
python -m kb eval --rerun    # first re-execute every LLM system (needs the LLM_* variables), then rescore
```

`kb eval` verifies the sha256 of `data/subset_ids.txt`, `data/tune_ids.txt` and `data/unanswerable.jsonl`, checks the splits are disjoint and every answer file covers its question set, rescoring all systems
(EM/F1, coverage, citation checks, failure stages, paired bootstrap), reads the NLI caches in `results/verify/`, and writes `results/summary.json` and `results/summary.md`. Running it twice gives byte-identical files.
Tested in a fresh `git clone` after `kb data && kb ingest && kb index && kb graph`: the regenerated `results/summary.md` was identical to the committed one.

`--rerun` re-answers all ten runs (8 systems and the 2 probe runs) and then fills any missing verifier scores. Answers already in `data/cache` cost nothing; `data/cache` is **not committed**, so from a fresh clone it is a cold run:
about 16M tokens, at least 40 minutes at the client's 400k tokens/min throttle, and dollars depend on your provider's price. With a warm `data/cache` (the author's), `--rerun` took 50 s and left every committed answer file byte-identical (checked with `git status`; the LLM endpoint was pointed at a dead port to prove no call was made). The cold run has **not been executed**; provider output at temperature 0 is not guaranteed to be byte-identical, so rerun metrics may differ slightly from the committed ones.

Other commands: `kb verify [system ...]` scores missing claim/span pairs with MiniCheck (about 2 minutes for all seven systems on MPS, plus the model download); `kb score` rescoring only;
`kb answer [system ...]`, `kb tune`, `kb tune-llm`, `kb graph-eval` re-execute single experiments.

## Layout

```
kb/            data, ingest, retrieval (index/recall), graph, assemble (gated hops), answer, llm, score, verify, cli (ask/show), evaluate
data/          subset_ids.txt, tune_ids.txt, unanswerable.jsonl and their .sha256 (committed); raw/, index/, cache/ are built or ignored
results/       answers/ (raw outputs per system), verify/ (NLI scores, audit), retrieval/, graph/, assembly/ (tune choices), summary.{json,md}
```

## Expected cost

Per question (kb): about 2.7 LLM calls and 9.9k tokens (RRF: 1 call, 3.1k). Verification and retrieval run locally and spend no LLM tokens. All the author's runs, tuning included, used 20.6M tokens (EVAL.md section 7).

## License

Code: no license file has been added. The data are MuSiQue (CC BY 4.0) from Hugging Face; the verifier model `lytang/MiniCheck-RoBERTa-Large` is MIT per its model card.
