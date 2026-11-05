# Walkthrough script (about 9 minutes)

Goal: show one multi-document answer with provenance, then one failure from the evaluation, and say honestly what the numbers do and do not show.
Before recording: indexes built (`kb data && kb ingest && kb index && kb graph`), `LLM_*` variables exported (needed only for the live `kb ask`; every `kb show` and `kb eval`
step works offline), terminal font large, `results/summary.md` open in a second pane.

## 0:00 - 1:00  The problem (slide: docs/architecture.html)

- A multi-hop question needs an *evidence set*: one paragraph per reasoning step, from at least two documents (our 400-question subset averages 2.84 gold documents per question).
- Hop 1 is similar to the question; hop 2 is not (it is about a bridge entity found only after reading hop 1). Top-k retrieval therefore misses later hops by construction:
  hybrid RRF has all gold documents in a 10-paragraph context for 25% of questions, 2% for 4-hop.
- The design is four layers: index with structure (sentence spans with stable ids, title graph), gated multi-hop assembly with a facts ledger, cited synthesis, verification.
  Point at the diagram; say that provenance is a lineage record answer -> claim -> span -> paragraph -> document -> which hop found it.

## 1:00 - 2:15  One command, and the headline

```bash
python -m kb eval
```

- Prints "expected runtime", checks the three id-file hashes, rebuilds every table from the committed raw outputs with no network and no LLM calls, writes `results/summary.md`.
- Show the first table: same model, prompt and 10-paragraph budget everywhere. closed-book EM 0.250; BM25 0.130; dense 0.188; RRF 0.195; **kb 0.383**.
- kb minus RRF: +0.188 EM, 95% CI [+0.145, +0.230] (paired bootstrap). Say up front: kb uses 2.7 calls and 9.9k tokens per question vs 1 and 3.1k, and we did not run an equal-token baseline.

## 2:15 - 5:00  A successful multi-document answer

1. The baseline fails (id `2hop__265989_531490`, "Who is the mother of the screenwriter of WarGames?", gold Jane Greer):

```bash
python -m kb show 2hop__265989_531490 --system rrf
```
   RRF has the WarGames paragraph, not Lawrence Lasker's; it abstains with NOT ANSWERABLE. Failure stage: hop_miss.

2. The same question through the kb pipeline, live (3 LLM calls; served from cache if you ran it before):

```bash
python -m kb ask "Who is the mother of the screenwriter of WarGames?"
```
   Walk the output top to bottom:
   - **Hop trace:** hop 0 retrieves, the sufficiency check says "not sufficient" and records the fact "WarGames was written by Lawrence Lasker and Walter F. Parkes" in the ledger; hop 1 resolves the title `Lawrence Lasker` (lookup path) and pulls that document.
   - **Claims:** two claims, each resolved to `Document title - paragraph p, sentence s: "sentence text"`, each labelled with the hop that found it.
   - **Distinct source documents cited: 2** (WarGames, Lawrence Lasker). That is the multi-document requirement, shown on the citations, not just on retrieval.
3. A 3-hop version, offline: `python -m kb show 3hop1__136016_87694_64412` (Leeds Minster -> St. Peter's Basilica at hop 1 -> Vatican City at hop 2, three documents cited, answer 11 February 1929 = gold). Note the NLI score printed per claim (local MiniCheck, no LLM tokens).

Talking point: the provenance numbers over all 785 kb claims: span ids valid 100%, quotes verbatim 99.8%, NLI-supported 94.0%.

## 5:00 - 7:30  A failure and an edge case from the evaluation

**Failure 1: wrong bridge, 3-hop (`3hop1__136016_87694_124169`, hop_miss).**

```bash
python -m kb show 3hop1__136016_87694_124169
```
- Gold answer 1952; kb answers "22 February 2001". Hop 1 searches Vatican City and pulls in the `Governor of Vatican City` document, but `St. Peter's Basilica` never enters the pool, then hop 2 searches a name that leads nowhere.
- Point at the claim with NLI p=0.37: the verifier flags it. Today `kb ask` only reports; wiring the verifier in as a gate is a next step.
- Failure stage counts: kb hop_miss 114 vs RRF 236, but assembly_drop 19 and generation_wrong 55 appear as the evidence reaches the model.

**Failure 2: kb abstains less than the baseline (`3hop1__163024_443779_52195`, unanswerable probe).**

```bash
python -m kb show 3hop1__163024_443779_52195 --system rrf_unans
python -m kb show 3hop1__163024_443779_52195 --system kb_unans
```
- The document that answers the last step (East Timor) is masked, so the right output is NOT ANSWERABLE. RRF abstains. kb chains Ambelau -> Indonesia-Timor Leste commission -> a search that returns the page of a later official (`Estanislau da Silva`) and answers "José Ramos-Horta" with a real, cited, topically close sentence.
- Aggregate on 60 probe questions: abstain RRF 0.733, kb 0.617; different answer 0.133 vs 0.200. n=60 so within noise, but the direction is the opposite of what we want and no abstain threshold was tuned.

**Edge case (optional, live):** `python -m kb ask "Where was the director of WarGames born?"` -> NOT ANSWERABLE with one cited claim ("WarGames was directed by John Badham"); the corpus has no birthplace paragraph for him. Good behaviour, and a reminder the corpus is a pooled dev set, not Wikipedia.

## 7:30 - 8:45  What the ablation says

- Removing the hop step: EM falls by 0.155 [0.115, 0.198]. That is the whole measured gain.
- Entity-first lookup: +0.003 [-0.015, +0.020]. No measurable effect on answers, although its paragraphs are purer (75% in gold documents vs 16% for search).
- Graph expansion: -0.019 [-0.064, +0.026], n=267: nothing measurable once hops are on; it helps only without hops (0.228 vs 0.195).
- Other honest points: closed-book already scores 0.250 (contamination); 91 of 199 correct kb answers do not cite a gold document (citation_gap); 4-hop EM is 0.200.

## 8:45 - 9:30  Limits and next steps

- Not run, listed as next steps: cross-encoder reranker, contextual chunk prefix, equal-token baseline, verifier as a gate, abstain threshold.
- Hop cap 2 was chosen on a two-question tune margin; the unanswerable probe has n=60; kb_no_graph covers 267 of 400 questions.
- Everything is reproducible with `kb eval`; raw outputs are in `results/answers/` and `results/verify/`.
