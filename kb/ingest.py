"""Step 2: paragraph -> sentence spans with exact character offsets."""
import re

from kb.common import INDEX, read_jsonl, write_jsonl

# Candidate boundary: sentence punctuation (+ closing quotes/brackets), whitespace, then a
# capital, digit or opening quote/bracket.
BOUNDARY = re.compile(r"""[.!?]+["')\]]*\s+(?=["'(\[]?[A-Z0-9])""")
ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "gen", "col", "lt", "sgt",
    "capt", "cmdr", "adm", "gov", "sen", "rep", "rev", "hon", "pres", "vs", "etc", "no", "vol",
    "inc", "ltd", "co", "corp", "dept", "est", "approx", "fig", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k", "e.g", "i.e", "ca",
}


def split_sentences(text: str) -> list[tuple[int, int]]:
    # ponytail: regex + abbreviation list; misses unlisted abbreviations ("Ave.", "Bros.") and
    # lowercase-initial sentences. Upgrade to a trained splitter (e.g. pysbd) if citation
    # spans measurably split mid-sentence.
    cuts = [0]
    for m in BOUNDARY.finditer(text):
        word = re.search(r"(\S+)$", text[: m.start()])
        token = word.group(1).lower().strip("\"'([") if word else ""
        if text[m.start()] == "." and m.group().startswith(".") and len(m.group().rstrip()) == 1:
            if token in ABBREV or re.fullmatch(r"[a-z]", token) or re.fullmatch(r"(?:[a-z]\.)+[a-z]", token):
                continue  # abbreviation, single initial ("J. R. R."), or "U.S"
        cuts.append(m.end())
    cuts.append(len(text))
    spans = []
    for a, b in zip(cuts, cuts[1:]):
        seg = text[a:b]
        s = a + (len(seg) - len(seg.lstrip()))
        e = a + len(seg.rstrip())
        if e > s:
            spans.append((s, e))
    return spans


def run() -> None:
    corpus = read_jsonl(INDEX / "corpus.jsonl")
    spans, lens = [], []
    for para in corpus:
        text = para["text"]
        sp = split_sentences(text)
        assert sp, f"paragraph without spans: {para['para_id']}"
        for s, (start, end) in enumerate(sp):
            sent = text[start:end]
            assert sent and sent == sent.strip(), f"empty/unstripped span {para['para_id']}:{s}"
            spans.append({
                "span_id": f"{para['para_id']}:{s}", "para_id": para["para_id"],
                "doc_id": para["doc_id"], "start": start, "end": end, "text": sent,
            })
        lens.append(len(sp))
    # Round-trip against the raw corpus, independent of the stored text field.
    by_para = {p["para_id"]: p["text"] for p in corpus}
    for sp in spans:
        assert by_para[sp["para_id"]][sp["start"]:sp["end"]] == sp["text"], sp["span_id"]
    assert len({s["span_id"] for s in spans}) == len(spans)
    write_jsonl(INDEX / "spans.jsonl", spans)
    print(f"paragraphs={len(corpus)} spans={len(spans)} "
          f"sentences/paragraph mean={len(spans)/len(corpus):.2f} max={max(lens)} "
          f"single-sentence paragraphs={sum(l == 1 for l in lens)}")
    print("self-check passed: round-trip, no empty spans, every paragraph >=1 span")
