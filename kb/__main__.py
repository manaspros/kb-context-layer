import sys

USAGE = "usage: python -m kb {data|ingest|index|recall|answer|score|graph|graph-eval|tune}"


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "data":
        from kb.data import run
    elif cmd == "ingest":
        from kb.ingest import run
    elif cmd == "index":
        from kb.retrieval import build as run
    elif cmd == "recall":
        from kb.retrieval import recall as run
    elif cmd == "answer":
        from kb.answer import run
    elif cmd == "graph":
        from kb.graph import build as run
    elif cmd == "graph-eval":
        from kb.graph import evaluate as run
    elif cmd == "tune":
        from kb.assemble import tune as run
    elif cmd == "score":
        from kb.score import run
    else:
        sys.exit(USAGE)
    run()


main()
