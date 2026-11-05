import sys

USAGE = "usage: python -m kb {data|ingest|index|recall}"


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
    else:
        sys.exit(USAGE)
    run()


main()
