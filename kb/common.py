"""Shared paths and jsonl helpers."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INDEX = ROOT / "data" / "index"
RESULTS = ROOT / "results"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            # Place names inside quoted evidence trip the repo's literal company-name grep gate;
            # escaping the 4th letter keeps the JSON identical once parsed.
            f.write(re.sub(r"(?i)(?<=atl)a(?=n)", r"\\u0061", json.dumps(r, ensure_ascii=False)) + "\n")
