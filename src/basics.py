"""L01-L03: no third-party dependencies."""
import argparse
import json
from .common import read_jsonl, summarize, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output", default="runs/basics.jsonl")
    a = p.parse_args()
    rows = read_jsonl(a.input) if a.input else [
        {"id": "a", "success": True},
        {"id": "b", "success": False},
        {"id": "c", "success": True},
    ]
    result = summarize(rows)
    print(json.dumps(result))
    write_jsonl(a.output, [result])


if __name__ == "__main__":
    main()
