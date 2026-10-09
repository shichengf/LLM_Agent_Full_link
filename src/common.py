import json
import os
import random
import tempfile
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        tmp = f.name
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def summarize(rows):
    if not rows:
        return {"n": 0, "success_rate": None}
    if any(type(r.get("success")) is not bool for r in rows):
        raise ValueError("success must be a Boolean in every record")
    return {"n": len(rows), "success_rate": sum(r["success"] for r in rows) / len(rows)}
