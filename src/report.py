import argparse
import json
import math
from collections import Counter
from .common import read_jsonl


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    c = (p + z*z/(2*n)) / (1+z*z/n)
    d = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [c-d, c+d]


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    i = (len(values)-1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return values[lo] + (values[hi]-values[lo]) * (i-lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    a = p.parse_args()
    for filename in a.files:
        rows = read_jsonl(filename)
        n = len(rows)
        k = sum(r.get("success", False) for r in rows)
        print(json.dumps({"file": filename, "n": n, "success_rate": k/n if n else None,
            "wilson95": wilson(k, n),
            "latency_p50_s": percentile([r["latency_s"] for r in rows if "latency_s" in r], .5),
            "latency_p95_s": percentile([r["latency_s"] for r in rows if "latency_s" in r], .95),
            "total_tokens": sum(r.get("total_tokens", 0) for r in rows),
            "errors": dict(Counter(r.get("error") for r in rows if r.get("error")))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
