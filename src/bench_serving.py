"""Closed-loop SSE benchmark. Records actual output tokens and content-first TTFT."""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.request
from .common import write_jsonl
from .report import percentile


def request_one(index,a):
    words=a.lengths[index%len(a.lengths)]
    prompt=("Explain one useful fact about distributed computing. " * (words//8+1))
    payload={"model":a.model,"messages":[{"role":"user","content":prompt}],"stream":True,
             "stream_options":{"include_usage":True},"temperature":0.,"max_tokens":a.max_tokens}
    headers={"Content-Type":"application/json"}
    if os.getenv("COURSE_API_KEY"):headers["Authorization"]="Bearer "+os.environ["COURSE_API_KEY"]
    request=urllib.request.Request(a.base_url.rstrip("/")+"/chat/completions",data=json.dumps(payload).encode(),headers=headers)
    start=time.perf_counter();first=None;usage={};error=None
    try:
        with urllib.request.urlopen(request,timeout=a.timeout) as response:
            for raw in response:
                line=raw.decode().strip()
                if not line.startswith("data:"):continue
                data=line[5:].strip()
                if data=="[DONE]":break
                event=json.loads(data)
                if event.get("usage"):usage=event["usage"]
                choices=event.get("choices") or []
                if choices and choices[0].get("delta",{}).get("content") and first is None:first=time.perf_counter()
    except Exception as exc:error=f"{type(exc).__name__}: {exc}"
    end=time.perf_counter();output=usage.get("completion_tokens")
    return {"id":str(index),"success":error is None and first is not None,"error":error,
            "ttft_s":first-start if first else None,"latency_s":end-start,
            "output_tokens":output,"prompt_tokens":usage.get("prompt_tokens"),
            "tpot_s":(end-first)/(output-1) if first and output and output>1 else None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--base-url",default="http://127.0.0.1:8000/v1")
    p.add_argument("--model",default="course-model");p.add_argument("--requests",type=int,default=64)
    p.add_argument("--concurrency",type=int,default=8);p.add_argument("--max-tokens",type=int,default=128)
    p.add_argument("--lengths",type=int,nargs="+",default=[64,512,2048]);p.add_argument("--timeout",type=float,default=180)
    p.add_argument("--out",default="runs/serving.jsonl");a=p.parse_args()
    warmup=request_one(0,a)
    if not warmup["success"]:raise SystemExit(f"warmup failed: {warmup}")
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:rows=list(pool.map(lambda i:request_one(i,a),range(a.requests)))
    elapsed=time.perf_counter()-start;write_jsonl(a.out,rows)
    print(json.dumps({"wall_s":elapsed,"requests":len(rows),"failed":sum(not r["success"] for r in rows),
        "output_tokens_s":sum(r["output_tokens"] or 0 for r in rows)/elapsed,
        "ttft_p95_s":percentile([r["ttft_s"] for r in rows if r["ttft_s"] is not None],.95),
        "latency_p95_s":percentile([r["latency_s"] for r in rows],.95),
        "note":"closed-loop load; repeated prompts have warm-cache effects; TPOT is an aggregate approximation"}))


if __name__=="__main__":main()
