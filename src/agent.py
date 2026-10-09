"""Same harness for a deterministic oracle, a mock policy, and real HTTP serving."""
import argparse
import concurrent.futures
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from .common import read_jsonl, write_jsonl
from .tasks import SYSTEM, call_tool, oracle_turns, verify


def chat(base_url, model, messages, temperature=0.0, max_tokens=192, timeout=120):
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("COURSE_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["COURSE_API_KEY"]
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"], result.get("usage", {})


def run_task(task, backend="oracle", base_url="http://127.0.0.1:8000/v1", model="course-model",
             max_turns=6, delay=0.0, fail_tool=False, temperature=0.0):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    start = time.perf_counter()
    trace, final, error, tokens = [], None, None, 0
    for turn in range(max_turns):
        try:
            if backend == "oracle":
                text, usage = oracle_turns(task)[min(turn, 2)], {}
            elif backend == "bad":
                text, usage = "not JSON", {}
            else:
                text, usage = chat(base_url, model, messages, temperature)
            tokens += usage.get("total_tokens", 0)
            trace.append({"turn": turn, "assistant": text, "usage": usage})
            messages.append({"role": "assistant", "content": text})
            action = json.loads(text)
            if not isinstance(action, dict):
                raise ValueError("action must be an object")
            if "final" in action:
                final = action["final"]
                break
            if delay:
                time.sleep(delay)
            if fail_tool:
                raise TimeoutError("injected tool timeout")
            observation = call_tool(task, action.get("tool"), action.get("args"))
            trace[-1]["observation"] = observation
            messages.append({"role": "user", "content": "TOOL_RESULT " + json.dumps(observation)})
        except (ValueError, TypeError, KeyError, TimeoutError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    if final is None and error is None:
        error = "max_turns"
    return {"id": task["id"], "split": task["split"], "success": verify(task, final),
            "final": final, "error": error, "latency_s": time.perf_counter() - start,
            "total_tokens": tokens, "tool_calls": sum("observation" in t for t in trace), "trace": trace}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/dev.jsonl")
    p.add_argument("--backend", choices=["oracle", "bad", "http"], default="oracle")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="course-model")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--delay", type=float, default=0)
    p.add_argument("--fail-tool", action="store_true")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--out", default="runs/agent.jsonl")
    p.add_argument("--db", help="Persistent completion ledger; one driver per database")
    a = p.parse_args()
    if not 0 <= a.shard < a.shards or a.workers < 1:
        p.error("invalid shard or workers")
    tasks = read_jsonl(a.tasks)[a.shard::a.shards]
    db = None
    done = {}
    if a.db:
        Path(a.db).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(a.db)
        db.execute("CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, result TEXT NOT NULL)")
        done = {row[0]: json.loads(row[1]) for row in db.execute("SELECT id,result FROM results")}
    pending = [t for t in tasks if t["id"] not in done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run_task, t, a.backend, a.base_url, a.model, a.max_turns,
                               a.delay, a.fail_tool, a.temperature) for t in pending]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            done[result["id"]] = result
            if db:
                db.execute("INSERT OR REPLACE INTO results VALUES (?,?)", (result["id"], json.dumps(result)))
                db.commit()
    if db:
        db.close()
    rows = [done[t["id"]] for t in tasks]
    write_jsonl(a.out, rows)
    print(json.dumps({"n": len(rows), "resumed": len(tasks) - len(pending),
                      "success_rate": sum(r["success"] for r in rows) / len(rows) if rows else None}))


if __name__ == "__main__":
    main()
