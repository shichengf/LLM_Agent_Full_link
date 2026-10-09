"""Evaluation harness for checkpoints trained with native function tools in verl."""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.request
from .common import read_jsonl,write_jsonl
from .tasks import call_tool,verify
from .verl_reward import compute_score
from .tool_schema import TOOLS

def run(task,a):
    messages=[{"role":"system","content":"Use lookup and calculate tools for private records. End with one JSON object containing the numeric final answer, for example {\"final\": 12}."},{"role":"user","content":task['prompt']}]
    start=time.perf_counter();success=False;error=None;tokens=0;calls=0
    try:
        for _ in range(a.max_turns):
            body={"model":"course-model","messages":messages,"tools":TOOLS,"tool_choice":"auto","temperature":0.,"max_tokens":256}
            headers={"Content-Type":"application/json"}
            if os.getenv('COURSE_API_KEY'):headers['Authorization']='Bearer '+os.environ['COURSE_API_KEY']
            req=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers=headers)
            with urllib.request.urlopen(req,timeout=120) as r:response=json.load(r)
            tokens+=response.get('usage',{}).get('total_tokens',0)
            m=response['choices'][0]['message']
            # Keep only supported chat fields; do not echo engine-specific metadata.
            m={k:m[k] for k in ('role','content','tool_calls') if k in m and m[k] is not None}
            messages.append(m)
            if m.get('tool_calls'):
                for c in m['tool_calls']:
                    fn=c['function'];obs=call_tool(task,fn['name'],json.loads(fn['arguments']));calls+=1
                    messages.append({'role':'tool','tool_call_id':c['id'],'content':json.dumps(obs)})
            else:
                success=bool(compute_score('course_records',m.get('content',''),json.dumps(task['answer'])))
                break
        else:error='max_turns'
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    return {'id':task['id'],'success':success,'error':error,'latency_s':time.perf_counter()-start,'total_tokens':tokens,'tool_calls':calls,'trace':messages}


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',default='data/dev.jsonl');p.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--max-turns',type=int,default=6);p.add_argument('--out',default='runs/native.jsonl');a=p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(lambda t:run(t,a),read_jsonl(a.tasks)))
    write_jsonl(a.out,rows);print('completed',len(rows))


if __name__=='__main__':main()
