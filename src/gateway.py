"""Small authenticated HTTP job service with SQLite completion persistence.
This course service uses a single controller and read-only tools. It is not an HA service.
"""
import argparse
import json
import os
import queue
import sqlite3
import threading
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from .agent import run_task
from .common import read_jsonl


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8080)
    p.add_argument('--tasks',default='data/dev.jsonl');p.add_argument('--db',default='runs/gateway.sqlite')
    p.add_argument('--backend',choices=['oracle','http'],default='oracle');p.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--delay',type=float,default=0.);a=p.parse_args()
    token=os.environ.get('COURSE_GATEWAY_TOKEN')
    if not token:raise SystemExit('Set COURSE_GATEWAY_TOKEN before starting the service')
    tasks={t['id']:t for t in read_jsonl(a.tasks)};Path(a.db).parent.mkdir(parents=True,exist_ok=True)
    lock=threading.Lock();work=queue.Queue(maxsize=128)
    def sql(statement,params=(),fetch=False):
        with sqlite3.connect(a.db,timeout=30) as db:
            cursor=db.execute(statement,params)
            return cursor.fetchall() if fetch else None
    sql('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, task_id TEXT, status TEXT, result TEXT)')
    def worker():
        while True:
            job_id,task_id=work.get()
            try:
                sql('UPDATE jobs SET status=? WHERE id=?',('running',job_id))
                result=run_task(tasks[task_id],a.backend,a.base_url,delay=a.delay)
                sql('UPDATE jobs SET status=?,result=? WHERE id=?',('done',json.dumps(result),job_id))
            except Exception as exc:
                sql('UPDATE jobs SET status=?,result=? WHERE id=?',('failed',json.dumps({'error':str(exc)}),job_id))
            finally:work.task_done()
    for _ in range(a.workers):threading.Thread(target=worker,daemon=True).start()
    # Restart at task boundary. Read-only tools make re-execution safe in this teaching service.
    for job_id,task_id in sql("SELECT id,task_id FROM jobs WHERE status IN ('queued','running')",fetch=True):work.put((job_id,task_id))
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,payload):
            body=json.dumps(payload).encode();self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def allowed(self):
            if self.headers.get('Authorization')!='Bearer '+token:self.respond(401,{'error':'unauthorized'});return False
            return True
        def do_POST(self):
            if not self.allowed():return
            if self.path!='/jobs':self.respond(404,{'error':'not found'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8192:raise ValueError('invalid request size')
                body=json.loads(self.rfile.read(length));task_id=body['task_id'];job_id=body.get('job_id') or uuid.uuid4().hex
                if task_id not in tasks:raise ValueError('unknown task_id')
                if not isinstance(job_id,str) or len(job_id)>128:raise ValueError('invalid job_id')
                with lock:
                    old=sql('SELECT task_id,status FROM jobs WHERE id=?',(job_id,),True)
                    if old:
                        if old[0][0]!=task_id:self.respond(409,{'error':'job_id already bound to another task'});return
                        self.respond(200,{'job_id':job_id,'status':old[0][1]});return
                    if work.full():self.respond(503,{'error':'queue full'});return
                    sql('INSERT INTO jobs VALUES (?,?,?,?)',(job_id,task_id,'queued',None));work.put_nowait((job_id,task_id))
                self.respond(202,{'job_id':job_id,'status':'queued'})
            except (ValueError,KeyError,TypeError) as exc:self.respond(400,{'error':str(exc)})
        def do_GET(self):
            if not self.allowed():return
            job_id=self.path.removeprefix('/jobs/')
            rows=sql('SELECT task_id,status,result FROM jobs WHERE id=?',(job_id,),True)
            if not rows:self.respond(404,{'error':'not found'});return
            task_id,status,result=rows[0];self.respond(200,{'job_id':job_id,'task_id':task_id,'status':status,'result':json.loads(result) if result else None})
    server=ThreadingHTTPServer((a.host,a.port),Handler)
    print(f'gateway listening on {a.host}:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()


if __name__=='__main__':main()
