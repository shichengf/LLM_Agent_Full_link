import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from src.common import write_jsonl
from src.tasks import make_task


class GatewayTests(unittest.TestCase):
    def test_submit_query_and_idempotency(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)/'tasks.jsonl';write_jsonl(data,[make_task(0,'dev'),make_task(1,'dev')])
            env=dict(os.environ,COURSE_GATEWAY_TOKEN='local-test-token')
            proc=subprocess.Popen([sys.executable,'-m','src.gateway','--port',str(port),'--tasks',str(data),'--db',str(Path(directory)/'jobs.sqlite')],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            def call(path,body=None,auth=True):
                headers={'Content-Type':'application/json'}
                if auth:headers['Authorization']='Bearer local-test-token'
                req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
                with urllib.request.urlopen(req,timeout=2) as response:return json.load(response)
            try:
                for _ in range(100):
                    try:
                        result=call('/jobs',{'task_id':'dev-0','job_id':'test-job'});break
                    except urllib.error.URLError:time.sleep(.02)
                else:self.fail('gateway did not start')
                for _ in range(100):
                    result=call('/jobs/test-job')
                    if result['status']=='done':break
                    time.sleep(.01)
                self.assertTrue(result['result']['success'])
                self.assertEqual(call('/jobs',{'task_id':'dev-0','job_id':'test-job'})['job_id'],'test-job')
                with self.assertRaises(urllib.error.HTTPError) as ctx:call('/jobs',{'task_id':'dev-1','job_id':'test-job'})
                self.assertEqual(ctx.exception.code,409)
                with self.assertRaises(urllib.error.HTTPError) as ctx:call('/jobs/test-job',auth=False)
                self.assertEqual(ctx.exception.code,401)
            finally:
                proc.terminate();proc.wait(timeout=5)


if __name__=='__main__':unittest.main()
