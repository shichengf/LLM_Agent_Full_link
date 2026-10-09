"""Collect useful environment facts without dumping credentials."""
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path


def main():
    packages={}
    for name in ("torch","transformers","peft","vllm","sglang","ray","verl"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    data={"host":socket.gethostname(),"python":platform.python_version(),"cwd":str(Path.cwd()),"packages":packages,
          "disk_free_GB":shutil.disk_usage('.').free/1e9}
    for cmd,key in [(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv"],"gpus"),
                    (["nvidia-smi","topo","-m"],"topology")]:
        try:data[key]=subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT,timeout=20)
        except (FileNotFoundError,subprocess.SubprocessError) as exc:data[key]=str(exc)
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
