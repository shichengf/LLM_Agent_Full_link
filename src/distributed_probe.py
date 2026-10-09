"""All ranks participate; NCCL on GPU, Gloo CPU fallback for correctness checks."""
import argparse
import datetime
import json
import os
import socket
import time
import torch
import torch.distributed as dist


def main():
    p=argparse.ArgumentParser();p.add_argument("--mb",type=int,default=64);p.add_argument("--iters",type=int,default=20)
    p.add_argument("--straggler-ms",type=int,default=0);a=p.parse_args()
    local=int(os.getenv("LOCAL_RANK","0"));gpu=torch.cuda.is_available()
    if gpu:torch.cuda.set_device(local)
    dist.init_process_group("nccl" if gpu else "gloo",timeout=datetime.timedelta(minutes=3))
    rank,world=dist.get_rank(),dist.get_world_size();device=f"cuda:{local}" if gpu else "cpu"
    print(json.dumps({"host":socket.gethostname(),"rank":rank,"local_rank":local,"world":world,"device":device}),flush=True)
    x=torch.ones(max(1,a.mb*1024*1024//4),device=device)
    for _ in range(5):x.fill_(1);dist.all_reduce(x)
    assert torch.all(x==world),"incorrect all-reduce sum"
    dist.barrier()
    if gpu:torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(a.iters):
        if rank==world-1 and a.straggler_ms:time.sleep(a.straggler_ms/1000)
        x.fill_(1);dist.all_reduce(x)
    if gpu:torch.cuda.synchronize()
    elapsed=torch.tensor([time.perf_counter()-t],device=device);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    if rank==0:
        seconds=elapsed.item()/a.iters
        print(json.dumps({"world":world,"payload_MB":x.numel()*4/1e6,"mean_collective_s":seconds,
                          "payload_GB_s":x.numel()*4/seconds/1e9,"note":"payload/time, not NCCL bus bandwidth"}))
    dist.destroy_process_group()


if __name__=="__main__":main()
