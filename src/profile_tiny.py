import argparse
from pathlib import Path
import torch
from .tiny_lm import TinyLM


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/profile.json');a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu';m=TinyLM(64,width=128).to(device)
    x=torch.randint(0,64,(8,64),device=device);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    for _ in range(3):opt.zero_grad();m(x).square().mean().backward();opt.step()
    with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as prof:
        for _ in range(3):
            with torch.profiler.record_function('training_step'):
                opt.zero_grad();m(x).square().mean().backward();opt.step()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);prof.export_chrome_trace(a.out)
    print(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=12))


if __name__=='__main__':main()
