"""Readable decoder-only LM: explicit attention, deterministic batches, resumable training."""
import argparse
import json
import math
import os
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .common import seed_all


class Attention(nn.Module):
    def __init__(self, width, heads, causal=True):
        super().__init__()
        assert width % heads == 0
        self.heads, self.causal = heads, causal
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.reshape(b, t, self.heads, c//self.heads).transpose(1, 2) for a in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(c//self.heads)
        if self.causal:
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = scores.softmax(-1) @ v
        return self.proj(out.transpose(1, 2).contiguous().reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, width, heads, causal=True):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = Attention(width, heads, causal)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ff(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab, width=64, heads=4, layers=2, context=64, causal=True):
        super().__init__()
        self.context = context
        self.tokens, self.positions = nn.Embedding(vocab, width), nn.Embedding(context, width)
        self.blocks = nn.Sequential(*[Block(width, heads, causal) for _ in range(layers)])
        self.norm, self.output = nn.LayerNorm(width), nn.Linear(width, vocab)

    def forward(self, ids):
        x = self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))
        return self.output(self.norm(self.blocks(x)))


def batch(data, n, context, generator, device):
    starts = torch.randint(len(data)-context-1, (n,), generator=generator)
    x = torch.stack([data[i:i+context] for i in starts])
    y = torch.stack([data[i+1:i+context+1] for i in starts])
    return x.to(device), y.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/tiny")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-causal", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    seed_all(a.seed)
    # Original course-generated corpus. Structured language, not a natural-language benchmark.
    sentences = [f"item {i}: {i%10} plus one equals {(i%10)+1}.\n" for i in range(3000)]
    train_text, val_text = "".join(sentences[:2400]), "".join(sentences[2400:])
    chars = sorted(set(train_text + val_text))
    encode = {c:i for i,c in enumerate(chars)}
    train, val = [torch.tensor([encode[c] for c in text]) for text in (train_text, val_text)]
    model = TinyLM(len(chars), a.width, layers=a.layers, context=a.context, causal=not a.no_causal).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    start = 0
    if a.resume:
        saved = torch.load(out/"checkpoint.pt", map_location=a.device, weights_only=False)
        if saved["chars"] != chars or saved["config"] != {"width":a.width,"layers":a.layers,"context":a.context,"causal":not a.no_causal}:
            raise ValueError("checkpoint configuration mismatch")
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu())
        start = saved["step"]
    t0 = time.perf_counter()
    for step in range(start, a.steps):
        x,y = batch(train,a.batch,a.context,generator,a.device)
        opt.zero_grad(set_to_none=True)
        logits=model(x); loss=F.cross_entropy(logits.flatten(0,1), y.flatten())
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (step+1)%20==0 or step==start:
            vg = torch.Generator().manual_seed(7)
            with torch.no_grad():
                vx,vy=batch(val,a.batch,a.context,vg,a.device)
                vl=F.cross_entropy(model(vx).flatten(0,1),vy.flatten()).item()
            if a.device.startswith("cuda"): torch.cuda.synchronize()
            row={"step":step+1,"train_loss":loss.item(),"val_loss":vl,
                 "tokens_s":(step-start+1)*a.batch*a.context/(time.perf_counter()-t0)}
            print(json.dumps(row),flush=True)
            with open(out/"metrics.jsonl","a") as f:f.write(json.dumps(row)+"\n")
    saved={"model":model.state_dict(),"optimizer":opt.state_dict(),"generator":generator.get_state(),
           "step":a.steps,"chars":chars,"config":{"width":a.width,"layers":a.layers,"context":a.context,"causal":not a.no_causal}}
    torch.save(saved,out/"checkpoint.tmp");os.replace(out/"checkpoint.tmp",out/"checkpoint.pt")
    ids=torch.tensor([[encode['i']]],device=a.device)
    sample_gen=torch.Generator(device=a.device).manual_seed(123)
    with torch.no_grad():
        for _ in range(100):
            probs=model(ids[:,-a.context:])[:,-1].softmax(-1)
            ids=torch.cat([ids,torch.multinomial(probs,1,generator=sample_gen)],dim=1)
    print("".join(chars[i] for i in ids[0].tolist()))


if __name__ == "__main__": main()
