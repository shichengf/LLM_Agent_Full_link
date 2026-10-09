"""Transparent SFT / continued-pretraining, single GPU, DDP or FSDP1.
Fixed examples per update, per-example token-mean objective, stateless batch sampling.
FSDP full checkpoint is intentionally expensive: measure it before using at scale.
"""
import argparse
import contextlib
import datetime
import functools
import json
import os
import random
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .common import read_jsonl, seed_all


def encode_row(row, tok, max_length):
    if "input_ids" in row:
        ids,labels=row["input_ids"],row["labels"]
        if len(ids)!=len(labels) or len(ids)>max_length:raise ValueError("invalid pretokenized sample length")
        if not any(x!=-100 for x in labels[1:]):raise ValueError("zero supervised tokens")
        return ids,labels
    if "text" in row:
        ids=tok(row["text"],add_special_tokens=False)["input_ids"]+[tok.eos_token_id]
        ids=ids[:max_length]
        return ids,ids.copy()
    prompt=tok.apply_chat_template(row["messages"],tokenize=True,add_generation_prompt=True)
    target=tok(row["response"],add_special_tokens=False)["input_ids"]+[tok.eos_token_id]
    if len(prompt)+len(target)>max_length:
        raise ValueError(f"sample {row.get('id')} exceeds max length; do not silently drop its answer")
    return prompt+target,[-100]*len(prompt)+target


def collate(rows,pad,device):
    length=max(len(r[0]) for r in rows)
    ids=torch.full((len(rows),length),pad,dtype=torch.long,device=device)
    labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
    for i,(x,y) in enumerate(rows):
        ids[i,:len(x)]=torch.tensor(x,device=device);labels[i,:len(y)]=torch.tensor(y,device=device);mask[i,:len(x)]=1
    return ids,mask,labels


def example_mean_loss(logits,labels):
    shifted=labels[:,1:];valid=shifted.ne(-100)
    if not torch.all(valid.sum(-1)>0):raise ValueError("zero supervised tokens")
    token_loss=F.cross_entropy(logits[:,:-1].float().transpose(1,2),shifted,ignore_index=-100,reduction="none")
    return (token_loss.sum(-1)/valid.sum(-1)).mean()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True);p.add_argument("--data",default="data/train_sft.jsonl")
    p.add_argument("--mode",choices=["single","ddp","fsdp"],default="single")
    p.add_argument("--steps",type=int,default=100);p.add_argument("--micro-batch",type=int,default=1)
    p.add_argument("--global-batch",type=int,default=32);p.add_argument("--max-length",type=int,default=2048)
    p.add_argument("--lr",type=float,default=2e-5);p.add_argument("--seed",type=int,default=17)
    p.add_argument("--lora",action="store_true");p.add_argument("--grad-checkpoint",action="store_true")
    p.add_argument("--out",default="runs/sft");p.add_argument("--save-every",type=int,default=50)
    p.add_argument("--resume",action="store_true");a=p.parse_args()
    distributed=a.mode!="single";local=int(os.getenv("LOCAL_RANK","0"))
    if not torch.cuda.is_available():raise SystemExit("HF training labs require a GPU; CPU foundation labs are separate")
    torch.cuda.set_device(local);device=torch.device("cuda",local)
    if distributed:dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=20))
    rank=dist.get_rank() if distributed else 0;world=dist.get_world_size() if distributed else 1
    if a.global_batch%(world*a.micro_batch):raise ValueError("global batch must divide world_size * micro_batch")
    accum=a.global_batch//(world*a.micro_batch)
    if a.lora and a.mode=="fsdp":raise ValueError("course FSDP lab uses full parameters; LoRA lab uses single/DDP")
    seed_all(a.seed)
    tok=AutoTokenizer.from_pretrained(a.model);tok.pad_token=tok.eos_token if tok.pad_token_id is None else tok.pad_token
    encoded=[encode_row(r,tok,a.max_length) for r in read_jsonl(a.data)]
    if not encoded:raise ValueError("empty training data")
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,attn_implementation="sdpa")
    model.config.use_cache=False
    if a.lora:
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
    if a.grad_checkpoint:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    if a.mode=="fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision,StateDictType,FullStateDictConfig,FullOptimStateDictConfig
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        layer_type=type(model.model.layers[0])
        policy=functools.partial(transformer_auto_wrap_policy,transformer_layer_cls={layer_type})
        model=FSDP(model,auto_wrap_policy=policy,device_id=device,use_orig_params=True,
                   mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16))
    else:
        model=model.to(device)
        if distributed:model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/"checkpoint.pt";start=0
    if a.resume:
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        for key in ("global_batch","micro_batch","seed","data","lora","max_length"):
            if saved["args"][key]!=getattr(a,key):raise ValueError(f"resume mismatch: {key}")
        if saved["world"]!=world:raise ValueError("exact continuation lab requires same world size")
        if a.mode=="fsdp":
            with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=False)):
                model.load_state_dict(saved["model"])
            optim_state=FSDP.scatter_full_optim_state_dict(saved["optimizer"] if rank==0 else None,model)
            optimizer.load_state_dict(optim_state)
        else:
            (model.module if distributed else model).load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        start=saved["step"]
    for step in range(start,a.steps):
        # No dropout in the chosen Qwen2.5 configuration; deterministic step seed still supports replay.
        seed_all(a.seed+step*world+rank)
        rng=random.Random(a.seed+step)
        global_indices=[rng.randrange(len(encoded)) for _ in range(a.global_batch)]
        indices=global_indices[rank::world]
        optimizer.zero_grad(set_to_none=True);total_loss=0.;tokens=0
        torch.cuda.synchronize();t0=time.perf_counter()
        for micro in range(accum):
            rows=[encoded[i] for i in indices[micro*a.micro_batch:(micro+1)*a.micro_batch]]
            ids,mask,labels=collate(rows,tok.pad_token_id,device)
            # FSDP synchronizes each microbatch to avoid full-gradient no_sync memory spikes.
            sync=model.no_sync() if a.mode=="ddp" and micro<accum-1 else contextlib.nullcontext()
            with sync,torch.autocast("cuda",dtype=torch.bfloat16):
                loss=example_mean_loss(model(input_ids=ids,attention_mask=mask).logits,labels)
                (loss/accum).backward()
            total_loss+=loss.detach().float()/accum;tokens+=mask.sum().item()
        grad=model.clip_grad_norm_(1.) if a.mode=="fsdp" else torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-t0
        stats=torch.tensor([float(total_loss),tokens],device=device)
        duration=torch.tensor(elapsed,device=device)
        peak=torch.tensor(torch.cuda.max_memory_allocated()/1e9,device=device)
        if distributed:
            dist.all_reduce(stats);dist.all_reduce(duration,op=dist.ReduceOp.MAX);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if rank==0:
            row={"step":step+1,"world":world,"loss":stats[0].item()/world,"global_batch":a.global_batch,
                 "tokens_s":stats[1].item()/duration.item(),"step_s":duration.item(),
                 "peak_GB_max_rank":peak.item(),"grad_norm":float(grad)}
            print(json.dumps(row),flush=True)
            with open(out/"metrics.jsonl","a") as f:f.write(json.dumps(row)+"\n")
        if (step+1)%a.save_every==0 or step+1==a.steps:
            save_start=time.perf_counter()
            if a.mode=="fsdp":
                with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True),FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
                    state=model.state_dict();opt_state=FSDP.optim_state_dict(model,optimizer)
            else:
                state=(model.module if distributed else model).state_dict();opt_state=optimizer.state_dict()
            if rank==0:
                torch.save({"model":state,"optimizer":opt_state,"step":step+1,"args":vars(a),"world":world},out/"checkpoint.tmp")
                os.replace(out/"checkpoint.tmp",checkpoint);tok.save_pretrained(out/"tokenizer")
                print(json.dumps({"checkpoint_seconds":time.perf_counter()-save_start}),flush=True)
            if distributed:dist.barrier()
    if distributed:dist.destroy_process_group()


if __name__=="__main__":main()
