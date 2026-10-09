"""Single-GPU multi-turn REINFORCE with a leave-one-out baseline.
Not a production GRPO implementation. Uses exact generated token IDs at each turn.
Sampling temperature=1, top_k=0, top_p=1 so optimized log-probs match sampling.
"""
import argparse
import json
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from .common import read_jsonl,seed_all
from .tasks import SYSTEM,call_tool,verify


@torch.no_grad()
def rollout(model,tok,task,max_turns,max_new_tokens):
    messages=[{"role":"system","content":SYSTEM},{"role":"user","content":task["prompt"]}]
    segments=[];final=None
    for _ in range(max_turns):
        prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt").to(model.device)
        output=model.generate(prompt,attention_mask=torch.ones_like(prompt),do_sample=True,
            temperature=1.,top_k=0,top_p=1.,repetition_penalty=1.,max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        generated=output[:,prompt.shape[1]:]
        # Keep generated IDs, not a retokenized version of their decoded text.
        segments.append((output.detach(),prompt.shape[1]))
        text=tok.decode(generated[0],skip_special_tokens=True)
        messages.append({"role":"assistant","content":text})
        try:
            action=json.loads(text)
            if "final" in action:final=action["final"];break
            obs=call_tool(task,action["tool"],action["args"])
            messages.append({"role":"user","content":"TOOL_RESULT "+json.dumps(obs)})
        except (ValueError,KeyError,TypeError):break
    return segments,float(verify(task,final)),messages


def sequence_logprob(model,ids,prompt_length):
    logits=model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,:-1].float()
    targets=ids[:,1:]
    selected=logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    return selected[:,prompt_length-1:].sum()


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--data",default="data/train.jsonl")
    p.add_argument("--steps",type=int,default=20);p.add_argument("--group",type=int,default=4)
    p.add_argument("--max-turns",type=int,default=4);p.add_argument("--max-new-tokens",type=int,default=128)
    p.add_argument("--lr",type=float,default=1e-6);p.add_argument("--out",default="runs/rl-teaching")
    a=p.parse_args();assert a.group>1
    seed_all(17);tasks=read_jsonl(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(a.model);tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,attn_implementation="sdpa").cuda()
    # eval disables dropout without disabling autograd.
    model.eval();optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    for step in range(a.steps):
        task=tasks[step%len(tasks)]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            episodes=[rollout(model,tok,task,a.max_turns,a.max_new_tokens) for _ in range(a.group)]
        rewards=[e[1] for e in episodes];advantages=[r-(sum(rewards)-r)/(a.group-1) for r in rewards]
        optimizer.zero_grad(set_to_none=True)
        # Backward each turn immediately: less activation memory. Parameters stay fixed until the group ends.
        for (segments,_,_),adv in zip(episodes,advantages):
            for ids,n_prompt in segments:
                with torch.autocast("cuda",dtype=torch.bfloat16):loss=-sequence_logprob(model,ids,n_prompt)*adv/a.group
                loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        # Skip the update when all returns agree; AdamW decay alone would otherwise still change weights.
        if any(advantages):optimizer.step()
        row={"step":step+1,"reward_mean":sum(rewards)/a.group,"rewards":rewards,"advantages":advantages,
             "grad_norm":float(norm),"algorithm":"on-policy REINFORCE, leave-one-out baseline, no KL"}
        print(json.dumps(row),flush=True)
        with open(out/"metrics.jsonl","a") as f:f.write(json.dumps(row)+"\n")
        with open(out/"traces.jsonl","a") as f:
            for _,reward,messages in episodes:f.write(json.dumps({"task":task["id"],"reward":reward,"messages":messages})+"\n")
    model.to(torch.bfloat16).save_pretrained(out/"model");tok.save_pretrained(out/"model")


if __name__=="__main__":main()
