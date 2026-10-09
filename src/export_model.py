"""CPU full-checkpoint export. Requires RAM for full weights; not optimizer shards."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    state=torch.load(Path(a.run)/"checkpoint.pt",map_location="cpu",weights_only=False)
    cfg=state["args"];model=AutoModelForCausalLM.from_pretrained(cfg["model"],torch_dtype=torch.float32)
    if cfg["lora"]:
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
    model.load_state_dict(state["model"])
    if cfg["lora"]:model=model.merge_and_unload()
    model.to(torch.bfloat16).save_pretrained(a.out,safe_serialization=True)
    AutoTokenizer.from_pretrained(Path(a.run)/"tokenizer").save_pretrained(a.out)
    print(a.out)


if __name__=="__main__":main()
