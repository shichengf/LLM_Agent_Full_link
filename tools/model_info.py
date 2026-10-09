import argparse
import json
from transformers import AutoConfig

p=argparse.ArgumentParser();p.add_argument("model");p.add_argument("--tp",type=int,default=1);a=p.parse_args()
c=AutoConfig.from_pretrained(a.model)
heads=c.num_attention_heads;kv=getattr(c,"num_key_value_heads",heads)
print(json.dumps({"attention_heads":heads,"kv_heads":kv,"layers":c.num_hidden_layers,
                  "tp":a.tp,"attention_heads_divisible":heads%a.tp==0,
                  "kv_layout_divisible_or_replicable":kv%a.tp==0 or a.tp%kv==0}))
if heads%a.tp:raise SystemExit("Invalid attention-head partition for ordinary TP; choose a different TP or model.")
