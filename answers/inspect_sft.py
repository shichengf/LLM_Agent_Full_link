import sys
from transformers import AutoTokenizer
from src.common import read_jsonl
from src.hf_train import encode_row

tok=AutoTokenizer.from_pretrained(sys.argv[1] if len(sys.argv)>1 else 'models/qwen05')
row=read_jsonl('data/train_sft.jsonl')[0];ids,labels=encode_row(row,tok,2048)
for token,label in zip(ids,labels):print(token,repr(tok.decode([token])),'TARGET' if label!=-100 else 'CONTEXT')
assert any(x==-100 for x in labels) and any(x!=-100 for x in labels)
