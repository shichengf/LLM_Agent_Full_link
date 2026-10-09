import sys
from transformers import AutoTokenizer

t=AutoTokenizer.from_pretrained(sys.argv[1]);messages=[{'role':'user','content':'What is 2 + 3?'}]
text=t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
ids=t.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
print(repr(text));print(ids);print('eos',t.eos_token,t.eos_token_id,'pad',t.pad_token_id)
