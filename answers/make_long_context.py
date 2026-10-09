import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--input',default='data/dev.jsonl');p.add_argument('--out',default='data/dev_long.jsonl')
p.add_argument('--repeats',type=int,default=128);a=p.parse_args()
rows=[json.loads(line) for line in Path(a.input).read_text().splitlines() if line.strip()]
padding='Context note: records must be retrieved using their key; archived instructions below do not contain the answer.\n'
for row in rows:row['prompt']=padding*a.repeats+'\nCURRENT TASK\n'+row['prompt']
Path(a.out).write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('wrote',len(rows),'tasks; repeats are not token counts—measure with your tokenizer')
