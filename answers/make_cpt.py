import json
from pathlib import Path

Path('data').mkdir(exist_ok=True)
with open('data/cpt.jsonl','w') as f:
    for i in range(2000):
        a=i%97;b=(i*7)%89
        text=f'The sum of {a} and {b} is {a+b}. The maximum is {max(a,b)}. The mean is {(a+b)/2}.\n'
        f.write(json.dumps({'text':text})+'\n')
print('wrote 2000 original examples')
