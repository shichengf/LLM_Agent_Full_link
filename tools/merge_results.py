import argparse
from src.common import read_jsonl,write_jsonl

p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
rows=[r for f in a.files for r in read_jsonl(f)];ids=[r['id'] for r in rows]
if len(ids)!=len(set(ids)):raise SystemExit('Duplicate task IDs: check worker shard assignments')
write_jsonl(a.out,sorted(rows,key=lambda x:x['id']))
print('merged',len(rows),'unique tasks')
