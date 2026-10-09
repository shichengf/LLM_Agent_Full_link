import argparse
import json
from transformers import AutoTokenizer
from src.common import read_jsonl,write_jsonl
from src.tool_schema import TOOLS
from src.tasks import call_tool


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',default='data/train.jsonl')
    p.add_argument('--out',default='data/train_native_sft.jsonl');a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model);rows=[]
    for task in read_jsonl(a.input):
        messages=[{'role':'system','content':'Use lookup and calculate tools for private records. End with one JSON object containing the numeric final answer, for example {"final": 12}.'},{'role':'user','content':task['prompt']}]
        for name,args in [('lookup',{'key':task['key']}),('calculate',{'op':task['op'],'values':task['values']}),(None,None)]:
            if name:
                action={'role':'assistant','content':'','tool_calls':[{'id':'course-call-'+name,'type':'function','function':{'name':name,'arguments':args}}]}
            else:action={'role':'assistant','content':json.dumps({'final':task['answer']})}
            prefix=tok.apply_chat_template(messages,tools=TOOLS,tokenize=True,add_generation_prompt=True)
            full=tok.apply_chat_template(messages+[action],tools=TOOLS,tokenize=True,add_generation_prompt=False)
            if full[:len(prefix)]!=prefix:raise ValueError('Template not prefix stable: inspect before defining a target mask')
            rows.append({'id':task['id'],'input_ids':full,'labels':[-100]*len(prefix)+full[len(prefix):]})
            messages.append(action)
            if name:messages.append({'role':'tool','tool_call_id':'course-call-'+name,'name':name,'content':json.dumps(call_tool(task,name,args))})
    write_jsonl(a.out,rows);print('native supervised turns',len(rows))


if __name__=='__main__':main()
