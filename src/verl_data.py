"""Native function-tool dataset for the current documented verl interface."""
import argparse
import json
from pathlib import Path
from datasets import Dataset
from .common import read_jsonl


def main():
    p=argparse.ArgumentParser();p.add_argument("--data-dir",default="data");a=p.parse_args()
    for split in ("train","dev","test","ood"):
        source=Path(a.data_dir)/f"{split}.jsonl"
        if not source.exists():continue
        rows=[]
        for task in read_jsonl(source):
            rows.append({"data_source":"course_records","agent_name":"tool_agent",
                "prompt":[{"role":"system","content":"Use lookup and calculate tools for private records. End with one JSON object containing the numeric final answer, for example {\"final\": 12}."},
                          {"role":"user","content":task["prompt"]}],
                "ability":"tool_use","reward_model":{"style":"rule","ground_truth":json.dumps(task["answer"])},
                "extra_info":{"id":task["id"],"split":task["split"]}})
        Dataset.from_list(rows).to_parquet(str(Path(a.data_dir)/f"{split}.parquet"))
        print(split,len(rows))


if __name__=="__main__":main()
