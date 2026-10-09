"""Native function tools. COURSE_DATA_DIR must point to a shared task directory."""
import functools
import os
from pathlib import Path
from verl.tools.function_tool import function_tool
from src.common import read_jsonl
from src.tasks import call_tool
from src.tool_schema import TOOLS


@functools.lru_cache(maxsize=1)
def records():
    directory=Path(os.environ["COURSE_DATA_DIR"])
    result={}
    for split in ("train","dev","test","ood"):
        filename=directory/f"{split}.jsonl"
        if filename.exists():
            result.update({t["key"]:t for t in read_jsonl(filename)})
    return result


@function_tool(schema=TOOLS[0])
def lookup(key: str) -> dict:
    """Read one private record by its exact key.

    Args:
        key: The record key provided in the task.
    """
    task=records().get(key)
    if task is None:return {"error":"unknown record key"}
    return {"values":task["values"]}


@function_tool(schema=TOOLS[1])
def calculate(op: str, values: list[float]) -> dict:
    """Compute a supported statistic of supplied numbers.

    Args:
        op: One of sum, max, mean, or count_positive.
        values: Between 1 and 128 finite numbers.
    """
    try:return call_tool({},"calculate",{"op":op,"values":values})
    except ValueError as exc:return {"error":str(exc)}
