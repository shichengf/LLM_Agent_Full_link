import json
import math


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Reward only a final JSON object at the end of the assistant solution.
    Never use numbers in tool observations as the answer. A numeric answer is required.
    """
    if data_source!="course_records":raise ValueError(data_source)
    text=solution_str.strip()
    # Permit model EOS markers, but not arbitrary trailing prose or tool results.
    for marker in ("<|im_end|>","<|endoftext|>"):
        if text.endswith(marker):text=text[:-len(marker)].rstrip()
    position=text.rfind('{')
    if position<0:return 0.
    try:obj=json.loads(text[position:]);answer=obj["final"];target=json.loads(ground_truth)
    except (ValueError,KeyError,TypeError):return 0.
    return float(type(answer) in (int,float) and math.isfinite(answer) and
                 math.isclose(answer,target,rel_tol=1e-6,abs_tol=1e-6))
