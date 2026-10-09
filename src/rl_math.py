"""Inspect advantages and clipping without needing any machine learning package."""
import json
import math


def group_advantages(rewards):
    mean=sum(rewards)/len(rewards)
    variance=sum((r-mean)**2 for r in rewards)/len(rewards)
    return [(r-mean)/max(math.sqrt(variance),1e-8) for r in rewards]


def clipped_objective(ratio,advantage,eps=.2):
    clipped=min(max(ratio,1-eps),1+eps)
    return min(ratio*advantage,clipped*advantage)


if __name__=="__main__":
    print(json.dumps({"rewards":[0,0,1,1],"advantages":group_advantages([0,0,1,1]),
                      "all_equal":group_advantages([1,1,1,1])}))
    for adv in (-1,1):
        for ratio in (.6,1.,1.4):print("adv",adv,"ratio",ratio,"objective",clipped_objective(ratio,adv))
