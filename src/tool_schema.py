"""Single schema shared by native SFT, evaluation, and the verl adapter."""
TOOLS=[
    {"type":"function","function":{"name":"lookup","description":"Read one private record by its exact key.","parameters":{"type":"object","properties":{"key":{"type":"string","description":"The record key provided in the task."}},"required":["key"]}}},
    {"type":"function","function":{"name":"calculate","description":"Compute a supported statistic of supplied numbers.","parameters":{"type":"object","properties":{"op":{"type":"string","description":"One of sum, max, mean, or count_positive.","enum":["sum","max","mean","count_positive"]},"values":{"type":"array","items":{"type":"number"},"description":"Between 1 and 128 finite numbers."}},"required":["op","values"]}}}
]
