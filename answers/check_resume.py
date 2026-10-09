import argparse
import torch

p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');a=p.parse_args()
x=torch.load(a.a,map_location='cpu',weights_only=False)['model']
y=torch.load(a.b,map_location='cpu',weights_only=False)['model']
diff=max((x[k]-y[k]).abs().max().item() for k in x)
print('maximum_parameter_difference',diff)
assert diff<1e-6
