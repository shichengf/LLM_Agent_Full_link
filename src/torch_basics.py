import argparse
import torch
from .common import seed_all
from .tiny_lm import TinyLM


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["tensor","regression","causal","policy"],default="tensor")
    a=p.parse_args();seed_all(7)
    if a.mode=="tensor":
        x=torch.arange(24,dtype=torch.float32).reshape(2,3,4)
        print("shape",x.shape,"mean",x.mean(-1),"broadcast",(x+torch.ones(4)).shape)
        w=torch.tensor(2.,requires_grad=True);loss=(w*3-9)**2;loss.backward()
        print("loss",loss.item(),"gradient",w.grad.item())
        assert w.grad.item()==-18
    elif a.mode=="regression":
        x=torch.linspace(-1,1,100).unsqueeze(1);y=3*x+2
        m=torch.nn.Linear(1,1);o=torch.optim.SGD(m.parameters(),lr=.1)
        for _ in range(300):
            loss=(m(x)-y).square().mean();o.zero_grad();loss.backward();o.step()
        print("weight",m.weight.item(),"bias",m.bias.item(),"loss",loss.item())
        assert loss.item()<1e-6
    elif a.mode=="causal":
        ids=torch.randint(0,20,(2,12));changed=ids.clone();changed[:,7:]=(changed[:,7:]+1)%20
        for causal in (True,False):
            m=TinyLM(20,causal=causal).eval()
            with torch.no_grad():d=(m(ids)[:,:7]-m(changed)[:,:7]).abs().max().item()
            print("causal",causal,"prefix_difference",d)
            assert (d<1e-6) if causal else (d>1e-6)
    else:
        logits=torch.nn.Parameter(torch.zeros(2));opt=torch.optim.SGD([logits],lr=.1)
        for _ in range(200):
            dist=torch.distributions.Categorical(logits=logits)
            actions=dist.sample((64,));rewards=actions.float()
            loss=-(dist.log_prob(actions)*(rewards-rewards.mean())).mean()
            opt.zero_grad();loss.backward();opt.step()
        print("action_probabilities",logits.softmax(-1).tolist())
        assert logits.softmax(-1)[1]>.9


if __name__=="__main__":main()
