"""RoPE and KV-cache equivalence. No engine implementation is hidden here."""
import math
import torch


def rope(x,offset=0):
    # x: B,H,T,D. Adjacent-pair rotation convention, consistently used for Q and K.
    d=x.shape[-1];assert d%2==0
    inv=10000.**(-torch.arange(0,d,2,device=x.device).float()/d)
    angles=torch.arange(offset,offset+x.shape[-2],device=x.device)[:,None]*inv[None,:]
    c,s=angles.cos(),angles.sin();even,odd=x[...,0::2],x[...,1::2]
    return torch.stack((even*c-odd*s,even*s+odd*c),dim=-1).flatten(-2)


def attend(q,k,v,past=0):
    scores=q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])
    qpos=torch.arange(past,past+q.shape[-2],device=q.device)[:,None]
    kpos=torch.arange(k.shape[-2],device=q.device)[None,:]
    return scores.masked_fill(kpos>qpos,float('-inf')).softmax(-1)@v


def main():
    torch.manual_seed(9)
    q,k,v=[torch.randn(2,4,12,16) for _ in range(3)]
    full=attend(rope(q),rope(k),v)
    keys,values=[],[];parts=[]
    for pos in range(12):
        keys.append(rope(k[:,:,pos:pos+1],pos));values.append(v[:,:,pos:pos+1])
        parts.append(attend(rope(q[:,:,pos:pos+1],pos),torch.cat(keys,2),torch.cat(values,2),pos))
    cached=torch.cat(parts,2)
    print('max_difference',(full-cached).abs().max().item())
    assert torch.allclose(full,cached,atol=1e-5,rtol=1e-5)
    print('norm_difference',(rope(q).norm(dim=-1)-q.norm(dim=-1)).abs().max().item())


if __name__=='__main__':main()
