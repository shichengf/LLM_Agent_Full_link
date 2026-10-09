import unittest
try:
    import torch
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False


@unittest.skipUnless(HAS_TORCH,'torch not installed')
class TorchTests(unittest.TestCase):
    def test_causality(self):
        from src.tiny_lm import TinyLM
        torch.manual_seed(1);m=TinyLM(10);a=torch.randint(0,10,(2,8));b=a.clone();b[:,4:]=(b[:,4:]+1)%10
        self.assertTrue(torch.allclose(m(a)[:,:4],m(b)[:,:4],atol=1e-6))
    def test_cache_and_rope(self):
        from src.attention_internals import main
        main()
    def test_optimizer_changes(self):
        from src.tiny_lm import TinyLM
        m=TinyLM(10);x=torch.randint(0,10,(2,8));before=m.tokens.weight.detach().clone()
        m(x).square().mean().backward();torch.optim.SGD(m.parameters(),lr=.1).step()
        self.assertFalse(torch.equal(before,m.tokens.weight))


if __name__=='__main__':unittest.main()
