import unittest
try:
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    HAS_HF=True
except ImportError:
    HAS_HF=False


@unittest.skipUnless(HAS_HF,'transformers/torch not installed')
class HFTests(unittest.TestCase):
    def test_mask_and_gradient(self):
        from src.hf_train import collate,example_mean_loss
        config=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
        m=Qwen2ForCausalLM(config)
        ids,mask,labels=collate([([1,2,3,4],[-100,-100,3,4]),([2,3],[-100,3])],0,'cpu')
        self.assertEqual(labels[1,-1].item(),-100)
        self.assertEqual(mask[1,-1].item(),0)
        loss=example_mean_loss(m(input_ids=ids,attention_mask=mask).logits,labels)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(any(p.grad is not None for p in m.parameters()))
    def test_zero_targets_rejected(self):
        from src.hf_train import example_mean_loss
        with self.assertRaises(ValueError):example_mean_loss(torch.randn(1,4,10),torch.full((1,4),-100))


if __name__=='__main__':unittest.main()
