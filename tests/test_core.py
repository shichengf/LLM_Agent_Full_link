import json
import math
import tempfile
import unittest
from pathlib import Path
from src.agent import run_task
from src.common import read_jsonl,summarize,write_jsonl
from src.tasks import make_task,call_tool,verify,sft_examples
from src.report import wilson
from src.rl_math import group_advantages,clipped_objective
from src.verl_reward import compute_score


class CoreTests(unittest.TestCase):
    def test_empty_and_types(self):
        self.assertIsNone(summarize([])['success_rate'])
        with self.assertRaises(ValueError):summarize([{'success':'False'}])
    def test_io(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.jsonl';write_jsonl(p,[{'a':'中文'}]);self.assertEqual(read_jsonl(p),[{'a':'中文'}])
    def test_oracle_all_operations(self):
        for split in ('train','dev','test','ood'):
            for i in range(16):
                t=make_task(i,split);r=run_task(t)
                self.assertTrue(r['success']);self.assertEqual(r['tool_calls'],2)
    def test_no_answer_in_prompt(self):
        t=make_task(0);self.assertNotIn(str(t['values']),t['prompt'])
        self.assertNotEqual(make_task(0,'train')['key'],make_task(0,'test')['key'])
    def test_task_determinism(self):self.assertEqual(make_task(7),make_task(7))
    def test_tools_boundaries(self):
        t=make_task(0)
        for name,args in [('lookup',{'key':'bad'}),('calculate',{'op':'sum','values':[True]}),('calculate',{'op':'sum','values':[float('nan')]}),('shell',{})]:
            with self.assertRaises(ValueError):call_tool(t,name,args)
    def test_failure_modes(self):
        self.assertFalse(run_task(make_task(0),'bad')['success'])
        self.assertIn('TimeoutError',run_task(make_task(0),fail_tool=True)['error'])
        self.assertEqual(run_task(make_task(0),max_turns=1)['error'],'max_turns')
    def test_sft_no_final_target_leak(self):
        rows=sft_examples(make_task(1));self.assertEqual(len(rows),3)
        for r in rows:self.assertNotEqual(r['messages'][-1]['content'],r['response'])
    def test_reward_parser(self):
        self.assertEqual(compute_score('course_records','{"final":12}','12'),1)
        for text in ['{"result":12}','{"final":true}','{"final":12} extra','12','{"final":NaN}']:
            self.assertEqual(compute_score('course_records',text,'12'),0)
    def test_advantages(self):
        self.assertEqual(group_advantages([1,1]),[0,0])
        self.assertEqual(group_advantages([0,0,1,1]),[-1,-1,1,1])
        self.assertAlmostEqual(clipped_objective(1.4,1),1.2)
        self.assertAlmostEqual(clipped_objective(.6,-1),-.8)
    def test_wilson(self):
        lo,hi=wilson(10,10);self.assertLess(lo,1);self.assertAlmostEqual(hi,1)


if __name__=='__main__':unittest.main()
