# 常见错误的排查顺序

| 症状 | 先检查什么 | 不要先做什么 |
| --- | --- | --- |
| No module named src | 当前目录与 python -m 调用方式 | 修改全局 Python 路径掩盖错误 |
| loss 不下降 | 标签错位、mask、梯度、学习率 | 直接扩大模型与卡数 |
| loss 很低但生成差 | 因果泄漏、验证数据、采样与模板 | 宣称预训练成功 |
| OOM 在加载模型时 | 权重精度、完整副本、主机内存 | 只减 max_new_tokens |
| OOM 在生成时 | KV cache、上下文、并发 | 只减少优化器状态 |
| DDP 多卡没有加速 | 有效 batch、通信、计时窗口、数据供给 | 比较不同计算预算后直接归因 |
| 所有 rank 等待 | 最早失败的 rank、地址端口、进程数 | 只看最后一条 timeout |
| 多节点 NCCL 卡住 | 先跑 collective probe，再查分配的网卡与互联 | 照抄别的集群的网卡名称 |
| model heads 不整除 TP | model_info.py 与模型配置 | 根据 GPU 数强行设置 TP |
| JSON 可解析但任务失败 | 工具参数、数据、最终状态 verifier | 给格式正确就打满分 |
| RL reward 长期为零 | 抽查 rollout、模板、工具协议、reward | 盲目增加训练轮数 |
| 全组 reward 相同 | advantage 是否为零、任务难度与采样 | 把无梯度更新当训练稳定 |
| 恢复后混入旧模型结果 | 是否复用了旧任务 DB | 把缓存结果作为新评测 |
| verl unknown config key | configs/verl.commit 与解析配置 | 使用 ++ 静默创建无效字段 |

提交错误时提供完整命令、最早相关 traceback、环境版本、卡数与节点数、已经尝试过的改动。去掉凭证和公司敏感数据。
