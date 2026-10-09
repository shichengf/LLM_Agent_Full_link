# 官方接口与代码依据

访问与课程编写日期：2026-10-09 UTC。以下链接用于接口核对和进一步阅读，讲义与教学代码为本课程编写。

- Python 教程：https://docs.python.org/3/tutorial/
- PyTorch 基础：https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- PyTorch Transformer 组件：https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
- PyTorch torchrun：https://docs.pytorch.org/docs/stable/elastic/run.html
- vLLM 快速开始：https://docs.vllm.ai/en/stable/getting_started/quickstart/
- vLLM 分布式推理：https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
- SGLang：https://docs.sglang.io/
- verl Agent Loop：https://verl.readthedocs.io/en/latest/advance/agent_loop.html
- verl 工具接口：https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html
- verl 多节点：https://verl.readthedocs.io/en/latest/start/multinode.html
- verl 配置：https://verl.readthedocs.io/en/latest/examples/config.html
- NVIDIA H100：https://www.nvidia.com/en-us/data-center/h100/

课程的 verl 配置与适配器核对了以下不可变源码快照：

https://github.com/verl-project/verl/tree/9e914d5086ab4160ba8d07dac9d2e5c8bcc56851

核对内容包括 function_tool 注册接口、tool_agent 名称、rollout 多轮配置键、主训练入口、Ray 初始化、模型导出入口和依赖声明。源码由 Apache 2.0 许可发布，本课程包不重新分发该项目源码。源码检查不等于 GPU 集群运行成功，实际测试范围见 VALIDATION.md。
