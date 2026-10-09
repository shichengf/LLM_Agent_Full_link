# 多节点 LLM 与 Agent 实践课程

从 [COURSE.md](COURSE.md) 开始，或下载仓库后用浏览器打开 `课程讲义.html`。讲义包含 32 课正文及主要源码和答案，离线可读。

**交互式入门：** 用浏览器打开 [visuals/attention.html](visuals/attention.html)，先点击 token，看清“只能看到过去和当前位置”，再进入第 06 课。GitHub 文件页面展示源码；克隆后双击 HTML 才能操作。

第一课不需要 GPU 或第三方库：

```bash
git clone https://github.com/shichengf/LLM_Agent_Full_link.git
cd LLM_Agent_Full_link
python3 -m src.basics
python3 -m unittest discover -s tests -v
```

按课号完成，不按日历推进。每课有解释、命令、结果判断、练习和验收；答案在正文及 `answers/`。

后续更新在仓库目录运行 `git pull --ff-only`；有自己的修改时，先提交到个人练习分支。私有仓库克隆需要你已有的 GitHub 登录方式。

课程源文件、练习和集群脚本以本仓库为准；HTML 是阅读版本，ZIP 只用作阶段快照。当前交互内容只有 attention 入门，其余章节仍以文字、代码和实验为主。可视化的讲解标准见 [TEACHING.md](TEACHING.md)。

修改讲义后重新生成 HTML：

```bash
python3 -m venv .venv-book
source .venv-book/bin/activate
python -m pip install Markdown==3.7
python tools/build_book.py
```

| 课号 | 本阶段实际完成的东西 |
| --- | --- |
| 01–05 | Python、文件、命令行、tensor、梯度与优化器 |
| 06–10 | 自己能读懂的 Transformer、RoPE、KV cache、预训练和恢复 |
| 11–15 | 开源 LLM 的模板、SFT、LoRA、持续预训练与显存诊断 |
| 16–19 | DDP、FSDP、节点间通信和 8/16/24/32 卡训练 |
| 20–22 | vLLM、SGLang、压测、TP/PP 与多节点推理 |
| 23–26 | Agent 循环、HTTP 任务服务、并行执行、恢复和评测 |
| 27–30 | 策略梯度、真实轨迹更新、verl 工具适配与多节点 RL |
| 31–32 | 规模压力实验、故障演练与完整研究交付 |

集群信息只填写 `configs/cluster.env`。无需提供账号密码；后续进入实际集群时按公司现有调度与镜像规范使用。训练、vLLM、SGLang、verl 分环境，不在一个环境混装。

`VALIDATION.md` 明确区分本地通过的检查与尚需 GPU/集群实测的部分。课程包不含公司数据、模型权重或第三方项目源码；相关权重和框架按讲义从授权位置获取。
