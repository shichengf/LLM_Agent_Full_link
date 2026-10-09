# 从 Python 到多节点 Agent 训练与部署

这是可以按顺序操作的课程，不是周计划。每课给出讲解、已有完整源码、运行步骤、结果判断和练习答案。完成前一课的验收再进入下一课。所有命令默认在本课程根目录执行。前十课可以在个人电脑完成；GPU 课必须在公司分配给你的计算节点上运行。

你有研究背景，但编程手感需要恢复，所以这里不假设你熟悉 Python 工程、PyTorch 或分布式命令。第一次读到不认识的语法，先在解释器中运行一小段，再回到完整程序。源码不是黑盒：每课会指定需要阅读的函数。

资源阶梯为 CPU、单卡、单节点八卡、两节点十六卡，以及三到四节点二十四至三十二卡。卡数提高与模型、上下文、环境并发和训练时长分别做对照。多节点不是最后一节附录，而是训练、推理、Agent 并发和 RL 中的独立实验。

## 开始之前

克隆仓库后进入 `LLM_Agent_Full_link`（旧 ZIP 解压后的目录名为 `agent_systems_course`）。先打开本文件或 `课程讲义.html`。源码在 `src/`，多节点命令在 `cluster/`，练习参考答案在 `answers/`。第 01 课直接开始，不需要 GPU，也不需要安装第三方包。

希望先建立直觉，可以先用浏览器打开 [attention 交互实验](visuals/attention.html)。它只演示单个注意力头的一行权重，不需要 Python 或 GPU；页面包含操作、解释和答案。初次阅读跳过页面末尾的公式即可。

```bash
python3 --version
python3 -m src.basics
python3 -m unittest discover -s tests -v
```

未安装 PyTorch 时，TorchTests 会显示 skipped。这是明确的缺少依赖，不是通过。课程作者的实际验证范围见 `VALIDATION.md`。GPU、NCCL 和公司调度器部分必须在你的环境进行验收；课程不声称已经连接或验证了公司集群。

### 只配置一次的集群信息

第 16 课之前复制 `configs/cluster.env.example` 为 `configs/cluster.env`，填写共享目录、Python 环境、节点私网地址和模型目录。课程同时提供 Slurm 与手动启动方式：Slurm 用户选择 sbatch；其他调度器用户在已分配节点内使用手动命令。已有公司镜像优先复用，不在节点上更换系统驱动。

`COURSE_MODEL` 是多节点推理实验的模型，示例为 Qwen2.5-72B-Instruct；`COURSE_TRAIN_MODEL` 是训练模型，示例为 Qwen2.5-7B-Instruct。早期练习使用 Qwen2.5-0.5B-Instruct。已有经过批准的本地权重可以替换，先运行第 11 课的配置检查。所有节点必须看到相同的路径与权重版本。

72B 推理实验不是在强调必须使用大模型：它提供跨节点 TP/PP 的具体载体。7B 的多节点训练与并行 rollout 同样是完整系统实验。资源或主机内存不足时减小模型，保留实验结构。

## 第 01 课 用 Python 计算成功率

Python 列表保存一组对象，字典按键保存字段，函数把输入转成输出。`True` 是布尔值，`"True"` 是字符串。统计任务成功率需要真正的布尔值，否则字符串 `"False"` 也可能被当成真。`None` 表示没有结果；空评测集的成功率应该是 None，而不是 0%。

打开 `src/basics.py` 和 `src/common.py`，阅读 `summarize`。先跟踪三个样本如何得到 2/3，再解释函数为何首先判断空输入。

```bash
python3 -m src.basics
python3 -m src.basics --output runs/my_first_result.jsonl
```

预期终端显示 `n` 为 3、`success_rate` 约为 0.6667。文件是结果，不是程序；修改结果文件不会改变下一次运行。

练习：给默认列表增加一条成功记录，先写下你预测的成功率。然后把其中一个 success 改成字符串，读完整 traceback，指出最底部异常类型和报错位置。参考答案：四条中三条成功，成功率 0.75；字符串应触发 ValueError。恢复代码后再继续。

验收：新建一个文件，不复制原函数，重新实现成功率统计。能解释 `len`、`sum`、字典索引、函数返回值，即可进入下一课。

## 第 02 课 JSONL 文件与错误定位

JSON 是文本格式，Python 字典是内存对象，两者通过序列化转换。JSONL 每行是一条 JSON，很适合逐条记录轨迹。一个 100 行的文件可能只有第 73 行坏了；错误信息必须包含文件名和行号，而不是一律返回空列表。

```bash
python3 -m src.tasks --n 16 --split train
python3 -m src.tasks --n 16 --split dev
python3 -m src.agent --backend oracle --tasks data/dev.jsonl --out runs/oracle16.jsonl
python3 -m src.basics --input runs/oracle16.jsonl
```

`oracle` 是课程提供的确定性正确策略，用来测试系统管道；它不是模型的成绩。预期 16 个任务全部成功。阅读 `read_jsonl` 的 enumerate：行号从 1 开始，便于你在编辑器定位。

练习：复制一份任务文件，删掉某一行末尾的右花括号再读取。然后观察 `write_jsonl` 的临时文件与 `os.replace`：它减少写入中途留下半个结果文件的风险。参考答案：解析器应指出损坏行；原子替换不等于跨机器事务，也不保证外部副作用只执行一次。

验收：能从磁盘读取任务、运行程序、写出结果，并根据行号修复坏数据。

## 第 03 课 命令行 环境与 Git

终端的当前目录决定相对路径指向哪里。`python -m src.agent` 表示把模块作为程序运行；直接运行 `python src/agent.py` 可能破坏相对导入。虚拟环境隔离 Python 依赖，不会自动安装或隔离 GPU 驱动。

```bash
pwd
python3 -m src.agent --help
python3 -m src.preflight > runs/local_environment.json
git status
git switch -c practice/lesson-03
git diff
```

课程仓库已经初始化；上面的命令建立你自己的练习分支，不会上传任何内容。修改代码后，用 `git add 文件名` 暂存、`git commit -m "Complete lesson 03"` 保存。若 Git 未配置身份，按公司要求配置后提交；不要把 API key、公司数据或模型权重加进 Git。课程只使用自生成数据。阅读 argparse：每个参数有名称、类型与默认值，命令行是实验配置的一部分。

练习：把 `--workers 4` 改成 `--workers four`。参考答案：参数类型检查应在运行任务前报错。再执行 `git diff`，理解它展示的是尚未提交的内容。

验收：关闭终端重新打开后，仍能进入目录并运行第 02 课，能说明代码、数据、环境和结果分别在哪里。

## 第 04 课 Tensor 与形状

进入安装了 PyTorch 的环境。没有现成环境时，在允许安装依赖的节点运行下方脚本。它创建专用环境，不修改系统 Python。

```bash
bash tools/bootstrap_train.sh
source .venv-train/bin/activate
python -m src.torch_basics --mode tensor
```

tensor 是带形状和 dtype 的数组。`[2,3,4]` 可以理解为两个样本，每个样本三个位置，每个位置四个特征。对最后一维求平均得到 `[2,3]`；加一个 `[4]` 向量，broadcasting 会让每个位置使用同一个向量。`device` 决定它在 CPU 还是 GPU。

阅读 `torch_basics.py`，在每次 reshape 和 mean 前先预测形状。把 `torch.ones(4)` 改成 `torch.ones(5)`，观察维度不匹配。

预期梯度例子输出 loss=9、gradient=-18。先不要背 autograd，下一课手工验证这个数。验收：能区分 shape、dtype、device，并主动打印它们排查错误。

## 第 05 课 反向传播和优化器

例子中预测为 3w，目标为 9，loss=(3w-9)^2。导数是 6(3w-9)，在 w=2 时等于 -18。梯度为负，梯度下降会增加 w。`backward()` 累计梯度，所以每个更新开始前通常调用 `zero_grad()`。

```bash
python -m src.torch_basics --mode regression
```

程序学习 y=3x+2。阅读初始化、前向、loss、清梯度、反向、step 六个步骤。预期 weight 接近 3、bias 接近 2，loss 低于 1e-6。

练习：暂时注释 `optimizer.step()`；再恢复并把 lr 改成 10。参考答案：前者参数不会更新，后者可能震荡或发散。不要把所有不收敛都归因于模型结构。记录第一次出现异常的步骤以及当时的 loss。

验收：能不用照抄写出一个线性回归训练循环，并手工解释一次参数更新方向。

## 第 06 课 从线性层到语言模型接口

语言模型输入是整数 token IDs `[B,T]`，embedding 把它映射到 `[B,T,C]`，最后输出每个位置对词表的 logits `[B,T,V]`。logits 不是概率，softmax 才得到归一化概率。训练预测下一个 token，所以输入与标签错开一位。

阅读 `TinyLM.forward` 和 `batch`。`batch` 从长文本取一段 x，再从下一个位置取同长度 y。位置 embedding 为序列顺序提供信息，残差连接让每层在已有表示上修正。

```bash
python -m src.tiny_lm --device cpu --steps 2 --batch 2 --context 16 --out runs/tiny-shapes
```

这一步只验证前向、反向、存盘，不判断语言能力。加入打印，确认 ids、embedding 和 logits 的形状；打印后移除，避免真实训练日志过大。

练习：如果 V=100、B=2、T=16，logits 有多少个数？参考答案：3200。为什么 labels 不需要 one-hot？交叉熵接口接收正确类别的整数索引。

验收：能解释一个输入 token 到 loss 的完整路径。

## 第 07 课 手写 Attention 与因果性

注意力计算先把表示投影为 Q、K、V。Q 与 K 的点积形成每个位置对其他位置的分数，除以 head_dim 的平方根后 softmax，再加权汇总 V。多头把特征拆成多个子空间，最后拼接。

阅读 `Attention.forward`：Q/K/V 的形状从 `[B,T,C]` 变为 `[B,H,T,D]`，分数是 `[B,H,T,T]`。因果 mask 把未来位置分数设为负无穷，使其 softmax 权重为零。

```bash
python -m src.torch_basics --mode causal
```

程序修改序列后半段，比较前半段 logits。因果模型差异应接近零；不加 mask 时通常非零。这个测试比“loss 看起来正常”更能发现标签泄漏。

练习：运行 `python -m src.tiny_lm --no-causal --steps 100 --out runs/leaky`，观察 loss。参考答案：能偷看未来的模型可能更容易降低训练 loss，但它违反自回归生成条件，不能据此宣称模型更好。

验收：能从张量索引解释为什么使用上三角 mask，能识别错误方向的 mask。

## 第 08 课 RoPE 与 KV Cache

RoPE 对 Q/K 的成对维度旋转，旋转角随位置变化。课程用相邻维成对约定，和某些框架的维度排列不同，但内部保持一致。KV cache 保存过去的 K/V，生成新 token 时只计算新位置的表示，再查询已有缓存。

```bash
python -m src.attention_internals
```

阅读 `rope`、`attend` 和逐位置循环。程序比较一次处理整段与逐 token 缓存处理的输出。最大误差应在 1e-5 附近的容差内，旋转前后向量范数也应近似相等。

练习：把逐 token 的 RoPE offset 固定成 0，再运行。参考答案：缓存路径的位置错误，应破坏等价性。KV cache 节省重复计算，但占用随上下文和并发增加；它不是零成本优化。

验收：能解释 prefill 处理已有输入、decode 逐步生成的区别，并指出 cache 中存的是哪些张量。

## 第 09 课 完成一次小模型预训练

现在运行完整小模型。数据为课程自生成结构化文本，不是自然语言能力基准。训练和验证按原始记录划分；验证 loss 用固定抽样，便于跨步骤比较。

```bash
python -m src.tiny_lm --steps 200 --out runs/tiny-main
```

有 GPU 时脚本默认使用 GPU。CPU 较慢可减小 width、batch 和 steps。阅读 train loss、val loss、tokens/s。不要把刚开始编译、初始化或很短窗口的速度当稳定吞吐。

练习：只改变 width=64 到 128，其余不变，各运行相同步数。比较参数量、训练速度、loss 和实际 token 数。如果增加步数，再单独记录额外计算预算。

预期是能完成训练与生成、loss 有学习迹象；不承诺某个固定 loss 或流畅对话。训练未改善时检查标签、学习率和因果性。验收：保存两组结果，解释你改变了什么、什么保持不变。

## 第 10 课 保存恢复与可复现

checkpoint 不只是模型权重。要继续同一优化过程，还需要 optimizer 状态、数据进度与随机状态。课程小模型保存采样 generator，恢复时继续取同一序列的训练 batch。不同硬件上的浮点运算仍可能有差异。

```bash
python -m src.tiny_lm --device cpu --steps 20 --out runs/resume-demo
python -m src.tiny_lm --device cpu --steps 40 --resume --out runs/resume-demo
python -m src.tiny_lm --device cpu --steps 40 --out runs/uninterrupted
```

`--steps 40` 是总步数，不是再训练 40 步。比较两次最终权重，用 `answers/check_resume.py`，应在同环境下相等或误差很小。

```bash
python answers/check_resume.py runs/resume-demo/checkpoint.pt runs/uninterrupted/checkpoint.pt
```

练习：恢复时改 context，确认程序拒绝不兼容 checkpoint。参考答案：模型维度变化不是普通继续训练，不能静默忽略不匹配键。

验收：能从中断点继续，并解释为什么只加载权重不等于精确恢复。

## 第 11 课 开源模型与 Chat Template

从模型仓库获取的是权重、配置与 tokenizer。聊天模板把 role 和内容编码成模型熟悉的格式；模板错误可能让“模型退化”看起来像训练失败。

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/qwen05'))"
python tools/model_info.py models/qwen05 --tp 1
python answers/inspect_tokens.py models/qwen05
```

下载需遵守公司模型访问规则；已有本地权重时直接使用其路径。`inspect_tokens.py` 打印真实模板文本和 token IDs，不打印账户凭证。把 messages 中 role 从 user 改成 assistant，对照生成起始位置。

大模型并行前运行 `model_info.py MODEL --tp N`。head 数不整除普通 TP 时脚本直接报错；KV heads 少于 TP 时还涉及后端的复制支持，不能只看总显存。

验收：知道模型 revision、chat template、EOS、pad token 的作用，并能定位一次模板差异。

## 第 12 课 SFT 与 Loss Mask

SFT 让模型学习给定上下文下的目标回复。课程把每个 assistant 回合单独构造成一个样本：前面的消息是条件，当前回复是目标。把 prompt labels 设成 -100，交叉熵会忽略它们。padding 同样不应计入 loss。

```bash
python -m src.tasks --n 1024 --split train
python -m src.tasks --n 128 --split dev
python -m src.hf_train --model models/qwen05 --steps 100 --global-batch 16 --out runs/sft05
python -m src.export_model --run runs/sft05 --out models/sft05
```

阅读 `encode_row`、`collate`、`example_mean_loss`。课程使用每个样本内部的 token 平均，再对样本平均。它与对整个 batch 所有 token 平均不同，变长任务会受不同权重。模型输出和标签内部再错位一次，因此不要提前重复错位。

练习：用 `answers/inspect_sft.py` 查看监督位置。参考答案：prompt 为 -100，response 和 EOS 有监督，padding 为 -100；每个样本至少有一个有效监督位置。

验收：训练与导出完成，并能明确说明哪些 token 在学习。成功率要等部署后测，不能只凭 SFT loss 下结论。

## 第 13 课 LoRA 与模型导出

LoRA 在指定线性层增加低秩可训练更新，冻结大部分原权重。它减少可训练参数与优化器状态，但不会消除基础模型和激活显存。课程示例只作用于 q_proj、v_proj，便于观察；这不是所有任务的最佳选择。

```bash
python -m src.hf_train --model models/qwen05 --lora --steps 100 --global-batch 16 --out runs/lora05
python -m src.export_model --run runs/lora05 --out models/lora05-merged
```

阅读 peft 包装与 merge_and_unload。导出脚本重建模型、加载参数，再把适配器合并为供推理使用的权重。CPU 导出需要足够内存，不能把 72B 权重在小内存登录节点展开。

练习：统计 requires_grad=True 的参数，与全参数训练比较。参考答案：LoRA 的训练参数明显更少；要同时比较显存、速度和下游成功率，不能据此断言效果相同。

验收：得到可以被模型服务重新加载的导出目录，保留原始 checkpoint 供恢复。

## 第 14 课 持续预训练与 Mid Training

持续预训练对文本继续做 next-token 学习，SFT 对指定回复监督，两者的数据与 mask 不同。mid-training 在不同团队中的定义不同；实践中应写清训练目标和数据分配，而不是仅给阶段贴标签。

```bash
python answers/make_cpt.py
python -m src.hf_train --model models/qwen05 --data data/cpt.jsonl --steps 50 --global-batch 16 --out runs/cpt05
python -m src.export_model --run runs/cpt05 --out models/cpt05
```

`make_cpt.py` 生成解释统计操作的自有文本。`encode_row` 遇到 text 字段时监督整段 token。不要把私有答案数据直接混进泛化测试集。

练习：比较原始模型、CPT 后模型和 SFT 后模型在同一 dev 集的表现；运行命令见第 23 课。参考答案：领域文本 loss 降低不保证工具调用能力提高，可能还需专门的交互训练。

验收：能区分“知识数据变化”和“动作轨迹监督”，并说明如何检查能力退化。

## 第 15 课 显存 时间和 OOM

训练显存由参数、梯度、优化器状态、激活与临时 buffer 构成；推理另有 KV cache。模型权重可以放下，不代表训练能放下。课程全参数训练以 FP32 参数配合 BF16 autocast，和纯 BF16 参数训练的内存开销不同。

```bash
python -m src.hf_train --model models/qwen05 --steps 20 --micro-batch 1 --global-batch 16 --out runs/memory-a
python -m src.hf_train --model models/qwen05 --steps 20 --micro-batch 2 --global-batch 16 --grad-checkpoint --out runs/memory-b
```

这里一次改变两项只用于探索；正式对照需把 micro-batch 与 checkpointing 分开做四格实验。阅读 metrics 中 step_s、tokens_s、peak_GB_max_rank。计时窗口包含训练步骤，不包含 checkpoint；完整成本还要加初始化与保存时间。

运行 `python -m src.profile_tiny --out runs/profile.json`，用 torch.profiler 导出时间线；阅读终端算子耗时表。trace 可在支持 Chrome trace 的离线性能查看器打开。观察 attention 矩阵乘法、softmax、反向和 optimizer；profiling 会增加开销，不能拿开启 profiler 的速度与关闭时直接比较。

练习：逐级增加 micro-batch，接近边界即停止，不必故意耗尽共享机器。若 OOM，先记录哪个阶段失败，再减小微批或序列长度。不要一上来关闭所有优化器状态或改变训练目标。

验收：能给出一张“配置—显存—吞吐”表，并区分计算瓶颈与内存容量限制。

## 第 16 课 单节点 DDP

DDP 每张卡保留完整模型，各卡处理不同样本，通过梯度通信保持参数一致。它主要扩大数据处理能力，不直接解决单卡放不下模型的问题。rank 是全局进程编号，local_rank 是当前节点的设备编号。

```bash
torchrun --standalone --nproc-per-node=2 -m src.distributed_probe --mb 16
torchrun --standalone --nproc-per-node=2 -m src.hf_train --model models/qwen05 --mode ddp --steps 50 --global-batch 32 --out runs/ddp2
torchrun --standalone --nproc-per-node=8 -m src.hf_train --model models/qwen05 --mode ddp --steps 50 --global-batch 32 --out runs/ddp8
```

global_batch = world_size × micro_batch × accumulation_steps。脚本要求整除，并用确定性的全局样本索引按 rank 切分。不能把每卡 batch 不变造成的总 batch 变大误解为完全相同的训练。

练习：在 global_batch=32 下比较 2 卡和 8 卡；再把 8 卡的 global_batch 改成 128，明确这是另一种实验。参考答案：前者接近强扩展对照，后者改变了优化更新的样本规模。

验收：world_size 正确、没有重复占用 GPU、所有 rank 完成同样步数。

## 第 17 课 FSDP 与分片

FSDP 把参数、梯度和优化器状态分散到多个 rank，需要时收集参数参与计算，再进行必要的同步。它降低每卡常驻状态，但增加通信。课程源码使用稳定易读的 FSDP1 API；FSDP2 是后续迁移练习，不把两者接口混在一起。

```bash
torchrun --standalone --nproc-per-node=8 -m src.hf_train --model models/qwen05 --mode fsdp --steps 30 --global-batch 32 --out runs/fsdp05
source configs/cluster.env
torchrun --standalone --nproc-per-node=8 -m src.hf_train --model "$COURSE_TRAIN_MODEL" --mode fsdp --grad-checkpoint --steps 30 --global-batch 32 --out runs/fsdp7b
```

先用小模型确认分片代码，再使用 7B。当前教学加载器每个进程先在 CPU 建立全模型；八进程的主机内存需求会很大。观察主机 RSS，不能只看显存。生产系统的 meta 初始化与分布式加载可以优化此处。

课程保存 full checkpoint 到 rank0，方便理解与导出，但大规模会出现主机内存与 I/O 瓶颈。第 30 课用真实 RL 框架的分布式 checkpoint 对照。

验收：记录 FSDP 相对 DDP 的显存变化与通信代价，并成功保存恢复一次。

## 第 18 课 节点间通信与慢进程

跨节点前先验证 all-reduce。每个 rank 把张量设成 1，求和结果必须等于 world_size。通信连通性检查应该早于大模型加载，否则一次启动会把网络错误藏在大量日志中。

```bash
cp configs/cluster.env.example configs/cluster.env
```

编辑配置。手动方式在节点 0 运行下面第一条，在节点 1 运行第二条；两边保持进程运行，命令可在两个终端分别启动。

```bash
COURSE_NODE_RANK=0 bash cluster/torchrun_node.sh src.distributed_probe --mb 64
COURSE_NODE_RANK=1 bash cluster/torchrun_node.sh src.distributed_probe --mb 64
```

Slurm 用户改用下方命令，按公司实际 partition/account 追加参数。

```bash
mkdir -p runs
sbatch --nodes=2 cluster/train.sbatch src.distributed_probe --mb 64
```

预期 16 个唯一 rank、正确求和、无超时。将 --mb 分别设为 1、64、256，比较延迟与吞吐。再加 `--straggler-ms 200`，观察一个慢 rank 如何拖慢整体。payload_GB_s 只是数据量/时间，不能当作 NCCL bus bandwidth。

验收：能根据主机名、rank、设备与 timeout 判断问题位置；不要未经依据把所有 NCCL 参数改一遍。

## 第 19 课 两到四节点训练

现在同一代码跨节点训练。所有节点必须共享源码、模型和输出目录，并使用相同依赖。手动启动时只改变 COURSE_NODE_RANK；Slurm 脚本自动使用 SLURM_PROCID。每节点一个 torchrun，再派生八个进程，不是先启动八个 launcher。

```bash
source configs/cluster.env
sbatch --nodes=2 cluster/train.sbatch src.hf_train --model "$COURSE_TRAIN_MODEL" --mode fsdp --grad-checkpoint --steps 60 --global-batch 96 --out runs/fsdp16
sbatch --nodes=3 cluster/train.sbatch src.hf_train --model "$COURSE_TRAIN_MODEL" --mode fsdp --grad-checkpoint --steps 60 --global-batch 96 --out runs/fsdp24
sbatch --nodes=4 cluster/train.sbatch src.hf_train --model "$COURSE_TRAIN_MODEL" --mode fsdp --grad-checkpoint --steps 60 --global-batch 96 --out runs/fsdp32
```

96 能被 8、16、24、32 整除。先完成一个作业再启动下一组，不默认同时占用全部节点。排除头几个预热步骤，使用相同统计窗口计算速度；保存期间的暂停单独记录。

强扩展效率用 E_N = throughput_N / (throughput_8 × N/8)。增加节点后效率下降并不自动表示错误，可能是单次更新计算量太小、跨节点通信或存储竞争。

练习：同样资源下做 global_batch 随卡数增加的实验，明确它改变了样本/更新。验收：至少完成单节点和双节点；三四节点可用后执行相同命令，并解释实际瓶颈。

## 第 20 课 vLLM 单卡服务

推理服务负责接收请求、调度生成和返回结果。训练环境与 serving 环境分开，避免推理框架替换训练所需的 PyTorch 版本。首次安装后保存 freeze，之后同一次对照实验不升级依赖。

```bash
bash tools/bootstrap_serving.sh vllm
source .venv-vllm/bin/activate
vllm serve models/sft05 --served-model-name course-model --host 127.0.0.1 --port 8000 --max-model-len 4096
```

保持服务运行，在另一个终端进入课程目录，发起真实 Agent 请求。

```bash
python3 -m src.agent --backend http --tasks data/dev.jsonl --workers 4 --out runs/sft05-agent.jsonl
python3 -m src.report runs/sft05-agent.jsonl
```

错误是证据：HTTP 404 先查 URL 和 model 名称；连接拒绝先看服务是否就绪；服务启动 OOM 与请求中 OOM 需要分别检查权重和 KV cache。

验收：保存一条完整 trace，解释请求经过模型服务后怎样回到工具执行循环。

## 第 21 课 SGLang 与公平压测

关闭自己启动的 vLLM 服务后，在独立环境部署同一模型。不要杀死节点上其他人的进程。

```bash
bash tools/bootstrap_serving.sh sglang
source .venv-sglang/bin/activate
python -m sglang.launch_server --model-path models/sft05 --served-model-name course-model --host 127.0.0.1 --port 8000 --context-length 4096
```

另一个终端运行压测；对两个后端各做一遍。

```bash
python3 -m src.bench_serving --concurrency 1 --requests 32 --out runs/c1.jsonl
python3 -m src.bench_serving --concurrency 8 --requests 128 --out runs/c8.jsonl
python3 -m src.bench_serving --concurrency 32 --requests 256 --out runs/c32.jsonl
```

TTFT 从请求发出到首个内容块，latency 到整个响应结束。SSE 一个内容块可能包含多个 token，所以脚本 TPOT 是基于总 token 数的近似平均，不是逐 token 延迟测量。当前压测是 closed-loop 固定并发，不能等同固定到达率的线上流量。

练习：比较重复 prompt 与不同 prompt，以及短输入和长输入。记录实际 prompt/output token 数、失败率、缓存状态和模型版本。验收：不能仅拿最快的一个吞吐数宣称哪个框架更好。

## 第 22 课 多节点推理与 TP PP

TP 在同一层内切分张量，PP 在层之间切分阶段，服务副本则复制模型处理不同请求。对于单节点能放下的模型，增加副本有时更合适；跨节点切分并非默认更快。

先将推理环境同步到所有已分配节点，激活相同环境。设置 COURSE_NNODES=2、GPUS_PER_NODE=8、推理模型为头数兼容的 72B。每个节点的 COURSE_NODE_IP 是自己的私网 IP。

```bash
# 节点 0
COURSE_NODE_RANK=0 COURSE_NODE_IP=节点0私网IP bash cluster/ray_node.sh
# 节点 1
COURSE_NODE_RANK=1 COURSE_NODE_IP=节点1私网IP bash cluster/ray_node.sh
# 节点 0 另一个终端
source configs/cluster.env
ray status --address="$COURSE_MASTER_ADDR:$COURSE_RAY_PORT"
bash cluster/vllm_head.sh
```

中文 IP 占位符只在本课首次配置时替换；其余参数由配置读取。预期 Ray 看到 16 GPU，vLLM 使用 TP8×PP2。再从可访问节点对私网 endpoint 执行第 21 课压测。

SGLang 原生多节点实验不需要复用这组 Ray 服务。在其独立环境每节点运行 `COURSE_NODE_RANK=N COURSE_NODE_IP=本机私网IP bash cluster/sglang_node.sh`，课程使用 TP16。检查模型 head 划分、互联和上下文限制。

验收：将单节点、跨节点 TP/PP 和多副本的资源与瓶颈区分清楚，记录权重加载、预热与稳定请求阶段。只在自己专属的作业资源内启停 Ray，控制面不开放到公网。

## 第 23 课 手写 Agent Harness

Agent 的核心循环是：给模型上下文，解析动作，调用工具，加入观察，继续生成或结束。模型服务并不替你管理任务状态。课程的 JSON 动作协议刻意简单，便于观察控制流；它与后面的原生 function calling 是两种不同协议。

```bash
python3 -m src.agent --backend oracle --tasks data/dev.jsonl --out runs/oracle.jsonl
python3 -m src.agent --backend bad --tasks data/dev.jsonl --out runs/bad.jsonl
python3 -m src.agent --backend oracle --tasks data/dev.jsonl --fail-tool --out runs/timeout.jsonl
```

阅读 `run_task`，跟踪 lookup、calculate、final 三轮。真实值不在 prompt 中，lookup 返回记录，calculate 执行受限统计；没有 eval 或任意 shell。任务正确性由独立 final verifier 判定。

练习：`--max-turns 1` 会怎样？参考答案：oracle 也来不及完成任务，应返回 max_turns，而不是“模型不知道答案”。再对实际服务比较原模型和 SFT 模型，使用不同结果目录。

验收：能把一个失败归到格式、工具、预算、推理或环境，而不是统称模型错误。

## 第 24 课 把 Agent 部署成任务服务

API 服务接收任务，后台 worker 执行，SQLite 保存状态。客户端可以查询 job_id，不必一直保持 HTTP 连接。该教学实现是单控制器，不具备高可用；但提交、幂等标识和持久化路径是真实的。

```bash
export COURSE_GATEWAY_TOKEN=course-local-demo-token
python3 -m src.gateway --backend oracle --delay 0.2
```

另一个终端执行：

```bash
curl -s -H 'Authorization: Bearer course-local-demo-token' -H 'Content-Type: application/json' -d '{"task_id":"dev-0","job_id":"first-job"}' http://127.0.0.1:8080/jobs
curl -s -H 'Authorization: Bearer course-local-demo-token' http://127.0.0.1:8080/jobs/first-job
```

同一个 job_id 重复提交同一个 task 不会新建第二个任务；不同 task 复用该 ID 返回冲突。示例 token 仅用于本地课程，真实内部服务用独立凭证。阅读 worker 和 SQL 更新，理解 queued、running、done、failed 的状态变化。

容器实验：`docker build -f Dockerfile.gateway -t course-gateway .`，再用 `docker run --rm -p 127.0.0.1:8080:8080 -e COURSE_GATEWAY_TOKEN=course-local-demo-token course-gateway`。若公司使用其他容器运行时，使用相同镜像内容与挂载约定，不要求修改主机权限。

验收：能提交、查询和重复提交，并说明服务端进程与模型服务的区别。

## 第 25 课 多节点执行与任务恢复

模型服务可以和 Agent worker 分开部署。worker 多数时间处理网络和环境 I/O，并不是每个 worker 都需要 GPU。把同一任务列表按 shard 划分可避免重复执行；每个 worker 使用独立完成记录。

```bash
python3 -m src.tasks --n 512 --split test
python3 -m src.agent --backend oracle --tasks data/test.jsonl --shards 2 --shard 0 --db runs/shard0.sqlite --out runs/shard0.jsonl
python3 -m src.agent --backend oracle --tasks data/test.jsonl --shards 2 --shard 1 --db runs/shard1.sqlite --out runs/shard1.jsonl
python3 -m tools.merge_results runs/shard0.jsonl runs/shard1.jsonl --out runs/all512.jsonl
```

实际多节点时在不同机器运行 `cluster/agent_worker.sh`，用 COURSE_WORKER_RANK 和 COURSE_WORKERS 指定全局 worker 编号。重复运行同样命令会跳过已完成任务。不同模型或 harness 实验必须用新的 DB，避免把旧结果当新结果。

练习：延迟设为 0.2，中途 Ctrl-C，再恢复。参考答案：完成的任务保留，未完成任务从任务起点重跑；这不是逐 token 恢复。课程工具只读，因此重跑安全；支付、发信等工具必须额外考虑业务幂等。

验收：合并后没有重复 ID，所有预期任务都有终态，能说明恢复的准确粒度。

## 第 26 课 评测与 Reward 不是同一个概念

训练 reward 是用于优化的信号，评测应独立检验目标行为。能输出一个格式正确的 JSON 不代表查到了正确记录；工具调用次数增加也不自动代表能力提高。课程 verifier 用独立实现的数值答案验证 final。

```bash
python3 -m src.tasks --n 256 --split ood
python3 -m src.report runs/oracle.jsonl runs/bad.jsonl runs/timeout.jsonl
python3 -m unittest discover -s tests -p test_core.py -v
```

OOD 集将记录长度增加到 24，检验长度迁移，不能称为跨任务家族泛化。当前四种操作都在训练出现过；跨家族泛化必须另建留出操作，并记录训练中从未出现的规则。

report 输出样本数、成功率和 Wilson 95% 区间。这个区间把任务视为近似独立的伯努利样本；同模板相关性较强时，应采用按任务家族分组的统计方式，不能照搬区间作强结论。

练习：给 reward 输入 {"final":true}、NaN、只有工具结果或多余文本。参考答案：均不得得到正确答案奖励。验收：能从原始轨迹解释至少五个失败，并说明指标的分母。

## 第 27 课 从 Policy Gradient 到 GRPO

policy gradient 增加高回报动作的 log-prob。baseline 帮助降低方差，但不应引入依赖于当前采样动作的错误偏差。group-relative 方法用同任务的多条结果比较；全组回报一样时可能没有可用的相对信号。

```bash
python -m src.torch_basics --mode policy
python -m src.rl_math
```

第一个实验是两动作问题，动作 1 得奖励，训练后其概率应大于 0.9。第二个打印组标准化 advantage 和 clipped objective。ratio = exp(new_logprob-old_logprob)，clip 只是限制目标的一部分，不表示所有梯度都会被强行截断到某范围。

练习：手算 rewards=[0,0,1,1] 的 advantage；再计算 advantage=-1、ratio=0.6、eps=0.2 时的最小项。答案分别为 [-1,-1,1,1] 和 -0.8。

DPO 使用偏好对，PPO/GRPO 使用 rollout 和奖励，它们不是同一个训练管道的不同名字。本课实现可检查的数学核心；真正的 GRPO 工程在第 29–30 课。

验收：能解释 reward、advantage、ratio、KL 和 clipping 各自解决的问题。

## 第 28 课 亲手运行多轮策略更新

`rl_teaching.py` 是可读的多轮 REINFORCE 加 leave-one-out baseline，不冒充生产 GRPO。模型查记录、调用计算工具、给出最终答案，回报来自最终结果。每轮保存实际生成的 token IDs；训练时只对这轮生成部分计算 log-prob，不对工具观察计算 policy loss。

```bash
source .venv-train/bin/activate
python -m src.rl_teaching --model models/sft05 --steps 20 --group 4 --out runs/rl-readable
```

生成使用 temperature=1、top_p=1、top_k=0，使优化的原始分布与采样一致。当前实现没有 KL、PPO clipping 或高速批量推理，所以只用于理解。参数在同一组采样和梯度计算期间保持不变，再更新一次。

练习：在 traces 中找一个奖励为 0 的轨迹，人工核对 reward。若所有组全对或全错，观察 advantages 与 grad_norm，解释为什么更新很少。程序在全组 advantage 为零时跳过 optimizer，避免只发生 AdamW decay。

验收：从一条完整轨迹定位所有被监督的 token，解释为何不能把工具输出也当模型动作。

## 第 29 课 接入真实 verl 训练管道

现在换成真实框架：训练 worker、推理引擎、工具执行与奖励管理协同工作。课程已核对 function-tool 接口与配置，并把源码固定在 `configs/verl.commit` 中的 commit。第三方依赖仍需在公司 GPU 镜像中验收，所以配置检查是实验步骤，不是事后忽略报错。

```bash
bash tools/prepare_verl.sh
source .venv-verl/bin/activate
python -m src.tasks --n 4096 --split train
python -m src.tasks --n 256 --split dev
python -m src.verl_data
python -m unittest discover -s tests -p test_core.py -v
```

阅读 `verl_data.py`：prompt 给模型，ground_truth 给 reward，二者必须分开。阅读 `verl_tools.py`：lookup、calculate 用正式工具 schema；与第 23 课纯 JSON 协议不同。function tools 使用共享只读记录，不包含私有执行状态。

先在一节点八卡、独立 Ray 作业内启动。设置 COURSE_NNODES=1、相应地址和环境；同一作业的所有 Ray 进程都使用 verl 环境。

```bash
COURSE_NODE_RANK=0 COURSE_NODE_IP=本机私网IP bash cluster/ray_node.sh
# 同节点另一个终端
bash cluster/verl_train.sh --cfg job > runs/verl_resolved.yaml
bash cluster/verl_train.sh trainer.total_training_steps=2 trainer.save_freq=1 trainer.test_freq=1
```

`--cfg job` 只解析配置，不启动训练。若某键不存在，停止并核对锁定版本的配置，不使用 + 或 ++ 把未知字段强塞进去。先确认当前接口与课程适配器匹配；错误堆栈和 commit 足以定位具体差异。

验收：完成真实 rollout、reward、update、checkpoint 两步，日志显示模型确实使用工具。单纯 Ray 节点上线不算完成。

## 第 30 课 多节点 Agent RL

两节点各八卡先建立独立 Ray 集群，方法与第 22 课相同，但必须使用 verl 环境，不能连接旧 vLLM 作业。把 COURSE_NNODES 改为 2，确认 Ray 资源为 16 GPU。多节点 RL 默认仍是框架的资源协作布局，不等于自动把一个节点只用于 trainer、另一个只用于 rollout。

```bash
source configs/cluster.env
ray status --address="$COURSE_MASTER_ADDR:$COURSE_RAY_PORT"
bash cluster/verl_train.sh --cfg job > runs/verl16_resolved.yaml
bash cluster/verl_train.sh trainer.total_training_steps=20 trainer.save_freq=5
```

预期看到至少三类耗时：生成交互、log-prob/更新、权重同步或保存。GPU 空闲可能是环境等待、长尾轨迹、采样不均、权重同步，也可能是 CPU 数据供给，不能只凭利用率截图判断。

依次比较一节点和两节点，保持训练样本、rollout.n、上下文上限与评估一致。三节点与四节点时把 train_batch_size、ppo_mini_batch_size 设为 192，保证与 24/32 卡的基本整除关系；仍需以实际资源组布局检查有效 batch。先在 8/16 卡用同样的 192 重新建立基线。

```bash
bash cluster/verl_train.sh data.train_batch_size=192 actor_rollout_ref.actor.ppo_mini_batch_size=192 trainer.total_training_steps=100
```

重启实验使用相同输出目录，先检查 resume_mode=auto 的解析值与日志恢复步数；不要仅因为程序启动就声称恢复成功。框架 checkpoint 导出见 `answers/verl_export.md`。

验收：完成至少一次双节点真实训练和恢复，报告性能与 reward 的共同变化，不把更大 batch 带来的奖励差异归于通信优化。

## 第 31 课 Scale 压力实验与故障演练

规模实验必须一次只扩大一个维度。准备独立 run_id 与输出目录，记录软件 commit、模型版本、卡数、数据、batch、上下文和工具预算。资源只在已分配作业内使用。

实验 A：第 19 课的 8/16/24/32 卡强扩展。实验 B：第 21 课并发 1/8/32/64 的吞吐与尾延迟。实验 C：第 25 课 512/4096 个任务和 1/2/4 个 worker 进程。实验 D：第 30 课 rollout.n 从 4 到 8，单独记录样本数量与计算预算改变。

```bash
python3 -m src.tasks --n 4096 --split test
python3 -m src.agent --backend oracle --tasks data/test.jsonl --workers 32 --delay 0.05 --db runs/load.sqlite --out runs/load.jsonl
python3 -m src.report runs/load.jsonl
```

这是环境执行压力测试，不是模型能力测试。真实模型负载用 backend=http 重跑，并保留 token 计量。长上下文实验用 `answers/make_long_context.py` 生成额外上下文，先测 tokenizer 长度再调整上限；把 max_response_length 改大并不保证实际轨迹变长。

故障演练：只中断自己的 worker 或作业；记录哪些任务已提交、完成、写盘，再恢复检查缺失与重复。模型训练先正常保存再停止，用 checkpoint 恢复；未持久化部分通常需要重算，不承诺任意位置零损失恢复。

验收：至少写出三个只在规模增加后观察到的问题及证据。允许结果显示“尚未碰到瓶颈”，但要说明测试范围，而不是编造故障。

## 第 32 课 完整交付与研究实验

毕业交付包含两个协议内的对照。JSON 协议比较原模型、JSON SFT 与教学 RL；原生工具协议比较原模型与 verl RL。不要把切换协议的收益混成训练收益。若要做严格的统一 SFT→RL 对照，先用 `answers/native_sft.md` 的转换与 mask 验证步骤建立原生工具 SFT 数据。

在 vLLM 服务原生工具模型时启用对应 parser，再运行专门的原生工具 harness。

```bash
vllm serve 模型导出目录 --served-model-name course-model --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000
# 另一个终端
python3 -m src.native_agent --tasks data/test.jsonl --out runs/native-test.jsonl
python3 -m src.native_agent --tasks data/ood.jsonl --out runs/native-ood.jsonl
python3 -m src.report runs/native-test.jsonl runs/native-ood.jsonl
```

模型实际模板和 parser 必须一致；上面的 hermes 对应课程的工具格式。成功但完全没有调用工具时，抽查任务是否泄漏或答案可猜，不能只看 reward 数字。

交付 `answers/REPORT_TEMPLATE.md` 中的各项：问题、配置、结果、预算、失败案例、统计限制、恢复证据与下一步。保留原始 JSONL、解析后的框架配置、训练曲线和 checkpoint 来源。

最后做独立修改：增加一个统计工具和新的任务家族，修改 schema、环境、verifier、数据与评估；完成单卡调试后，在双节点真实框架上验证。这次只允许 AI 审查与给提示，核心逻辑由你实现。验收不要求刷高某个分数，要求能解释行为、定位问题并复现结论。

## 课程边界与继续深入

这套课程给出从基础代码到多节点真实框架的可执行起点。真实框架实验有完整配置与适配器，但尚需你在公司集群做硬件验收。FSDP2、Megatron、CUDA/Triton、MoE、全异步分离式 trainer/rollout 和跨机房容错是后续专项，不把运行一次脚本称为精通全部系统。

首轮推荐按课号执行，恢复熟练后按验收跳过已掌握内容。每遇到规模问题，先缩小为可重复案例，再回到原规模验证修复。这种往返是课程的一部分。
