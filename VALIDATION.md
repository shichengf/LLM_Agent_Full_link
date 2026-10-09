# 验证记录

本记录区分实际运行、源码核对与尚待集群验证，避免把语法正确误认为多节点可运行。

## 本地实际通过

- 17 项 unittest：数据读写与类型检查、四种任务与 oracle、工具输入边界、异常与超时、SFT 轨迹、reward 解析、advantage 与 clipping、统计区间、HTTP 服务提交与查询、重复任务标识与认证、HF mask 与反向传播、attention 因果性、KV cache/RoPE 等价、优化器更新。
- PyTorch 2.6.0 CPU 小模型训练 40 步：验证 loss 从约 3.205 降到约 0.944。该数字只说明教学程序发生了学习，不代表实用语言能力。
- 20 步保存后恢复到 40 步，与同环境连续 40 步的最大参数差为 0。
- KV cache 逐 token 与整段计算最大差约 3.58e-7；RoPE 范数差约 4.77e-7。
- Transformers 4.51.3 的小型随机初始化 Qwen2 模型前向、masked loss 与反向传播通过。
- 使用真实 Qwen2.5-0.5B-Instruct tokenizer 检查普通 JSON SFT 的 mask，以及 8 个任务、24 条原生工具 SFT 样本的模板前缀一致性与 mask。
- Python 源码编译与 Shell 语法检查通过。

## 源码与配置核对

- 已读取 configs/verl.commit 指定的真实官方源码。
- 课程 verl_train.sh 的 41 个配置键均存在于该快照的生成配置中。
- function_tool 的显式 schema、tool_agent 注册名、Ray 初始化和模型合并入口已核对。
- 已读取模型配置：Qwen2.5-7B 为 28 attention heads、4 KV heads、28 layers；72B 为 64 attention heads、8 KV heads、80 layers。因此普通 TP8/16 不能直接用于课程的 7B 模型。

## 尚未通过硬件运行验证

- 当前编写环境没有 H100 或公司集群，因此没有运行 HF GPU 训练、CUDA/NCCL、FSDP 多卡、vLLM/SGLang GPU 服务、verl RL、Slurm、跨节点网络与故障恢复。
- 尝试的本地两进程 Gloo probe 在本机网络初始化阶段失败，未得到 collective 性能结果。不能把该项列为通过。课程第 18 课要求在实际分配节点重新验收。
- serving 与 verl 的 Python 依赖仍需在公司允许的镜像和驱动组合上安装、通过 pip check 并冻结；锁定源码不自动锁定系统 CUDA 与所有二进制依赖。
- HTML 完成结构检查；未进行真实浏览器截图比对。

实际集群第一次运行按 preflight → collective → 配置解析 → 两步训练 → 短实验 → 扩展实验的顺序进行。某一关失败时保留证据，先解决该关，不继续提交昂贵作业。

## 迁入 GitHub 与网页课程

- 32 课网站数据生成成功；网站 JavaScript 语法检查通过。
- 三份标准 Notebook 的全部代码格在本地 Python 中按顺序执行通过。
- 迁移后 17 项既有 unittest 再次通过（PyTorch 2.6.0 CPU / Transformers 4.51.3）。
- 网站提供浏览器编辑、当前浏览器自动保存、停止运行和下载 Notebook。浏览器 Python 使用固定 Pyodide 0.27.7；仅面向标准库入门练习。
- Chromium 浏览器功能测试通过：32 课导航、真实 Pyodide 执行、编辑后运行、刷新后保留代码、下载修改后的 Notebook、停止无限循环、重新启动 Python、Attention 因果 mask，以及 390px 窄屏无横向页面溢出。
- 浏览器测试把 CDN 请求映射到相同版本官方 Pyodide npm 包的本地文件，验证了浏览器运行逻辑；未把这一结果当作用户网络访问 CDN 的保证。直接读取官方 CDN 的 pyodide.mjs 返回正常。
- jsdom + 实际 Pyodide 的独立逻辑检查也通过。
- Pages 是否上线以 GitHub Actions 的 deploy 结果为准，不以本地 build 成功替代。
