# 导出与恢复 verl Checkpoint

先分清两个用途。恢复训练使用原始分布式 checkpoint 和 optimizer 状态；推理评估使用合并后的 Hugging Face 权重。不要为了部署而覆盖训练 checkpoint。

在已锁定的 verl 环境里执行：

```bash
python -m verl.model_merger merge --help
```

若当前版本提供该命令，按照打印的参数执行 FSDP 合并；常见入口是：

```bash
python -m verl.model_merger merge --backend fsdp --local_dir runs/你的实验/global_step_20/actor --target_dir models/native-rl
```

将“你的实验”换为实际 trainer.default_local_dir，先列出实际保存目录，不凭空假设 step 已存在。如果锁定版本将导出改为保存时 hf_model，使用解析出的 checkpoint.save_contents 配置并保留原始分片。CLI 帮助和保存日志是该版本的权威接口。

导出目录必须含 config、tokenizer/chat template 与模型权重。用 vLLM 加载后，对同一 dev/test 文件运行 src.native_agent。恢复则使用相同 run_id 和目录，确认 trainer.resume_mode 的实际值与恢复日志；看到 global_step 继续增加才算完成。
