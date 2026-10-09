# 统一原生工具协议的 SFT

不要直接把 JSON 动作模型拿来与原生工具 RL 做严格归因。先构造同协议的 SFT。`answers/native_sft.py` 已提供转换器：每个目标回合用 tokenizer 的原生工具模板生成序列，并显式保存目标 token IDs。

运行：

```bash
python -m answers.native_sft --model models/qwen05
python -m src.hf_train --model models/qwen05 --data data/train_native_sft.jsonl --steps 100 --global-batch 16 --out runs/native-sft
python -m src.export_model --run runs/native-sft --out models/native-sft
```

输出包含 input_ids/labels，避免 JSON function call 被当成普通字符串。转换器要求模板的 prompt token 序列是完整回合序列的前缀；不成立时显式报错，不静默裁剪。换模板后应重新验证。

然后把 COURSE_TRAIN_MODEL 指向导出的 native-sft 模型，作为 verl 初始 checkpoint。这样可以在同一原生工具 harness 下比较 Base、SFT、SFT+RL。
