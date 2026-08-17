# Top1 training data

实际数据不提交到仓库。默认文件名为 `train.jsonl`；验证集是可选的，可通过
`TOP1_VALIDATION_DATA` 指定。

每行格式：

```json
{"messages":[{"role":"user","content":"查询贵州茅台"}],"target_candidate_name":"StockQuery"}
```

候选名必须存在于 `configs/top1_candidates.json`。训练启动后会在输出目录生成经过
实际 tokenizer 长度裁剪的 `prepared/train.sft.jsonl`，验证集对应
`prepared/validation.sft.jsonl`；相邻的 `*_profile.json` 只保存统计诊断，不复制正文。

默认 memorization 数据为 `top1_labeldesc_paper_v1.jsonl`，来源文件是 PromptGen 的
同名 42 行 LabelDesc 数据。它仍属于训练数据且被 `.gitignore` 忽略；向 NVIDIA 服务器
同步仓库时需要与 `train.jsonl` 一样单独传输。训练会校验稳定 `id`、`source_type`、
`description_type`、单条 user message、候选名及全候选覆盖。
