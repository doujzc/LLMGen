# Top1 training data

实际数据不提交到仓库。默认文件名为 `train.jsonl`；验证集是可选的，可通过
`TOP1_VALIDATION_DATA` 指定。

每行格式：

```json
{"messages":[{"role":"user","content":"查询贵州茅台"}],"target_candidate_name":"StockQuery"}
```

候选名必须存在于 `configs/top1_candidates.json`。训练启动后会在输出目录生成经过
实际 tokenizer 长度裁剪的 `sft_input.jsonl`。
