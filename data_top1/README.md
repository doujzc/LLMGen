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
训练和评测都会把较早消息按时间顺序写入固定 Markdown `## Dialogue Context`，历史
只使用紧凑的 `user:`、`assistant:` 或 `tool:` 行，并把最后一轮 user 消息放入
`## Turn T - Current Customer Utterance`；反斜杠和换行控制字符会进行单行转义。

默认 memorization 数据为 `top1_labeldesc_paper_v1.jsonl`，来源文件是 PromptGen 的
同名 42 行 LabelDesc 数据。它是 `data_top1/` 中唯一取消 `.gitignore` 的数据文件；
其它训练数据仍需单独传输。训练会校验稳定 `id`、`source_type`、`description_type`、
单条 user message、候选名及全候选覆盖。

## IntentChange augmentation

`scripts/build_top1_intent_change.py` 从具有显式结构化标签的 PromptGen train split
构造当前目标覆盖历史目标的多轮样本。生成时排除 dev、test 和所有 reserved audit
cohort；固定覆盖 7 个候选之间的 42 个有向切换组合。默认每个组合 10 条，共 420 条，
每个来源候选和目标候选各 60 条。最后一轮直接使用目标意图的自然 query，不添加
“换个问题”“等一下”或“不用了”等显式切换提示。

```bash
uv run --no-sync python scripts/build_top1_intent_change.py \
  --source-data ../PromptGen/data/xiaoyi_intent_v1.jsonl \
  --reserved-id-dir ../PromptGen/eval/cohorts
```

输出 `top1_intent_change_v1.jsonl` 和相邻的 summary。JSONL 使用标准 Top1 训练格式，
可在构建最终 `train.jsonl` 时与主训练数据合并；两份生成文件继续由 `.gitignore`
排除，不进入 Git。
