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

## Reviewed 1,000-row training set

`scripts/build_top1_training_v1.py` 按 PromptGen 当前七候选语义边界构造均衡训练集。
它不会粗粒度映射整个旧数据集：只选取与当前 taxonomy 一致的场景族，并加入 46 条
重新核对过的生产难例。基础数据每类 100 条且单轮/多轮各半；随后覆盖全部 42 个有向
候选切换，共加入 300 条直接 IntentChange，最终得到 1,000 条。

```bash
uv run --no-sync python scripts/build_top1_training_v1.py
```

输出 `top1_train_v1.jsonl` 和 `top1_train_v1_summary.json`。这两个版本化文件作为
可复现实验数据提交到仓库；其它临时训练数据仍由 `.gitignore` 排除。PromptGen 的
历史 audit cohort 一旦被该训练集复用，就不能再作为无偏评测集；summary 会明确记录
复用数量和所有源文件哈希。

## 独立的可控多轮合成

`scripts/generate_top1_multiturn.py` 是与现有训练集构建器完全分离的 LLM 合成流程。
它先固化结构化对话计划，再由生成模型逐轮实现；GLM 与 Qwen 在看不到计划标签的情况
下独立盲判。只有计划标签、两次盲判标签、对话现象和质量门全部一致的样本才进入
`train.jsonl`。IntentChange 还要求两模型分别确认末轮只包含新需求，不承接、确认或评价
上一轮，也不使用切换元话语；任一模型否决就重新生成。

默认计划 800 条，其中 420 条以每个有向组合 10 条的配额覆盖全部 42 个 IntentChange，
其余覆盖渐进披露、上下文省略、修正澄清、assistant 干扰和冗余表达。运行过程支持断点
续跑；相同输出目录只能继续同一个 manifest，配置或输入变化时必须使用新目录。

```bash
uv run --no-sync python scripts/generate_top1_multiturn.py \
  --credentials-file ~/Codes/api_keys/llm_api.txt
```

默认产物位于 `data_top1/generated/top1_controlled_multiturn_v1/`，包含不可变 manifest、
计划、模型原始响应、逐次接受/拒绝记录、最终 `train.jsonl` 和质量汇总。该目录仍由
`.gitignore` 排除，不改变现有 `top1_train_v1.jsonl`。只检查计划而不调用 API 时使用
`--plan-only`。

生成数据可直接作为独立训练输入：

```bash
TOP1_TRAIN_DATA=data_top1/generated/top1_controlled_multiturn_v1/train.jsonl \
  bash scripts/train_top1.sh
```

## 可直接训练的 combined v1

`scripts/build_top1_combined_v1.py` 严格合并 reviewed 1,000 条基础集和 controlled
multiturn 800 条增强集。构建时拒绝重复 ID、完全重复对话、非法候选和源数据版本漂移，
并记录两份输入及最终输出的 SHA256。

```bash
uv run --no-sync python scripts/build_top1_combined_v1.py
```

输出 `top1_train_combined_v1.jsonl` 和相邻 summary，共 1,800 条。`configs/top1.env`
已将它设为默认训练数据，因此构建后可直接启动：

```bash
bash scripts/train_top1.sh
```
