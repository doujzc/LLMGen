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

普通电商商品的购买前品牌/型号选择、比较、价格优惠、性能评价和适用性判断统一属于
`EcommerceProduct`，即使 query 已经给出具体型号或没有点名京东、淘宝。药品、整车、
房屋、服务和软件，以及已有商品的使用、故障、售后和订单事务仍属于
`GeneralProduct`。对已生成数据进行人工边界复核时，使用显式 ID 清单应用修订：

```bash
uv run --no-sync python scripts/repair_top1_ecommerce_labels.py
```

修订不会改写原始双模型判断；每条修订记录在 `label_review_correction`，summary 同时记录
修订清单、输入和修订后数据的 SHA256。

## 普通零售资格边界数据

`scripts/build_top1_retail_boundary_v1.py` 不按商品名称穷举，而是围绕四个结构化边界轴
生成成对的最小差异样本：登记资产/零售模型、权利许可/商品副本、服务/零售工具、定制
工程/家用成品。同一对使用完全相同的购买动作表达，只改变准确交易对象，避免模型继续把
“买、多少钱、下单”当作 `EcommerceProduct` 的充分条件。

```bash
uv run --no-sync python scripts/build_top1_retail_boundary_v1.py
```

训练集 `top1_retail_boundary_v1.jsonl` 包含 192 条、96 对单轮样本，并自动合入默认 combined
数据。`top1_retail_boundary_v1_validation.jsonl` 包含 64 条、32 对，仅用于边界评测；其对象
家族与训练集完全隔离，不会合入训练。相邻 summary 记录抽象轴、候选分布和两个 split 的
SHA256。

## 可直接训练的 combined v1

`scripts/build_top1_combined_v1.py` 严格合并 reviewed 1,000 条基础集、controlled
multiturn 800 条增强集和 retail-boundary 192 条训练集。构建时拒绝重复 ID、完全重复
对话、非法候选和源数据版本漂移，并记录三份输入及最终输出的 SHA256。边界 validation
不参与合并。

```bash
uv run --no-sync python scripts/build_top1_combined_v1.py
```

输出 `top1_train_combined_v1.jsonl` 和相邻 summary，共 1,992 条。`configs/top1.env`
已将它设为默认训练数据，因此构建后可直接启动：

```bash
bash scripts/train_top1.sh
```
