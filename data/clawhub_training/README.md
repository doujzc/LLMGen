# ClawHub-1000 Skill 路由数据集

## 概览

ClawHub-1000 是面向个人手机 Agent 的中文闭集 Skill 路由数据集。给定一条自然
语言 query，目标是在固定的 ClawHub Top-1,000 候选中识别一个或多个相关 Skill。
候选、训练集、验证集和测试集共享同一份 `skills.jsonl`，不包含 unseen-skill
评估。

候选来自 2026-07-22 校验后的 ClawHub Top-1,000 归档快照，全部 1,000 个候选
均被保留，没有按 `mobile_fit` 或训练覆盖率再次过滤。`Qwen3.6-Plus` 用于生成
能力画像和合成 query，`GLM-5.1` 用于独立质量复核。当前最终快照生成于
2026-07-22，格式版本为 1，随机种子为 `20260720`。

## 数据规模

| 子集 | Query 数 | 正例关系数 | 每条 Query 的目标数 |
| --- | ---: | ---: | ---: |
| 单 Skill 对齐集 | 5,963 | 5,963 | 1 |
| 多 Skill 训练集 | 12,360 | 34,176 | 2–4 |
| 多 Skill 验证集 | 190 | 522 | 2–4 |
| 多 Skill 测试集 | 161 | 432 | 2–4 |

多 Skill 训练集包含 4,959 条不同的语义样本；目标顺序增广后得到 12,360 条
训练序列，平均每条语义样本对应 2.492 个目标排列。验证集和测试集没有顺序增广。

其他统计如下：

- 导出的多 Skill query 共 12,711 条，其中双目标 5,195 条、三目标 5,324
  条、四目标 2,192 条；
- 跨领域 query 共 11,321 条；
- 隐式意图 query 共 5,278 条，占 41.52%；显式意图 query 共 7,433 条；
- 单 Skill 对齐集覆盖 1,000/1,000 个候选，每个候选至少 5 条，平均 5.963 条；
- 多 Skill 训练集覆盖 997/1,000 个候选；
- 按“单 Skill 对齐样本 + 未增广的多 Skill 训练样本”统计，1,000 个候选均有
  正监督，每个候选至少 10 条，平均 19.033 条；
- 质量审计状态为 `pass`，目标位置覆盖率为 1.0，未发现近重复样本。

隐式/显式意图数量以及目标数量分布按最终导出序列统计，因此训练集的目标顺序变体
会分别计数。

## 最终数据文件

可消费的数据位于 `final/`：

| 文件 | 说明 |
| --- | --- |
| `skills.jsonl` | 唯一的 1,000-Skill 候选注册表 |
| `queries_alignment.jsonl` | 单 Skill 能力对齐 query |
| `qrels_alignment.jsonl` | 单 Skill query 与正例 Skill 的关系 |
| `queries_train.jsonl` | 经过目标顺序增广的多 Skill 训练集 |
| `queries_validation.jsonl` | 未增广的多 Skill 验证集 |
| `queries_test.jsonl` | 未增广的多 Skill 测试集 |
| `qrels_train.jsonl` | 训练 query 与正例 Skill 的关系 |
| `qrels_validation.jsonl` | 验证 query 与正例 Skill 的关系 |
| `qrels_test.jsonl` | 测试 query 与正例 Skill 的关系 |
| `queries.jsonl` | 多 Skill 数据的聚合审计视图，含划分和复核信息 |
| `manifest.json` | 数据规模、覆盖率、分布、增广信息及文件 SHA-256 |
| `quality_report.json` | 隐式意图、顺序增广和候选覆盖质量报告 |
| `rejected_near_duplicates.jsonl` | 近重复拒绝记录；当前为空 |

`queries.jsonl` 是审计视图，不是额外训练集；不能再与三个 split 文件拼接，否则
会重复计入相同样本。

目录根部的 `skill_profiles.jsonl`、`workflows.jsonl`、
`queries.generated.jsonl`、`query_reviews.jsonl` 及对应 manifest/error 文件
保存生成和复核过程的中间数据，不属于最终训练/评估划分。

## 样本语义

### 单 Skill 对齐

每条对齐 query 只对应一个 Skill，用于直接表达该候选的典型用户意图。样本均为
显式意图，并带有 query 内的证据片段。

### 多 Skill 协作

每条多 Skill query 需要 2–4 个候选协作完成，覆盖办公文档、通信、知识管理、
金融、媒体、出行、天气等 20 个领域。样本强调有上下文关系的完整任务，而不是
互不相关请求的简单拼接。

`intent_mode="implicit"` 表示至少一个目标能力没有被直接点名，但被认为是完成
请求所必需的。例如，行程规划请求可能隐式需要天气查询。每个隐式目标都在
`implicit_skill_ids` 和 `implicit_rationales` 中显式记录。

### 目标顺序增广

训练集中同一语义 query 可具有 2 或 3 个不同的 `skill_ids` 排列。
`source_query_id` 将这些变体关联起来，`target_order_variant` 标识排列编号。
排列只表示自回归监督顺序，不表示 Skill 的实际执行依赖。属于同一工作流的样本
不会跨训练、验证和测试划分。

## 字段

### `skills.jsonl`

| 字段 | 含义 |
| --- | --- |
| `skill_id` | 唯一且稳定的 ClawHub 候选标识 |
| `name` | Skill 显示名 |
| `description` | ClawHub 能力描述 |
| `capability_zh` | 中文能力摘要 |
| `domain` | 领域标签 |
| `roles` | 能力角色，如检索、规划、创建或执行 |
| `mobile_fit` | 手机 Agent 适配度元数据；不参与候选过滤 |
| `rank` | Top-1,000 快照中的排名 |
| `canonical_url` | ClawHub Skill 页面 |

### Query 文件

| 字段 | 含义 |
| --- | --- |
| `id` | 当前样本的唯一标识 |
| `query` | 中文用户请求 |
| `skill_ids` | 有序目标 Skill 列表 |
| `workflow_id` | 多 Skill 样本所属的语义工作流 |
| `anchor_skill_id` | 构造工作流时的锚点 Skill |
| `domains` | 样本覆盖的领域 |
| `intent_mode` | `explicit` 或 `implicit` |
| `target_intents` | 每个目标 Skill 的显式/隐式属性 |
| `evidence` | query 中支持每个目标的原文片段 |
| `implicit_skill_ids` | 隐式目标 Skill |
| `implicit_rationales` | 选择各隐式目标的理由 |
| `quality_scores` | 连贯性、复杂度、手机风格、具体性和目标必要性评分 |
| `source_query_id` | 顺序增广前的语义样本 ID |
| `target_order_variant` | 当前目标排列编号 |

单 Skill 文件还包含 `curriculum_phase="single_skill_alignment"`，且没有多
Skill 工作流字段。验证集和测试集没有 `source_query_id` 与
`target_order_variant`。

### Qrels 文件

每行包含 `query_id`、`skill_id` 和 `relevance`。当前所有已列出的关系均为
`relevance=1`；未列出的候选只是未标注为正例，并非人工确认的困难负例。

## 数据来源与限制

- 候选源自 ClawHub 排名快照，会继承抓取时点、下载量排序和平台生态的分布偏差；
- Query、标签和质量评分均包含模型生成或模型复核成分，不等同于真实用户日志或
  人工金标；
- 数据只覆盖固定的 1,000 个候选，不包含候选集外请求、未知 Skill 或无可用
  Skill 的拒识样本；
- qrels 只提供正例关系，不提供人工标注的困难负例；
- ClawHub Skill 包是未受信任的第三方内容；本数据集仅使用其描述和元数据，不
  表示对包内代码进行安全认证；
- Skill 的归属、许可和使用条件以 `canonical_url` 指向的上游页面和原始包为准，
  本数据集不统一授予第三方内容的许可。
