# ClawHub Training 数据集

这是一个面向个人手机 Agent（小艺）Skill 路由任务的中文合成数据集。给定一条用户
query，模型需要从固定的 ClawHub Top-1,000 候选集中生成一个或多个相关 Skill。
数据集同时提供单 Skill 能力对齐样本和复杂多 Skill 协作样本，可直接用于本仓库的
两阶段 Router 训练。

当前快照生成于 2026-07-22，随机种子为 `20260720`，格式版本为 `1`。权威统计、
文件校验和与质量门结果分别记录在 `final/manifest.json` 和
`final/quality_report.json` 中。

## 数据规模

| 子集 | Query 数 | 正例关系数 | 每条 Query 的目标数 |
| --- | ---: | ---: | --- |
| 单 Skill 对齐集 | 5,963 | 5,963 | 1 |
| 多 Skill 训练集 | 12,360 | 34,176 | 2–4 |
| 多 Skill 验证集 | 190 | 522 | 2–4 |
| 多 Skill 测试集 | 161 | 432 | 2–4 |

- 候选 Skill：1,000 个，全部来自经过校验的 ClawHub Top-1,000 快照；没有按
  `mobile_fit` 或训练覆盖率再次筛选。
- 原始多 Skill 语义样本：5,310 条，其中训练 4,959 条、验证 190 条、测试
  161 条。
- 训练集目标顺序增广后：12,360 条；加上未增广的验证集和测试集，共 12,711 条。
- 隐式意图样本：5,278 条，占已导出多 Skill 样本的 41.52%。
- 跨领域样本：11,321 条，覆盖办公文档、通信、软件开发、知识管理、金融、媒体、
  出行、天气等 20 个领域。
- 每个候选至少有 5 条单 Skill 对齐样本；按“单 Skill 对齐样本 + 未增广的多
  Skill 训练样本”统计，每个候选至少有 10 条训练正例，均值为 19.033 条。

训练、验证、测试和推理共享同一个 `skills.jsonl` 候选集，因此这是**闭集路由**
数据，不是 unseen-skill 泛化数据集。

## 样本设计

### 单 Skill 能力对齐

`queries_alignment.jsonl` 为每个 Skill 构造至少 5 条只对应该 Skill 的显式 query。
训练时这部分数据先于多 Skill 数据使用，让模型先建立“Skill 功能—离散编码”的
基本映射，再学习复杂组合选择。

### 多 Skill 协作

多 Skill query 模拟用户向个人手机 Agent 发出的自然请求，具有以下特点：

- 一条 query 需要 2–4 个 Skill 协作完成，而不是若干互不相关需求的简单拼接；
- 覆盖多个生活和工作领域，并尽量形成有上下文联系的完整工作流；
- 同时包含显式意图和隐式意图。例如用户只说“规划一下五一”，标签可以包含行程
  规划、天气查询等完成该请求所必需、但没有被直接点名的 Skill；
- 保存每个目标的文本证据、显式/隐式属性、隐式选择理由和 1--5 分质量评分，便于
  审计标签。

### 目标顺序增广

多 Skill Router 采用完整自回归序列作为监督。为减弱标签排列顺序带来的偏置，训练
集中的每条原始语义样本生成 2 或 3 个不同的目标排列，最终从 4,959 条扩增到
12,360 条。验证集和测试集不做该增广。

`skill_ids` 的排列是生成训练顺序，不应解释为 Skill 的执行依赖顺序；路由评估按
目标集合计算 Recall 等指标。同一原始 query 的排列变体由 `source_query_id` 关联，
并且不会跨训练、验证和测试划分泄漏。

## 文件说明

所有可直接消费的数据都位于 `final/`：

| 文件 | 说明 |
| --- | --- |
| `skills.jsonl` | 唯一的 1,000-Skill 候选注册表 |
| `queries_alignment.jsonl` | 单 Skill 课程学习 query |
| `qrels_alignment.jsonl` | 单 Skill query 与正例 Skill 的关系 |
| `queries_train.jsonl` | 做过目标顺序增广的多 Skill 训练集 |
| `queries_validation.jsonl` | 未增广的多 Skill 验证集 |
| `queries_test.jsonl` | 未增广的多 Skill 测试集 |
| `qrels_{train,validation,test}.jsonl` | 各划分的 query—正例 Skill 关系 |
| `queries.jsonl` | 上述多 Skill 划分的聚合审计视图，额外包含 `split` 和完整复核信息 |
| `manifest.json` | 数据来源、规模、覆盖率、分布、增广信息和 SHA-256 校验和 |
| `quality_report.json` | 覆盖率、隐式意图、目标顺序等质量门结果 |
| `rejected_near_duplicates.jsonl` | 被近重复规则拒绝的样本；当前快照为空 |

`queries.jsonl` 是聚合审计视图，不是额外训练数据；不要再与三个拆分文件拼接，否则
会重复计入样本。

## 字段定义

### `skills.jsonl`

| 字段 | 含义 |
| --- | --- |
| `skill_id` | 稳定候选标识，例如 `@owner/skill-name` |
| `name` | Skill 名称 |
| `description` | Skill 的原始能力描述 |
| `capability_zh` | 用于生成与检查的中文能力摘要 |
| `domain` | 领域分类 |
| `roles` | 在工作流中的能力角色 |
| `mobile_fit` | 手机 Agent 适配度元数据；不参与候选过滤 |
| `rank` | 在 Top-1,000 快照中的排名 |
| `canonical_url` | ClawHub 上游页面 |

### Query 文件

| 字段 | 含义 |
| --- | --- |
| `id` | 当前样本的唯一标识 |
| `query` | 中文用户请求 |
| `skill_ids` | 目标 Skill 列表 |
| `workflow_id` | 原始工作流标识 |
| `anchor_skill_id` | 构造工作流时的锚点 Skill |
| `domains` | 样本涉及的领域 |
| `intent_mode` | `explicit` 或 `implicit` |
| `target_intents` | 每个目标 Skill 的显式/隐式属性 |
| `evidence` | query 中支持各目标的文本证据 |
| `implicit_skill_ids` | 未被直接点名但完成任务所需的目标 |
| `implicit_rationales` | 选择隐式目标的原因 |
| `quality_scores` | 连贯性、复杂度、手机输入风格、具体性和目标必要性等评分 |
| `source_query_id` | 训练集中目标顺序变体对应的原始 query |
| `target_order_variant` | 当前目标排列的变体编号 |

单 Skill 文件还包含 `curriculum_phase="single_skill_alignment"`。验证集和测试集没有
顺序增广，因此不包含 `source_query_id` 和 `target_order_variant`。

### Qrels 文件

每行包含 `query_id`、`skill_id` 和 `relevance`。当前只提供
`relevance=1` 的正例关系；其他候选是闭集训练或评估时的非标注候选，而不是经过
人工确认的困难负例。

## 校验与训练

克隆仓库后可先校验文件、覆盖率和数据约束：

```bash
python scripts/clawhub_data/05_validate_dataset.py \
  --dataset-dir data/clawhub_training/final \
  --expected-candidates 1000
```

完整训练默认先执行单 Skill 对齐，再执行多 Skill Retrieval：

```bash
export EMBEDDING_MODEL=/models/Qwen3-Embedding-8B
export ROUTER_MODEL=/models/Qwen3-1.7B
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
bash scripts/router_pipeline.sh clawhub full
```

默认参数位于 `configs/clawhub.env`；环境安装、Embedding 服务、四卡 ZeRO-3 训练和
分步调试方法见仓库根目录 `README.md`。

## 重新生成

训练现有快照不需要访问外部模型。若要从候选 Skill 重新生成并复核整个数据集，在
仓库外配置 `~/llm_api.txt`：

```yaml
base_url: http://host:port/v1
api_key: YOUR_KEY
model: Qwen3.6-Plus
```

然后执行：

```bash
bash scripts/run_clawhub_data.sh
```

当前脚本默认使用 Qwen3.6-Plus 做能力画像和样本生成、GLM-5.1 做独立复核，并关闭
thinking。只有完整重建通过覆盖率与质量门后，才会替换 `final/` 快照；API key 不会
写入数据元信息。各阶段可在 `scripts/clawhub_data/` 中单独运行和调试。

## 使用限制

- Query 和标签由大模型生成并经另一模型复核，不等同于真实用户流量或人工金标；
  `quality_scores` 也属于模型评估结果。
- 数据只覆盖固定 Top-1,000 候选集，不包含未知 Skill、无可用 Skill 或候选集外请求。
- 当前 qrels 只有正例，不适合直接研究人工困难负例排序。
- Top-1,000 快照会引入 ClawHub 排名和时间截面的分布偏差。
- Skill 的归属和许可仍以 `canonical_url` 指向的上游项目为准；本数据集不统一授予
  第三方 Skill 内容的许可。
