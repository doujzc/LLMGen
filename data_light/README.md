# Light-301 Skill 路由数据集

Light-301 是面向个人手机 Agent 的中文闭集 Skill 路由数据集。训练、验证、测试、
导出和 Web 解码共享 `final/skills.jsonl` 中同一套 301 个候选，不包含
unseen-skill 评估。

## 数据规模

| 数据 | Query 数 | 目标数 |
| --- | ---: | ---: |
| 单 Skill 能力课程 | 5,636 | 每条 1 个 |
| 多 Skill 训练集 | 40,572 | 每条 2–4 个 |
| 多 Skill 验证集 | 530 | 每条 2–4 个 |
| 多 Skill 测试集 | 631 | 每条 2–4 个 |
| 线上 bad-case 回归集 | 31 | 每条 1 个 |

多 Skill 训练集包含 12,812 条不同语义样本。训练 split 对目标顺序做最多 4 个
确定性旋转或排列，得到 40,572 条自回归序列；验证集和测试集不做顺序增广。
每个 query-target 对都至少一次位于目标序列首位。

单 Skill 课程覆盖 301/301 个候选，每个候选至少 15 条，平均 18.72 条。按
“通过审核的单 Skill 样本 + 未增广的多 Skill 训练样本”统计，每个候选至少有
100 条正监督，平均 137.54 条。

## 候选画像

`final/skills.jsonl` 保留原始 `description`，并包含以下路由元数据：

- `aliases`：真实用户可能使用的产品名或别名；
- `capability_facets`：不可由一句短摘要替代的具体能力切面；
- `trigger_phrases`：品牌、格式、失败状态等区分性触发条件；
- `negative_boundaries` 和 `confusable_skill_ids`：能力边界与近邻候选；
- `routing_mode`：`atomic`、`composite` 或 `meta`。

`composite` 表示一个候选本身覆盖紧密相关的多步骤或多产物，例如脑洞设定、荒诞
新闻和生图提示词；`meta` 表示由失败次数、停滞状态或 Agent 配置等上下文触发。

## 单 Skill 课程

单 Skill 样本除普通核心请求外，还显式覆盖四类路由场景：

| 场景 | 通过样本数 |
| --- | ---: |
| Composite 能力整体请求 | 710 |
| 品牌、平台或特有产物显式出现 | 348 |
| Meta 能力的具体任务与状态触发 | 89 |
| 核心能力后的 LLM 原生文本处理 | 1,380 |

原生文本处理包括总结、翻译、简评、表格/清单/纪要、图片提示词以及把文字组织成
Word、PPT 或 HTML 内容。它们不额外增加外部候选；平台访问、抓取、浏览器验证、
发送、下单、预订和真实文件读写仍按候选能力边界标注。

Qwen3.7-Plus 生成的单 Skill 样本由 GLM-5.2 独立复核。另有 18 条由仓库显式
维护、未经过 GLM 独立复核的精修样本，来源见 `manual_alignment.jsonl`；为满足
每候选 100 条正监督，还去重迁移了上一版中 160 条已通过模型复核的单 Skill
样本，来源统计见
`legacy_alignment_import.json`。最终行内分别以 `curation_source` 或
`review_source` 保留来源，不冒充新版模型审核。

## 多 Skill 数据

多 Skill 样本覆盖显式和隐式意图。`intent_mode="implicit"` 表示至少一个支持
候选未被直接点名，但被时间、地点、风险、偏好或总目标强蕴含。最终序列中有
15,942 条隐式意图样本，占 38.20%。

本快照的多 Skill 语义样本沿用上一版由 Qwen3.7-Plus 生成、GLM-5.2 审核通过的
集合；本次重新应用了完整候选画像、主/支持候选元数据、四目标顺序增广和新版质量
门禁。未完成的新版多 Skill 重生成结果没有混入最终数据。

## 文件

| 路径 | 说明 |
| --- | --- |
| `candidates.jsonl` | 301 个原始候选的 ID、名称和描述 |
| `manual_alignment.jsonl` | 透明人工精修的单 Skill 样本源 |
| `legacy_alignment_import.json` | 旧版已审核样本的迁移数量与来源 |
| `regression/queries.jsonl` | 31 条 held-out 线上 bad case |
| `regression/qrels.jsonl` | bad case 的期望候选 |
| `final/skills.jsonl` | 唯一候选注册表 |
| `final/queries_alignment.jsonl` | 单 Skill 能力课程 |
| `final/qrels_alignment.jsonl` | 单 Skill 正例关系 |
| `final/queries_{train,validation,test}.jsonl` | 多 Skill 划分 |
| `final/qrels_{train,validation,test}.jsonl` | 多 Skill 正例关系 |
| `final/queries.jsonl` | 含 split 与审核信息的聚合审计视图 |
| `final/manifest.json` | 数量、覆盖率、增广配置和文件 SHA-256 |
| `final/quality_report.json` | 数据质量硬门禁报告 |

`final/queries.jsonl` 是审计视图，不能与三个 split 文件再次拼接训练。

## 质量状态与限制

当前 `final/quality_report.json` 状态为 `pass`：候选覆盖、每候选正监督、隐式
意图、专项单技能场景和目标首位监督均无缺口。

数据仍包含模型生成与模型审核成分，不等同于真实用户日志或完整人工金标；qrels
只标注正例，没有人工确认的负例；回归集只有 31 条，应与常规验证/测试指标结合
使用。
