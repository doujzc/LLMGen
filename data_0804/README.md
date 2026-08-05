# 0804-166 Skill 路由数据集

0804-166 是面向个人手机 Agent“小艺”的中文闭集 Skill 路由数据集。训练、验证、
测试和推理共享 `final/skills.jsonl` 中同一套 166 个候选，不包含 unseen-skill
评估，也没有从输入候选中筛除 Skill。

候选来源为仓库根目录的 `candidates_0804.jsonl`。Qwen3.7-Plus 根据候选原始描述
生成单 Skill 能力对齐样本以及需要 2–4 个 Skill 协作的复杂请求，GLM-5.2 对样本
进行独立复核。数据同时包含显式请求和隐式意图，并对多目标训练样本执行确定性的
目标顺序增广。

## 数据规模

| 数据 | Query 数 | 每条目标数 |
| --- | ---: | ---: |
| 单 Skill 能力对齐 | 3,050 | 1 |
| 多 Skill 训练集 | 21,146 | 2–4 |
| 多 Skill 验证集 | 297 | 2–4 |
| 多 Skill 测试集 | 380 | 2–4 |

多 Skill 训练集来自 6,450 条不同语义样本。根据目标数，每条训练 query 保留 2
或 4 种目标顺序；验证集和测试集不做顺序增广。最终多 Skill 数据包含 8,090 条
隐式意图序列，占 37.1%。

单 Skill 对齐覆盖 166/166 个候选，每个候选至少 12 条。按“单 Skill 对齐 +
未增广多 Skill 训练样本”统计，每个候选至少有 35 条正监督，平均 131.1 条。

## 导出状态

当前快照是一次显式的 provisional 导出：生成阶段共有 11,523 条多 Skill query，
其中 11,457 条完成复核，7,127 条通过；API 预算耗尽时尚有 66 条未复核，这些样本
被直接排除，没有被伪造为通过。

本次导出采用实际可满足的综合覆盖下限 35 和单 Skill 下限 12，低于原计划的
100/15。文件结构、候选一致性、训练规模、目标顺序、隐式意图和路由场景覆盖均已
通过质量审计，但该快照不代表原计划覆盖门槛已经完成。例外原因及全部缺失 query
ID 记录在 `final/manifest.json` 和 `final/quality_report.json`。

## 文件

| 路径 | 说明 |
| --- | --- |
| `../candidates_0804.jsonl` | 166 个原始候选的名称与描述 |
| `final/skills.jsonl` | 训练、评估和推理共用的唯一候选注册表 |
| `final/queries_alignment.jsonl` | 单 Skill 能力课程 |
| `final/qrels_alignment.jsonl` | 单 Skill 正例关系 |
| `final/queries_{train,validation,test}.jsonl` | 多 Skill 数据划分 |
| `final/qrels_{train,validation,test}.jsonl` | 多 Skill 正例关系 |
| `final/queries.jsonl` | 多 Skill 聚合审计视图，不是额外训练集 |
| `final/manifest.json` | 数据统计、provisional 说明和文件 SHA-256 |
| `final/quality_report.json` | 最终质量审计结果 |

## 限制

- Query 与标签来自模型生成和模型复核，不等同于真实用户日志；
- 66 条未完成复核的生成样本未进入任何数据划分；
- 41 个候选没有达到原计划的每候选 100 条正监督，最低候选为 35 条；
- qrels 只提供正例，没有完整的困难负例标注；
- 训练后仍应结合真实请求和人工 bad case 评估路由效果。
