# 0804-166 Skill 路由数据集

该数据集面向个人手机 Agent“小艺”的中文闭集 Skill 路由。训练、验证、测试与
推理共用 `final/skills.jsonl` 中同一套 166 个候选；候选不经过筛除。两个存在
名称/描述冲突的候选通过 `metadata_patches.jsonl` 显式修正，Skill ID 不变。

## 构建方法

DeepSeek Flash 根据候选画像和历史能力组合生成新 query；独立严格复核模型逐条
检查两个目标是否形成依赖、共享对象或共享目标，并拒绝 mere conjunction（无关
任务硬拼）。复核后按每个候选的训练正例缺口，从已通过样本派生训练专用 workflow，
重新生成并再次复核。

外部测试 CSV 只在本地提取目标数量、长度、句式和领域等聚合统计，以及执行
exact/near-duplicate 门禁；测试原句和具体 target 对不会进入任何模型 prompt，
也不会写入本数据集。生成及复核结果均支持断点续跑。

所有多 Skill 样本均包含两个目标。每个 workflow 生成 5 种场景，其中 4 条显式、
1 条隐式；训练样本再生成正反两种 target 顺序。该设计对应外部测试集中“短 query、
双目标、无句末标点”的分布。

## 数据规模与门禁

精确规模与文件哈希见 `final/manifest.json`，分布、覆盖及泄露结果见
`final/teststyle_audit.json`。最终导出必须同时满足：166 个候选完全一致、每个候选
至少 15 条单 Skill 对齐样本和 100 条多 Skill 训练正例、目标数恒为 2、测试集
exact/near 泄露为 0。训练样本保留正反两种 target 顺序，验证与测试不做顺序增广。

## 文件

| 路径 | 说明 |
| --- | --- |
| `final/skills.jsonl` | 唯一候选注册表 |
| `final/queries_alignment.jsonl` | 单 Skill 课程数据 |
| `final/queries_{train,validation,test}.jsonl` | 多 Skill 数据划分 |
| `final/qrels_*.jsonl` | Query–Skill 正例关系 |
| `final/manifest.json` | 数据规模、来源和文件哈希 |
| `final/quality_report.json` | 通用质量审计 |
| `final/teststyle_audit.json` | 分布、覆盖和泄露专项审计 |
| `distribution_profile.json` | 不含测试原句和 Skill ID 的聚合分布 |

## 限制

- Query 与标签仍属于模型生成数据，不等同于真实用户日志或人工逐条标注；
- 独立语义复核能过滤明显硬拼，但不能保证消除所有标签噪声；
- 训练集在保证每个候选充分覆盖后再参考测试领域分布，领域比例不会逐项完全相等；
- qrels 仅包含正例，没有穷举困难负例；
- 外部测试 CSV 不属于仓库，训练后仍应在该 held-out 集及真实 bad case 上评估。
