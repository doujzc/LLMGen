# Light-301 Skill 路由数据集

Light-301 是面向个人手机 Agent 的中文闭集 Skill 路由数据集。训练、验证、测试和
推理共享 `final/skills.jsonl` 中同一套 301 个候选，不包含 unseen-skill
评估。

当前快照以 Git 修订 `f325809` 的上一版数据为基线。多 Skill 数据和目标顺序
增广均恢复为该版本，只对单 Skill 能力对齐数据做了小规模、可审计的定向修补。

## 数据规模

| 数据 | Query 数 | 每条目标数 |
| --- | ---: | ---: |
| 单 Skill 能力对齐 | 5,576 | 1 |
| 多 Skill 训练集 | 33,098 | 2–4 |
| 多 Skill 验证集 | 530 | 2–4 |
| 多 Skill 测试集 | 631 | 2–4 |
| bad-case 回归集 | 31 | 1 |

多 Skill 训练集包含 12,812 条不同语义样本，按上一版策略为每条 query 生成
2–3 个目标顺序，共得到 33,098 条训练序列。验证集和测试集不做顺序增广。

单 Skill 对齐覆盖 301/301 个候选，每个候选至少 15 条。按“单 Skill 对齐 +
未增广的多 Skill 训练样本”统计，每个候选仍至少有 100 条正监督。

## 定向修补

修补源位于 `manual_alignment.jsonl`，共 167 条，覆盖 26 个候选：

| 类型 | 数量 | 目的 |
| --- | ---: | --- |
| 品牌或 Skill 名直呼 | 65 | 覆盖“华泰”“涨乐”“中金财富”“东方财富”等真实说法 |
| 语义矫正 | 45 | 强化候选真正的核心能力，避免被泛化成普通文案或人设修改 |
| 边界消歧 | 28 | 区分相邻金融、会议、音频、邮件和研究类候选 |
| bad-case 改写 | 29 | 使用线上问题的等价改写训练，原始 bad case 继续留作回归测试 |

其中 47 条基线样本被替换，而不是简单叠加：

- `brainhole-factory`：删除泛化的起名、广告文案等样本，改为平行宇宙、荒诞
  新闻、后果推演、meme 和图片提示词组合任务；
- `pua`：删除错误的人格、语气调整样本，改为连续失败后的主动排查、换方法、
  穷尽假设和验证闭环；
- `tencent-meeting-mcp`：删除超出候选描述的会议控制样本，改为会议管理、
  成员、录制、转写搜索和 AI 纪要。

`ai-zhangle-skills` 的候选信息补充了“华泰证券”“涨乐财富通”身份，并加入
12 条品牌直呼样本；`speech-to-text` 的错误显示名“语音合成”已更正为
“ElevenLabs语音转写”。

修补可追溯信息、替换清单和类别计数记录在
`manual_alignment.manifest.json`。`final/manifest.json` 记录最终文件哈希和
每个候选的增补数量。

## 文件

| 路径 | 说明 |
| --- | --- |
| `candidates.jsonl` | 301 个原始候选的 ID、名称和描述 |
| `skill_metadata_patches.jsonl` | 两项候选名称/描述纠错 |
| `manual_alignment.jsonl` | 定向单 Skill 对齐样本 |
| `manual_alignment.manifest.json` | 基线版本、替换策略及数量 |
| `regression/queries.jsonl` | 31 条不进入训练的原始 bad case |
| `regression/qrels.jsonl` | bad case 的期望候选 |
| `final/skills.jsonl` | 唯一候选注册表 |
| `final/queries_alignment.jsonl` | 单 Skill 能力课程 |
| `final/qrels_alignment.jsonl` | 单 Skill 正例关系 |
| `final/queries_{train,validation,test}.jsonl` | 多 Skill 数据划分 |
| `final/qrels_{train,validation,test}.jsonl` | 多 Skill 正例关系 |
| `final/queries.jsonl` | 多 Skill 聚合审计视图，不是额外训练集 |
| `final/manifest.json` | 数据统计、补丁来源和文件 SHA-256 |
| `final/quality_report.json` | 数据质量门禁结果 |

## 限制

- Query 和标签包含模型生成、模型复核与少量人工精修，不等同于真实用户日志；
- qrels 只提供正例，没有完整的困难负例标注；
- `hithink-iwencai` 与 `hithink-wencai-suite` 的名称和能力描述几乎相同，仅凭
  自然语言 query 无法稳定区分，当前没有人为编造二者边界；
- 回归集规模较小，应同时关注常规验证集、测试集和人工 bad-case 结果。
