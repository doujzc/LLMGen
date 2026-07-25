# SkillRet v1.0 数据快照

## 概览

该目录保存官方 Hugging Face 数据集 `ThakiCloud/SKILLRET` 的完整 SkillRet
v1.0 快照，固定于 revision
`7cae7cfbad2b0e1ebc9170892f568993aae543b0`。快照下载于
2026-07-13，共 11 个上游文件、725,301,219 bytes。

SkillRet 是英文 Agent Skill 检索基准：给定自然语言 query，从对应候选 Skill
语料中检索一个或多个相关 Skill。Skill 文档来自公开 GitHub 仓库，query 为合成
数据，qrels 提供二元正例关系。

## 数据规模

| 数据 | Train | Test | 全量 |
| --- | ---: | ---: | ---: |
| Skills | 10,123 | 6,660 | 17,810 |
| Queries | 63,259 | 4,997 | — |
| Qrels | 127,190 | 8,347 | — |

Train 与 Test 的 Skill 集不重叠，query ID 也不重叠。全量候选表还包含 1,027 个
不属于 Train/Test Skill 语料的候选。当前快照的所有 qrels 都是
`relevance=1`。

## 文件

| 路径 | 说明 |
| --- | --- |
| `data/skills/train.jsonl` | Train 候选 Skill 语料 |
| `data/skills/test.jsonl` | Test 候选 Skill 语料 |
| `data/queries/train.jsonl` | Train query |
| `data/queries/test.jsonl` | Test query |
| `data/qrels/train.jsonl` | Train query—Skill 正例关系 |
| `data/qrels/test.jsonl` | Test query—Skill 正例关系 |
| `data/skills.jsonl` | 含 17,810 个候选的完整 Skill 目录 |
| `data/taxonomy.json` | 两级功能分类体系 |
| `README.md` | 上游原始数据卡 |
| `croissant-rai.json` | 上游 Croissant/RAI 元数据 |
| `download_manifest.json` | 固定 revision、文件哈希、数量和本地验证结果 |
| `SHA256SUMS` | 上游文件的 SHA-256 清单 |

上游 `README.md` 是下载快照的一部分；`LOCAL_README.md` 只记录该本地快照的
结构和经验证状态。

## 字段

### Skill

| 字段 | 含义 |
| --- | --- |
| `id` | 唯一 Skill 标识 |
| `name`、`namespace` | Skill 名称与公开命名空间 |
| `description` | 简短能力描述 |
| `skill_md`、`body` | 完整 Skill Markdown；两字段内容重复 |
| `author`、`repo` | GitHub 作者与仓库 |
| `source_url`、`raw_url` | Skill 目录和原始 Markdown 地址 |
| `stars`、`installs` | 公开仓库/市场统计 |
| `license` | `MIT` 或 `Apache-2.0` |
| `major`、`sub` | 两级功能分类 |
| `primary_action` | 主要动作标签 |
| `primary_object` | 主要对象标签 |
| `domain` | 领域标签 |

分类体系包含 6 个 major categories 和 18 个 sub-categories；当前快照中的 Skill
分类值均能在 `taxonomy.json` 中解析。

### Query

| 字段 | 含义 |
| --- | --- |
| `id` | 唯一 query ID |
| `original_id` | 生成阶段的原始 ID |
| `query` | 英文用户请求 |
| `skill_ids` | 正例 Skill ID 列表 |
| `skill_names` | 正例 Skill 名称列表 |
| `k` | 正例数量 |
| `generator_model` | 生成 query 的模型 |

### Qrels

每行包含 `query_id`、`skill_id` 和 `relevance`。qrels 与 query 中的
`skill_ids` 已验证一致；未列出的候选不是正例，但不代表经过人工确认的困难负例。

## 完整性与许可

本地校验结果为：

- 所有 JSONL 均可解析；
- 每个 Skill 语料中的 ID 唯一；
- 每个 query split 中的 ID 唯一；
- 每个 qrel pair 唯一；
- Train/Test Skill 和 query 均无交集；
- 所有 Skill 分类标签有效；
- SHA-256 清单验证通过。

完整目录包含 15,570 个 MIT Skill 和 2,240 个 Apache-2.0 Skill。基准元数据、
query 和 taxonomy 使用 Apache-2.0；每份 Skill 文档保留其记录的原始许可。

## 数据限制

- Skill 来源限于公开 GitHub 生态，且领域分布偏向软件工程；
- Skill 语料以英文为主，不代表多语言 Skill 分布；
- query 由模型合成，不能完整替代真实用户流量；
- Train/Test 使用不同候选语料，评估协议与本仓库的闭集 ClawHub/Light 数据不同；
- qrels 只包含二元正例，不提供人工困难负例或安全性标签；
- 检索指标不能被解释为 Agent 执行安全、事实正确性或整体任务完成能力。
