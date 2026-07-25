# ClawHub Top-1,000 Skill 原始快照

## 概览

该目录保存从 ClawHub 公共 API 获取的 Top-1,000 Skill 原始快照。API 排名快照
采集于 2026-07-20，最终归档包含 1,000 个 Skill、1,000 份元数据记录和
1,000 个安全解压但未执行的 Skill 包；下载失败数为 0。

快照先冻结候选列表，再下载包文件，避免下载行为改变候选排名。候选选择规则为：

1. 仅保留 ClawHub API 未标记为 suspicious 的记录；
2. 以 `downloads` 降序选择；
3. 下载量相同时依次按 `stars`、`installs`、`updated_at` 降序和 `slug`
   升序确定顺序。

采集时共读取 1,190 条记录，选择前 1,000 条；入选边界的下载量为 5,230。
快照格式版本为 1，数据源为 `https://clawhub.ai/api/v1/skills`。

## 文件与目录

| 路径 | 说明 |
| --- | --- |
| `catalog.jsonl` | 规范化候选注册表，每个 Skill 一行，按最终排名排列 |
| `catalog_snapshot.json` | 下载前冻结的原始 API 响应和选择元数据 |
| `catalog_resolved.json` | 候选版本与归档解析结果 |
| `manifest.json` | 来源、时间、数量、筛选规则以及快照/目录 SHA-256 |
| `metadata/<owner>/<slug>.json` | 单个 Skill 的原始详情、版本和包来源信息 |
| `skills/<owner>/<slug>/` | 对应版本的安全解压内容 |
| `errors.jsonl` | 下载或解析失败记录；当前为空 |

`catalog.jsonl` 的主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `skill_id` | 稳定标识，格式通常为 `@owner/slug` |
| `owner`、`slug` | ClawHub 所有者和 Skill 路径名 |
| `display_name` | 展示名称 |
| `description`、`summary` | ClawHub 提供的能力描述 |
| `rank` | 当前快照中的排名 |
| `stats` | 下载、安装、星标、评论和版本数量 |
| `latest_version` | 快照解析到的最新版本 |
| `tags`、`topics` | ClawHub 标签和主题 |
| `canonical_url` | ClawHub Skill 页面 |
| `artifact` | 下载版本、时间、归档哈希和文件清单 |

`metadata/` 保留更完整的 API 详情和归档来源，`skills/` 保留上游包的原始目录
结构。快照不对包内脚本做执行、依赖安装或功能验证。

## 完整性与信任边界

`manifest.json` 记录 `catalog.jsonl` 和原始 API 快照的 SHA-256；每条候选记录的
`artifact.archive_sha256` 记录对应归档哈希。候选条目、元数据和解压目录应通过
`skill_id`、`owner`、`slug` 与版本号关联。

Skill 包属于未受信任的第三方内容。快照中的下载量、星标、描述和 suspicious
状态均来自采集时的 ClawHub 数据，不构成安全审核、质量认证或长期有效保证。执行
包内脚本前需要独立审查代码、依赖、权限和网络行为。

`manifest.json` 记录 ClawHub 对发布包许可的说明；具体 Skill 的归属、许可与使用
条件仍应以对应元数据、包内容和 `canonical_url` 指向的上游页面为准。
