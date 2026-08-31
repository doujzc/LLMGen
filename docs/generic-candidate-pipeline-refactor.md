# 任意候选 Skill 训练闭环重构设计

状态：Phase 1 已实现；Phase 2～4 按本文契约渐进迁移
范围：输入候选 Skill 名称与描述，完成监督数据与 qrels 生成、Skill 编码学习、
Router 训练、评估和自包含模型导出。

> 兼容性原则：本次落地只增加编排、状态、配置、日志和 artifact 契约，不修改现有
> 数据生成算法、prompt、RQ-VAE、Router loss 或约束解码语义。新 Runner 通过薄适配器
> 调用现有 Python/Shell 入口；旧入口仍可独立运行。

## 1. 背景

项目已经具备以下训练能力：

- 使用候选 Skill 文档生成 embedding 和协同图；
- 训练 ToolWeaver 风格的层级 Skill Tokenizer；
- 导出固定长度的虚拟 token code；
- 构造 Memorization、Alignment 和 Retrieval 三类 Router SFT 数据；
- 分阶段微调 Router，并进行约束解码评估；
- 导出 `router_manifest.json`、`skill_decode_map.json` 和
  `virtual_tokens.txt` 等部署文件。

现有入口 [`scripts/router_pipeline.sh`](../scripts/router_pipeline.sh) 主要面向
ClawHub 1000 候选集和 Light 301 候选集。候选数据构造、训练配置和导出流程分别位于
多组 Shell 脚本、环境变量配置及 Python 入口中。它们能够复现实验，但还不能直接作为
“任意候选输入到可部署模型”的通用产品化流水线，主要问题包括：

1. 数据集名称、目录、编码空间和训练资源仍带有特定实验的固定值；
2. 数据生成与 Router 训练是两套入口，缺少统一的运行状态和 artifact 血缘；
3. 长阶段虽会写部分文件或 checkpoint，但没有统一的阶段完成、失效和恢复协议；
4. 配置主要通过多层 Shell 环境变量叠加，难以审阅某次实验的最终生效值；
5. qrels 只校验目标集合，尚未把多 Skill 执行顺序作为显式格式契约；
6. 日志分散在不同子进程中，难以从大量混合日志中快速定位某个 Run 和 Stage；
7. 修改一个上游参数后，系统不能自动判定哪些阶段可以复用、哪些阶段必须重跑。

本次重构保留现有算法实现，先建立统一编排、输入输出和恢复协议，再逐步把现有阶段迁移
到新的 Python 模块中。

## 2. 目标与非目标

### 2.1 目标

新的流水线应支持：

1. 输入任意非空候选集，最小字段为 Skill 名称和描述；
2. 自动生成单 Skill 和多 Skill 查询、审核结果及有序 qrels；
3. 自动补齐每个候选的训练覆盖率；
4. 根据候选数量规划单层或多层 Skill code；
5. 完成 Memorization、Alignment、Retrieval 课程训练；
6. 评估通过后导出可以被当前服务脚本加载的自包含模型目录；
7. 每个阶段及时持久化中间文件，并支持阶段内、阶段间恢复；
8. 可以独立执行某一阶段，或执行一个连续阶段区间；
9. 保存足够的结构化日志、原始 LLM 响应、训练指标和错误样本；
10. 通过不可变配置快照和 artifact 哈希完整复现实验。

目标命令为：

```bash
python scripts/train_candidates.py run \
  --candidates /path/to/candidates.jsonl \
  --config configs/router_pipeline.yaml \
  --output runs/my-router
```

正式交付目录固定为：

```text
runs/my-router/export/model/
```

### 2.2 非目标

首轮重构不包括：

- 重写 RQ-VAE、约束解码或 Router loss；
- 用 Web 服务替代命令行编排；
- 在流水线中保存明文 API Key；
- 保证任意质量的名称和描述都能生成高质量监督数据；
- 让修改后的候选集合与旧模型 codebook 无条件兼容；
- 把所有实验参数自动调优问题同时解决。

这里的“任意候选集”指任意有限、ID 可唯一化且描述足以区分能力的候选集合，不代表
首期适配器已经支持任意语言、任意 prompt 或任意 Skill 组合分布。当前复用的数据生成
算法仍使用既有中文质量门槛、prompt 和 2～4 Skill 工作流分布；配置与这些约束冲突时
必须在 Run 创建时明确失败，不能静默忽略。当前实现要求至少两个候选；单候选降级为
仅 Memorization/Alignment 是后续能力。

## 3. 核心设计原则

### 3.1 Run 目录是唯一事实来源

一次运行的输入快照、最终配置、阶段状态、日志、中间 artifact、checkpoint、评估和导出
全部位于同一个 `run_dir`。阶段不得依赖未登记的临时路径，也不得从当前工作目录猜测
上游文件。

### 3.2 阶段只通过 artifact 契约连接

每个阶段声明逻辑输入和输出，例如 `candidates.normalized`、
`dataset.queries.train` 或 `model.retrieval`。路径由 Artifact Registry 解析，阶段代码不
直接拼接其他阶段的内部目录。

### 3.3 配置与结果不可静默漂移

Run 创建后，其候选快照和 resolved config 视为不可变。修改配置应创建派生 Run，而
不是直接修改历史 Run。阶段恢复必须同时验证输入哈希、相关配置哈希和输出哈希。

### 3.4 长阶段按小批次原子提交

目标态下，LLM 请求、embedding、数据审核和训练不能等全部完成后才产生一个大文件。
每完成一个 batch 或 checkpoint，就写入独立分片并更新进度。正式汇总 artifact 仅在
阶段完成时发布，避免下游误读不完整文件。Phase 1 已提供 Stage 边界恢复、子进程心跳
和现有训练 checkpoint 透传；LLM/embedding 的不可变请求分片与去重账本属于 Phase 2。

### 3.5 原始证据与正式结果分离

LLM 原始请求、响应、被拒绝样本和 traceback 都要保存，但不能混入正式训练数据。
正式 artifact 只包含通过格式校验和质量门禁的数据。

## 4. 总体流程

```text
pipeline.yaml + candidates.jsonl
                 │
                 ▼
              创建 Run
                 │
     ┌───────────┴────────────┐
     │ resolved config        │
     │ run manifest           │
     │ artifact registry      │
     └───────────┬────────────┘
                 ▼
00 ingest
    → 01 enrich
    → 02 plan queries
    → 03 generate queries
    → 04 review and backfill
    → 05 finalize dataset
    → 06 train codebook
    → 07 assign codes
    → 08 build SFT
    → 09 train memorization
    → 10 train alignment
    → 11 train retrieval
    → 12 evaluate
    → 13 export
```

阶段名称是稳定 API。目录中的数字仅用于排序，不应被内部依赖关系用作标识。

## 5. Run 目录

```text
runs/my-router/
├── run_manifest.json
├── artifact_registry.json
├── config/
│   ├── pipeline.source.yaml
│   ├── pipeline.resolved.yaml
│   ├── overrides.json
│   └── environment.json
├── source/
│   ├── candidates.input.jsonl
│   ├── candidates.normalized.jsonl
│   └── candidate_manifest.json
├── stages/
│   ├── 00_ingest/
│   ├── 01_enrich/
│   ├── 02_plan_queries/
│   ├── 03_generate_queries/
│   ├── 04_review_queries/
│   ├── 05_finalize_dataset/
│   ├── 06_train_codebook/
│   ├── 07_assign_codes/
│   ├── 08_build_sft/
│   ├── 09_train_memorization/
│   ├── 10_train_alignment/
│   ├── 11_train_retrieval/
│   ├── 12_evaluate/
│   └── 13_export/
├── cache/
│   ├── llm/
│   └── embeddings/
├── logs/
│   ├── pipeline.log
│   ├── pipeline.jsonl
│   └── stages/
└── export/
    ├── model/
    └── report/
```

每个 Stage 目录使用一致结构：

```text
stages/03_generate_queries/
├── stage_state.json
├── input_manifest.json
├── progress.json
├── attempts/
│   └── 0001/
│       ├── requests/
│       ├── responses/
│       ├── drafts/
│       ├── failures.jsonl
│       ├── subprocess.log
│       └── debug.jsonl
└── output/
    ├── query_drafts.jsonl
    └── manifest.json
```

同一配置下的重试使用新的 `attempts/NNNN`，不能覆盖旧失败记录。`output/` 只指向或
复制当前成功 attempt 的正式输出。

## 6. 统一候选输入

最小输入格式为 JSONL：

```json
{"id":"weather","name":"天气查询","description":"查询指定城市的天气"}
{"id":"calendar","name":"日程管理","description":"创建、修改和查询日程"}
```

字段约束：

| 字段 | 必需 | 说明 |
|---|---|---|
| `id` | 推荐 | 稳定 Skill ID；缺失时按配置生成 |
| `name` | 是 | 展示名称和语义输入的一部分 |
| `description` | 是 | 描述能力、边界和适用场景 |
| `metadata` | 否 | 原样保留的扩展字段，不参与默认契约 |

规范化记录格式：

```json
{
  "skill_id": "weather",
  "name": "天气查询",
  "description": "查询指定城市的天气",
  "metadata": {},
  "source_line": 1
}
```

Stage 00 必须校验：

- 输入非空；
- ID 和名称非空；
- ID 唯一；
- 描述非空；
- JSON 对象格式正确；
- 标准化后的候选顺序稳定；
- 原始文件和标准化文件 SHA-256 已记录。

## 7. qrels 生成契约

### 7.1 先确定目标，再生成查询

LLM 不应自由决定 qrels。流程应为：

1. 程序根据候选画像、embedding 和覆盖率选择目标 Skill 组合；
2. 组合中明确记录 Skill 执行顺序；
3. 生成模型根据指定组合编写用户查询；
4. 独立审核模型验证每个目标 Skill 是否必要，并提取原文证据；
5. 程序根据审核通过的有序目标列表确定性生成 qrels。

这样可以把 qrels 从不可控的 LLM 自由输出，转换为可审计的程序派生 artifact。

### 7.2 Workflow 计划

```json
{
  "workflow_id": "workflow-000001",
  "skill_ids": ["weather", "calendar"],
  "selection_reason": "complementary",
  "required_order": true,
  "generation_variants": 3
}
```

目标组合应覆盖：

- 每个候选的单 Skill Alignment 查询；
- 语义相近、容易混淆的 Skill 组合；
- 能形成真实多步骤任务的互补 Skill 组合；
- 少量经过审核的跨域组合；
- 对覆盖不足候选的定向回填组合。

### 7.3 查询草稿

```json
{
  "draft_id": "draft-000001",
  "workflow_id": "workflow-000001",
  "query": "查询明天北京天气，然后加入我的日程。",
  "skill_ids": ["weather", "calendar"],
  "intent_mode": "explicit",
  "generator_request_id": "req-generate-000001"
}
```

### 7.4 审核结果

```json
{
  "draft_id": "draft-000001",
  "accepted": true,
  "skill_reviews": {
    "weather": {
      "necessary": true,
      "evidence": "查询明天北京天气"
    },
    "calendar": {
      "necessary": true,
      "evidence": "加入我的日程"
    }
  },
  "issues": [],
  "review_request_id": "req-review-000001"
}
```

被拒绝的数据不得删除，应进入 `rejected_queries.jsonl` 并记录拒绝原因，供提示词和
候选描述调优。

### 7.5 有序 qrels

```json
{"query_id":"query-000001","skill_id":"weather","relevance":1,"position":0}
{"query_id":"query-000001","skill_id":"calendar","relevance":1,"position":1}
```

`query.skill_ids` 是执行顺序的权威来源。数据校验必须满足：

```text
qrels 按 position 排序后的 skill_id == query.skill_ids
```

不能只比较集合。Stage 08 构造 Retrieval target 时也必须使用这个顺序，不能依赖
JSONL 行的偶然出现顺序。

此外还必须满足：`query_id` 在 split 内唯一；`query.skill_ids` 非空、无重复且全部属于
候选快照；每个正 qrel 的 `relevance` 为正；同一 query 的每个 Skill 恰有一行 qrel；
`position` 从 0 连续递增且无重复。Alignment 是单 target 特例，固定只有
`position=0`。任何 qrels 读取入口都必须显式按 `position` 排序；旧格式缺少 position
时只能在 `finalize-dataset` 中根据权威 `query.skill_ids` 确定性补齐并重新校验。

### 7.6 覆盖与切分

每个候选分别统计：

- Alignment 查询数；
- Retrieval 语义正例数；
- 显式和隐式正例数；
- 在二、三、四 Skill 工作流中的出现次数；
- Train、Validation 和 Test 分布。

覆盖不足时生成定向 backfill round：

```text
stages/04_review_queries/backfill/
├── round-01/
├── round-02/
└── coverage_report.json
```

切分单位使用 `workflow_id` 或规范化查询语义组，不能让同一查询的改写、顺序变体或
同源工作流跨越 Train、Validation 和 Test。

Phase 1 保持现有算法：`SHA-256(seed, "workflow_split", workflow_id)` 的前 64 bit
对 20 取模，bucket 0 为 Validation、bucket 1 为 Test、其余为 Train；coverage backfill
与 recovery workflow 永远进入 Train。小数据集不为凑比例移动样本，manifest 应记录
实际计数。顺序增强沿用源 `workflow_id` 的 split，禁止产生跨 split 派生样本。

## 8. 配置模型

统一配置使用带 schema version 的 YAML，并按职责分组：

```yaml
schema_version: 1

run:
  name: my-router
  output_dir: runs/my-router
  seed: 42

input:
  candidates: data/candidates.jsonl
  id_policy: explicit_or_name
  preserve_metadata: true

providers:
  generation:
    type: openai_compatible
    base_url: "${GENERATION_API_BASE}"
    api_key_env: GENERATION_API_KEY
    model: Qwen3.7-Plus
    concurrency: 12
    timeout_seconds: 300
    max_retries: 3
  review:
    type: openai_compatible
    base_url: "${REVIEW_API_BASE}"
    api_key_env: REVIEW_API_KEY
    model: GLM-5.2
    concurrency: 12
  embedding:
    type: openai_compatible
    base_url: "${EMBEDDING_API_BASE}"
    model: Qwen3-Embedding-8B
    batch_size: 8

data_generation:
  alignment_queries_per_skill: 10
  retrieval_positives_per_skill: 20
  skills_per_query:
    min: 2
    max: 4
  explicit_variants: 3
  implicit_variants: 1
  order_variants: 2
  max_backfill_rounds: 5
  split:
    train: 0.90
    validation: 0.05
    test: 0.05

code:
  mode: auto
  latency_priority: balanced
  spare_capacity_ratio: 1.25
  max_virtual_tokens: 512
  max_branching_factor: 256
  assignment: balanced_hierarchical

router:
  base_model: /models/Qwen3-1.7B
  finetune_mode: full
  precision: bf16
  max_length: 4096
  memorization:
    epochs: 10
    learning_rate: 2.0e-5
  alignment:
    enabled: true
    epochs: 3
    learning_rate: 2.0e-5
  retrieval:
    epochs: 10
    learning_rate: 2.0e-5
    alignment_replay_fraction: 0.15
    memorization_replay_fraction: 0.05

runtime:
  devices: auto
  distributed: auto
  deepspeed: auto
  dataloader_workers: 4

checkpointing:
  llm_batch_records: 20
  embedding_batch_records: 100
  training_save_steps: 100
  keep_last: 3

evaluation:
  query_split: test
  cutoffs: [1, 2, 5]
  require_format_valid_rate: 0.99
  require_candidate_coverage: 1.0
  metric_thresholds:
    recall@1: 0.80
    ordered_code_exact_match: 0.95

export:
  output_dir: export/model
  require_all_gates: true
  smoke_test: false

logging:
  console_level: INFO
  file_level: DEBUG
  marker: "[[LLMGEN-PIPELINE]]"
  progress_interval_seconds: 30
  capture_subprocess: true
  save_llm_requests: false
  save_llm_responses: false
  console_text_preview: false
  file_text_preview_chars: 1000
```

配置优先级固定为：

```text
代码默认值 < pipeline.yaml < CLI --set key=value
```

所有非敏感最终值写入 `pipeline.resolved.yaml`。API Key 只允许通过所引用的环境变量
读取，不写入配置快照、日志或 manifest。未知键、类型错误和不满足约束的组合应在 Run
创建时失败，不能静默忽略。

配置哈希按 Stage 所使用的字段子集计算。例如修改 Retrieval 学习率不应使 embedding
和 qrels artifact 失效；修改候选输入则必须使全部下游 Stage 失效。

Phase 1 的严格配置策略如下：

| 配置能力 | 当前行为 | 后续目标 |
|---|---|---|
| `skills_per_query` | 仅接受既有算法的 2～4 | 通用组合策略 |
| `split` | 仅接受既有 90/5/5 | 可配置、稳定的语义组切分 |
| `single_candidate_policy` | 仅接受 `error` | 支持 Alignment-only 降级 |
| `router.finetune_mode` | 仅接受 `full`，保证导出自包含 | 合并或显式打包 LoRA base model |
| `save_llm_requests/responses` | 必须为 `false` | 私有、不可变请求/响应分片 |
| `runtime.distributed` | 仅接受 `auto` | 显式资源计划与校验 |
| `evaluation.protocol` | 仅接受 `closedset` | 统一 closed-set/unseen 报告 |

任何尚未被适配器消费的行为型配置都应由 schema 拒绝。Provider 凭证只读取
`api_key_env` 指向的环境变量；resolved config、环境快照、registry 和全局日志不保存
凭证值。

## 9. 自动 CodePlan

现有 `32 × 16` 和 `128 × 128` 属于数据集专用设置。通用流水线应先生成并冻结
`code_plan.json`：

```json
{
  "schema_version": 1,
  "candidate_count": 301,
  "mode": "auto",
  "num_levels": 2,
  "branching_factors": [32, 16],
  "capacity": 512,
  "virtual_token_count": 48,
  "spare_capacity": 211
}
```

自动规划满足：

- code 容量不小于候选数和目标预留容量；
- 每级 codebook 大小不超过当前算法允许范围；
- tokenizer batch size 不小于最大 codebook；
- 控制新增 special token 总数；
- 在生成长度与词表扩张之间按配置权衡；
- 小候选集可以选择单 token，大候选集自动选择两级或更多级；
- 用户显式指定 `num_levels` 或 `branching_factors` 时严格校验，不自动修正。

CodePlan 是 Stage 06 和后续阶段的正式输入。修改 CodePlan 必须重跑 codebook、code
分配、SFT 构造、Router 训练、评估和导出。

## 10. 阶段输入输出契约

| Stage | 逻辑输入 | 正式输出 |
|---|---|---|
| `ingest` | 原始候选 JSONL、输入配置 | 标准化候选、候选 manifest、校验报告 |
| `enrich` | 标准化候选、Provider 配置 | Skill 画像、embedding 分片、embedding 矩阵 |
| `plan-queries` | 候选、画像、embedding | Alignment 任务、Skill 组合、workflow 计划 |
| `generate-queries` | workflow 计划、生成配置 | 原始请求、响应、query drafts、失败记录 |
| `review-queries` | query drafts、审核配置 | 审核结果、接受/拒绝查询、覆盖报告、回填轮次 |
| `finalize-dataset` | 审核通过查询、候选快照 | skills、queries、ordered qrels、split manifest |
| `train-codebook` | embedding、Train qrels | 协同图、RQ-VAE checkpoints、训练指标 |
| `assign-codes` | 最优 codebook、候选、CodePlan | codes、registry、virtual tokens、质量报告 |
| `build-sft` | dataset、codes、registry | Memorization、Alignment、Retrieval SFT 数据 |
| `train-memorization` | 基座模型、Memorization 数据 | 模型、checkpoint、trainer state、指标 |
| `train-alignment` | Memorization 模型、Alignment 数据 | 模型、checkpoint、trainer state、指标 |
| `train-retrieval` | 前置模型、Retrieval 与 replay 数据 | 最终 Router、checkpoint、trainer state、指标 |
| `evaluate` | 最终 Router、测试集、decoder artifact | 预测明细、错误样本、评估与门禁报告 |
| `export` | 模型、tokenizer、registry、评估结果 | 自包含模型、导出报告、artifact 血缘 |

### 10.1 Code artifact

```json
{
  "skill_id": "weather",
  "indices": [12, 3],
  "tokens": ["<SK_L1_12>", "<SK_L2_3>"],
  "code_text": "<SK_L1_12><SK_L2_3>"
}
```

### 10.2 Router SFT artifact

```json
{
  "phase": "retrieval",
  "group_id": "workflow-000001",
  "query_id": "query-000001",
  "input_text": "查询明天北京天气，然后加入我的日程。",
  "target_skill_ids": ["weather", "calendar"],
  "target_paths": [
    ["<SK_L1_12>", "<SK_L2_3>"],
    ["<SK_L1_7>", "<SK_L2_11>"]
  ],
  "target_text": "<SK_L1_12><SK_L2_3>\n<SK_L1_7><SK_L2_11>"
}
```

JSONL 适合记录级 artifact；JSON 适合 manifest、报告和状态；NumPy 或 Safetensors
适合矩阵与权重；PyTorch checkpoint 仅作为训练恢复格式，最终模型应导出为标准
HuggingFace 目录。

## 11. Artifact Registry

Stage 之间通过 `artifact_registry.json` 查找逻辑 artifact：

```json
{
  "schema_version": 1,
  "artifacts": {
    "candidates.normalized": {
      "path": "source/candidates.normalized.jsonl",
      "format": "jsonl",
      "artifact_schema": "candidate/v1",
      "producer": "ingest",
      "sha256": "...",
      "rows": 301,
      "bytes": 123456
    },
    "dataset.queries.train": {
      "path": "stages/05_finalize_dataset/output/queries_train.jsonl",
      "format": "jsonl",
      "artifact_schema": "router_query/v1",
      "producer": "finalize-dataset",
      "sha256": "...",
      "rows": 33098,
      "bytes": 4567890
    }
  }
}
```

每项至少包含：

- 逻辑名称；
- 相对 `run_dir` 的路径；
- artifact schema 和 schema version；
- producer Stage；
- SHA-256；
- 文件大小；
- 可用时的行数、shape、dtype 或模型摘要；
- 输入 artifact 和配置哈希形成的 lineage。

Registry 更新必须持锁并原子替换，不能被并发 Stage 写坏。

## 12. Stage 状态与恢复

每个 Stage 保存 `stage_state.json`：

```json
{
  "schema_version": 1,
  "stage": "generate-queries",
  "status": "running",
  "attempt": 2,
  "started_at": "2026-08-31T10:00:00Z",
  "updated_at": "2026-08-31T10:30:00Z",
  "input_artifacts": {
    "query.plan": {"sha256": "..."}
  },
  "config_hash": "...",
  "progress": {
    "completed": 1200,
    "total": 3000,
    "succeeded": 1170,
    "failed": 30
  },
  "outputs": {},
  "last_error": null
}
```

状态枚举为：

```text
pending
running
completed
failed
invalidated
skipped
```

Stage 仅在以下条件同时满足时可复用：

1. 状态为 `completed`；
2. 输入 artifact 哈希未变化；
3. Stage 相关配置哈希未变化；
4. 输出文件存在且哈希匹配；
5. 输出格式和质量门禁通过。

正式输出采用临时文件加原子 rename。`COMPLETED` 标记最后写入。运行中断时，
`running` 状态不会被当作完成，但新 attempt 可以从 batch ledger、embedding shard 或
训练 checkpoint 恢复。

强制重跑上游 Stage 后，Runner 根据依赖 DAG 把所有受影响的下游 Stage 标记为
`invalidated`。未受影响的 artifact 保持可复用。

同一 `run_dir` 同时只允许一个 Runner 执行，使用非阻塞进程文件锁；锁被占用时立即
返回可定位错误，进程退出后操作系统自动释放锁，不以可遗留的 PID 文件判断存活。
上次进程留下的 `running` 状态在下一次执行时创建新 attempt，旧 attempt 保留用于
审计。成功提交顺序固定为：验证输出并计算哈希 → 写 output manifest → 原子登记
artifact → 将 Stage 标记为 `completed` → 最后写 `COMPLETED` 标记。`--force-stage`
移除该 Stage 及下游的正式 registry 记录并标记失效，但保留历史 attempt。显式
checkpoint 只传给对应训练 Stage；目标态还需在 checkpoint 内校验输入 lineage 与配置
哈希。

## 13. 及时写出中间文件

本节描述最终恢复粒度。当前适配层会立即保存 Stage state、attempt、命令记录、原始
子进程日志、心跳、正式输出 manifest 和训练 checkpoint；但现有数据生成脚本尚未提供
统一的 request/response shard ledger。因此 Phase 1 可保证“已完成 Stage 不重跑”和
“训练从显式 checkpoint 继续”，不能承诺生成/embedding Stage 崩溃后逐请求零重复。
为避免形成虚假安全感，对应日志开关在 ledger 落地前由配置校验直接拒绝。

### 13.1 LLM 生成与审核

每完成一个 batch 即写入一个不可变分片：

```text
requests/part-000001.jsonl
responses/part-000001.jsonl
drafts/part-000001.jsonl
```

分片记录至少包含：

- request ID 和 prompt hash；
- Provider、模型和非敏感请求参数；
- 尝试次数、状态码、耗时；
- 原始响应或解析错误；
- 生成的正式记录 ID；
- 是否进入重试、拒绝或接受集合。

Stage 完成时按分片 manifest 合并正式输出。恢复时跳过已经成功并通过哈希校验的
request，不重复付费调用。

### 13.2 Embedding

```text
embeddings/
├── shards/
│   ├── 000000-000099.npy
│   └── 000100-000199.npy
├── shard_manifest.json
└── embeddings.npy
```

每个 shard 记录 Skill ID 顺序、shape、dtype、模型、维度和哈希。合并前严格校验候选
顺序，防止 embedding 与候选错位。

### 13.3 Codebook

```text
checkpoints/
├── checkpoint-000100.pt
├── checkpoint-000200.pt
└── best.pt
metrics.jsonl
training_state.json
```

训练 checkpoint 必须包含模型、optimizer、scheduler、AMP、RNG、epoch、step 和输入
数据血缘。

### 13.4 Router

继续使用 HuggingFace checkpoint 目录：

```text
checkpoint-100/
checkpoint-200/
trainer_state.json
router_manifest.json
```

Memorization、Alignment 和 Retrieval 是三个独立 Stage，不能再隐藏在一个不可拆分的
Shell 命令中。

## 14. CLI 与执行语义

### 14.1 创建并完整执行

```bash
python scripts/train_candidates.py run \
  --candidates candidates.jsonl \
  --config configs/router_pipeline.yaml \
  --output runs/my-router
```

### 14.2 查看状态

```bash
python scripts/train_candidates.py status \
  --run-dir runs/my-router
```

### 14.3 单独执行 Stage

```bash
python scripts/train_candidates.py stage generate-queries \
  --run-dir runs/my-router
```

单独执行时必须先验证全部上游 artifact，不能隐式补跑上游。缺少输入时打印逻辑
artifact 名称、期望 producer 和建议命令。

### 14.4 从某阶段继续

```bash
python scripts/train_candidates.py run \
  --run-dir runs/my-router \
  --from review-queries
```

### 14.5 执行阶段区间

```bash
python scripts/train_candidates.py run \
  --run-dir runs/my-router \
  --from train-codebook \
  --to build-sft
```

### 14.6 强制重跑

```bash
python scripts/train_candidates.py run \
  --run-dir runs/my-router \
  --force-stage assign-codes
```

### 14.7 从训练 checkpoint 恢复

```bash
python scripts/train_candidates.py stage train-retrieval \
  --run-dir runs/my-router \
  --resume-checkpoint checkpoint-500
```

### 14.8 派生实验

```bash
python scripts/train_candidates.py fork \
  --from-run runs/my-router \
  --output runs/my-router-one-token \
  --set code.num_levels=1 \
  --set router.retrieval.learning_rate=1e-5
```

派生 Run 记录 `parent_run_id`、配置差异和复用的 artifact。典型失效边界：

| 变更 | 最早重跑 Stage |
|---|---|
| 候选输入 | `ingest` |
| 生成模型或提示词 | `generate-queries` |
| 审核模型或审核规则 | `review-queries` |
| 数据切分 | `finalize-dataset` |
| CodePlan 或 codebook 参数 | `train-codebook` |
| code 分配策略 | `assign-codes` |
| SFT prompt | `build-sft` |
| Retrieval 学习率 | `train-retrieval` |
| 评估阈值 | `evaluate` |
| 导出格式 | `export` |

## 15. 日志设计

### 15.1 标准输出

所有流水线日志使用统一标记：

```text
[[LLMGEN-PIPELINE]] event=stage.begin run_id=my-router stage=generate-queries attempt=1
[[LLMGEN-PIPELINE]] event=stage.progress run_id=my-router stage=generate-queries completed=1200 total=3000 rate_per_sec=4.2 eta_seconds=429
[[LLMGEN-PIPELINE]] event=llm.batch_complete run_id=my-router stage=generate-queries succeeded=18 failed=2 latency_ms=3280
[[LLMGEN-PIPELINE]] event=stage.complete run_id=my-router stage=generate-queries elapsed_ms=714329 outputs=3000
```

标准字段至少包括：

```text
timestamp
run_id
stage
attempt
event
level
elapsed_ms
pid
host
config_hash
```

### 15.2 文件日志

日志分层保存：

- `logs/pipeline.log`：人类可读的全局 INFO 日志；
- `logs/pipeline.jsonl`：结构化全局事件；
- `logs/stages/<stage>.log`：阶段标准输出；
- `stages/<stage>/attempts/<id>/debug.jsonl`：DEBUG 细节；
- `subprocess.log`：训练等子进程原始 stdout/stderr；
- `metrics.jsonl`：可绘图的指标流；
- `failures.jsonl`：失败请求或数据记录。

控制台默认不打印完整 prompt、查询和模型响应，只打印请求 ID、长度、状态和耗时。
原始文本根据配置写入权限受限的中间 artifact。所有异常在 DEBUG 文件中保留完整
traceback，在 stdout 中打印异常类型、摘要和定位路径。

全局文本/JSONL 日志对嵌套对象和列表递归脱敏，并使用配置引用到的 secret 值以及
常见 credential 形态做替换。原始 request、response、traceback 与子进程 stdout/stderr
只允许进入对应 attempt 的私有文件，权限为 `0600`；它们不会复制到全局结构化日志。
所有子进程事件带 `command_index`，未来 Provider 分片事件分别带 `request_id` 与
`artifact_id`，从而可以沿 `run_id → stage → attempt → command/request → artifact`
关联定位。安静子进程的心跳由 Runner 定时产生，不依赖子进程主动输出。

### 15.3 进度心跳

长 Stage 至少每 `logging.progress_interval_seconds` 输出一次：

- 已完成/总数；
- 成功、失败、重试数；
- 当前吞吐和 ETA；
- LLM 请求延迟、token 或费用统计（Provider 可提供时）；
- 训练 step、epoch、loss、学习率；
- GPU/NPU 内存峰值（可获取时）；
- 最近 checkpoint 或 shard 路径。

## 16. 质量门禁

### 16.1 数据门禁

- 候选 ID 唯一；
- 查询与 qrels 全部引用已知候选；
- qrels `position` 与 `query.skill_ids` 顺序一致；
- evidence 是 query 原文片段；
- workflow 和语义查询组不跨 split；
- 每个候选达到配置要求的 Alignment 和 Retrieval 覆盖；
- accepted、rejected、failed 数量可以对账；
- manifest 的行数、文件大小和 SHA-256 一致。

### 16.2 Code 门禁

- CodePlan 容量充足；
- 所有 code 路径长度一致；
- token namespace 唯一；
- 碰撞率和最大 bucket 大小达标；
- 各层利用率和归一化熵达标；
- registry、codes 和候选集合完全一致。

### 16.3 模型门禁

- 虚拟 token 对 tokenizer 是唯一原子 token；
- Memorization code accuracy 达标；
- Retrieval 格式合法率达标；
- Recall@K、Exact Match 等指标达标；
- 所有活跃候选在有效训练监督中至少出现一次；
- 约束解码不会产生 registry 外路径。

### 16.4 导出门禁

- 模型和 tokenizer 能从导出目录独立加载；
- manifest、decode map、virtual tokens 相互一致；
- 权重、配置和 decoder artifact 文件齐全；
- 最小约束解码 smoke test 通过；
- 导出目录不依赖仓库其他文件；
- 所有 artifact lineage 和文件哈希写入导出报告。

指标阈值由 `evaluation.metric_thresholds` 显式给出；空映射表示实验尚未设定准确率下限，
但 `metrics.json` 仍必须存在且格式有效。报告必须区分结构/可加载门禁与业务指标门禁，
不能仅因预测含非空 `paths` 就声称检索质量达标。

`export.smoke_test=false` 允许实验阶段发布结构完整的 bundle，但报告与 manifest 必须将
模型标为 `deployment_qualified=false`、smoke 状态标为 `not_requested`。只有显式启用并
通过“从导出目录加载 tokenizer 与模型”的 smoke test，且其余门禁全部通过时，才能
标记 `deployment_qualified=true`。

默认门禁失败就停止导出。实验需要绕过时必须在创建 Run 时显式传入
`--allow-failed-gates`；已有 Run 应通过
`fork --set export.allow_failed_gates=true` 创建派生实验。模型 manifest 和导出报告
必须写入失败门禁、绕过标记及 `deployment_qualified=false`。绕过仅用于诊断，不得把
模型标记为部署合格。

## 17. 最终输出契约

```text
export/
├── model/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── chat_template.jinja
│   ├── router_manifest.json
│   ├── skill_decode_map.json
│   └── virtual_tokens.txt
└── report/
    ├── run_summary.json
    ├── run_summary.md
    ├── evaluation.json
    ├── failed_examples.jsonl
    ├── quality_gates.json
    └── artifact_lineage.json
```

`router_manifest.json` 至少记录：

- 候选集合哈希；
- 数据集 manifest 哈希；
- embedding、协同图和 codebook 哈希；
- Skill codes、registry 和 virtual tokens 哈希；
- 各训练阶段配置、输入模型和最终 checkpoint；
- 基座模型标识；
- 训练数据及 replay 比例；
- 评估结果和门禁状态；
- 代码版本和 Git commit；
- 最终模型文件哈希。

最终 `export/model` 必须兼容当前服务约定，服务只需要模型目录即可获得模型、
tokenizer、prompt manifest 和候选 decode map。

## 18. 目标代码结构

```text
src/llmgen/pipeline/
├── config.py
├── schema.py
├── runner.py
├── state.py
├── artifacts.py
├── logging.py
├── providers.py
└── stages/
    ├── ingest.py
    ├── enrich.py
    ├── plan_queries.py
    ├── generate_queries.py
    ├── review_queries.py
    ├── finalize_dataset.py
    ├── train_codebook.py
    ├── assign_codes.py
    ├── build_sft.py
    ├── train_router.py
    ├── evaluate.py
    └── export.py

scripts/train_candidates.py
configs/router_pipeline.yaml
```

Phase 1 已落地 `config/schema/runner/state/artifacts/logging`、CLI 和默认配置；为保证首轮
变更不触碰算法，具体 Stage 暂集中在 `stages/legacy.py`，Provider 适配也由该文件把配置
转换为现有脚本参数与子进程环境。Phase 2 再按上图拆分 `providers.py` 和各 Stage 文件，
拆分前后 artifact 名称、状态语义及配置哈希边界保持不变。

职责划分：

- `config.py`：配置加载、合并、校验、Stage 配置投影和哈希；
- `schema.py`：候选、查询、qrels、artifact 和状态 schema；
- `runner.py`：DAG、阶段区间、恢复、失效和锁；
- `state.py`：Run/Stage 状态的原子读写；
- `artifacts.py`：Artifact Registry 和血缘；
- `logging.py`：统一标记、结构化日志和子进程日志；
- `providers.py`：LLM/Embedding Provider、缓存、重试和并发；
- `stages/`：薄 Stage 适配器，复用现有算法函数；
- `scripts/train_candidates.py`：不承载算法逻辑的 CLI。

## 19. 与现有实现的迁移

迁移应保持现有 ClawHub 和 Light 实验可复现。

### 当前过渡边界

- 新入口 `scripts/train_candidates.py` 负责 Run、DAG、状态、哈希、恢复和报告，并通过
  `src/llmgen/pipeline/stages/legacy.py` 调用现有算法入口；
- `scripts/router_pipeline.sh` 与 `scripts/skillret/*.sh` 保持兼容入口身份，它们不会反向
  创建或维护新 Runner 的状态；
- Alignment 与 Retrieval 已拆成可独立调用的 `06a`、`06b` 阶段，原 `06` 脚本保留为
  行为兼容 wrapper，因此新 Runner 不会在 Retrieval 阶段重复覆盖 Alignment；
- 旧数据经过 `finalize-dataset` 的显式迁移与 ordered-qrels 校验后才能进入新 Stage 08，
  下游不能依赖 JSONL 物理顺序推断执行顺序；
- Stage 边界恢复、配置失效传播、artifact 血缘和导出报告只由新入口提供；旧入口仍按
  原有目录和环境变量规则运行；
- 不支持的通用化参数由新配置 schema 拒绝。禁止“接受配置但仍使用旧硬编码值”。

### Phase 1：编排基础设施

实现：

- typed config；
- Run 初始化；
- Artifact Registry；
- Stage State；
- 结构化日志；
- `run`、`stage`、`status`、`from/to` 和 resume CLI。

这一阶段通过适配器调用现有脚本，不修改算法。

### Phase 2：通用数据生成

将现有 `scripts/clawhub_data/` 和 `scripts/light_data/` 的共用逻辑迁入 Python
模块，并完成：

- 通用候选输入；
- embedding/画像复用；
- workflow 计划；
- 请求缓存和分片；
- ordered qrels；
- coverage backfill；
- split 和 dataset manifest。

### Phase 3：动态 CodePlan 与 Router Stage

实现：

- 根据候选数量自动生成 CodePlan；
- 单 token、多 token 和任意固定长度 code 的统一校验；
- Memorization、Alignment、Retrieval 独立 Stage；
- 单卡、多卡和 DeepSpeed 自动资源配置；
- checkpoint 恢复和 artifact lineage。

### Phase 4：评估与统一导出

实现：

- 统一质量门禁；
- 错误样本导出；
- 自包含模型目录；
- 服务兼容 smoke test；
- Run summary 和可复现报告。

迁移完成前，现有 [`scripts/router_pipeline.sh`](../scripts/router_pipeline.sh) 和
`scripts/skillret/*.sh` 保留；新模块把它们作为算法适配入口调用。新流水线稳定后，训练
控制台应把新的 CLI 作为执行边界，而不是复制 Runner 逻辑。

## 20. 测试策略

### 20.1 单元测试

- 候选规范化、ID 生成和重复检查；
- ordered qrels 的集合和顺序校验；
- 配置类型、未知字段和配置投影；
- Artifact Registry 原子更新；
- Stage 输入和配置哈希失效传播；
- CodePlan 对不同候选数量的容量性质；
- JSONL 分片恢复和重复请求消除；
- 日志脱敏和 marker；
- export manifest 一致性。

### 20.2 Mock 端到端测试

使用少量候选和确定性 Mock Provider 验证：

```text
candidates
→ query/qrels
→ mock embedding
→ code assignment
→ SFT
→ mock training
→ evaluation
→ export
```

测试应在无 GPU、无网络环境中完成，并覆盖中途失败、恢复、强制重跑和派生实验。

### 20.3 小模型 smoke test

使用本地微型 HuggingFace CausalLM 和单卡完成真实 tokenizer resize、target-only
训练、checkpoint 恢复、约束生成和模型导出。

### 20.4 回归测试

使用当前 ClawHub 和 Light 数据验证：

- 候选顺序和哈希不变；
- SFT 样本数与现有逻辑一致；
- code artifact 和 registry 契约兼容；
- 旧服务能加载新导出目录；
- 新 qrels 顺序校验不会改变当前合法数据。

## 21. 验收标准

重构完成需同时满足：

1. 仅提供候选 JSONL、Pipeline 配置、LLM/Embedding 服务和基座模型即可完成全流程；
2. 不需要新增数据集专用 Shell 脚本或修改硬编码数据集名称；
3. 每个 Stage 都有可审计的输入、输出、状态、日志和 manifest；
4. 任意 Stage 可以单独执行，缺失依赖时给出准确错误；
5. Pipeline 可以从 Stage 边界和长 Stage 内部 checkpoint 恢复；
6. 修改配置后只使真正受影响的 Stage 及其下游失效；
7. LLM 和 embedding 请求恢复时不会重复执行已经成功的分片；
8. qrels 明确保存并严格校验 Skill 执行顺序；
9. 所有活跃候选都有足够监督并存在于最终 decode map；
10. 最终 `export/model` 不依赖仓库文件，可被当前服务直接加载；
11. Mock 端到端、小模型 smoke test 和现有数据集回归测试全部通过；
12. Run summary 可以从候选输入追溯到最终每个模型文件。

## 22. 实施建议

优先实现 Runner、Stage State、Artifact Registry、typed config 和日志。这些能力决定后续
算法实验是否可恢复、可比较和可追溯。第一阶段不要同时重写数据生成和训练算法；应先
用 Stage 适配器包裹现有入口，建立稳定执行协议，再逐步把核心逻辑从 Shell 脚本迁入
Python 模块。

ordered qrels 是数据格式重构中应最先落地的算法契约。CodePlan 自动化和资源自动配置
可以随后实现，因为它们会使 codebook 及模型训练产物失效，适合在统一 artifact 血缘
建立后再迭代。
