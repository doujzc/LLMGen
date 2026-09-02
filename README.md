# LLMGen：层级 Agent Skill 路由

LLMGen 面向一个给定的候选 Skill 集合，自动生成 closed-set 路由数据，学习每个 Skill 的
短层级编码，并微调 Router 模型根据用户 Query 自回归输出一个或多个 Skill code：

```text
<SK_L1_1><SK_L2_7>
<SK_L1_3><SK_L2_4>
```

当前主入口是 [`scripts/train_candidates.py`](scripts/train_candidates.py)。它把数据生成、
Codebook、Router 训练、评估和模型导出组织为 14 个可单独执行、可恢复的 Stage：

```text
00 ingest
→ 01 enrich
→ 02 plan-queries
→ 03 generate-queries
→ 04 review-queries
→ 05 finalize-dataset
→ 06 train-codebook
→ 07 assign-codes
→ 08 build-sft
→ 09 train-memorization
→ 10 train-alignment
→ 11 train-retrieval
→ 12 evaluate
→ 13 export
```

本文主要说明两件事：如何一条命令运行完整算法，以及如何逐 Stage 单独运行。

## 1. 安装

要求 Python 3.10～3.12。开发和测试环境：

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e '.[train,test]'
```

CUDA 12.4 训练环境可使用仓库锁定的依赖：

```bash
python -m pip install -r requirements/cuda124.txt
python -m pip install --no-build-isolation deepspeed==0.16.4
python -m pip install -e '.[train,test]'
```

确认训练设备：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count())"
```

## 2. 准备候选 Skill

输入是 JSONL，每个非空行是一个候选 Skill：

```jsonl
{"id":"weather","name":"天气查询","description":"查询指定城市和日期的天气"}
{"id":"calendar","name":"日程管理","description":"创建、修改和查询日程","metadata":{"domain_hint":"productivity"}}
```

字段要求：

| 字段 | 必需 | 说明 |
|---|---:|---|
| `id` 或 `skill_id` | 推荐 | 稳定且唯一的候选 ID；默认可从 `name` 生成 |
| `name` | 是 | 面向用户的 Skill 名称 |
| `description` 或 `desc` | 是 | 能力、边界和适用场景的事实来源 |
| `metadata` | 否 | 扩展对象；默认保留但不参与核心路由算法 |

候选文件不能为空，候选 ID 不能重复。单候选默认使用 `alignment_only` 模式，只构造
Alignment、Memorization 和单 Skill 路由数据，不伪造多 Skill Workflow。

可选人工 Alignment 文件同样使用 JSONL：

```jsonl
{"skill_id":"weather","query":"看看杭州明天下不下雨","category":"manual"}
```

它必须在创建 Run 时通过配置项 `data_generation.manual_alignment_path` 指定。

## 3. 配置模型服务和训练模型

Stage 01～05 使用三个 OpenAI-compatible Provider：

| Provider | 用途 | 环境变量 |
|---|---|---|
| generation | 路由画像、Query 和回填数据 | `GENERATION_API_BASE`、`GENERATION_API_KEY`、`GENERATION_MODEL` |
| review | Alignment/Retrieval 独立审核 | `REVIEW_API_BASE`、`REVIEW_API_KEY`、`REVIEW_MODEL` |
| embedding | 候选 Skill Embedding | `EMBEDDING_API_BASE`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL` |

例如：

```bash
export GENERATION_API_BASE=http://127.0.0.1:8000/v1
export GENERATION_API_KEY=EMPTY
export GENERATION_MODEL=Qwen3.7-Plus

export REVIEW_API_BASE=http://127.0.0.1:8000/v1
export REVIEW_API_KEY=EMPTY
export REVIEW_MODEL=GLM-5.2

export EMBEDDING_API_BASE=http://127.0.0.1:8001/v1
export EMBEDDING_API_KEY=EMPTY
export EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

export ROUTER_BASE_MODEL=/models/Qwen3-1.7B
export PIPELINE_DEVICE=cuda
```

`ROUTER_BASE_MODEL` 必须是已经下载并挂载的本地 Hugging Face 模型目录，不能使用
`org/model` 或 `org/model@revision` 形式的远程 ID。流水线会把本地模型文件身份写入
provenance，供训练、评估和导出校验。

默认配置是 [`configs/router_pipeline.yaml`](configs/router_pipeline.yaml)。常用配置组：

| 配置组 | 内容 |
|---|---|
| `input` | 候选规范化和单候选策略 |
| `providers` | generation、review、embedding 服务 |
| `data_generation` | Query 数量、审核、回填、切分和增强 |
| `code` | CodePlan、Codebook 训练和 code 质量门禁 |
| `router` | 基座模型、三阶段课程训练和 replay 比例 |
| `runtime` | Python、设备和分布式运行方式 |
| `checkpointing` | Provider 调度和训练 checkpoint |
| `evaluation` | closed-set 评估及指标门禁 |
| `export` | 最终模型目录和导出门禁 |

## 4. 一条命令运行完整算法

以下命令创建一个新 Run，并从 Stage 00 一直执行到 Stage 13：

```bash
python scripts/train_candidates.py run \
  --candidates /absolute/path/to/candidates.jsonl \
  --config configs/router_pipeline.yaml \
  --output runs/my-router
```

`--output` 必须指向尚不存在的目录。创建 Run 时会冻结：

- 候选输入和可选人工 Alignment；
- 展开环境变量及 `--set` 后的完整配置；
- Python、依赖、设备、Git 和本地基座模型 provenance；
- Stage DAG 和输入 SHA-256。

需要在新 Run 中覆盖配置时，可以重复使用 `--set KEY=YAML_VALUE`：

```bash
python scripts/train_candidates.py run \
  --candidates /absolute/path/to/candidates.jsonl \
  --config configs/router_pipeline.yaml \
  --output runs/my-router-lora \
  --set router.finetune_mode=lora \
  --set router.retrieval.epochs=5 \
  --set 'runtime.devices=[0,1]'
```

如果只需要生成训练数据，不训练 Router，则运行到 Stage 08：

```bash
python scripts/train_candidates.py run \
  --candidates /absolute/path/to/candidates.jsonl \
  --config configs/router_pipeline.yaml \
  --output runs/my-router-data \
  --to build-sft
```

Stage 08 完成后，`dataset/`、Embedding、Skill code 和 Router SFT 数据都已生成。

## 5. 单独运行各个步骤

### 5.1 创建 Run 并只完成第一个步骤

`stage` 命令只能操作已经存在的 Run，不负责创建 Run。若计划手动逐步执行，先创建 Run 并
只运行 `ingest`：

```bash
export RUN_DIR=runs/my-router-step-by-step

python scripts/train_candidates.py run \
  --candidates /absolute/path/to/candidates.jsonl \
  --config configs/router_pipeline.yaml \
  --output "$RUN_DIR" \
  --to ingest
```

### 5.2 按顺序执行 Stage 01～13

每条命令只执行指定 Stage，不会隐式补跑缺失的上游依赖：

```bash
# 01：生成每个 Skill 的路由画像
python scripts/train_candidates.py stage enrich --run-dir "$RUN_DIR"

# 02：确定性规划多 Skill Workflow
python scripts/train_candidates.py stage plan-queries --run-dir "$RUN_DIR"

# 03：生成单 Skill Alignment 和多 Skill Retrieval Query
python scripts/train_candidates.py stage generate-queries --run-dir "$RUN_DIR"

# 04：独立审核 Query，并回填 Alignment/Retrieval 覆盖
python scripts/train_candidates.py stage review-queries --run-dir "$RUN_DIR"

# 05：导出正式 Dataset、ordered qrels、协同图和 Embedding
python scripts/train_candidates.py stage finalize-dataset --run-dir "$RUN_DIR"

# 06：规划编码空间并训练层级 Codebook
python scripts/train_candidates.py stage train-codebook --run-dir "$RUN_DIR"

# 07：为候选分配 Skill code，并执行碰撞率/利用率/熵门禁
python scripts/train_candidates.py stage assign-codes --run-dir "$RUN_DIR"

# 08：构造 Memorization、Alignment 和 Retrieval SFT 数据
python scripts/train_candidates.py stage build-sft --run-dir "$RUN_DIR"

# 09：训练 Skill 文档到 code 的 Memorization 阶段
python scripts/train_candidates.py stage train-memorization --run-dir "$RUN_DIR"

# 10：训练单 Skill Query Alignment 阶段
python scripts/train_candidates.py stage train-alignment --run-dir "$RUN_DIR"

# 11：训练最终多 Skill Retrieval Router
python scripts/train_candidates.py stage train-retrieval --run-dir "$RUN_DIR"

# 12：在 frozen closed-set test split 上评估并执行指标门禁
python scripts/train_candidates.py stage evaluate --run-dir "$RUN_DIR"

# 13：导出自包含 Hugging Face 模型和部署所需 Router 文件
python scripts/train_candidates.py stage export --run-dir "$RUN_DIR"
```

Stage 与主要产物的对应关系：

| Stage | 是否调用模型/训练 | 主要正式产物 |
|---|---|---|
| `ingest` | 否 | frozen candidates、catalog、candidate manifest |
| `enrich` | generation | `skill_profiles.jsonl` |
| `plan-queries` | 否 | `workflows.jsonl` |
| `generate-queries` | generation | Alignment/Retrieval Query 草稿 |
| `review-queries` | generation + review | reviewed Query、Review、回填 Workflow |
| `finalize-dataset` | embedding | Dataset、ordered qrels、processed、Embedding、协同图 |
| `train-codebook` | 训练 | CodePlan、`best.pt` |
| `assign-codes` | 推理/分配 | code registry、decode map 的上游索引、virtual tokens |
| `build-sft` | 否 | 三类 target-only SFT JSONL |
| `train-memorization` | 训练 | Memorization Router checkpoint |
| `train-alignment` | 训练 | Alignment Router checkpoint |
| `train-retrieval` | 训练 | 最终 Retrieval Router checkpoint |
| `evaluate` | 推理 | 指标、质量门禁和预测结果 |
| `export` | 导出 | 自包含模型和完整 lineage report |

## 6. 执行一个 Stage 范围

`run --from/--to` 可以执行闭区间。区间内已经完成且 lineage 匹配的 Stage 会自动复用：

```bash
python scripts/train_candidates.py run \
  --run-dir "$RUN_DIR" \
  --from review-queries \
  --to build-sft
```

从某个失败步骤继续执行到最终导出：

```bash
python scripts/train_candidates.py run \
  --run-dir "$RUN_DIR" \
  --from review-queries
```

查看 Run 和全部 Stage 状态：

```bash
python scripts/train_candidates.py status --run-dir "$RUN_DIR"
```

## 7. 重跑、Checkpoint 和实验分支

### 7.1 强制重跑

强制失效某个 Stage 及其所有下游，然后只重跑该 Stage：

```bash
python scripts/train_candidates.py run \
  --run-dir "$RUN_DIR" \
  --from review-queries \
  --to review-queries \
  --force-stage review-queries
```

旧 attempts 和 Provider ledger 会保留。注意：`--force-stage` 不代表强制重新发送已经在
ledger 中记录为成功的 Provider 请求；相同 request ID 会复用旧响应。

### 7.2 恢复训练 Checkpoint

训练 Stage 会自动选择最新且 lineage 兼容的 checkpoint。也可以显式指定完整路径：

```bash
python scripts/train_candidates.py stage train-retrieval \
  --run-dir "$RUN_DIR" \
  --resume-checkpoint /absolute/path/to/checkpoint-500
```

显式 checkpoint 默认必须带有匹配当前 Run、Stage、输入 artifact、CodePlan 和配置的
lineage sidecar。`--allow-legacy-checkpoint` 只应用于明确接受风险的旧 checkpoint。

### 7.3 修改配置开展新实验

不要直接修改已有 Run 的 `config/pipeline.resolved.yaml`。使用 `fork` 创建派生 Run：

```bash
python scripts/train_candidates.py fork \
  --from-run "$RUN_DIR" \
  --output runs/my-router-exp2 \
  --set router.retrieval.learning_rate=1e-5 \
  --set data_generation.order_variants=2

python scripts/train_candidates.py run --run-dir runs/my-router-exp2
```

只有输入哈希、Stage 配置投影、实现哈希和输出哈希仍兼容的结果会被复用。

## 8. Run 目录和日志

```text
runs/my-router/
├── run_manifest.json
├── artifact_registry.json
├── config/                       # resolved config、输入指纹、环境和 provenance
├── source/                       # 候选及人工 Alignment 的冻结快照
├── stages/
│   ├── 00_ingest/
│   ├── ...
│   └── 13_export/
│       ├── stage_state.json
│       ├── ledger/               # 适用 Stage 的 Provider 请求/响应分片
│       ├── attempts/0001/         # 命令状态、日志、traceback、中间输出
│       └── output/                # 当前发布目录；是否可用以 Registry 为准
├── logs/
│   ├── pipeline.log
│   ├── pipeline.jsonl
│   └── stages/
├── models/                       # Router 三阶段训练目录
└── export/
    ├── model/
    └── report/
```

Stage 间只通过 `artifact_registry.json` 解析正式输入。失败 attempt 的中间结果仍会保留供
诊断，但没有登记到 Registry 的物理文件不能作为下游输入。

最终 `export/model/` 是部署输入，至少包含：

- Hugging Face 模型权重和 `config.json`；
- Tokenizer 文件和 chat template；
- `router_manifest.json`；
- `skill_decode_map.json`；
- `virtual_tokens.txt`。

## 9. 关键文档

- [数据生成算法设计](docs/data-generation-algorithm-design.md)：Stage 00～08 的算法、Schema、审核门禁、覆盖和恢复边界；
- [通用候选训练闭环设计](docs/generic-candidate-pipeline-refactor.md)：Runner、Artifact Registry、checkpoint 和 Stage 09～13；
- [整体算法说明](docs/algorithm.md)：层级编码和 Router 学习原理；
- [训练控制台](training_console/README.md)：浏览器控制台和后台任务；
- [增量候选设计](docs/incremental-candidates.md)：冻结 Codebook 后增删候选。

## 10. 兼容的固定数据集入口

`scripts/router_pipeline.sh` 仍用于仓库已有的 `clawhub` 和 `light` 固定数据集，以及训练
控制台的兼容流程。新候选集应优先使用 `scripts/train_candidates.py`。

```bash
export DATASET=light
export EMBEDDING_MODEL=/models/Qwen3-Embedding-8B
export ROUTER_MODEL=/models/Qwen3-1.7B
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/router_pipeline.sh "$DATASET" paths
bash scripts/router_pipeline.sh "$DATASET" full
```

查看旧入口的全部命令：

```bash
bash scripts/router_pipeline.sh --help
```

固定数据说明见 [ClawHub Training](data/clawhub_training/README.md) 和
[Light](data_light/README.md)。

## 11. 测试

运行完整测试：

```bash
python -m pytest
```

通用闭环的重点回归：

```bash
python -m pytest -q \
  tests/test_pipeline_provider_e2e.py \
  tests/test_pipeline_dataset_regression.py \
  tests/test_pipeline_data_stages.py \
  tests/test_pipeline_training_stage_modules.py \
  tests/test_pipeline_stage_evaluate_export.py \
  tests/test_generic_pipeline.py
```
