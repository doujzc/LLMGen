# LLMGen：层级 Agent Skill 路由

LLMGen 将固定候选集中的 Agent Skills 编码为短层级 token，并微调 Qwen3 Router
根据用户 query 自回归生成一个或多个 Skill code：

```text
<SK_L1_1><SK_L2_7>
<SK_L1_3><SK_L2_4>
```

训练、评估、解码和 Web 调试始终使用同一候选集。目前提供两套闭集数据：

| 数据集 | 候选数 | 默认 codebook | 配置 | 默认运行目录 |
|---|---:|---:|---|---|
| `clawhub` | 1,000 | `128×128` | `configs/clawhub.env` | `runs/clawhub-top1000-qwen3-1.7b-full-v1` |
| `light` | 301 | `32×16` | `configs/light.env` | `runs/light301-qwen3-1.7b-full-v3` |

数据说明见
[ClawHub Training](data/clawhub_training/README.md) 和
[Light](data_light/README.md)。

## 任意候选集训练闭环

通用入口 [`scripts/train_candidates.py`](scripts/train_candidates.py) 接收候选 JSONL，
依次执行 qrels 构造、层级 code 学习、Router 课程训练、约束评估和自包含模型导出。
默认 DAG 包含 14 个可恢复 Stage：

```text
ingest → enrich → plan-queries → generate-queries → review-queries
→ finalize-dataset → train-codebook → assign-codes → build-sft
→ train-memorization → train-alignment → train-retrieval → evaluate → export
```

每行候选必须是 JSON 对象，`name` 和 `description` 必填；`id` 推荐提供，默认
`input.id_policy=explicit_or_name` 时可由名称稳定生成；`metadata` 可选：

```jsonl
{"id":"weather","name":"天气查询","description":"查询指定城市的天气","metadata":{"domain":"utility"}}
{"id":"calendar","name":"日程管理","description":"创建、修改和查询日程"}
```

候选 ID 必须唯一，文件不能为空。默认允许单候选输入，并以 `alignment_only` 模式跳过
无意义的多 Skill Retrieval；可用 `input.single_candidate_policy=error` 改为拒绝。

先配置 OpenAI-compatible 的生成、审核和 embedding 服务以及本地基座模型，再创建一个
全新的 Run 目录（`--output` 指向的目录不能已存在）：

```bash
export GENERATION_API_BASE=http://127.0.0.1:8000/v1
export REVIEW_API_BASE=http://127.0.0.1:8000/v1
export EMBEDDING_API_BASE=http://127.0.0.1:8001/v1
export GENERATION_API_KEY=EMPTY
export REVIEW_API_KEY=EMPTY
export EMBEDDING_API_KEY=EMPTY

python scripts/train_candidates.py run \
  --candidates /path/to/candidates.jsonl \
  --config configs/router_pipeline.yaml \
  --output runs/my-router \
  --set router.base_model=/models/Qwen3-1.7B
```

`router.base_model` 必须是已经物化并挂载的本地 Hugging Face 模型目录。流水线拒绝
`org/model` 和 `org/model@revision` 形式的远程模型 ID，避免 provenance 冻结的版本与
训练、评估或导出实际加载的版本不一致。

Run 创建时会冻结候选输入、resolved config、代码/依赖/设备 provenance。常用恢复和
单阶段命令如下：

```bash
# 查看 Run 和所有 Stage 状态
python scripts/train_candidates.py status --run-dir runs/my-router

# 从失败 Stage 继续；已完成且 lineage 匹配的 Stage 自动复用
python scripts/train_candidates.py run \
  --run-dir runs/my-router --from review-queries

# 只执行一个 Stage；不会隐式补跑缺失的上游依赖
python scripts/train_candidates.py stage evaluate --run-dir runs/my-router

# 执行闭区间，或强制失效并重跑某 Stage 及其下游
python scripts/train_candidates.py run \
  --run-dir runs/my-router --from train-codebook --to build-sft
python scripts/train_candidates.py run \
  --run-dir runs/my-router --from assign-codes --force-stage assign-codes

# 训练 Stage 会自动选择最新的 lineage 兼容 checkpoint，也可显式指定完整路径
python scripts/train_candidates.py stage train-retrieval \
  --run-dir runs/my-router \
  --resume-checkpoint /abs/path/to/checkpoint-500
```

不要直接修改已有 Run 的 resolved config。调整实验参数时使用 `fork`；Runner 只复用输入
哈希、Stage 配置投影和输出哈希都兼容的 artifact：

```bash
python scripts/train_candidates.py fork \
  --from-run runs/my-router \
  --output runs/my-router-lr-1e-5 \
  --set router.retrieval.learning_rate=1e-5
```

关键产物布局为：

```text
runs/my-router/
├── run_manifest.json
├── artifact_registry.json
├── config/                       # 配置、输入指纹、环境与 provenance
├── source/                       # 候选及可选人工对齐输入的冻结快照
├── stages/00_ingest/ ... 13_export/
│   ├── stage_state.json
│   ├── attempts/0001/            # 命令状态/PID、子进程日志、traceback、输出
│   ├── ledger/                   # 适用 Stage 的 LLM/embedding 不可变分片
│   └── output/                   # 最近一次成功 attempt 的正式输出
├── logs/                         # marker 日志和逐 Stage 日志
└── export/
    ├── model/                    # 自包含 HuggingFace + Router 解码文件
    └── report/                   # 质量门禁、评估、文件哈希和完整 lineage
```

`export/model/` 是部署输入；其中包含模型、tokenizer、`router_manifest.json`、
`skill_decode_map.json` 和 `virtual_tokens.txt`。实现与验收细节见
[任意候选 Skill 训练闭环重构设计](docs/generic-candidate-pipeline-refactor.md)。

通用闭环的重点验收可单独运行：

```bash
python -m pytest -q \
  tests/test_pipeline_provider_e2e.py \
  tests/test_pipeline_dataset_regression.py \
  tests/test_pipeline_data_stages.py \
  tests/test_pipeline_training_stage_modules.py \
  tests/test_pipeline_stage_evaluate_export.py \
  tests/test_generic_pipeline.py
```

## 安装

要求 Python 3.10--3.12。CUDA 12.5 驱动可运行项目使用的 CUDA 12.4 PyTorch
wheel。

```bash
conda create -n llmgen python=3.10 -y
conda activate llmgen

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements/cuda124.txt
python -m pip install --no-build-isolation deepspeed==0.16.4
python -m pip install -e '.[train,test]'

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count())"
```

## 配置

选择数据集并设置目标机器上的模型路径：

```bash
export DATASET=clawhub  # 或 light
export EMBEDDING_MODEL=/models/Qwen3-Embedding-8B
export ROUTER_MODEL=/models/Qwen3-1.7B
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DEVICE=cuda
```

默认使用 4 卡、DeepSpeed ZeRO-3 和全参数微调：

```bash
export ROUTER_NUM_GPUS=4
export ROUTER_FINETUNE_MODE=full
export ROUTER_DEEPSPEED_CONFIG=configs/deepspeed_zero3.json
```

三卡训练只需改为：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
export ROUTER_NUM_GPUS=3
```

LoRA 训练设置：

```bash
export ROUTER_FINETUNE_MODE=lora
```

所有配置都可被同名环境变量覆盖。自定义运行目录时设置 `RUN_DIR`；其他产物路径
默认随之派生：

```bash
export RUN_DIR=runs/my-run
bash scripts/router_pipeline.sh "$DATASET" paths
```

每次训练或恢复前都建议用 `paths` 核对实际生效的数据、checkpoint 和输出目录。

## 训练控制台

激活安装了 LLMGen 的 Conda 环境后，可以通过独立开发者控制台管理参数和可编辑
配置：

```bash
conda activate llmgen
bash scripts/serve_training_console.sh
```

默认地址为 `http://127.0.0.1:8090`。控制台通过既有
`scripts/router_pipeline.sh` 提交 detached 任务；关闭浏览器或控制台服务不会
终止已经启动的训练。完整操作和远程访问方式见
[训练控制台说明](training_console/README.md)，架构边界见
[设计文档](docs/training-console/design.md)。

## Embedding 预处理

预处理通过 OpenAI-compatible API 调用 Embedding 模型。建议在独立 Conda 环境中
启动 vLLM，避免改变训练环境的 PyTorch 依赖：

```bash
conda create -n llmgen-vllm python=3.10 -y
conda activate llmgen-vllm
python -m pip install -r requirements/vllm-cu124.txt

MODEL=/models/Qwen3-Embedding-8B \
SERVED_MODEL_NAME=/models/Qwen3-Embedding-8B \
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=4096 \
VLLM="$(which vllm)" \
bash scripts/serve_qwen3_embedding.sh
```

在训练环境的另一个终端执行：

```bash
conda activate llmgen
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY

bash scripts/router_pipeline.sh "$DATASET" prepare
```

预处理完成后可关闭 Embedding 服务，释放 GPU。已有匹配的 `processed/` 和
`embeddings/` 时无需重复执行。

## 训练与评估

从 Stage 1 开始执行完整流程：

```bash
SKIP_PREPARE=1 bash scripts/router_pipeline.sh "$DATASET" full
```

完整流程依次执行层级 Tokenizer、code 导出与质量门禁、Router 数据构造、
Memorization、单 Skill Retrieval Alignment、多 Skill Retrieval 和闭集评估。
`light` 的扩充版数据默认使用 10 个 Memorization epochs、1 个 Alignment epoch
和 1 个 Retrieval epoch；`clawhub` 的默认值由对应配置文件定义。
闭集 Retrieval 默认按 `80%` 多 Skill、`15%` Alignment、`5%` Memorization
混合；后两项可分别用 `ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION` 和
`ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION` 覆盖。

分阶段调试：

```bash
bash scripts/router_pipeline.sh "$DATASET" train-tokenizer
bash scripts/router_pipeline.sh "$DATASET" export-codes
bash scripts/router_pipeline.sh "$DATASET" build-router-data
bash scripts/router_pipeline.sh "$DATASET" train-memorization
bash scripts/router_pipeline.sh "$DATASET" train-retrieval
bash scripts/router_pipeline.sh "$DATASET" evaluate
```

Stage 参数可以直接追加到统一入口，例如：

```bash
bash scripts/router_pipeline.sh "$DATASET" train-tokenizer \
  --no-edge-aware-batches
```

`export-codes` 会在 Router 训练前检查碰撞率、码本利用率和熵；质量门禁失败时不应
直接放宽阈值继续训练。

查看全部命令：

```bash
bash scripts/router_pipeline.sh --help
```

## Checkpoint 恢复

以下示例假设 `RUN_DIR` 已设置为当前运行目录。Stage 1 只能在训练配置兼容时恢复：

```bash
export TOKENIZER_RESUME="$RUN_DIR/stage1/last.pt"
bash scripts/router_pipeline.sh "$DATASET" train-tokenizer
```

恢复 Router：

```bash
export ROUTER_RESUME_MEMORIZATION=latest
bash scripts/router_pipeline.sh "$DATASET" train-memorization

export ROUTER_RESUME_ALIGNMENT=latest       # 仅恢复单 Skill Alignment
export ROUTER_RESUME_RETRIEVAL=latest       # 恢复多 Skill Retrieval
bash scripts/router_pipeline.sh "$DATASET" train-retrieval
```

## 主要产物

```text
<RUN_DIR>/
├── processed/
├── embeddings/
├── stage1/
│   ├── best.pt
│   ├── last.pt
│   └── summary.json
├── index/
│   ├── manifest.json
│   ├── train_codes.jsonl
│   ├── train_registry.json
│   └── virtual_tokens.txt
├── router_data/
├── router/
│   ├── memorization/
│   ├── retrieval_alignment/
│   └── retrieval/
├── evaluation/
└── diagnostics/
```

Router 模型目录同步保存 `skill_decode_map.json` 和 `virtual_tokens.txt`，用于将生成
token 解码回原始 Skill。

## 诊断

分析数据覆盖、code 分布、训练记录和现有预测：

```bash
bash scripts/router_pipeline.sh "$DATASET" diagnose
```

加载模型计算 teacher-forced code 准确率：

```bash
DIAG_WITH_MODEL=1 DEVICE=cuda:0 \
  bash scripts/router_pipeline.sh "$DATASET" diagnose
```

比较 Memorization 与 Retrieval checkpoint，检查是否发生遗忘：

```bash
DIAG_SAMPLE_SIZE=256 DEVICE=cuda:0 \
  bash scripts/router_pipeline.sh "$DATASET" diagnose-memorization
```

报告默认写入 `$RUN_DIR/diagnostics/`。

## 导出与 Web 调试

导出最终 Retrieval 模型：

```bash
bash scripts/router_pipeline.sh "$DATASET" export-web
```

也可直接导出训练中的 Retrieval checkpoint：

```bash
bash scripts/router_pipeline.sh "$DATASET" export-web \
  "$RUN_DIR/router/retrieval/checkpoint-500"
```

训练控制台中选择 `10 · export-web` 后，在“待导出模型目录”填写上述任一
目录即可；checkpoint 的输出目录、Tokenizer 来源和模板 Manifest 可在高级
设置中覆盖。最终模型导出沿用其 `router_manifest.json` 中记录的 Replay 配比。

启动人工测试界面：

```bash
bash scripts/router_pipeline.sh "$DATASET" web \
  --device cuda:0 \
  --dtype bfloat16
```

默认地址为 `http://127.0.0.1:8080`。远程机器可使用：

```bash
ssh -L 8080:127.0.0.1:8080 user@server
```

界面默认使用 Greedy 自回归生成多条 Skill code。切换为 Beam Search 后只生成
一条固定长度 code，并将 Beam 宽度 K 对应的前 K 个 code 作为检索候选；
`Skill 候选 Top K` 再限制 code 碰撞桶展开后的 Skill 数量。
`Greedy + Beam 补全` 会优先保留 Greedy 的候选；仅在数量不足时执行单行
Beam Search，去重追加到 `Skill 候选 Top K` 或耗尽配置的 Beam 宽度。
切换到“批量 TXT”后，每个非空行会作为一个 Query 分批推理，结果可以逐条检查
并下载为 JSONL。

## 候选增删基线

增量更新冻结 Stage 1 codebook，已有 Skill code 不变。候选状态是小型 overlay，
不复制模型权重。以下变量只需指向现有产物和一个新 Skill JSON：

```bash
export SOURCE_ROUTER=runs/my-run/router/retrieval
export STAGE1_CHECKPOINT=runs/my-run/stage1/best.pt
export SKILL_JSON=/path/to/new_skill.json
export INC_ROOT=runs/my-run/incremental/new-skill
```

仅计算新 code 并启用对应 trie path：

```bash
python scripts/incremental/01_add_candidate.py \
  --source-state-dir "$SOURCE_ROUTER" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --skill "$SKILL_JSON" \
  --output-dir "$INC_ROOT/state" \
  --update-mode index_only

WEB_MODEL_DIR="$SOURCE_ROUTER" \
WEB_CANDIDATE_STATE_DIR="$INC_ROOT/state" \
  bash scripts/router_pipeline.sh "$DATASET" web
```

对新增 Skill 做增量 LoRA（只训练 1 条 Memorization 和默认 10 条 query）：

```bash
python scripts/incremental/01_add_candidate.py \
  --source-state-dir "$SOURCE_ROUTER" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --skill "$SKILL_JSON" \
  --output-dir "$INC_ROOT/state" \
  --update-mode lora_train

python scripts/incremental/02_build_training_data.py \
  --candidate-state-dir "$INC_ROOT/state" \
  --output-dir "$INC_ROOT/data"

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/incremental/03_train_lora.sh \
  "$SOURCE_ROUTER" "$INC_ROOT/state" "$INC_ROOT/data" "$INC_ROOT/router"

python -m web_server.server \
  --model-dir "$INC_ROOT/router/retrieval" \
  --device cuda:0 --dtype bfloat16
```

`02_build_training_data.py` 默认用 `~/llm_api.txt` 和 `Qwen3.6-Plus` 生成 query；
也可传 `--queries-txt queries.txt`，每个非空行一条。源 Router 若是 PEFT adapter，
可用 `INCREMENTAL_BASE_MODEL=/models/Qwen3-1.7B` 覆盖其 base model 路径。
用一份独立的 held-out TXT 对 index-only 或 LoRA 模型批量验证：

```bash
python scripts/infer_router.py \
  --model-name-or-path "$INC_ROOT/router/retrieval" \
  --candidate-state-dir "$INC_ROOT/router/retrieval" \
  --query-txt heldout.txt \
  --output-jsonl "$INC_ROOT/heldout.predictions.jsonl" \
  --device cuda:0 --dtype bfloat16
```

验证 index-only 时，分别把上面的模型目录和候选目录改为
`$SOURCE_ROUTER`、`$INC_ROOT/state`。

删除候选只更新 active decode map 和 trie。若该 code 是碰撞桶，默认仅移除目标
Skill 并保留同 path 的其他成员；显式传 `--disable-shared-path` 才会删除整个桶：

```bash
python scripts/incremental/01_remove_candidate.py \
  --source-state-dir "$SOURCE_ROUTER" \
  --skill-id '@owner/skill' \
  --output-dir "$INC_ROOT/removed-state"
```

后续连续增删时，把上一次的 `state` 目录作为新的 `--source-state-dir`。详细约束见
[增量候选设计](docs/incremental-candidates.md)。

## 显存不足

`CUDA_VISIBLE_DEVICES` 会让 PyTorch 重新编号设备。例如配置物理 GPU `4,6` 后，
OOM 中的 `GPU 0` 指物理 GPU 4，并不表示训练跑到了物理 0 号卡。通过训练控制台
启动时，先在运行监控中核对 `cuda:N -> nvidia-smi GPU N -> UUID` 和“最后观测”
记录；映射正确时，OOM 属于所选卡上的真实显存压力。

优先启用 ZeRO-3 CPU parameter offload：

```bash
export ROUTER_DEEPSPEED_CONFIG=configs/deepspeed_zero3_offload.json
```

仍然 OOM 时再缩短 Router 上下文：

```bash
export ROUTER_MAX_LENGTH=768
```

## 数据生成

各数据目录 README 只描述数据本身；下载、生成和校验操作统一记录在这里。

获取或更新原始 ClawHub Top-1,000 快照：

```bash
python scripts/download_clawhub_skills.py
```

下载并校验固定版本的官方 SkillRet 快照：

```bash
python scripts/download_skillret.py
```

仓库已经包含可训练的最终数据，不需要重新调用生成模型。需要重新构建时分别使用：

```bash
bash scripts/run_clawhub_data.sh
bash scripts/run_light_data.sh
```

生成接口默认从 `~/llm_api.txt` 读取 OpenAI-compatible 配置。独立校验当前最终
数据：

```bash
python scripts/clawhub_data/05_validate_dataset.py \
  --dataset-dir data/clawhub_training/final \
  --expected-candidates 1000

python scripts/clawhub_data/05_validate_dataset.py \
  --dataset-dir data_light/final \
  --expected-candidates 301
```

数据格式、统计、来源和限制见各数据目录 README。

## 测试

```bash
python -m pytest
bash scripts/run_skillret_smoke.sh
```
