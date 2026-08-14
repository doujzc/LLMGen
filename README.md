# LLMGen：层级 Agent Skill 路由

LLMGen 将固定候选集中的 Agent Skills 编码为短层级 token，并微调 Qwen3 Router
根据用户 query 自回归生成一个或多个 Skill code：

```text
<SK_L1_1><SK_L2_7>
<SK_L1_3><SK_L2_4>
```

## 多轮 Top1 直接路由

项目支持不经过 RQ-VAE/codebook 的直接候选名模式。
输入保留 `user`、`assistant`、`tool` 多轮记录，训练目标仅为
`候选英文名 + EOS`，推理始终返回一个候选。输出空间为两个真实路由
`StockQuery`、`Ecommerce` 和五个虚拟路由 `StockAdvice`、`StockOther`、
`ProductOther`、`ChitChat`、`NoAvailable`。

训练直接读取用户提供的 JSONL，不依赖任何上游数据仓库或转换步骤。设置数据和模型路径
即可进行 4 卡 ZeRO-3 全参训练：

```bash
export ROUTER_MODEL=/models/Qwen3-1.7B
export TOP1_TRAIN_DATA=/data/my_router/train.jsonl
export TOP1_VALIDATION_DATA=/data/my_router/validation.jsonl
bash scripts/top1/01_train.sh
```

评估默认使用每卡 batch 1，并仅保留 loss，避免长多轮样本的全词表 logits 在 eval
开始时造成显存峰值。显存充足时可用 `ROUTER_PER_DEVICE_EVAL_BATCH_SIZE` 调大。

训练启动时会自动将实际使用的标准 system/user/assistant 消息写入：

```bash
$TOP1_RUN_DIR/router/retrieval/sft_input.jsonl
```

每行仅包含 `{"messages":[...]}`，且使用训练时已经加载的 tokenizer 复现历史清洗
和长度裁剪，不需要额外配置。也可在不训练时运行 `bash scripts/top1/export_sft.sh`
单独生成（独立脚本不加载模型 tokenizer，因此不执行 token 长度裁剪）。通用框架
应启用 assistant-only/response-only loss，以保持只拟合候选名和 EOS。

新训练默认使用 `standalone_request_v2` 上下文模板：模型先在内部把末轮还原为
可独立理解的请求，只补充历史中不可缺少的信息；末轮已经完整或切换目标时忽略旧
目标，然后仍只生成候选名。它不使用关键词/正则，也不增加第二次模型调用。模板版本
写入 `router_manifest.json`，推理会自动复现训练格式；旧模型继续使用原模板。

有测试集时执行评测，或者直接运行训练加评测：

```bash
export TOP1_TEST_DATA=/data/my_router/test.jsonl
bash scripts/top1/02_evaluate.sh
bash scripts/top1/02_evaluate.sh --route-threshold 0.6  # 低置信度真实路由改为无候选
bash scripts/top1/full.sh
```

`--route-threshold` 取值为 `[0, 1]`，比较的是合法候选名称整条生成路径的归一化
概率；不传时保持原有行为。阈值只拦截 `StockQuery`、`Ecommerce` 等真实路由，
不会改变模型直接生成的虚拟无路由候选。预测文件同时保留原始候选、置信度和是否触发
阈值，指标文件额外报告路由覆盖率、拒绝率和路由决策准确性。

可用标签文件绘制正、负样本准确率随阈值变化的曲线。标签按预测结果的行序对齐，
每行是 intent label（JSON 字符串或裸字符串），负样本写 `null`：

```bash
python scripts/top1/visualize_threshold.py \
  --predictions "$TOP1_RUN_DIR/evaluation/predictions.jsonl" \
  --labels /data/my_router/test.labels.jsonl \
  --output "$TOP1_RUN_DIR/evaluation/threshold.html"
```

非 `null` 标签是正样本，要求阈值处理后的 intent 与标签完全一致；`null` 标签是负样本，
要求处理后不路由。只有原始 intent 非空且 `candidate_confidence >= threshold` 才保留路由。
生成的 HTML 内嵌 Plotly.js，可离线打开并悬停查看同一阈值的两项准确率，也支持缩放和
导出图片。脚本同时在同目录写出 `threshold.json`，保存每个阈值的曲线数值。

单条和多轮推理分别使用：

```bash
python scripts/infer_candidate_router.py \
  --model-name-or-path runs/qwen3-1.7b-top1/router/retrieval \
  --query "看看贵州茅台今天的走势" \
  --output-jsonl /tmp/prediction.jsonl

python scripts/infer_candidate_router.py \
  --model-name-or-path runs/qwen3-1.7b-top1/router/retrieval \
  --messages-json /path/to/conversation.json \
  --output-jsonl /tmp/prediction.jsonl
```

`conversation.json` 可以是消息数组，也可以是 `{"messages": [...]}`。训练时不会新增
候选 code token，也不会调整模型词表；全参/LoRA、DeepSpeed 和 checkpoint 恢复仍复用
原训练器。

训练、评估、解码和 Web 调试始终使用同一候选集。目前提供两套闭集数据：

| 数据集 | 候选数 | 默认 codebook | 配置 | 默认运行目录 |
|---|---:|---:|---|---|
| `clawhub` | 1,000 | `128×128` | `configs/clawhub.env` | `runs/clawhub-top1000-qwen3-1.7b-full-v1` |
| `light` | 301 | `32×16` | `configs/light.env` | `runs/light301-qwen3-1.7b-full-v3` |

数据说明见
[ClawHub Training](data/clawhub_training/README.md) 和
[Light](data_light/README.md)。

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
HELDOUT_CSV=/path/to/result.csv bash scripts/run_0804_data.sh
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

python scripts/0804_data/05_audit_final.py \
  --dataset-dir data_0804/final \
  --candidates candidates_0804.jsonl \
  --heldout /path/to/result.csv \
  --distribution-profile data_0804/distribution_profile.json
```

0804 流程默认读取 `~/deepseek_api_key.txt`：`deepseek-v4-flash` 负责生成，独立严格
复核默认使用 `deepseek-reasoner`。复核会拒绝仅用“并/再”连接的无关 Skill，并按
候选覆盖缺口自动生成训练专用回填 workflow。生成与复核均可断点续跑；复核默认每
500 条落盘。held-out CSV 只用于不含原句/Skill ID 的聚合分布和本地泄露门禁，
不会进入模型 prompt 或训练文件。

同一个 `WORK_DIR` 重跑会复用 workflow 和 API 结果。如修改候选、held-out 或种子
数据，使用 `REBUILD_STATIC=1` 并指定新的 `WORK_DIR`，避免混用旧缓存。

数据格式、统计、来源和限制见各数据目录 README。

## 测试

```bash
python -m pytest
bash scripts/run_skillret_smoke.sh
```
