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
| `light` | 301 | `32×16` | `configs/light.env` | `runs/light301-qwen3-1.7b-full-v2` |

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

激活安装了 LLMGen 的 Conda 环境后，可以通过独立开发者控制台管理参数和不可变
配置版本：

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
切换到“批量 TXT”后，每个非空行会作为一个 Query 分批推理，结果可以逐条检查
并下载为 JSONL。

## 显存不足

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
