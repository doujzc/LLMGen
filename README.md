# LLMGen：ClawHub 全参数训练

本仓库提供基于 1,000 个 ClawHub Agent Skills 的完整生成式路由训练流程。默认配置为：

- `Qwen3-Embedding-8B` 生成 Skill embeddings；
- `Qwen3-1.7B` 作为 Router；
- 两层 `128×128` Skill Code；
- 多 Skill 完整自回归输出，每条 code 占一行；
- 单机 4 卡 DeepSpeed ZeRO-3 全参数训练；
- 训练和测试共享同一套 1,000 Skill 候选集。

模型输出示例：

```text
<SK_L1_1><SK_L2_7>
<SK_L1_3><SK_L2_4>
```

## 1. 克隆仓库

```bash
git clone https://github.com/doujzc/LLMGen.git
cd LLMGen

git rev-parse --short HEAD
```

ClawHub 的 1,000 个候选和 4,200 条多 Skill query 已提交在
`data/clawhub_training/final/`，无需重新爬取或调用模型生成数据。

## 2. 安装训练环境

要求 Python 3.10--3.12。使用 Conda 创建训练环境：

```bash
conda create -n llmgen python=3.10 -y
conda activate llmgen

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements/cuda124.txt
python -m pip install --no-build-isolation deepspeed==0.16.4
python -m pip install -e '.[train,test]'
```

检查 PyTorch、CUDA 和可见 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count())"
```

预期输出类似：

```text
2.6.0+cu124 12.4
4
```

CUDA 12.4 的 PyTorch wheel 可以在支持 CUDA 12.5 的 NVIDIA 驱动上运行。

## 3. 配置模型和训练参数

将模型路径改成目标机器上的实际位置：

```bash
export EMBEDDING_MODEL=/models/Qwen3-Embedding-8B
export ROUTER_MODEL=/models/Qwen3-1.7B

export RUN_DIR=runs/clawhub-qwen3-1.7b-full-v3
export PROCESSED_DIR="$RUN_DIR/processed"
export EMBEDDING_DIR="$RUN_DIR/embeddings"

export ROUTER_FINETUNE_MODE=full
export ROUTER_NUM_GPUS=4
export ROUTER_DEEPSPEED_CONFIG=configs/deepspeed_zero3.json

export ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE=1
export ROUTER_GRADIENT_ACCUMULATION_STEPS=8
export ROUTER_PRECISION=bf16
export ROUTER_GRADIENT_CHECKPOINTING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DEVICE=cuda
```

对应的有效全局 batch size 为：

```text
4 GPUs × micro batch 1 × gradient accumulation 8 = 32
```

ClawHub 默认执行 10 个 Memorization epochs 和 15 个 Retrieval epochs；Retrieval
训练中混入 20% Memorization replay，避免 code 映射被覆盖。

## 4. 启动 Embedding 服务

建议为 vLLM 创建独立 Conda 环境，避免它修改训练环境中的 PyTorch/CUDA 依赖：

```bash
conda create -n llmgen-vllm python=3.10 -y
conda activate llmgen-vllm

cd /path/to/LLMGen
python -m pip install -r requirements/vllm-cu124.txt
```

启动 OpenAI-compatible Embedding 服务：

```bash
MODEL=/models/Qwen3-Embedding-8B \
SERVED_MODEL_NAME=/models/Qwen3-Embedding-8B \
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=4096 \
VLLM="$(which vllm)" \
bash scripts/serve_qwen3_embedding.sh
```

如果单卡显存足够，可以将 `TENSOR_PARALLEL_SIZE` 改成 `1`。

在另一个终端检查服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

## 5. 预处理 ClawHub 数据

回到训练环境：

```bash
conda activate llmgen
cd /path/to/LLMGen

export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

重新执行第 3 节中的环境变量配置，然后运行：

```bash
bash scripts/clawhub_train/01_prepare.sh
```

完成后关闭 vLLM Embedding 服务，释放 GPU。Embedding 模型只用于预处理，不参与
后续 Router 训练。

## 6. 开始全参数训练

一次执行剩余完整流程：

```bash
SKIP_PREPARE=1 bash scripts/run_clawhub_full.sh
```

脚本依次执行：

1. 训练两层层级 Skill Tokenizer；
2. 为 1,000 个 Skills 导出固定 code，并执行碰撞、利用率和熵质量门禁；
3. 构造多 Skill 自回归 SFT 数据；
4. 全参数训练 Memorization 阶段；
5. 从 Memorization 模型继续全参数训练 Retrieval 阶段；
6. 在共享的 1,000 Skill 候选集上评估。

每一步也可以单独运行，便于调试和恢复：

```bash
bash scripts/clawhub_train/02_train_tokenizer.sh
bash scripts/clawhub_train/03_export_codes.sh
bash scripts/clawhub_train/04_build_router_data.sh
bash scripts/clawhub_train/05_train_memorization.sh
bash scripts/clawhub_train/06_train_retrieval.sh
bash scripts/clawhub_train/07_evaluate.sh
```

旧版 `64×64 / clawhub-v2` 的 Stage 1 和 Router checkpoint 不兼容，必须使用新的
`RUN_DIR` 从 Stage 1 重新训练。导出成功后可快速确认 code 质量：

```bash
python -c 'import json,os; x=json.load(open(os.path.join(os.environ["RUN_DIR"],"index/manifest.json"))); print(json.dumps(x["splits"]["train"]["quality_gate"], indent=2))'
```

`passed` 必须为 `true`；否则脚本会在训练 Router 前直接退出。

## 7. 输出目录

主要产物为：

```text
runs/clawhub-qwen3-1.7b-full-v3/
├── processed/
├── embeddings/
├── stage1/
│   └── best.pt
├── index/
│   ├── train_codes.jsonl
│   ├── train_registry.json
│   └── virtual_tokens.txt
├── router/
│   ├── memorization/
│   └── retrieval/
└── evaluation/
    ├── predictions.jsonl
    └── metrics.json
```

确认两个 Router 阶段均使用全参数训练：

```bash
grep -n '"finetune_mode"' \
  "$RUN_DIR/router/memorization/router_manifest.json" \
  "$RUN_DIR/router/retrieval/router_manifest.json"
```

应输出：

```text
"finetune_mode": "full"
```

## 8. Checkpoint 恢复

恢复 Stage 1 Tokenizer：

```bash
export TOKENIZER_RESUME="$RUN_DIR/stage1/last.pt"
bash scripts/clawhub_train/02_train_tokenizer.sh
```

恢复 Memorization：

```bash
export ROUTER_RESUME_MEMORIZATION=latest
bash scripts/clawhub_train/05_train_memorization.sh
```

恢复 Retrieval：

```bash
export ROUTER_RESUME_RETRIEVAL=latest
bash scripts/clawhub_train/06_train_retrieval.sh
```

## 9. 显存不足

默认 `Qwen3-1.7B + 4 GPU + ZeRO-3` 通常比较宽裕。若更换为 Qwen3-4B/8B 后发生
OOM，优先启用 CPU parameter offload：

```bash
export ROUTER_DEEPSPEED_CONFIG=configs/deepspeed_zero3_offload.json
SKIP_PREPARE=1 bash scripts/run_clawhub_full.sh
```

单卡 micro batch 已经是 `1`，继续降低 gradient accumulation 不会减少单步 activation
显存。必要时可以缩短上下文：

```bash
export ROUTER_MAX_LENGTH=768
```

缩短上下文可能截断较长 query，因此应优先使用 ZeRO-3 offload。

## 10. 低召回诊断

先分析数据、code 分布、训练日志和现有预测：

```bash
bash scripts/clawhub_train/08_diagnose.sh
```

再加载最终 checkpoint，对训练集和测试集各抽样 128 条计算 teacher-forced token
准确率：

```bash
DIAG_WITH_MODEL=1 DEVICE=cuda:0 \
  bash scripts/clawhub_train/08_diagnose.sh
```

报告写入 `$RUN_DIR/diagnostics/test.json`。如需判断是否只记住训练数据，再生成训练集预测：

```bash
QUERY_SET=train EVAL_DIR="$RUN_DIR/evaluation-train" \
  bash scripts/clawhub_train/07_evaluate.sh

DIAG_SPLIT=train \
DIAG_PREDICTIONS="$RUN_DIR/evaluation-train/predictions.jsonl" \
DIAG_OUTPUT="$RUN_DIR/diagnostics/train.json" \
  bash scripts/clawhub_train/08_diagnose.sh
```

分别检查 Memorization checkpoint 是否学会 code，以及 Retrieval 后是否遗忘：

```bash
DIAG_SAMPLE_SIZE=256 DEVICE=cuda:0 \
  bash scripts/clawhub_train/09_diagnose_memorization.sh
```

查看两个报告中的 `teacher_forcing.train.categories.code.constrained_accuracy`：
Memorization checkpoint 应达到 95% 左右，Retrieval checkpoint 不应显著下降。

## 11. 测试

```bash
python -m pytest
```

CPU 端到端 smoke：

```bash
bash scripts/run_skillret_smoke.sh
```
