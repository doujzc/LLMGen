# LLMGen

面向 Agent Skills 的低时延生成式召回：模型每条候选只生成可配置的 `L` 个层级
special tokens，再由 collision bucket 展开为多个 skills。

项目同时保留可解释 taxonomy tokenizer 与学习式平衡 tokenizer；SkillRet 完整训练
链路使用后者。默认通过 OpenAI-compatible API 调用 `Qwen3-Embedding-8B`，Router
使用 `Qwen3-1.7B`。训练所需的 ToolWeaver RQ-VAE 已固定版本并内置在仓库中。
设计见 [docs/design.md](docs/design.md)，上游来源见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 安装与数据

要求 Python 3.10--3.12。若 `nvidia-smi` 显示最高 CUDA 12.5/12.6，使用
CUDA 12.4 wheel，避免安装需要更新 NVIDIA 驱动的 CUDA 12.8/13.x 版本：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements/cuda124.txt
.venv/bin/pip install --no-build-isolation deepspeed==0.16.4
.venv/bin/pip install -e '.[train,test]'
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)'
.venv/bin/python scripts/download_skillret.py
```

版本检查应输出 `2.6.0+cu124 12.4`。DeepSpeed 固定为 `0.16.4`，以兼容 ZeRO-3、
PEFT `modules_to_save` 和新增 code-token embeddings；在已有 PyTorch 的环境中构建，
避免隔离构建重新解析 PyTorch。已有环境若曾安装其它 CUDA wheel，建议删除 `.venv`
后按上述命令重建，不要只覆盖安装 `torch`。

SkillRet 固定为 `ThakiCloud/SKILLRET@7cae7cf`；当前快照已下载到
`data/skillret/`，11 个文件的校验值见 `data/skillret/SHA256SUMS`。

## 训练与推理

先启动提供 `/v1/embeddings` 的服务；仓库给出了 vLLM 启动脚本：

```bash
# vLLM 使用独立环境，避免它改写训练环境中的 PyTorch/CUDA 依赖
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -r requirements/vllm-cu124.txt
VLLM=.venv-vllm/bin/vllm bash scripts/serve_qwen3_embedding.sh
```

这里固定 `vLLM==0.8.5.post1`、`torch==2.6.0+cu124`；8B 模型可用
`TENSOR_PARALLEL_SIZE` 调整卡数。

完整流程默认使用 `Qwen/Qwen3-Embedding-8B`、`L=3, K=64`、末层 Sinkhorn、
`Qwen/Qwen3-1.7B` LoRA。Stage 2 默认用单机 4 卡 DeepSpeed ZeRO-3，分片
参数、梯度和优化器状态；每卡 batch 1、梯度累积 8，global batch 32：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
bash scripts/run_skillret_full.sh
```

单卡 Stage 2：

```bash
ROUTER_NUM_GPUS=1 ROUTER_DEEPSPEED_CONFIG=none \
  ROUTER_GRADIENT_ACCUMULATION_STEPS=32 \
  bash scripts/run_skillret_full.sh
```

显存仍不足时将参数 offload 到 CPU；如需恢复普通 DDP，设置配置为 `none`：

```bash
ROUTER_DEEPSPEED_CONFIG=configs/deepspeed_zero3_offload.json \
  bash scripts/run_skillret_full.sh

ROUTER_DEEPSPEED_CONFIG=none bash scripts/run_skillret_full.sh
```

全参数 SFT：

```bash
ROUTER_FINETUNE_MODE=full \
  bash scripts/run_skillret_full.sh
```

其它 Qwen3 Causal LM 与可裁剪 embedding 维度：

```bash
ROUTER_MODEL=Qwen/Qwen3-4B EMBEDDING_DIMENSIONS=1024 \
  bash scripts/run_skillret_full.sh
```

若 embedding 服务和训练共享 GPU，先单独执行 `prepare_skillret.py`，停止服务释放
显存，再用 `SKIP_PREPARE=1` 运行 full script。

Qwen3 官方小尺寸型号是 `1.7B`，没有 `1.5B`；如需严格的 1.5B 模型，可通过
`ROUTER_MODEL` 指向其它 `AutoModelForCausalLM` 兼容模型。

各阶段可独立执行和恢复：

```bash
.venv/bin/python scripts/prepare_skillret.py \
  --embedding-provider openai \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --embedding-base-url http://127.0.0.1:8000/v1 --batch-size 8
.venv/bin/python scripts/train_tokenizer.py --data-root data/skillret \
  --output-dir runs/skillret/stage1 \
  --num-levels 3 --branching-factors 64 64 64 --sk-epsilons 0 0 0.01 \
  --epochs 100 --batch-size 512 --device cuda --amp-dtype bf16
.venv/bin/python scripts/export_skill_codes.py \
  --checkpoint runs/skillret/stage1/best.pt --output-dir runs/skillret/index \
  --device cuda
.venv/bin/python scripts/build_router_data.py \
  --catalog data/skillret/processed/catalog_train.jsonl \
  --queries data/skillret/processed/queries_train.jsonl \
  --qrels data/skillret/processed/qrels_train.jsonl \
  --codes runs/skillret/index/train_codes.jsonl \
  --virtual-tokens runs/skillret/index/virtual_tokens.txt \
  --output-dir runs/skillret/router_data
```

Stage-2 的 `train_router.py` 与 `infer_router.py` 完整参数已封装在 full script；前者将
memorization/retrieval 作为两个独立的 ZeRO-3 launch，支持 Qwen3 系列的 full/LoRA 和
checkpoint resume；DeepSpeed 下的 activation checkpoint 默认使用完整重计算的
reentrant 实现，避免重计算时读取到 ZeRO-3 的零尺寸参数占位。后者执行固定 `L` 的
trie-constrained beam search，并输出 NDCG、Recall、MAP、MRR 与 Completeness。
同时报告不受 bucket 内部同分顺序影响的 code recall 与 bucket-expanded recall。

使用 train skills 作为共享候选库，并在训练时留出的 2% train-query groups 上做
closed-set 验证：

```bash
bash scripts/eval_skillret_closedset.sh
```

仅用于检查训练集拟合程度时运行：

```bash
QUERY_SET=train bash scripts/eval_skillret_closedset.sh
```

验证结果分别写入
`runs/skillret/evaluation/closedset-validation/` 和 `closedset-train/`；官方 disjoint
test-skills 评估仍由 full script 保留。

CPU 端到端验收与测试：

```bash
bash scripts/run_skillret_smoke.sh
.venv/bin/pytest
```
