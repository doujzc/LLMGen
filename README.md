# LLMGen

面向 Agent Skills 的低时延生成式召回：模型每条候选只生成可配置的 `L` 个层级
special tokens，再由 collision bucket 展开为多个 skills。

项目同时保留可解释 taxonomy tokenizer 与学习式平衡 tokenizer；SkillRet 完整训练
链路使用后者。默认通过 OpenAI-compatible API 调用 `Qwen3-Embedding-8B`，Router
使用 `Qwen3-1.7B`。训练所需的 ToolWeaver RQ-VAE 已固定版本并内置在仓库中。
设计见 [docs/design.md](docs/design.md)，上游来源见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 安装与数据

要求 Python 3.10+。

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[train,test]'
.venv/bin/python scripts/download_skillret.py
```

SkillRet 固定为 `ThakiCloud/SKILLRET@7cae7cf`；当前快照已下载到
`data/skillret/`，11 个文件的校验值见 `data/skillret/SHA256SUMS`。

## 训练与推理

先启动提供 `/v1/embeddings` 的服务；仓库给出了 vLLM 启动脚本：

```bash
# 在独立服务环境安装 vLLM；8B 可用 TENSOR_PARALLEL_SIZE 调整卡数
python -m pip install vllm
bash scripts/serve_qwen3_embedding.sh
```

完整流程默认使用 `Qwen/Qwen3-Embedding-8B`、`L=3, K=64`、末层 Sinkhorn、
`Qwen/Qwen3-1.7B` LoRA：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
bash scripts/run_skillret_full.sh
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

Stage-2 的 `train_router.py` 与 `infer_router.py` 完整参数已封装在 full script；前者支持
Qwen3 系列的 full/LoRA、DeepSpeed 和 checkpoint resume，后者执行固定 `L` 的 trie-constrained
beam search，并输出 NDCG、Recall、MAP、MRR 与 Completeness。
同时报告不受 bucket 内部同分顺序影响的 code recall 与 bucket-expanded recall。

CPU 端到端验收与测试：

```bash
bash scripts/run_skillret_smoke.sh
.venv/bin/pytest
```
