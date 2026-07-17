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

所有阶段共享 [configs/skillret.env](configs/skillret.env)。在新机器上通常只需修改文件
开头的 `EMBEDDING_MODEL` 和 `ROUTER_MODEL` 两个模型路径；数据路径、训练超参数、
DeepSpeed 和评估参数均已有默认值，也可通过同名环境变量临时覆盖。
脚本会优先使用仓库的 `.venv`，未找到时使用当前激活的 Conda 环境中的 `python`。

先启动提供 `/v1/embeddings` 的服务：

```bash
# vLLM 使用独立环境，避免它改写训练环境中的 PyTorch/CUDA 依赖
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -r requirements/vllm-cu124.txt
VLLM=.venv-vllm/bin/vllm bash scripts/serve_qwen3_embedding.sh
```

这里固定 `vLLM==0.8.5.post1`、`torch==2.6.0+cu124`；8B 模型可用
`TENSOR_PARALLEL_SIZE` 调整卡数。

完整流程默认使用 `L=3, K=64`、末层 Sinkhorn、LoRA 和单机 4 卡 DeepSpeed
ZeRO-3。Memorization 使用全部 train skills，Retrieval 留出 2% train queries；训练和
验证共用 train-skills 候选库。总入口只负责依次调用下方的阶段脚本：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
bash scripts/run_skillret_full.sh
```

每个步骤均可独立重复执行；所有脚本从仓库根目录或其它目录调用均可：

| 步骤 | 脚本 | 主要输出 |
|---:|---|---|
| 00 | `scripts/skillret/00_download.sh` | `data/skillret/` |
| 01 | `scripts/skillret/01_prepare.sh` | `processed/`、`embeddings/` |
| 02 | `scripts/skillret/02_train_tokenizer.sh` | `runs/skillret/stage1/best.pt` |
| 03 | `scripts/skillret/03_export_codes.sh` | `runs/skillret/index/` |
| 04 | `scripts/skillret/04_build_router_data.sh` | `runs/skillret/router_data/` |
| 05 | `scripts/skillret/05_train_memorization.sh` | `router/memorization/` |
| 06 | `scripts/skillret/06_train_retrieval.sh` | `router/retrieval/` |
| 07 | `scripts/skillret/07_evaluate.sh` | `evaluation/metrics.json` |

例如从 tokenizer 训练开始重跑：

```bash
bash scripts/skillret/02_train_tokenizer.sh
bash scripts/skillret/03_export_codes.sh
bash scripts/skillret/04_build_router_data.sh
bash scripts/skillret/05_train_memorization.sh
bash scripts/skillret/06_train_retrieval.sh
bash scripts/skillret/07_evaluate.sh
```

单次调试参数可直接附在对应脚本末尾，优先于默认值：

```bash
bash scripts/skillret/02_train_tokenizer.sh --epochs 2
bash scripts/skillret/06_train_retrieval.sh --retrieval-epochs 1
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

若 embedding 服务和训练共享 GPU，先单独执行步骤 01，停止服务释放显存，再用
`SKIP_PREPARE=1` 运行 full script；也可以直接从步骤 02 开始运行。

Qwen3 官方小尺寸型号是 `1.7B`，没有 `1.5B`；如需严格的 1.5B 模型，可通过
`ROUTER_MODEL` 指向其它 `AutoModelForCausalLM` 兼容模型。

阶段 02、05、06 支持从 checkpoint 恢复：

```bash
TOKENIZER_RESUME=runs/skillret/stage1/last.pt \
  bash scripts/skillret/02_train_tokenizer.sh
ROUTER_RESUME_MEMORIZATION=latest \
  bash scripts/skillret/05_train_memorization.sh
ROUTER_RESUME_RETRIEVAL=latest \
  bash scripts/skillret/06_train_retrieval.sh
```

`build_router_data.py` 的默认划分同样是 Memorization 不留出 skills、Retrieval 留出
2% query groups；可用 `--retrieval-validation-fraction` 调整比例。

Stage-2 参数集中在公共配置中，步骤 05/06 分别执行独立的 ZeRO-3 launch，支持 Qwen3
系列的 full/LoRA 和 checkpoint resume。DeepSpeed 下的 activation checkpoint 默认
使用完整重计算的 reentrant 实现，避免重计算时读取到 ZeRO-3 的零尺寸参数占位。步骤
07 执行固定 `L` 的 trie-constrained beam search，并输出 NDCG、Recall、MAP、MRR、
Completeness、code recall 与 bucket-expanded recall。

步骤 07 默认使用 train skills 作为共享候选库，在留出的 2% train-query groups 上做
closed-set 验证：

```bash
bash scripts/skillret/07_evaluate.sh
```

仅用于检查训练集拟合程度时运行：

```bash
QUERY_SET=train bash scripts/skillret/07_evaluate.sh
```

结果默认写入 `runs/skillret/evaluation/`。官方 disjoint test-skills 协议需显式运行：

```bash
EVAL_PROTOCOL=unseen bash scripts/run_skillret_full.sh
EVAL_PROTOCOL=both bash scripts/run_skillret_full.sh
```

CPU 端到端验收与测试：

```bash
bash scripts/run_skillret_smoke.sh
.venv/bin/pytest
```
