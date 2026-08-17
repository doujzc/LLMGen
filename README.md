# Top1 Router Training

本仓库只训练多轮对话的直接候选名 Top1 路由器。它不构建 embedding、RQ-VAE、
层级 code 或虚拟 token；模型直接学习生成一个候选英文名和 EOS。

## 环境

要求 Python 3.10--3.12：

本机只做代码验证，不安装 PyTorch、DeepSpeed 或 CUDA：

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv run --no-sync python -m unittest discover -s tests -v
```

NVIDIA Linux 服务器安装训练环境。下面以 CUDA 12.4 为例；如果服务器 CUDA
版本不同，应替换 `--torch-backend`，不要在 Mac 上执行：

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python \
  --torch-backend=cu124 torch==2.6.0 setuptools wheel packaging ninja
uv pip install --python .venv/bin/python \
  --torch-backend=cu124 --no-build-isolation-package deepspeed \
  -e '.[train]'
```

DeepSpeed 固定为 0.16.4。先安装 PyTorch 和构建工具，是为了让 DeepSpeed 的源码
元数据与扩展构建能够读取服务器上的 PyTorch；本机验证环境刻意不启用 `train` extra。

## 数据

训练和验证文件使用 JSONL。每行只要求两个字段：

```json
{"messages":[{"role":"user","content":"推荐一款耳机"},{"role":"assistant","content":"预算是多少？"},{"role":"user","content":"500 元以内"}],"target_candidate_name":"Ecommerce"}
```

- `messages` 支持 `user`、`assistant`、`tool` 和 `system`；最后一条非 system
  消息必须是 `user`。
- 数据自带的 system 消息会被丢弃，训练始终使用
  `configs/top1_system_prompt.md`。
- `target_candidate_name` 必须存在于 `configs/top1_candidates.json`。
- 不要求训练集覆盖全部候选，其他元数据字段会被忽略。

仓库不包含实际训练数据。把数据放到 `data_top1/`，或通过环境变量指向外部文件。
默认还会读取 `data_top1/top1_labeldesc_paper_v1.jsonl`，先执行 description → label
memorization；该文件和主训练数据一样不进入 Git，需要单独同步到训练服务器。

## 训练

```bash
TOP1_MODEL=/models/Qwen3-1.7B \
TOP1_TRAIN_DATA=/data/router/train.jsonl \
TOP1_VALIDATION_DATA=/data/router/validation.jsonl \
bash scripts/train_top1.sh
```

memorization 默认执行 20 个 optimizer steps，可独立调节：

```bash
TOP1_MEMORIZATION_DATA=/data/router/top1_labeldesc_paper_v1.jsonl \
TOP1_MEMORIZATION_STEPS=20 \
TOP1_TRAIN_DATA=/data/router/train.jsonl \
bash scripts/train_top1.sh
```

将 `TOP1_MEMORIZATION_DATA` 显式设为空字符串可以运行无 memorization 的消融基线。

只有训练集时省略 `TOP1_VALIDATION_DATA`。LoRA 训练：

```bash
TOP1_FINETUNE_MODE=lora \
TOP1_TRAIN_DATA=/data/router/train.jsonl \
bash scripts/train_top1.sh
```

单卡且不使用 DeepSpeed：

```bash
TOP1_NUM_GPUS=1 \
TOP1_DEEPSPEED_CONFIG=none \
TOP1_TRAIN_DATA=/data/router/train.jsonl \
bash scripts/train_top1.sh
```

断点恢复可将 `TOP1_RESUME` 设为 checkpoint 路径，或设为 `latest` 自动选择输出目录
中的最新 checkpoint。恢复时必须用 `TOP1_OUTPUT_DIR` 或 `TOP1_RUN_ID` 指向原 run，
已完成的 run 不允许原地恢复或覆盖。完整默认配置见 `configs/top1.env`；临时参数也可以
直接追加到训练命令末尾。

默认输出目录不是固定名称，而是：

```text
runs/top1/<experiment_name>/<UTC时间>-<git短SHA>/
```

`TOP1_EXPERIMENT_NAME` 用于归组同一实验，`TOP1_RUN_ID` 标识一次训练；也可以通过
`TOP1_OUTPUT_DIR` 显式给出完整 run 目录。

## 训练语义

训练和产物只有一套固定契约：

1. 较早消息序列化为 `history`，最后一轮 user 消息为 `current_user_request`。
2. `standalone_request_v2` 指示模型在内部补全当前请求所需的最少上下文。
3. 超长输入先删除最早历史，再从中间截断当前请求。
4. prompt 始终为“最长候选 token 路径 + EOS”预留空间，绝不根据当前 label 改变裁剪。
5. label 仅覆盖 `候选名原生 tokenizer tokens + EOS`，prompt 部分全部为 `-100`。
6. Top1 模式不新增 token，也不调整模型词表。
7. 启用 memorization 时，42 条 description → candidate 样本按照固定 seed 重排并重复，
   先执行配置数量的完整 optimizer steps；随后才进入主训练样本。

两阶段共用同一个 Trainer、模型、优化器、候选 token、system prompt 和
`prepare_router_prompt()`。memorization 样本数按有效全局 batch 对齐，阶段边界不会把
description 样本与主训练样本混入同一次梯度更新。Transformers 5.14.1 的 sequential
sampling 保证预构建课程顺序在 Trainer 内不被再次打乱；断点恢复也按同一顺序跳过数据。
memorization 结束时会额外请求一次 checkpoint，避免必须等到常规 `save_steps` 才具备
可恢复点；该 checkpoint 后续仍受 `save_total_limit` 管理。

训练与评测只调用 `prepare_router_prompt()` 这一套实现。最终 bundle 还固化 prompt 实现
哈希、system prompt/候选表哈希、Transformers 版本、tokenizer/chat-template 指纹、
EOS/PAD、候选 token 序列、`max_length` 和 `trust_remote_code`。评测逐项验证后才加载
模型，任意一项不一致都会直接终止。

## 训练运行与产物

每次训练使用一套固定目录：

```text
run_manifest.json                 # 不可变：输入哈希、配置、代码版本
status.json                       # 可变：CREATED/RUNNING/COMPLETED/FAILED
prepared/
  memorization.sft.jsonl          # 42 条 description → label 规范化消息
  memorization_profile.json
  train.sft.jsonl                 # 主训练数据的规范化消息（顺序见 schedule）
  validation.sft.jsonl
  train_profile.json              # 类别、token 长度、截断与长度偏置风险
  validation_profile.json
  tokenizer/
  training_schedule.json          # 阶段边界、重复量、顺序哈希和预期 step
logs/
  events.jsonl                    # append-only 生命周期与 Trainer 日志
  trainer_history.jsonl           # 完整 Trainer log_history
  system.json                     # Python/依赖/CUDA/GPU 信息
  torchrun/                       # 多卡各 rank 的 stdout/stderr
checkpoints/
  checkpoint-*/                   # 可恢复，默认只保留最近/最佳所需的两个
  last_checkpoint.json
  best_checkpoint.json
final/
  model/                           # 唯一可部署目录
    model_artifact.json            # 全部模型文件哈希与稳定 model_id
    candidate_registry.json
    router_system_prompt.md
    router_manifest.json
  curves.json                      # train/eval loss、LR、grad norm 曲线数据
  summary.json                     # 最终指标、best checkpoint、model_id
```

共享文件只由 global rank 0 写入，JSON 状态和汇总采用原子替换，事件采用追加写；训练
失败时保留已写日志并把状态置为 `FAILED`。`prepared/*.sft.jsonl` 与训练编码来自同一次
规范化过程，可用于检查模型实际看到的 system/user/assistant 消息。`runs/` 默认不进入
Git。

`events.jsonl` 保留训练阶段、step、epoch、loss、eval loss、learning rate、grad norm、Trainer
吞吐指标、进程内存、GPU allocated/reserved/peak memory 和非有限数告警。数据 profile
不复制原始文本，只记录候选分布、输入/目标 token 分位数、长度利用率、历史裁剪、当前
请求裁剪、候选 token 路径长度和首 token 冲突。

## 独立评测

同一个最终模型可以针对不同数据或推理参数执行任意多次评测。每次调用都会创建一个
独立、不可变的 Evaluation Run，不修改训练 run：

```bash
uv run --no-sync python scripts/evaluate_top1.py \
  --model-dir runs/top1/<experiment>/<run_id>/final/model \
  --data /data/router/test.jsonl \
  --suite-id baseline-v1 \
  --batch-size 32
```

评分阶段会显示按数据行计数的进度条；每一行完成全部候选路径评分（以及可选的
history ablation）后推进一次。

分析多轮历史的净收益，并对比候选名长度偏置：

```bash
uv run --no-sync python scripts/evaluate_top1.py \
  --model-dir runs/top1/<experiment>/<run_id>/final/model \
  --data /data/router/test.jsonl \
  --history-ablation
```

评测目录为：

```text
runs/evaluations/top1/<model_id前缀>/<evaluation_id>/
  eval_manifest.json              # 模型/数据/参数/代码及 evaluation_signature
  status.json
  logs/events.jsonl
  logs/system.json
  predictions.jsonl               # 无原始文本的逐样本候选分数与诊断
  metrics.json
  confusion_matrix.json
  summary.json
runs/evaluations/top1/evaluation_index.jsonl
runs/evaluations/top1/suites/<suite_id>/members.jsonl
```

`model_id` 来自最终模型目录内所有文件的 SHA256；每次评测前都会重新校验，确认模型没有
被改动。`evaluation_id` 每次都不同，而相同模型、数据快照和语义推理参数会得到相同的
`evaluation_signature`，可用于发现重复实验。`batch_size`、设备和精度作为执行参数记录，
不会混入语义签名；数据哈希和 history ablation 会混入签名。
`evaluation_index.jsonl` 是所有评测的追加式索引；用同一个 `--suite-id` 可把一组数据集或
参数扫描聚合到同一 suite 的 `members.jsonl`，而每个成员仍保持独立、不可变。

正式预测固定使用候选完整路径（含 EOS）的 `sum_logprob`，与 bundle 中的推理契约一致；
`mean_logprob` 只作为长度偏置诊断自动计算，不能切换为正式决策。评测也不能覆盖训练时
的 prompt、候选表、`max_length` 或 tokenizer 行为。LoRA bundle 会额外绑定 base model：
Hub 模型固定到训练时 revision，本地模型复核完整目录内容哈希。

指标包括 Top1 accuracy、候选级 precision/recall/NLL、混淆矩阵、margin、熵、ECE
校准误差、单轮/多轮分层，以及 `sum_logprob` 与 `mean_logprob` 的准确率和预测分歧。
开启 history ablation 后还会记录历史导致的预测变化、帮助数、伤害数和净收益。逐样本
结果不保存对话正文，使用原数据行号回查；`metrics.json` 还给出最低 margin 和高置信
错误的行号清单。

## 检查

```bash
uv run --no-sync python -m unittest discover -s tests -v
bash -n scripts/train_top1.sh
uv run --no-sync python -m compileall -q src scripts tests
uv run --no-sync python scripts/evaluate_top1.py --help
```
