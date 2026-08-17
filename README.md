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

## 训练

```bash
TOP1_MODEL=/models/Qwen3-1.7B \
TOP1_TRAIN_DATA=/data/router/train.jsonl \
TOP1_VALIDATION_DATA=/data/router/validation.jsonl \
bash scripts/train_top1.sh
```

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
中的最新 checkpoint。完整默认配置见 `configs/top1.env`；临时参数也可以直接追加到
训练命令末尾。

## 训练语义

训练和产物只有一套固定契约：

1. 较早消息序列化为 `history`，最后一轮 user 消息为 `current_user_request`。
2. `standalone_request_v2` 指示模型在内部补全当前请求所需的最少上下文。
3. 超长输入先删除最早历史，再从中间截断当前请求。
4. label 仅覆盖 `候选名原生 tokenizer tokens + EOS`，prompt 部分全部为 `-100`。
5. Top1 模式不新增 token，也不调整模型词表。

## 产物

默认输出到 `runs/qwen3-1.7b-top1/`：

```text
model/tokenizer files
checkpoint-*/
candidate_registry.json
router_system_prompt.md
router_manifest.json
sft_input.jsonl
```

`sft_input.jsonl` 与训练编码来自同一次规范化过程，可用于检查模型实际看到的
system/user/assistant 消息。`router_manifest.json` 记录数据哈希、候选集合、对话裁剪
策略和有效全局 batch。

## 检查

```bash
uv run --no-sync python -m unittest discover -s tests -v
bash -n scripts/train_top1.sh
uv run --no-sync python -m compileall -q src scripts tests
```
