# 轻量候选训练集

该目录是一套独立的 301-candidate 闭集路由数据，不会读取或覆盖
`data/clawhub_training/final/` 中原有的 1000-candidate 数据。

`candidates.jsonl` 是唯一候选来源，每行必须包含非空的 `id`、`name` 和
`desc`。生成脚本使用 `~/deepseek_api_key.txt` 中的单行密钥，通过
`deepseek-v4-flash` 完成能力画像、样本生成和复核：

```bash
bash scripts/run_light_data.sh
```

脚本会构造每个候选的单 Skill 对齐样本，以及含显式/隐式意图的多 Skill
样本，并对训练集目标顺序做自回归顺序增强。断点保存在 `data_light/work/`，
最终可训练文件写入 `data_light/final/`；重复执行会从断点续跑。

当前数据由 DeepSeek V4 Flash 生成并复核，质量门禁结果为 `pass`：

- 候选：301，全部保留；
- 单 Skill 对齐样本：941，每个候选至少 3 条；
- 多 Skill 样本：train 1838、validation 66、test 48；
- 每个候选的 train 正例至少 4 条，隐式意图样本占 52.0%；
- train 中每条语义样本包含 2 个目标顺序变体。

两个候选 ID `hithink-iwencai` 与 `hithink-wencai-suite` 共用显示名
“同花顺问财”；训练和解码均使用唯一 ID，因此不会合并标签。

## 一键训练

先启动 README 主文档所述的 OpenAI-compatible Qwen3 Embedding 服务，然后设置
目标机器上的模型路径：

```bash
export EMBEDDING_MODEL=/models/Qwen3-Embedding-8B
export ROUTER_MODEL=/models/Qwen3-1.7B
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/run_light_full.sh
```

该入口固定使用 `configs/light.env` 和 `data_light/final/`，依次执行预处理、
层级 Tokenizer、Skill code 导出、Router 数据构造、Memorization、Retrieval
和闭集评估。默认是 4 卡 ZeRO-3 全参数训练。

如果已经完成预处理并关闭了 Embedding 服务：

```bash
SKIP_PREPARE=1 bash scripts/run_light_full.sh
```

## 分阶段执行

调试或恢复时，先切换到独立配置：

```bash
export SKILLRET_CONFIG=configs/light.env
bash scripts/clawhub_train/01_prepare.sh
bash scripts/clawhub_train/02_train_tokenizer.sh
bash scripts/clawhub_train/03_export_codes.sh
bash scripts/clawhub_train/04_build_router_data.sh
bash scripts/clawhub_train/05_train_memorization.sh
bash scripts/clawhub_train/06_train_retrieval.sh
bash scripts/clawhub_train/07_evaluate.sh
```
