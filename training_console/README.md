# LLMGen 训练控制台

训练控制台是现有训练 CLI 的独立配置与观测层。它不导入模型或训练代码：

- 配置以可编辑版本保存在 `.llmgen/training-console/profiles/`，每次保存递增
  revision；
- 提交时在 `.llmgen/training-console/runs/` 写入独立快照；
- detached runner 调用现有 `scripts/router_pipeline.sh`；
- 关闭浏览器或 Web 服务不会终止训练；
- 页面重启后从磁盘恢复任务、日志和 checkpoint 状态；
- “运行监控”页集中显示历史任务、流水线阶段、进度、PID、GPU、checkpoint
  与实时日志尾部。

加载已保存的 `vN` 后，点击“保存修改”会原子覆盖同一个 `vN.json`，不会创建
新版本；`revision` 用于阻止多个页面静默互相覆盖。已经提交的 run 始终读取自己的
不可变 `config.json`，后续修改配置不会改变正在运行或历史任务。

## 配置如何落盘

未点击保存的表单只存在于浏览器内存。保存后，Web 服务重新解析默认配置并校验，
然后在文件锁内把内容写入临时文件、执行 `fsync`，最后用原子 rename 替换：

```text
.llmgen/training-console/
├── profiles/<profile-id>/v0001.json   # 可编辑；revision 递增
└── runs/<run-id>/
    ├── config.json                     # 提交时冻结，不再修改
    ├── config.env                      # 仅用于审阅/导出
    ├── run.json                        # 状态、PID、exit code、checkpoint
    ├── runner.log
    └── train.log
```

profile JSON 同时保存相对默认值的 `overrides` 和完整生效值 `resolved`，但不保存
API key。默认状态根目录相对仓库，也可通过 `--state-root` 或
`LLMGEN_TRAINING_CONSOLE_STATE` 修改。

## 训练工作目录

`RUN_DIR` 是一次训练流程的统一工作目录。以下产物目录默认随它联动：

```text
$RUN_DIR/
├── processed/
├── embeddings/
├── stage1/
├── index/
├── router_data/
├── router/
└── evaluation/
```

未覆盖的目录字段直接显示 `$RUN_DIR/processed`、`$RUN_DIR/index` 等符号默认值；
校验、提交和运行快照中才安全展开为实际路径。修改 `RUN_DIR` 不需要同步修改这些
字段，也不会清除某个目录已有的独立覆盖；单项覆盖也可以继续使用
`$RUN_DIR/custom_path`。原始 `DATASET_DIR` 和控制台自己的 state root 不属于该
工作目录。

配置检查失败不会锁死“保存配置”按钮。修正字段后可以等待自动检查，也可以直接
点击“重新检查并保存”；无效配置仍不会写入磁盘或提交训练。

Stage 02“层级 Tokenizer”会同时显示 Stage 03 执行的 Code 质量门参数，包括
`CODE_MAX_RAW_COLLISION_RATE`、碰撞桶大小、各层 raw utilization 和 normalized
entropy。碰撞率、利用率和熵阈值在控制台中限制为 `0..1`，控制台不会自动放宽
质量门禁。

## 运行监控与停止

顶栏可在“配置工作台”和“运行监控”之间切换。监控页每 3 秒从磁盘状态、日志和
`nvidia-smi` 刷新一次；它不使用浏览器心跳维持任务，页面崩溃不会影响训练。

“可见 GPU”接受 `nvidia-smi` 数字编号或 GPU UUID。控制台默认使用
`CUDA_DEVICE_ORDER=PCI_BUS_ID`，Runner 启动时还会把数字编号解析成当时主机上
对应的完整 GPU UUID，再把 UUID 掩码交给训练进程。监控页分别展示请求设备、
Runtime UUID 绑定和属于本任务进程组的实际 CUDA 占用；其余整机 GPU 会弱化
显示，不再把其他任务的利用率误认为本任务。

“停止训练”不会由浏览器直接 kill PID。Web 服务只把停止请求原子写入对应
`run.json`，独立 Runner 读取后终止自己创建的训练进程组：先发送 `SIGTERM`，
12 秒后仍未退出才发送 `SIGKILL`。日志、运行快照和已保存 Checkpoint 均保留。

完整设计与安全边界见
[设计与实现规范](../docs/training-console/design.md)。

## 启动

```bash
bash scripts/serve_training_console.sh
```

默认地址：`http://127.0.0.1:8090`。

也可以从已安装的 wheel 或 editable package 启动，但需要明确给出代码仓：

```bash
python -m training_console.server --repo-root /path/to/LLMGen
```

指定地址或状态目录：

```bash
bash scripts/serve_training_console.sh \
  --host 127.0.0.1 \
  --port 8090 \
  --state-root /data/llmgen-console-state
```

只验证界面、禁止启动训练：

```bash
bash scripts/serve_training_console.sh --no-launch
```

远程机器建议使用 SSH 隧道：

```bash
ssh -L 8090:127.0.0.1:8090 user@server
```

服务会拒绝绑定 `0.0.0.0` 或其他非 loopback 地址，并拒绝非 loopback HTTP
Host。需要共享时优先使用上面的 SSH 隧道；如必须使用带鉴权的反向代理，代理还
需要把 Host 和 Origin 重写为 `127.0.0.1:8090`，否则服务会返回 403。

原始训练日志只允许当前 OS 用户读取；浏览器读取日志时还会对已知密钥值和常见
凭证格式进行脱敏，并采用有界文件尾读取，避免大日志拖垮 Web 服务。不要让下游
训练脚本主动打印凭证。
