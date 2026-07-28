# LLMGen 训练控制台

训练控制台是现有训练 CLI 的独立配置与观测层。它不导入模型或训练代码：

- 配置以可编辑版本保存在 `.llmgen/training-console/profiles/`，每次保存递增
  revision；
- 提交时在 `.llmgen/training-console/runs/` 写入独立快照；
- detached runner 调用现有 `scripts/router_pipeline.sh`；
- 关闭浏览器或 Web 服务不会终止训练；
- 页面重启后从磁盘恢复任务、日志和 checkpoint 状态。

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
