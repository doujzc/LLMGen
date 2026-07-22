# Web manual test UI

服务启动时加载一次 Router，之后复用同一模型执行约束自回归推理。模型目录必须包含
训练时自动生成的 `skill_decode_map.json` 和 `virtual_tokens.txt`。

```bash
python -m web_server.server \
  --model-dir runs/clawhub/router/retrieval \
  --device cuda:0 \
  --dtype bfloat16
```

打开 `http://127.0.0.1:8080`。常用参数：

```text
--base-model-name-or-path PATH  LoRA 的 base model 路径覆盖
--host 127.0.0.1               监听地址
--port 8080                    监听端口
--max-code-paths 8             单次生成路径的服务端上限
--max-input-length N           可选 prompt token 上限
```

JSON API：

- `GET /api/health`
- `GET /api/catalog?q=weather&limit=20&supervised_only=true`
- `GET /api/skill?id=@owner/skill-name`
- `POST /api/infer`，body 为
  `{"query":"...","max_code_paths":4,"top_k":10}`

为旧模型生成自包含解码文件：

重新导出也会将训练候选文本写入解码文件，供界面的 Skill 详情弹窗展示。

```bash
python scripts/export_router_bundle.py \
  --model-dir runs/clawhub/router/retrieval \
  --catalog runs/clawhub/processed/catalog_train.jsonl \
  --codes runs/clawhub/index/train_codes.jsonl \
  --registry runs/clawhub/index/train_registry.json \
  --virtual-tokens runs/clawhub/index/virtual_tokens.txt \
  --training-data runs/clawhub/router_data/retrieval_train.jsonl \
  --phase retrieval
```
