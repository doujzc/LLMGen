# Web manual test UI

服务启动时加载一次 Router，之后复用同一模型执行约束自回归推理。模型目录必须包含
训练时自动生成的 `skill_decode_map.json` 和 `virtual_tokens.txt`。

```bash
bash scripts/router_pipeline.sh clawhub web
bash scripts/router_pipeline.sh light web
```

入口根据数据集配置自动定位最终 Retrieval 模型。也可以指定导出的 checkpoint bundle：

```bash
bash scripts/router_pipeline.sh light web \
  runs/light301-qwen3-1.7b-full-v1/exports/retrieval-checkpoint-500
```

打开 `http://127.0.0.1:8080`。额外参数会原样传递给 Web 服务，常用参数：

```text
--base-model-name-or-path PATH  LoRA 的 base model 路径覆盖
--host 127.0.0.1               监听地址
--port 8080                    监听端口
--max-code-paths 8             单次生成路径的服务端上限
--max-num-beams 8              Web 请求允许的最大 Beam 宽度
--max-input-length N           可选 prompt token 上限
```

JSON API：

- `GET /api/health`
- `GET /api/catalog?q=weather&limit=20`
- `GET /api/skill?id=@owner/skill-name`
- `POST /api/infer`，body 为
  `{"query":"...","max_code_paths":4,"top_k":10,"decoding_mode":"greedy"}`

Beam Search 会搜索完整的多 Skill 自回归输出序列，而不是把不同 beam 当作多个
Skill。启用方式为
`{"query":"...","decoding_mode":"beam_search","num_beams":4}`；Beam 宽度越大，
时延和显存开销越高。默认仍为 Greedy。

为已训练模型生成自包含解码文件：

重新导出也会将唯一训练候选集及其文本写入解码文件，供约束解码和 Skill 详情弹窗共同使用。

```bash
bash scripts/router_pipeline.sh clawhub export-web
bash scripts/router_pipeline.sh light export-web
```
