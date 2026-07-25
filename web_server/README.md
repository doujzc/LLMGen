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
--max-code-paths 8             Greedy 单次生成路径的服务端上限
--max-num-beams 8              Beam 单次返回 code 数量的服务端上限
--max-batch-queries 1000       单个 TXT 允许的最大 Query 数
--max-batch-size 8             单次模型前向允许的最大 Query 数
--max-input-length N           可选 prompt token 上限
```

界面可切换到“批量 TXT”：文件中每个非空行作为一个 Query，空行会忽略，重复行
和原始顺序会保留。运行期间按选择的模型 Batch Size 分批处理，完成后可逐条查看
并下载 JSONL 结果。

JSON API：

- `GET /api/health`
- `GET /api/catalog?q=weather&limit=20`
- `GET /api/skill?id=@owner/skill-name`
- `POST /api/infer`，body 为
  `{"query":"...","max_code_paths":4,"top_k":10,"decoding_mode":"greedy"}`
- `POST /api/infer-batch`，body 为
  `{"queries":["query 1","query 2"],"batch_size":2,"max_code_paths":4,"top_k":10}`

Greedy 会自回归生成一条包含多个 Skill code 的换行分隔序列。Beam Search 则只
生成一条固定长度 code：`num_beams=K` 时返回概率最高的 K 个单行 code，再按
`top_k` 截断解码、碰撞桶展开后的 Skill 列表。Beam 模式忽略
`max_code_paths`（有效值固定为 1）：

```json
{
  "query": "...",
  "decoding_mode": "beam_search",
  "num_beams": 4,
  "top_k": 10
}
```

K 越大，时延和显存开销越高。默认仍为多路径 Greedy。

命令行批量推理也支持同一 TXT 格式，使用
`python scripts/infer_router.py --query-txt queries.txt`；其余模型、索引及输出
参数与 JSONL 推理相同。

为已训练模型生成自包含解码文件：

重新导出也会将唯一训练候选集及其文本写入解码文件，供约束解码和 Skill 详情弹窗共同使用。

```bash
bash scripts/router_pipeline.sh clawhub export-web
bash scripts/router_pipeline.sh light export-web
```
