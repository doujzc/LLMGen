# 数据生成算法设计

状态：当前实现契约  
最后核对：2026-09-01  
范围：从任意候选 Skill 集合生成 closed-set 数据集、Embedding、Skill 编码及 Router SFT 数据

## 1. 文档目的

本文精确描述通用候选流水线 Stage 00～08 的当前算法和数据契约：

```text
候选 Skills
→ 候选规范化
→ 路由画像
→ Workflow 规划
→ Alignment/Retrieval Query 生成
→ 独立审核与覆盖回填
→ Dataset、ordered qrels 与 Embedding
→ Codebook 与 Skill code
→ Router SFT 数据
```

本文以代码行为为准，不描述尚未实现的目标态。Stage 09～11 的 Router 模型训练、
Stage 12 的评估和 Stage 13 的模型导出只在说明上下游边界时提及。

主要实现入口：

- [`scripts/train_candidates.py`](../scripts/train_candidates.py)：统一 CLI；
- [`src/llmgen/pipeline/stages/`](../src/llmgen/pipeline/stages/)：Stage 00～13；
- [`src/llmgen/clawhub_dataset.py`](../src/llmgen/clawhub_dataset.py)：画像、多 Skill Workflow、Query、审核、回填和 Dataset；
- [`src/llmgen/clawhub_alignment.py`](../src/llmgen/clawhub_alignment.py)：单 Skill Alignment 数据；
- [`scripts/prepare_closedset.py`](../scripts/prepare_closedset.py)：processed 数据、协同图和 Embedding；
- [`src/llmgen/pipeline/code_plan.py`](../src/llmgen/pipeline/code_plan.py)：编码空间规划；
- [`scripts/build_router_data.py`](../scripts/build_router_data.py)：Router SFT 数据构造。

## 2. 范围和假设

### 2.1 输入

算法的业务输入是一个非空候选 JSONL。每行是一个 Skill：

```json
{"id":"weather","name":"天气查询","description":"查询指定城市和日期的天气","metadata":{"domain_hint":"utility"}}
```

字段契约：

| 字段 | 必需 | 说明 |
|---|---:|---|
| `id` | 推荐 | 稳定候选 ID；缺失时由 `input.id_policy` 决定是否从名称生成 |
| `name` | 是 | 用户可理解的能力名称 |
| `description` | 是 | 能力、边界和适用场景的事实来源 |
| `metadata` | 否 | 原样保留的扩展信息；默认不参与核心路由算法 |

还可以提供可选的人工 Alignment JSONL。候选和人工 Alignment 文件都会在 Run 创建时
冻结；后续 Stage 只读取冻结副本，不重新读取可能已经变化的外部文件。

### 2.2 外部服务

数据生成使用三个可独立配置的 OpenAI-compatible Provider：

| Provider | 用途 | 默认模型变量 |
|---|---|---|
| generation | 画像、Alignment Query、Retrieval Query、回填 Query | `GENERATION_MODEL` |
| review | Alignment 和 Retrieval Query 的独立审核 | `REVIEW_MODEL` |
| embedding | 候选文档向量 | `EMBEDDING_MODEL` |

Provider 可以指向同一个服务，但请求配置、ledger namespace 和模型身份彼此独立。

### 2.3 当前任务边界

当前算法是 closed-set Router 数据构造：

- 所有正例必须属于输入候选快照；
- 不允许 LLM 引入候选集以外的 target；
- 不单独构造显式负例，候选全集中未标为正例的 Skill 构成闭集隐式负类；
- 数据风格和审核规则当前面向中文手机个人智能体；
- 单候选默认进入 `alignment_only`，不伪造无意义的多 Skill Retrieval 数据。

## 3. 核心不变量

### 3.1 先确定 target，再生成 Query

Retrieval qrels 不能由 LLM 自由决定。严格顺序是：

1. 程序确定有序目标 Skill 集合；
2. generation LLM 为固定目标集合编写自然 Query；
3. 本地代码校验输出结构和证据；
4. review LLM 独立判断所有 target 是否必要；
5. 程序从审核通过 Query 的 `skill_ids` 确定性生成 ordered qrels。

因此 LLM 不能增加、删除或重排正式 target。

### 3.2 候选 Registry 唯一

候选顺序、Dataset、Embedding、Codebook、Skill code、SFT 和最终服务解码必须共享同一
候选 Registry。任何阶段都不能自行重新排序或过滤候选。

### 3.3 qrels 是有序序列

每条正式 qrel 必须包含 `position`：

```json
{"query_id":"q1","skill_id":"weather","relevance":1,"position":0}
{"query_id":"q1","skill_id":"calendar","relevance":1,"position":1}
```

强制满足：

```text
sort(qrels[query_id], key=position).skill_id == query.skill_ids
```

集合相等但顺序不同仍视为错误。

### 3.4 语义样本和增强样本分离

覆盖率、协同图和数据质量以原始语义 Query 为统计单位。Target-order augmentation
生成的训练副本通过 `source_query_id` 归并，不能虚增语义覆盖。

### 3.5 可恢复且可审计

每个 Stage 都保存输入 artifact 哈希、配置投影、实现文件哈希、中间文件、Provider
ledger、错误记录和输出 manifest。Provider transport/JSON 解析层会只重试 ledger 中尚无
耐久 `succeeded` 记录的请求；算法 schema/质量校验发生在其外层，且 Provider 返回到
ledger 提交之间仍有崩溃窗口，完整恢复边界见 16.4。

## 4. 总体 Stage 图

```text
Run 创建：冻结候选、人工数据、配置和 provenance
    │
    ▼
00 ingest                确定性
    │
    ▼
01 enrich                generation LLM
    │
    ▼
02 plan-queries          确定性
    │
    ▼
03 generate-queries      generation LLM
    │
    ▼
04 review-queries        review LLM + generation LLM 回填
    │
    ▼
05 finalize-dataset      确定性 + embedding Provider
    │
    ▼
06 train-codebook        神经 Codebook 训练
    │
    ▼
07 assign-codes          确定性编码分配与质量门禁
    │
    ▼
08 build-sft             确定性 Join 和切分
```

### 4.1 Stage 00～05 的正式输入输出契约

下表中的路径均相对对应 Stage 的正式 `output/`；Stage 00 的文件位于 Run `source/`。

| Stage | Required artifacts | 正式输出：路径（artifact schema） |
|---|---|---|
| `ingest` | 冻结输入指纹 | `candidates.input`：`source/candidates.input.jsonl` (`candidate_input/v1`)<br>`candidates.normalized`：`source/candidates.normalized.jsonl` (`candidate/v1`)<br>`candidates.catalog`：`source/catalog.jsonl` (`candidate_catalog/v1`)<br>`candidates.manifest`：`source/candidate_manifest.json` (`candidate_manifest/v1`)<br>`inputs.manual_alignment`：`source/manual_alignment.input.jsonl` (`manual_alignment_input/v1`) |
| `enrich` | `candidates.catalog` | `data.profiles`：`skill_profiles.jsonl` (`skill_profile/v1`)<br>`ledger.enrich.generation`：Stage `ledger/generation/` (`provider_ledger/v1`) |
| `plan-queries` | `candidates.manifest`、`data.profiles` | `data.workflows`：`workflows.jsonl` (`workflow_plan/v1`) |
| `generate-queries` | `candidates.manifest`、`data.profiles`、`data.workflows` | `data.queries.generated`：`queries.generated.jsonl` (`query_draft/v1`)<br>`data.queries.alignment.generated`：`queries.alignment.generated.jsonl` (`alignment_query_draft/v1`)<br>`ledger.generate-queries.generation`：Stage `ledger/generation/` (`provider_ledger/v1`) |
| `review-queries` | `candidates.manifest`、`data.profiles`、`data.workflows`、`data.queries.generated`、`data.queries.alignment.generated`、`inputs.manual_alignment` | `data.workflows.reviewed`：`workflows.jsonl` (`workflow_plan/v1`)<br>`data.queries.reviewed`：`queries.generated.jsonl` (`query_draft/v1`)<br>`data.reviews`：`query_reviews.jsonl` (`query_review/v1`)<br>`data.queries.alignment.reviewed`：`queries.alignment.generated.jsonl` (`alignment_query_draft/v1`)<br>`data.reviews.alignment`：`query_alignment_reviews.jsonl` (`alignment_query_review/v1`)<br>`ledger.review-queries.generation`、`ledger.review-queries.review` (`provider_ledger/v1`) |
| `finalize-dataset` | `candidates.manifest`、`candidates.catalog`、`data.profiles` 和 Stage 04 的五个 reviewed artifacts | `dataset.directory`：`dataset/` (`closedset_dataset/v3`)<br>`dataset.manifest`：`dataset/manifest.json` (`closedset_manifest/v3`)<br>`dataset.queries.{split}`：`dataset/queries_{split}.jsonl` (`router_querie/1`)<br>`dataset.qrels.{split}`：`dataset/qrels_{split}.jsonl` (`router_qrel/1`)<br>`processed.directory`：`processed/` (`processed_closedset/v1`)<br>`processed.manifest`：`processed/manifest.json` (`processed_manifest/v1`)<br>`embeddings.directory`：`embeddings/` (`embedding_bundle/v1`)<br>`embeddings.manifest`：`embeddings/manifest.json` (`embedding_manifest/v1`)<br>`ledger.finalize-dataset.embedding` (`provider_ledger/v1`) |
| `train-codebook` | `candidates.manifest`、`dataset.directory`、`processed.directory`、`processed.manifest`、`embeddings.directory`、`embeddings.manifest` | `code.plan`：`code_plan.json` (`code_plan/v1`)<br>`codebook.directory`：`stage1/` (`toolweaver_codebook/v1`)<br>`codebook.best`：`stage1/best.pt` (`toolweaver_checkpoint/v1`) |
| `assign-codes` | `code.plan`、`codebook.best`、`candidates.manifest`、`processed.directory`、`embeddings.directory` | `codes.directory`：`index/` (`skill_code_index/v1`)<br>`codes.train`：`index/train_codes.jsonl` (`skill_code/v1`)<br>`codes.registry`：`index/train_registry.json` (`skill_registry/v1`)<br>`codes.virtual_tokens`：`index/virtual_tokens.txt` (`virtual_tokens/v1`)<br>`codes.manifest`：`index/manifest.json` (`code_index_manifest/v1`) |
| `build-sft` | `code.plan`、`candidates.manifest`、`processed.directory`、`codes.train`、`codes.registry`、`codes.virtual_tokens` | `sft.directory`：`router_data/` (`router_sft_bundle/v1`)<br>`sft.manifest`：`router_data/manifest.json` (`router_sft_manifest/v1`)<br>`sft.memorization.train`、`sft.memorization.validation`、`sft.retrieval.alignment.train`、`sft.retrieval.train`、`sft.retrieval.validation`：对应 JSONL (`router_sft/v1`) |

Generation ledger 的 operation 子目录当前包括 `profile-candidates`、`generate-alignment`、
`generate-multiskill`、各轮 `alignment-backfill-*` 和 `coverage-backfill-*`；Review ledger 包括
`review-alignment`、`review-multiskill`。逻辑 ledger artifact 指向 Provider 根目录，内部
manifest 分别验证每个 operation。

## 5. 规范数据对象

### 5.1 标准候选

```json
{
  "skill_id": "weather",
  "name": "天气查询",
  "description": "查询指定城市和日期的天气",
  "metadata": {},
  "source_line": 1
}
```

### 5.2 路由画像

```json
{
  "profile_schema_version": 3,
  "source_signature": "...",
  "rank": 1,
  "skill_id": "weather",
  "owner": "generic-candidates",
  "slug": "weather",
  "display_name": "天气查询",
  "summary": null,
  "description": "查询指定城市和日期的天气",
  "domain": "weather_environment",
  "roles": ["retrieve"],
  "capability_zh": "查询天气、温度和降水信息",
  "aliases": ["天气", "天气预报"],
  "capability_facets": ["城市天气", "未来预报"],
  "trigger_phrases": ["明天会下雨吗"],
  "negative_boundaries": ["不能创建日历事项"],
  "routing_mode": "atomic",
  "mobile_fit": "high",
  "unsafe_action": false,
  "confusable_skill_ids": ["climate-report"]
}
```

### 5.3 Workflow

```json
{
  "workflow_id": "wf-0123456789abcdef",
  "routing_schema_version": 3,
  "anchor_skill_id": "trip-planner",
  "anchor_round": 1,
  "split_hint": "train",
  "skill_ids": ["weather", "trip-planner", "calendar"],
  "target_count": 3,
  "domains": ["weather_environment", "travel_local", "productivity_planning"],
  "cross_domain": true,
  "unsafe_action": false,
  "targets": [
    {
      "skill_id": "weather",
      "name": "天气查询",
      "aliases": ["天气", "天气预报"],
      "capability": "查询天气、温度和降水信息",
      "facets": ["城市天气", "未来预报"],
      "trigger_phrases": ["明天会下雨吗"],
      "negative_boundaries": ["不能创建日历事项"],
      "routing_mode": "atomic",
      "domain": "weather_environment",
      "roles": ["retrieve"],
      "unsafe_action": false,
      "original_description": "查询指定城市和日期的天气",
      "confusable_alternatives": []
    },
    {"skill_id": "trip-planner", "name": "行程规划", "aliases": [], "capability": "规划行程", "facets": ["路线规划"], "trigger_phrases": ["帮我排个行程"], "negative_boundaries": [], "routing_mode": "atomic", "domain": "travel_local", "roles": ["plan"], "unsafe_action": false, "original_description": "规划旅行路线", "confusable_alternatives": []},
    {"skill_id": "calendar", "name": "日程管理", "aliases": [], "capability": "创建日程", "facets": ["日程创建"], "trigger_phrases": ["加到日历"], "negative_boundaries": [], "routing_mode": "atomic", "domain": "productivity_planning", "roles": ["schedule"], "unsafe_action": false, "original_description": "创建和管理日历事项", "confusable_alternatives": []}
  ]
}
```

`targets` 保存每个 Skill 的名称、aliases、能力、facets、触发语句、负边界、路由类型、
原始描述和易混淆候选，是 generation/review Prompt 的事实上下文。

### 5.4 Retrieval Query 草稿

```json
{
  "data_schema_version": 4,
  "query_id": "cq-0123456789abcdef-v0",
  "workflow_id": "wf-0123456789abcdef",
  "anchor_skill_id": "trip-planner",
  "primary_skill_ids": ["trip-planner"],
  "support_skill_ids": ["weather", "calendar"],
  "anchor_round": 1,
  "variant": 0,
  "query": "查周末杭州天气，据此规划三天亲子行程，再把每天安排写进日历。",
  "skill_ids": ["weather", "trip-planner", "calendar"],
  "evidence": {
    "weather": "查周末杭州天气",
    "trip-planner": "规划三天亲子行程",
    "calendar": "把每天安排写进日历"
  },
  "intent_mode": "explicit",
  "target_intents": {
    "weather": "explicit",
    "trip-planner": "explicit",
    "calendar": "explicit"
  },
  "implicit_skill_ids": [],
  "implicit_rationales": {},
  "domains": ["weather_environment", "travel_local", "productivity_planning"],
  "cross_domain": true,
  "unsafe_action": false,
  "query_hash": "..."
}
```

### 5.5 Review

```json
{
  "review_schema_version": 4,
  "query_id": "cq-0123456789abcdef-v0",
  "query_hash": "...",
  "workflow_id": "wf-0123456789abcdef",
  "intent_mode": "explicit",
  "scores": {
    "mobile_style": 5,
    "complexity": 4,
    "target_necessity": 5,
    "coherence": 5,
    "specificity": 4
  },
  "missing_skill_ids": [],
  "redundant_skill_ids": [],
  "unsafe": false,
  "pass": true,
  "model_pass": true,
  "issues": []
}
```

`model_pass` 是审核模型的原始声明；`pass` 由本地阈值重新计算，是正式选择依据。

### 5.6 Alignment Query 和 Review

Alignment Query 是 Retrieval Query 的单 target 特例，但额外保存语义覆盖任务：

```json
{
  "data_schema_version": 4,
  "query_id": "ca-0123456789abcdef-v0",
  "variant": 0,
  "generation_requirements": ["identity_explicit"],
  "routing_mode": "atomic",
  "query": "用天气查询看看杭州明天下不下雨",
  "query_hash": "...",
  "skill_ids": ["weather"],
  "primary_skill_ids": ["weather"],
  "support_skill_ids": [],
  "evidence": {"weather": "天气查询看看杭州明天下不下雨"},
  "intent_mode": "explicit",
  "target_intents": {"weather": "explicit"},
  "implicit_skill_ids": [],
  "implicit_rationales": {},
  "domain": "weather_environment"
}
```

对应 Review：

```json
{
  "review_schema_version": 4,
  "query_id": "ca-0123456789abcdef-v0",
  "query_hash": "...",
  "skill_id": "weather",
  "scores": {
    "mobile_style": 5,
    "target_relevance": 5,
    "specificity": 4,
    "coherence": 5
  },
  "missing": false,
  "extra_capability_needed": false,
  "requirement_satisfied": true,
  "unsafe": false,
  "pass": true,
  "model_pass": true,
  "issues": []
}
```

人工或旧数据适配时还可带 `curation_source`、`review_source`、`legacy_source_query_id`；
回填行可带 `backfill_round`。这些来源字段必须保留，不能与模型原生样本混淆。

## 6. Run 创建和输入冻结

在 Stage 00 前，Runner 完成以下工作：

1. 解析 YAML、环境变量和 CLI overrides；
2. 将候选原始字节复制到 `source/candidates.input.jsonl`；
3. 将可选人工 Alignment 复制到 `source/manual_alignment.input.jsonl`；未配置时写入空文件；
4. 保存两个输入的字节数和 SHA-256；
5. 保存 resolved config、Python/包/设备、代码和本地基座模型 provenance；
6. 创建 14 个 Stage 的初始状态。

原始外部文件后续即使被修改或删除，也不会改变已创建 Run 的语义。修改候选或人工数据
必须创建新 Run，或通过 `fork` 创建派生实验。

## 7. Stage 00：候选规范化

处理步骤：

1. 校验冻结候选文件 SHA-256；
2. 逐行解析 JSON 对象；
3. 按 `input.id_policy` 生成或规范化 `skill_id`；
4. 校验输入非空、ID 唯一、名称非空、描述非空；
5. 保持输入顺序，写出 normalized candidates 和兼容 catalog；
6. 校验人工 Alignment 冻结文件 SHA-256；
7. 写出候选 manifest。

输入兼容 `id` 或 `skill_id`，以及 `description` 或 `desc`。每个非空 JSONL 行必须是对象；
`metadata` 若存在必须是对象。`skill_id` 必须唯一，但当前实现不禁止不同候选使用相同
`name`，只在 manifest 中记录 `unique_name_count`。规范化后的行额外保存从 1 开始的
`source_line`，用于定位坏输入。

候选 manifest 至少保存 `candidate_count`、`ordered_skill_ids`、`execution_mode`，以及 input、
normalized、catalog 三份文件的字节数和 SHA-256。

单候选模式：

```text
input.single_candidate_policy=alignment_only → execution_mode=alignment_only
input.single_candidate_policy=error          → Stage 失败
```

正式 artifact：

| 逻辑名称 | 文件 | Schema |
|---|---|---|
| `candidates.input` | `source/candidates.input.jsonl` | `candidate_input/v1` |
| `candidates.normalized` | `source/candidates.normalized.jsonl` | `candidate/v1` |
| `candidates.catalog` | `source/catalog.jsonl` | `candidate_catalog/v1` |
| `candidates.manifest` | `source/candidate_manifest.json` | `candidate_manifest/v1` |
| `inputs.manual_alignment` | `source/manual_alignment.input.jsonl` | `manual_alignment_input/v1` |

## 8. Stage 01：生成路由画像

### 8.1 输入和调用方式

`enrich` 读取完整 catalog，按 `data_generation.profile_batch_size` 分批，调用 generation
Provider。默认 batch size 为 10，Provider 默认并发为 12。

### 8.2 画像内容

模型需要从名称和原始描述中抽取：

- `domain`：20 个预定义领域之一；
- `roles`：`retrieve/perceive/analyze/plan/create/communicate/store/schedule/act/monitor/automate/protect/meta`；
- `capability_zh`；
- 用户可说出的 `aliases`；
- 能力切面 `capability_facets`；
- 正向触发语句 `trigger_phrases`；
- 不应路由到该 Skill 的 `negative_boundaries`；
- `routing_mode`：`atomic/composite/meta`；
- `mobile_fit`；
- `unsafe_action`。

本地代码严格校验枚举、数量、长度和 composite facets。批次失败后会拆为单候选请求，
隔离格式异常。最终必须实现 catalog 全覆盖，否则 Stage 失败。

主要本地约束：

| 字段 | 约束 |
|---|---|
| `domain` | 必须属于预定义领域枚举 |
| `roles` | 1～3 个，必须属于 role 枚举，去重后保序 |
| `capability_zh` | 4～80 字 |
| `aliases` | 最多 6 个，每项最多 80 字；有 display name 时至少一个 |
| `capability_facets` | 1～10 个，每项最多 80 字 |
| `trigger_phrases` | 2～8 个，每项最多 80 字 |
| `negative_boundaries` | 0～5 个，每项最多 100 字 |
| `routing_mode` | `atomic`、`composite` 或 `meta` |
| `mobile_fit` | `high`、`medium` 或 `low` |

`routing_mode=composite` 时必须至少有两个 capability facet。可选字段缺失时，本地代码使用
名称、capability 或 roles 进行有界回退，但不会放宽枚举和长度门禁。

### 8.3 易混淆候选

画像生成后，本地代码计算文本特征：

```text
features = 小写拉丁/数字 token ∪ 中文二元 gram
similarity(a,b) = |features(a) ∩ features(b)| / |features(a) ∪ features(b)|
```

特征来源包括名称、能力摘要、aliases、facets、trigger phrases、摘要和最多 1600 字原始描述。
每个 Skill 在同领域或 role 重叠的候选中选出最多 3 个高分近邻，写入
`confusable_skill_ids`。这些近邻用于阻止错误组合，并作为 generation/review 的 hard-negative
上下文。

近邻分数为：

```text
confusable_score(a,b)
= similarity(a,b)
+ 0.08 × I(domain 相同)
+ 0.02 × role_overlap_count
```

只有同领域或至少一个 role 重叠的候选才进入排序；同分时再按 catalog rank 和
`skill_id` 稳定决胜。

当前实现不会在 `enrich` 阶段生成 Embedding；神经 Embedding 位于 Stage 05。

输出：`data.profiles`，以及 generation request/response ledger artifact。

## 9. Stage 02：确定性 Workflow 规划

### 9.1 目标数量

基础 Workflow 不调用 LLM。每个候选依次作为 anchor。默认配置：

```yaml
workflows_per_skill: 3
skills_per_query:
  min: 2
  max: 4
```

当候选数允许时，每个 anchor 的三个 round 分别请求 2、3、4 个 target。候选不足时，
目标数截断到候选总数。例如 2 个候选时三个 Workflow 都是 2-target。

### 9.2 候选池

对当前已选集合，先汇总同领域和 `DOMAIN_NEIGHBORS` 邻近领域候选；数量不足时回退到
全候选。候选先按全局低使用次数和稳定 seed 排序，并截取最多 256 个进入精排。

### 9.3 精排函数

对候选 `c` 和已选集合 `S`：

```text
score(c,S) = Σ domain_bonus(c,s)
           + Σ role_compatibility(c,s)
           - Σ 14 × max(0, similarity(c,s) - 0.25)
           + mobile_fit_bonus
           - 0.16 × usage_count(c)
           + stable_jitter(seed, round, rank)
```

领域分数：

| 关系 | round 0/2 | round 1/3 |
|---|---:|---:|
| 同领域 | +2.5 | +0.5 |
| 邻近领域 | +2.0 | +3.5 |
| 其他领域 | -1.5 | -1.5 |

Role 分数：

- 每个命中预定义上下游 `ROLE_EDGES` 的 role 对加 2.5；
- role 集合完全相同减 5；
- role 有交集但不完全相同再减 1。

`mobile_fit=high` 加 1.5；同时选择多个 `mobile_fit=low` 会产生额外惩罚。

相似度大于 0.72 且 role 集合相同的候选被视为近似替代品，正常情况下禁止进入同一
Workflow。极小或高度同质候选集无法构造时，只放宽这条近似替代限制，不放宽候选唯一性。

完整 target set 尽量不重复。每个 Skill 独立作为 anchor，因此基础计划天然保证 anchor
覆盖；manifest 还会校验每个候选恰好拥有配置数量的 anchor Workflow。

### 9.4 Target 顺序和 Workflow ID

选定集合后按最早 role 排序：

```text
protect/meta
→ retrieve/perceive
→ monitor
→ analyze
→ plan
→ create
→ schedule
→ act/automate
→ communicate/store
```

同一位置再按 catalog rank、`skill_id` 排序。该顺序写入 `workflow.skill_ids`，成为后续
Query、qrels 和 SFT 的权威初始顺序。

Workflow ID 由 routing schema、anchor、round 和排序后的 target set 做 SHA-256 后截断：

```text
wf-{16 hex chars}
```

单候选 `alignment_only` 模式写出一个合法但为空的 Workflow 文件，不执行多 Skill 规划。

## 10. Stage 03：Retrieval Query 生成

### 10.1 调用参数

默认关键参数：

```yaml
explicit_variants: 3
implicit_variants: 1
query_batch_size: 4
validation_retry_rounds: 3
min_completion_rate: 0.95
```

当前 `explicit_variants` 名称具有误导性：Stage 将其传为 `--variants 3`，底层解释为
“每个 Workflow 的总变体数”。因此普通 Workflow 默认实际生成：

```text
2 explicit + 1 implicit = 3 total
```

若 Workflow 的全部 target 都是 `unsafe_action=true`，没有可作为隐式目标的安全 Skill，
则生成 `3 explicit + 0 implicit`。绝不是 3 explicit 加 1 implicit。

### 10.2 Prompt 输入

每个 Workflow 的 Prompt 项包含：

- `workflow_id`；
- `primary_skill_id`；
- 完整且有序的 `required_target_ids`；
- explicit/implicit 数量计划；
- 是否包含高影响动作；
- 可选人工 recovery scenario；
- 每个 target 的名称、aliases、能力、facets、触发语句、负边界、路由类型、领域、roles、
  高影响标志、最多 1600 字原始描述及 confusable alternatives。

模型只能围绕固定 target 编写不同自然场景，不能分摊、增加、删除或重排 target。

### 10.3 Query 语言要求

Prompt 要求：

- 中文手机用户自然口语；
- 所有 target 构成一个有依赖关系的现实任务；
- 包含时间、地点、对象、条件、偏好、截止时间或输出格式等具体约束；
- 不得泄漏 Skill ID、ClawHub、target、路由或数据集术语；
- 不得声称操作已经完成；
- composite Skill 应体现两个以上能力切面；
- meta Skill 应体现真实触发状态；
- 品牌是能力边界时不得把相近平台互换；
- 未经用户明确要求，不得扩展发布、交易、删除、部署等高影响动作。

Explicit 样本必须明确表达全部 target 动作。Implicit 样本至少有一个显式 target 和一个
隐式 target；隐式能力必须被用户目标或约束强蕴含，删除它后任务无法满足，而不是仅仅
“可能有帮助”。`unsafe_action=true` 的 target 永远不能隐式出现。

### 10.4 模型返回 Schema

```json
{
  "items": [
    {
      "workflow_id": "wf-0123456789abcdef",
      "variants": [
        {
          "intent_mode": "explicit",
          "query": "...",
          "evidence": {
            "skill-a": "query 中逐字出现的片段",
            "skill-b": "query 中另一个逐字片段"
          },
          "implicit_skill_ids": [],
          "implicit_rationales": {}
        }
      ]
    }
  ]
}
```

### 10.5 本地严格校验

模型输出不会直接成为正式数据。每个 variant 必须通过：

1. Query 实际长度为 25～220 字；
2. 至少有 12 个中文字符，中文占语言字符比例不低于 45%；
3. 不包含禁用的数据集/实现话术或 `@owner/slug`；
4. 不以“用户希望”“请求：”“Query”等数据描述开头；
5. 至少命中一个时间、对象、条件、顺序、数量或格式上下文 marker；
6. variant index 对应的 explicit/implicit 类型正确；
7. `implicit_skill_ids` 是 target 子集；
8. implicit 数量属于 `1..target_count-1`，从而同时保留显式和隐式 target；
9. unsafe target 不在 `implicit_skill_ids`；
10. `implicit_rationales` key 与隐式 target 完全相同，每条实际长度为 8～80 字；
11. `evidence` key 与 Workflow 全部 target 完全相同；
12. 每段 evidence 是 Query 中 2～80 字的逐字子串；
13. 不同 target 不复用同一 evidence；
14. 同一 Workflow 的变体规范化 Query hash 互不重复；
15. anchor 必须属于 target 集合。

Prompt 文案要求 Query 25～180 字，但当前本地校验实际允许 25～220 字。正式接受边界以
本地校验为准；这是当前实现差异，不应在实验分析中混淆。

模型无权提供正式 `skill_ids`。校验通过后，代码直接复制 `workflow.skill_ids`，并生成：

```text
query_id = cq-{workflow_id 去掉 wf- 前缀}-v{variant_index}
```

### 10.6 批次、修复和完成条件

初始请求按 4 个 Workflow 一批。底层将最多 200 个 API batch 作为一个 checkpoint chunk，
每个 chunk 后原子刷新中间 `queries.generated.jsonl`。

失败分两类：

- transport/HTTP/JSON 失败：由 Provider 按 `max_retries` 重试；
- 本地 schema/质量失败：进入 Workflow 级 repair，默认最多 3 轮。

Repair Prompt 包含具体本地错误、上一次输出和纠错规则，并要求重新返回该 Workflow 的
全部 3 个变体。每轮后重新校验和刷新中间文件。

只有 variant index `0..2` 全部存在、类型数量正确且 Query hash 唯一的 Workflow 才完整
保留。若最终仍有缺失：

```text
completion_rate < 0.95  → Stage 失败；新 attempt 复用 Provider ledger 后重新执行算法
completion_rate >= 0.95 → 缺失 Workflow 记为 abandoned，Stage 可成功
```

被 abandon 的 Workflow ID 写入本次 attempt 的 manifest，供本次输出审计。新的 pipeline
attempt 使用新的 attempt output，当前不会继承上一 attempt 的 abandoned manifest；因此
它会重新评估这些 Workflow，但 Provider 层通常会重放相同 ledger payload。

### 10.7 Alignment Query

Stage 03 总是先为每个候选生成单 Skill Alignment Query，默认每 Skill 10 条；之后才在
非单候选模式生成 Retrieval Query。Alignment 的 target 固定为一个 Skill，且所有样本均为
`intent_mode=explicit`。

算法按 variant index 确定性分配语义覆盖任务：

- `routing_mode=composite`：前 `ceil(variants/3)` 条要求 `composite_bundle`；
- `routing_mode=meta`：前 `ceil(variants/2)` 条要求 `meta_task_context`；
- 最后 `ceil(variants/4)` 条要求 `native_followup`；
- 第 0 条在存在用户可见名称/alias 时要求 `identity_explicit`；
- 未命中特殊要求的 variant 标记为 `core`。

一条 variant 可以同时带多个 requirement。它们写入 `generation_requirements`，供后续 Review
逐项检查，不能只依赖 Query 与 Skill 的粗粒度相关性。

Alignment 本地 validator 执行：

1. Query 实际长度 6～180 字；Prompt 文案要求 6～140 字，正式接受边界仍以 validator 为准；
2. 禁止泄漏带分隔符的 opaque Skill ID、`@owner/slug` 和数据集/路由话术；
3. 默认至少 3 个中文字符，且中文占中英文字符至少 12%；若 Query 含画像声明的 `/command`
   触发词，则允许技术命令特例；
4. evidence 应为 Query 中的逐字片段；单 target 情况下若模型的 evidence 引号或省略写法无效，
   本地安全回退为完整 Query，而不是丢弃正确样本；
5. 同一候选的 variant index 必须完整，Query hash 必须唯一；最终还校验跨候选全局 hash
   不重复。

初始请求按 `alignment_batch_size=3` 分批；失败 batch 拆成单候选重试。任何候选缺少完整的
10 条基础 variant，Stage 直接失败，不应用 Retrieval 的 95% 完成率豁免。单候选模式在此
结束多 Skill 生成分支，并写出合法的空 Retrieval Query artifact。

## 11. Stage 04：独立审核与覆盖回填

### 11.1 初始文件

Stage 开始时把 Workflow、Retrieval Query 和 Alignment Query 复制到本次 attempt 的
输出目录。后续审核和回填只修改这些 attempt-owned 文件；handler 返回后的目录换入、
Registry 登记和失败边界见 17.2。

### 11.2 Alignment 审核和回填

Review Provider 首先审核所有 Alignment Query。审核关注单一目标是否与 Query 直接匹配、
表达是否自然具体、是否超出能力边界以及是否包含不安全扩展。

模型返回 1～5 分的 `mobile_style`、`target_relevance`、`specificity`、`coherence`，以及
`missing`、`extra_capability_needed`、`requirement_satisfied` 和 `unsafe`。正式通过条件由本地
代码重新计算：

```text
mobile_style >= 3
and target_relevance >= 4
and specificity >= 3
and coherence >= 4
and missing is false
and extra_capability_needed is false
and requirement_satisfied is true
and unsafe is false
```

模型返回的 `pass` 同样只保存为 `model_pass`。Review 按 `review_batch_size=10` 分批，失败
batch 拆成单 Query 重试；最终 Review 数不等于 Query 数时 Stage 失败。

然后执行最多 `alignment_backfill_rounds` 轮，默认 5 轮。每轮：

1. 统计每个 Skill 已审核通过的 Alignment 数；
2. 找出低于 `alignment_queries_per_skill` 的候选；
3. generation Provider 定向生成缺失样本；
4. review Provider 重新审核完整 Alignment 集合。

这里不是按精确 deficit 只生成若干行：每个被选中的候选在每轮生成完整的
`variants=alignment_queries_per_skill` 批次，默认 10 条，再由 Review 筛选，因此可能超过
最低数量。回填行保存 `backfill_round`；同一 round 已经为某候选落盘后，再执行该 round
不会重复追加。

回填不仅检查通过数量，还检查 requirement 最低覆盖：每个候选至少 1 条
`native_followup`；存在用户可见名称/alias 时至少 1 条 `identity_explicit`；
`routing_mode=composite` 时至少 2 条 `composite_bundle`；`routing_mode=meta` 时至少 2 条
`meta_task_context`。数量已经达标但 requirement 不足时仍会进入回填。

如果配置 `data_generation.manual_alignment_path`，流水线调用人工 Alignment 适配器，将 Run
创建时冻结的 curated 数据并入 Query/Review。人工来源和审核来源写入记录，不能伪装为
模型生成数据。人工行仍经过单 Skill Query 本地 validator 和全局 hash 去重；随后写入
`curation_source=manual_alignment`、`review_source=manual_curation` 的显式通过记录，不再
调用 Review Provider。人工输入引用未知 Skill、requirement 非法或与已有 Query 重复时
Stage 失败。

### 11.3 Retrieval 审核

非单候选模式下，Review Prompt 获得：

- Query 文本；
- primary Skill；
- explicit/implicit 声明和隐式理由；
- 全部 target 的原始描述、aliases、facets、触发语句、负边界和路由类型；
- confusable alternatives；
- 高影响动作标记。

模型逐项返回 1～5 分：

| 指标 | 含义 | 本地通过阈值 |
|---|---|---:|
| `mobile_style` | 是否像真实手机用户 | 3 |
| `complexity` | 是否是有依赖的多能力任务 | 4 |
| `target_necessity` | 每个 target 是否不可缺少 | 4 |
| `coherence` | 场景和步骤是否连贯 | 4 |
| `specificity` | 对象和约束是否充分 | 3 |

同时返回：

- `missing_skill_ids`：目标集合中未被 Query 表达或强蕴含的 Skill；
- `redundant_skill_ids`：重复、可替代或只是可选增强的 Skill；
- `unsafe`：Query 未授权却扩大的高影响动作；
- `issues`：最多 3 个简短问题标签；
- 模型自己的 `pass`。

本地代码重新计算正式 `pass`：

```text
all score thresholds satisfied
and missing_skill_ids == []
and redundant_skill_ids == []
and unsafe is false
and issues == []
```

模型声明的 `pass` 仅保存为 `model_pass`。审核失败记录不会删除，后续正式 Dataset 只过滤
`pass=true`。

已有 Review 只有在 schema version 和 `query_hash` 都匹配时才可复用；Query 文本改变会
强制重新审核。Retrieval Review 按 `review_batch_size=10` 分批，失败 batch 再拆成单 Query
重试；最终 Review 数少于 Query 数时 Stage 失败。

### 11.4 Retrieval 覆盖回填

基础生成完成后，算法统计每个候选的训练正例。当前覆盖统计口径为：

```text
审核通过的 Alignment Query
+ 审核通过、近重复去除、确定性分到 train 的未增强 Retrieval Query
```

Target-order augmentation 不计入覆盖。默认最低正例数来自：

```yaml
retrieval_positives_per_skill: 20
```

对 deficit 大于 0 的 Skill，规划回填 Workflow 数：

```text
planned(skill) = ceil(
    deficit(skill) / variants_per_workflow × coverage_oversample_factor
)
```

默认 `variants_per_workflow=3`、`coverage_oversample_factor=3.0`。

回填规划优先把多个欠覆盖 Skill 放入同一个 Workflow，然后考虑邻近领域、role 兼容、
相似度和 target-set 去重。相似度大于 0.78 且 role 相同的候选通常不组合。回填 Workflow
带有：

```json
{
  "coverage_backfill": true,
  "coverage_round": 1,
  "split_hint": "train"
}
```

因此它们强制进入 train。每轮执行：

```text
统计缺口
→ 规划新增 Workflow
→ generation 只补新增 Workflow
→ review 新 Query
→ 重新统计
```

默认最多 `max_backfill_rounds=5`。在默认 `skills_per_query=2..4` 配置下，当前回填实现的
目标数计算只会选择 2 或 3 个 target；基础 Workflow 仍覆盖 2、3、4 target。这是当前
实现细节。

### 11.5 最终 Alignment 回填

Retrieval 回填之后，再执行最多 `final_alignment_backfill_rounds` 轮，默认 5 轮。该步骤
基于“Alignment + Train Retrieval”合计覆盖，使用 Alignment Query 补足仍未达到最低正例
数的候选。

这里存在一个当前实现偏差：Stage 04 的最终 Alignment 回填在计算多 Skill Train 覆盖时，
调用 `workflow_split(..., seed=20260720)` 且使用默认 90/5/5，而没有传入 `run.seed` 和
`data_generation.split`。Stage 05 的正式 Dataset 则使用冻结配置，默认 `run.seed=42`。
因此 Stage 04 的数值只是 provisional coverage，可能少补或多补；Stage 05 以正式 split
重新计算并执行权威覆盖门禁，少补时会写 `coverage_failure.json` 后失败。实验分析必须以
Stage 05 manifest 和 coverage 文件为准。

### 11.6 输出

| 逻辑名称 | 内容 |
|---|---|
| `data.workflows.reviewed` | 基础和回填 Workflow |
| `data.queries.reviewed` | 全部 Retrieval Query，包括未通过样本 |
| `data.reviews` | Retrieval Review |
| `data.queries.alignment.reviewed` | Alignment Query |
| `data.reviews.alignment` | Alignment Review |

Stage 同时登记 generation 和 review ledger artifact。

## 12. Stage 05：正式 Dataset、qrels、协同图和 Embedding

### 12.1 接受和近重复去除

正式导出只选择 Review `pass=true` 的 Retrieval Query。随后对规范化 Query 做字符
3-gram Jaccard 近重复检测，阈值为 0.86。

发生冲突时的保留优先级：

```text
Review 五项总分更高
→ Query 更长
→ Query ID 稳定排序
```

被删除项写入 `rejected_near_duplicates.jsonl`，包含 `duplicate_of` 和 Jaccard 分数。

### 12.2 Workflow 级切分

切分单位是完整 Workflow，不是单条 Query。一个 Workflow 的全部语义变体必须属于同一
split，防止近同义改写泄漏到训练和评估两侧。

默认 90/5/5 保持历史算法：

```text
bucket = stable_hash(seed, "workflow_split", workflow_id) % 20
bucket == 0 → validation
bucket == 1 → test
其他        → train
```

非默认合法比例使用相同稳定哈希映射到配置区间。Coverage 和 recovery Workflow 始终为
train。

若某候选在多 Skill train 中完全没有正例，算法从 validation/test 选择包含该候选且
Review 总分最高的 Query，并将其整个 Workflow 移入 train；不会只移动单一变体。

### 12.3 覆盖门禁

导出再次计算：

```text
combined_positive_count(skill)
= accepted Alignment
+ accepted, unaugmented, semantic Train Retrieval
```

任何候选低于 `retrieval_positives_per_skill` 时，写出 `coverage_failure.json` 并使 Stage
失败。顺序增强不会用于通过该门禁。

Alignment Dataset 还有独立门禁：每个候选的通过数必须至少为
`alignment_queries_per_skill`，否则写 `alignment_coverage_failure.json` 并失败。

### 12.4 Target-order augmentation

只对 train Retrieval Query 执行。算法对原始 `skill_ids`：

1. 保留原顺序；
2. 优先生成循环旋转；
3. 其余排列按稳定哈希排序；
4. 取前 `min(order_variants, factorial(target_count))` 个。

默认 `order_variants=4`：

| target 数 | 实际最多顺序版本 |
|---:|---:|
| 2 | 2 |
| 3 | 4 |
| 4 | 4 |

增强版本不修改 Query 文本，只修改 target 顺序并增加：

```json
{
  "source_query_id": "cq-...-v0",
  "target_order_variant": 1,
  "id": "cq-...-v0--ord-1"
}
```

这类样本用于让不同 target 出现在不同自回归位置，不代表新增语义 Query。

增强完成后，若 Train 行数低于 `min_augmented_train_queries`，写
`training_scale_failure.json` 并失败；当前默认值为 0，因此默认不设置额外的数据规模下限。

### 12.5 qrels 生成和规范化

Dataset exporter 先按每条 Query 的 `skill_ids` 顺序写正 qrel。随后三层代码共同完成门禁：

1. `ensure_ordered_qrels()` 检查 Query ID 唯一、target 非空且无重复、同 split 的 qrel 集合与
   target 集合完全相同、没有 orphan qrel；再按权威 `query.skill_ids` 补齐或重写
   `position`，并原子更新 manifest 哈希；
2. `validate_ordered_qrels()` 专门检查每条 Query 的 position 是否为从 0 开始的连续整数，
   以及按 position 排序后的 Skill 序列是否与 Query 完全一致；
3. `05_validate_dataset.py` 的完整 Dataset audit 再检查 target 全部属于候选 Registry、
   primary/support 分区、implicit 语义、held-out target-count 是否在 Train 出现、manifest
   大小和 SHA-256 等 Query/Dataset 质量契约。

Exporter 固定写 `relevance=1`；`ensure_ordered_qrels()` 对已有 relevance 原样保留，
`validate_ordered_qrels()` 本身不单独判断 relevance 正负。当前正式流水线的正 relevance
由 exporter 的受控写入保证，不应把这一点误归因于 ordered-qrels validator。

Alignment 是单 target 特例，position 固定为 0。

### 12.6 Dataset 文件

```text
dataset/
├── skills.jsonl
├── queries_train.jsonl
├── qrels_train.jsonl
├── queries_validation.jsonl
├── qrels_validation.jsonl
├── queries_test.jsonl
├── qrels_test.jsonl
├── queries_alignment.jsonl
├── qrels_alignment.jsonl
├── queries.jsonl
├── rejected_near_duplicates.jsonl
└── manifest.json
```

`manifest.json` 保存候选数、Review 完成率、语义/增强样本数、split 分布、target 数分布、
跨域统计、覆盖统计、顺序增强统计，以及所有正式文件的字节数和 SHA-256。

### 12.7 processed 数据

正式 Dataset 通过完整审计后，`prepare_closedset.py` 生成稳定训练格式：

```text
processed/
├── catalog_train.jsonl
├── queries_train.jsonl
├── qrels_train.jsonl
├── queries_validation.jsonl
├── qrels_validation.jsonl
├── queries_test.jsonl
├── qrels_test.jsonl
├── queries_alignment.jsonl
├── qrels_alignment.jsonl
├── collab_graph_train.npz
└── manifest.json
```

候选训练文本为：

```text
name | capability_zh | description
```

processed qrels 保留可选 `position` 字段。

### 12.8 Skill 协同图

协同图只使用 Train qrels。顺序增强版本先按 `source_query_id` 去重，然后以同一语义 Query
中共同出现的 Skill 建立边。边权归一化为：

```text
co_use_count / sqrt(skill_frequency_product)
```

输出 `collab_graph_train.npz`，包括 `src`、`dst`、`weight`、`num_nodes` 和候选顺序哈希。

### 12.9 Embedding

Embedding Provider 对 `catalog_train.jsonl` 的候选训练文本编码。默认：

```yaml
batch_size: 8
max_batch_chars: 12000
timeout_seconds: 600
max_retries: 5
```

向量归一化后按候选 Registry 顺序保存为：

```text
embeddings/train.npy
embeddings/manifest.json
```

Embedding manifest 记录模型、endpoint、requested dimensions、shape、SHA-256、候选顺序
哈希和 ledger 统计。Embedding 不参与 Stage 02 的基础 Workflow 规划；当前只用于 Codebook
学习。

Stage 05 正式 artifact 包括 `dataset.directory`、`processed.directory`、
`embeddings.directory`、各自 manifest、各 split Query/qrels 以及 embedding ledger。

## 13. Stage 06：CodePlan 和 Codebook

### 13.1 CodePlan 输入

CodePlan 的首要输入是候选数 `N`。默认自动配置：

```yaml
mode: auto
latency_priority: balanced
spare_capacity_ratio: 1.25
max_virtual_tokens: 512
max_branching_factor: 256
```

目标容量：

```text
N == 1: target_capacity = 1
N > 1 : target_capacity = max(N, ceil(1.25 × N))
```

自动规划遍历 1～8 层。每层 branching factor：

- 至少为 2，单候选特例可以为 1；
- 不超过 `min(N, max_branching_factor)`；
- 所有 factor 的乘积必须达到 `target_capacity`；
- factor 之和，即 virtual token 数，不超过 `max_virtual_tokens`。

`balanced` 优先级的选择代价为：

```text
16 × num_levels + virtual_token_count
```

之后依次比较层数、token 数、剩余容量和 factor 序列。该代价使小候选集倾向单 token，
较大候选集避免扩张为 N 个新增 token。

Manual 模式直接使用 `branching_factors`，但仍校验层数、容量、候选数上限和 virtual-token
预算。

### 13.2 Codebook 训练

Stage 先验证 Run 的训练 provenance，再生成并冻结 `code_plan.json`、将其纳入 checkpoint
lineage，然后使用：

- `embeddings/train.npy`；
- 候选稳定顺序；
- `collab_graph_train.npz`；
- RQ/ToolWeaver 配置；

训练层级 Codebook。训练支持从 lineage 完全匹配的 checkpoint 恢复。不同候选、CodePlan、
配置或实现版本的 checkpoint 默认拒绝。

当前默认训练参数：

```yaml
rq_layers: [512, 256, 128]
embedding_dim: 64
beta: 0.25
epochs: 100
batch_size: 512
learning_rate: 1.0e-4
scheduler: cosine
warmup_ratio: 0.05
eval_every: 1
graph_lambda: 0.001
amp_dtype: bf16
```

虚拟 token 格式为 `<SK_L{level}_{index}>`。为保证单个 codebook 层的全部中心能参与相关
计算，底层 tokenizer 训练实际 batch size 取：

```text
max(code.batch_size, max(branching_factors))
```

训练输入同时绑定 processed manifest、embedding manifest、图哈希、候选顺序和 CodePlan。
`last.pt` 可用于同 lineage 恢复，最优 checkpoint 写为 `best.pt`。

输出：

| Artifact | 说明 |
|---|---|
| `code.plan` | 冻结的层数、branching factors、容量和 token 预算 |
| `codebook.directory` | Stage-1 训练目录 |
| `codebook.best` | 最佳 Codebook checkpoint |

## 14. Stage 07：Skill code 分配

Stage 使用最佳 Codebook 计算每个候选的 raw nearest code，再按
`code.assignment` 执行最终分配。默认策略是 `balanced_hierarchical`，并使用冻结的固定
层数和 branching factors。

质量门禁包括：

- 最终 collision rate；
- raw collision rate；
- 最大 bucket size；
- 各层 token 利用率；
- normalized entropy；
- raw 各层利用率和 entropy；
- 所有 token 必须属于 virtual-token namespace；
- 所有 code path 长度必须等于 CodePlan 层数。

默认门禁：

```yaml
max_collision_rate: 0.01
max_raw_collision_rate: 1.0
max_bucket_size: 2
min_level_utilization: 0.0
min_normalized_entropy: 0.0
min_raw_level_utilization: []
min_raw_normalized_entropy: 0.0
```

每个候选输出一行：

```json
{
  "skill_id": "weather",
  "indices": [12, 3],
  "tokens": ["<SK_L1_12>", "<SK_L2_3>"]
}
```

消费者应从 `tokens` 拼接 code path；当前 code 行不保证存在可选的 `code_text`。Registry
至少保存 `num_levels`、`branching_factors`、`token_format`、`assignment_mode`、
`ordered_skill_ids_sha256` 和 bucket 映射：

```json
{"buckets":{"12/3":["weather"]}}
```

默认先记录 frozen encoder 的 raw nearest code，再执行 `balanced_hierarchical` 最终分配。
Collision 并非绝对禁止：registry bucket 可以包含多个 Skill，但必须满足配置的 collision
rate 和最大 bucket 门禁。

输出：

```text
index/
├── train_codes.jsonl
├── train_registry.json
├── virtual_tokens.txt
└── manifest.json
```

逻辑 artifact 为 `codes.directory`、`codes.train`、`codes.registry`、
`codes.virtual_tokens` 和 `codes.manifest`。

## 15. Stage 08：Router SFT 数据

### 15.1 Memorization

对每个有 code 的候选构造：

```json
{
  "phase": "memorization",
  "group_id": "weather",
  "skill_id": "weather",
  "input_text": "天气查询 | 查询天气、温度和降水信息 | 查询指定城市和日期的天气",
  "target_paths": [["<SK_L1_0>", "<SK_L2_3>"]],
  "target_tokens": ["<SK_L1_0>", "<SK_L2_3>"],
  "target_text": "<SK_L1_0><SK_L2_3>",
  "target_skill_ids": ["weather"]
}
```

默认 Memorization validation fraction 为 0，因此所有候选都进入训练覆盖。

### 15.2 Alignment

Alignment Query 必须恰有一个 qrel。Join 后构造成单 code path Retrieval 格式，写入
`retrieval_alignment_train.jsonl`；该 split 的 validation fraction 固定为 0。

### 15.3 Retrieval

对每条多 Skill Query：

1. 按 qrel `position` 取得有序 positive Skill；
2. 查找每个 Skill 的 code path；
3. 相同 code bucket 的 Skill 合并到同一 path；
4. 不同 path 以换行分隔；
5. 保留 `positive_skill_ids`、`path_skill_ids` 和 `target_paths`。

```json
{
  "phase": "retrieval",
  "group_id": "规范化 Query 组 ID",
  "query_id": "cq-...",
  "input_text": "查周末杭州天气……",
  "target_paths": [
    ["<SK_L1_0>", "<SK_L2_3>"],
    ["<SK_L1_1>", "<SK_L2_4>"]
  ],
  "target_text": "<SK_L1_0><SK_L2_3>\n<SK_L1_1><SK_L2_4>",
  "target_skill_ids": ["weather", "calendar"],
  "path_skill_ids": [["weather"], ["calendar"]],
  "positive_skill_ids": ["weather", "calendar"]
}
```

### 15.4 SFT 切分

训练/验证按 `group_id` 切分，避免相同规范化 Query 的变体跨 split。Retrieval 默认使用
`router.validation_fraction=0.02`，并确保 validation 中的每个 target 在 train 中仍有
足够 group 覆盖。

输出：

```text
router_data/
├── memorization_train.jsonl
├── memorization_validation.jsonl
├── retrieval_alignment_train.jsonl
├── retrieval_train.jsonl
├── retrieval_validation.jsonl
└── manifest.json
```

单候选 `alignment_only` 模式不构造多 Skill Retrieval。为保持下游文件契约稳定，算法将
Alignment 数据作为 retrieval passthrough，并在 manifest 中明确标记。

### 15.5 Join 门禁和 Artifact

构造前后必须同时满足：

- 新流水线 qrels 都带 `position`，且 position 连续、唯一；兼容旧输入时，无 position 的
  qrels 仅保留文件物理顺序，不能自行重排；
- Query 引用的每个 Skill 都必须存在 code；
- 每个 code index 都位于对应层 branching 范围；
- 每个 code token 都属于 `virtual_tokens.txt`；
- 去重后的不同 code path 保持第一次出现的 target 顺序；
- collision bucket 的多个 Skill 合并为一条 path，同时完整保留 `path_skill_ids`；
- Retrieval validation 中出现的每个 target 在 train 仍有覆盖。

逻辑 artifact 为：

```text
sft.directory
sft.manifest
sft.memorization.train
sft.memorization.validation
sft.retrieval.alignment.train
sft.retrieval.train
sft.retrieval.validation
```

底层 builder 还会写 `retrieval_alignment_validation.jsonl`，当前通常为空，且未单独注册为
逻辑 artifact；它仍被 `sft.directory` 的目录哈希覆盖。Router 训练采用 target-only
监督，即损失只计算输出的 code token，不把输入 Prompt 当成目标文本。

## 16. Provider Ledger、重试和恢复边界

### 16.1 Ledger 的作用域

Generation、Review 和 Embedding 的调用结果不保存在某次 attempt 内，而保存在 Stage 级
目录：

```text
stages/<stage>/ledger/<provider>/<operation>/
```

这样即使 attempt 因进程退出、网络错误或后续质量门禁失败而未完成，下次 attempt 仍能复用
已经耐久提交的 Provider `succeeded` 响应。默认 pending 调度上限为：

```yaml
checkpointing:
  llm_batch_records: 20
  embedding_batch_records: 100
```

这些值是 ledger 一次 `schedule_requests`/`schedule_embeddings` 可选择的 pending work 上限，
不改变 generation 的 Prompt batch 或 embedding API batch 大小，也不是“每个 shard 必须累积
多少行才落盘”。当前 LLM client 每次只向 ledger schedule 一个请求，因此
`llm_batch_records=20` 通常不会把 20 个 LLM 请求合并为一次提交；Embedding 调用才会明显
利用多记录调度上限。

### 16.2 稳定请求身份

LLM 请求 ID 由 ledger namespace、Prompt 内容哈希和请求语义共同确定；请求语义至少绑定
Provider operation、endpoint、model、temperature、最大输出 token 等影响响应的参数。
Embedding ID 同时绑定 namespace、候选项、输入文本和 embedding model。

因此：

- 完全相同且已持久化记录为 Provider `succeeded` 的请求直接复用；
- Prompt、模型或关键参数变化会得到新的请求身份；
- 失败响应保留作审计，但不阻止同一工作项重试；
- Provider 原始请求/响应只作为私有恢复证据，不直接进入正式 Dataset。

### 16.3 不可变分片和原子提交

Ledger 使用不可变 JSONL shard。Manifest 记录每个 shard 的路径、行数、大小和 SHA-256，
并通过临时文件加原子替换提交。恢复前必须验证：

1. manifest schema 和 ledger identity；
2. manifest 声明的 shard 全部存在；
3. shard 大小、哈希和记录数未变化；
4. 请求、响应或 embedding 行能按 schema 解析；
5. 稳定 ID 无冲突。

若进程在“shard 已落盘、manifest 尚未更新”的窗口退出，恢复逻辑可以在完整性校验通过后
收养 orphan shard；无法证明完整性的孤立文件不会被静默采用。

### 16.4 Provider 级重试和算法级 Repair

两类重试不能混为一谈：

| 层次 | 触发条件 | 处理方式 |
|---|---|---|
| Provider 重试 | 超时、HTTP 错误、响应不可解析 | 按 Provider `max_retries` 重发同一语义请求 |
| 算法 Repair | JSON 可读，但违反 Query schema 或本地语义规则 | 构造包含错误原因和旧输出的新 Prompt |
| Batch 降级 | 批量画像、Alignment 或 Review 调用失败 | 拆成单记录调用，隔离坏样本 |
| Stage 重试 | 进程退出或 Stage 门禁失败 | 新建 attempt，复用成功 ledger 记录 |

Retrieval Query 的 Repair 默认最多 `validation_retry_rounds=3`。Repair Prompt 改变后具有
新的 Prompt 哈希，但仍归属于同一 Workflow 工作项；最终只发布最新且完整有效的 variant
集合。

当前 ledger 的成功边界是“Provider 返回内容已解析为 JSON 对象”，不是“该 JSON 已通过
画像、Query 或 Review 的算法 validator”。具体顺序是：

```text
HTTP 成功 → JSON object 解析成功 → ledger 记录 succeeded
         → 调用方执行算法 schema/质量校验
```

因此，若初始 Prompt 和全部 Repair Prompt 都返回“JSON 可解析但算法无效”的固定结果，
下一 attempt 会从 ledger 重放相同 payload，不会自动重新付费请求，Stage 可能重复失败。
普通 `--force-stage` 会保留 Stage ledger，也不能改变这个结果。此时只有改变 Prompt/实现、
model 或影响 request ID 的语义参数以创建新请求身份，或者由运维显式处理对应私有 ledger，
才能真正重新调用 Provider。后者涉及恢复证据变更，必须在保留备份和审计记录的前提下进行。

“不重复付费调用”只适用于 `succeeded` response 已经耐久写入 ledger 的情况。Provider 已
返回、但进程在 ledger 成功记录提交前退出或持久化失败时，恢复无法证明该请求已经完成，
后续可能再次调用 Provider。这是外部调用与本地 durable commit 之间不可消除的窗口。

## 17. Run、Attempt 和 Artifact 数据布局

### 17.1 Run 目录

```text
<run_dir>/
├── run_manifest.json
├── artifact_registry.json
├── config/
│   ├── pipeline.source.yaml
│   ├── pipeline.resolved.yaml
│   ├── overrides.json
│   ├── environment.json
│   ├── provenance.json
│   ├── candidate_input.json
│   └── manual_alignment_input.json
├── source/
│   ├── candidates.input.jsonl
│   ├── manual_alignment.input.jsonl
│   ├── candidates.normalized.jsonl
│   ├── catalog.jsonl
│   └── candidate_manifest.json
├── stages/
│   ├── 00_ingest/
│   ├── 01_enrich/
│   ├── 02_plan_queries/
│   ├── 03_generate_queries/
│   ├── 04_review_queries/
│   ├── 05_finalize_dataset/
│   ├── 06_train_codebook/
│   ├── 07_assign_codes/
│   └── 08_build_sft/
└── logs/
    ├── pipeline.log
    ├── pipeline.jsonl
    └── stages/
```

### 17.2 Stage 目录

```text
stages/03_generate_queries/
├── stage_state.json
├── input_manifest.json
├── progress.json
├── ledger/
├── attempts/
│   ├── 0001/
│   │   ├── checkpoint_lineage.json
│   │   ├── commands.jsonl
│   │   ├── command_state/
│   │   ├── subprocess.log
│   │   ├── traceback.txt
│   │   └── output/
│   └── 0002/
└── output/
```

失败 attempt 的中间结果、原始日志和 traceback 永久保留；新的重试写入新的编号目录。
当前 Runner 的提交顺序是：

1. handler 完成自身 schema/质量门禁并返回；
2. Runner 在 attempt output 写 Stage manifest；
3. 通过 `os.replace` 把 attempt output 原子换入 Stage `output/`；
4. 对换入后的内容计算哈希并登记 `artifact_registry.json`；
5. 标记 Stage `completed`，最后写 `COMPLETED`。

因此物理 `output/` 的换入早于 Registry 哈希登记。若第 4 步因重复 logical name、读取失败
或 Registry 写入失败，Stage 状态为 failed，但 `output/` 可能已经包含这次未登记内容。
下游只通过 Registry 解析 artifact，所以不会消费它；下一 attempt 发布时会先把现有
`output/` 移到该 attempt 的 `previous-output/`，再换入新结果。

### 17.3 Artifact Registry

Stage 依赖逻辑 artifact 名，而不是猜测物理路径。每个记录至少绑定：

```json
{
  "logical_name": "dataset.queries.train",
  "path": "stages/05_finalize_dataset/output/dataset/queries_train.jsonl",
  "format": "jsonl",
  "artifact_schema": "router_querie/1",
  "producer": "finalize-dataset",
  "sha256": "...",
  "bytes": 123456,
  "created_at": "2026-09-01T12:00:00Z",
  "rows": 1000,
  "inputs": {"data.reviews": "..."},
  "config_hash": "...",
  "metadata": {}
}
```

`router_querie/1` 是当前 Registry 实现由 `queries.rstrip("s")` 产生的兼容拼写；qrels 对应
`router_qrel/1`。它们是实现中的实际 schema 标识，不应在消费端擅自规范化为另一个字符串。

已完成 Stage 只有在以下条件全部满足时才可复用：

1. `stage_state.status=completed` 且 `COMPLETED` 标记存在；
2. 上游 artifact 哈希不变；
3. Stage config view 哈希不变，其中包含相关配置和声明的实现文件哈希；
4. 对训练/编码 Stage，config view 中绑定的冻结 Run provenance 不变；
5. 每个输出仍能由 Registry 验证路径、内容哈希和 producer，且与 Stage state 记录一致。

任一条件变化都会使该 Stage 及依赖它的下游失效，不能以“文件还在”为理由继续使用。
复用检查不会重新运行该 Stage 的 schema validator 或质量门禁，也不会重新探测当前机器并
把它与冻结 provenance 比较；它信任成功执行时已经通过的门禁和已冻结的 provenance。

### 17.4 中断进程的恢复前提

同一 Run 同时只允许一个 Runner 持有执行锁。Stage 中断后，新 attempt 创建前会检查旧
command state 中的 host、PID、PGID 和进程出生身份：

- 确认本机进程及其进程组已经退出，或 PID 已被其他进程复用：允许标记 stale 并恢复；
- 本机仍有存活 leader/后代：拒绝启动新 attempt；
- 记录来自其他主机，或当前无法可靠确认进程状态：fail closed，拒绝恢复。

因此“Stage 进程退出”本身不是无条件可恢复事件。流水线必须先证明旧子进程不会与新
attempt 并发写相同数据；必要的终止流程使用 `SIGTERM`、有界等待、`SIGKILL`、再次有界
确认，无法确认退出时仍拒绝继续。

## 18. 配置模型和当前默认值

配置以 `configs/router_pipeline.yaml` 为基线。创建 Run 时完成环境变量展开和 CLI `--set`
覆盖，并把最终配置冻结为 `config/pipeline.resolved.yaml`。已有 Run 不接受临时 `--set`；
需要改变实验配置时应 fork 新 Run，以保留父实验的输入和血缘。

### 18.1 输入、Provider 和持久化

| 配置 | 默认值 | 算法作用 |
|---|---:|---|
| `run.seed` | 42 | Workflow 选择、切分和顺序增强的稳定随机源 |
| `input.id_policy` | `explicit_or_name` | 缺失 ID 时允许从名称生成 |
| `input.preserve_metadata` | `true` | 保留候选 metadata |
| `input.single_candidate_policy` | `alignment_only` | 单候选退化策略 |
| `providers.generation.concurrency` | 12 | generation 并发上限 |
| `providers.generation.timeout_seconds` | 300 | 单次 generation 超时 |
| `providers.generation.max_retries` | 3 | generation Provider 重试 |
| `providers.review.concurrency` | 12 | review 并发上限 |
| `providers.review.timeout_seconds` | 300 | 单次 review 超时 |
| `providers.review.max_retries` | 3 | review Provider 重试 |
| `checkpointing.llm_batch_records` | 20 | LLM ledger 单次 pending 调度上限 |
| `checkpointing.embedding_batch_records` | 100 | Embedding ledger 单次 pending 调度上限 |

### 18.2 Query 构造和审核

| 配置 | 默认值 | 算法作用 |
|---|---:|---|
| `alignment_queries_per_skill` | 10 | 每候选 Alignment 最低数 |
| `retrieval_positives_per_skill` | 20 | Alignment + 语义 Train Retrieval 联合覆盖下限 |
| `skills_per_query.min/max` | 2 / 4 | 基础 Workflow target 数范围 |
| `workflows_per_skill` | 3 | 每 anchor 的基础 Workflow 数 |
| `explicit_variants` | 3 | 当前实现中的**总 variant 数** |
| `implicit_variants` | 1 | 允许的 implicit variant 数；受总数和安全规则约束 |
| `order_variants` | 4 | 每条 Train Query 的 target 顺序版本上限 |
| `profile_batch_size` | 10 | 画像 Prompt batch |
| `alignment_batch_size` | 3 | Alignment Prompt batch |
| `query_batch_size` | 4 | Retrieval Prompt batch |
| `review_batch_size` | 10 | Review Prompt batch |
| `validation_retry_rounds` | 3 | Retrieval 本地校验 Repair 轮数 |
| `min_completion_rate` | 0.95 | 允许 abandon 少量 Workflow 的完成率门槛 |
| `min_augmented_train_queries` | 0 | 顺序增强后 Train Query 总数下限 |
| `alignment_backfill_rounds` | 5 | 初次 Alignment 回填上限 |
| `max_backfill_rounds` | 5 | Retrieval coverage 回填上限 |
| `final_alignment_backfill_rounds` | 5 | 最终联合覆盖 Alignment 回填上限 |
| `coverage_oversample_factor` | 3.0 | 按 Review 淘汰率放大的回填量 |
| `split.train/validation/test` | .90 / .05 / .05 | Dataset 语义 Workflow 切分 |

### 18.3 Embedding、Code 和 SFT

| 配置 | 默认值 | 算法作用 |
|---|---:|---|
| `providers.embedding.batch_size` | 8 | 单次 Embedding API batch |
| `providers.embedding.max_batch_chars` | 12000 | 一个 batch 的累计字符上限 |
| `providers.embedding.timeout_seconds` | 600 | 单次 Embedding 请求超时 |
| `providers.embedding.max_retries` | 5 | Embedding Provider 重试 |
| `code.mode` | `auto` | 自动或手工规划 CodePlan |
| `code.latency_priority` | `balanced` | 自动 CodePlan 目标 |
| `code.spare_capacity_ratio` | 1.25 | 编码容量余量 |
| `code.max_virtual_tokens` | 512 | 所有层 virtual token 总上限 |
| `code.max_branching_factor` | 256 | 单层 branching 上限 |
| `code.assignment` | `balanced_hierarchical` | 最终 code 分配策略 |
| `router.validation_fraction` | .02 | Retrieval SFT validation 比例 |
| `router.data_seed` | 42 | SFT group 切分稳定随机源 |

## 19. 端到端算法伪代码

```text
function BUILD_ROUTER_DATA(candidates_file, optional_manual_alignment, config):
    run = CREATE_RUN_AND_FREEZE_INPUTS(candidates_file,
                                       optional_manual_alignment,
                                       resolved_config=config)

    candidates = INGEST(run.frozen_candidates)
    assert candidates is not empty
    registry = FREEZE_ORDERED_CANDIDATE_REGISTRY(candidates)

    profiles = GENERATE_PROFILES_WITH_LEDGER(registry)
    profiles = ADD_CONFUSABLE_CANDIDATES(profiles)
    assert one compatible profile per candidate

    if registry.size == 1 and single_candidate_policy == alignment_only:
        workflows = []
    else:
        workflows = PLAN_DETERMINISTIC_WORKFLOWS(profiles, seed)

    alignment_queries = GENERATE_ALIGNMENT_QUERIES(profiles)
    retrieval_queries = GENERATE_AND_LOCALLY_VALIDATE_QUERIES(workflows)
    assert completion_rate(retrieval_queries, workflows) >= min_completion_rate

    alignment_reviews = REVIEW_ALIGNMENT(alignment_queries)
    alignment_queries, alignment_reviews = BACKFILL_ALIGNMENT_UNTIL_TARGET(...)
    alignment_queries, alignment_reviews = MERGE_FROZEN_MANUAL_ALIGNMENT(...)

    if workflows is not empty:
        retrieval_reviews = REVIEW_RETRIEVAL(retrieval_queries)
        repeat at most max_backfill_rounds:
            accepted = FILTER_LOCAL_PASS(retrieval_queries, retrieval_reviews)
            deficits = JOINT_TRAIN_COVERAGE_DEFICITS(alignment_queries, accepted)
            if deficits is empty: break
            new_workflows = PLAN_COVERAGE_WORKFLOWS(deficits, profiles)
            new_queries = GENERATE_AND_LOCALLY_VALIDATE_QUERIES(new_workflows)
            new_reviews = REVIEW_RETRIEVAL(new_queries)
            append all generated and review evidence

    alignment_queries, alignment_reviews = FINAL_ALIGNMENT_BACKFILL(...)

    accepted_alignment = FILTER_LOCAL_PASS(alignment_queries, alignment_reviews)
    accepted_retrieval = FILTER_LOCAL_PASS(retrieval_queries, retrieval_reviews)
    semantic_retrieval = DEDUPE_BY_NORMALIZED_CHAR_3GRAM(accepted_retrieval, 0.86)
    split = SPLIT_BY_WHOLE_WORKFLOW(semantic_retrieval, seed)
    split = REPAIR_ZERO_TRAIN_RETRIEVAL_COVERAGE(split)
    assert JOINT_COVERAGE(accepted_alignment, split.train) >= configured minima

    augmented_train = TARGET_ORDER_AUGMENT(split.train, order_variants, seed)
    dataset = WRITE_DATASET_AND_ORDERED_QRELS(registry,
                                               accepted_alignment,
                                               augmented_train,
                                               split.validation,
                                               split.test)
    VALIDATE_ORDERED_QRELS(dataset)

    processed = NORMALIZE_CLOSED_SET_DATASET(dataset)
    graph = BUILD_TRAIN_COLLAB_GRAPH(processed, dedupe_by_source_query_id=true)
    embeddings = EMBED_NORMALIZE_AND_ORDER_CANDIDATES(processed.catalog)

    code_plan = PLAN_CODE_SPACE(candidate_count, config.code)
    codebook = TRAIN_CODEBOOK(embeddings, graph, code_plan)
    codes = ASSIGN_AND_VALIDATE_CODES(codebook, embeddings, registry)

    sft_memorization = JOIN_CANDIDATE_DOCS_TO_CODES(registry, codes)
    sft_alignment = JOIN_ORDERED_QRELS_TO_CODES(dataset.alignment, codes)
    sft_retrieval = JOIN_ORDERED_QRELS_TO_CODES(dataset.train, codes)
    sft_train, sft_validation = GROUP_SPLIT_WITH_COVERAGE_GUARD(sft_retrieval)

    return {
        dataset, processed, embeddings, graph,
        code_plan, codebook, codes,
        sft_memorization, sft_alignment, sft_train, sft_validation
    }
```

## 20. 质量门禁和失败语义

### 20.1 可局部恢复的问题

以下问题保留证据并重试，不应污染正式输出：

- Provider 超时、限流、临时 HTTP 错误；
- Batch 中个别记录返回非法 JSON；
- Query 缺少 evidence、文本过短、语言比例错误或 implicit 声明不一致；
- Review batch 失败；
- Embedding batch 中断；
- Stage 进程意外退出，且 Runner 能证明相关本机进程组已经终止。

这里的“重试”不保证换取新的 Provider 响应：若 payload 已作为 JSON 成功写入 ledger、只是
外层算法校验失败，后续 attempt 会重放它，必须按 16.4 改变请求身份或显式处理 ledger。

### 20.2 Stage 必须失败的问题

以下问题说明正式数据无法证明正确，Stage 不得标记完成：

- 冻结候选文件哈希漂移、候选为空、Skill ID 重复；
- 路由画像或 Alignment variant 未覆盖全部候选；
- Retrieval Workflow 完成率低于 `min_completion_rate`；
- Review 与 Query 的 schema 或 `query_hash` 不匹配；
- 每候选 Alignment 或联合 Train 正例覆盖低于配置门槛；
- 顺序增强后的 Train Query 数低于 `min_augmented_train_queries`；
- Query、qrels、候选 Registry、position 任一不一致；
- Dataset manifest 声明的大小或 SHA-256 与文件不一致；
- Embedding shape、顺序 hash 或模型 provenance 不一致；
- CodePlan 容量不足，或 code collision/utilization/entropy 门禁失败；
- SFT 中存在无 code 的 Skill、未知 virtual token 或有序 target 漂移。
- 上一 attempt 的子进程仍存活，位于其他主机，或其终止状态无法可靠确认。

失败 Stage 不登记可供下游解析的正式 artifact，也不会写 `COMPLETED`；已经落盘的 attempt
和 ledger 仍可用于诊断与恢复。按 17.2 的当前提交顺序，失败后物理 `output/` 目录可能已
被换入，但未登记内容不具备 artifact 身份，不能手工绕过 Registry 使用。

### 20.3 最终接受条件

Stage 08 完成至少意味着：

1. 候选 Registry 非空且顺序稳定；
2. 每候选存在规定数量的通过审核 Alignment 数据；
3. 多候选模式下，每候选满足联合 Train 正例覆盖；
4. 所有正式 Query 的 qrels 完整、有序且仅指向候选集；
5. 评估 split 与 Train 不共享同一 Workflow；
6. 顺序增强没有虚增覆盖统计和协同图边权；
7. Embedding、Codebook、Skill code 与相同候选顺序绑定；
8. 每个 SFT target 都能被 `virtual_tokens.txt` 和 code registry 解释；
9. 所有正式文件均已登记 artifact schema、hash 和 lineage。

## 21. 边界情况与当前实现注意事项

### 21.1 单候选

默认 `input.single_candidate_policy=alignment_only`：

- Stage 02 写空 Workflow；
- Stage 03 写空 Retrieval Query，但正常生成 Alignment；
- Stage 04 仅执行 Alignment 审核和回填；
- Stage 05 仍生成 Candidate、Alignment、Embedding 和单节点图；
- CodePlan 容量为 1，允许一个单 token code；
- Stage 08 以 Alignment 作为 Retrieval passthrough。

若 policy 为 `error`，Stage 00 直接失败。

### 21.2 不同长度的 Skill code

CodePlan 可以为小候选集选择一层，因此“每个候选一个 token”是合法结果；多层时所有候选
在同一 Run 内使用固定且相同的 path 长度。SFT consumer 必须读取 CodePlan 和 registry，
不能写死两层或三层编码。

### 21.3 Collision bucket

Code collision 受质量阈值控制，并非无条件禁止。若 registry 的同一 bucket 含多个 Skill，
SFT 中这些 Skill 共享一条 target path，`path_skill_ids` 保存 bucket 成员；不能为同一路径
重复输出 token。

### 21.4 五个容易误读的配置和统计口径

1. `explicit_variants=3` 在当前下层接口中表示总 variant 数；默认是 2 条 explicit 加
   1 条 implicit，而不是 3 加 1。全 target 都是高影响动作时改为 3 条 explicit。
2. `retrieval_positives_per_skill=20` 是 Alignment 与未增强 Train Retrieval 的联合覆盖，
   不是要求每个 Skill 有 20 条纯多 Skill Query。
3. Embedding 在 Stage 05 `finalize-dataset` 生成，不在 Stage 01 `enrich`。
4. Generation Prompt 要求 Query 为 25～180 字；当前本地 validator 实际接受 25～220 字。
   正式数据以本地 validator 为硬门禁。
5. `order_variants=4` 是上限；2-target 最多只有 `2! = 2` 种不同顺序。

### 21.5 当前未生成的内容

- 没有独立显式负例文件；
- 没有单独的正式 `rejected_queries.jsonl`；审核失败证据保存在 Review artifact 和错误伴随文件；
- Embedding 不参与基础 Workflow 规划；
- `split_hint` 不是现代基础 Workflow 的最终切分权威，最终 split 由稳定 hash 决定；
- 人工 Alignment 只能来自 Run 创建时冻结的输入，不能在运行中悄悄替换。

## 22. 独立执行、恢复和实验迭代

创建新 Run 并执行到数据生成完成：

```bash
python scripts/train_candidates.py run \
  --candidates data/my_candidates.jsonl \
  --config configs/router_pipeline.yaml \
  --output runs/my-router \
  --to build-sft
```

恢复已有 Run 的某一范围：

```bash
python scripts/train_candidates.py run \
  --run-dir runs/my-router \
  --from review-queries \
  --to build-sft
```

仅执行一个 Stage：

```bash
python scripts/train_candidates.py stage finalize-dataset \
  --run-dir runs/my-router
```

查看 Stage 状态和 artifact：

```bash
python scripts/train_candidates.py status --run-dir runs/my-router
```

若要修改参数开展新实验，使用 fork，而不是修改已有 Run 的 resolved config：

```bash
python scripts/train_candidates.py fork \
  --from-run runs/my-router \
  --output runs/my-router-exp2 \
  --set data_generation.order_variants=2
```

`--force-stage <name>` 会使指定 Stage 及其下游正式结果失效并重跑，但保留旧 attempts 和
Provider ledger。只要请求身份没有变化，Provider `succeeded` 结果仍会复用；这也包括
“JSON 可解析但未通过外层算法 validator”的结果，详见 16.4。

## 23. 实现映射

| 算法模块 | 当前主要实现 |
|---|---|
| Run 创建、输入冻结、恢复 | `src/llmgen/pipeline/runner.py` |
| Stage 状态 | `src/llmgen/pipeline/state.py` |
| Artifact Registry | `src/llmgen/pipeline/artifacts.py` |
| Provider ledger | `src/llmgen/pipeline/ledger.py`、`providers.py` |
| Stage 00～08 编排 | `src/llmgen/pipeline/stages/` |
| 画像、Workflow、Retrieval Query、Review、Dataset | `src/llmgen/clawhub_dataset.py` |
| Alignment Query | `src/llmgen/clawhub_alignment.py` |
| Dataset prepare、图、Embedding | `scripts/prepare_closedset.py` |
| CodePlan | `src/llmgen/pipeline/code_plan.py` |
| Codebook 训练和编码导出 | `scripts/train_tokenizer.py`、`scripts/export_skill_codes.py` |
| Ordered qrels | `src/llmgen/pipeline/schema.py` |
| Router SFT | `scripts/build_router_data.py`、`src/llmgen/router.py` |

本文所说的“默认值”均指当前 `configs/router_pipeline.yaml`；CLI override、环境变量或 fork
后的 resolved config 可以改变数值，但不能绕过 schema、候选 Registry、ordered qrels、
lineage 和质量门禁等正确性不变量。
