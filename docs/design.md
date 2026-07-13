# LLMGen：面向动态 Skill 集合的层次生成式 Tokenizer

状态：v0.1（已落地完整训练与推理闭环）
范围：给定用户 query，LLM 以极少的生成 token 召回一个或多个 Agent Skills。

## 1. 设计结论

在线模型不生成 skill 名称、描述或 JSON，而只生成固定长度的层次码：

```text
query -> <SK_L1_...> <SK_L2_...> ... <SK_LL_...>
                                      |
                                      v
                         code -> [skill_id, ...]
```

其中：

- 层数 `L = num_levels` 可配置；一条候选路径始终只生成 `L` 个 token。
- 每层 token 都注册为一个不可再切分的 special token。
- 一个完整 code 对应一个小 bucket，而不是强制对应唯一 skill。
- 多候选优先通过 constrained beam search 得到多条长度为 `L` 的路径，再由 bucket 展开；不让模型在线生成长候选列表。
- tokenizer 支持两种可替换策略：
  - `interpretable`：由人工 taxonomy / Skill Card 字段构成可解释层次；
  - `balanced`：学习 ToolWeaver 风格的多级残差 codebook，并用 Sinkhorn 平衡分配抑制 code collapse。
- 日常新增/删除只修改 tokenizer registry 和约束解码 trie。旧 skill 的 code 不变，也不修改 LLM 词表。
- 全量重训、层数变化或 codebook 变化必须产生新 `codebook_version`，不能原地覆盖。

这套设计刻意把三个职责拆开：

```text
层次构造策略 -> 固定长度 HierarchicalCode -> 动态 SkillRegistry / 解码 Trie
```

两种 tokenizer 只负责构造 code；候选展开、增删、序列化和约束解码共用同一套逻辑。

## 2. 为什么采用 bucket code

若要求“一条 code 唯一对应一个 skill”，新增 skill 常常需要拆桶、重聚类或追加唯一后缀，最终导致 code drift 或生成长度增长。这里允许多个 skill 落入同一叶 bucket，并在外部做小规模精排：

```text
<SK_L1_3><SK_L2_7><SK_L3_1>
  -> [calendar.create, calendar.reschedule, calendar.cancel]
  -> reranker(query, Skill Cards)
  -> selected skills
```

好处是：

- 在线生成长度只取决于 `L`，不随 skill 总数增长；
- 新增 skill 可以直接加入现有 bucket；
- 删除 skill 只移除 registry membership；
- code 冲突成为可控的候选召回，而不是编码错误。

因此本项目优化的是“高召回的短码路由”，而不是让生成模型独自完成最终精排。

## 3. 与相关工作的关系

### 3.1 ToolGen

[ToolGen](https://arxiv.org/abs/2410.03439) 提出把工具选择转为生成特殊 token：模型输出短 token 序列，运行时再 grounding 到真实工具。不同点是本方案不为每个 skill 永久绑定一个独立原子 token，而使用可组合层次 token，控制词表规模并支持动态集合。其官方实现见 [Reason-Wang/ToolGen](https://github.com/Reason-Wang/ToolGen)。

### 3.2 ToolWeaver

[ToolWeaver](https://arxiv.org/abs/2601.21947) 用 `L` 个 codebook、每层 `K` 个离散 code 表示工具，词表开销是 `L*K`，组合容量是 `K^L`；其量化器使用工具固有语义和协同使用信号，并通过最优传输 / Sinkhorn-Knopp 平衡分配。官方研究代码见 [Fwibo/ToolWeaver](https://github.com/Fwibo/ToolWeaver)。

本项目保留 NumPy residual-centroid 后端用于算法单测和 registry 原型；正式 SkillRet
训练链路将 ToolWeaver 的 `RQVAE`、`ResidualVectorQuantizer`、`VectorQuantizer` 和
Sinkhorn 四个模型模块固定在 `src/llmgen/vendor/toolweaver/`，训练完整 MLP encoder /
decoder 与多层 codebook。官方来源固定为 commit
`3a102bad2d85f9674a7febdbaed0235d137e7222`，并记录实际复制所用本地 checkout 的
commit `a7684edaf2bb3af7ff6928c34e27a324599deda0`；checkpoint 同时记录上游与仓库内副本的
SHA-256，恢复训练和加载索引时会拒绝源码漂移。新机器只需 clone 本仓库。

正式后端额外实现：

- 只用训练集 qrels 构造稀疏 collaborative graph；
- edge-aware sampler，确保协同边真正进入 batch；
- reconstruction、residual VQ、稀疏图损失的联合反向传播；
- optimizer、scheduler、AMP、RNG 和断点恢复的完整 checkpoint；
- codebook 冻结后，以逐层残差最近邻为新 / test skill 编码；
- code 到 skill 的一对多关系由外部 registry 管理。

ToolWeaver 本身不是动态目录协议。这里的“冻结 codebook + registry 增删 + 版本化迁移”是为动态 Skill 集合增加的系统层设计，不能表述为论文原生能力。

实现核对时重点参考官方仓库的 [RQ-VAE 外壳](https://github.com/Fwibo/ToolWeaver/blob/main/index/models/rqvae.py)、[多层残差量化](https://github.com/Fwibo/ToolWeaver/blob/main/index/models/rq.py)、[VQ 与 Sinkhorn](https://github.com/Fwibo/ToolWeaver/blob/main/index/models/vq.py)、[KMeans/Sinkhorn 基础层](https://github.com/Fwibo/ToolWeaver/blob/main/index/models/layers.py)和 [Trie 约束](https://github.com/Fwibo/ToolWeaver/blob/main/training/models/utils.py)。本实现额外修正以下工程边界：

- 对所有逐层配置做严格长度校验，避免 `zip` 截断后静默少建层；
- codebook 使用 list 表达，允许每层 `K_l` 不同；
- token 名称按任意 `L` 动态生成，不硬编码二至五层；
- `balance_scope` 显式选择 `last` 或 `all`，避免论文描述和代码默认值差异被隐藏；
- 不复用其碰撞后追加可变长 suffix 的导出逻辑，始终保持固定 `L`，碰撞交给 bucket。

### 3.3 GRLM / Structured Term Identifiers

[GRLM](https://aclanthology.org/2026.findings-acl.984/) 说明有顺序、结构化的短语义标识符能同时提供一致性和局部区分度。本方案吸收“前层稳定、可共享，后层负责区分”的原则，但不直接生成自然语言 term：自然语言 term 往往会被拆成多个 subtoken，延迟和格式约束都弱于 special-token code。

## 4. 在线输出契约与时延

设每层分支数为 `K_1 ... K_L`：

- 新增 special token 数：`sum(K_l)`；
- 理论 code 容量：`product(K_l)`；
- 单条路径的生成长度：严格为 `L`；
- beam size 为 `B` 时，得到至多 `B` 个 code hypotheses，但每个 hypothesis 仍只有 `L` 步。

例如 `L=3, K=(64, 64, 64)`：

- 词表仅增加 192 个 special tokens；
- 理论上有 262,144 条 code 路径；
- 单路径只生成 3 个 token。

约束解码规则：

1. 第 `l` 步只开放第 `l` 层 namespace 中、且能通向非空 bucket 的 token；
2. 生成满 `L` 层后只允许 EOS；
3. 删除最后一个 bucket member 后，该完整路径立即从 active trie 隐藏；
4. 已预留但当前为空的路径不会出现在 `valid_next_tokens` 中。

多 skill 有两种语义，必须区分：

- **候选召回**：使用 beam 的多条 code 或同一 bucket 的多个 skill，不增加单条序列长度；
- **必须同时执行的 skill chain**：属于后续 planner 阶段，不在 tokenizer 输出中串联任意数量的 code。

## 5. 配置模型

统一配置示意：

```json
{
  "strategy": "balanced",
  "num_levels": 3,
  "branching_factors": [16, 16, 16],
  "codebook_version": "skills-v1",
  "token_format": "<SK_L{level}_{index}>",
  "random_seed": 7,
  "bucket_capacity": null,
  "overflow_policy": "allow",
  "balance_scope": "all",
  "sinkhorn_temperature": 0.05,
  "sinkhorn_iterations": 50,
  "clustering_iterations": 20,
  "collaborative_weight": 0.25,
  "dynamic_balance_weight": 0.1
}
```

约束：

- `num_levels >= 1`；
- `len(branching_factors) == num_levels`；
- 每个 `branching_factor >= 1`；
- `token_format` 必须同时包含 `{level}` 和 `{index}`；
- 改变层数、分支数、token 格式或已训练 codebook 时必须改变 `codebook_version`；
- `bucket_capacity=null` 表示允许 bucket 自然碰撞，由下游精排控制最终候选数。
- `bucket_capacity` 非空且 `overflow_policy=error` 时是硬上限；`allow` 时容量仅作为监控阈值，仍允许共享 bucket。

`balance_scope` 支持：

- `all`：每层都做平衡分配，适合独立 tokenizer，希望最大化各层 code 利用率；
- `last`：仅最后一层使用 Sinkhorn，更接近 ToolWeaver 的实现选择；前层采用普通残差聚类。

## 6. 统一数据与接口

### 6.1 SkillRecord

```python
SkillRecord(
    skill_id: str,
    name: str,
    description: str,
    hierarchy: tuple[str, ...],
    embedding: tuple[float, ...],
    collaborative_embedding: tuple[float, ...],
    metadata: dict[str, object],
)
```

- `interpretable` 至少需要长度为 `L` 的 `hierarchy`；
- `balanced` 至少需要非空、等维的 `embedding`；
- collaborative embedding 是可选项，缺失时以全零向量处理；
- embedding 的生成不属于 tokenizer 在线路径，应在 Skill Card 入库时离线完成。

### 6.2 HierarchicalCode

```python
HierarchicalCode(
    indices: tuple[int, ...],
    tokens: tuple[str, ...],
    codebook_version: str,
)
```

`indices` 和 `tokens` 长度都必须等于 `num_levels`。持久化时同时保留整数 code 和 token 字符串，加载时交叉校验，防止 token 映射被静默重排。

### 6.3 SkillTokenizer

```python
fit(skills) -> None
encode(skill_or_id) -> HierarchicalCode
decode(prefix) -> tuple[str, ...]
add(skill) -> HierarchicalCode
remove(skill_id) -> bool
valid_next_tokens(prefix) -> tuple[str, ...]
snapshot() -> dict
```

语义：

- `decode(full_code)` 返回叶 bucket；
- `decode(prefix)` 返回该子树下所有 active skills，方便调试和分层召回；
- `add` 不允许覆盖已有 `skill_id`；
- `remove` 幂等，返回是否实际删除；
- `fit` 是创建一个 codebook 版本的离线动作，不是日常增删接口。

## 7. Tokenizer A：可解释层次

### 7.1 输入

每个 Skill Card 提供一条 taxonomy path，例如三层：

```text
productivity / calendar / mutate-event
productivity / calendar / read-event
engineering  / source-control / pull-request
```

当 `num_levels=2` 时只读取前两层；当 path 短于 `L` 时拒绝入库，不用隐式 `unknown` 掩盖数据质量问题。

### 7.2 编码

每个 prefix 内维护 append-only 的 `label -> child_index` 映射：

```text
()                         productivity -> 0, engineering -> 1
(0,)                       calendar -> 0
(0, 0)                     mutate-event -> 0, read-event -> 1
(1,)                       source-control -> 0
```

同一个 token（如 `<SK_L2_0>`）的含义由其 prefix 决定。这种上下文相关 child index 可以用 `K^L` 个组合表达 taxonomy，同时每层仍只需 `K` 个 special token。

稳定性规则：

- 新 label 使用当前 prefix 下最小的未占用 index；
- 删除 skill 不回收 label index，避免旧 checkpoint 的语义漂移；
- 超过该层 `branching_factor` 时按 `overflow_policy` 报错或显式允许共享；首版默认报错；
- 多个 skill 拥有相同完整 hierarchy 时自然进入同一 bucket。

### 7.3 可解释性输出

manifest 保存每个 prefix 的 `index -> label`，因此可以把模型 code 还原为：

```text
<SK_L1_0><SK_L2_0><SK_L3_1>
  == productivity / calendar / read-event
```

这份映射用于审计、数据构造和 UI，不让 LLM 额外生成 label 文本。

## 8. Tokenizer B：学习式平衡残差聚类

### 8.1 语义输入与协同图

语义输入使用 SkillRet 官方定义的文本：

```text
name | description | skill_md
```

默认由独立部署的 `Qwen/Qwen3-Embedding-8B` 通过 OpenAI-compatible
`/v1/embeddings` 离线生成向量，客户端校验响应顺序并做 L2 归一化；本地
Sentence Transformer 后端仅作为兼容与 smoke 路径。embedding 模型不参加
RQ-VAE 反向传播。协同信号不拼接成另一个 embedding，而是由 train qrels
直接构造图。设 `C_uv` 是 skill `u,v` 同为一个 query
正例的次数，`C_uu` 是 `u` 出现的 query 数：

```text
A_uv = C_uv / sqrt(C_uu * C_vv)
```

图只保存非零无向边；test qrels 不允许进入 embedding、图或训练数据，只在最终
评测读取。训练时先采样边端点，再补充随机节点，因此不会出现“随机 batch 中几乎
没有协同边”的稀疏性问题。

### 8.2 多级残差 codebook

初始化 `r_0 = x`，对每层 `l = 1..L`：

```text
i_l = assign(r_{l-1}, C_l)
r_l = r_{l-1} - C_l[i_l]
code(x) = (i_1, ..., i_L)
```

`C_l` 有 `K_l` 个可训练码字。前层拟合主要语义，后层拟合 residual；层数和
每层 `K_l` 均由 list 配置，允许异构分支数，并严格要求
`len(K)==len(sk_epsilons)==L`。

神经训练目标为：

```text
L_total = L_reconstruction
        + quant_weight * (
            mean_l(L_codebook_l + beta * L_commit_l)
            + graph_lambda * sum_(u,v) A_uv ||zq_u-zq_v||^2 / sum_(u,v) A_uv
          )
```

需要平衡的层在训练 assignment 时运行 ToolWeaver 的 Sinkhorn。默认采用论文代码
常见的 `last` 配置（只平衡末层），也可给每层非零 epsilon。导出与新增 skill 时
固定 `use_sk=False`，只做 `L` 次最近码字查找，不把 Sinkhorn 带入在线路径。

### 8.3 动态新增

codebook 冻结后，新 skill 逐层执行：

```text
score(i) = squared_distance(residual, centroid_i)
           + dynamic_balance_weight * usage_ratio(i)
```

其中 `usage_ratio(i)` 是该层当前落到 code `i` 的 skill 数除以该层 active skill 总数。选择最小 score，减去该 centroid 后进入下一层。`dynamic_balance_weight=0` 即纯最近邻；较小正值能减缓持续新增导致的热点，并让参数尺度不随目录总量线性增长，但不会移动旧 skill。

这保证：

- 新增不触发全量聚类；
- 旧 skill code 完全不变；
- 在线只做有限矩阵距离计算；
- 新旧 skill 可以共享叶 code，由 bucket 展开。

当新增数据分布与训练集明显不同，最近邻仍会工作，但 bucket 会失衡、语义会变差。应监控并在阈值触发后离线训练新版本，而不是偷偷改旧 code。

### 8.4 动态删除

删除仅执行：

1. `skill_id -> code` 删除；
2. `code -> members` 删除该 member；
3. 更新每层 usage；
4. 若 bucket 为空，从 active trie 隐藏路径。

centroid 和 token ID 不删除、不重排。

## 9. Registry 与 constrained decoding

Registry 至少持有：

```text
skill_id -> SkillRecord
skill_id -> full code
full code -> ordered skill_ids
active prefixes -> valid next indices
```

`valid_next_tokens(prefix)` 从 active bucket 派生。例如：

```text
active codes = {(0, 1, 2), (0, 1, 4), (3, 2, 1)}

prefix ()       -> L1: {0, 3}
prefix (0,)     -> L2: {1}
prefix (0, 1)   -> L3: {2, 4}
prefix (0, 2)   -> {}
```

这可以直接接 Hugging Face `prefix_allowed_tokens_fn` 或其他推理引擎的 logits processor。具体模型适配器不放进 tokenizer 核心包，避免绑定某个 serving 框架。

Tokenizer 外层使用同一把可重入锁串行化 `fit/add/remove/snapshot` 以及策略 usage/taxonomy 更新，保证 registry 与 strategy state 是一个事务。`SkillRegistry` 仍单独保护自身读写，但调用方不应绕过 tokenizer 直接修改其公开 registry 对象。

## 10. 版本与迁移

每份快照包含：

- `schema_version`；
- `codebook_version`；
- tokenizer 完整配置；
- token namespace；
- strategy-specific state；
- Skill Records、skill-code assignments、bucket memberships；
- 随机种子和 embedding 维度。

兼容性原则：

| 变化 | 是否原地兼容 | 处理方式 |
|---|---:|---|
| 新增 skill | 是 | 冻结 tokenizer，更新 registry/trie |
| 删除 skill | 是 | 更新 registry/trie |
| 修改 skill 描述但不重编码 | 是 | 只更新 metadata/reranker index |
| 重新编码已有 skill | 否 | 新 codebook version |
| 修改 `num_levels` | 否 | 新 special-token 集、新训练数据、新 checkpoint |
| 修改 branching factors | 否 | 新 codebook version |
| 全量重聚类 | 否 | 新 codebook version，双读迁移 |

推荐发布协议：

1. 离线生成 `v_next` codebook 和训练数据；
2. 训练 / 对齐能输出 `v_next` token 的 router checkpoint；
3. 线上短期同时加载 `v_current` 和 `v_next` registry；
4. 灰度切换 router；
5. 观察召回、空 bucket、bucket size、code usage entropy；
6. 下线旧版本。

## 11. 完整训练与推理接口

流水线分两阶段：

```text
SkillRet skill text -> semantic embeddings
train qrels         -> sparse collaborative graph
                     -> Stage 1 ToolWeaver RQ-VAE
frozen RQ-VAE       -> train/test fixed-L codes + collision buckets
train queries/qrels -> Stage 2 causal-LM SFT
test active trie    -> fixed-L constrained beam -> skill candidates
```

Stage 1 只拟合 train skill；test skill 模拟新增目录项，由冻结 encoder/codebook 编码。
Stage 2 默认使用 `Qwen/Qwen3-1.7B`，也接受其它 Hugging Face
`AutoModelForCausalLM` 兼容的 Qwen3 模型，并支持 LoRA 或全参数训练。Qwen3
chat template 的 thinking 模式固定关闭，保证 assistant 起始位置直接监督 code。
Stage 2 先做 `skill text -> code` memorization，再做 `query -> code` retrieval。prompt
部分 label 设为 `-100`，只对恰好 `L` 个 code token 与 EOS 计算 loss。多正例 query
按不同 code path 展开为多条样本，而不是生成一个很长的 code 列表。

单正例样本：

```json
{
  "query": "把明天下午的周会改到三点",
  "target_tokens": ["<SK_L1_0>", "<SK_L2_0>", "<SK_L3_0>"]
}
```

多个等价 skill/code 时，每个 distinct 正例 code 各构造一条训练样本；同 bucket
内的多个 skill 不重复构造相同 target。

推理：

1. 对 `L` 步做 trie-constrained beam search；
2. 将 beam code 展开为 skill buckets；
3. 去重并限制候选数量；
4. 用轻量 cross-encoder / LLM scorer 读取 query 与短 Skill Cards 精排；
5. 返回 top-k skills 给 planner 或执行器。

## 12. 监控指标

离线 tokenizer：

- 每层 code usage entropy；
- 每层最大 / 最小 usage；
- 空 code 比例；
- 叶 bucket size 的 p50 / p95 / max；
- 同 taxonomy / 高相似 skill 的 prefix overlap；
- 新增 skill 的量化距离和 residual norm。

在线 router：

- valid-path rate；
- code Recall@B；
- bucket-expanded Skill Recall@K；
- 精排后的 Skill Recall / MRR；
- 平均生成步数（应等于 `L`）；
- 首 token / 完整 code 延迟；
- 删除 skill 被召回率（应为 0）；
- 新增 skill 冷启动召回率。

触发新版本的建议条件不是固定常数，应由数据规模标定。首版至少暴露：bucket p95、usage entropy、量化距离 p95 和 active path 数，便于制定阈值。

## 13. 代码范围与产物

本仓库实现：

- 可配置 `num_levels` 和逐层 branching factors；
- 统一数据模型、token 格式和 registry；
- 可解释 taxonomy tokenizer；
- NumPy 多级残差 centroid 参考后端；
- 内置固定版本的 ToolWeaver RQ-VAE，提供完整神经 Stage 1；
- SkillRet 固定 revision 下载、严格校验、OpenAI-compatible 语义 embedding 与
  train-only 协同图；
- edge-aware 图训练、AMP、scheduler、完整 checkpoint/resume 和量化指标；
- train/test 固定长度 code、完整 special-token namespace 与 collision buckets；
- Qwen3 causal LLM 的 memorization/retrieval 两阶段 SFT，支持 full、LoRA 和
  DeepSpeed；
- active-trie constrained beam 推理及 NDCG/Recall/MAP/MRR/Completeness；
- 不依赖 bucket 内任意 tie-break 的 code recall 与 bucket-expanded recall；
- skill 动态 add/remove；
- prefix decode 与 valid-next-token 查询；
- JSON 快照保存 / 恢复；
- full/smoke 脚本和可人工验证的单元测试。

每个阶段写入 manifest，并绑定 dataset revision、ordered skill-ID hash、embedding 和
上游 checkpoint hash。当前不包含 bucket 内 reranker、多版本线上控制面和生产 serving
封装；它们消费稳定的 `code -> [skill_id,...]` 契约，不影响训练与离线评测闭环。
