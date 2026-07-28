# 冻结 Codebook 的候选增删基线

## 设计目标

这一版只验证低成本增量是否可行，保持两个不变量：

1. Stage 1 ToolWeaver encoder 和各层 codebook 完全冻结，已有 Skill 的 code 永不改变。
2. 推理时只有一套 active candidates。`skill_decode_map.json` 中的 Skill、code bucket
   和 trie path 始终同步增删。

候选状态采用小型 overlay，不复制 Router 权重：

```text
candidate-state/
├── skill_decode_map.json
├── virtual_tokens.txt
└── operation.json
```

Web Runtime 仍从原目录加载模型和 tokenizer，但从 overlay 的 active paths 重建
`MultiPathTokenTrie`。每次操作记录父状态 hash 和全部祖先 hash，因此可以连续增删，
同时仍能校验它来自当前 Router 的原始候选集。

## 删除

删除会从 decode map 移除 Skill，并同步重建 trie。若多个 Skill 共用同一 code，
默认只移除目标 Skill，并保留该 path 及桶内其他成员。显式传
`--disable-shared-path` 才会删除整个碰撞桶及对应 trie path。

## 新增

新 Skill 文档先由与原索引一致的 Embedding 服务编码，再经过冻结的 Stage 1
encoder 和 residual codebooks。

- `nearest`：标准逐层最近 code，允许落入已有碰撞桶。
- `nearest_available`（默认）：旧 code 不动，以 beam 搜索选择距离最近的空闲完整
  path，保证新增 Skill 可以独立启用和删除。

两种运行模式共用同一个候选状态：

- `index_only`：只启用新 path，直接验证原 Router 对新能力的零样本泛化。
- `lora_train`：额外构造 1 条 Skill 文档到 code 的 Memorization 样本，以及默认
  10 条“小艺”风格的单 Skill query；仅用这些样本训练增量 LoRA。

增量 LoRA 从已训练 Router 初始化。若源 Router 是全参数模型，输出是纯 LoRA
delta，既有虚拟 token embedding 保持冻结；若源 Router 本身是 PEFT adapter，
则在一个新输出目录中继续训练该 adapter，源目录不会被修改。

## 当前边界

- `index_only` 的效果完全依赖原 Router 的语义泛化能力。
- `lora_train` 按基线要求不 replay 旧候选，可能改变旧候选分布；评估时应同时测试
  新 Skill 命中率与原测试集回归。
- 空闲 path 搜索受 `--assignment-beam-size` 限制；找不到时可扩大 beam，或使用
  `nearest` 接受碰撞。
- overlay 变化后需重启 Web Server；当前没有进程内热更新。
