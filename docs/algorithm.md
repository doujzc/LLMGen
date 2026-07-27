# LLMGen 层级生成式 Skill 路由：训练与推理算法

本文档以论文 Method Section 的形式定义当前 LLMGen 的训练与推理算法。讨论范围仅包括问题建模、层级 Skill 编码、Router 课程学习、约束解码及其性质；不涉及数据文件、训练脚本、分布式框架、硬件或部署。

当前默认协议是闭集路由：训练、验证、测试和推理共享同一个候选 Skill 集合。层数 \(L\) 可配置，当前主要实例均取 \(L=2\)。默认编码器是 ToolWeaver 风格的学习式平衡残差量化器；可解释 taxonomy 编码是可替换的非默认方案。

## 1. 问题定义

给定候选 Skill 集合

$$
\mathcal S=\{s_i\}_{i=1}^{N},
$$

其中 \(s_i\) 包含名称、能力说明和 Skill 文档 \(d_i\)。对于用户请求 \(q\)，目标是从同一候选集合中选出一个或多个 Skill：

$$
\mathcal Y_q=(s_{q,1},s_{q,2},\ldots,s_{q,m_q}),\qquad s_{q,j}\in\mathcal S.
$$

\(\mathcal Y_q\) 是有序序列，而不是无序标签集合。一个请求可以同时包含显式意图和隐式意图；只要某个 Skill 是完成请求所必需的，它就进入监督序列。

直接让语言模型生成 Skill 名称或完整工具描述会产生较长输出。为此，为每个 Skill 分配一个固定长度的层级标识符：

$$
c(s_i)=
\left(t^{(1)}_{a_i^{(1)}},t^{(2)}_{a_i^{(2)}},\ldots,
t^{(L)}_{a_i^{(L)}}\right),
$$

其中第 \(\ell\) 层有 \(K_\ell\) 个互斥的原子 token，

$$
t^{(\ell)}_k=\texttt{<SK\_L}\ell\texttt{\_}k\texttt{>},
\qquad 0\le k<K_\ell.
$$

层级码空间容量为 \(\prod_{\ell=1}^{L}K_\ell\)，但新增词表规模仅为
\(\sum_{\ell=1}^{L}K_\ell\)。单个 Skill 始终只需生成 \(L\) 个 code token。

主要符号如下。

| 符号 | 含义 |
|---|---|
| \(N\) | 候选 Skill 数 |
| \(d_i,x_i\) | Skill 文档及其固定语义向量 |
| \(L,K_\ell\) | 层数及第 \(\ell\) 层码本大小 |
| \(E_\phi,D_\psi\) | RQ-VAE 编码器和解码器 |
| \(\mathbf C^{(\ell)}\) | 第 \(\ell\) 层向量码本 |
| \(c(s_i)\) | Skill \(s_i\) 的离散层级路径 |
| \(\delta\) | 两条路径间的换行分隔 token 序列 |
| \(P_\theta\) | 自回归 Router |
| \(\mathcal B(c)\) | code \(c\) 对应的 Skill 碰撞桶 |

整体算法为：

```text
Skill 文档 ──> 固定语义向量 ──┐
                              ├─> 图正则 RQ-VAE ─> 层级平衡硬分配 ─> Skill codes
训练 query 的 Skill 共现图 ───┘                                      │
                                                                    v
Skill 文档/单 Skill query/多 Skill query ─> 三阶段 Router SFT ─> 约束生成
                                                                    │
                                                                    v
                                                    code 路径 ─> 碰撞桶展开 ─> Skills
```

## 2. 层级 Skill 标识符学习

### 2.1 语义表征与协同图

使用冻结的文本表征函数 \(f_{\mathrm{emb}}\) 将 Skill 文档映射为向量：

$$
x_i=\operatorname{Normalize}\!\left(f_{\mathrm{emb}}(d_i)\right)
\in\mathbb R^{d}.
$$

Embedding 模型不参与后续梯度更新。它只用于离线学习 Skill code；在线 Router 推理不调用 Embedding 模型。

为保留多 Skill 协作关系，从训练 query 的正例构造无向加权图
\(\mathcal G=(\mathcal S,\mathcal E)\)。记 \(\mathcal T(q)\) 为 query \(q\) 的目标 Skill 集，定义

$$
f_i=\sum_q\mathbf 1[s_i\in\mathcal T(q)],
\qquad
c_{ij}=\sum_q\mathbf 1[s_i,s_j\in\mathcal T(q)].
$$

边权使用余弦式共现归一化：

$$
w_{ij}=\frac{c_{ij}}{\sqrt{f_i f_j}},\qquad c_{ij}>0.
$$

同一语义 query 的不同目标顺序增强样本只计数一次，避免随机排列人为放大共现边。

### 2.2 多层残差量化

编码器首先产生低维连续隐变量：

$$
z_i=E_\phi(x_i)\in\mathbb R^{d_e}.
$$

当前 \(E_\phi\) 是带 ReLU 的多层感知机，\(D_\psi\) 使用反向层宽构成对称解码器；最后一层不施加激活。层宽和层数不影响后续离散算法的定义。

第 \(\ell\) 层码本为

$$
\mathbf C^{(\ell)}
=\{e^{(\ell)}_k\in\mathbb R^{d_e}\}_{k=0}^{K_\ell-1}.
$$

令初始残差 \(r_i^{(0)}=z_i\)。在第 \(\ell\) 层，计算残差到所有 code 向量的平方距离：

$$
D_{ik}^{(\ell)}
=\left\|r_i^{(\ell-1)}-e_k^{(\ell)}\right\|_2^2.
$$

训练时不直接使用逐样本最近邻，而是在每个 mini-batch 内执行熵正则最优传输。对经过仿射尺度归一化的距离 \(\bar D^{(\ell)}\)，Sinkhorn 迭代近似求解

$$
\begin{aligned}
\Pi^{(\ell)}
=\arg\min_{\Pi\ge0}\quad&
\langle \Pi,\bar D^{(\ell)}\rangle
-\epsilon_\ell H(\Pi),\\
\text{s.t.}\quad&
\Pi\mathbf 1=\mathbf 1,\qquad
\Pi^\top\mathbf 1=\frac{B}{K_\ell}\mathbf 1,
\end{aligned}
$$

其中 \(B\) 是 batch size，\(\epsilon_\ell\) 控制语义保真与均衡程度。硬索引取为

$$
a_i^{(\ell)}=\arg\max_k\Pi_{ik}^{(\ell)}.
$$

选中的码本向量为 \(e_i^{(\ell)}=e_{a_i^{(\ell)}}^{(\ell)}\)。采用 straight-through estimator：

$$
\tilde e_i^{(\ell)}
=r_i^{(\ell-1)}
+\operatorname{sg}\!\left(
e_i^{(\ell)}-r_i^{(\ell-1)}
\right),
$$

并递归更新

$$
r_i^{(\ell)}
=r_i^{(\ell-1)}-\tilde e_i^{(\ell)},\qquad
\hat z_i=\sum_{\ell=1}^{L}\tilde e_i^{(\ell)},\qquad
\hat x_i=D_\psi(\hat z_i).
$$

\(\operatorname{sg}(\cdot)\) 表示停止梯度。每层码本在首次使用时由当前残差的 K-means 中心初始化。

### 2.3 训练目标

重构损失为

$$
\mathcal L_{\mathrm{rec}}
=\frac{1}{B}\sum_i\|\hat x_i-x_i\|_2^2.
$$

第 \(\ell\) 层的向量量化损失为

$$
\mathcal L_{\mathrm{vq}}^{(\ell)}
=
\left\|e_i^{(\ell)}
-\operatorname{sg}(r_i^{(\ell-1)})\right\|_2^2
+\beta
\left\|\operatorname{sg}(e_i^{(\ell)})
-r_i^{(\ell-1)}\right\|_2^2,
$$

并对层和 batch 求均值：

$$
\mathcal L_{\mathrm{vq}}
=\frac{1}{L}\sum_{\ell=1}^{L}
\mathbb E_i[\mathcal L_{\mathrm{vq}}^{(\ell)}].
$$

对于当前 batch 内诱导出的协同边 \(\mathcal E_B\)，图正则项直接约束每层选中的码本向量：

$$
\mathcal L_{\mathrm{graph}}
=\frac{1}{L}\sum_{\ell=1}^{L}
\frac{
\sum_{(i,j)\in\mathcal E_B}
w_{ij}\|e_i^{(\ell)}-e_j^{(\ell)}\|_2^2
}{
\sum_{(i,j)\in\mathcal E_B}w_{ij}
}.
$$

因此 Stage 1 的完整目标为

$$
\boxed{
\mathcal L_{\mathrm{S1}}
=\mathcal L_{\mathrm{rec}}
+\lambda_{\mathrm{q}}
\left(
\mathcal L_{\mathrm{vq}}
+\lambda_{\mathrm{g}}\mathcal L_{\mathrm{graph}}
\right)
}.
$$

训练 batch 采用 edge-aware sampling：先采样一组基础节点，再按 \(w_{ij}\) 为每个基础节点采一个邻居，最后随机补齐 batch。这样无需构造稠密邻接矩阵，也能让多数 batch 含有有效协同边。

评估码本时关闭 Sinkhorn，使用普通残差最近邻得到 raw codes。当前模型选择准则按

$$
\left(
\operatorname{CollisionRate}_{\mathrm{raw}},
\mathcal L_{\mathrm{S1}}
\right)
$$

做字典序最小化：优先选择 raw code 碰撞率更低的 checkpoint，碰撞率相同时再比较损失。

### 2.4 冻结码本上的层级平衡硬分配

Sinkhorn 仅保证 mini-batch 内近似平衡，且逐行 \(\arg\max\) 后仍可能发生全局坍缩。因此，训练结束后冻结 \(E_\phi\) 与全部码本，在完整候选集合上执行一次确定性的层级硬分配。

首先计算全部隐变量和冻结码本距离。第 \(\ell\) 层的目标是尽量最小化

$$
\min_{\{a_i^{(\ell)}\}}
\sum_i D_{i,a_i^{(\ell)}}^{(\ell)}
$$

并施加两类约束。

第一，全局容量尽量严格平衡：

$$
n_k^{(\ell)}
=\sum_i\mathbf 1[a_i^{(\ell)}=k]
\in
\left\{
\left\lfloor\frac{N}{K_\ell}\right\rfloor,
\left\lceil\frac{N}{K_\ell}\right\rceil
\right\}.
$$

多出的容量分配给最近邻需求量最大的 code，整数索引用于确定性打破平局。

第二，在相同历史前缀的 Skill 组

$$
\mathcal G_p^{(\ell)}
=\{i:(a_i^{(1)},\ldots,a_i^{(\ell-1)})=p\}
$$

内部，只要 \(|\mathcal G_p^{(\ell)}|\le K_\ell\)，就要求当前层索引互异：

$$
a_i^{(\ell)}\ne a_j^{(\ell)},
\quad
i\ne j,\ i,j\in\mathcal G_p^{(\ell)}.
$$

当前求解器是确定性的分层近似：

1. 第一层通过带重复容量槽位的 Hungarian assignment 求解容量约束最小代价匹配；超大问题使用容量约束 deferred assignment。
2. 后续层按前缀组大小降序处理。在全局容量仍可行时，每组执行 Hungarian assignment，同时满足全局 floor/ceil 容量和组内唯一性。
3. 若某个前缀组大于当前码本，退化为组内 floor/ceil 平衡；小组精确求解，大组使用 deferred assignment。
4. 每层分配完成后，从残差中减去选中中心，再按新前缀重新分组。

该过程近似求解全局组合优化问题，但每个 Hungarian 子问题是精确的。对于当前两层设置，只要

$$
K_1K_2\ge N,
$$

第一层平衡便给出
\(\max_p|\mathcal G_p^{(2)}|\le K_2\)，第二层的前缀内唯一性因而可以产生无碰撞路径。

最终离散标识符为

$$
c(s_i)=
\left(
t^{(1)}_{a_i^{(1)}},
\ldots,
t^{(L)}_{a_i^{(L)}}
\right).
$$

若容量不足或约束不可满足，多个 Skill 可以共享同一路径。算法不会静默丢弃它们，而是保留碰撞桶：

$$
\mathcal B(c)=\{s_i:c(s_i)=c\}.
$$

### 2.5 Code 质量判据

对第 \(\ell\) 层计数 \(n_k^{(\ell)}\)，定义利用率和归一化熵：

$$
U_\ell
=\frac{|\{k:n_k^{(\ell)}>0\}|}{K_\ell},
$$

$$
\bar H_\ell
=-\frac{1}{\log K_\ell}
\sum_{k:n_k^{(\ell)}>0}
p_k^{(\ell)}\log p_k^{(\ell)},
\qquad
p_k^{(\ell)}=\frac{n_k^{(\ell)}}{N}.
$$

完整路径碰撞率定义为

$$
R_{\mathrm{coll}}
=1-\frac{|\{c(s_i):s_i\in\mathcal S\}|}{N}.
$$

Router 训练前同时检查 raw nearest codes 与最终 balanced codes 的层利用率、熵、碰撞率和最大桶大小。Raw 指标用于判断学习到的连续空间是否健康；balanced 指标用于保证最终生成任务可辨识。硬分配不能替代健康的 raw codebook。

### 2.6 可解释层级编码

当 Skill 自带长度至少为 \(L\) 的人工 taxonomy

$$
h_i=(h_i^{(1)},\ldots,h_i^{(L)}),
$$

可以跳过 RQ-VAE，使用 append-only 的前缀局部映射

$$
\mu_p^{(\ell)}:h^{(\ell)}\mapsto
\{0,\ldots,K_\ell-1\}.
$$

对每个前缀 \(p\)，新标签占用第一个未使用的 child index；已有标签永远复用原索引。删除 Skill 不回收标签含义。分支耗尽时，默认拒绝编码；显式允许 overflow 时，使用 codebook version、前缀和标签的稳定哈希映射到共享槽位。

这种方案的路径可直接解释为 taxonomy 标签链，但平衡性和碰撞率依赖人工 taxonomy。后续 Router 训练与推理算法对两种编码方式完全相同。当前默认训练链路采用上一节的学习式平衡编码。

## 3. Router 监督序列

### 3.1 原子词表

将所有层级 token 加入语言模型词表：

$$
\mathcal V'
=\mathcal V\cup
\bigcup_{\ell=1}^{L}
\{t_k^{(\ell)}\}_{k=0}^{K_\ell-1}.
$$

每个层级 token 必须恰好对应一个 tokenizer ID，且不同层使用互不相交的命名空间。输入和输出 embedding 随 Router 一起训练。

### 3.2 三类监督

记 \(d_i\) 为 Skill 文档，\(q_i^{(1)}\) 为只需要一个 Skill 的自然语言请求，\(q_n\) 为多 Skill 请求。

Memorization 数据为

$$
\mathcal D_{\mathrm{mem}}
=\{(d_i,c(s_i))\}_{i=1}^{N}.
$$

它学习“Skill 能力描述 \(\rightarrow\) 固定 code”，使每个候选 Skill 在多 Skill 学习前都获得直接监督。

Single-skill retrieval alignment 数据为

$$
\mathcal D_{\mathrm{align}}
=\{(q_i^{(1)},c(s_i))\}.
$$

它继续使用 retrieval 指令，将输入分布从 Skill 文档桥接到真实用户 query，同时保持单一、低歧义的目标。

Multi-skill retrieval 数据为

$$
\mathcal D_{\mathrm{ret}}
=\{(q_n,\mathcal Y_n)\}.
$$

三类输入均采用因果语言模型的对话上下文。Memorization 指令要求把给定 Skill 文档映射为固定长度 code；alignment 与 retrieval 指令要求选出请求所需的全部 Skill，并且每行只输出一条 code，不输出解释。记对应 prompt 为

$$
X_i^{\mathrm{mem}}=\operatorname{Chat}(h_{\mathrm{mem}},d_i),
\qquad
X_n^{\mathrm{ret}}=\operatorname{Chat}(h_{\mathrm{ret}},q_n).
$$

若目标 Skill 顺序为
\(\mathcal Y_n=(s_{n,1},\ldots,s_{n,m_n})\)，监督字符串为

$$
y_n=
c(s_{n,1})\Vert\delta\Vert
c(s_{n,2})\Vert\cdots\Vert\delta\Vert
c(s_{n,m_n})\Vert\mathrm{EOS}.
$$

其中 \(\delta\) 是换行符在基础 tokenizer 下对应的一个或多个 token。若多个目标 Skill 共享相同 code，该 code 在目标字符串中只出现一次；精确 Skill 集仍由其碰撞桶表示。

在当前默认的两层设置中，目标形式为：

```text
<SK_L1_a><SK_L2_b><EOS>
```

或

```text
<SK_L1_a><SK_L2_b>
<SK_L1_c><SK_L2_d><EOS>
```

含 \(m\) 条路径的目标 token 数为

$$
T_{\mathrm{out}}
=mL+(m-1)|\delta|+1.
$$

因此，算法始终生成短 code，而不是 Skill 名称、JSON 参数或自然语言解释。

### 3.3 隐式意图与顺序增强

隐式意图不引入独立分类头。对于没有被 query 直接点名、但完成任务所必需的 Skill，仍将其 code 放入 \(y_n\)，并通过同一自回归目标学习。

对于同一个语义请求，可以构造多个允许的目标排列
\(\Pi_n=\{\pi_1,\ldots,\pi_R\}\)。每个排列形成一个独立监督样本：

$$
y_{n,\pi}
=c(s_{n,\pi(1)})\Vert\delta\Vert\cdots
\Vert c(s_{n,\pi(m_n)})\Vert\mathrm{EOS}.
$$

Router 因而不依赖唯一的标注顺序。训练阶段不会把多 Skill 标签排序回 canonical order；给定样本中的顺序就是该样本的 teacher-forcing 顺序。

### 3.4 Target-only 自回归目标

对 prompt \(X_n\) 和目标 token
\(y_n=(y_{n,1},\ldots,y_{n,T_n})\)，Router 定义

$$
P_\theta(y_n\mid X_n)
=\prod_{t=1}^{T_n}
P_\theta(y_{n,t}\mid X_n,y_{n,<t}).
$$

只对 assistant 目标部分计算交叉熵；system prompt 和 user prompt 的 label 全部 mask：

$$
\boxed{
\mathcal L_{\mathrm{SFT}}(\theta;\mathcal D)
=-\frac{1}{\sum_nT_n}
\sum_n\sum_{t=1}^{T_n}
\log P_\theta(y_{n,t}\mid X_n,y_{n,<t})
}.
$$

Code token、换行边界和 EOS 使用相同的 token-level 权重。当前 Stage 2 不增加额外的对比损失、排序损失或 code-level 正则项；默认 weight decay 也为零。全参数微调与 LoRA 只改变可优化参数子空间，不改变上述目标。

## 4. 三阶段课程学习

设 \(\theta_0\) 为扩展层级 token 词表后的预训练语言模型参数。

第一阶段执行 Skill-code memorization：

$$
\theta_{\mathrm{mem}}
=\arg\min_\theta
\mathcal L_{\mathrm{SFT}}(\theta;\mathcal D_{\mathrm{mem}}),
\qquad \theta\leftarrow\theta_0.
$$

第二阶段从 \(\theta_{\mathrm{mem}}\) 初始化，执行单 Skill query 对齐：

$$
\theta_{\mathrm{align}}
=\arg\min_\theta
\mathcal L_{\mathrm{SFT}}(\theta;\mathcal D_{\mathrm{align}}),
\qquad \theta\leftarrow\theta_{\mathrm{mem}}.
$$

第三阶段从 \(\theta_{\mathrm{align}}\) 初始化，执行完整多 Skill 自回归训练。为缓解对固定 code 的灾难性遗忘，从
\(\mathcal D_{\mathrm{mem}}\) 中采样 replay 子集
\(\mathcal R_\rho\)。目标 replay 比例为 \(\rho\)，对应请求样本数

$$
R_{\mathrm{req}}
=
\begin{cases}
0,&\rho=0,\\
\max\!\left(
1,
\operatorname{round}
\left(
\dfrac{\rho}{1-\rho}|\mathcal D_{\mathrm{ret}}|
\right)
\right),&0<\rho<1.
\end{cases}
$$

当前算法从 \(\mathcal D_{\mathrm{mem}}\) 中确定性打乱后无放回采样，因此

$$
|\mathcal R_\rho|
=\min\left(|\mathcal D_{\mathrm{mem}}|,R_{\mathrm{req}}\right),
\qquad
\rho_{\mathrm{actual}}
=\frac{|\mathcal R_\rho|}
{|\mathcal D_{\mathrm{ret}}|+|\mathcal R_\rho|}.
$$

当 memorization 样本不足时，实际 replay 比例会低于请求的 \(\rho\)。

最终优化

$$
\boxed{
\theta_\star
=\arg\min_\theta
\mathcal L_{\mathrm{SFT}}
\left(
\theta;
\mathcal D_{\mathrm{ret}}\cup\mathcal R_\rho
\right),
\qquad
\theta\leftarrow\theta_{\mathrm{align}}
}.
$$

Replay 样本保留 memorization 指令，多 Skill 样本使用 retrieval 指令。当前默认闭集配置请求 \(\rho=0.2\)。

查询划分以规范化后的 query 文本为组，完全相同的请求不会跨训练集和验证集。验证组的每个目标 Skill 在训练监督中至少保留一个正例，避免把闭集评估误变为 unseen-target 评估。

算法 1 总结完整训练过程。

```text
Algorithm 1: Hierarchical Generative Skill Router Training
Input:
    candidate skills S, multi-skill supervision Dret,
    single-skill supervision Dalign,
    levels L, codebook sizes {K_l}, replay ratio rho
Output:
    code registry R and router parameters theta*

1:  x_i <- frozen_embed(skill_document(s_i)) for every s_i in S
2:  G <- normalized_co_use_graph(Dret)
3:  (E, D, {C^(l)}) <- train_graph_regularized_RQ_VAE({x_i}, G)
4:  {c(s_i)} <- hierarchical_balanced_assignment(E({x_i}), {C^(l)})
5:  reject the index if raw or assigned code quality violates the gates
6:  R <- group every active skill by its complete code path
7:  extend the LLM vocabulary with all level-specific code tokens
8:  theta_mem   <- target_only_SFT(theta_0, Dmem)
9:  theta_align <- target_only_SFT(theta_mem, Dalign)
10: R_rho <- sample_memorization_replay(Dmem, rho)
11: theta* <- target_only_SFT(theta_align, Dret union R_rho)
12: return R, theta*
```

## 5. 约束解码空间

令当前 active Skill 集为 \(\mathcal A\subseteq\mathcal S\)，其去重后的合法路径集合为

$$
\mathcal C_{\mathcal A}
=\{c(s):s\in\mathcal A\}.
$$

算法在 \(\mathcal C_{\mathcal A}\) 上构造 token trie。推理时，非法层级 token 在 softmax 前被置为 \(-\infty\)。对当前合法 token 集 \(\mathcal V_{\mathrm{allow}}(h)\)，约束概率为

$$
\tilde P_\theta(v\mid h)
=
\begin{cases}
\displaystyle
\frac{P_\theta(v\mid h)}
{\sum_{u\in\mathcal V_{\mathrm{allow}}(h)}
P_\theta(u\mid h)},&
v\in\mathcal V_{\mathrm{allow}}(h),\\[10pt]
0,&\text{otherwise}.
\end{cases}
$$

因此输出在算法层面保证是 active registry 中的合法 code，而不是仅依赖 prompt 学会格式。

### 5.1 多路径语法

Greedy 模式使用如下文法：

$$
\mathrm{PATH}\;(\delta\;\mathrm{PATH})^\star\;\mathrm{EOS},
$$

其中每个 \(\mathrm{PATH}\) 恰好包含 \(L\) 个 token。

解码状态包含：

1. 已完成路径集合 \(\mathcal H\)；
2. 当前路径前缀 \(p\)；
3. 当前位于路径、分隔符还是路径边界；
4. 已完成路径数 \(m\)。

若正在生成路径，合法下一 token 是所有尚未生成路径在前缀 \(p\) 后的 trie child：

$$
\mathcal V_{\mathrm{allow}}(p,\mathcal H)
=
\left\{
c_{|p|+1}:
c\in\mathcal C_{\mathcal A}\setminus\mathcal H,
c_{1:|p|}=p
\right\}.
$$

完成一条路径后：

- 若达到最大路径数 \(M\)，下一 token 只能是 EOS；
- 否则模型在 EOS 与 \(\delta\) 的首 token 之间自回归决策；
- 一旦选择 \(\delta\)，其余分隔 token 被文法强制生成；
- 已完成路径不能再次出现。

### 5.2 Greedy：完整多 Skill 自回归输出

Greedy 模式在每一步选择

$$
y_t^\star
=\arg\max_{v\in\mathcal V_{\mathrm{allow}}(y_{<t})}
\tilde P_\theta(v\mid q,y_{<t}).
$$

模型自行决定输出几条路径，并生成一个完整序列：

$$
c_1\Vert\delta\Vert c_2
\Vert\cdots\Vert\delta\Vert c_m\Vert\mathrm{EOS},
\qquad 1\le m\le M.
$$

第 \(j\) 条路径的展示分数只累加其 \(L\) 个 code token 的约束 log-probability：

$$
S_j^{\mathrm{greedy}}
=\sum_{\ell=1}^{L}
\log\tilde P_\theta
\left(
c_{j,\ell}
\mid q,c_{<j},\delta,c_{j,<\ell}
\right).
$$

分隔符与 EOS 的概率不并入路径分数。Greedy 的路径次序表示模型生成的选择/执行次序，而不是按 \(S_j^{\mathrm{greedy}}\) 重新排序。

```text
Algorithm 2: Constrained Multi-path Greedy Decoding
Input: query q, active path trie T, maximum paths M
Output: ordered paths C

1: C <- []; H <- empty set; state <- root(T)
2: while true:
3:     A <- legal_next_tokens(state, H, M)
4:     v <- argmax_{u in A} P_theta(u | q, generated_prefix)
5:     append v to generated_prefix and update state
6:     if one path c is completed:
7:         append c to C; add c to H
8:     if v is EOS:
9:         break
10: return C
```

### 5.3 Beam：单条路径的 Top-\(K\) code 检索

Beam 模式具有不同语义：它不搜索多行输出，也不是 Greedy 多路径解码的增强版。每个 beam 只表示一条固定长度路径，算法恰好生成 \(L\) 个 code token，然后返回得分最高的 \(K\) 个独立 code。

对单条路径 \(c\in\mathcal C_{\mathcal A}\)，定义

$$
S^{\mathrm{beam}}(c\mid q)
=\sum_{\ell=1}^{L}
\log\tilde P_\theta
\left(c_\ell\mid q,c_{<\ell}\right).
$$

宽度为 \(K\) 的标准 trie-constrained beam search 在每层展开合法 child，并保留累计分数最高的 \(K\) 个前缀。所有候选长度相同，故无需长度归一化。最终结果近似为

$$
\operatorname{TopK}_{c\in\mathcal C_{\mathcal A}}
S^{\mathrm{beam}}(c\mid q).
$$

该模式不生成换行符或 EOS。这样 code 排名不会混入“生成下一行还是停止”的概率。

```text
Algorithm 3: Constrained Single-code Top-K Beam Decoding
Input: query q, active path trie T, beam width K, path length L
Output: K ranked code paths

1: B_0 <- {(empty_prefix, score=0)}
2: for l = 1 ... L:
3:     E <- empty list
4:     for (prefix p, score s) in B_(l-1):
5:         for token v in trie_children(T, p):
6:             add (p + v, s + log P_tilde_theta(v | q, p)) to E
7:     B_l <- K highest-scoring entries in E
8: return B_L in descending score order
```

两种模式的语义对比如下。

| 属性 | Greedy | Beam |
|---|---|---|
| 搜索对象 | 一个完整的多路径序列 | \(K\) 条独立单路径 |
| 每个结果长度 | \(mL+(m-1)|\delta|+1\) | 恰好 \(L\) |
| 是否生成换行/EOS | 是 | 否 |
| Skill 数量决策 | 模型在边界选择 EOS 或继续 | 不做多 Skill 数量决策 |
| 输出路径数 | \(1\) 到 \(M\) | 最多 \(K\) 个 code |
| 排序含义 | 自回归生成/执行顺序 | 单 code 累计概率顺序 |

### 5.4 Code 到 Skill 的碰撞桶展开

对生成路径序列 \((c_1,\ldots,c_r)\)，依次展开

$$
\mathcal B_{\mathcal A}(c_j)
=\{s\in\mathcal A:c(s)=c_j\}.
$$

候选 Skill 排名遵循：

1. 先按路径排名；
2. 同一碰撞桶内使用确定性顺序；
3. Skill 继承所属路径的分数；
4. 展开并去重后，再截断 Skill candidate top-\(k\)。

因此，Beam width \(K\) 与 Skill candidate top-\(k\) 是两个不同参数：前者控制返回多少条 code 路径，后者控制桶展开后最多保留多少个 Skill。

当前算法没有桶内 reranker。若 \(|\mathcal B(c)|>1\)，仅凭生成 code 无法区分桶内 Skill；降低碰撞率是 Stage 1 的必要目标，而不是可由 Stage 2 自动补救的问题。

## 6. 算法性质与复杂度

### 6.1 输出效率

选择 \(m\) 个 Skill 的自回归输出长度为

$$
O(mL),
$$

与 Skill 名称或文档长度无关。当前 \(L=2\)，因此每条路径只包含两个新增 token。

### 6.2 约束有效性

只要 trie、registry 和 Router 使用同一 codebook version：

1. 每条输出路径长度恒为 \(L\)；
2. 每层 token 必属于对应层命名空间；
3. 每条路径必属于 active code 集；
4. Greedy 序列内不会重复完整路径；
5. 删除 active Skill 后，其空桶路径会自然从 trie 消失。

### 6.3 计算复杂度

对 batch size \(B\)、隐维度 \(d_e\)，一次 RQ 前向的距离计算复杂度为

$$
O\!\left(
B d_e\sum_{\ell=1}^{L}K_\ell
\right),
$$

Sinkhorn 额外需要

$$
O\!\left(
B I_{\mathrm{SK}}\sum_{\ell=1}^{L}K_\ell
\right).
$$

冻结码本后的距离计算对 \(N\) 个 Skill 为

$$
O\!\left(
N d_e\sum_{\ell=1}^{L}K_\ell
\right).
$$

Hungarian 子问题为立方复杂度，因此只对受控大小的组精确求解；更大组使用容量约束 deferred assignment。

使用 KV cache 时，Greedy 只需对实际生成的
\(mL+(m-1)|\delta|+1\) 个位置做增量解码。单路径 Beam 仅搜索 \(L\) 层，其 beam 状态数为 \(K\)，不会随最大多路径数 \(M\) 增长。

## 7. 当前默认算法实例

当前闭集训练采用以下算法性设定：

| 组件 | 当前默认 |
|---|---|
| 候选协议 | 训练、验证、测试和推理共享唯一候选集 |
| 层级长度 | \(L=2\) |
| 码本学习 | 图正则 RQ-VAE + 分层 Sinkhorn |
| 隐空间维度 | \(d_e=64\) |
| 量化权重 | \(\lambda_{\mathrm q}=1,\ \beta=2.25\) |
| 图正则 | \(\lambda_{\mathrm g}=10^{-3}\) |
| Sinkhorn 温度 | \((\epsilon_1,\epsilon_2)=(0.003,0.01)\) |
| 最终分配 | 全候选集上的 balanced hierarchical assignment |
| Router 目标 | target-only causal cross-entropy |
| 课程 | memorization \(\rightarrow\) single-skill alignment \(\rightarrow\) multi-skill retrieval |
| 最终 replay | memorization 请求比例 \(\rho=0.2\)，不足时按实际可采样量截断 |
| 默认推理 | constrained multi-path Greedy |
| 可选推理 | constrained single-code Top-\(K\) Beam |

ClawHub 1,000-candidate 实例使用 \(128\times128\) 的两层码本；Light 301-candidate 实例使用 \(32\times16\)。二者都让 \(\prod_\ell K_\ell\ge N\)，并通过第二层前缀内唯一性消除可避免碰撞。

## 8. 方法边界

当前算法的结论限定在以下范围内：

1. 默认协议是共享候选集的闭集检索，不以 unseen-skill 泛化为优化目标。
2. Embedding、层级码本与 Router 分阶段训练，而非端到端联合优化。
3. Registry 删除只改变 active 解码空间；在保持全局平衡约束的同时加入新 Skill，通常需要创建新 codebook version 并重新分配。
4. 随机目标顺序增强提高顺序鲁棒性，但若所有排列都被视为等价，就不能再把输出先后严格解释为唯一执行依赖。
5. 碰撞桶内没有二阶段语义消歧，精确 Skill 可辨识性的上限由 code 碰撞率决定。
