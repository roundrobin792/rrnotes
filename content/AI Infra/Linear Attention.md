---
title: "Linear Attention"
weight: 1
# bookFlatSection: false
bookToc: true
# bookHidden: false
# bookCollapseSection: false
# bookComments: true
# bookSearchExclude: false
# bookHref: ''
# bookIcon: ''
---

{{< katex />}}


# Linear Attention

这是本人写的第一篇 AI Infra 相关的博客，因此有必要说一嘴：根据个人经验，AI 乃至整个机器学习领域，包括 NLP 相关的一些研究，很多时候是从一些“有道理但模糊”的 intuition 开始的，很多时候构造的公式、算法其实只是对现实规律的一个近似模拟，甚至有些时候都不能确定现实是否真的如此，只是研究者们提出的一种观点，并不具有严格的数学证明，大家都认可的原因仅仅是因为它有效。所以，在很多时候，只需要有一个可说的 intuition 来解释“为什么这样设计”就行了，多关注算法层面的、看得清摸得着的东西（比如能看懂公式，写出代码），不要在本身就没办法给出所谓的“严谨、完备”的解释的问题上钻牛角尖，没有任何意义。本文中的 GDN 一节便是如此。

## 概述

{{% details "**什么是 Linear Attention（LA）？**" %}}
简言之，就是 Transformer 架构中 Attention 机制的一种在**长序列任务**下更高效的实现方式，相较于传统的 Full Attention，能够将单个注意力头的前向时间复杂度由 $O(n^2d)$ 降为 $O(nd^2)$（$n$ 为序列长度，$d$ 为 hidden size，在 $n \gg d$ 时，即长序列场景下才成立）。根据语境，LA 有狭义和广义两种理解方式：
  * 狭义的 LA：特指 2020 年原始论文中提出的实现方式[[1]](#Katharopoulos2020)。这也是最朴素的 LA 实现，通过将 softmax 替换成可拆分的核函数，以将 Full Attention 公式修改为可逐 token 线性递归的形式，从而将复杂度降为 $O(nd^2)$（后文会详细推导）。
  * 广义的 LA：在朴素的 LA 基础上，通过改进递归公式进行改进后的算法的伞称，包括 RetNet、RWKV、GLA、Mamba-2、DeltaNet 和如今的 Gated DeltaNet 等等。这些算法仍然继承了朴素 LA 核心的可递归形式，保持了 $O(nd^2)$ 复杂度，只是使用门控和 Delta Rule 等机制提高 LA 的建模能力。
{{% /details %}}

{{% details "**为什么需要 LA？**" %}}
LLM 支持输入的序列越来越长，Full Attention 的单头前向复杂度为 $O(n^2d)$（$n \gg d$），逐渐成为了不可忽视的性能瓶颈。Q1 中简要介绍了 LA 在长序列场景下的性能优势，这也正是 LA 开始被使用的原因。
{{% /details %}}

{{% details "**LA 是否有缺点？**" %}}
任何工程决策都有所取舍。LA 带来的效率提升难免会降低注意力层的建模能力，因为在递归计算过程中仅通过有限大小的记忆空间（$d \times d$）来压缩历史信息，会导致信息损失，且序列越长损失越明显。
{{% /details %}}

## 原理

### 朴素 Linear Attention

本节介绍如何从经典的 Full Attention 公式推导出最原始的 Linear Attention 公式，说明其核心部分（可递归状态 $S$）的由来。本节内容参考 LA 原始论文[[1]](#Katharopoulos2020)。

假定注意力层的输入序列长度为 $n$，输出矩阵为 $O$，且 $Q$、$K$ 矩阵的 hidden size 为 $d_{k}$，$V$ 的为 $d_{v}$，那么根据 Full Attention 的定义，在加了 causal mask 的情况下 $O$ 的第 $t$ 行的表达式为：

$$
\begin{equation}
O_{t} = \sum_{i=1}^t \frac{\operatorname{exp}(\frac{Q_{t}K_{i}^T}{\sqrt{d_{k}}})}{\sum_{j=1}^{t}\operatorname{exp}(\frac{Q_{t}K_{j}^T}{\sqrt{d_{k}}})}V_{i}
\end{equation}
$$

由此估算 Full Attention 的时间复杂度：计算特定的一行输出 $O_{t}$，首先要算出 $\operatorname{exp}(\frac{Q_{t}K_{i}^T}{\sqrt{d_{k}}})$（$1 \le i \le t$）并保存，这一步复杂度显然是 $O(td_{k})$；然后求和得出 $\sum_{j=1}^{t}\operatorname{exp}(\frac{Q_{t}K_{j}^T}{\sqrt{d_{k}}})$，这一步为 $O(t)$；最后一步则是使用前两步的结果算出 $O_{t}$，这一步是 $O(td_{v})$。因此算第 $t$ 行的总时间复杂度为 $O((d_{k} + d_{v} + 1)t)$，由于 $O$ 共有 $n$ 行，故复杂度为 $O((d_{k} + d_{v} + 1)\frac{n^2 + n}{2})$。如果不考虑 $d_{k}$ 和 $d_{v}$ 差异，近似处理为单一的变量 $d$，且在 $n \gg d$ 的长序列场景下，复杂度近似为 $O(n^2d)$。

从代数结构上分析，$(1)$ 式的中的 $\operatorname{exp}(\cdot)$ 函数可以看成是计算 $Q_{t}$ 与其它 token 的 key 的“相似度”的函数，中间的 $\frac{\operatorname{exp}(\frac{Q_{t}K_{i}^T}{\sqrt{d_{k}}})}{\sum_{j=1}^{t}\operatorname{exp}(\frac{Q_{t}K_{j}^T}{\sqrt{d_{k}}})}$ 这一部分可以看成是将与序列中所有之前的 token 的相似度归一化，以算出 $V_{i}$ 在 $O_{t}$ 中所占的权重。

假设我们将“相似度”函数记为 $\operatorname{sim}(q, k)$，那么论文所做的核心修改就在于**将其修改为“可分离”的版本，即能够将其写成 $\operatorname{sim}(q, k) = \phi(q)\phi(k)$ 的形式**（其中 $\phi(\cdot)$ 叫做特征映射，用于将 $q$、$k$ 映射到特征空间使得其内积保持原来相似度函数的效果），而这对于 Full Attention 中的 $\operatorname{sim}(q, k) = \operatorname{exp}(q, k)$ 来说是做不到的。

那么这样做究竟能产生什么样的效果？我们将其带入 $(1)$ 进行推导：

$$
\begin{align*}
O_{t} &= \sum_{i=1}^t \frac{\operatorname{sim}(Q_{t}, K_{i}^T)}{\sum_{j=1}^{t}\operatorname{sim}(Q_{t}, K_{j}^T)}V_{i} \\\\
      &= \sum_{i=1}^t \frac{\phi(Q_{t})\phi(K_{i})^T}{\sum_{j=1}^t \phi(Q_{t})\phi(K_{j})^T}V_{i} \\\\
      &= \frac{\phi(Q_{t})\sum_{i=1}^t \phi(K_{i})^TV_{i}}{\phi(Q_{t}) \sum_{j=1}^t \phi(K_{j})^T}
\end{align*}
$$

令 $S_t = \sum_{i=1}^t \phi(K_{i})^TV_{i}$，$z_t = \sum_{j=1}^t \phi(K_{j})^T$，则：

$$
\begin{equation}
O_{t} = \frac{\phi(Q_{t})S_t}{\phi(Q_{t})z_t}
\end{equation}
$$

这就是我们得到的最终的计算公式，此时计算 $O$ 矩阵的整体时间复杂度已经降为了 $O(nd^2)$，因为 $S_t$ 和 $z_t$ 存在递归式：

$$
S_t = S_{t-1} + \phi(K_t)^TV_t \\\\
z_t = z_{t-1} + \phi(K_t)^T
$$

这意味着，在 $O_t$ 的时候，只需设置一块空间存储 $S_t$ 和 $z_t$ 的值，下轮求 $O_{t+1}$ 的时候，直接在已有的基础上递推得到 $S_{t+1}$ 和 $z_{t+1}$，而无需重复计算，从而降低了计算量。

简要估算上述 LA 公式的时间复杂度：求 $O_t$ 时，递推 $S_t$ 和 $z_t$ 的复杂度分别为 $O(d_kd_v)$、$O(d_k)$；计算 $(2)$ 式分子分母的复杂度分别为 $O(d_kd_v)$ 和 $O(d_k^2)$。可得计算 $O_t$ 的整体复杂度为 $O(d_k^2 + 2d_kd_v + d_k + d_v)$，由于 $O$ 矩阵有 $n$ 行，故整个 LA 的时间复杂度为 $O((d_k^2 + 2d_kd_v + d_k + d_v)n)$，近似处理后即得 $O(nd^2)$。

将计算改为可递归的形式是 LA 降低时间复杂度的核心。不管是朴素的 LA 还是后续的改进版本，都遵循了类似的递归结构。


### Gated DeltaNet（GDN）

上一节推导出的朴素 LA 公式如下：

$$
S_t = S_{t-1} + K_t^TV_t \\\\
O_{t} = Q_{t}S_t
$$

从这里开始，我们会省略掉用于归一化的分母和特征映射 $\phi$，现实中实现各类 LA 的时候，往往会将相同功能的操作放在其它位置，上面的递归式才是 LA 的核心部分。

虽然能降低复杂度，但是实验证明这种方法的实际效果明显落后于传统的 Full Attention。可能的解释是，$S_t$ 承载了包括当前和之前所有 token 的“记忆”（通过单纯累加 key 和 value 的外积），由于状态 $S_t$ 可存储的 token 信息数量有限，当输入序列长度超出上限的时候，就会导致不同 token 之间的信息相互干扰。对此，更加详细的 intuition 可以参考[[6]](#Schlag2021)。

既然能力下降的根本原因是 $S_t$ 信息过载，那么优化的方向就自然是让 $S_t$ 在递推更新时能够适当的“忘记”之前的信息，最好可以有选择性地忘记不重要的信息。为此，GDN 糅合了两种被证明可行的机制：

* **衰减门控 $\alpha$**：根据时间维度的距离逐渐忘记较远 token 的信息。
* **Delta Rule（配合更新门控 $\beta$）**：在往状态中添加当前 token 的信息时针对性地删除与之重合的信息。

本节将详细讲解上述两种机制的实现方式，内容参考 GDN 原始论文[[2]](#Yang2024)与 Qwen3-Next 的实现[[3]](#Qwen3Next2025)。

#### Delta Rule（配合更新门控 $\beta$）

根据论文[[6]](#Schlag2021)，从数学形式上，朴素 LA 的状态 $S$ 可以被理解为一张以 key 为索引、以 value 为内容的记忆表（key-value accociative memory，写入的语义是累加 token 的 key、value 的外积，召回的语义是 $\tilde{V}_{t} = K_{t} S$）。$S_t = S_{t-1} + K_t^TV_t$ 的语义是单纯添加信息，前文说过，当序列较长时，会超出状态 $S$ 可存储 token 信息的上限。

Delta Rule 的思路是在添加新的 token 信息时，同时抹除一部分与之重叠的信息。公式如下：

$$
\tilde{V}_t = K_tS_{t-1} \\\\
V_t' = \beta_tV_t + (1 - \beta_t) \tilde{V}_t \\\\
S_t = S_{t-1} - \underbrace{K_t^T \tilde{V}_t}_{\text{抹除的信息}} + \underbrace{K_t^TV_t'}_{\text{添加的信息}}
$$

代入化简即得：

$$
S_t = S_{t-1} + \beta_tK_t^T(V_t - K_tS_{t-1}) \tag{3}
$$

$(3)$ 式中的 $\beta_t \in (0, 1)$ 被称为**更新门控（update gate）**，是一个数值，从上面的公式不难看出，它控制的是抹除和写入信息的强度。

所有 token 的更新门控在进入 Delta Rule 之前计算好，以 $\beta \in \mathbb{R}^n$ 的形式传入。一种简单的实现是，通过对输入做一层线性投影，再经过一层 sigmoid 激活得到。

#### 衰减门控 $\alpha$

**衰减门控**是另一种自动抹除 $S$ 中信息的方式，其核心思想是每一步迭代都先对之前的记忆用衰减系数 $\alpha_t$ 做统一的衰减，然后再写入新信息，公式如下：

$$
S_t = \alpha_tS_{t-1} + (\text{新信息的写入项})
$$

每个 token 都有自己的衰减系数，以控制对之前的 token 信息是“多忘”还是“少忘”。一个比较好理解的例子是：如果一个 token 前是句号，那么该 token 是新句子的开头，则需要多忘；而如果和前面的 token 构成连贯的句子，则需要少忘。早期的 LA（如 RetNet）就只用了一个固定的衰减因子，事实证明恒定的遗忘速度不够灵活。

$\alpha_t$ 的计算略微复杂一点，做线性投影后并不是简单地过一个 sigmoid，而是（以 Qwen3-Next 的实现为例）：

$$
\alpha = \operatorname{exp}(-A \odot \operatorname{softplus}(w_{\alpha}X^T + b_{\alpha})) \tag{4}
$$

其中 $A = \operatorname{exp}(A_{\log})$ 是一个可学习的正参数。该式结构上保证了 $\alpha_t \in (0, 1)$（$\operatorname{softplus}(\cdot)$ 的输出恒正，$A$ 恒正，所以指数部分恒负）。这种写法继承自 Mamba 的离散化参数化，好处是衰减率在对数空间中是线性的，长序列下连乘时数值更稳定。

需要注意的是，在 Qwen3-Next 中 $\alpha_t$ 是 **per-head 的标量**（每个注意力头一个值），而 Kimi Linear 的 KDA 机制将其细化为了 **per-channel 的向量**（每个特征维度一个值），以获得更精细的记忆控制能力[[4]](#KimiLinear2025)。

#### 完整的递推公式

把两项改进合并，并注意实现中的执行顺序是**先衰减、再读、后写**（即 Delta Rule 读取的是已经衰减过的状态），可得 GDN 的完整递推式：

$$
S_t = \alpha_tS_{t-1} + \beta_tK_t^T(V_t - K_t \cdot \alpha_tS_{t-1}) \tag{5}
$$

$$
O_t = Q_tS_t \tag{6}
$$

在这里省略详细的复杂度分析，不难得出时间复杂度仍然是 $O(nd^2)$。

#### 其余组件

除了上述递推的核心，一个完整的 GDN 层还包含几个辅助组件。

**Causal Conv1d（因果深度卷积）**。在进入递推之前，$Q$、$K$、$V$ 会先沿序列维度过一个短的因果深度可分离卷积（kernel size 通常为 4）。它的作用是引入**局部信息混合**：递归状态 $S$ 擅长的是携带全局的、被压缩过的历史，而对于"前几个 token 说了什么"这类短程依赖，让卷积直接处理要比走 $S$ 这个瓶颈高效得多。这个设计同样继承自 Mamba。

**l2 归一化**。在递推之前，$Q$ 与 $K$ 会先做 l2 归一化（$Q$ 额外除以 $\sqrt{d_k}$）。对 $K$ 做归一化不只是为了数值稳定，它对递推的收敛性是**结构性的保障**：考察传递算子中的 $I - \beta_tK_t^TK_t$，由于 $K_t^TK_t$ 是秩为 1 的矩阵，其唯一的非零特征值为 $\lVert K_t \rVert^2$，因此该算子的特征值为 $1 - \beta_t\lVert K_t \rVert^2$（沿 $K_t$ 方向）和 $1$（其余方向）。当 $\lVert K_t \rVert = 1$ 且 $\beta_t \in (0, 1)$ 时，特征值恒落在 $(0, 1]$ 内，算子是非扩张的，$S$ 在长序列上不会发散；而若 $\lVert K_t \rVert$ 可以任意大，$1 - \beta_t\lVert K_t \rVert^2$ 可能是绝对值很大的负数，连乘后状态迅速爆炸。

**输出门控**。递推得到的 $O_t$ 会先过一个 per-head 的 RMSNorm，再与一个由输入算出的门相乘：

$$
\tilde{O_t} = \operatorname{RMSNorm}(O_t) \odot \operatorname{silu}(X_tW_g)
$$

这与 Qwen3-Next 在其 Full Attention 层上使用的 gated attention 是同一个思路，区别只在于激活函数用的是 $\operatorname{silu}$ 而非 $\operatorname{sigmoid}$。Qwen3-Next 的作者称这类输出门控有助于消除 Attention Sink 和 Massive Activation 等现象，提升数值稳定性。

#### 整体流程

综合上述内容，单个 GDN 层的前向流程为：

1. 由输入 $X$ 线性投影得到 $Q$、$K$、$V$，以及三个门控信号 $\alpha$、$\beta$、$g$（其中 $\alpha$ 按 $(4)$ 式参数化，$\beta = \operatorname{sigmoid}(XW_{\beta})$）；
2. 对 $Q$、$K$、$V$ 施加因果深度卷积，混合局部信息；
3. 对 $Q$、$K$ 做 l2 归一化；
4. 初始化 $S_0 = 0$，按 $(5)$ 式逐 token 递推更新状态，并按 $(6)$ 式读出每一步的输出；
5. 对输出施加 RMSNorm 与 $\operatorname{silu}$ 门控，最后经输出投影得到该层结果。

## 实现

理论部分讲清楚了 GDN 在做什么，本节则对照一份简化的 PyTorch 实现，看看这些机制在代码中分别对应哪几行。代码参考自 Sebastian Raschka 对 Qwen3-Next 官方实现的简化改写[[5]](#Raschka2025)，为便于阅读，省略了因果卷积部分，只保留递推的核心逻辑。

### 门控信号的计算

先看三个门控信号是怎么从输入算出来的。$Q$、$K$、$V$ 的投影与标准注意力无异，此处略过：

```python
beta = torch.sigmoid(self.W_beta(x))
alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
    self.W_alpha(x) + self.dt_bias
)
alpha = alpha_log.exp()
gate = self.W_gate(x)
```

$\beta$ 的计算最直接，一个线性层加 sigmoid，把更新强度压到 $(0, 1)$ 内。

$\alpha$ 则对应理论部分的 $(4)$ 式。这里把它拆成了 `alpha_log` 和 `alpha` 两步，正是为了让 $\alpha_t = \operatorname{exp}(-A \cdot \operatorname{softplus}(\cdot))$ 的结构显式可见：`self.A_log.exp()` 即 $A$，恒正；`F.softplus(...)` 恒正；两者相乘取负后再取指数，结构上保证了 $\alpha \in (0, 1)$。需要留意 `self.W_alpha` 的输出维度是 `num_heads` 而非 `d_out`，这印证了前文提到的——Qwen3-Next 中的 $\alpha$ 是 per-head 的标量。

`gate` 暂时只是个线性投影，它要到最后的输出门控才会用上。

### l2 归一化

```python
queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
keys = l2norm(keys, dim=-1)
```

这两行对应理论部分关于传递算子非扩张性的讨论。`keys` 被归一化到单位长度，从而保证 $I - \beta_tK_t^TK_t$ 的特征值落在 $(0, 1]$ 内；`queries` 额外除以 $\sqrt{d_k}$，作用类似标准注意力中的缩放因子。

### 递推主循环

这是整个 GDN 的核心，也是与标准注意力差异最大的地方：

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

for t in range(num_tokens):
    k_t, q_t, v_t = keys[:, :, t], queries[:, :, t], values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t                                        # ① 衰减
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)       # ② 读
    delta = (v_t - kv_mem) * b_t                       # ③ 算差
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)    # ④ 写
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)          # ⑤ 读出
```

这五行与理论部分的 $(5)$、$(6)$ 式一一对应：

* ① 对应 $\alpha_tS_{t-1}$，按当前 token 决定的衰减率淡化旧记忆；
* ②③④ 合起来就是 Delta Rule 的"读—算差—写"三步，即 $\beta_tK_t^T(V_t - K_tS_{t-1})$。注意 ② 读取的是**已经衰减过的** $S$，这与前文强调的执行顺序一致；
* ⑤ 对应 $O_t = Q_tS_t$。

有两处实现细节值得说明。其一，`(S * k_t.unsqueeze(-1)).sum(dim=-2)` 是在用广播加求和来表达向量与矩阵的乘法 $K_tS$，`k_t.unsqueeze(-1) * delta.unsqueeze(-2)` 则是外积 $K_t^T\Delta$，两者都避免了显式的 `matmul` 调用。其二，全程没有出现任何 $n \times n$ 的中间张量——`S` 的形状是 `(b, h, d, d)`，与 `num_tokens` 无关，这正是 GDN 显存优势的来源。

{{% details "**这个 for 循环不会很慢吗？**" %}}
会。逐 token 的 Python 循环无法利用 GPU 的并行能力，这份实现的目的是把递推逻辑讲清楚，而非用于生产。

实际训练中采用的是 **chunk-wise 并行**策略：把序列切成若干固定大小的 chunk，chunk 内部用矩阵乘法一次性算完（充分利用 tensor core），chunk 之间才串行地传递状态 $S$。这样既保持了 $O(nd^2)$ 的复杂度，又把并行度提到了可接受的水平。相关的融合算子可以在 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 中找到。
{{% /details %}}

### 输出门控

```python
context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
context = context.view(b, num_tokens, self.num_heads, self.head_dim)

context = self.norm(context)
context = context * F.silu(gate)
```

把每一步的 `y_t` 堆叠回序列后，先过 per-head 的 RMSNorm（`self.norm` 的归一化维度是 `head_dim`），再与前面算好的 `gate` 相乘。这里用的是 $\operatorname{silu}$ 而非 $\operatorname{sigmoid}$，与 Qwen3-Next 在 Full Attention 层上使用的 gated attention 形成对照。

值得回顾的是，理论部分提到 GDN 去掉了朴素 LA 的归一化分母 $z_t$——`self.norm` 这一行正是它的替代品：归一化的职责从递推内部转移到了输出侧，既省去了 $z_t$ 的递归维护，也让数值范围的控制更直接。

### 小结

对照完代码可以看出，GDN 相对朴素 LA 的改动集中在递推主循环的五行之内：加一行衰减、把一行累加拆成读—算差—写三行。其余部分——投影、归一化、输出门控——都是围绕这个核心的辅助设施。这也解释了为什么 GDN 能在不改变 $O(nd^2)$ 复杂度的前提下显著提升建模能力：所有改进都作用在**如何维护 $S$** 这件事上，而没有触碰"$S$ 大小固定"这个前提。

## Chunk-wise Parallelism

理论上，LA 确实降低了长序列场景下的时间复杂度。然而，时间复杂度衡量的仅仅是“计算所需的操作数”，在串行计算时确实能加速，但是在支持并行运算时并不一定如此。

现实中 LA 确实遇到了这个问题。为方便理解，我们此处以朴素 LA 为例，解释问题的根源，以及什么是、为什么需要 chunk-wise 实现。

朴素 LA 的公式有两种表示法：

1. **循环形式（Recurrent form）**：$S_t = S_{t-1} + K_t^TV_t$，$O_t = Q_t S_t$。
2. **并行形式（Parallel form）**：$O = (QK^T \odot M)V$，其中 $M$ 为 causal mask。

从循环形式可以导出并行形式，推导此处省略。那么，为什么叫循环形式和并行形式？循环形式比较好理解，即需要在一个逐 token 迭代的循环中一行一行地求出输出矩阵 $O$，每一轮迭代的状态 $S_t$ 都依赖于上一轮的 $S_{t-1}$（因此每轮迭代具有严格的先后次序关系，无法并行）；至于并行形式，其实时间复杂度仍为 $O(n^2d)$，但是由于是矩阵乘法，而 GPU 的 matmul 并行加速机制可能反而使得采用并行形式要快得多（对于序列长度和 hidden size 不过分大的情况下，甚至可能在 $O(1)$ 的时间内解决）。

## 参考文献
<a id="Katharopoulos2020"></a>
[1] [Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML 2020*.](https://arxiv.org/abs/2006.16236)

<a id="Yang2024"></a>
[2] [Yang, S., Kautz, J., & Hatamizadeh, A. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. *arXiv:2412.06464*.](https://arxiv.org/abs/2412.06464)

<a id="Qwen3Next2025"></a>
[3] [Qwen Team. (2025). Qwen3-Next: Towards Ultimate Training & Inference Efficiency.](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)

<a id="KimiLinear2025"></a>
[4] [Kimi Team. (2025). Kimi Linear: An Expressive, Efficient Attention Architecture. *arXiv:2510.26692*.](https://arxiv.org/abs/2510.26692)

<a id="Raschka2025"></a>
[5] [Raschka, S. Gated DeltaNet for Linear Attention. *LLMs-from-scratch*, ch04/08_deltanet.](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet)

<a id="Schlag2021"></a>
[6] [Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. *ICML 2021*.](https://proceedings.mlr.press/v139/schlag21a.html)
