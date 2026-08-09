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

假设我们将“相似度”函数记为 $\operatorname{sim}(q, k)$，那么论文所做的核心修改就在于**将其修改为“可分离”的版本，即能够将其写成 $\operatorname{sim}(q, k) = \phi(q)\phi(k)$ 的形式**，而这对于 Full Attention 中的 $\operatorname{sim}(q, k) = \operatorname{exp}(q, k)$ 来说是做不到的。

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

上一节推导出的朴素 LA 已经把复杂度降到了 $O(nd^2)$，但它的递归式过于简单：

$$
S_t = S_{t-1} + \phi(K_t)^TV_t
$$

新信息只是被**无条件地累加**进 $S$，既没有"该忘掉什么"的机制，也没有"该覆盖什么"的机制。GDN 正是沿着这两个方向对递归式做改进的：用**衰减门控 $\alpha$** 控制旧记忆的保留比例，用 **Delta Rule（配合更新门控 $\beta$）** 控制新信息的写入方式。本节内容参考 GDN 原始论文[[2]](#Yang2024)与 Qwen3-Next 的实现[[3]](#Qwen3Next2025)。

#### 改进一：Delta Rule —— 把"累加"改成"覆盖"

先看朴素 LA 写入方式的问题。把 $S$ 理解成一张以 key 为索引、以 value 为内容的记忆表，那么 $S_t = S_{t-1} + \phi(K_t)^TV_t$ 做的事情相当于 `mem[k] += v`，是**累加**语义。这带来的后果是：如果序列中先后出现了两个相近的 key，它们对应的 value 会被叠加在一起，而不是后者替换前者。换句话说，这张记忆表只能追加，不能修改。

Delta Rule 的思路是把它改成 `mem[k] = v` 的**覆盖**语义。具体做法分三步：

1. **读**：先用当前的 $K_t$ 去查询已有的记忆，得到记忆当前对这个 key 的"预测值"$\tilde{V_t} = K_tS_{t-1}$；
2. **算差**：真实值与预测值之差 $V_t - \tilde{V_t}$，即所谓的 delta（$\Delta$），代表"记忆还差多少才对"；
3. **写**：只把这个差值写回记忆，并用一个系数 $\beta_t$ 控制写入的强度。

写成公式即为：

$$
S_t = S_{t-1} + \beta_tK_t^T(V_t - K_tS_{t-1}) \tag{3}
$$

$(3)$ 式中的 $\beta_t \in (0, 1)$ 就是**更新门控**，它决定这一步"改写"得有多彻底：$\beta_t \to 0$ 时记忆几乎不动，$\beta_t \to 1$ 时旧值被完全替换成新值。可以把它类比为梯度下降中的学习率——事实上，Delta Rule 正是最小二乘意义下对 $\lVert V_t - K_tS \rVert^2$ 做一步梯度下降的结果，这也是它名字的由来。

把 $(3)$ 式展开整理，可以得到一个更能说明问题的形式：

$$
\begin{align*}
S_t &= S_{t-1} + \beta_tK_t^TV_t - \beta_tK_t^TK_tS_{t-1} \\\\
    &= (I - \beta_tK_t^TK_t)S_{t-1} + \beta_tK_t^TV_t
\end{align*}
$$

对比朴素 LA 的 $S_t = S_{t-1} + \phi(K_t)^TV_t$ 可以看出，Delta Rule 的本质是：**旧状态不再原封不动地传递下去，而是先被一个由当前输入决定的矩阵 $(I - \beta_tK_t^TK_t)$ 作用一次**。这个矩阵是单位阵的一个 rank-1 修正，它的作用是在 $K_t$ 所指的那个方向上"擦除"旧记忆，从而为新值腾出位置，而其它方向不受影响。

#### 改进二：衰减门控 $\alpha$ —— 让模型学会遗忘

Delta Rule 解决了写入的问题，但还有一个问题没解决：$S$ 的容量是固定的（$d_k \times d_v$），如果只写不忘，历史信息会不断累积、相互干扰，序列越长越严重。

早期的改进（如 RetNet）给递归式加了一个固定的衰减因子 $\gamma$，即 $S_t = \gamma S_{t-1} + \phi(K_t)^TV_t$，让远处的信息随距离自然淡出。但固定的 $\gamma$ 意味着所有 token、所有任务都用同一个遗忘速度，显然不够灵活。GDN 把它换成了**由当前输入决定的** $\alpha_t$，让模型自己判断此刻应该多记还是多忘：

$$
S_t = \alpha_tS_{t-1} + (\text{新信息的写入项})
$$

$\alpha_t$ 的参数化方式值得单独一提。它并不是简单地过一个 sigmoid，而是（以 Qwen3-Next 的实现为例）：

$$
\alpha_t = \operatorname{exp}(-A \cdot \operatorname{softplus}(X_tW_{\alpha} + b_{dt})) \tag{4}
$$

其中 $A = \operatorname{exp}(A_{\log})$ 是一个可学习的正参数。这个式子看起来绕，但结构上保证了 $\alpha_t \in (0, 1)$：$\operatorname{softplus}(\cdot)$ 的输出恒正，$A$ 恒正，所以指数部分恒负，取指数后必然落在 $(0, 1)$ 内，无需额外的截断。这种写法继承自 Mamba 的离散化参数化，好处是衰减率在对数空间中是线性的，长序列下连乘时数值更稳定。

需要注意的是，在 Qwen3-Next 中 $\alpha_t$ 是 **per-head 的标量**（每个注意力头一个值），而 Kimi Linear 的 KDA 机制将其细化为了 **per-channel 的向量**（每个特征维度一个值），以获得更精细的记忆控制能力[[4]](#KimiLinear2025)。

#### 完整的递推公式

把两项改进合并，并注意实现中的执行顺序是**先衰减、再读、后写**（即 Delta Rule 读取的是已经衰减过的状态），可得 GDN 的完整递推式：

$$
\begin{align*}
S_t &= \alpha_tS_{t-1} + \beta_tK_t^T(V_t - K_t \cdot \alpha_tS_{t-1}) \\\\
    &= \alpha_t(I - \beta_tK_t^TK_t)S_{t-1} + \beta_tK_t^TV_t
\end{align*}
\tag{5}
$$

$$
O_t = Q_tS_t \tag{6}
$$

对照上一节朴素 LA 的结论，有两处差异值得说明：

其一，$(5)$ 式中旧状态的传递算子是 $\alpha_t(I - \beta_tK_t^TK_t)$，它同时受 $\alpha_t$、$\beta_t$、$K_t$ 三者影响，全部依赖于当前输入。这与朴素 LA 中恒为 $I$、RetNet 中恒为 $\gamma I$ 的传递算子形成鲜明对比。

其二，$(6)$ 式**不再有归一化分母**，即上一节推导出的 $z_t$ 递归被去掉了。原因是 GDN 转而使用 RMSNorm 来做输出侧的归一化（见后文），而且经过 l2 归一化后的 $Q$、$K$ 本身数值范围就已经受控，分母带来的收益不足以抵消它的计算开销。这也意味着 GDN 中的 $\phi(\cdot)$ 已经不再承担"保证分母非零"的职责，它退化成了一个普通的激活层。

{{% details "**GDN 还能写成 Full Attention 那样的逐行公式吗？**" %}}
不能。把 $(5)$ 式反复展开到底，可以得到：

$$
S_t = \sum_{i=1}^t \left( \prod_{l=i+1}^t \alpha_l(I - \beta_lK_l^TK_l) \right) \beta_iK_i^TV_i
$$

代入 $(6)$ 式后，$V_i$ 前面的系数是一个**连乘积**，它依赖于 $i$ 与 $t$ 之间**所有中间 token** 的 $\alpha_l$、$\beta_l$ 和 $K_l$，而不只是 $Q_t$ 和 $K_i$。这意味着无法写出一个 $A_{ti} = f(Q_t, K_i)$ 形式的注意力权重，也就没有对应的 $n \times n$ 注意力矩阵。

朴素 LA 与 RetNet 之所以还能写成注意力矩阵形式（后者对应一个固定的指数衰减掩码），是因为它们的传递算子与输入无关。转折点发生在 Delta Rule 引入 $(I - \beta_tK_t^TK_t)$ 的那一刻——从这里开始，LA 就彻底脱离了 Attention 的框架，更适合被理解为一个**隐状态为矩阵的非线性 RNN**。
{{% /details %}}

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

#### 复杂度

GDN 相对朴素 LA 增加了这么多机制，复杂度是否还保持线性？简要估算递推一步的开销：衰减 $\alpha_tS_{t-1}$ 为 $O(d_kd_v)$；读取 $K_tS_{t-1}$ 为 $O(d_kd_v)$；外积写入 $\beta_tK_t^T(\cdot)$ 为 $O(d_kd_v)$；读出 $Q_tS_t$ 为 $O(d_kd_v)$。四项均为 $O(d_kd_v)$，故单步复杂度为 $O(d_kd_v)$，$n$ 步累计即 $O(nd_kd_v)$，近似处理后仍为 $O(nd^2)$，与朴素 LA 同阶。

关键在于，Delta Rule 引入的 $(I - \beta_tK_t^TK_t)$ 虽然形式上是一个 $d_k \times d_k$ 的矩阵，但由于它是单位阵的 rank-1 修正，实际计算时无需显式构造该矩阵，只需按 $(3)$ 式的"读—算差—写"三步走，每步都是向量与矩阵的运算，因此没有引入额外的复杂度。这正是 Delta Rule 能够被"免费"加入的原因。

除时间复杂度外，GDN 在推理时的显存优势同样显著。Full Attention 的 KV cache 大小为：

$$
b \times n \times h \times d_{head} \times 2 \times \mathrm{bytes}
$$

而 GDN 需要缓存的状态矩阵 $S$ 大小为：

$$
b \times h \times d_{head}^2 \times \mathrm{bytes}
$$

后者**不含 $n$ 项**，即显存占用与上下文长度无关。虽然引入了 $d_{head}^2$ 的平方项，但 $d_{head}$ 通常较小（Qwen3-Next 中为 128），在长上下文场景下这笔交换是相当划算的。

{{% details "**既然 GDN 这么好，为什么不完全取代 Full Attention？**" %}}
因为 GDN 的建模能力仍受限于 $S$ 的固定容量。无论门控多灵活、写入多精准，把任意长度的历史压缩进一个 $d_k \times d_v$ 的矩阵都是**有损**的，序列越长损失越明显。这在需要**精确检索**的任务上尤其致命——比如从长文档中定位某一具体事实，Full Attention 保留了所有 token 的完整表示，可以精确定位；而 GDN 中该信息早已被混合、衰减进 $S$，难以无损取回。

正因如此，Qwen3-Next 与 Kimi Linear 都采用了**混合架构**：每三个 GDN 层搭配一个 Full Attention 层（3:1 比例），用少量的全注意力层承担精确检索的职责，其余层则享受线性复杂度带来的效率收益。
{{% /details %}}

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
