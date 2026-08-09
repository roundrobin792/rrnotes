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

## 实现

## 参考文献
<a id="Katharopoulos2020"></a>
[1] [Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML 2020*.](https://arxiv.org/abs/2006.16236)