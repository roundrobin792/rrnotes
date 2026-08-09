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


### Gated DeltaNet（GDN）

## 3

## 参考文献
<a id="Katharopoulos2020"></a>
[1] [Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML 2020*.](https://arxiv.org/abs/2006.16236)