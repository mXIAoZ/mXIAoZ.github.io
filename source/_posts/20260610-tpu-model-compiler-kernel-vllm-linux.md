---
title: 从大模型到 TPU 执行：框架、算子、内核、DLC、vLLM 与 Linux 内核
date: 2026/06/10
categories:
- AI Infra
- TPU
tags: [TPU,LLM,vLLM,Compiler,Linux,Kernel]
mathjax: true
---

最近在梳理大模型基础设施的系统分层图：大模型、深度学习框架、Tensor、算子、算子内核、DLC/编译器、vLLM、TPU Runtime、Linux 内核和 TPU 硬件分别处在哪一层，各自解决什么问题。

这篇文章试图用更工程化的方式回答这个问题。最核心的关系是：**大模型定义计算目标，框架表达计算图，算子定义数学语义，算子内核定义硬件实现，编译器/DLC 生成目标硬件可执行产物，vLLM 管理在线推理调度与 KV cache，Runtime 通过 Linux TPU 驱动把任务提交给 TPU 执行。**

<!--more-->

## 先分清两个 Kernel

“Kernel” 在这条栈里至少有两个完全不同的含义。第一个是 **Linux Kernel**，也就是操作系统内核；第二个是 **Operator Kernel**，也就是某个算子在 GPU/TPU/NPU 上执行的计算内核。

| 名称 | 专业含义 | 运行位置 | 主要职责 |
| --- | --- | --- | --- |
| Linux Kernel | 操作系统内核 | CPU 特权态 | 进程调度、虚拟内存、文件系统、设备驱动、中断处理、权限隔离 |
| TPU Operator Kernel | TPU 算子内核 | TPU/NPU/GPU 加速器 | 执行 MatMul、Softmax、Attention、LayerNorm 等算子的具体硬件实现 |

Linux Kernel 通常不理解 MatMul、Softmax、Attention 的数学语义；它负责让用户态程序安全、稳定地访问 TPU 设备。真正执行矩阵乘的是 TPU 上的 operator kernel。

## 整体分层关系

下面这张图把推理/编译/执行链路放在一张分层图中。图中有两条容易混淆的路径：DLC/Compiler 偏离线编译，vLLM 偏在线推理服务调度。二者最终都需要通过后端 Runtime、Linux Driver 和硬件执行能力完成计算。

<div class="svg-figure">
<svg viewBox="0 0 1180 940" width="100%" role="img" aria-label="LLM to TPU layered architecture">
  <defs>
    <marker id="arrow-main" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M1,1 L9,5 L1,9 Z" fill="#4b5563"/></marker>
    <style>
      .box{fill:#ffffff;stroke:#cbd5e1;stroke-width:2;rx:16;filter:drop-shadow(0 8px 12px rgba(15,23,42,.08));}
      .blue{fill:#eff6ff;stroke:#93c5fd}.green{fill:#ecfdf5;stroke:#86efac}.amber{fill:#fffbeb;stroke:#fcd34d}.purple{fill:#f5f3ff;stroke:#c4b5fd}.gray{fill:#f8fafc;stroke:#cbd5e1}.cyan{fill:#ecfeff;stroke:#67e8f9}
      .title{font:700 18px 'Times New Roman','SimSun',serif;fill:#111827}.sub{font:14px 'Times New Roman','SimSun',serif;fill:#4b5563}.arrow{stroke:#4b5563;stroke-width:2.2;fill:none;marker-end:url(#arrow-main)}
    </style>
  </defs>
  <rect x="300" y="40" width="580" height="84" class="box blue"/><text x="590" y="74" text-anchor="middle" class="title">Large Language Model / 大语言模型</text><text x="590" y="100" text-anchor="middle" class="sub">Transformer Blocks, Weights, Tokenizer-facing Semantics</text>
  <path d="M590 124 L590 160" class="arrow"/>
  <rect x="300" y="160" width="580" height="84" class="box blue"/><text x="590" y="194" text-anchor="middle" class="title">Deep Learning Framework / 深度学习框架</text><text x="590" y="220" text-anchor="middle" class="sub">PyTorch, TensorFlow, JAX, Transformers</text>
  <path d="M590 244 L590 280" class="arrow"/>
  <rect x="300" y="280" width="580" height="84" class="box green"/><text x="590" y="314" text-anchor="middle" class="title">Computational Graph / 计算图</text><text x="590" y="340" text-anchor="middle" class="sub">Operators as Nodes, Tensors as Edges</text>
  <path d="M590 364 C590 400 350 392 330 430" class="arrow"/>
  <path d="M590 364 C590 400 830 392 850 430" class="arrow"/>
  <rect x="90" y="430" width="480" height="92" class="box purple"/><text x="330" y="466" text-anchor="middle" class="title">Compiler / DLC Toolchain</text><text x="330" y="494" text-anchor="middle" class="sub">Graph Optimization, Lowering, Code Generation</text>
  <rect x="610" y="430" width="480" height="92" class="box amber"/><text x="850" y="466" text-anchor="middle" class="title">vLLM Serving Engine</text><text x="850" y="494" text-anchor="middle" class="sub">Scheduler, Continuous Batching, PagedAttention, KV Cache</text>
  <path d="M330 522 C330 570 560 560 590 600" class="arrow"/>
  <path d="M850 522 C850 570 620 560 590 600" class="arrow"/>
  <rect x="300" y="600" width="580" height="84" class="box gray"/><text x="590" y="634" text-anchor="middle" class="title">Runtime / SDK / 用户态运行时</text><text x="590" y="660" text-anchor="middle" class="sub">Load Artifacts, Allocate Buffers, Submit Command Buffers</text>
  <path d="M590 684 L590 720" class="arrow"/>
  <rect x="300" y="720" width="580" height="84" class="box gray"/><text x="590" y="754" text-anchor="middle" class="title">Linux Kernel + TPU Device Driver</text><text x="590" y="780" text-anchor="middle" class="sub">ioctl, mmap, DMA Mapping, Command Queue, Interrupts</text>
  <path d="M590 804 L590 840" class="arrow"/>
  <rect x="300" y="840" width="580" height="84" class="box cyan"/><text x="590" y="874" text-anchor="middle" class="title">TPU Hardware / TPU 硬件</text><text x="590" y="900" text-anchor="middle" class="sub">Global Memory, Local SRAM, DMA, Matrix Units, Vector Units</text>
</svg>
</div>

## 术语逐层解释

### 大模型：高层函数与参数集合

大模型可以看作一个由大量参数组成的函数。以 LLM 为例，模型通常由 Embedding、若干 Transformer Block、Normalization、MLP、Attention 和 LM Head 组成。大模型本身描述的是整体函数形式，而不是目标硬件上的具体执行策略。

### 框架：模型表达、张量管理与图导出

深度学习框架负责描述模型结构、管理 Tensor、加载权重、执行自动微分、导出计算图，并调用后端 Runtime 或编译器。常见框架包括 PyTorch、TensorFlow、JAX、MindSpore 和 PaddlePaddle。

例如 PyTorch 代码：

```python
y = torch.softmax(x @ w, dim=-1)
```

从图语义上看，这段代码至少包含矩阵乘和 Softmax 两类算子。框架层负责表达这个关系，但不一定决定它最终在 TPU 上如何分块、如何搬运数据、如何复用片上缓存。

### Tensor：数据与元数据的组合

Tensor 不只是多维数组，还包含一组影响编译和执行的重要元数据。

| 属性 | 含义 | 示例 |
| --- | --- | --- |
| shape | 维度大小 | $[B,S,H]$ |
| dtype | 元素类型 | FP32、FP16、BF16、INT8 |
| layout | 内存布局 | NCHW、NHWC、blocked layout |
| stride | 各维度内存步长 | contiguous / non-contiguous |
| device | 所在设备 | CPU、GPU、TPU、NPU |

LLM 中常见 hidden states 可以表示为 $[B,S,H]$，例如 $[1,2048,4096]$。

### Operator：计算图中的数学语义节点

Operator 定义“算什么”。例如 Elementwise Add、Reduction Sum、MatMul、Softmax、RMSNorm、Attention 都是算子。以矩阵乘为例：

$$
C_{i,j}=\sum_{k=1}^{K}A_{i,k}B_{k,j}
$$

如果 $A\in\mathbb{R}^{M\times K}$，$B\in\mathbb{R}^{K\times N}$，那么输出 $C\in\mathbb{R}^{M\times N}$。这个定义描述的是数学语义，不描述具体硬件上如何高效执行。

### Operator Kernel：目标硬件上的执行实现

Operator Kernel 定义“怎么在硬件上算”。同一个 MatMul operator 可以对应 CPU kernel、CUDA kernel、Triton kernel、TPU kernel 或 NPU kernel。TPU MatMul kernel 通常会涉及 tile 划分、global memory 到 local SRAM 的搬运、DMA、matrix unit 调用、partial sum 累加和写回。

<div class="svg-figure">
<svg viewBox="0 0 1120 430" width="100%" role="img" aria-label="operator kernel hardware mapping">
  <defs>
    <marker id="arrow-ok" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M1,1 L9,5 L1,9 Z" fill="#475569"/></marker>
    <style>.okbox{fill:#fff;stroke:#cbd5e1;stroke-width:2;rx:16}.okblue{fill:#eff6ff;stroke:#93c5fd}.okamber{fill:#fffbeb;stroke:#fcd34d}.okgreen{fill:#ecfdf5;stroke:#86efac}.okt{font:700 17px 'Times New Roman','SimSun',serif;fill:#111827}.oks{font:14px 'Times New Roman','SimSun',serif;fill:#475569}.oka{stroke:#475569;stroke-width:2.2;fill:none;marker-end:url(#arrow-ok)}</style>
  </defs>
  <rect x="60" y="90" width="280" height="160" class="okbox okblue"/><text x="200" y="130" text-anchor="middle" class="okt">Operator / 算子</text><text x="200" y="166" text-anchor="middle" class="oks">数学语义</text><text x="200" y="196" text-anchor="middle" class="oks">Cᵢⱼ = Σₖ AᵢₖBₖⱼ</text>
  <rect x="420" y="90" width="280" height="160" class="okbox okamber"/><text x="560" y="130" text-anchor="middle" class="okt">Operator Kernel / 算子内核</text><text x="560" y="166" text-anchor="middle" class="oks">tiling, scheduling</text><text x="560" y="196" text-anchor="middle" class="oks">load → compute → store</text>
  <rect x="780" y="90" width="280" height="160" class="okbox okgreen"/><text x="920" y="130" text-anchor="middle" class="okt">Hardware / TPU 硬件</text><text x="920" y="166" text-anchor="middle" class="oks">SRAM, DMA</text><text x="920" y="196" text-anchor="middle" class="oks">matrix units, vector units</text>
  <path d="M340 170 L420 170" class="oka"/><path d="M700 170 L780 170" class="oka"/>
  <text x="380" y="150" text-anchor="middle" class="oks">lowering</text><text x="740" y="150" text-anchor="middle" class="oks">execute</text>
</svg>
</div>

## DLC / 编译器：从图语义到硬件执行计划

DLC 在不同厂商语境下可能表示 **Deep Learning Compiler**，也可能表示 **Compiled Model Artifact**。这两种含义都围绕同一件事：把高层模型图转换为目标硬件上可执行、正确且高效的执行计划。

| 生态 | 常见编译产物 |
| --- | --- |
| NVIDIA TensorRT | engine / plan |
| Qualcomm SNPE | dlc |
| Ascend | om |
| Sophon | bmodel |
| 其他 TPU/NPU 厂商 | model binary / compiled artifact |

典型编译链路如下：

<div class="svg-figure">
<svg viewBox="0 0 1260 460" width="100%" role="img" aria-label="deep learning compiler pipeline">
  <defs>
    <marker id="arrow-compile" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M1,1 L9,5 L1,9 Z" fill="#2563eb"/></marker>
    <style>.cbox{fill:#fff;stroke:#cbd5e1;stroke-width:2;rx:14}.c1{fill:#eff6ff;stroke:#93c5fd}.c2{fill:#ecfdf5;stroke:#86efac}.c3{fill:#fffbeb;stroke:#fcd34d}.ct{font:700 15px 'Times New Roman','SimSun',serif;fill:#111827}.cs{font:13px 'Times New Roman','SimSun',serif;fill:#475569}.ca{stroke:#2563eb;stroke-width:2.2;fill:none;marker-end:url(#arrow-compile)}</style>
  </defs>
  <rect x="40" y="70" width="170" height="86" class="cbox c1"/><text x="125" y="104" text-anchor="middle" class="ct">Model Export</text><text x="125" y="130" text-anchor="middle" class="cs">PyTorch / ONNX</text>
  <rect x="260" y="70" width="170" height="86" class="cbox c1"/><text x="345" y="104" text-anchor="middle" class="ct">Graph IR</text><text x="345" y="130" text-anchor="middle" class="cs">operators + tensors</text>
  <rect x="480" y="70" width="180" height="86" class="cbox c2"/><text x="570" y="104" text-anchor="middle" class="ct">Graph Optimize</text><text x="570" y="130" text-anchor="middle" class="cs">fusion, folding</text>
  <rect x="710" y="70" width="180" height="86" class="cbox c2"/><text x="800" y="104" text-anchor="middle" class="ct">Lowering</text><text x="800" y="130" text-anchor="middle" class="cs">Graph IR → Tensor IR</text>
  <rect x="940" y="70" width="180" height="86" class="cbox c3"/><text x="1030" y="104" text-anchor="middle" class="ct">Scheduling</text><text x="1030" y="130" text-anchor="middle" class="cs">tiling, layout</text>
  <path d="M210 113 L260 113" class="ca"/><path d="M430 113 L480 113" class="ca"/><path d="M660 113 L710 113" class="ca"/><path d="M890 113 L940 113" class="ca"/>
  <rect x="260" y="260" width="180" height="86" class="cbox c3"/><text x="350" y="294" text-anchor="middle" class="ct">Memory Planning</text><text x="350" y="320" text-anchor="middle" class="cs">buffers, reuse</text>
  <rect x="530" y="260" width="180" height="86" class="cbox c3"/><text x="620" y="294" text-anchor="middle" class="ct">Code Generation</text><text x="620" y="320" text-anchor="middle" class="cs">kernel binary</text>
  <rect x="800" y="260" width="180" height="86" class="cbox c1"/><text x="890" y="294" text-anchor="middle" class="ct">Compiled Artifact</text><text x="890" y="320" text-anchor="middle" class="cs">DLC / model binary</text>
  <rect x="1070" y="260" width="150" height="86" class="cbox c2"/><text x="1145" y="294" text-anchor="middle" class="ct">Runtime</text><text x="1145" y="320" text-anchor="middle" class="cs">load + execute</text>
  <path d="M1030 156 C1030 210 350 205 350 260" class="ca"/><path d="M440 303 L530 303" class="ca"/><path d="M710 303 L800 303" class="ca"/><path d="M980 303 L1070 303" class="ca"/>
</svg>
</div>

其中，Graph IR 保留较高层的算子语义；Tensor IR 更接近循环、索引和 buffer；Kernel IR 更接近目标硬件执行方式。编译器的核心任务不是“把 Python 翻译成二进制”这么简单，而是同时决定算子融合、layout、tiling、memory planning、kernel selection 和 code generation。

## vLLM：在线推理服务引擎，不是离线编译器

vLLM 的定位是 **LLM inference serving engine**。它关注在线推理系统问题：如何接收多用户请求，如何调度 prefill 和 decode，如何做 continuous batching，如何管理 KV cache，如何通过后端 Runtime 调用硬件 kernel。

<div class="svg-figure">
<svg viewBox="0 0 1120 620" width="100%" role="img" aria-label="vLLM serving stack">
  <defs>
    <marker id="arrow-vllm" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M1,1 L9,5 L1,9 Z" fill="#15803d"/></marker>
    <style>.vbox{fill:#fff;stroke:#cbd5e1;stroke-width:2;rx:14}.v1{fill:#eff6ff;stroke:#93c5fd}.v2{fill:#ecfdf5;stroke:#86efac}.v3{fill:#fffbeb;stroke:#fcd34d}.v4{fill:#f5f3ff;stroke:#c4b5fd}.vt{font:700 16px 'Times New Roman','SimSun',serif;fill:#111827}.vs{font:13px 'Times New Roman','SimSun',serif;fill:#475569}.va{stroke:#15803d;stroke-width:2.2;fill:none;marker-end:url(#arrow-vllm)}</style>
  </defs>
  <rect x="60" y="60" width="250" height="78" class="vbox v1"/><text x="185" y="92" text-anchor="middle" class="vt">Client Requests</text><text x="185" y="116" text-anchor="middle" class="vs">OpenAI API / Python API</text>
  <rect x="435" y="60" width="250" height="78" class="vbox v1"/><text x="560" y="92" text-anchor="middle" class="vt">Tokenizer</text><text x="560" y="116" text-anchor="middle" class="vs">text ↔ token ids</text>
  <rect x="810" y="60" width="250" height="78" class="vbox v2"/><text x="935" y="92" text-anchor="middle" class="vt">Request Scheduler</text><text x="935" y="116" text-anchor="middle" class="vs">admission and priority</text>
  <path d="M310 99 L435 99" class="va"/><path d="M685 99 L810 99" class="va"/>
  <rect x="160" y="240" width="280" height="92" class="vbox v2"/><text x="300" y="276" text-anchor="middle" class="vt">Continuous Batching</text><text x="300" y="304" text-anchor="middle" class="vs">merge prefill and decode workloads</text>
  <rect x="680" y="240" width="280" height="92" class="vbox v3"/><text x="820" y="276" text-anchor="middle" class="vt">PagedAttention</text><text x="820" y="304" text-anchor="middle" class="vs">logical tokens → physical KV blocks</text>
  <path d="M935 138 C935 195 820 190 820 240" class="va"/><path d="M900 138 C900 200 300 190 300 240" class="va"/><path d="M440 286 L680 286" class="va"/>
  <rect x="110" y="460" width="260" height="86" class="vbox v4"/><text x="240" y="494" text-anchor="middle" class="vt">Model Executor</text><text x="240" y="520" text-anchor="middle" class="vs">prefill, decode, sampling</text>
  <rect x="430" y="460" width="260" height="86" class="vbox"/><text x="560" y="494" text-anchor="middle" class="vt">Backend Runtime</text><text x="560" y="520" text-anchor="middle" class="vs">PyTorch / XLA / vendor SDK</text>
  <rect x="750" y="460" width="260" height="86" class="vbox v2"/><text x="880" y="494" text-anchor="middle" class="vt">GPU / TPU / NPU</text><text x="880" y="520" text-anchor="middle" class="vs">operator kernels execute</text>
  <path d="M300 332 L240 460" class="va"/><path d="M820 332 C820 410 260 400 240 460" class="va"/><path d="M370 503 L430 503" class="va"/><path d="M690 503 L750 503" class="va"/>
</svg>
</div>

vLLM 与 DLC/Compiler 的边界可以这样理解：vLLM 管在线请求与 KV cache，编译器管图优化与 kernel 生成，Linux driver 管设备访问，TPU hardware 管实际计算。若要让 vLLM 使用 TPU/NPU，需要 XLA/torch_xla、厂商 runtime adapter 或专门的 backend 支持。

## Prefill、Decode、KV Cache 与 PagedAttention

LLM 推理通常分成两个阶段：Prefill 和 Decode。Prefill 处理完整 prompt，计算量大但序列维度并行度高；Decode 每次生成一个 token，单步计算较小，但需要反复读取历史 KV cache。

KV cache 的近似内存占用可以写成：

$$
KV_{bytes}=2 \times L \times B \times S \times H_{kv} \times D_{head} \times bytes(dtype)
$$

其中 <script type="math/tex">2</script> 表示 K 和 V，<script type="math/tex">L</script> 是层数，<script type="math/tex">B</script> 是 batch size，<script type="math/tex">S</script> 是序列长度，<script type="math/tex">H_{kv}</script> 是 KV head 数，<script type="math/tex">D_{head}</script> 是每个 head 的维度。

<div class="svg-figure">
<svg viewBox="0 0 1120 520" width="100%" role="img" aria-label="prefill decode pagedattention kv cache">
  <defs>
    <marker id="arrow-kv" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M1,1 L9,5 L1,9 Z" fill="#15803d"/></marker>
    <style>.kbox{fill:#fff;stroke:#cbd5e1;stroke-width:2;rx:14}.k1{fill:#eff6ff;stroke:#93c5fd}.k2{fill:#ecfdf5;stroke:#86efac}.k3{fill:#fffbeb;stroke:#fcd34d}.kt{font:700 16px 'Times New Roman','SimSun',serif;fill:#111827}.ks{font:13px 'Times New Roman','SimSun',serif;fill:#475569}.ka{stroke:#15803d;stroke-width:2.2;fill:none;marker-end:url(#arrow-kv)}</style>
  </defs>
  <rect x="70" y="60" width="220" height="80" class="kbox k1"/><text x="180" y="92" text-anchor="middle" class="kt">Prompt Tokens</text><text x="180" y="116" text-anchor="middle" class="ks">input sequence</text>
  <rect x="390" y="60" width="220" height="80" class="kbox k2"/><text x="500" y="92" text-anchor="middle" class="kt">Prefill</text><text x="500" y="116" text-anchor="middle" class="ks">build KV cache</text>
  <rect x="710" y="60" width="220" height="80" class="kbox k3"/><text x="820" y="92" text-anchor="middle" class="kt">Decode</text><text x="820" y="116" text-anchor="middle" class="ks">read cache, append KV</text>
  <path d="M290 100 L390 100" class="ka"/><path d="M610 100 L710 100" class="ka"/>
  <rect x="90" y="250" width="940" height="190" class="kbox"/><text x="560" y="286" text-anchor="middle" class="kt">PagedAttention: logical token blocks → physical KV blocks</text>
  <rect x="160" y="330" width="110" height="48" class="kbox k1"/><text x="215" y="359" text-anchor="middle" class="ks">logical 0</text>
  <rect x="290" y="330" width="110" height="48" class="kbox k1"/><text x="345" y="359" text-anchor="middle" class="ks">logical 1</text>
  <rect x="420" y="330" width="110" height="48" class="kbox k1"/><text x="475" y="359" text-anchor="middle" class="ks">logical 2</text>
  <rect x="640" y="330" width="110" height="48" class="kbox k3"/><text x="695" y="359" text-anchor="middle" class="ks">page 17</text>
  <rect x="770" y="330" width="110" height="48" class="kbox k3"/><text x="825" y="359" text-anchor="middle" class="ks">page 04</text>
  <rect x="900" y="330" width="110" height="48" class="kbox k3"/><text x="955" y="359" text-anchor="middle" class="ks">page 29</text>
  <path d="M270 354 L640 354" class="ka"/><path d="M400 354 L770 354" class="ka"/><path d="M530 354 L900 354" class="ka"/>
</svg>
</div>

PagedAttention 的核心思想类似虚拟内存分页：逻辑上连续的 token block 不要求物理上连续存放。这样可以降低 KV cache 碎片，提高长上下文和多请求并发时的设备内存利用率。

## TPU 与 Linux 内核的关系

TPU 是加速器硬件，Linux Kernel 是操作系统内核。用户态 Runtime 通常通过 `ioctl`、`mmap`、`poll` 等接口与 TPU device driver 通信；驱动再负责 DMA 映射、command queue、interrupt handling 和错误状态返回。

<div class="svg-figure">
<svg viewBox="0 0 1180 620" width="100%" role="img" aria-label="linux kernel tpu driver architecture">
  <defs>
    <marker id="arrow-linux" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M1,1 L9,5 L1,9 Z" fill="#b91c1c"/></marker>
    <style>.lbox{fill:#fff;stroke:#cbd5e1;stroke-width:2;rx:14}.l1{fill:#eff6ff;stroke:#93c5fd}.l2{fill:#fffbeb;stroke:#fcd34d}.l3{fill:#ecfdf5;stroke:#86efac}.lt{font:700 16px 'Times New Roman','SimSun',serif;fill:#111827}.ls{font:13px 'Times New Roman','SimSun',serif;fill:#475569}.la{stroke:#b91c1c;stroke-width:2.2;fill:none;marker-end:url(#arrow-linux)}</style>
  </defs>
  <rect x="40" y="40" width="1100" height="130" class="lbox l1"/><text x="80" y="75" class="lt">User Space / 用户态</text>
  <rect x="170" y="90" width="220" height="58" class="lbox"/><text x="280" y="125" text-anchor="middle" class="lt">Application</text>
  <rect x="520" y="90" width="220" height="58" class="lbox"/><text x="630" y="125" text-anchor="middle" class="lt">TPU Runtime / SDK</text>
  <rect x="860" y="90" width="220" height="58" class="lbox"/><text x="970" y="125" text-anchor="middle" class="lt">DLC / Model Binary</text>
  <path d="M390 119 L520 119" class="la"/><path d="M860 119 L740 119" class="la"/>
  <rect x="40" y="245" width="1100" height="150" class="lbox l2"/><text x="80" y="280" class="lt">Kernel Space / Linux 内核态</text>
  <rect x="160" y="310" width="220" height="58" class="lbox"/><text x="270" y="345" text-anchor="middle" class="lt">ioctl / mmap / poll</text>
  <rect x="480" y="310" width="240" height="58" class="lbox"/><text x="600" y="345" text-anchor="middle" class="lt">TPU Device Driver</text>
  <rect x="820" y="310" width="220" height="58" class="lbox"/><text x="930" y="345" text-anchor="middle" class="lt">DMA / IOMMU</text>
  <path d="M630 148 L630 310" class="la"/><path d="M380 339 L480 339" class="la"/><path d="M720 339 L820 339" class="la"/>
  <rect x="40" y="470" width="1100" height="110" class="lbox l3"/><text x="80" y="505" class="lt">Device / TPU 硬件</text>
  <rect x="220" y="525" width="190" height="40" class="lbox"/><text x="315" y="551" text-anchor="middle" class="ls">Command Queue</text>
  <rect x="500" y="525" width="190" height="40" class="lbox"/><text x="595" y="551" text-anchor="middle" class="ls">TPU Cores</text>
  <rect x="780" y="525" width="190" height="40" class="lbox"/><text x="875" y="551" text-anchor="middle" class="ls">Interrupt</text>
  <path d="M930 368 L930 525" class="la"/><path d="M410 545 L500 545" class="la"/><path d="M690 545 L780 545" class="la"/><path d="M875 525 C1060 450 1030 350 1040 339" class="la"/>
</svg>
</div>

因此，一个 MatMul 从 Python 到 TPU 的路径可以用专业语言概括为：框架识别 `aten::matmul` 或等价图节点，导出到 ONNX 或厂商 IR，编译器执行 lowering 和 kernel selection，Runtime 加载 compiled artifact 并提交 command buffer，Linux TPU driver 完成设备通信，TPU hardware 执行目标 MatMul kernel。

## 例子一：Linear + Bias + ReLU 融合

框架层表达式如下：

```python
y = torch.relu(x @ w + b)
```

数学形式是：

$$
Y=max(0,XW+b)
$$

如果不融合，执行路径包含 MatMul、Add、ReLU 三个算子，可能产生两个中间张量；如果融合，编译器可以生成一个 `Fused MatMul + Bias + ReLU Kernel`，在累加完成后直接加 bias、应用激活函数并写回最终输出。

<div class="svg-figure">
<svg viewBox="0 0 1120 460" width="100%" role="img" aria-label="linear bias relu fusion">
  <defs>
    <marker id="arrow-fuse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M1,1 L9,5 L1,9 Z" fill="#2563eb"/></marker>
    <style>.fbox{fill:#fff;stroke:#cbd5e1;stroke-width:2;rx:14}.f1{fill:#eff6ff;stroke:#93c5fd}.f2{fill:#ecfdf5;stroke:#86efac}.ft{font:700 16px 'Times New Roman','SimSun',serif;fill:#111827}.fs{font:13px 'Times New Roman','SimSun',serif;fill:#475569}.fa{stroke:#2563eb;stroke-width:2.2;fill:none;marker-end:url(#arrow-fuse)}</style>
  </defs>
  <text x="60" y="50" class="ft">Before Fusion / 融合前</text>
  <rect x="70" y="90" width="150" height="68" class="fbox f1"/><text x="145" y="130" text-anchor="middle" class="ft">MatMul</text>
  <rect x="310" y="90" width="150" height="68" class="fbox f1"/><text x="385" y="130" text-anchor="middle" class="ft">Add</text>
  <rect x="550" y="90" width="150" height="68" class="fbox f1"/><text x="625" y="130" text-anchor="middle" class="ft">ReLU</text>
  <rect x="790" y="90" width="150" height="68" class="fbox"/><text x="865" y="130" text-anchor="middle" class="ft">Output</text>
  <path d="M220 124 L310 124" class="fa"/><path d="M460 124 L550 124" class="fa"/><path d="M700 124 L790 124" class="fa"/>
  <text x="265" y="178" text-anchor="middle" class="fs">tmp1</text><text x="505" y="178" text-anchor="middle" class="fs">tmp2</text>
  <text x="60" y="280" class="ft">After Fusion / 融合后</text>
  <rect x="120" y="320" width="460" height="78" class="fbox f2"/><text x="350" y="352" text-anchor="middle" class="ft">Fused MatMul + Bias + ReLU Kernel</text><text x="350" y="378" text-anchor="middle" class="fs">accumulate → add bias → activation → store once</text>
  <rect x="760" y="325" width="160" height="68" class="fbox"/><text x="840" y="365" text-anchor="middle" class="ft">Output</text>
  <path d="M580 359 L760 359" class="fa"/>
</svg>
</div>

融合的直接收益是减少中间 tensor、减少 global memory 读写、减少 kernel launch overhead，并提升端到端吞吐。

## 例子二：Scaled Dot-Product Attention

标准注意力公式是：

$$
O=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

普通实现可能 materialize `scores` 和 `probs` 两个 $[B,H,S,S]$ 规模的中间矩阵。当序列长度 $S$ 很大时，内存读写会成为主要瓶颈。

<div class="svg-figure">
<svg viewBox="0 0 1180 520" width="100%" role="img" aria-label="scaled dot product attention">
  <defs>
    <marker id="arrow-attn" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M1,1 L9,5 L1,9 Z" fill="#7c3aed"/></marker>
    <style>.abox{fill:#fff;stroke:#cbd5e1;stroke-width:2;rx:14}.a1{fill:#eff6ff;stroke:#93c5fd}.a2{fill:#fffbeb;stroke:#fcd34d}.a3{fill:#ecfdf5;stroke:#86efac}.at{font:700 16px 'Times New Roman','SimSun',serif;fill:#111827}.as{font:13px 'Times New Roman','SimSun',serif;fill:#475569}.aa{stroke:#7c3aed;stroke-width:2.2;fill:none;marker-end:url(#arrow-attn)}</style>
  </defs>
  <rect x="60" y="70" width="130" height="60" class="abox a1"/><text x="125" y="107" text-anchor="middle" class="at">Q</text>
  <rect x="60" y="210" width="130" height="60" class="abox a1"/><text x="125" y="247" text-anchor="middle" class="at">K</text>
  <rect x="60" y="350" width="130" height="60" class="abox a1"/><text x="125" y="387" text-anchor="middle" class="at">V</text>
  <rect x="300" y="145" width="170" height="76" class="abox a2"/><text x="385" y="178" text-anchor="middle" class="at">QKᵀ</text><text x="385" y="202" text-anchor="middle" class="as">scores</text>
  <rect x="560" y="145" width="170" height="76" class="abox a2"/><text x="645" y="178" text-anchor="middle" class="at">Scale</text><text x="645" y="202" text-anchor="middle" class="as">1 / sqrt(dₖ)</text>
  <rect x="820" y="145" width="170" height="76" class="abox a3"/><text x="905" y="178" text-anchor="middle" class="at">Softmax</text><text x="905" y="202" text-anchor="middle" class="as">probabilities</text>
  <rect x="560" y="340" width="170" height="76" class="abox a2"/><text x="645" y="373" text-anchor="middle" class="at">MatMul V</text><text x="645" y="397" text-anchor="middle" class="as">weighted sum</text>
  <rect x="820" y="340" width="170" height="76" class="abox"/><text x="905" y="373" text-anchor="middle" class="at">Output O</text><text x="905" y="397" text-anchor="middle" class="as">[B,H,S,D]</text>
  <path d="M190 100 C245 100 250 183 300 183" class="aa"/><path d="M190 240 C245 240 250 183 300 183" class="aa"/><path d="M470 183 L560 183" class="aa"/><path d="M730 183 L820 183" class="aa"/><path d="M905 221 C905 300 730 300 730 378" class="aa"/><path d="M190 380 L560 380" class="aa"/><path d="M730 378 L820 378" class="aa"/>
  <rect x="1010" y="165" width="130" height="190" class="abox"/><text x="1075" y="200" text-anchor="middle" class="at">Optimization</text><text x="1075" y="232" text-anchor="middle" class="as">blocking</text><text x="1075" y="260" text-anchor="middle" class="as">online softmax</text><text x="1075" y="288" text-anchor="middle" class="as">avoid S×S</text><text x="1075" y="316" text-anchor="middle" class="as">less HBM traffic</text>
</svg>
</div>

FlashAttention 类优化的核心不是改变数学公式，而是改变执行方式：按 block 计算 $QK^T$，在线维护 softmax 的 max 和 sum，不完整写出 scores/probs，从而减少 HBM/global memory traffic。

## 优化到底在优化什么

| 优化项 | 专业含义 | 主要目的 | 例子 |
| --- | --- | --- | --- |
| Operator Fusion | 多个相邻算子合成 fused kernel | 减少中间张量和 kernel launch | MatMul + Bias + ReLU |
| Layout Optimization | 选择硬件友好的内存布局 | 提高连续访存和对齐效率 | NCHW、NHWC、blocked layout |
| Memory Planning | 规划 buffer 生命周期和复用 | 降低峰值设备内存 | activation buffer reuse |
| Kernel Selection | 根据 shape/dtype/layout 选实现 | 提高单算子性能 | small GEMM vs large GEMM |
| Tiling / Blocking | 把大计算切成 tile | 提高片上缓存复用 | MatMul 的 M/N/K tile |
| Quantization | 降低数值精度 | 降低带宽和存储，提高吞吐 | FP16、BF16、INT8、INT4 |
| DMA/Compute Overlap | 搬运和计算流水化 | 隐藏 DMA latency | double buffering |
| Continuous Batching | 动态合并在线请求 | 提高 LLM serving 吞吐 | vLLM scheduler |
| KV Cache Paging | 按 block 管理 KV cache | 降低碎片，提高并发 | vLLM PagedAttention |

Arithmetic intensity 可以帮助判断算子更可能受计算限制还是带宽限制：

$$
AI=\frac{FLOPs}{Bytes\ moved}
$$

如果 $AI$ 很低，算子通常更容易受内存带宽限制；如果 $AI$ 很高，才更可能受计算单元吞吐限制。

## 最后总结

整条链路可以概括为：Large Model 定义任务语义，Framework 或 vLLM 负责模型表达与在线服务，Computational Graph 将模型分解为 Operators，Compiler 或 Backend Runtime 选择和生成 Operator Kernels，Runtime 通过 Linux Driver 提交硬件任务，TPU Hardware 最终执行这些算子内核。

更短地说：模型由算子组成，算子由硬件内核执行，Runtime 通过 Linux 驱动把硬件任务提交给 TPU。Linux kernel 让 TPU 能被系统安全访问；TPU operator kernel 让算子能在 TPU 上高效计算；vLLM 让大模型在线推理能高吞吐地调度请求和管理 KV cache。
