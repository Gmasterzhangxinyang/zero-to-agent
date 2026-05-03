# LLM 关键专业词汇速查

---

## 硬件与存储

### 三层存储结构
```
SSD（存储）→ 读取到 → 内存/显存（运行）→ 断电清空 → 重新从SSD读
持久化，断电不丢    临时，断电清空
慢（几GB/s）        快（几十~几百GB/s）
```

- **SSD**：模型文件、代码、checkpoint 都存这里
- **内存（RAM）**：系统、应用、CPU跑模型时的权重
- **显存（VRAM）**：GPU跑模型时的权重、KV Cache、中间计算结果

### CUDA / GPU
NVIDIA 发明的 GPU 通用计算框架，让 GPU 不只能渲染图形，还能做矩阵运算。

```
GPU：几千个弱核心，擅长大量简单并行计算（矩阵乘法）
CPU：几十个强核心，擅长复杂逻辑串行计算

模型推理 = 大量矩阵乘法 → 天然适合GPU
```

PyTorch 底层调 CUDA，你只需要写 `model.to("cuda")`，底下的事 PyTorch 全帮你处理。

### 为什么显存是瓶颈，不是算力
```
7B 模型权重：float16 → 14GB 显存
仅仅装进去就要14GB，还没开始算KV Cache...

算力不够 → 生成慢，但至少能跑
显存不够 → 直接报错，跑不起来
```

4090 算力比 3090 强很多，但两者显存都是 24GB，能跑的模型一样大。

### Mac M系列的优势（统一内存）
```
普通电脑：内存（CPU用）+ 显存（GPU用）= 两块独立的，互相搬数据有延迟
Mac M系列：统一内存，CPU 和 GPU 共享同一块，没有搬运延迟
```

M4 24GB 统一内存全都能给模型用，相当于 24GB 显存，能流畅跑 13B 模型。

### 没有GPU能跑吗
可以，但慢：
```
有GPU（显存8GB）：  装得下 → 快（50~100 token/s）
纯CPU（内存32GB）： 装得下 → 慢（3~10 token/s）
内存不够：          装不下 → 报错
```
瓶颈从显存变成了内存带宽+CPU算力，反而更严重。

---

## 架构相关

### Transformer
目前所有主流大模型的基础架构（GPT、LLaMA、Qwen 都是）。

```
输入文字
  ↓ Tokenizer（文字→token ID）
  ↓ Embedding（token ID→高维向量）
  ↓ N × Transformer Block
      ├── Self-Attention：让每个token"看到"其他所有token
      └── FFN：对每个token做非线性变换
  ↓ 输出层（向量→每个token的概率分布）
  ↓ 采样（按temperature选下一个token）
```

### Attention / Self-Attention
模型计算每个 token 和其他所有 token 的相关性，决定"生成这个词时应该重点关注哪些上下文"。

```
"它吃了苹果因为它饿了"
生成第二个"它"时，Attention 会高度关注"苹果"和"饿了"，判断指的是谁
```

### KV Cache
**推理加速的核心：用显存换速度，避免重复计算。**

模型生成每个 token 时，需要对所有历史 token 做 Attention 计算（算 Key 和 Value 矩阵）。KV Cache 把这些中间结果缓存起来，下次直接用。

```
没有 KV Cache：生成第100个token，要重新算前99个token的K/V
有  KV Cache：前99个K/V已缓存，只算第100个，速度快几十倍
```

代价：占显存，上下文越长 KV Cache 越大，这也是长对话越来越慢的原因。

**对你透明**：KV Cache 是推理框架（Ollama/vLLM）自动管理的，调接口时完全感知不到，只有自己部署服务时才需要关心。

### MHA / GQA / MQA
Attention 的变体，影响速度和显存：

| 名称 | 全称 | 特点 |
|------|------|------|
| MHA | Multi-Head Attention | 标准版，每个头独立K/V |
| MQA | Multi-Query Attention | 所有头共享K/V，省显存快 |
| GQA | Grouped-Query Attention | 折中，LLaMA3/Qwen2用这个 |

### FFN / MLP
Transformer Block 里 Attention 之后的前馈网络，做非线性变换。参数量占模型总参数的 2/3 左右。

### MoE（Mixture of Experts）
把 FFN 拆成多个"专家"，每次只激活其中几个。
- 总参数大，但计算量小
- 代表模型：Mixtral、DeepSeek、GPT-4（据传）

---

## 训练相关

### Checkpoint
训练过程中定期把显存里的模型权重保存到 SSD 的快照。

```
训练进度：
step 1000 → 保存 checkpoint-1000/（SSD）
step 2000 → 保存 checkpoint-2000/（SSD）
step 3000 → 服务器断电！从 checkpoint-2000 继续，不用从头来
```

checkpoint 本质就是一堆文件：
```
checkpoint-2000/
├── model.safetensors  # 模型权重（最大，几GB）
├── config.json        # 模型结构配置
├── tokenizer.json     # tokenizer
└── trainer_state.json # 当前step、loss等训练状态
```

HuggingFace 上下载的模型本质上也是一个 checkpoint，只是训练完的最终版本。

### Pre-training（预训练）
在海量文本上做 next-token prediction，让模型学会语言规律。
数据量：万亿 token 级别，训练时间：数月，成本：数千万美元。

### SFT（Supervised Fine-Tuning，监督微调）
在预训练模型基础上，用高质量的"问题-答案"对继续训练，让模型学会按指令回答，而不只是续写文本。

### RLHF（Reinforcement Learning from Human Feedback）
用人类偏好数据训练一个奖励模型，再用强化学习让 LLM 输出更符合人类偏好。ChatGPT 用这个方法对齐。

### DPO（Direct Preference Optimization）
RLHF 的简化版，不需要单独训练奖励模型，直接用偏好数据对比训练。更稳定，现在更流行。

### LoRA（Low-Rank Adaptation）
高效微调方法。不改动原始权重，只在旁边加两个小矩阵（A 和 B）：

```
原始权重 W（冻结，不训练）
新增 ΔW = A × B（只训练这两个小矩阵）
实际输出 = W + ΔW
```

参数量只有全量微调的 1%，效果接近。

### QLoRA
LoRA + 量化。把原始模型用 4bit 量化压缩（省显存），再用 LoRA 微调。24GB 显存可以微调 70B 模型。

---

## 推理相关

### Tokenizer
把文字转成 token ID 的工具，每个模型有自己的 tokenizer。同一段文字，不同 tokenizer 切出来的 token 数不同。

```
tiktoken（OpenAI）：为英文优化，中文被拆碎（"世" → 2个token）
Qwen tokenizer：   为中文优化，1个汉字基本就是1个token
```

### Context Window（上下文窗口）
模型一次能处理的最大 token 数（输入+输出）。超出就截断，模型"忘记"最早的内容。

```
qwen2.5:3b  → 32k tokens  ≈ 2.5万汉字
GPT-4o      → 128k tokens ≈ 10万汉字
Claude 3.5  → 200k tokens ≈ 15万汉字
```

### 多轮对话的本质
模型本身无状态，"记忆"靠每次把完整历史传进去：

```python
history = [
    {"role": "user",      "content": "我叫Bobby"},
    {"role": "assistant", "content": "你好Bobby"},
    {"role": "user",      "content": "我叫什么？"},  # 模型能看到前两条才能回答
]
```

history 越长，消耗 token 越多，超出 context window 就会忘记最早的内容。

### Prefill vs Decode
推理分两个阶段：
- **Prefill**：一次性处理整个输入 prompt，生成 KV Cache，很快
- **Decode**：逐个生成输出 token，每次用 KV Cache 加速，这是慢的部分

### TTFT / TPS
- **TTFT**（Time To First Token）：从发请求到收到第一个 token 的延迟，取决于 Prefill 速度
- **TPS**（Tokens Per Second）：生成速度

### Quantization（量化）
把模型权重从 float32/float16 压缩到 int8/int4，减少显存占用和计算量。精度略有损失，但通常可接受。

```
原始 7B 模型：float16 → 14GB 显存
量化后 7B 模型：4bit   → 4GB 显存
```

### Sampling（采样）
模型每次输出的是所有 token 的概率分布，采样策略决定怎么从中选一个：

- **Greedy**：每次选概率最高的（temperature=0），确定性强
- **Top-k**：从概率最高的 k 个里随机选
- **Top-p**：从累积概率达到 p 的候选里随机选（nucleus sampling）
- **Beam Search**：同时维护多条候选序列，取最优

---

## 工程相关

### 推理框架对比
| | Ollama | vLLM | HuggingFace Transformers |
|---|---|---|---|
| 适合场景 | 本地开发/学习 | 生产部署 | 研究/训练 |
| 并发能力 | 1~2个请求 | 几百个并发 | 低 |
| 硬件要求 | 普通Mac/PC | GPU服务器 | 任意 |
| 安装难度 | 一条命令 | 需要配置 | pip install |
| 核心技术 | 自动量化+Metal/CUDA | PagedAttention | 标准实现 |

同一个模型权重可以用任意框架跑，框架只影响速度和并发，不影响模型能力。

### HuggingFace / transformers
最主流的模型库，几乎所有开源模型都在上面，提供统一的加载/训练接口。

---

## 评估相关

### Perplexity（困惑度）
模型对文本的"意外程度"，越低说明模型预测越准。训练时用来监控进度，但不能代表实际能力（背下训练集也能困惑度极低）。

### Benchmark
标准化考试，客观对比不同模型能力。一个模型可能 MMLU 高但代码差，所以要同时看多个：

| Benchmark | 测什么 | 示例 |
|---|---|---|
| MMLU | 57学科选择题，测通用知识广度 | 青霉素的发现者是？ |
| HumanEval | Python编程题+单元测试，测代码能力 | 补全函数实现 |
| GSM8K | 小学数学应用题，测多步推理能力 | 小明有5个苹果... |
