# Day 1 — LLM 推理基础

## 今天学什么

```
LLM工作原理 → Token → 推理参数 → 本地模型(Ollama) → 云端API
```

---

## 环境

```bash
# 启动 Ollama 服务
ollama serve &

# 激活 Python 环境
conda activate py310

# 进入目录
cd "/Users/bobby/Desktop/Personal/AI project/day1_llm_basics"
```

---

## 代码文件

| 文件 | 内容 | 运行命令 |
|------|------|---------|
| `01_basic_inference.py` | 基础调用、流式输出、多轮对话 | `python 01_basic_inference.py` |
| `02_parameters.py` | temperature / top_p 实验 | `python 02_parameters.py` |
| `03_tokenization.py` | Token 可视化 | `python 03_tokenization.py` |
| `04_anthropic_sdk.py` | 云端 API（需要 ANTHROPIC_API_KEY） | `python 04_anthropic_sdk.py` |

---

## 核心概念

### LLM 的本质
**next-token prediction**：给定前文，预测下一个最可能的 token，不断重复。没有"理解"，只有概率。

### 完整推理流程
```
输入文字
  ↓ tokenizer.json（查词表，文字→token ID）
  ↓ embedding层（查model.safetensors，token ID→3072维向量）
  ↓ 加位置编码（告诉模型每个token在第几位）
  ↓ N层 Transformer Block（Self-Attention + FFN，反复提炼语义）
  ↓ 线性层（3072维→150,000维，每个token一个分数）
  ↓ softmax（变成概率分布）
  ↓ 采样（按temperature选下一个token）
  ↓ tokenizer反查（token ID→文字）
输出文字
```

### 三层存储
```
SSD（持久化）→ 读取到 → 内存/显存（运行）→ 断电清空
模型文件、checkpoint      权重、KV Cache、中间计算
```

### 文件结构（一个模型的 checkpoint）
```
qwen2.5-7b/
├── tokenizer.json       # 词表映射表（token↔ID），几MB，训练完不变
├── model.safetensors    # 模型权重（embedding+Attention+FFN），几GB
├── config.json          # 模型结构配置（层数、维度等）
└── tokenizer_config.json
```

### 关键参数
| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `temperature` | 采样随机性，0=确定，1+=随机 | 0（代码）/ 0.7（通用）|
| `top_p` | 候选token范围 | 0.9 |
| `max_tokens` | 最大输出长度 | 按需 |

### 多轮对话的本质
模型无状态，"记忆"靠每次传完整 history：
```python
history = [
    {"role": "system",    "content": "你是..."},
    {"role": "user",      "content": "我叫Bobby"},
    {"role": "assistant", "content": "你好Bobby"},
    {"role": "user",      "content": "我叫什么？"},  # 模型看到完整历史才能回答
]
```

### 推理框架
同一个模型权重，不同框架跑，能力一样，速度不同：
- **Ollama**：本地开发，一条命令，自动选硬件（Mac用Metal，NVIDIA用CUDA）
- **vLLM**：生产部署，PagedAttention，高并发
- **HuggingFace Transformers**：研究/训练，最灵活

---

## 今天的关键结论

1. LLM = next-token prediction，temperature 控制采样随机性
2. tokenizer.json = 词表映射表（文字↔ID），model.safetensors = 模型权重（含embedding）
3. 多轮对话靠传完整 history，模型本身无状态
4. 显存是瓶颈：装不下就跑不起来，CPU跑能跑但慢5~10倍
5. Mac M4 统一内存 = CPU/GPU共享，24GB全给模型用，能跑13B
6. Checkpoint = 训练中把显存权重存到SSD的快照，断点续训用

---

## 词汇表

详见 `glossary.md`，涵盖：架构（Transformer/Attention/KV Cache/MoE）、训练（预训练/SFT/LoRA/QLoRA）、推理（量化/采样/Prefill/Decode）、工程（Ollama/vLLM）、评估（Perplexity/MMLU）

---

## 明天预告：Day 2 — LoRA/QLoRA 微调

在远程服务器上，用 QLoRA 微调一个 7B 模型，让它学会特定风格或知识。
需要：SSH 连接远程服务器 + NVIDIA GPU（至少16GB显存）
