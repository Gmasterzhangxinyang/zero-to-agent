# Day 2 — LoRA/QLoRA 微调

## 今天学什么

```
为什么微调 → LoRA原理 → QLoRA量化 → 数据准备 → 远程训练 → 推理验证
```

---

## 环境

### 本地（准备数据）
```bash
conda activate py310
cd "/Users/bobby/Desktop/Personal/AI project/day2_finetuning"
pip install datasets pandas
```

### 远程服务器（训练）
```bash
# SSH 连接到远程服务器
ssh user@your-gpu-server

# 安装依赖
pip install torch transformers peft datasets bitsandbytes accelerate
```

**硬件要求**：
- NVIDIA GPU，至少 16GB 显存（推荐 24GB+）
- QLoRA 可以在 16GB 显存上微调 7B 模型
- 如果只有 CPU，可以用更小的模型（1B-3B）但会很慢

---

## 代码文件

| 文件 | 内容 | 运行位置 |
|------|------|---------|
| `01_prepare_data.py` | 数据集准备和格式化 | 本地 |
| `02_lora_basics.py` | LoRA 原理演示 | 本地/远程 |
| `03_qlora_finetune.py` | QLoRA 微调脚本 | 远程服务器 |
| `04_inference.py` | 微调后模型推理 | 远程/本地 |
| `sample_data.jsonl` | 示例训练数据 | - |

---

## 核心概念

### 为什么需要微调？

预训练模型是"通才"，微调让它成为"专家"：

| 场景 | 预训练模型 | 微调后 |
|------|-----------|--------|
| 客服对话 | 通用回答 | 符合公司话术、产品知识 |
| 代码生成 | 通用代码 | 符合团队规范、内部框架 |
| 医疗问答 | 可能幻觉 | 基于专业文献、准确性高 |

### 全量微调 vs LoRA

```
全量微调（Full Fine-tuning）
├── 更新所有参数（7B模型 = 70亿参数）
├── 显存需求：~80GB（7B模型）
└── 训练时间：数小时到数天

LoRA（Low-Rank Adaptation）
├── 只训练小矩阵（~0.1%参数）
├── 显存需求：~24GB（7B模型）
└── 训练时间：数十分钟到数小时
```

### LoRA 原理

不修改原始权重 `W`，而是加一个低秩分解：

```
原始：h = W·x                    （W是7B参数）
LoRA：h = W·x + B·A·x            （B·A只有几百万参数）
      ↑固定   ↑可训练
```

**关键参数**：
- `r`（rank）：低秩矩阵的秩，越大能力越强但显存越多，推荐 8-64
- `alpha`：缩放因子，通常设为 `r` 的 2 倍
- `target_modules`：对哪些层应用 LoRA，通常是 `q_proj, v_proj`

### QLoRA = LoRA + 量化

```
QLoRA 的三个技巧：
1. 4-bit 量化：把 W 从 FP16 压缩到 4-bit（显存减少 75%）
2. NF4 数据类型：专为神经网络设计的 4-bit 格式
3. 双重量化：连量化参数本身也量化
```

**显存对比**（7B 模型）：
- 全量微调：~80GB
- LoRA（FP16）：~24GB
- QLoRA（4-bit）：~12GB ✅ 单卡 RTX 3090 可跑

### 训练数据格式

```jsonl
{"instruction": "用户问题", "input": "", "output": "期望回答"}
{"instruction": "用户问题", "input": "补充信息", "output": "期望回答"}
```

或 ChatML 格式：
```json
{
  "messages": [
    {"role": "system", "content": "你是..."},
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "回答"}
  ]
}
```

---

## 训练流程

### 1. 数据准备
```bash
python 01_prepare_data.py
# 生成 train.jsonl 和 eval.jsonl
```

### 2. 远程训练
```bash
# 在远程服务器上
python 03_qlora_finetune.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_file train.jsonl \
  --output_dir ./output \
  --num_epochs 3 \
  --batch_size 4
```

### 3. 推理验证
```bash
python 04_inference.py --model_path ./output
```

---

## 关键参数调优

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `learning_rate` | 学习率 | 1e-4 到 5e-4 |
| `num_epochs` | 训练轮数 | 3-5（数据少）/ 1-2（数据多）|
| `batch_size` | 批次大小 | 4-8（显存允许的最大值）|
| `lora_r` | LoRA 秩 | 8-64 |
| `lora_alpha` | LoRA 缩放 | `r * 2` |
| `max_seq_length` | 最大序列长度 | 512-2048 |

---

## 常见问题

### Q1: CUDA out of memory
```bash
# 减小 batch_size
--batch_size 2

# 启用梯度累积
--gradient_accumulation_steps 4

# 减小序列长度
--max_seq_length 512
```

### Q2: 训练不收敛
- 检查数据质量（是否有噪声、格式错误）
- 降低学习率（1e-5）
- 增加训练轮数

### Q3: 过拟合
- 减少训练轮数
- 增加数据量
- 使用更小的 `lora_r`

---

## 今天的关键结论

1. **LoRA 原理**：冻结原始权重，只训练低秩矩阵 `B·A`，显存需求降低 70%
2. **QLoRA = LoRA + 4-bit 量化**，12GB 显存可微调 7B 模型
3. **数据质量 > 数据数量**：100 条高质量数据胜过 1000 条噪声数据
4. **训练监控**：关注 `loss` 曲线，过拟合时及时停止
5. **LoRA 权重可合并**：训练完可以合并回原模型，或单独保存（几十 MB）

---

## 明天预告：Day 3 — RAG 基础

学习如何用向量数据库构建本地知识库，让 LLM 回答私有文档中的问题。
需要：Ollama + ChromaDB + Sentence Transformers
