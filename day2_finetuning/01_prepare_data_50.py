"""
Day 2 - 01: 训练数据准备（50条扩充版）
演示如何准备和格式化微调数据集
"""

import json
import random
from pathlib import Path


# ─── 示例数据：让模型学会"简洁技术风格"回答 ───────────────────────────────────

RAW_DATA = [
    {
        "instruction": "解释什么是梯度下降",
        "output": "梯度下降是优化算法的核心。计算损失函数对每个参数的偏导数（梯度），然后沿梯度反方向更新参数：θ = θ - lr × ∇L。重复直到收敛。学习率 lr 控制步长：太大震荡，太小收敛慢。"
    },
    {
        "instruction": "什么是 Batch Normalization？",
        "output": "BatchNorm 在每个 mini-batch 内对激活值做归一化（均值0、方差1），然后用可学习的 γ、β 缩放平移。作用：缓解梯度消失/爆炸、允许更大学习率、有轻微正则化效果。通常加在线性层之后、激活函数之前。"
    },
    {
        "instruction": "Dropout 的原理是什么？",
        "output": "训练时随机将部分神经元输出置零（概率 p，通常 0.1-0.5），强迫网络不依赖特定神经元，减少过拟合。推理时关闭 Dropout，但将权重乘以 (1-p) 保持期望值一致。本质是对指数级子网络做集成。"
    },
    {
        "instruction": "解释 Attention 机制",
        "output": "Attention 让模型动态决定关注哪些位置。核心公式：Attention(Q,K,V) = softmax(QK^T / √d_k) × V。Q（查询）和 K（键）计算相似度，softmax 归一化为权重，加权聚合 V（值）。Self-Attention 中 Q=K=V 来自同一序列，捕捉序列内部依赖。"
    },
    {
        "instruction": "什么是 KV Cache？",
        "output": "推理时，每个 token 的 K、V 矩阵只需计算一次，后续 token 生成时直接复用缓存。避免重复计算，将推理复杂度从 O(n²) 降到 O(n)。代价是显存占用随序列长度线性增长，这是长上下文推理的主要瓶颈。"
    },
    {
        "instruction": "LoRA 和全量微调有什么区别？",
        "output": "全量微调更新所有参数（7B模型≈70亿），显存需求约80GB。LoRA 冻结原始权重 W，只训练低秩矩阵 B·A（参数量约0.1%），显存降到24GB。推理时 h = Wx + BAx，可合并为 W' = W + BA，无额外延迟。"
    },
    {
        "instruction": "解释 Transformer 的位置编码",
        "output": "Transformer 本身无序列顺序感知，位置编码（PE）注入位置信息。原始论文用正弦/余弦函数：PE(pos,2i) = sin(pos/10000^(2i/d))。现代模型多用 RoPE（旋转位置编码），将位置信息编码进 Q、K 的旋转变换中，支持外推到训练时未见过的更长序列。"
    },
    {
        "instruction": "什么是混合精度训练？",
        "output": "用 FP16（半精度）做前向/反向传播，FP32（单精度）维护主权重副本。好处：显存减半、计算加速（Tensor Core 对 FP16 有硬件加速）。风险：FP16 范围小易溢出，用 loss scaling 解决（放大 loss 再缩小梯度）。PyTorch 的 autocast + GradScaler 自动处理。"
    },
    {
        "instruction": "解释 SFT（监督微调）的流程",
        "output": "SFT 是在预训练模型上用（指令, 回答）对做有监督训练。流程：1) 准备高质量指令数据集；2) 格式化为模型输入模板；3) 只对 assistant 部分计算 loss（忽略 prompt 的 loss）；4) 用较小学习率（1e-5 到 5e-4）训练 1-3 个 epoch。目的是让模型学会遵循指令格式。"
    },
    {
        "instruction": "什么是 RLHF？",
        "output": "RLHF（人类反馈强化学习）分三步：1) SFT 得到初始模型；2) 收集人类偏好数据，训练奖励模型（RM）；3) 用 PPO 算法，以 RM 分数为奖励优化语言模型，同时加 KL 散度惩罚防止偏离太远。ChatGPT 和 Claude 都用了 RLHF 或其变体（如 DPO）。"
    },
    {
        "instruction": "什么是 Transformer？",
        "output": "Transformer 是基于 Self-Attention 的序列模型，完全抛弃了 RNN 的递归结构。核心组件：Multi-Head Attention（多头注意力）+ FFN（前馈网络）+ 残差连接 + LayerNorm。并行计算所有位置，训练速度远快于 RNN。GPT、BERT、Qwen 都是 Transformer 变体。"
    },
    {
        "instruction": "解释残差连接（Residual Connection）",
        "output": "残差连接将层的输入直接加到输出：output = F(x) + x。作用：缓解深层网络的梯度消失问题，让梯度可以直接流过跳跃连接。即使 F(x) 学到的是零，网络也能保持恒等映射，训练更稳定。ResNet 和 Transformer 都依赖残差连接。"
    },
    {
        "instruction": "什么是 Layer Normalization？",
        "output": "LayerNorm 对单个样本的所有特征维度做归一化（均值0、方差1），再用可学习的 γ、β 缩放平移。与 BatchNorm 的区别：BatchNorm 跨样本归一化，LayerNorm 跨特征归一化，不依赖 batch size，适合序列模型。Transformer 每个子层后都有 LayerNorm。"
    },
    {
        "instruction": "什么是 Adam 优化器？",
        "output": "Adam 结合了 Momentum（动量）和 RMSprop（自适应学习率）。维护两个移动平均：一阶矩 m（梯度均值）和二阶矩 v（梯度平方均值）。更新公式：θ = θ - lr × m / (√v + ε)。优点：自适应学习率、收敛快、对超参数不敏感。深度学习默认优化器。"
    },
    {
        "instruction": "解释 Cross Entropy Loss",
        "output": "交叉熵衡量预测分布和真实分布的差异。公式：L = -Σ y_i log(p_i)，y_i 是真实标签（one-hot），p_i 是预测概率。分类任务的标准 loss，等价于最大化正确类别的对数似然。数值稳定实现：先算 logits 的 log_softmax，再取负。"
    },
    {
        "instruction": "什么是 Embedding？",
        "output": "Embedding 将离散符号（token ID）映射到连续向量空间。本质是查表：embedding_matrix[token_id] → d 维向量。预训练时学习，语义相近的词向量距离近。Transformer 第一层就是 Embedding，将 token ID 转成模型能处理的向量表示。"
    },
    {
        "instruction": "解释 Softmax 函数",
        "output": "Softmax 将任意实数向量转成概率分布。公式：softmax(x_i) = exp(x_i) / Σ exp(x_j)。保证输出非负且和为 1。分类任务最后一层用 Softmax，将 logits 转成类别概率。数值稳定技巧：先减去 max(x) 防止 exp 溢出。"
    },
    {
        "instruction": "什么是 Tokenization？",
        "output": "Tokenization 将文本切分成模型能处理的最小单元（token）。常见方法：BPE（字节对编码）、WordPiece、SentencePiece。平衡词汇表大小和表达能力：高频词单独成 token，低频词拆成子词。中文通常按字或子词切分。GPT 用 BPE，词汇表约 5 万。"
    },
    {
        "instruction": "解释 Temperature 参数",
        "output": "Temperature 控制采样随机性。公式：p_i = exp(logits_i / T) / Σ exp(logits_j / T)。T=1 保持原始分布，T→0 趋向贪心（选概率最大的），T>1 分布更平滑（更随机）。代码生成用 T=0，创意写作用 T=0.7-1.0。"
    },
    {
        "instruction": "什么是 Top-p 采样？",
        "output": "Top-p（nucleus sampling）动态选择候选 token。累积概率达到 p 时截断，只从这些 token 中采样。与 Top-k 的区别：Top-k 固定候选数量，Top-p 根据分布自适应。p=0.9 是常用值，平衡多样性和质量。比纯随机采样更可控。"
    },
    {
        "instruction": "解释 Beam Search",
        "output": "Beam Search 是贪心搜索的扩展，保留 k 个最优候选（beam）。每步扩展所有候选，保留总概率最高的 k 个。优点：比贪心搜索质量高。缺点：输出趋向通用、缺乏多样性，且计算量是贪心的 k 倍。机器翻译常用，对话生成少用。"
    },
    {
        "instruction": "什么是 Perplexity？",
        "output": "Perplexity（困惑度）衡量语言模型的预测能力。公式：PPL = exp(-1/N Σ log P(w_i))，即平均负对数似然的指数。PPL 越低，模型越确定。可理解为模型平均在多少个词中犹豫。评估语言模型的标准指标。"
    },
    {
        "instruction": "解释 Gradient Clipping",
        "output": "梯度裁剪防止梯度爆炸。方法：如果梯度范数超过阈值，按比例缩放。公式：g = g × threshold / ||g|| if ||g|| > threshold。RNN 训练必备，Transformer 也常用。PyTorch 的 clip_grad_norm_ 自动处理。阈值通常设为 1.0 或 5.0。"
    },
    {
        "instruction": "什么是 Learning Rate Warmup？",
        "output": "Warmup 让学习率从 0 线性增长到目标值。训练初期模型不稳定，大学习率会震荡。Warmup 给模型适应期，通常占总步数的 5%-10%。Transformer 训练标配，公式：lr = lr_max × min(step / warmup_steps, 1)。"
    },
    {
        "instruction": "解释 Weight Decay",
        "output": "Weight Decay 是 L2 正则化的实现，惩罚大权重。更新公式：θ = θ - lr × (∇L + λθ)，λ 是衰减系数（通常 0.01）。作用：防止过拟合、让权重趋向 0。AdamW 将 weight decay 从梯度中解耦，效果更好。"
    },
    {
        "instruction": "什么是 Multi-Head Attention？",
        "output": "Multi-Head Attention 并行运行多个 Attention，每个 head 学习不同的关注模式。公式：MultiHead(Q,K,V) = Concat(head_1,...,head_h) × W_O，每个 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)。多头让模型同时关注不同位置和不同语义维度。"
    },
    {
        "instruction": "解释 Feed Forward Network（FFN）",
        "output": "Transformer 中每个 Attention 层后跟一个 FFN：FFN(x) = max(0, xW_1 + b_1)W_2 + b_2。两层线性变换，中间用 ReLU 或 GELU 激活。维度先扩大 4 倍再压缩回来（如 4096→16384→4096）。作用：引入非线性，增强表达能力。"
    },
    {
        "instruction": "什么是 Causal Mask？",
        "output": "Causal Mask（因果掩码）让每个 token 只能看到自己和之前的 token，不能看未来。实现：Attention 矩阵的上三角设为 -∞，softmax 后变为 0。保证自回归生成的合法性：预测第 i 个 token 时只用前 i-1 个。GPT 系列必须用，BERT 不用。"
    },
    {
        "instruction": "解释 Quantization（量化）",
        "output": "量化将浮点权重压缩为低精度整数。FP16→INT8 显存减半，FP16→INT4 减少 75%。方法：找到权重的范围 [min, max]，线性映射到整数区间。代价：精度损失，通常 INT8 几乎无损，INT4 有轻微下降。QLoRA 用 NF4（4-bit 非均匀量化）。"
    },
    {
        "instruction": "什么是 Prefill 和 Decode？",
        "output": "推理分两阶段。Prefill：并行处理输入 prompt，计算所有 token 的 KV Cache，速度快。Decode：逐个生成输出 token，每步只计算一个新 token，速度慢（受显存带宽限制）。长 prompt 的瓶颈在 Prefill，长输出的瓶颈在 Decode。"
    },
    {
        "instruction": "解释 Flash Attention",
        "output": "Flash Attention 重新设计 Attention 的计算顺序，减少 HBM（显存）读写次数。标准 Attention 需要存储 N×N 的注意力矩阵，Flash Attention 分块计算，显存从 O(N²) 降到 O(N)。速度提升 2-4 倍，是长上下文训练的关键优化。"
    },
    {
        "instruction": "什么是 MoE（Mixture of Experts）？",
        "output": "MoE 用多个专家网络（FFN）替换单个 FFN，每个 token 只激活部分专家（如 8 个中选 2 个）。Router 网络决定选哪些专家。优点：参数多但计算量不变，提升模型容量。缺点：训练不稳定、负载不均衡。Mixtral、DeepSeek 用 MoE。"
    },
    {
        "instruction": "解释 RoPE（旋转位置编码）",
        "output": "RoPE 将位置信息编码进 Q、K 的旋转变换中。公式：q_m = R_m q, k_n = R_n k，R 是旋转矩阵。相对位置 m-n 自然体现在内积 q_m^T k_n 中。优点：支持外推到更长序列、计算高效。Llama、Qwen 都用 RoPE。"
    },
    {
        "instruction": "什么是 GQA（Grouped Query Attention）？",
        "output": "GQA 是 MHA 和 MQA 的折中。MHA 每个 head 独立的 K、V，MQA 所有 head 共享 K、V，GQA 将 head 分组共享。如 32 个 head 分 8 组，每组共享 K、V。平衡推理速度和质量，Llama 3 用 GQA。"
    },
    {
        "instruction": "解释 Sliding Window Attention",
        "output": "Sliding Window Attention 限制每个 token 只关注窗口内的 token（如前后各 2048）。复杂度从 O(N²) 降到 O(N×W)，W 是窗口大小。代价：长距离依赖需要多层传递。Mistral 用 4096 窗口，配合 RoPE 外推支持 32k 上下文。"
    },
    {
        "instruction": "什么是 Speculative Decoding？",
        "output": "Speculative Decoding 用小模型快速生成候选 token，大模型并行验证。小模型生成 k 个 token，大模型一次前向验证全部，接受正确的、拒绝错误的。加速比取决于小模型准确率，理想情况 2-3 倍。无损加速技术。"
    },
    {
        "instruction": "解释 Prompt Engineering",
        "output": "Prompt Engineering 是设计输入提示词让模型输出符合预期。技巧：Few-shot（给示例）、Chain-of-Thought（让模型逐步推理）、Role Prompting（设定角色）。好 prompt 能显著提升效果，是使用 LLM 的核心技能。"
    },
    {
        "instruction": "什么是 RAG？",
        "output": "RAG（检索增强生成）在生成前先检索相关文档。流程：1) 将知识库向量化存入数据库；2) 用户提问时检索最相关的文档；3) 将文档和问题一起送给 LLM 生成回答。解决 LLM 知识截止和幻觉问题，无需微调即可注入私有知识。"
    },
    {
        "instruction": "解释 Vector Database",
        "output": "向量数据库存储和检索高维向量。核心操作：给定查询向量，找最近邻（ANN）。常用算法：HNSW（分层图）、IVF（倒排索引）。代表产品：Chroma、Pinecone、Weaviate、Milvus。RAG 系统的核心组件，存储文档的 Embedding。"
    },
    {
        "instruction": "什么是 Embedding 模型？",
        "output": "Embedding 模型将文本转成固定维度的向量，语义相近的文本向量距离近。常用模型：text-embedding-ada-002（OpenAI）、BGE（BAAI）、E5。评估指标：MTEB 榜单。RAG 系统用 Embedding 模型将文档和查询向量化，再做相似度检索。"
    },
    {
        "instruction": "解释 Cosine Similarity",
        "output": "余弦相似度衡量两个向量的方向相似性。公式：cos(θ) = (A·B) / (||A|| × ||B||)，范围 [-1, 1]。1 表示完全相同方向，0 表示正交，-1 表示相反。向量检索的标准度量，不受向量长度影响，只看方向。"
    },
    {
        "instruction": "什么是 Chain-of-Thought？",
        "output": "Chain-of-Thought（CoT）让模型在给出答案前先逐步推理。方法：在 prompt 中加入推理示例，或直接说'让我们一步步思考'。效果：显著提升复杂推理任务（数学、逻辑）的准确率。原理：中间步骤作为草稿纸，减少一步到位的错误。"
    },
    {
        "instruction": "解释 Function Calling",
        "output": "Function Calling 让 LLM 调用外部工具。流程：1) 定义工具的 JSON Schema；2) LLM 决定调用哪个工具、传什么参数；3) 执行工具获取结果；4) LLM 基于结果生成最终回答。Agent 的核心能力，让 LLM 能查天气、搜索、执行代码。"
    },
    {
        "instruction": "什么是 ReAct 框架？",
        "output": "ReAct 交替进行推理（Reasoning）和行动（Acting）。流程：Thought（分析当前状态）→ Action（调用工具）→ Observation（获取结果）→ 循环直到完成任务。优点：推理过程可解释，错误可追溯。LangChain Agent 的默认框架。"
    },
    {
        "instruction": "解释 Context Window",
        "output": "Context Window 是模型一次能处理的最大 token 数。超出则截断或报错。GPT-4 支持 128k，Claude 支持 200k。长上下文的挑战：显存随长度平方增长（KV Cache），注意力计算复杂度高。实际使用中，过长的上下文会导致'中间遗忘'问题。"
    },
    {
        "instruction": "什么是 System Prompt？",
        "output": "System Prompt 是对话开始前设定模型角色和行为的指令。放在 messages 的第一条，role 为 system。作用：设定人格、限制话题、指定输出格式、注入背景知识。模型会在整个对话中遵循 system prompt 的约束。"
    },
    {
        "instruction": "解释 DPO（Direct Preference Optimization）",
        "output": "DPO 是 RLHF 的简化替代，直接用偏好数据优化模型，不需要训练奖励模型。数据格式：(prompt, chosen, rejected) 三元组。损失函数：最大化 chosen 和 rejected 的对数概率差。比 PPO 稳定、简单，效果相当。Llama 3 用 DPO。"
    },
    {
        "instruction": "什么是 Instruction Tuning？",
        "output": "Instruction Tuning 用多样化的指令数据微调预训练模型，让它能遵循各种指令。数据覆盖：问答、摘要、翻译、代码、推理等任务。效果：泛化到未见过的指令类型。FLAN、InstructGPT 是代表工作。本质是 SFT，但数据更多样。"
    },
    {
        "instruction": "解释 Hallucination（幻觉）",
        "output": "幻觉是 LLM 生成看似合理但实际错误的内容。原因：训练目标是预测下一个 token，不是保证事实正确；训练数据有噪声。缓解方法：RAG（检索真实文档）、RLHF（人类反馈惩罚错误）、温度设低（减少随机性）。幻觉是 LLM 的根本局限之一。"
    },
]


def format_alpaca(item: dict) -> dict:
    return {
        "instruction": item["instruction"],
        "input": item.get("input", ""),
        "output": item["output"],
    }


def format_chatml(item: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]},
        ]
    }


def split_dataset(data: list, eval_ratio: float = 0.1) -> tuple[list, list]:
    random.seed(42)
    shuffled = data.copy()
    random.shuffle(shuffled)
    split = max(1, int(len(shuffled) * eval_ratio))
    return shuffled[split:], shuffled[:split]


def save_jsonl(data: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples → {path}")


def main():
    output_dir = Path(__file__).parent

    alpaca_data = [format_alpaca(d) for d in RAW_DATA]
    chatml_data = [format_chatml(d) for d in RAW_DATA]

    train_alpaca, eval_alpaca = split_dataset(alpaca_data)
    train_chatml, eval_chatml = split_dataset(chatml_data)

    save_jsonl(train_alpaca, output_dir / "train_alpaca.jsonl")
    save_jsonl(eval_alpaca,  output_dir / "eval_alpaca.jsonl")
    save_jsonl(train_chatml, output_dir / "train_chatml.jsonl")
    save_jsonl(eval_chatml,  output_dir / "eval_chatml.jsonl")

    print(f"\n总计：{len(RAW_DATA)} 条，训练 {len(train_chatml)} 条，验证 {len(eval_chatml)} 条")


if __name__ == "__main__":
    main()
