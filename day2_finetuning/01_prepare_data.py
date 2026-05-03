"""
Day 2 - 01: 训练数据准备
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
]


def format_alpaca(item: dict) -> dict:
    """Alpaca 格式：instruction + input + output"""
    return {
        "instruction": item["instruction"],
        "input": item.get("input", ""),
        "output": item["output"],
    }


def format_chatml(item: dict) -> dict:
    """ChatML 格式：messages 列表"""
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

    # 格式化数据
    alpaca_data = [format_alpaca(d) for d in RAW_DATA]
    chatml_data = [format_chatml(d) for d in RAW_DATA]

    # 划分训练/验证集
    train_alpaca, eval_alpaca = split_dataset(alpaca_data)
    train_chatml, eval_chatml = split_dataset(chatml_data)

    # 保存
    save_jsonl(train_alpaca, output_dir / "train_alpaca.jsonl")
    save_jsonl(eval_alpaca,  output_dir / "eval_alpaca.jsonl")
    save_jsonl(train_chatml, output_dir / "train_chatml.jsonl")
    save_jsonl(eval_chatml,  output_dir / "eval_chatml.jsonl")

    # 预览
    print("\n--- 数据预览（Alpaca 格式）---")
    sample = train_alpaca[0]
    print(f"instruction: {sample['instruction']}")
    print(f"output: {sample['output'][:80]}...")

    print("\n--- 数据预览（ChatML 格式）---")
    sample = train_chatml[0]
    for msg in sample["messages"]:
        print(f"[{msg['role']}] {msg['content'][:60]}...")

    print(f"\n总计：{len(RAW_DATA)} 条，训练 {len(train_alpaca)} 条，验证 {len(eval_alpaca)} 条")


if __name__ == "__main__":
    main()
