"""
Day 2 — LoRA 原理演示
用 numpy 手动实现 LoRA，理解低秩分解的本质
"""

import numpy as np


def demo_lora_math():
    """演示 LoRA 的数学原理"""
    print("=" * 50)
    print("LoRA 数学原理演示")
    print("=" * 50)

    # 模拟一个小的权重矩阵（真实模型里是 4096x4096）
    d_in, d_out = 8, 8
    r = 2  # rank，远小于 d_in 和 d_out

    # 原始权重（冻结，不训练）
    W = np.random.randn(d_out, d_in) * 0.1
    print(f"\n原始权重 W: shape={W.shape}, 参数量={W.size}")

    # LoRA 矩阵（可训练）
    # A: 初始化为随机小值
    # B: 初始化为全零（确保训练开始时 ΔW = B·A = 0）
    A = np.random.randn(r, d_in) * 0.01
    B = np.zeros((d_out, r))
    print(f"LoRA A: shape={A.shape}, 参数量={A.size}")
    print(f"LoRA B: shape={B.shape}, 参数量={B.size}")
    print(f"LoRA 总参数量: {A.size + B.size}（原始的 {(A.size + B.size) / W.size * 100:.1f}%）")

    # 前向传播
    x = np.random.randn(d_in)

    # 原始输出
    h_original = W @ x

    # LoRA 输出：原始 + 低秩增量
    alpha = 4  # 缩放因子
    delta_W = B @ A  # 低秩矩阵
    h_lora = W @ x + (alpha / r) * delta_W @ x

    print(f"\n输入 x: {x.round(3)}")
    print(f"原始输出 h = W·x: {h_original.round(3)}")
    print(f"LoRA 输出 h = W·x + (α/r)·B·A·x: {h_lora.round(3)}")
    print(f"（训练初始时 B=0，所以 ΔW=0，输出相同）")


def demo_rank_effect():
    """演示 rank 对表达能力的影响"""
    print("\n" + "=" * 50)
    print("Rank 对表达能力的影响")
    print("=" * 50)

    d = 64
    target = np.random.randn(d, d)  # 目标矩阵

    for r in [1, 2, 4, 8, 16, 32, 64]:
        A = np.random.randn(r, d)
        B = np.random.randn(d, r)
        approx = B @ A

        # 用 Frobenius 范数衡量近似误差
        error = np.linalg.norm(target - approx) / np.linalg.norm(target)
        params = A.size + B.size
        print(f"rank={r:2d}: 参数量={params:5d} ({params/d**2*100:4.1f}%), 近似误差={error:.3f}")


def demo_lora_merge():
    """演示训练完成后合并 LoRA 权重"""
    print("\n" + "=" * 50)
    print("LoRA 权重合并")
    print("=" * 50)

    d_in, d_out, r = 8, 8, 2
    alpha = 4

    W = np.random.randn(d_out, d_in) * 0.1
    A = np.random.randn(r, d_in) * 0.01
    B = np.random.randn(d_out, r) * 0.01  # 训练后 B 不再是全零

    x = np.random.randn(d_in)

    # 推理时：分开计算
    h_separate = W @ x + (alpha / r) * (B @ A) @ x

    # 合并后：等价但更快
    W_merged = W + (alpha / r) * B @ A
    h_merged = W_merged @ x

    print(f"分开计算: {h_separate.round(6)}")
    print(f"合并计算: {h_merged.round(6)}")
    print(f"结果相同: {np.allclose(h_separate, h_merged)}")
    print(f"\n合并的好处：推理时不需要额外计算，速度与原始模型相同")


def demo_target_modules():
    """演示 LoRA 应用在哪些层"""
    print("\n" + "=" * 50)
    print("LoRA 应用的目标层")
    print("=" * 50)

    # Transformer 中的注意力层
    modules = {
        "q_proj": "Query 投影（最常用）",
        "k_proj": "Key 投影",
        "v_proj": "Value 投影（最常用）",
        "o_proj": "输出投影",
        "gate_proj": "FFN 门控",
        "up_proj": "FFN 上投影",
        "down_proj": "FFN 下投影",
    }

    print("\n常见配置：")
    print("  最小（省显存）: q_proj, v_proj")
    print("  标准: q_proj, k_proj, v_proj, o_proj")
    print("  全量: 所有层（效果最好，显存最多）")

    print("\n各层说明：")
    for name, desc in modules.items():
        print(f"  {name:12s}: {desc}")


if __name__ == "__main__":
    demo_lora_math()
    demo_rank_effect()
    demo_lora_merge()
    demo_target_modules()

    print("\n" + "=" * 50)
    print("关键结论：")
    print("1. LoRA 用 B·A 近似 ΔW，参数量从 d² 降到 2·d·r")
    print("2. B 初始化为 0，保证训练开始时不破坏原始模型")
    print("3. rank 越大，表达能力越强，但显存越多")
    print("4. 训练完可以合并，推理无额外开销")
    print("=" * 50)
