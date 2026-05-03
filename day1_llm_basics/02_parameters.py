"""
Day 1 - Part 2: 关键参数实验
理解 temperature、top_p、max_tokens 对输出的影响
"""

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:3b"
PROMPT = "给我一个创意公司名字，只输出名字本身"


def generate(prompt: str, **params) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": params},
    )
    return response.json()["response"].strip()


if __name__ == "__main__":
    # ── temperature ──────────────────────────────────────────────
    # 控制"随机性"：0 = 确定性最强，1+ = 越来越随机
    print("=== temperature 实验 ===")
    for temp in [0.0, 0.5, 1.0, 1.5]:
        results = [generate(PROMPT, temperature=temp) for _ in range(3)]
        print(f"temperature={temp}: {results}")

    # ── top_p (nucleus sampling) ─────────────────────────────────
    # 只从累积概率前 p% 的 token 里采样
    # 通常 temperature 和 top_p 二选一调，不要同时改
    print("\n=== top_p 实验 ===")
    for top_p in [0.1, 0.5, 0.9]:
        result = generate(PROMPT, temperature=1.0, top_p=top_p)
        print(f"top_p={top_p}: {result}")

    # ── max_tokens ───────────────────────────────────────────────
    # 控制最大输出长度（token 不等于字符，中文约 1.5~2 字/token）
    print("\n=== max_tokens 实验 ===")
    long_prompt = "详细解释一下机器学习的发展历史"
    for max_tok in [20, 100, 500]:
        result = generate(long_prompt, num_predict=max_tok)  # ollama 用 num_predict
        print(f"max_tokens={max_tok}: {result[:80]}...")

    # ── 关键结论 ─────────────────────────────────────────────────
    print("""
关键结论：
- temperature=0   → 每次输出相同，适合需要确定性的任务（代码、分类）
- temperature=0.7 → 平衡创意与稳定，大多数场景的默认值
- temperature>1   → 输出更随机，适合头脑风暴
- top_p=0.9       → 业界常用默认值
- max_tokens      → 按需设置，太小会截断，太大浪费算力
""")
