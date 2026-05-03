"""
Day 1 - Part 4: 用 Anthropic SDK 调用 Claude（云端 API）
对比本地 Ollama 和云端 API 的接口差异
"""

# pip install anthropic
import anthropic
import os

# API Key 从环境变量读取，不要硬编码！
# export ANTHROPIC_API_KEY="sk-ant-..."
client = anthropic.Anthropic()  # 自动读取 ANTHROPIC_API_KEY


# ── 1. 基础调用 ──────────────────────────────────────────────────
def basic_call():
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "用一句话解释什么是 Transformer"}],
    )
    return message.content[0].text


# ── 2. 带 system prompt ──────────────────────────────────────────
def with_system():
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="你是一个简洁的技术导师，每次回答不超过 3 句话",
        messages=[{"role": "user", "content": "什么是 attention mechanism？"}],
    )
    return message.content[0].text


# ── 3. 流式输出 ──────────────────────────────────────────────────
def streaming_call():
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": "给我讲一个关于程序员的笑话"}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()


# ── 4. 查看 token 用量 ───────────────────────────────────────────
def check_usage():
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "你好"}],
    )
    print(f"输入 tokens: {message.usage.input_tokens}")
    print(f"输出 tokens: {message.usage.output_tokens}")


if __name__ == "__main__":
    print("=== 1. 基础调用 ===")
    print(basic_call())

    print("\n=== 2. System Prompt ===")
    print(with_system())

    print("\n=== 3. 流式输出 ===")
    streaming_call()

    print("\n=== 4. Token 用量 ===")
    check_usage()
