"""
Day 1 - Part 1: 最基础的 LLM 推理
运行前提: ollama serve 已启动，且已 pull 了模型
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:3b"  # 可换成你 pull 的任意模型


# ── 1. 最简单的调用 ──────────────────────────────────────────────
def simple_chat(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
    )
    return response.json()["response"]


# ── 2. 流式输出（streaming）─────────────────────────────────────
def stream_chat(prompt: str):
    """逐 token 打印，体感更好"""
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": True},
        stream=True,
    ) as resp:
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                print(chunk["response"], end="", flush=True)
                if chunk.get("done"):
                    break
    print()


# ── 3. 带 system prompt 的对话格式 ──────────────────────────────
def chat_with_system(system: str, user_message: str) -> str:
    """
    messages 格式是所有主流模型的标准接口
    role 只有三种: system / user / assistant
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        },
    )
    return response.json()["message"]["content"]


# ── 4. 多轮对话（维护 history）───────────────────────────────────
def multi_turn_demo():
    history = []

    def chat(user_input: str) -> str:
        history.append({"role": "user", "content": user_input})
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "messages": history, "stream": False},
        )
        reply = response.json()["message"]["content"]
        history.append({"role": "assistant", "content": reply})
        return reply

    print("=== 多轮对话 ===")
    print("User: 我叫 Bobby")
    print("AI:", chat("我叫 Bobby"))
    print("User: 我叫什么名字？")
    print("AI:", chat("我叫什么名字？"))  # 模型应该记得


if __name__ == "__main__":
    print("=== 1. 简单调用 ===")
    print(simple_chat("用一句话解释什么是神经网络"))

    print("\n=== 2. 流式输出 ===")
    stream_chat("给我讲个笑话")

    print("\n=== 3. System Prompt ===")
    reply = chat_with_system(
        system="你是一个只会用中文回答的 Python 专家，回答要简洁",
        user_message="什么是装饰器？",
    )
    print(reply)

    print("\n=== 4. 多轮对话 ===")
    multi_turn_demo()
