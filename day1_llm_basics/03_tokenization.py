"""
Day 1 - Part 3: Token 可视化
理解 tokenization 是理解 LLM 的基础
"""

# 需要安装: pip install tiktoken
# tiktoken 是 OpenAI 的 tokenizer，大多数模型用类似逻辑

try:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 用的编码

    def show_tokens(text: str):
        tokens = enc.encode(text)
        decoded = [enc.decode([t]) for t in tokens]
        print(f"文本: {text!r}")
        print(f"Token 数量: {len(tokens)}")
        print(f"Token 列表: {decoded}")
        print(f"Token IDs: {tokens}")
        print()

    print("=== Token 可视化 ===\n")
    show_tokens("Hello, world!")
    show_tokens("你好，世界！")
    show_tokens("def hello(): pass")
    show_tokens("ChatGPT is amazing!!!")

    print("""
关键结论：
- 英文单词通常 1 个词 = 1~2 个 token
- 中文通常 1 个字 = 1~2 个 token（比英文"贵"）
- 代码的 token 效率较高
- Context Window（上下文窗口）= 模型能处理的最大 token 数
  - qwen2.5:3b  → 32k tokens
  - GPT-4o      → 128k tokens
  - Claude 3.5  → 200k tokens
- 计费和速度都按 token 算，所以 prompt 要精简
""")

except ImportError:
    print("请先安装: pip install tiktoken")
    print("或者直接访问 https://platform.openai.com/tokenizer 可视化体验")
