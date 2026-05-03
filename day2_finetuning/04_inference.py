"""
Day 2 — 微调后模型推理
对比基础模型和微调后模型的输出差异
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora_path", default="./output/lora_weights",
                        help="LoRA 权重路径（03_qlora_finetune.py 的输出）")
    parser.add_argument("--merge", action="store_true",
                        help="将 LoRA 权重合并到基础模型")
    return parser.parse_args()


TEST_QUESTIONS = [
    "解释什么是梯度下降",
    "LoRA 和全量微调有什么区别？",
    "什么是 KV Cache？",
]


def generate(model, tokenizer, question: str, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只返回新生成的部分
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"缺少依赖：{e}")
        return

    # ─── 加载基础模型 ─────────────────────────────────────────────────────────
    print(f"加载基础模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ─── 加载 LoRA 权重 ───────────────────────────────────────────────────────
    print(f"加载 LoRA 权重: {args.lora_path}")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_path)

    if args.merge:
        print("合并 LoRA 权重到基础模型...")
        lora_model = lora_model.merge_and_unload()
        print("合并完成，推理速度与原始模型相同")

    # ─── 对比推理 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("基础模型 vs 微调模型 对比")
    print("=" * 60)

    for question in TEST_QUESTIONS:
        print(f"\n问题: {question}")
        print("-" * 40)

        base_answer = generate(base_model, tokenizer, question)
        print(f"[基础模型]\n{base_answer}")

        lora_answer = generate(lora_model, tokenizer, question)
        print(f"\n[微调模型]\n{lora_answer}")
        print("-" * 40)


if __name__ == "__main__":
    main()
