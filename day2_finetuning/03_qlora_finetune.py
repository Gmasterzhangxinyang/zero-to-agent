"""
Day 2 — QLoRA 微调脚本（在远程 GPU 服务器上运行）

依赖：
    pip install torch transformers peft datasets bitsandbytes accelerate

运行：
    python 03_qlora_finetune.py
    python 03_qlora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --epochs 3
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace 模型 ID（小模型用于测试）")
    parser.add_argument("--train_file", default="train_chatml.jsonl")
    parser.add_argument("--eval_file", default="eval_chatml.jsonl")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="启用 4-bit 量化（QLoRA）")
    return parser.parse_args()


def load_jsonl(path: str) -> list:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_prompt(messages: list, tokenizer) -> str:
    """将 messages 格式化为模型输入"""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    args = parse_args()

    # ─── 导入（放在函数内，避免本地没装包时报错）─────────────────────────────
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"缺少依赖：{e}")
        print("请在 GPU 服务器上运行：pip install torch transformers peft datasets bitsandbytes accelerate")
        return

    print(f"模型: {args.model}")
    print(f"使用 4-bit 量化: {args.use_4bit}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")

    # ─── 量化配置（QLoRA 核心）────────────────────────────────────────────────
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4 格式，专为神经网络设计
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,      # 双重量化，进一步节省显存
        )

    # ─── 加载模型和 tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",          # 自动分配到可用 GPU
        trust_remote_code=True,
    )
    model.config.use_cache = False  # 训练时关闭 KV Cache

    # ─── LoRA 配置 ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力层
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 输出类似：trainable params: 6,815,744 || all params: 1,543,714,816 || trainable%: 0.44

    # ─── 数据处理 ─────────────────────────────────────────────────────────────
    train_raw = load_jsonl(args.train_file)
    eval_raw = load_jsonl(args.eval_file)

    def tokenize(examples):
        texts = [format_prompt(item["messages"], tokenizer) for item in examples["data"]]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
        # 只对 assistant 部分计算 loss（将 prompt 部分的 label 设为 -100）
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_dataset = Dataset.from_dict({"data": train_raw}).map(
        tokenize, batched=True, remove_columns=["data"]
    )
    eval_dataset = Dataset.from_dict({"data": eval_raw}).map(
        tokenize, batched=True, remove_columns=["data"]
    )

    # ─── 训练参数 ─────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,     # 等效 batch_size * 4
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,                          # 混合精度训练
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",                   # 不上报到 wandb
    )

    # ─── 训练 ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print("\n开始训练...")
    trainer.train()

    # ─── 保存 ─────────────────────────────────────────────────────────────────
    output_path = Path(args.output_dir)
    model.save_pretrained(output_path / "lora_weights")
    tokenizer.save_pretrained(output_path / "lora_weights")
    print(f"\n训练完成，LoRA 权重保存到: {output_path / 'lora_weights'}")


if __name__ == "__main__":
    main()
