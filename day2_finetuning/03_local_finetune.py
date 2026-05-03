"""
Day 2 — 本地 LoRA 微调（Mac M4 / Apple Silicon 版本）

使用 MPS 加速，不需要 NVIDIA GPU
适合在 Mac 上跑小模型（1.5B-3B）

运行：
    conda activate py310
    python 03_local_finetune.py
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace 模型 ID")
    parser.add_argument("--train_file", default="train_chatml.jsonl")
    parser.add_argument("--eval_file", default="eval_chatml.jsonl")
    parser.add_argument("--output_dir", default="./output_local")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
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

    print("=" * 60)
    print("本地 LoRA 微调（Apple Silicon MPS 加速）")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"设备: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print()

    # ─── 加载模型和 tokenizer ─────────────────────────────────────────────────
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # 移动到 MPS 设备
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.config.use_cache = False

    # ─── LoRA 配置 ────────────────────────────────────────────────────────────
    print("应用 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # 最小配置，省显存
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ─── 数据处理 ─────────────────────────────────────────────────────────────
    print("\n加载数据...")
    train_raw = load_jsonl(args.train_file)
    eval_raw = load_jsonl(args.eval_file)
    print(f"训练样本: {len(train_raw)}, 验证样本: {len(eval_raw)}")

    def tokenize(examples):
        texts = [format_prompt(item["messages"], tokenizer) for item in examples["data"]]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
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
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        use_cpu=False,  # 让 Trainer 使用 model 所在设备
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
    print("（Mac 上会比 GPU 慢，1.5B 模型预计 10-20 分钟）")
    trainer.train()

    # ─── 保存 ─────────────────────────────────────────────────────────────────
    output_path = Path(args.output_dir)
    model.save_pretrained(output_path / "lora_weights")
    tokenizer.save_pretrained(output_path / "lora_weights")
    print(f"\n✅ 训练完成，LoRA 权重保存到: {output_path / 'lora_weights'}")
    print(f"权重大小: {sum(f.stat().st_size for f in (output_path / 'lora_weights').rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
