"""SFT 训练脚本（TRL SFTTrainer）

SFT（Supervised Fine-Tuning）：
- 目标：让模型学会 ReAct 输出格式与基本的工具选择
- 数据：scripts/generate_sft_data.py 产出的 prompt-completion 会话式数据集
- 输出：SFT 模型（作为 DPO/GRPO 的起点）

用法：
    python -m rl.train_sft
    python -m rl.train_sft --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3

说明：默认基座沿用项目原定的 Qwen2.5-1.5B-Instruct。
无 GPU 的机器上可换成更小的模型跑通流程验证，例如：
    python -m rl.train_sft --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("STORE_BACKEND", "memory")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def pick_device_flags():
    """按硬件挑精度：只有 CUDA 才开 fp16，否则会直接报错。"""
    import torch
    if torch.cuda.is_available():
        return {"fp16": True}
    return {}


def main(
    model: str = DEFAULT_MODEL,
    train_data: str = "data/sft/sft_train.jsonl",
    output_dir: str = "outputs/sft",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 2,
    lr: float = 2e-5,
    max_length: int = 3072,
    max_samples: int = 0,
):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    data_path = REPO_ROOT / train_data
    if not data_path.exists():
        print(f"数据集不存在: {data_path}")
        print("请先运行: python scripts/generate_sft_data.py")
        return

    print(f"加载模型: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(model)

    print(f"加载数据集: {data_path}")
    train_dataset = load_dataset("json", data_files=str(data_path), split="train")
    if max_samples and max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_samples))
    print(f"  样本数: {len(train_dataset)}")

    # 截断守卫：ReAct prompt 很长（含完整工具 JSON Schema，实测 ~2300 tokens），
    # 而 completion 在序列末尾。max_length 设小了会把 assistant 目标整段截掉，
    # 训练照常跑完但模型什么都学不到——这种失败是完全静默的，必须显式检查。
    lengths = []
    for row in train_dataset:
        ids = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        lengths.append(len(ids))
    over = sum(1 for n in lengths if n > max_length)
    print(f"  token 长度: p50={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")
    if over:
        raise SystemExit(
            f"有 {over}/{len(lengths)} 条样本超过 max_length={max_length}，"
            f"会截掉 assistant 目标。请改用 --max_length {max(lengths) + 64}。"
        )

    out_dir = REPO_ROOT / output_dir
    config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_length=max_length,          # TRL>=0.20 用 max_length（旧版是 max_seq_length）
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        **pick_device_flags(),
    )

    print("开始 SFT 训练...")
    trainer = SFTTrainer(
        model=hf_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,     # TRL>=0.13 用 processing_class（旧版是 tokenizer）
    )
    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"SFT 完成，模型已保存: {final_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SFT 训练")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train_data", default="data/sft/sft_train.jsonl")
    p.add_argument("--output_dir", default="outputs/sft")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=3072)
    p.add_argument("--max_samples", type=int, default=0)
    a = p.parse_args()
    main(a.model, a.train_data, a.output_dir, a.epochs, a.batch_size,
         a.grad_accum, a.lr, a.max_length, a.max_samples)
