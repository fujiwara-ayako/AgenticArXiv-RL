"""DPO 训练脚本（TRL DPOTrainer）

DPO（Direct Preference Optimization）：
- 目标：在同一状态下偏好"正确的工具调用"，拒绝"乱码 / 选错工具 / 提前结束"
- 数据：scripts/generate_dpo_data.py 产出的 (prompt, chosen, rejected)
- 输出：DPO 模型（作为 GRPO 的起点）

用法：
    python -m rl.train_dpo
    python -m rl.train_dpo --model outputs/sft/final --epochs 2
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("STORE_BACKEND", "memory")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def pick_device_flags():
    import torch
    if torch.cuda.is_available():
        return {"fp16": True}
    return {}


def main(
    model: str = "outputs/sft/final",
    train_data: str = "data/dpo/dpo_train.jsonl",
    output_dir: str = "outputs/dpo",
    epochs: int = 2,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 5e-6,
    beta: float = 0.1,
    max_length: int = 3072,
    max_samples: int = 0,
):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    model_path = REPO_ROOT / model
    if model_path.exists():
        resolved = str(model_path)
    elif "/" in model and not model.startswith(("outputs/", "./", "/")):
        resolved = model                       # 形如 org/name 的 HF 仓库
    else:
        raise SystemExit(
            f"未找到本地模型 {model_path}。\n"
            f"（若上一阶段训练失败，这里会把路径当成 HF repo id，报出令人误解的 "
            f"'Repo id must be in the form ...'，因此这里提前拦下。）\n"
            f"请先运行: python -m rl.train_sft"
        )

    data_path = REPO_ROOT / train_data
    if not data_path.exists():
        print(f"数据集不存在: {data_path}")
        print("请先运行: python scripts/generate_dpo_data.py")
        return

    print(f"加载模型: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(resolved)
    ref_model = AutoModelForCausalLM.from_pretrained(resolved)

    print(f"加载数据集: {data_path}")
    train_dataset = load_dataset("json", data_files=str(data_path), split="train")
    if max_samples and max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_samples))
    print(f"  偏好对数: {len(train_dataset)}")

    # 截断守卫：DPOConfig 的 max_length 默认只有 1024，而本项目的 ReAct prompt
    # 实测 ~2300 tokens；truncation_mode 默认 keep_start，会保留 prompt、切掉
    # chosen/rejected 的回答部分，导致偏好信号消失且不报任何错。
    lengths = []
    for row in train_dataset:
        for side in ("chosen", "rejected"):
            ids = tokenizer.apply_chat_template(row["prompt"] + row[side], tokenize=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            lengths.append(len(ids))
    over = sum(1 for n in lengths if n > max_length)
    print(f"  token 长度: p50={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")
    if over:
        raise SystemExit(
            f"有 {over}/{len(lengths)} 条序列超过 max_length={max_length}，"
            f"keep_start 截断会切掉回答部分。请改用 --max_length {max(lengths) + 64}。"
        )

    out_dir = REPO_ROOT / output_dir
    config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        max_length=max_length,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        # DPO 要同时前向 policy/ref × chosen/rejected 四路，logits 是
        # 序列长 × 词表(49152)，2900 tokens 下单张 logits 就 ~570MB。
        # batch_size=2 实测在 30GB MPS 上 OOM，因此默认 batch=1 + 梯度检查点。
        gradient_checkpointing=True,
        **pick_device_flags(),
    )

    print("开始 DPO 训练...")
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"DPO 完成，模型已保存: {final_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DPO 训练")
    p.add_argument("--model", default="outputs/sft/final")
    p.add_argument("--train_data", default="data/dpo/dpo_train.jsonl")
    p.add_argument("--output_dir", default="outputs/dpo")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=3072)
    p.add_argument("--max_samples", type=int, default=0)
    a = p.parse_args()
    main(a.model, a.train_data, a.output_dir, a.epochs, a.batch_size,
         a.grad_accum, a.lr, a.beta, a.max_length, a.max_samples)
