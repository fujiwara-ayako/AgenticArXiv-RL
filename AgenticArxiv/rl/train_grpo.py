"""GRPO 训练脚本（TRL GRPOTrainer）

GRPO（Group Relative Policy Optimization）：
- 目标：用 verifiable reward 在线训练，无需 reward model、无需 value model
- 数据：scripts/generate_grpo_data.py 产出的 (prompt, gold)
- 奖励：rl.reward.grpo_reward_func —— 规则判定，RLVR

用法：
    python -m rl.train_grpo
    python -m rl.train_grpo --model outputs/dpo/final --num_generations 4

与旧版的区别
------------
旧版 reward_fn 是 `return [0.0 for _ in responses]` 的占位实现，
且 GRPOTrainer 整段被注释掉，运行只会打印一行 TODO 提示；
配置项 `model_name` / `num_sample_generations` 也不是 TRL GRPOConfig 的字段。
现在奖励函数已实现（见 rl/reward.py 的 compute_action_reward），训练可真实执行。
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
    model: str = "outputs/dpo/final",
    train_data: str = "data/grpo/grpo_train.jsonl",
    output_dir: str = "outputs/grpo",
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 2,
    lr: float = 1e-5,
    beta: float = 0.04,
    num_generations: int = 4,
    max_prompt_length: int = 768,
    max_completion_length: int = 96,
    max_samples: int = 0,
):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from rl.reward import grpo_reward_func

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
            f"请先运行 train_sft / train_dpo，或用 --model 指定别的检查点"
        )

    data_path = REPO_ROOT / train_data
    if not data_path.exists():
        print(f"数据集不存在: {data_path}")
        print("请先运行: python scripts/generate_grpo_data.py")
        return

    print(f"加载模型: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(resolved)

    print(f"加载数据集: {data_path}")
    train_dataset = load_dataset("json", data_files=str(data_path), split="train")
    if max_samples and max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_samples))
    print(f"  样本数: {len(train_dataset)}")

    # GRPO 要求 batch_size 能被 num_generations 整除
    if batch_size % num_generations != 0:
        batch_size = num_generations
        print(f"  调整 per_device_train_batch_size = {batch_size}（需被 num_generations 整除）")

    # 生成长度守卫：max_completion_length 若短于标准动作，模型永远吐不出完整 JSON，
    # 奖励恒为解析失败的下限 → reward_std=0 → 组内优势恒为 0 → 完全没有梯度。
    # 这种失败在日志里表现为 completions/clipped_ratio=1、frac_reward_zero_std=1，
    # 很容易被误读成"模型太差"，其实是配置问题，所以这里显式检查。
    import json as _json
    gold_lens = []
    for row in train_dataset:
        g = row.get("gold")
        if not g:
            continue
        try:
            parsed = _json.loads(g)
        except (_json.JSONDecodeError, TypeError):
            continue
        if parsed == "FINISH":
            text = "Thought: 任务已完成\nAction: FINISH"
        else:
            text = f"Thought: xxx\nAction: {_json.dumps(parsed, ensure_ascii=False)}"
        gold_lens.append(len(tokenizer(text)["input_ids"]))
    if gold_lens:
        need = max(gold_lens)
        print(f"  标准动作 token 长度: p50={sorted(gold_lens)[len(gold_lens)//2]} max={need}")
        if max_completion_length < need:
            raise SystemExit(
                f"max_completion_length={max_completion_length} 小于标准动作最大长度 {need}，"
                f"模型不可能生成出完整动作，奖励会恒为下限、梯度为 0。"
                f"请改用 --max_completion_length {need + 32}。"
            )

    out_dir = REPO_ROOT / output_dir
    cfg_kwargs = {
        "output_dir": str(out_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "beta": beta,
        "num_generations": num_generations,       # 旧版误写成 num_sample_generations
        "max_prompt_length": max_prompt_length,   # 部分 TRL 版本没有这个字段
        "max_completion_length": max_completion_length,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        **pick_device_flags(),
    }

    # TRL 的 GRPOConfig 字段在各版本间变动较大（例如 0.29 已无 max_prompt_length）。
    # 这里按实际安装版本过滤，避免因为一个参数名不存在就整个训练起不来。
    import dataclasses
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in cfg_kwargs if k not in valid)
    if dropped:
        print(f"  提示：当前 TRL 不支持这些 GRPOConfig 参数，已忽略 -> {dropped}")
    config = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in valid})

    print(f"开始 GRPO 训练... (每个 prompt 采样 {num_generations} 条，规则奖励)")
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=grpo_reward_func,            # 旧版参数名 reward_model 已不存在
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"GRPO 完成，模型已保存: {final_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GRPO 训练")
    p.add_argument("--model", default="outputs/dpo/final")
    p.add_argument("--train_data", default="data/grpo/grpo_train.jsonl")
    p.add_argument("--output_dir", default="outputs/grpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_prompt_length", type=int, default=768)
    p.add_argument("--max_completion_length", type=int, default=96)
    p.add_argument("--max_samples", type=int, default=0)
    a = p.parse_args()
    main(a.model, a.train_data, a.output_dir, a.epochs, a.batch_size, a.grad_accum,
         a.lr, a.beta, a.num_generations, a.max_prompt_length,
         a.max_completion_length, a.max_samples)
