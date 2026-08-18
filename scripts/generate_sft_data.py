"""生成 SFT 训练数据（expert demonstrations）。

用法：
    python scripts/generate_sft_data.py                       # 专家策略，确定性
    python scripts/generate_sft_data.py --backend noisy --n 8 --min_reward 1.8

与旧版的区别
------------
旧版有两个致命问题：

1. **训练/推理分布不一致**：user content 是裸任务描述、assistant 是裸 JSON；
   而推理时送进模型的是完整 ReAct prompt，期望输出 `Thought: ...\\nAction: {...}`。
   两者对不上，训完在真实 ReAct 循环里不 work。
2. **多步任务标签互相矛盾**：同一个任务的每一步都生成 user content 完全相同、
   assistant 目标不同的样本，等于教模型在多个动作之间随机。

现在改为直接从执行链路上"录"下每一步真实的 (prompt, completion)：
prompt 天然包含工具描述、格式约束和当前步的历史，因此不同步骤的输入互不相同。

输出格式为 TRL 的 prompt-completion 会话式数据集，
prompt 部分不计 loss，只训练 assistant 的输出。
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "AgenticArxiv"))
os.environ.setdefault("STORE_BACKEND", "memory")

from rl.rollout import make_env, rollout_once  # noqa: E402
from rl.tasks import get_extended_tasks, split_train_eval  # noqa: E402


def generate(
    output: str = "data/sft/sft_train.jsonl",
    backend: str = "expert",
    n: int = 1,
    min_reward: float = 1.5,
    error_rate: float = 0.35,
    seed: int = 0,
    holdout: bool = True,
):
    tasks, eval_tasks = split_train_eval() if holdout else (get_extended_tasks(), [])
    env = make_env()

    print(f"SFT 数据生成 | 策略={backend} 任务={len(tasks)} 每任务{n}条 阈值={min_reward}")
    if holdout:
        print(f"  held-out 评估任务: {len(eval_tasks)} 条（不进训练集）")

    samples, kept, dropped = [], 0, 0
    for task_def in tasks:
        for trial in range(n):
            result, reward, metrics = rollout_once(
                task_def, env, backend=backend, trial=trial,
                seed=seed, error_rate=error_rate,
            )
            if reward < min_reward:
                dropped += 1
                continue
            kept += 1
            for rec in result.get("records", []):
                samples.append({
                    "prompt": rec["messages"],
                    "completion": [{"role": "assistant", "content": rec["completion"]}],
                })
    print(f"  已处理 {len(tasks)} 个任务")

    out_path = REPO_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n保留轨迹 {kept} 条、丢弃 {dropped} 条（reward < {min_reward}）")
    print(f"SFT 样本 {len(samples)} 条 → {out_path}")
    return samples


def main():
    p = argparse.ArgumentParser(description="生成 SFT 数据")
    p.add_argument("--output", default="data/sft/sft_train.jsonl")
    p.add_argument("--backend", default="expert", choices=["expert", "noisy", "remote", "local"])
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--min_reward", type=float, default=1.5)
    p.add_argument("--error_rate", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-holdout", dest="holdout", action="store_false")
    args = p.parse_args()
    generate(
        output=args.output, backend=args.backend, n=args.n,
        min_reward=args.min_reward, error_rate=args.error_rate,
        seed=args.seed, holdout=args.holdout,
    )


if __name__ == "__main__":
    main()
