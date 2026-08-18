"""生成 DPO 偏好数据（chosen / rejected）。

用法：
    python scripts/generate_dpo_data.py --n 8 --error_rate 0.45

与旧版的区别
------------
旧版取 `history[-1]["action"]` 作为 chosen/rejected，但**成功轨迹的最后一步动作
恒为字符串 "FINISH"**，于是 `chosen == rejected` → 全部被 `continue` 跳过，
实测产出 0 条样本；即便偶有产出，比较的也是终止标记而非工具选择偏好。

现在按「同一状态下的不同动作」构造偏好对：

    1. 每个任务用带噪策略采样 N 条轨迹，记录每一步真实的 (prompt, completion)
    2. 按 prompt 字符串分组 —— 同一个 prompt 就是同一个状态
       （ReAct prompt 里已经包含任务 + 完整历史，能唯一确定状态）
    3. 组内按所属轨迹的 return 排序，最高 vs 最低构成一对
    4. 过滤掉 completion 相同、或 return 差距过小的对

这本质上是 advantage-weighted preference：同状态下，偏好导致更高回报的动作。
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "AgenticArxiv"))
os.environ.setdefault("STORE_BACKEND", "memory")

from rl.rollout import make_env, rollout_once  # noqa: E402
from rl.tasks import split_train_eval  # noqa: E402


def generate(
    output: str = "data/dpo/dpo_train.jsonl",
    backend: str = "noisy",
    n: int = 8,
    error_rate: float = 0.45,
    seed: int = 0,
    min_margin: float = 0.3,
    holdout: bool = True,
):
    tasks, eval_tasks = split_train_eval() if holdout else (None, [])
    if tasks is None:
        from rl.tasks import get_extended_tasks
        tasks = get_extended_tasks()

    env = make_env()
    print(f"DPO 数据生成 | 策略={backend} 任务={len(tasks)} 每任务{n}条 最小 margin={min_margin}")

    pairs = []
    for task_def in tasks:
        # state(prompt) -> [(return, completion)]
        by_state = defaultdict(list)
        rewards = []

        for trial in range(n):
            result, reward, _ = rollout_once(
                task_def, env, backend=backend, trial=trial,
                seed=seed, error_rate=error_rate,
            )
            rewards.append(reward)
            for rec in result.get("records", []):
                key = json.dumps(rec["messages"], ensure_ascii=False, sort_keys=True)
                by_state[key].append((reward, rec["completion"], rec["messages"]))

        made = 0
        for _, entries in by_state.items():
            if len(entries) < 2:
                continue
            entries.sort(key=lambda x: x[0], reverse=True)
            best, worst = entries[0], entries[-1]
            if best[1] == worst[1]:
                continue                      # 同一个动作，没有偏好信息
            if best[0] - worst[0] < min_margin:
                continue                      # 回报差距太小，信号噪声比低
            pairs.append({
                "prompt": best[2],
                "chosen": [{"role": "assistant", "content": best[1]}],
                "rejected": [{"role": "assistant", "content": worst[1]}],
            })
            made += 1

        span = f"{min(rewards):+.2f}~{max(rewards):+.2f}" if rewards else "n/a"
        print(f"  {task_def['id']:<15} reward范围={span:<14} 新增偏好对={made}")

    out_path = REPO_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDPO 偏好对 {len(pairs)} 条 → {out_path}")
    return pairs


def main():
    p = argparse.ArgumentParser(description="生成 DPO 偏好数据")
    p.add_argument("--output", default="data/dpo/dpo_train.jsonl")
    p.add_argument("--backend", default="noisy")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--error_rate", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_margin", type=float, default=0.3)
    p.add_argument("--no-holdout", dest="holdout", action="store_false")
    args = p.parse_args()
    generate(
        output=args.output, backend=args.backend, n=args.n,
        error_rate=args.error_rate, seed=args.seed,
        min_margin=args.min_margin, holdout=args.holdout,
    )


if __name__ == "__main__":
    main()
