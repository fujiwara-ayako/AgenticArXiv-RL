"""生成 GRPO 训练数据（prompt + 该状态下的标准动作）。

用法：
    python scripts/generate_grpo_data.py

说明
----
TRL 的 GRPOTrainer 是「单轮生成 + 打分」的结构，不会替你跑完整 Agent 循环。
因此这里把多步 ReAct 轨迹拆成「每一步一个样本」：

    prompt = 该步真实送进模型的 ReAct prompt（含工具描述 + 历史）
    gold   = 该状态下的标准动作（工具名 + 参数），或 "FINISH"

训练时 GRPO 对同一个 prompt 采样 num_generations 条输出，
用 rl.reward.grpo_reward_func 逐条打分，组内做相对优势估计。
奖励完全由规则判定（RLVR），不需要 reward model。
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "AgenticArxiv"))
os.environ.setdefault("STORE_BACKEND", "memory")

from rl.reward import parse_action_text  # noqa: E402
from rl.rollout import make_env, rollout_once  # noqa: E402
from rl.tasks import split_train_eval  # noqa: E402


def generate(output: str = "data/grpo/grpo_train.jsonl", holdout: bool = True, seed: int = 0):
    tasks, eval_tasks = split_train_eval()
    if not holdout:
        from rl.tasks import get_extended_tasks
        tasks, eval_tasks = get_extended_tasks(), []

    env = make_env()
    print(f"GRPO 数据生成 | 任务={len(tasks)}（held-out {len(eval_tasks)} 条）")

    rows, skipped = [], 0
    for task_def in tasks:
        # 专家策略保证每一步的 gold 都是标准解
        result, reward, _ = rollout_once(task_def, env, backend="expert", trial=0, seed=seed)
        if reward < 1.5:
            skipped += 1
            continue
        for rec in result.get("records", []):
            kind, action = parse_action_text(rec["completion"])
            if kind == "parse_error":
                skipped += 1
                continue
            gold = "FINISH" if kind == "finish" else action
            rows.append({
                "prompt": rec["messages"],
                "gold": json.dumps(gold, ensure_ascii=False),
                "task_id": task_def["id"],
            })

    out_path = REPO_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_finish = sum(1 for r in rows if r["gold"] == '"FINISH"')
    print(f"样本 {len(rows)} 条（其中 FINISH 步 {n_finish} 条），跳过 {skipped}")
    print(f"→ {out_path}")
    return rows


def main():
    p = argparse.ArgumentParser(description="生成 GRPO 数据")
    p.add_argument("--output", default="data/grpo/grpo_train.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-holdout", dest="holdout", action="store_false")
    args = p.parse_args()
    generate(output=args.output, holdout=args.holdout, seed=args.seed)


if __name__ == "__main__":
    main()
