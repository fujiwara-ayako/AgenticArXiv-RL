"""在 held-out 任务上评估策略。

用法：
    # 评估基座模型 vs SFT 模型
    python eval/evaluate.py --backend local --model Qwen/Qwen2.5-1.5B-Instruct
    python eval/evaluate.py --backend local --model outputs/sft/final

    # 对照组：程序化专家（奖励上界）与带噪专家
    python eval/evaluate.py --backend expert
    python eval/evaluate.py --backend noisy --error_rate 0.4

    # 一次性对比多个检查点
    python eval/evaluate.py --compare base outputs/sft/final outputs/dpo/final

评估集是 rl.tasks.split_train_eval() 划出的 held-out 任务，
训练数据生成时已排除，因此这里衡量的是泛化而非记忆。
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "AgenticArxiv"))
os.environ.setdefault("STORE_BACKEND", "memory")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rl.rollout import make_env, rollout_once  # noqa: E402
from rl.tasks import split_train_eval  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def evaluate(backend="expert", model=None, n=1, error_rate=0.4, seed=0,
             limit=0, verbose=False, device=None):
    _, eval_tasks = split_train_eval()
    if limit:
        eval_tasks = eval_tasks[:limit]

    env = make_env()
    rows = []

    for task_def in eval_tasks:
        for trial in range(n):
            _, reward, m = rollout_once(
                task_def, env, backend=backend, trial=trial, seed=seed,
                error_rate=error_rate, model_name=model, device=device,
            )
            rows.append({
                "task_id": task_def["id"],
                "difficulty": task_def.get("difficulty"),
                "reward": reward,
                "success": bool(m.task_success),
                "tool_accurate": bool(m.tool_call_accurate),
                "arg_score": float(m.arg_score),
                "parse_failures": int(m.parse_failures),
                "tool_exec_failures": int(m.tool_exec_failures),
                "termination": m.termination_type,
            })
            if verbose:
                print(f"  {task_def['id']:<22} r={reward:+.2f} "
                      f"success={int(m.task_success)} pf={m.parse_failures}")

    return rows


def summarize(rows, label=""):
    if not rows:
        return {}
    n = len(rows)
    rewards = [r["reward"] for r in rows]
    out = {
        "label": label,
        "n": n,
        "reward_mean": round(statistics.mean(rewards), 3),
        "reward_std": round(statistics.pstdev(rewards) if n > 1 else 0.0, 3),
        "success_rate": round(sum(r["success"] for r in rows) / n, 3),
        "tool_accuracy": round(sum(r["tool_accurate"] for r in rows) / n, 3),
        "arg_score": round(statistics.mean(r["arg_score"] for r in rows), 3),
        "parse_fail_rate": round(sum(1 for r in rows if r["parse_failures"]) / n, 3),
    }
    return out


def print_table(summaries):
    cols = ["label", "n", "reward_mean", "reward_std", "success_rate",
            "tool_accuracy", "arg_score", "parse_fail_rate"]
    head = ["策略", "n", "平均奖励", "奖励std", "成功率", "工具准确率", "参数分", "解析失败率"]
    widths = [max(len(h), 12) for h in head]
    print(" | ".join(h.ljust(w) for h, w in zip(head, widths)))
    print("-|-".join("-" * w for w in widths))
    for s in summaries:
        vals = [str(s.get(c, "")) for c in cols]
        print(" | ".join(v.ljust(w) for v, w in zip(vals, widths)))


def main():
    p = argparse.ArgumentParser(description="held-out 评估")
    p.add_argument("--backend", default="expert",
                   choices=["expert", "noisy", "local", "remote"])
    p.add_argument("--model", default=None)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--error_rate", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", default=None, help="cpu / mps / cuda，默认自动")
    p.add_argument("--compare", nargs="+", default=None,
                   help="对比多个本地检查点；'base' 表示未微调的基座模型")
    p.add_argument("--out", default=None, help="把结果写成 JSON")
    args = p.parse_args()

    summaries, all_rows = [], {}

    if args.compare:
        for item in args.compare:
            if item == "base":
                model, label = BASE_MODEL, "base(未微调)"
            elif item in ("expert", "noisy"):
                model, label = None, item
            else:
                model = str(REPO_ROOT / item) if (REPO_ROOT / item).exists() else item
                label = item
            backend = item if item in ("expert", "noisy") else "local"
            print(f"\n评估 {label} ...")
            rows = evaluate(backend=backend, model=model, n=args.n,
                            error_rate=args.error_rate, seed=args.seed,
                            limit=args.limit, verbose=args.verbose, device=args.device)
            all_rows[label] = rows
            summaries.append(summarize(rows, label))
    else:
        model = args.model
        if model and (REPO_ROOT / model).exists():
            model = str(REPO_ROOT / model)
        label = model or args.backend
        rows = evaluate(backend=args.backend, model=model, n=args.n,
                        error_rate=args.error_rate, seed=args.seed,
                        limit=args.limit, verbose=args.verbose, device=args.device)
        all_rows[label] = rows
        summaries.append(summarize(rows, label))

    print()
    print_table(summaries)

    if args.out:
        out_path = REPO_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summaries": summaries, "rows": all_rows}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
