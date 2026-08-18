"""Rollout 循环（收集 trajectory）

完整链路：
    任务定义 → [setup 铺会话状态] → ReActAgent(policy, env, side_effects)
             → history → RewardCalculator → Trajectory → JSONL

关键设计
--------
* **完全离线**：store 走内存、工具走 MockArxivEnv 快照、翻译走 mock 句柄。
  跑 rollout 不需要 MySQL、不需要 API key、不发任何网络请求。
* **会话隔离**：每个 episode 用独立 session_id 并重置内存 store，
  避免上一条轨迹的 last_active_paper_id 泄漏到下一条。
* **setup 前置**：依赖类任务（下载/翻译/指代）先用 gold 动作把会话状态铺好，
  被评估的 episode 只包含目标任务本身，credit assignment 更干净。

用法：
    python -m rl.rollout single search_01
    python -m rl.rollout all --n 4 --backend noisy --error_rate 0.4
    python -m rl.rollout all --backend expert          # 专家上界对照
"""

import os
import statistics
import sys
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 必须在触碰 models.store 之前设置：让全链路走内存后端
os.environ.setdefault("STORE_BACKEND", "memory")

import fire

import tools.arxiv_tool  # noqa: F401
import tools.cache_status_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401

from agents.agent_engine import ReActAgent
from agents.side_effects import LocalSideEffectManager
from models.store import use_memory_store
from rl.env import MockArxivEnv
from rl.policy import RecordingPolicy, make_policy
from rl.reward import RewardCalculator
from rl.tasks import get_all_tasks, get_task_by_id
from rl.trajectory import create_trajectory, save_trajectory

DEFAULT_SNAPSHOT = Path(__file__).resolve().parent.parent.parent / "data" / "mock_arxiv_snapshot.json"
DEFAULT_TRACE_DIR = Path(__file__).resolve().parent.parent.parent / "traces" / "train"


def make_env(snapshot: Optional[str] = None, mode: str = "replay") -> MockArxivEnv:
    path = Path(snapshot) if snapshot else DEFAULT_SNAPSHOT
    if not path.exists():
        raise FileNotFoundError(
            f"快照不存在: {path}\n请先运行: cd AgenticArxiv && python -m rl.build_snapshot"
        )
    return MockArxivEnv(snapshot_path=path, mode=mode)


def apply_setup(task_def: Dict[str, Any], env: MockArxivEnv,
                side_fx: LocalSideEffectManager, session_id: str) -> None:
    """执行任务的前置动作，把会话状态铺好（不经过 LLM）。"""
    from models.schemas import Paper

    for action in task_def.get("setup") or []:
        name = action["name"]
        args = dict(action.get("args") or {})
        args["session_id"] = session_id

        if name == "translate_arxiv_pdf":
            # 注意：args 里已经带了 session_id，不能再显式传一次，否则重复关键字参数报错
            side_fx.enqueue_translate(**args)
            continue

        result = env.execute_tool(name, args)

        # 复刻 BaseAgent 里的会话写入逻辑，保证 setup 后的状态与真实执行一致
        if name == "get_recently_submitted_cs_papers" and isinstance(result, list) and result:
            side_fx.set_last_papers(session_id, [Paper(**p) for p in result])
        if isinstance(result, dict) and isinstance(result.get("paper_id"), str):
            side_fx.set_last_active_paper_id(session_id, result["paper_id"])


def rollout_once(
    task_def: Dict[str, Any],
    env: MockArxivEnv,
    backend: str = "expert",
    trial: int = 0,
    seed: int = 0,
    error_rate: float = 0.4,
    model_name: Optional[str] = None,
    reward_calc: Optional[RewardCalculator] = None,
    max_iterations: int = 5,
    strict_parse: bool = True,
    device: Optional[str] = None,
) -> Tuple[Dict[str, Any], float, Any]:
    """跑一条轨迹，返回 (result, reward, metrics)。

    result 里额外带一个 `records` 字段：本条轨迹每一步真实的
    (messages, completion) 对，供 SFT/DPO 数据生成直接使用。
    """
    task_id = task_def["id"]
    session_id = f"rl_{task_id}_t{trial}"

    # 会话隔离：每条轨迹从干净状态开始
    use_memory_store(reset=True)
    side_fx = LocalSideEffectManager()

    # 种子里混入 task_id，否则所有任务的第 k 条轨迹会注入同一种失败模式
    task_seed = (zlib.crc32(task_id.encode()) + seed * 1_000_003 + trial * 7919) % (2 ** 31)

    policy = make_policy(
        backend,
        task_def=task_def,
        seed=task_seed,
        error_rate=error_rate,
        model_name_or_path=model_name,
        device=device,
    )
    policy = RecordingPolicy(policy)
    policy.reset(task_def)

    apply_setup(task_def, env, side_fx, session_id)

    agent = ReActAgent(
        llm_client=policy,
        side_effect_mgr=side_fx,
        env=env,
        max_iterations=max_iterations,
        strict_parse=strict_parse,
    )
    result = agent.run(
        task_def["task"],
        agent_model=getattr(policy, "model", backend),
        session_id=session_id,
    )

    result["records"] = policy.records

    reward_calc = reward_calc or RewardCalculator()
    reward, metrics = reward_calc.compute_reward(
        task_def, result, agent_type="regex", trial=trial, session_id=session_id
    )
    return result, reward, metrics


def _to_trajectory(task_def, result, reward, metrics, session_id, model):
    return create_trajectory(
        task_id=task_def["id"],
        task=task_def["task"],
        session_id=session_id,
        history=result.get("history", []),
        final_reward=reward,
        metrics={
            "task_completed": metrics.task_completed,
            "task_success": metrics.task_success,
            "tool_call_accurate": metrics.tool_call_accurate,
            "arg_score": round(metrics.arg_score, 3),
            "parse_failures": metrics.parse_failures,
            "tool_exec_failures": metrics.tool_exec_failures,
            "termination_type": metrics.termination_type,
            "tool_call_sequence": metrics.tool_call_sequence,
            "expected_tools": metrics.expected_tools,
            "iteration_count": metrics.iteration_count,
            "total_tokens": metrics.total_tokens,
            "difficulty": task_def.get("difficulty", "unknown"),
        },
        model=model,
        termination_type=metrics.termination_type,
    )


def _print_row(task_def, trial, reward, metrics):
    status = "PASS" if metrics.task_success else f"FAIL({metrics.termination_type})"
    tools = " -> ".join(metrics.tool_call_sequence) or "(none)"
    print(
        f"  [{task_def['id']:<15} t{trial}] reward={reward:+.2f} {status:<18} "
        f"acc={int(metrics.tool_call_accurate)} arg={metrics.arg_score:.2f} "
        f"pf={metrics.parse_failures} tf={metrics.tool_exec_failures} | {tools}"
    )


def _summarize(rewards: List[float], label: str) -> None:
    if not rewards:
        print(f"\n{label}: 无数据")
        return
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    print(
        f"\n{label}: n={len(rewards)} mean={mean:+.3f} std={std:.3f} "
        f"min={min(rewards):+.2f} max={max(rewards):+.2f}"
    )
    if std < 1e-9:
        print("  ⚠️  奖励方差为 0 → GRPO 组内优势恒为 0，没有梯度信号。")


# ---------------- CLI ----------------

def single(
    task_id: str = "search_01",
    output_dir: str = None,
    backend: str = "expert",
    n: int = 1,
    seed: int = 0,
    error_rate: float = 0.4,
    snapshot: str = None,
    model_name: str = None,
    save: bool = True,
    strict_parse: bool = True,
):
    """对单个任务执行 rollout。"""
    task_def = get_task_by_id(task_id)
    if not task_def:
        print(f"任务 {task_id} 不存在。可用: {[t['id'] for t in get_all_tasks()]}")
        return

    env = make_env(snapshot)
    reward_calc = RewardCalculator()
    out_dir = Path(output_dir) if output_dir else DEFAULT_TRACE_DIR
    out_path = out_dir / f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    print(f"任务: {task_def['id']} [{task_def.get('difficulty')}] - {task_def['task']}")
    print(f"策略: {backend} | 轨迹数: {n}")

    rewards = []
    for trial in range(n):
        result, reward, metrics = rollout_once(
            task_def, env, backend=backend, trial=trial, seed=seed,
            error_rate=error_rate, model_name=model_name, reward_calc=reward_calc,
            strict_parse=strict_parse,
        )
        rewards.append(reward)
        _print_row(task_def, trial, reward, metrics)
        if save:
            traj = _to_trajectory(
                task_def, result, reward, metrics,
                f"rl_{task_id}_t{trial}", backend,
            )
            save_trajectory(traj, out_path)

    _summarize(rewards, "汇总")
    print(f"环境统计: {env.describe()}")
    if save:
        print(f"轨迹已保存: {out_path}")


def all(
    output_dir: str = None,
    backend: str = "expert",
    n: int = 1,
    seed: int = 0,
    error_rate: float = 0.4,
    snapshot: str = None,
    model_name: str = None,
    save: bool = True,
    strict_parse: bool = True,
    difficulty: str = None,
):
    """对所有任务执行 rollout。"""
    tasks = get_all_tasks()
    if difficulty:
        tasks = [t for t in tasks if t.get("difficulty") == difficulty]

    env = make_env(snapshot)
    reward_calc = RewardCalculator()
    out_dir = Path(output_dir) if output_dir else DEFAULT_TRACE_DIR
    out_path = out_dir / f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    print(f"任务数: {len(tasks)} | 策略: {backend} | 每任务 {n} 条轨迹")
    print("=" * 78)

    all_rewards: List[float] = []
    by_difficulty: Dict[str, List[float]] = {}
    n_pass = 0

    for task_def in tasks:
        for trial in range(n):
            result, reward, metrics = rollout_once(
                task_def, env, backend=backend, trial=trial, seed=seed,
                error_rate=error_rate, model_name=model_name, reward_calc=reward_calc,
                strict_parse=strict_parse,
            )
            all_rewards.append(reward)
            by_difficulty.setdefault(task_def.get("difficulty", "?"), []).append(reward)
            n_pass += int(metrics.task_success)
            _print_row(task_def, trial, reward, metrics)
            if save:
                traj = _to_trajectory(
                    task_def, result, reward, metrics,
                    f"rl_{task_def['id']}_t{trial}", backend,
                )
                save_trajectory(traj, out_path)

    print("=" * 78)
    for diff in ("easy", "medium", "hard"):
        if diff in by_difficulty:
            _summarize(by_difficulty[diff], f"难度={diff}")
    _summarize(all_rewards, "全部")
    print(f"任务成功率: {n_pass}/{len(all_rewards)} = {n_pass / max(1, len(all_rewards)):.0%}")
    print(f"环境统计: {env.describe()}")
    if save:
        print(f"轨迹已保存: {out_path}")


if __name__ == "__main__":
    fire.Fire({"single": single, "all": all})
