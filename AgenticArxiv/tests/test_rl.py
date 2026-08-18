#!/usr/bin/env python3
"""RL 链路回归测试 —— 每条用例对应一个已修复的缺陷，防止回归。

运行：
    cd AgenticArxiv && python tests/test_rl.py

不依赖 pytest，直接跑即可（与仓库里其他测试脚本风格一致）。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.arxiv_tool  # noqa: F401,E402
import tools.cache_status_tool  # noqa: F401,E402
import tools.pdf_download_tool  # noqa: F401,E402
import tools.pdf_translate_tool  # noqa: F401,E402

from agents.agent_engine import ReActAgent  # noqa: E402
from agents.base_agent import PARSE_FAILED  # noqa: E402
from agents.side_effects import LocalSideEffectManager  # noqa: E402
from models.store import use_memory_store  # noqa: E402
from rl.policy import NoisyExpertPolicy, ScriptedPolicy  # noqa: E402
from rl.reward import RewardCalculator, compute_action_reward  # noqa: E402
from rl.rollout import make_env, rollout_once  # noqa: E402
from rl.tasks import get_task_by_id  # noqa: E402

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def run_agent(task_id, policy, strict_parse=True, env=None):
    use_memory_store(reset=True)
    task = get_task_by_id(task_id)
    env = env or make_env()
    agent = ReActAgent(
        llm_client=policy, side_effect_mgr=LocalSideEffectManager(),
        env=env, strict_parse=strict_parse,
    )
    policy.reset(task)
    result = agent.run(task["task"], session_id="test")
    reward, metrics = RewardCalculator().compute_reward(task, result)
    return result, reward, metrics


def main():
    env = make_env()
    print("\n=== 1. 导入与依赖解耦 ===")
    # 缺陷：rl/reward.py 导入了不存在的 compute_metrics
    from benchmark.metrics import compute_metrics, extract_metrics
    check("rl.reward 可导入且 compute_metrics 别名存在", compute_metrics is extract_metrics)

    # 缺陷：requirements 声称去掉了 MySQL/FastAPI，但 RL 路径仍硬依赖
    check("RL 链路未加载 sqlalchemy", "sqlalchemy" not in sys.modules)
    check("RL 链路未加载 fastapi", "fastapi" not in sys.modules)

    # 缺陷：ReActAgent.__init__ 不接受 side_effect_mgr
    a = ReActAgent(llm_client=None, side_effect_mgr=LocalSideEffectManager(), env=env)
    check("ReActAgent 接受 side_effect_mgr / env", a.side_effects is not None and a.env is env)

    print("\n=== 2. 解析失败不再冒充任务成功 ===")
    # 缺陷：解析失败返回 None，与 FINISH 同信号 → 白拿 +1.0
    p = ScriptedPolicy(["Thought: 我随便说点什么，不给 Action"])
    _, reward, m = run_agent("search_01", p)
    check("无 Action 段计入 parse_failures", m.parse_failures >= 1, f"pf={m.parse_failures}")
    check("解析失败不算 task_success", not m.task_success)
    check("解析失败奖励为负", reward < 0, f"reward={reward:+.2f}")

    # 缺陷：JSON 坏掉时被"文本降级提取"静默修好，惩罚失效
    bad = "Thought: t\nAction: {'name': 'get_recently_submitted_cs_papers', 'args': {'days': 7,}}"
    _, r_lenient, m_lenient = run_agent("search_01", ScriptedPolicy([bad]), strict_parse=False)
    _, r_strict, m_strict = run_agent("search_01", ScriptedPolicy([bad]), strict_parse=True)
    check("strict_parse=True 时坏 JSON 被判解析失败",
          m_strict.parse_failures >= 1 and m_lenient.parse_failures == 0,
          f"strict pf={m_strict.parse_failures}, lenient pf={m_lenient.parse_failures}")
    check("strict 模式奖励显著低于宽松模式", r_strict < r_lenient,
          f"{r_strict:+.2f} < {r_lenient:+.2f}")

    print("\n=== 3. 空转 FINISH 不再白拿奖励 ===")
    # 缺陷：task_completed 只表示"干净终止"，直接 FINISH 也 +1.0
    _, reward, m = run_agent("search_01", ScriptedPolicy(["Thought: 完成了\nAction: FINISH"]))
    check("直接 FINISH 的 task_completed 为真（干净终止）", m.task_completed)
    check("直接 FINISH 的 task_success 为假", not m.task_success)
    check("直接 FINISH 奖励为 0", abs(reward) < 1e-6, f"reward={reward:+.2f}")

    print("\n=== 4. 参数级正确性 ===")
    # 缺陷：奖励只看工具名，参数填错也满分
    good = ('Thought: t\nAction: {"name":"get_recently_submitted_cs_papers",'
            '"args":{"aspect":"AI","days":7,"max_results":5}}')
    wrong = ('Thought: t\nAction: {"name":"get_recently_submitted_cs_papers",'
             '"args":{"aspect":"CR","days":30,"max_results":99}}')
    fin = "Thought: done\nAction: FINISH"
    _, r_good, m_good = run_agent("search_01", ScriptedPolicy([good, fin]))
    _, r_wrong, m_wrong = run_agent("search_01", ScriptedPolicy([wrong, fin]))
    check("参数正确得满分", m_good.arg_score == 1.0, f"arg={m_good.arg_score}")
    check("参数错误扣分", m_wrong.arg_score < 1.0, f"arg={m_wrong.arg_score}")
    check("参数错误奖励更低", r_wrong < r_good, f"{r_wrong:+.2f} < {r_good:+.2f}")

    print("\n=== 5. MockArxivEnv 真正接管工具执行 ===")
    e = make_env()
    _, _, _ = rollout_once(get_task_by_id("search_01"), e, backend="expert")
    check("检索命中快照、零真实网络调用",
          e.stats["hit"] >= 1 and e.stats["real_calls"] == 0, e.describe())
    e2 = make_env()
    _, _, m2 = rollout_once(get_task_by_id("composite_02"), e2, backend="expert")
    check("多步复合任务离线可完成", m2.task_success, f"tools={m2.tool_call_sequence}")
    check("下载走离线桩（不发 HTTP）", e2.stats["offline_stubs"] >= 1, e2.describe())

    print("\n=== 6. 指代消解（ref=null）===")
    e3 = make_env()
    _, r_ref, m_ref = rollout_once(get_task_by_id("referential_01"), e3, backend="expert")
    check("ref=null 指代任务可完成", m_ref.task_success, f"reward={r_ref:+.2f}")

    print("\n=== 7. 奖励方差（GRPO 前提）===")
    e4 = make_env()
    rewards = [rollout_once(get_task_by_id("search_01"), e4, backend="noisy",
                            trial=i, error_rate=0.5)[1] for i in range(12)]
    spread = max(rewards) - min(rewards)
    check("带噪策略产生奖励方差", spread > 0.5,
          f"range={min(rewards):+.2f}~{max(rewards):+.2f}")

    print("\n=== 8. 动作级奖励（GRPO reward_fn）===")
    gold = {"name": "get_recently_submitted_cs_papers",
            "args": {"aspect": "AI", "days": 7, "max_results": 5}}
    r_perfect = compute_action_reward(good, gold)
    r_wrongtool = compute_action_reward(
        'Thought: t\nAction: {"name":"download_arxiv_pdf","args":{"ref":1}}', gold)
    r_badjson = compute_action_reward("Thought: t\nAction: {oops", gold)
    r_equiv = compute_action_reward(
        'Thought: t\nAction: {"name":"get_recently_submitted_cs_papers",'
        '"args":{"aspect":"cs.AI","days":"7","max_results":"5"}}', gold)
    check("奖励序：正确 > 选错工具 > 坏JSON",
          r_perfect > r_wrongtool > r_badjson,
          f"{r_perfect:+.2f} > {r_wrongtool:+.2f} > {r_badjson:+.2f}")
    check("cs.AI 与 AI 视为等价", r_equiv == r_perfect, f"{r_equiv:+.2f}")
    check("GRPO reward_fn 非占位实现", r_perfect != 0.0 and r_badjson != 0.0)

    print("\n=== 9. Trajectory 往返 ===")
    from rl.trajectory import load_trajectories, save_trajectory
    from rl.rollout import _to_trajectory
    import tempfile
    res, rw, mt = rollout_once(get_task_by_id("search_01"), make_env(), backend="expert")
    traj = _to_trajectory(get_task_by_id("search_01"), res, rw, mt, "s", "expert")
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "t.jsonl"
        save_trajectory(traj, fp)
        loaded = load_trajectories(fp)
    check("Trajectory JSONL 读写往返一致",
          len(loaded) == 1 and abs(loaded[0].final_reward - rw) < 1e-9
          and len(loaded[0].steps) == len(traj.steps))

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        print("失败用例:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
