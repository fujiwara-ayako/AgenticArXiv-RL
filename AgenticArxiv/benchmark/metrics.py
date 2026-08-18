# AgenticArxiv/benchmark/metrics.py
"""从 Agent run() 结果中提取性能和准确性指标。"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class TaskMetrics:
    """单次任务执行的完整指标"""
    task_id: str
    agent_type: str
    trial: int
    session_id: str = ""

    # --- 性能 ---
    total_time_ms: int = 0
    iteration_count: int = 0
    total_llm_ms: int = 0
    total_tool_ms: int = 0
    framework_overhead_ms: int = 0
    avg_llm_ms: float = 0.0
    avg_tool_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # --- 准确性 ---
    task_completed: bool = False   # 干净终止（FINISH）——仅表示"没崩、没超时"
    task_success: bool = False     # 真正完成任务：干净终止 且 调用了预期工具
    termination_type: str = "UNKNOWN"
    tool_call_sequence: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    tool_call_accurate: bool = False
    parse_failures: int = 0
    tool_exec_failures: int = 0

    # --- 参数级准确性（RL 扩展）---
    arg_score: float = 0.0       # 0~1，命中的期望参数占比
    arg_match: bool = False      # arg_score == 1.0

    # --- 原始数据 ---
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tool_call_sequence"] = ",".join(d["tool_call_sequence"])
        d["expected_tools"] = ",".join(d["expected_tools"])
        return d


def extract_metrics(
    task_def: Dict[str, Any],
    result: Dict[str, Any],
    agent_type: str,
    trial: int,
    session_id: str = "",
) -> TaskMetrics:
    """从 agent.run() 返回值提取 TaskMetrics"""
    history = result.get("history", [])
    timing = result.get("timing", {})
    token_usage = result.get("token_usage", {})

    # --- 性能指标 ---
    total_time_ms = result.get("total_time_ms", 0)
    total_llm_ms = timing.get("total_llm_ms", 0)
    total_tool_ms = timing.get("total_tool_ms", 0)
    framework_overhead_ms = timing.get("framework_overhead_ms", total_time_ms - total_llm_ms - total_tool_ms)
    iteration_count = result.get("iteration_count", len(history))

    effective_steps = max(1, iteration_count)
    avg_llm_ms = round(total_llm_ms / effective_steps, 1)
    avg_tool_ms = round(total_tool_ms / effective_steps, 1)

    # --- 准确性指标 ---
    termination_type = _get_termination_type(history)
    task_completed = termination_type == "FINISH"

    tool_sequence = _extract_tool_sequence(history)
    expected_tools = task_def.get("expected_tools", [])
    tool_call_accurate = _check_tool_sequence(tool_sequence, expected_tools)

    parse_failures = _count_parse_failures(history)
    tool_exec_failures = _count_tool_failures(history)

    arg_score = _score_args(history, task_def.get("expected_args") or [])

    # 「干净终止」≠「完成任务」。
    # 只要求 FINISH 的话，"什么都不做直接 FINISH" 也能拿满 +1.0，
    # 这是策略最容易学到的 reward hacking 捷径（实测 completion_rate 恒为 100%）。
    # 因此把奖励挂到 task_success 上：必须真的调用了预期工具序列。
    task_success = task_completed and tool_call_accurate

    error = None
    if termination_type == "ERROR" and history:
        error = history[-1].get("observation", "")

    return TaskMetrics(
        task_id=task_def["id"],
        agent_type=agent_type,
        trial=trial,
        session_id=session_id,
        total_time_ms=total_time_ms,
        iteration_count=iteration_count,
        total_llm_ms=total_llm_ms,
        total_tool_ms=total_tool_ms,
        framework_overhead_ms=framework_overhead_ms,
        avg_llm_ms=avg_llm_ms,
        avg_tool_ms=avg_tool_ms,
        prompt_tokens=token_usage.get("prompt_tokens", 0),
        completion_tokens=token_usage.get("completion_tokens", 0),
        total_tokens=token_usage.get("total_tokens", 0),
        task_completed=task_completed,
        task_success=task_success,
        termination_type=termination_type,
        tool_call_sequence=tool_sequence,
        expected_tools=expected_tools,
        tool_call_accurate=tool_call_accurate,
        parse_failures=parse_failures,
        tool_exec_failures=tool_exec_failures,
        arg_score=arg_score,
        arg_match=(arg_score >= 1.0),
        error=error,
    )


# rl/ 与改造计划文档里一直用的是 compute_metrics 这个名字，提供别名避免二次踩坑
compute_metrics = extract_metrics


# 非工具调用的动作标记（与 agents.base_agent.TERMINAL_ACTIONS 保持一致）
NON_TOOL_ACTIONS = ("FINISH", "FORCE_STOP", "ERROR", "PARSE_ERROR")


def _get_termination_type(history: List[Dict]) -> str:
    if not history:
        return "NO_HISTORY"
    last_action = history[-1].get("action", "")
    if last_action == "FINISH":
        return "FINISH"
    elif last_action == "FORCE_STOP":
        return "FORCE_STOP"
    elif last_action == "ERROR":
        return "ERROR"
    elif last_action == "PARSE_ERROR":
        # 解析失败收尾（未走到 FORCE_STOP）：明确区别于正常完成
        return "PARSE_ERROR"
    # action 是 JSON 字符串（工具调用后没有正常终止）
    return "INCOMPLETE"


def _extract_tool_sequence(history: List[Dict]) -> List[str]:
    """从 history 中提取实际调用的工具名序列"""
    tools = []
    for step in history:
        action = step.get("action", "")
        if action in NON_TOOL_ACTIONS:
            continue
        try:
            action_dict = json.loads(action)
            name = action_dict.get("name", "")
            if name:
                tools.append(name)
        except (json.JSONDecodeError, TypeError):
            pass
    return tools


def _check_tool_sequence(actual: List[str], expected: List[str]) -> bool:
    """检查实际工具调用是否包含所有预期工具（顺序匹配）"""
    if not expected:
        return True
    ei = 0
    for tool in actual:
        if ei < len(expected) and tool == expected[ei]:
            ei += 1
    return ei == len(expected)


def _extract_tool_calls(history: List[Dict]) -> List[Dict[str, Any]]:
    """提取 [{name, args}] 序列（含参数，供参数级评分使用）。"""
    calls = []
    for step in history:
        action = step.get("action", "")
        if action in NON_TOOL_ACTIONS:
            continue
        try:
            d = json.loads(action)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(d, dict) and d.get("name"):
            calls.append({"name": d["name"], "args": d.get("args") or {}})
    return calls


def _normalize_arg(key: str, value: Any) -> Any:
    """参数归一化，避免把等价写法判成错误。

    - aspect: "cs.AI" / "ai" → "AI"
    - days / max_results / ref: 数字字符串 → int
    - ref=None 保持 None（指代语义，不能与 0/"" 混同）
    """
    if value is None:
        return None
    if key == "aspect":
        s = str(value).strip()
        if s.lower().startswith("cs."):
            s = s[3:]
        return s.upper()
    if key in ("days", "max_results", "ref"):
        if isinstance(value, bool):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value).strip().lower()
    return str(value).strip().lower()


def _score_args(history: List[Dict], expected_args: List[Dict[str, Any]]) -> float:
    """参数级准确率：命中的期望参数键占全部期望键的比例（0~1）。

    与 expected_tools 一样按"顺序对齐"匹配：第 i 组期望参数对应第 i 次工具调用。
    没有 expected_args 的任务返回 1.0（不惩罚）。
    """
    total = sum(len(a) for a in expected_args)
    if total == 0:
        return 1.0

    calls = _extract_tool_calls(history)
    hit = 0
    for i, exp in enumerate(expected_args):
        if i >= len(calls):
            break
        actual = calls[i].get("args") or {}
        for k, v in exp.items():
            if k in actual and _normalize_arg(k, actual[k]) == _normalize_arg(k, v):
                hit += 1
    return hit / total


def _count_parse_failures(history: List[Dict]) -> int:
    """统计解析失败次数。

    首选显式标记 `parse_failed`（由 BaseAgent 在遇到 PARSE_FAILED 哨兵时写入），
    这是唯一可靠的判据。另外两条是兼容旧轨迹的兜底规则。

    历史坑：旧实现只靠"FINISH 出现在非末尾"和 observation 含"无法解析"来猜，
    而旧 BaseAgent 遇到 FINISH 立即 break、解析失败又被伪装成 FINISH，
    导致这个计数恒为 0，-0.2 的惩罚项从未生效。
    """
    failures = 0
    for i, step in enumerate(history):
        # 1) 显式标记（新轨迹）
        if step.get("parse_failed"):
            failures += 1
            continue
        # 2) 动作标记（新轨迹，防御性）
        if step.get("action") == "PARSE_ERROR":
            failures += 1
            continue
        # 3) 旧轨迹兼容
        if step.get("action") == "FINISH" and i < len(history) - 1:
            failures += 1
        elif "无法解析" in step.get("observation", ""):
            failures += 1
    return failures


def _count_tool_failures(history: List[Dict]) -> int:
    """统计工具执行失败次数"""
    failures = 0
    error_markers = ["错误:", "工具执行失败:", "命令失败", "命令执行超时", "命令执行异常"]
    for step in history:
        action = step.get("action", "")
        if action in NON_TOOL_ACTIONS:
            continue
        obs = step.get("observation", "")
        if any(marker in obs for marker in error_markers):
            failures += 1
    return failures
