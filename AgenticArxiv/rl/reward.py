"""奖励计算器（复用 benchmark/metrics.py 的 verifiable 组件）

Verifiable Reward 设计：
- 任务成功: +1.0
- 工具调用准确: +0.5
- 解析错误: -0.2 per failure
- 工具执行失败: -0.3 per failure
- 超时(FORCE_STOP): -0.5
- 错误终止(ERROR): -1.0
- 解析错误终止(PARSE_ERROR): -0.5
- 不必要调用: -0.1 per call

所有分项都可由规则直接判定，无需人类标注 → RLVR。
"""

from typing import Dict, Any, Tuple, List

from benchmark.metrics import TaskMetrics, extract_metrics


class RewardCalculator:
    """基于 verifiable metrics 的奖励计算器"""

    def __init__(self, weights: Dict[str, float] = None):
        """初始化奖励权重"""
        self.weights = {
            "task_completed": 1.0,
            "tool_call_accurate": 0.5,
            "arg_score": 0.5,
            "parse_failure_penalty": -0.2,
            "tool_exec_failure_penalty": -0.3,
            "force_stop_penalty": -0.5,
            "error_penalty": -1.0,
            "parse_error_penalty": -0.5,
            "unnecessary_call_penalty": -0.1,
        }
        if weights:
            self.weights.update(weights)

    def compute_reward(
        self,
        task_def: Dict[str, Any],
        result: Dict[str, Any],
        agent_type: str = "regex",
        trial: int = 0,
        session_id: str = "rl_train",
    ) -> Tuple[float, TaskMetrics]:
        """计算 reward + 返回 TaskMetrics

        Args:
            task_def: 任务定义（来自 benchmark/tasks.py 或 rl/tasks.py）
            result: Agent 执行结果（history/timing/token_usage/iteration_count）
            agent_type: Agent 类型（默认 "regex"）
            trial: 试验次数（默认 0）
            session_id: 会话 ID

        Returns:
            (reward, metrics) 元组
        """
        metrics = extract_metrics(task_def, result, agent_type, trial, session_id)
        reward = self.reward_from_metrics(metrics)
        return reward, metrics

    def reward_from_metrics(self, metrics: TaskMetrics) -> float:
        """纯函数：只依赖 TaskMetrics，便于单测与离线重算。"""
        reward = 0.0

        # 正向奖励：挂在 task_success（干净终止 且 工具序列正确）上。
        # 若挂在 task_completed 上，"直接 FINISH 什么都不做" 也能白拿 +1.0。
        if metrics.task_success:
            reward += self.weights["task_completed"]

        if metrics.tool_call_accurate:
            reward += self.weights["tool_call_accurate"]

        # 参数级奖励（连续值，提供比 0/1 更细的梯度）
        reward += self.weights["arg_score"] * float(metrics.arg_score)

        # 负向惩罚
        reward += self.weights["parse_failure_penalty"] * metrics.parse_failures
        reward += self.weights["tool_exec_failure_penalty"] * metrics.tool_exec_failures

        # 终止类型惩罚
        if metrics.termination_type == "FORCE_STOP":
            reward += self.weights["force_stop_penalty"]
        elif metrics.termination_type == "ERROR":
            reward += self.weights["error_penalty"]
        elif metrics.termination_type == "PARSE_ERROR":
            reward += self.weights["parse_error_penalty"]

        # 不必要调用惩罚（调用次数超过 expected_tools 的部分）
        if metrics.expected_tools:
            extra_calls = len(metrics.tool_call_sequence) - len(metrics.expected_tools)
            if extra_calls > 0:
                reward += self.weights["unnecessary_call_penalty"] * extra_calls

        return reward

    def reward_breakdown(self, metrics: TaskMetrics) -> Dict[str, float]:
        """返回各分项贡献，便于调试 reward hacking。"""
        out: Dict[str, float] = {}
        if metrics.task_success:
            out["task_success"] = self.weights["task_completed"]
        if metrics.tool_call_accurate:
            out["tool_call_accurate"] = self.weights["tool_call_accurate"]
        if metrics.arg_score:
            out["arg_score"] = self.weights["arg_score"] * float(metrics.arg_score)
        if metrics.parse_failures:
            out["parse_failures"] = self.weights["parse_failure_penalty"] * metrics.parse_failures
        if metrics.tool_exec_failures:
            out["tool_exec_failures"] = (
                self.weights["tool_exec_failure_penalty"] * metrics.tool_exec_failures
            )
        key = {
            "FORCE_STOP": "force_stop_penalty",
            "ERROR": "error_penalty",
            "PARSE_ERROR": "parse_error_penalty",
        }.get(metrics.termination_type)
        if key:
            out[metrics.termination_type] = self.weights[key]
        if metrics.expected_tools:
            extra = len(metrics.tool_call_sequence) - len(metrics.expected_tools)
            if extra > 0:
                out["unnecessary_calls"] = self.weights["unnecessary_call_penalty"] * extra
        return out

    def get_weights(self) -> Dict[str, float]:
        """返回当前奖励权重配置"""
        return self.weights.copy()

    def set_weights(self, new_weights: Dict[str, float]) -> None:
        """更新奖励权重配置（部分更新）"""
        self.weights.update(new_weights)


def compute_step_reward(step_dict: Dict[str, Any], metrics: TaskMetrics = None) -> float:
    """计算单步奖励（用于 step-wise / 稠密奖励实验）

    Args:
        step_dict: 单步数据（thought/action/observation/parse_failed）
        metrics: 当前累积的 metrics（可选，保留接口兼容）

    Returns:
        单步奖励
    """
    step_reward = 0.0
    observation = step_dict.get("observation", "")

    if step_dict.get("parse_failed", False) or step_dict.get("action") == "PARSE_ERROR":
        return -0.2

    if any(m in observation for m in ("错误:", "工具执行失败:", "命令失败")):
        step_reward -= 0.3
    elif "成功" in observation:
        step_reward += 0.1

    return step_reward


def assign_step_rewards(history: List[Dict[str, Any]]) -> List[float]:
    """对整条轨迹逐步打分（credit assignment 的最简版本）。"""
    return [compute_step_reward(s) for s in history]


# ==================== 动作级奖励（GRPO 用） ====================

import json as _json  # noqa: E402
import re as _re  # noqa: E402

from benchmark.metrics import _normalize_arg  # noqa: E402

# 与 agents/agent_engine.py 的解析规则保持一致
_THOUGHT_RE = _re.compile(r"Thought:\s*(.*?)(?=\nAction:|$)", _re.DOTALL)
_ACTION_RE = _re.compile(r"Action:\s*(.*?)(?=\nObservation:|$)", _re.DOTALL)


def parse_action_text(completion: str):
    """从模型输出里解析动作。

    返回：
        ("finish", None)          模型输出 FINISH
        ("call",   {name, args})  合法工具调用
        ("parse_error", None)     无法解析
    """
    if not completion:
        return "parse_error", None

    m = _ACTION_RE.search(completion)
    if not m:
        return "parse_error", None
    action_text = m.group(1).strip()

    if action_text.upper().startswith("FINISH"):
        return "finish", None

    jm = _re.search(r"(\{.*\})", action_text, _re.DOTALL)
    if not jm:
        return "parse_error", None
    try:
        d = _json.loads(jm.group(1))
    except _json.JSONDecodeError:
        return "parse_error", None       # 严格 JSON：不做降级修复

    if not isinstance(d, dict) or "name" not in d:
        return "parse_error", None
    return "call", {"name": d["name"], "args": d.get("args") or {}}


# 动作级奖励区间：[-1.0, 2.0]
ACTION_REWARDS = {
    "parse_error": -1.0,     # 格式都没对，最重的惩罚
    "wrong_kind": -0.5,      # 该调工具却 FINISH，或该结束却继续调
    "wrong_tool": 0.0,       # 工具选错
    "right_tool": 1.0,       # 工具选对（再按参数加分，最多 +1.0）
    "correct_finish": 2.0,   # 该结束时正确结束
}


def compute_action_reward(completion: str, gold) -> float:
    """单步动作的可验证奖励：把模型输出与该状态下的标准动作对比。

    这是 GRPO 的 reward function —— 之前是 `return [0.0 for _ in responses]` 的占位实现。

    Args:
        completion: 模型这一步的原始输出文本
        gold: 该状态下的标准动作；None / "FINISH" 表示应当结束
    Returns:
        [-1.0, 2.0] 区间内的标量奖励
    """
    kind, action = parse_action_text(completion)

    if kind == "parse_error":
        return ACTION_REWARDS["parse_error"]

    gold_is_finish = gold is None or gold == "FINISH" or (
        isinstance(gold, dict) and not gold.get("name")
    )

    if kind == "finish":
        return ACTION_REWARDS["correct_finish"] if gold_is_finish else ACTION_REWARDS["wrong_kind"]

    # kind == "call"
    if gold_is_finish:
        return ACTION_REWARDS["wrong_kind"]

    if action["name"] != gold.get("name"):
        return ACTION_REWARDS["wrong_tool"]

    gold_args = gold.get("args") or {}
    if not gold_args:
        return ACTION_REWARDS["right_tool"] + 1.0

    actual = action.get("args") or {}
    hit = sum(
        1 for k, v in gold_args.items()
        if k in actual and _normalize_arg(k, actual[k]) == _normalize_arg(k, v)
    )
    return ACTION_REWARDS["right_tool"] + hit / len(gold_args)


def grpo_reward_func(completions, gold=None, **kwargs) -> List[float]:
    """TRL GRPOTrainer 的 reward function。

    TRL 会把数据集里除 prompt 外的列按列名作为 kwargs 传进来，
    因此数据集需要带一个 `gold` 列（JSON 字符串形式的标准动作）。
    """
    golds = gold if gold is not None else [None] * len(completions)
    rewards = []
    for comp, g in zip(completions, golds):
        # 会话式数据集下 completion 是 [{"role":..., "content":...}]
        if isinstance(comp, list):
            text = "\n".join(m.get("content", "") for m in comp)
        else:
            text = str(comp)
        if isinstance(g, str):
            try:
                g = _json.loads(g)
            except _json.JSONDecodeError:
                g = None
        rewards.append(compute_action_reward(text, g))
    return rewards
