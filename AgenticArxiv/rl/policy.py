"""策略（policy）后端：把"谁来产生 Action"这件事抽象出来。

所有后端都实现同一个契约（与 utils.llm_client.LLMClient 一致）：

    chat_completions(model, messages, temperature, max_tokens, stream, extra) -> dict
    返回 {"choices":[{"message":{"content": str}}], "usage": {...}}

因此可以直接塞进 BaseAgent，无需改动执行循环。

后端一览
--------
RemoteAPIPolicy   OpenAI 兼容 HTTP 接口（原 LLMClient，需要 LLM_API_KEY）
LocalHFPolicy     本地 transformers 模型（GRPO 训练 / 无 API key 时的真实策略）
ScriptedPolicy    预设脚本，按顺序返回（单测、回归）
ExpertPolicy      ★ 按任务的 gold_actions 程序化产出标准解法
                    → 没有强 LLM 也能造出 expert demonstrations
NoisyExpertPolicy ★ 在 Expert 基础上按概率注入 6 类真实失败模式
                    → 制造奖励方差，这是 DPO 偏好对与 GRPO 组内优势的前提
"""

import json
import random
from typing import Any, Dict, List, Optional

# 供 NoisyExpertPolicy 挑选"错误工具"
ALL_TOOLS = [
    "get_recently_submitted_cs_papers",
    "download_arxiv_pdf",
    "translate_arxiv_pdf",
    "get_paper_cache_status",
]

FAILURE_MODES = (
    "bad_json",      # JSON 语法错误 → 解析失败
    "no_action",     # 只有 Thought 没有 Action → 解析失败
    "wrong_tool",    # 选错工具 → tool_call_accurate=False
    "wrong_args",    # 参数填错 → arg_score 下降
    "early_finish",  # 什么都没做直接 FINISH → 无工具调用
    "extra_call",    # 多调一次无关工具 → unnecessary_call 惩罚
)


def _wrap(content: str, prompt_chars: int = 0) -> Dict[str, Any]:
    """把纯文本包装成 OpenAI 兼容响应。"""
    # 粗略 token 估计（4 字符 ≈ 1 token），仅用于统计，不影响训练
    p = max(1, prompt_chars // 4)
    c = max(1, len(content) // 4)
    return {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c},
    }


def render_action(name: str, args: Dict[str, Any]) -> str:
    """渲染成 ReAct 的 Action 行（严格 JSON，与 prompt 约束一致）。"""
    return json.dumps({"name": name, "args": args}, ensure_ascii=False)


def render_react_step(thought: str, action: Optional[str]) -> str:
    """渲染完整的一步 ReAct 输出。action=None 表示 FINISH。"""
    if action is None:
        return f"Thought: {thought}\nAction: FINISH"
    return f"Thought: {thought}\nAction: {action}"


class BasePolicy:
    """策略基类：默认无状态，可被有状态子类覆写 reset()。"""

    model: str = "policy"

    def reset(self, task_def: Optional[Dict[str, Any]] = None) -> None:
        pass

    def chat_completions(self, model=None, messages=None, temperature=0.1,
                         max_tokens=1000, stream=False, extra=None) -> Dict[str, Any]:
        raise NotImplementedError


class RemoteAPIPolicy(BasePolicy):
    """OpenAI 兼容远程接口（对 utils.llm_client.LLMClient 的薄封装）。"""

    def __init__(self, client=None):
        if client is None:
            from utils.llm_client import get_env_llm_client
            client = get_env_llm_client()
        self.client = client
        self.model = getattr(client, "model", "remote")

    def chat_completions(self, model=None, messages=None, temperature=0.1,
                         max_tokens=1000, stream=False, extra=None) -> Dict[str, Any]:
        return self.client.chat_completions(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=stream, extra=extra,
        )


class LocalHFPolicy(BasePolicy):
    """本地 HuggingFace 模型策略（无需 API key）。"""

    def __init__(self, model_name_or_path: str, device: str = None,
                 dtype: str = "auto", max_new_tokens: int = 160,
                 max_prompt_tokens: int = 4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.model = model_name_or_path
        self.max_new_tokens = max_new_tokens
        # ReAct prompt 实测 ~2300 tokens（含 4 个工具的完整 JSON Schema）。
        # 这里若设成 2048，HF 默认从右侧截断，会把「当前任务 + 历史」整段切掉，
        # 模型只看到工具说明书 → 评测结果失真。务必留足余量。
        self.max_prompt_tokens = max_prompt_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        self.hf_model.eval()

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return "\n".join(m.get("content", "") for m in messages)

    def chat_completions(self, model=None, messages=None, temperature=0.1,
                         max_tokens=1000, stream=False, extra=None) -> Dict[str, Any]:
        import torch

        prompt = self._render_prompt(messages or [])
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.max_prompt_tokens).to(self.device)
        do_sample = temperature is not None and temperature > 0
        with torch.no_grad():
            out = self.hf_model.generate(
                **inputs,
                max_new_tokens=min(self.max_new_tokens, max_tokens or self.max_new_tokens),
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else None,
                top_p=0.95 if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        content = self.tokenizer.decode(gen, skip_special_tokens=True)
        p = int(inputs["input_ids"].shape[1])
        c = int(gen.shape[0])
        return {
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c},
        }


class ScriptedPolicy(BasePolicy):
    """按顺序返回预设文本；耗尽后一直返回 FINISH。用于单测与回归。"""

    def __init__(self, script: List[str]):
        self.script = list(script)
        self._i = 0

    def reset(self, task_def=None) -> None:
        self._i = 0

    def chat_completions(self, model=None, messages=None, temperature=0.1,
                         max_tokens=1000, stream=False, extra=None) -> Dict[str, Any]:
        prompt_chars = sum(len(m.get("content", "")) for m in (messages or []))
        if self._i < len(self.script):
            content = self.script[self._i]
            self._i += 1
        else:
            content = render_react_step("任务已完成", None)
        return _wrap(content, prompt_chars)


class ExpertPolicy(BasePolicy):
    """按任务 gold_actions 逐步产出标准解法，最后 FINISH。

    作用：在没有强 LLM / 没有 API key 的情况下，也能得到「专家轨迹」，
    用于 SFT 数据生成、DPO 的 chosen 侧、以及作为奖励上界的对照组。
    """

    model = "expert"

    def __init__(self, task_def: Optional[Dict[str, Any]] = None):
        self.task_def: Dict[str, Any] = task_def or {}
        self._step = 0

    def reset(self, task_def: Optional[Dict[str, Any]] = None) -> None:
        if task_def is not None:
            self.task_def = task_def
        self._step = 0

    def gold_actions(self) -> List[Dict[str, Any]]:
        return self.task_def.get("gold_actions") or []

    def _thought_for(self, i: int, action: Dict[str, Any]) -> str:
        name = action["name"]
        mapping = {
            "get_recently_submitted_cs_papers": "需要先检索符合条件的论文列表",
            "download_arxiv_pdf": "已有论文列表，接下来下载指定论文的 PDF",
            "translate_arxiv_pdf": "PDF 已就绪，调用翻译工具处理",
            "get_paper_cache_status": "查询该论文的本地缓存状态",
        }
        return mapping.get(name, f"第 {i + 1} 步：调用 {name}")

    def next_content(self) -> str:
        golds = self.gold_actions()
        if self._step < len(golds):
            action = golds[self._step]
            content = render_react_step(
                self._thought_for(self._step, action),
                render_action(action["name"], action.get("args", {})),
            )
            self._step += 1
            return content
        return render_react_step("所有步骤已完成，任务结束", None)

    def chat_completions(self, model=None, messages=None, temperature=0.1,
                         max_tokens=1000, stream=False, extra=None) -> Dict[str, Any]:
        prompt_chars = sum(len(m.get("content", "")) for m in (messages or []))
        return _wrap(self.next_content(), prompt_chars)


class NoisyExpertPolicy(ExpertPolicy):
    """在 ExpertPolicy 基础上按概率注入失败模式，制造奖励方差。

    这是复现 RL 训练信号的关键：原 benchmark 任务集在强模型上 100% 通过、
    奖励恒为常数，GRPO 的组内优势恒为 0，梯度为零。
    用可控噪声策略采样，就能得到分布合理的 reward，从而验证
    「DPO 偏好对构造」和「GRPO 优势估计」这两条链路确实工作。
    """

    model = "noisy-expert"

    def __init__(self, task_def=None, error_rate: float = 0.4, seed: int = 0,
                 modes: Optional[List[str]] = None):
        super().__init__(task_def)
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.modes = list(modes or FAILURE_MODES)
        self.injected: List[str] = []

    def reset(self, task_def=None) -> None:
        super().reset(task_def)
        self.injected = []

    def _corrupt_args(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        bad = dict(args)
        if name == "get_recently_submitted_cs_papers":
            choice = self.rng.choice(["aspect", "max_results", "days"])
            if choice == "aspect":
                pool = [a for a in ("AI", "LG", "CL", "CV", "RO", "CR")
                        if a != str(args.get("aspect"))]
                bad["aspect"] = self.rng.choice(pool)
            elif choice == "max_results":
                bad["max_results"] = int(args.get("max_results", 5)) + self.rng.choice([1, 5, 10])
            else:
                bad["days"] = int(args.get("days", 7)) + self.rng.choice([1, 7, 14])
        else:
            cur = args.get("ref")
            bad["ref"] = 2 if cur in (1, None) else 1
        return bad

    def next_content(self) -> str:
        golds = self.gold_actions()

        # 已经走完 gold 序列 → 正常 FINISH
        if self._step >= len(golds):
            return render_react_step("所有步骤已完成，任务结束", None)

        if self.rng.random() >= self.error_rate:
            return super().next_content()

        mode = self.rng.choice(self.modes)
        self.injected.append(mode)
        action = golds[self._step]
        name, args = action["name"], action.get("args", {})

        if mode == "bad_json":
            # Python 风格字面量 + 尾随逗号：prompt 明令禁止，这里正是要触发解析失败
            self._step += 1
            return (
                "Thought: 调用工具获取结果\n"
                "Action: {'name': '%s', 'args': {'ref': None,}}" % name
            )

        if mode == "no_action":
            self._step += 1
            return "Thought: 我先想一想应该怎么做，稍后再决定调用哪个工具。"

        if mode == "wrong_tool":
            self._step += 1
            pool = [t for t in ALL_TOOLS if t != name]
            wrong = self.rng.choice(pool)
            return render_react_step("选择一个工具来完成任务", render_action(wrong, args))

        if mode == "wrong_args":
            self._step += 1
            return render_react_step(
                "调用工具，参数我估一个", render_action(name, self._corrupt_args(name, args))
            )

        if mode == "early_finish":
            self._step = len(golds)  # 直接跳到结尾
            return render_react_step("我认为任务已经完成了", None)

        if mode == "extra_call":
            # 不推进 _step：下一轮仍会尝试正确动作（模拟"多绕了一步"）
            pool = [t for t in ALL_TOOLS if t != name]
            extra = self.rng.choice(pool)
            return render_react_step(
                "先查一下别的信息", render_action(extra, {"ref": 1})
            )

        return super().next_content()


class RecordingPolicy(BasePolicy):
    """装饰器：记录每一步真实的 (messages, completion) 对。

    这是修复「SFT 数据与推理分布不一致」的关键。
    原 generate_sft_data.py 把 user content 写成裸任务描述、assistant 写成裸 JSON，
    而推理时 BaseAgent 送进去的是完整 ReAct prompt（工具描述 + 格式约束 + 历史），
    模型输出的是 `Thought: ...\\nAction: {...}`。两者对不上，训完在真实循环里不work。

    直接把训练样本从执行链路上"录"下来，就不存在对不上的问题。
    """

    def __init__(self, inner: BasePolicy):
        self.inner = inner
        self.records: List[Dict[str, Any]] = []

    @property
    def model(self):
        return getattr(self.inner, "model", "recorded")

    def reset(self, task_def=None) -> None:
        self.inner.reset(task_def)
        self.records = []

    def chat_completions(self, model=None, messages=None, temperature=0.1,
                         max_tokens=1000, stream=False, extra=None) -> Dict[str, Any]:
        resp = self.inner.chat_completions(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=stream, extra=extra,
        )
        content = resp["choices"][0]["message"]["content"]
        self.records.append({"messages": list(messages or []), "completion": content})
        return resp


def make_policy(backend: str = "expert", **kwargs) -> BasePolicy:
    """策略工厂。

    backend: expert | noisy | remote | local | scripted
    """
    backend = (backend or "expert").lower()
    if backend == "expert":
        return ExpertPolicy(kwargs.get("task_def"))
    if backend in ("noisy", "noisy_expert"):
        return NoisyExpertPolicy(
            kwargs.get("task_def"),
            error_rate=kwargs.get("error_rate", 0.4),
            seed=kwargs.get("seed", 0),
        )
    if backend == "remote":
        return RemoteAPIPolicy(kwargs.get("client"))
    if backend == "local":
        return LocalHFPolicy(
            kwargs.get("model_name_or_path") or kwargs.get("model")
            or "Qwen/Qwen2.5-1.5B-Instruct",
            device=kwargs.get("device"),
            max_new_tokens=kwargs.get("max_new_tokens", 160),
            max_prompt_tokens=kwargs.get("max_prompt_tokens", 4096),
        )
    if backend == "scripted":
        return ScriptedPolicy(kwargs.get("script") or [])
    raise ValueError(f"未知 policy backend: {backend}")
