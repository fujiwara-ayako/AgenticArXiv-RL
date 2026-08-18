# AgenticArxiv/agents/base_agent.py
import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

from utils.llm_client import LLMClient
from utils.logger import log
from config import settings
from models.schemas import Paper
from tools.tool_registry import registry
from agents.side_effects import SideEffectManager, LocalSideEffectManager


class _ParseFailed:
    """哨兵：表示 LLM 输出无法解析（区别于真正的 FINISH）。

    历史实现里解析失败与 FINISH 都返回 None，导致"输出乱码 == 任务成功 == +1.0 奖励"，
    是一个可被策略利用的 reward hacking 入口。引入哨兵后二者可区分。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<PARSE_FAILED>"

    def __bool__(self) -> bool:
        return False


PARSE_FAILED = _ParseFailed()

# 非工具调用的动作标记（用于指标提取时跳过）
TERMINAL_ACTIONS = ("FINISH", "FORCE_STOP", "ERROR", "PARSE_ERROR")


class BaseAgent(ABC):
    """所有 Agent 方案的基类，封装通用的循环控制、日志、SSE、副作用逻辑"""

    agent_type: str = "regex"  # 子类覆写

    def __init__(
        self,
        llm_client: LLMClient,
        side_effect_mgr: Optional[SideEffectManager] = None,
        env: Optional[Any] = None,
        max_iterations: int = 5,
    ):
        """
        Args:
            llm_client: LLM 调用客户端（策略）
            side_effect_mgr: 副作用管理器；默认 LocalSideEffectManager（无 DB/SSE）
            env: 可选的工具执行环境（需实现 execute_tool(name, args)）。
                 传入 MockArxivEnv 即可让 rollout 走快照回放而非真实 API。
            max_iterations: ReAct 最大迭代轮数
        """
        self.llm_client = llm_client
        self.side_effects = side_effect_mgr or LocalSideEffectManager()
        self.env = env
        self.max_iterations = max_iterations
        self.session_id = "default"

    # ---------- 子类必须实现 ----------

    @abstractmethod
    def discover_tools(self) -> List[Dict[str, Any]]:
        """返回可用工具列表 [{name, description, parameters}]"""

    @abstractmethod
    def build_messages(
        self, task: str, tools_info: List[Dict], history_text: str
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """构造 LLM 请求。返回 (messages, extra_payload)"""

    @abstractmethod
    def parse_response(self, raw_response: Dict) -> Tuple[str, Optional[Dict[str, Any]]]:
        """解析 LLM 响应。

        返回 (thought, action)：
          - action 为 dict          → 正常工具调用
          - action 为 None          → 模型主动 FINISH
          - action 为 PARSE_FAILED  → 无法解析（将计入 parse_failures 惩罚）
        """

    @abstractmethod
    def invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """执行工具，返回原始结果"""

    # ---------- 子类可选覆写 ----------

    def format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """将工具列表格式化为 prompt 中的文本描述，子类可覆写"""
        from agents.prompt_templates import format_tool_description
        return format_tool_description(tools)

    def format_history(self, steps: list) -> str:
        """将历史步骤格式化为 prompt 中的文本，子类可覆写"""
        parts = []
        for s in steps:
            parts.append(
                f"Thought: {s['thought']}\nAction: {s['action']}\nObservation: {s['observation']}"
            )
        return "\n\n".join(parts)

    # ---------- 通用执行循环 ----------

    def run(
        self, task: str, agent_model: str = None, session_id: str = "default"
    ) -> Dict[str, Any]:
        log.info(f"[{self.__class__.__name__}] 开始执行任务: {task}")
        run_start = time.time()
        self.session_id = session_id
        msg_id = uuid.uuid4().hex

        if agent_model is None:
            agent_model = settings.models.agent_model

        try:
            self.side_effects.create_chat_log(
                session_id, msg_id, "user", task, model=agent_model, agent_type=self.agent_type
            )
        except Exception as e:
            log.warning(f"Failed to log user message: {e}")

        tools = self.discover_tools()
        tools_description = self.format_tools_for_prompt(tools)

        # 注入会话上下文，避免 LLM 重复搜索已缓存的论文
        enriched_task = self._enrich_task_with_context(task, session_id)

        history: List[Dict[str, Any]] = []
        step_timings: List[Dict[str, int]] = []
        token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for iteration in range(self.max_iterations):
            log.info(f"第 {iteration + 1} 次迭代")

            history_text = self.format_history(history)
            messages, extra = self.build_messages(enriched_task, tools_description, history_text)

            llm_ms = 0
            tool_ms = 0
            thought = ""
            action_dict = None
            observation = ""

            try:
                t0 = time.time()
                response = self.llm_client.chat_completions(
                    model=agent_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000,
                    stream=False,
                    extra=extra or None,
                )
                llm_ms = int((time.time() - t0) * 1000)

                # 累计 token 用量
                usage = response.get("usage") or {}
                token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                token_usage["total_tokens"] += usage.get("total_tokens", 0)

                thought, action_dict = self.parse_response(response)
                log.info(f"Thought: {thought}")

                # --- 解析失败：不再冒充 FINISH，而是回灌错误并继续尝试 ---
                if action_dict is PARSE_FAILED:
                    log.warning("Action 解析失败，计入 parse_failures 并要求模型重试")
                    observation = (
                        "无法解析你的 Action。请严格按格式输出：\n"
                        'Action: {"name":"工具名","args":{...}}\n'
                        "或在任务完成时输出：Action: FINISH"
                    )
                    history.append({
                        "thought": thought,
                        "action": "PARSE_ERROR",
                        "observation": observation,
                        "parse_failed": True,
                    })
                    step_timings.append({"llm_ms": llm_ms, "tool_ms": 0})
                    self._log_step(
                        msg_id, iteration, thought, "PARSE_ERROR", "",
                        observation, llm_ms, 0, session_id,
                    )
                    if iteration == self.max_iterations - 1:
                        log.warning("达到最大迭代次数，强制结束")
                        history.append({
                            "thought": "达到最大迭代次数", "action": "FORCE_STOP",
                            "observation": "迭代限制", "parse_failed": False,
                        })
                        self._log_step(
                            msg_id, iteration + 1, "达到最大迭代次数", "FORCE_STOP",
                            "", "迭代限制", 0, 0, session_id,
                        )
                        break
                    continue

                if action_dict is None:
                    log.info("任务完成")
                    observation = "任务完成"
                    history.append({
                        "thought": thought, "action": "FINISH",
                        "observation": observation, "parse_failed": False,
                    })
                    step_timings.append({"llm_ms": llm_ms, "tool_ms": 0})
                    self._log_step(msg_id, iteration, thought, "FINISH", "{}", observation, llm_ms, 0, session_id)
                    break

                # 带副作用的工具执行
                t1 = time.time()
                observation = self._execute_with_side_effects(action_dict)
                tool_ms = int((time.time() - t1) * 1000)
                log.info(f"Observation: {observation[:200]}...")

                step_timings.append({"llm_ms": llm_ms, "tool_ms": tool_ms})

                action_str = json.dumps(action_dict, ensure_ascii=False)
                history.append({
                    "thought": thought, "action": action_str,
                    "observation": observation, "parse_failed": False,
                })

                self._log_step(
                    msg_id, iteration, thought,
                    action_dict.get("name", ""), json.dumps(action_dict.get("args", {}), ensure_ascii=False),
                    observation[:4000], llm_ms, tool_ms, session_id,
                )

                if iteration == self.max_iterations - 1:
                    log.warning("达到最大迭代次数，强制结束")
                    history.append({
                        "thought": "达到最大迭代次数", "action": "FORCE_STOP",
                        "observation": "迭代限制", "parse_failed": False,
                    })
                    self._log_step(msg_id, iteration + 1, "达到最大迭代次数", "FORCE_STOP", "", "迭代限制", 0, 0, session_id)
                    break

            except Exception as e:
                error_msg = f"LLM调用失败: {str(e)}"
                log.error(error_msg)
                history.append({
                    "thought": "LLM调用失败", "action": "ERROR",
                    "observation": error_msg, "parse_failed": False,
                })
                step_timings.append({"llm_ms": llm_ms, "tool_ms": 0})
                self._log_step(msg_id, iteration, "LLM调用失败", "ERROR", "", error_msg, llm_ms, 0, session_id)
                break

        final_observation = history[-1]["observation"] if history else "无执行结果"

        reply = ""
        for step in reversed(history):
            if step.get("action") not in TERMINAL_ACTIONS:
                reply = step.get("observation", "")
                break
        reply = reply or final_observation

        try:
            self.side_effects.create_chat_log(
                session_id, msg_id + "_reply", "assistant", reply,
                model=agent_model, agent_type=self.agent_type,
            )
        except Exception as e:
            log.warning(f"Failed to log assistant reply: {e}")

        total_time_ms = int((time.time() - run_start) * 1000)
        total_llm_ms = sum(s["llm_ms"] for s in step_timings)
        total_tool_ms = sum(s["tool_ms"] for s in step_timings)

        result = {
            "task": task,
            "msg_id": msg_id,
            "history": history,
            "final_observation": final_observation,
            "total_time_ms": total_time_ms,
            "iteration_count": len(history),
            "agent_type": self.agent_type,
            "timing": {
                "total_llm_ms": total_llm_ms,
                "total_tool_ms": total_tool_ms,
                "framework_overhead_ms": total_time_ms - total_llm_ms - total_tool_ms,
                "steps": step_timings,
            },
            "token_usage": token_usage,
        }
        log.info(f"任务执行完成，共 {len(history)} 步, 总耗时 {total_time_ms}ms (LLM {total_llm_ms}ms + Tool {total_tool_ms}ms)")
        log.info("-" * 80)
        return result

    # ---------- 会话上下文 ----------

    def _enrich_task_with_context(self, task: str, session_id: str) -> str:
        """向任务描述注入当前会话状态，帮助 LLM 避免重复操作"""
        try:
            papers = self.side_effects.get_last_papers(session_id)
            if papers:
                titles = [f"  {i+1}. {p.title}" for i, p in enumerate(papers[:10])]
                ctx = f"\n\n[会话上下文] 当前会话已有 {len(papers)} 篇论文:\n" + "\n".join(titles)
                ctx += "\n可直接用 ref 序号引用，无需重新搜索。"
                return task + ctx
        except Exception:
            pass
        return task

    # ---------- 工具执行（可被 env 接管） ----------

    def _dispatch_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """统一的工具执行入口：注入了 env 就走 env（快照回放），否则走子类实现。"""
        if self.env is not None:
            return self.env.execute_tool(tool_name, args)
        return self.invoke_tool(tool_name, args)

    # ---------- 通用副作用逻辑 ----------

    def _execute_with_side_effects(self, action_dict: Dict[str, Any]) -> str:
        """统一处理 session_id 覆盖、翻译异步、paper 状态写入等副作用"""
        try:
            tool_name = action_dict["name"]
            args = action_dict.get("args", {}) or {}

            log.info(f"执行工具: {tool_name}, 参数: {args}")

            # 强制覆盖 session_id
            try:
                tool = registry.get_tool(tool_name)
                props = (tool or {}).get("parameters", {}).get("properties", {})
                if isinstance(args, dict) and ("session_id" in props):
                    args["session_id"] = self.session_id
            except Exception:
                pass

            # 验证工具存在
            available_tools = [t["name"] for t in registry.list_tools()]
            if tool_name not in available_tools:
                return f"错误: 工具 '{tool_name}' 不存在。可用工具包括: {', '.join(available_tools)}"

            # 翻译工具异步 enqueue
            if tool_name == "translate_arxiv_pdf":
                t = self.side_effects.enqueue_translate(
                    session_id=self.session_id,
                    ref=args.get("ref", None),
                    force=bool(args.get("force", False)),
                    service=args.get("service") or settings.pdf2zh_service,
                    threads=int(args.get("threads") or settings.pdf2zh_threads),
                    keep_dual=bool(args.get("keep_dual", False)),
                    paper_id=args.get("paper_id"),
                    pdf_url=args.get("pdf_url"),
                    input_pdf_path=args.get("input_pdf_path"),
                )
                return (
                    f"已创建翻译任务 task_id={t.task_id}, paper_id={t.paper_id}，状态={t.status}。"
                    f"前端可订阅 SSE: /events?session_id={self.session_id}，"
                    f"任务完成后刷新 /translate/assets 或 /pdf/assets。"
                )

            # 调用工具（env 优先）
            result = self._dispatch_tool(tool_name, args)

            # paper_id 写入 last_active
            try:
                if isinstance(result, dict):
                    pid = result.get("paper_id")
                    if isinstance(pid, str) and pid.strip():
                        self.side_effects.set_last_active_paper_id(self.session_id, pid.strip())
            except Exception:
                pass

            # arxiv 搜索结果存入 session
            if tool_name == "get_recently_submitted_cs_papers":
                # MCP 模式可能返回 dict（如 {"error": "..."}），兼容处理
                if isinstance(result, dict):
                    if "error" in result:
                        return f"工具执行失败: {result['error']}"
                    vals = list(result.values())
                    if len(vals) == 1 and isinstance(vals[0], list):
                        result = vals[0]
                if isinstance(result, list):
                    if result:
                        papers_obj = [Paper(**p) for p in result]
                        self.side_effects.set_last_papers(self.session_id, papers_obj)
                        papers_count = len(result)
                        paper_titles = [paper.get("title", "无标题") for paper in result[:3]]
                        titles_str = "\n".join([f"  - {title}" for title in paper_titles])
                        if papers_count > 3:
                            return f"成功获取 {papers_count} 篇论文。示例论文:\n{titles_str}\n  ... 还有 {papers_count - 3} 篇论文"
                        else:
                            return f"成功获取 {papers_count} 篇论文:\n{titles_str}"
                    else:
                        return "未获取到任何论文记录，请尝试调整搜索参数（如增加天数范围）"
                else:
                    return f"工具返回结果格式异常: {type(result)}, 内容: {str(result)[:200]}"

            elif tool_name == "format_papers_console":
                return "FINISH"

            # 通用结果格式化
            if isinstance(result, list):
                return f"成功获取 {len(result)} 条记录"
            elif isinstance(result, str):
                return result[:1000] if len(result) > 1000 else result
            else:
                return str(result)[:1000]

        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            log.error(error_msg, exc_info=True)
            return error_msg

    # ---------- 日志 + SSE ----------

    def _log_step(
        self, msg_id: str, step_index: int,
        thought: str, action_name: str, action_args: str,
        observation: str, llm_ms: int, tool_ms: int,
        session_id: str,
    ):
        try:
            self.side_effects.save_agent_step(
                msg_id=msg_id, step_index=step_index,
                thought=thought, action_name=action_name,
                action_args=action_args, observation=observation,
                llm_latency_ms=llm_ms, tool_latency_ms=tool_ms,
            )
            self.side_effects.publish_sse(session_id, {
                "type": "agent_step",
                "step": {
                    "thought": thought, "action_name": action_name,
                    "observation": observation[:500], "step_index": step_index,
                    "llm_latency_ms": llm_ms, "tool_latency_ms": tool_ms,
                },
            })
        except Exception as e:
            log.warning(f"Failed to log step: {e}")
