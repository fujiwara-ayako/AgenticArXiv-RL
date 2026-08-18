"""RL 任务集（在 benchmark/tasks.py 基础上扩展）。

相比 benchmark 任务集，这里补了三样 RL 必需的东西：

1. **gold_actions**：每个任务的标准解法（工具名 + 参数）。
   有了它就能在没有强 LLM 的情况下程序化生成 expert demonstrations（SFT 数据），
   也能给 DPO 构造真正的 chosen 样本。

2. **expected_args**：参数级 ground truth。
   原奖励只看工具名，导致「给 cs.AI 的任务却检索 cs.CR」也能拿满分。
   现在参数正确性单独计分，奖励粒度更细、方差更大。

3. **setup**：前置动作（原 benchmark 用 depends_on + 重跑一遍 Agent）。
   RL 里我们直接用 gold 动作把会话状态铺好，让被评估的 episode 只包含目标任务本身，
   避免前置步骤的成败污染当前任务的 credit assignment。

难度分层（difficulty）用于课程学习与分层统计：
    easy   单工具检索
    medium 依赖会话记忆的单工具（下载/缓存/翻译）
    hard   多步复合、指代消解
"""

from typing import Any, Dict, List, Optional

# 所有任务共享的会话种子：先检索某个方向的论文，写入会话记忆
_SEED_AI = [{"name": "get_recently_submitted_cs_papers", "args": {"aspect": "AI", "days": 7, "max_results": 5}}]
_SEED_LG = [{"name": "get_recently_submitted_cs_papers", "args": {"aspect": "LG", "days": 7, "max_results": 10}}]

RL_TASKS: List[Dict[str, Any]] = [
    # ==================== easy：单工具检索 ====================
    {
        "id": "search_01",
        "task": "检索最近7天内人工智能(cs.AI)方向的论文，最多5篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_args": [{"aspect": "AI", "days": 7, "max_results": 5}],
        "gold_actions": [
            {"name": "get_recently_submitted_cs_papers",
             "args": {"aspect": "AI", "days": 7, "max_results": 5}}
        ],
        "expected_termination": "FINISH",
        "category": "search",
        "difficulty": "easy",
    },
    {
        "id": "search_02",
        "task": "获取最近3天机器学习(cs.LG)方向的最新论文，最多10篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_args": [{"aspect": "LG", "days": 3, "max_results": 10}],
        "gold_actions": [
            {"name": "get_recently_submitted_cs_papers",
             "args": {"aspect": "LG", "days": 3, "max_results": 10}}
        ],
        "expected_termination": "FINISH",
        "category": "search",
        "difficulty": "easy",
    },
    {
        "id": "search_03",
        "task": "搜索最近7天自然语言处理(cs.CL)方向的论文，最多5篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_args": [{"aspect": "CL", "days": 7, "max_results": 5}],
        "gold_actions": [
            {"name": "get_recently_submitted_cs_papers",
             "args": {"aspect": "CL", "days": 7, "max_results": 5}}
        ],
        "expected_termination": "FINISH",
        "category": "search",
        "difficulty": "easy",
    },
    {
        "id": "search_04",
        "task": "检索最近7天计算机视觉(cs.CV)方向的论文，最多3篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_args": [{"aspect": "CV", "days": 7, "max_results": 3}],
        "gold_actions": [
            {"name": "get_recently_submitted_cs_papers",
             "args": {"aspect": "CV", "days": 7, "max_results": 3}}
        ],
        "expected_termination": "FINISH",
        "category": "search",
        "difficulty": "easy",
    },
    {
        "id": "search_05",
        "task": "查找最近14天机器人学(cs.RO)方向的论文，最多8篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_args": [{"aspect": "RO", "days": 14, "max_results": 8}],
        "gold_actions": [
            {"name": "get_recently_submitted_cs_papers",
             "args": {"aspect": "RO", "days": 14, "max_results": 8}}
        ],
        "expected_termination": "FINISH",
        "category": "search",
        "difficulty": "easy",
    },

    # ==================== medium：依赖会话记忆 ====================
    {
        "id": "download_01",
        "task": "下载第1篇论文的PDF",
        "setup": _SEED_AI,
        "expected_tools": ["download_arxiv_pdf"],
        "expected_args": [{"ref": 1}],
        "gold_actions": [{"name": "download_arxiv_pdf", "args": {"ref": 1}}],
        "expected_termination": "FINISH",
        "category": "download",
        "difficulty": "medium",
    },
    {
        "id": "download_02",
        "task": "把第3篇论文的PDF下载下来",
        "setup": _SEED_LG,
        "expected_tools": ["download_arxiv_pdf"],
        "expected_args": [{"ref": 3}],
        "gold_actions": [{"name": "download_arxiv_pdf", "args": {"ref": 3}}],
        "expected_termination": "FINISH",
        "category": "download",
        "difficulty": "medium",
    },
    {
        "id": "cache_01",
        "task": "查看第1篇论文的缓存状态",
        "setup": _SEED_AI,
        "expected_tools": ["get_paper_cache_status"],
        "expected_args": [{"ref": 1}],
        "gold_actions": [{"name": "get_paper_cache_status", "args": {"ref": 1}}],
        "expected_termination": "FINISH",
        "category": "cache",
        "difficulty": "medium",
    },
    {
        "id": "translate_01",
        "task": "翻译第1篇论文",
        "setup": _SEED_AI + [{"name": "download_arxiv_pdf", "args": {"ref": 1}}],
        "expected_tools": ["translate_arxiv_pdf"],
        "expected_args": [{"ref": 1}],
        "gold_actions": [{"name": "translate_arxiv_pdf", "args": {"ref": 1}}],
        "expected_termination": "FINISH",
        "category": "translate",
        "difficulty": "medium",
    },

    # ==================== hard：多步复合 / 指代消解 ====================
    {
        "id": "composite_01",
        "task": "搜索最近7天计算机视觉(cs.CV)的论文(最多3篇)，然后下载第1篇",
        "expected_tools": ["get_recently_submitted_cs_papers", "download_arxiv_pdf"],
        "expected_args": [{"aspect": "CV", "days": 7, "max_results": 3}, {"ref": 1}],
        "gold_actions": [
            {"name": "get_recently_submitted_cs_papers",
             "args": {"aspect": "CV", "days": 7, "max_results": 3}},
            {"name": "download_arxiv_pdf", "args": {"ref": 1}},
        ],
        "expected_termination": "FINISH",
        "category": "composite",
        "difficulty": "hard",
    },
    {
        "id": "composite_02",
        "task": "搜索最近7天密码学与安全(cs.CR)方向的论文(最多5篇)，下载第1篇，然后查看它的缓存状态",
        "expected_tools": [
            "get_recently_submitted_cs_papers", "download_arxiv_pdf", "get_paper_cache_status",
        ],
        "expected_args": [
            {"aspect": "CR", "days": 7, "max_results": 5}, {"ref": 1}, {"ref": 1},
        ],
        "gold_actions": [
            {"name": "get_recently_submitted_cs_papers",
             "args": {"aspect": "CR", "days": 7, "max_results": 5}},
            {"name": "download_arxiv_pdf", "args": {"ref": 1}},
            {"name": "get_paper_cache_status", "args": {"ref": 1}},
        ],
        "expected_termination": "FINISH",
        "category": "composite",
        "difficulty": "hard",
    },
    {
        "id": "referential_01",
        "task": "把刚才那篇论文翻译成中文",
        # 会话里已经下载过第2篇 → last_active_paper_id 指向它
        "setup": _SEED_AI + [{"name": "download_arxiv_pdf", "args": {"ref": 2}}],
        "expected_tools": ["translate_arxiv_pdf"],
        "expected_args": [{"ref": None}],
        "gold_actions": [{"name": "translate_arxiv_pdf", "args": {"ref": None}}],
        "expected_termination": "FINISH",
        "category": "translate",
        "difficulty": "hard",
    },
]


# ---------- 查询辅助 ----------

def get_all_tasks() -> List[Dict[str, Any]]:
    return list(RL_TASKS)


def get_task_by_id(task_id: str) -> Optional[Dict[str, Any]]:
    for t in RL_TASKS:
        if t["id"] == task_id:
            return t
    return None


def get_tasks_by_category(category: str) -> List[Dict[str, Any]]:
    return [t for t in RL_TASKS if t.get("category") == category]


def get_tasks_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    return [t for t in RL_TASKS if t.get("difficulty") == difficulty]


def task_ids() -> List[str]:
    return [t["id"] for t in RL_TASKS]


# ---------- 参数化任务生成（扩充训练集规模） ----------

ASPECT_NAMES = {
    "AI": "人工智能",
    "LG": "机器学习",
    "CL": "自然语言处理",
    "CV": "计算机视觉",
    "RO": "机器人学",
    "CR": "密码学与安全",
}

_SEARCH_TEMPLATES = [
    "检索最近{days}天内{name}(cs.{code})方向的论文，最多{k}篇",
    "帮我找一下最近{days}天{name}(cs.{code})的新论文，最多{k}篇",
    "获取最近{days}天{name}方向(cs.{code})的最新论文，数量上限{k}篇",
    "搜索 cs.{code} 方向最近{days}天的论文，最多返回{k}篇",
]

_DOWNLOAD_TEMPLATES = [
    "下载第{i}篇论文的PDF",
    "把第{i}篇论文的PDF下载下来",
    "请下载列表里第{i}篇论文",
]

_CACHE_TEMPLATES = [
    "查看第{i}篇论文的缓存状态",
    "第{i}篇论文有没有已经下载过？",
    "查一下第{i}篇论文的本地缓存情况",
]


def make_parametric_tasks(
    aspects: Optional[List[str]] = None,
    days_grid: Optional[List[int]] = None,
    k_grid: Optional[List[int]] = None,
    refs: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """按参数网格批量生成任务，用于把训练集从十几条扩到几百条。

    原任务集只有 7~12 条，SFT 样本不足 20 条，模型学不到东西；
    而这些任务的结构是高度参数化的（方向 × 天数 × 数量 × 序号），
    直接按网格展开即可，且 gold_actions / expected_args 可同步推导出来。
    """
    aspects = aspects or list(ASPECT_NAMES.keys())
    days_grid = days_grid or [3, 7, 14]
    k_grid = k_grid or [3, 5, 10]
    refs = refs or [1, 2, 3]

    tasks: List[Dict[str, Any]] = []

    # --- 检索类 ---
    for ai, code in enumerate(aspects):
        for di, days in enumerate(days_grid):
            for ki, k in enumerate(k_grid):
                tmpl = _SEARCH_TEMPLATES[(ai + di + ki) % len(_SEARCH_TEMPLATES)]
                args = {"aspect": code, "days": days, "max_results": k}
                tasks.append({
                    "id": f"gen_search_{code}_{days}_{k}",
                    "task": tmpl.format(days=days, k=k, code=code, name=ASPECT_NAMES[code]),
                    "expected_tools": ["get_recently_submitted_cs_papers"],
                    "expected_args": [args],
                    "gold_actions": [{"name": "get_recently_submitted_cs_papers", "args": args}],
                    "expected_termination": "FINISH",
                    "category": "search",
                    "difficulty": "easy",
                    "generated": True,
                })

    # --- 下载 / 缓存类（带会话种子）---
    for ai, code in enumerate(aspects):
        seed = [{"name": "get_recently_submitted_cs_papers",
                 "args": {"aspect": code, "days": 7, "max_results": 10}}]
        for ri, ref in enumerate(refs):
            tasks.append({
                "id": f"gen_download_{code}_{ref}",
                "task": _DOWNLOAD_TEMPLATES[(ai + ri) % len(_DOWNLOAD_TEMPLATES)].format(i=ref),
                "setup": seed,
                "expected_tools": ["download_arxiv_pdf"],
                "expected_args": [{"ref": ref}],
                "gold_actions": [{"name": "download_arxiv_pdf", "args": {"ref": ref}}],
                "expected_termination": "FINISH",
                "category": "download",
                "difficulty": "medium",
                "generated": True,
            })
            tasks.append({
                "id": f"gen_cache_{code}_{ref}",
                "task": _CACHE_TEMPLATES[(ai + ri) % len(_CACHE_TEMPLATES)].format(i=ref),
                "setup": seed,
                "expected_tools": ["get_paper_cache_status"],
                "expected_args": [{"ref": ref}],
                "gold_actions": [{"name": "get_paper_cache_status", "args": {"ref": ref}}],
                "expected_termination": "FINISH",
                "category": "cache",
                "difficulty": "medium",
                "generated": True,
            })

    # --- 复合类 ---
    for ai, code in enumerate(aspects):
        for k in (3, 5):
            s_args = {"aspect": code, "days": 7, "max_results": k}
            tasks.append({
                "id": f"gen_composite_{code}_{k}",
                "task": (
                    f"搜索最近7天{ASPECT_NAMES[code]}(cs.{code})的论文(最多{k}篇)，然后下载第1篇"
                ),
                "expected_tools": ["get_recently_submitted_cs_papers", "download_arxiv_pdf"],
                "expected_args": [s_args, {"ref": 1}],
                "gold_actions": [
                    {"name": "get_recently_submitted_cs_papers", "args": s_args},
                    {"name": "download_arxiv_pdf", "args": {"ref": 1}},
                ],
                "expected_termination": "FINISH",
                "category": "composite",
                "difficulty": "hard",
                "generated": True,
            })

    return tasks


def get_extended_tasks() -> List[Dict[str, Any]]:
    """手写核心任务 + 参数化生成任务。"""
    return RL_TASKS + make_parametric_tasks()


def split_train_eval(
    tasks: Optional[List[Dict[str, Any]]] = None,
    eval_ids: Optional[List[str]] = None,
    eval_ratio: float = 0.15,
    seed: int = 0,
):
    """划分训练/评估集。

    默认对「扩展任务集」做分层抽样（按 difficulty 分层），保证 held-out 里
    三种难度都有代表；也可以显式给 eval_ids 精确指定。
    划分是确定性的（固定 seed），保证实验可复现。
    """
    import random

    pool = tasks if tasks is not None else get_extended_tasks()

    if eval_ids:
        train = [t for t in pool if t["id"] not in eval_ids]
        evalset = [t for t in pool if t["id"] in eval_ids]
        return train, evalset

    by_diff: Dict[str, List[Dict[str, Any]]] = {}
    for t in pool:
        by_diff.setdefault(t.get("difficulty", "unknown"), []).append(t)

    rng = random.Random(seed)
    eval_id_set = set()
    for diff, items in by_diff.items():
        items = sorted(items, key=lambda x: x["id"])
        k = max(1, int(round(len(items) * eval_ratio)))
        for t in rng.sample(items, min(k, len(items))):
            eval_id_set.add(t["id"])

    train = [t for t in pool if t["id"] not in eval_id_set]
    evalset = [t for t in pool if t["id"] in eval_id_set]
    return train, evalset
