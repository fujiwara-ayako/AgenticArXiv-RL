# AgenticArXiv-RL — Agentic RL 训练环境

> **基于 ReAct Agent + arXiv 工具的 Agentic RL 训练环境**  
> 支持 SFT/DPO/GRPO/PPO 渐进式训练路径，用于研究 LLM Agent 强化学习

---

## 🎯 项目定位

将 arXiv 论文检索/下载/翻译任务改造为**可训练的强化学习环境**，专注于：

1. **Verifiable Reward**：基于规则化奖励（工具调用准确度、任务完成度、解析错误等），无需人类标注
2. **渐进式训练**：SFT（监督微调）→ DPO（直接偏好优化）→ GRPO（组内相对策略优化）→ PPO（近端策略优化）
3. **轻量级工程**：纯 Python + JSONL 存储，无需 MySQL/FastAPI/前端，专注离线训练

**非目标**：生产级 arXiv 应用、Web UI、实时翻译服务。

> 说明：这些 Web 相关模块（`api/` `services/` `mcp_protocol/` `skill_cli/` `AgenticArxivWeb/`）
> **仍然保留在仓库里**并可正常使用（`archive/` 里装的是更早期的调研项目，与本项目无关）。
> RL 路径通过 `STORE_BACKEND=memory` + `agents/side_effects.py` 与它们完全解耦：
> 只装 `requirements.txt` 里的依赖就能跑完整条 RL 链路，不需要 fastapi / sqlalchemy / pymysql。

---

## 🚀 快速开始

### 前置要求

- Python 3.9+
- LLM API（支持 OpenAI API 格式，如 Claude、Gemini、Qwen 等）
- 使用 `.venv` 虚拟环境

### 1️⃣ 克隆项目

```bash
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
```

### 2️⃣ 环境配置

**创建虚拟环境**：
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

**安装依赖**：
```bash
pip install -r AgenticArxiv/requirements.txt
```

**配置 LLM API**：
```bash
cat > AgenticArxiv/.env << 'EOF'
# LLM API 配置
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo

# 可选：PDF 路径配置
PDF_RAW_PATH=./output/pdf_raw
PDF_TRANSLATED_PATH=./output/pdf_translated
EOF
```

**关于 LLM API**：**不配也能跑**。RL 链路默认使用离线环境 + 内置策略
（程序化专家 / 带噪专家 / 本地 HF 模型），不需要 API key、不需要 MySQL、不发网络请求。
只有 `rl.build_snapshot`（一次性生成论文快照）和使用 `--backend remote` 时才需要外网。

### 3️⃣ 生成离线快照（只需一次，唯一联网步骤）

```bash
cd AgenticArxiv
python -m rl.build_snapshot          # 抓 7 个方向 × 50 篇真实论文 → data/mock_arxiv_snapshot.json
```

### 4️⃣ 测试 Rollout

```bash
python -m rl.rollout single search_01 --backend expert
```

**期望输出**：
```
任务: search_01 [easy] - 检索最近7天内人工智能(cs.AI)方向的论文，最多5篇
策略: expert | 轨迹数: 1
  [search_01       t0] reward=+2.00 PASS  acc=1 arg=1.00 pf=0 tf=0 | get_recently_submitted_cs_papers
汇总: n=1 mean=+2.000 std=0.000
环境统计: mode=replay hit=1 miss=0 real_calls=0 offline_stubs=0
```

`real_calls=0` 表示这条轨迹完全走快照回放，没有触碰外网。

### 5️⃣ 跑回归测试

```bash
python tests/test_rl.py       # 24 条用例，覆盖全部已修复缺陷
```

---

## 📚 核心概念

### MDP 设计

| 维度 | 定义 |
|------|------|
| **State** | 任务描述 + 对话历史 + 工具结果 |
| **Action** | 4 个工具（arxiv搜索/下载/翻译/缓存查询）+ FINISH |
| **Reward** | Verifiable（任务成功+1.0、工具准确+0.5、解析错误-0.2 等） |
| **Transition** | `execute_tool(action) → observation` |

### 离线环境（MockArxivEnv）

RL 的状态转移由 `rl/env.py` 接管，四个工具各有策略：

| 工具 | 离线处理方式 |
|---|---|
| `get_recently_submitted_cs_papers` | 快照回放：按 aspect 取论文池、按 `max_results` 切片，参数变化也能正确响应 |
| `download_arxiv_pdf` | 离线桩：解析 ref → 写占位文件 → 更新 store，不发 HTTP |
| `translate_arxiv_pdf` | `LocalSideEffectManager` 返回确定性 mock 句柄，不起线程、不调 pdf2zh |
| `get_paper_cache_status` | 真实执行（纯本地读内存 store） |

`env.describe()` 会报告 `hit / miss / real_calls / offline_stubs`，可用来确认 rollout 真的没打外网。

### 动作空间（4 个工具）

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — 搜索 arXiv 论文
2. `download_arxiv_pdf(ref, session_id)` — 下载 PDF
3. `translate_arxiv_pdf(ref, session_id)` — 翻译 PDF
4. `get_paper_cache_status(ref, session_id)` — 查询缓存状态

### Verifiable Reward 组件（轨迹级）

| 维度 | 奖励 | 来源 |
|------|------|------|
| 任务成功 | +1.0 | `task_success` = 干净终止 **且** 工具序列正确 |
| 工具调用准确 | +0.5 | `tool_call_accurate`（`_check_tool_sequence`） |
| 参数正确 | +0.5 × 比例 | `arg_score`（对比 `expected_args`，连续值） |
| 解析错误 | -0.2 / 次 | `parse_failures`（由 `PARSE_FAILED` 哨兵显式标记） |
| 工具执行失败 | -0.3 / 次 | `tool_exec_failures` |
| 超时 | -0.5 | `termination_type == "FORCE_STOP"` |
| 错误终止 | -1.0 | `termination_type == "ERROR"` |
| 不必要调用 | -0.1 / 次 | 调用次数超出 `expected_tools` 的部分 |

奖励区间约 **[-1.5, +2.0]**。六类典型行为的实测取值：

| 行为 | reward |
|---|---|
| 完全正确 | +2.00 |
| 参数填错 | +1.50 ~ +1.83 |
| 工具选错 | +0.50 |
| 空转直接 FINISH | +0.00 |
| 输出乱码 / 缺 Action | -0.20 |
| 反复无关调用直到超时 | -1.50 |

> ⚠️ **两个曾经存在的 reward hacking 入口（已修复）**
> 1. 奖励原本挂在 `task_completed` 上，而它只表示"以 FINISH 结束"——
>    于是「什么都不做直接 FINISH」也能白拿 +1.0。现改挂 `task_success`。
> 2. 解析失败原本与 FINISH 返回同一个信号（`None`），乱码输出会被判成任务成功。
>    现用 `PARSE_FAILED` 哨兵区分，并计入 `parse_failures`。

### 动作级 Reward（GRPO 用）

`rl/reward.py::compute_action_reward` 把单步输出与该状态的标准动作对比，区间 `[-1.0, +2.0]`：

| 情况 | reward |
|---|---|
| 工具与参数全对 / 该结束时正确结束 | +2.00 |
| 工具对、参数部分对 | +1.00 ~ +2.00 |
| 工具选错 | 0.00 |
| 该调工具却 FINISH（或反之） | -0.50 |
| 无法解析 | -1.00 |

参数比较带归一化：`cs.AI` ≡ `AI`，`"5"` ≡ `5`，避免把等价写法误判为错误。

**关键**：所有奖励都是 **可验证的**（rule-based），无需人类标注 → 对应 RLVR（Reinforcement Learning with Verifiable Reward）框架。

---

## 📊 复现结果

完整的修复清单与实测数据见 **[`documents/rl_reproduction.md`](documents/rl_reproduction.md)**。要点：

| 结论 | 依据 |
|---|---|
| ✅ RL 环境可用 | 24/24 回归测试；奖励正确分层（`−1.0 / −0.5 / 0.0 / +2.0`）；GRPO 在未见 prompt 上 6/8 组组内方差非零 → 有真实梯度 |
| ✅ 训练管线正确 | 过拟合检验 `train_loss` 3.879 → 0.2172、输出 4/4 可解析 |
| ⚠️ 部分见效 | held-out 上解析失败率 **100% → 20%**、平均奖励 **−1.50 → −0.30**（格式学会并泛化到未见任务） |
| ❌ 未证明方法有效 | 成功率仍 0%（小样本导致塌缩到只输出 FINISH）；完整 SFT 因欠训练与基座输出逐字相同 |

要拿到有效训练结果，优先级依次是：换 1.5B 基座 → 提高 lr 到 1e-4~2e-4 → 压缩 ReAct prompt（现 ~2300 tokens，大部分是工具 JSON Schema）。

---

## ⚡ 一条命令跑通全流程

```bash
cd AgenticArxiv && python -m rl.build_snapshot    # ① 一次性联网，生成快照
cd ..
python scripts/generate_sft_data.py               # ② 专家演示 → SFT 数据
python scripts/generate_dpo_data.py --n 6         # ③ 带噪采样 → DPO 偏好对
python scripts/generate_grpo_data.py              # ④ 逐步 gold → GRPO 数据
cd AgenticArxiv
python -m rl.train_sft                            # ⑤ SFT
python -m rl.train_dpo                            # ⑥ DPO
python -m rl.train_grpo --max_samples 16          # ⑦ GRPO
cd ..
python eval/evaluate.py --compare base outputs/sft/final   # ⑧ held-out 对比评估
```

## 🛠️ 训练路径（SFT → DPO → GRPO）

### 阶段1：SFT（Supervised Fine-Tuning）

**目标**：让模型学会基本的工具调用格式。

**步骤**：
1. 生成 expert demonstrations（用 `ExpertPolicy` 按任务的 `gold_actions` 程序化产出，无需强 LLM）：
   ```bash
   python scripts/generate_sft_data.py
   ```
2. 训练：
   ```bash
   python -m rl.train_sft
   ```
3. 产出：`./outputs/sft/final` 模型

**数据格式**（`data/sft/sft_train.jsonl`，TRL prompt-completion 会话式）：
```json
{
  "prompt":     [{"role": "user", "content": "<完整 ReAct prompt：工具描述 + 格式约束 + 当前步历史>"}],
  "completion": [{"role": "assistant", "content": "Thought: 需要先检索符合条件的论文列表\nAction: {\"name\":\"get_recently_submitted_cs_papers\",\"args\":{...}}"}]
}
```

> 样本是**从执行链路上录下来的**（`rl/policy.py::RecordingPolicy`），
> prompt 就是推理时真正送进模型的那一串，completion 就是 `Thought/Action` 全文。
> 这样才能保证训练分布 == 推理分布。
> 旧版把 prompt 写成裸任务描述、completion 写成裸 JSON，且多步任务的每一步共用同一个
> prompt 却给不同标签（自相矛盾），训出来在真实 ReAct 循环里不 work。

---

### 阶段2：DPO（Direct Preference Optimization）

**目标**：让模型偏好正确的工具选择，拒绝错误路由。

**步骤**：
1. 带噪策略多次 rollout，按「同一状态下的不同动作」构造偏好对：
   ```bash
   python scripts/generate_dpo_data.py --n 6
   ```
2. 训练：
   ```bash
   python -m rl.train_dpo
   ```
3. 产出：`./outputs/dpo/final` 模型

**构造方式**：每个任务采样 N 条轨迹 → 按 prompt 分组（ReAct prompt 含任务+完整历史，
可唯一确定状态）→ 组内按轨迹 return 排序，最高 vs 最低成对 → 过滤掉动作相同或 margin < 0.3 的对。
本质是 advantage-weighted preference。

> 旧版取 `history[-1]["action"]` 作 chosen/rejected，但成功轨迹的最后一步恒为 `"FINISH"`，
> 于是 `chosen == rejected` 被全部跳过，**实测产出 0 条样本**。现版本在 97 个训练任务上产出 97 条偏好对。

---

### 阶段3：GRPO（Group Relative Policy Optimization）

**目标**：用 verifiable reward 在线训练，无需 value model。

**步骤**：
```bash
python scripts/generate_grpo_data.py      # 拆成"每步一个样本"：prompt + gold 动作
python -m rl.train_grpo --max_samples 16
```

**产出**：`./outputs/grpo/final` 模型

**奖励函数**：`rl/reward.py::grpo_reward_func` —— GRPO 对同一 prompt 采样
`num_generations` 条输出，逐条用规则打分，组内做相对优势估计，无需 reward model。

> 旧版 `reward_fn` 是 `return [0.0 for _ in responses]` 的占位实现，
> 且 `GRPOTrainer(...)` 整段被注释掉，运行只会打印一行 TODO。

**优势**：
- 无需 reward model（DPO 的缺点：无法在线学习）
- 无需 value model（PPO 的缺点：显存开销大）
- 适合小模型（如 Qwen2.5-1.5B）

---

## 📂 目录结构

```
AgenticArXiv-RL/
├─ AgenticArxiv/                     # Python 包
│  ├─ agents/                        # Agent 核心
│  │  ├─ base_agent.py              # 通用 ReAct 循环（副作用可注入 / env 可注入 / PARSE_FAILED 哨兵）
│  │  ├─ agent_engine.py            # ReActAgent（RL 策略），支持 strict_parse
│  │  ├─ prompt_templates.py
│  │  ├─ context_manager.py
│  │  └─ side_effects.py            # NoOp / Local / MySQL 三种副作用实现
│  ├─ tools/                         # 工具层（动作空间）
│  │  ├─ tool_registry.py
│  │  ├─ arxiv_tool.py
│  │  ├─ pdf_download_tool.py
│  │  ├─ pdf_translate_tool.py
│  │  └─ cache_status_tool.py
│  ├─ benchmark/                     # ⭐ Verifiable Reward 来源
│  │  ├─ metrics.py                 # TaskMetrics（含 task_success / arg_score）
│  │  ├─ tasks.py                   # 原 7 个 benchmark 任务
│  │  ├─ runner.py / report.py / run_benchmark.py
│  ├─ rl/                            # ⭐ RL 核心
│  │  ├─ env.py                     # MockArxivEnv（快照回放 + 离线下载桩）
│  │  ├─ policy.py                  # Expert / NoisyExpert / LocalHF / Remote / Scripted / Recording
│  │  ├─ reward.py                  # RewardCalculator + compute_action_reward + grpo_reward_func
│  │  ├─ trajectory.py              # Trajectory + JSONL 读写
│  │  ├─ tasks.py                   # RL 任务集（12 手写 + 102 参数化 = 114）
│  │  ├─ rollout.py                 # rollout 循环（CLI: single / all）
│  │  ├─ build_snapshot.py          # 生成离线快照（唯一联网步骤）
│  │  ├─ train_sft.py / train_dpo.py / train_grpo.py
│  ├─ models/
│  │  ├─ store.py                   # 后端分发器（memory / mysql，代理对象）
│  │  ├─ store_memory.py            # 内存实现（RL 用）
│  │  ├─ store_mysql.py             # MySQL 实现（Web 用）
│  │  ├─ db.py / orm.py / schemas.py
│  ├─ tests/
│  │  └─ test_rl.py                 # ⭐ 24 条 RL 回归测试
│  ├─ api/ services/ mcp_protocol/ skill_cli/   # Web 应用（RL 路径不依赖）
│  └─ requirements.txt
├─ traces/train/                     # Trajectory 存储（JSONL，运行时生成）
├─ data/
│  ├─ sft/sft_train.jsonl            # SFT 数据（207 条）
│  ├─ dpo/dpo_train.jsonl            # DPO 偏好对（97 条）
│  ├─ grpo/grpo_train.jsonl          # GRPO 数据（207 条）
│  ├─ mock_arxiv_snapshot.json       # MockEnv 快照（7 方向 × 50 篇真实论文）
│  └─ raw_data.csv / summary.json / report.md   # 原 benchmark 实验数据
├─ eval/
│  └─ evaluate.py                    # held-out 评估 + 多检查点对比
├─ scripts/
│  ├─ generate_sft_data.py
│  ├─ generate_dpo_data.py
│  └─ generate_grpo_data.py
├─ outputs/                          # 训练产物（sft/dpo/grpo 各自 final/）
├─ docs/rl_building.md               # 原改造计划
├─ documents/                        # 代码审阅与复现报告
└─ README.md
```

> **后端切换**：`STORE_BACKEND=memory|mysql|auto`（默认 auto —— 有 sqlalchemy 走 MySQL，
> 没有则自动回落内存）。RL 入口已强制 memory，因此 `pip install -r requirements.txt`
> 装的那些依赖就足够跑完整条链路，不需要 fastapi/sqlalchemy/pymysql。

## 🔬 使用示例

### 1. Rollout（收集 trajectory）

```bash
cd AgenticArxiv

# 单个任务
python -m rl.rollout search_01 ../traces/train/

# 批量 rollout
python -m rl.rollout --all ../traces/train/
```

### 2. 训练流程（SFT → DPO → GRPO）

```bash
# Step 1: 生成 SFT 数据
python scripts/generate_sft_data.py

# Step 2: SFT 训练
python -m rl.train_sft

# Step 3: 生成 DPO 数据（需要 SFT 模型）
python scripts/generate_dpo_data.py

# Step 4: DPO 训练
python -m rl.train_dpo

# Step 5: GRPO 训练
python -m rl.train_grpo
```

### 3. Reward 计算测试

```python
from rl.reward import RewardCalculator
from benchmark.tasks import get_task_by_id

task_def = get_task_by_id('search_01')
# 构造一个 mock result
result = {
    'history': [
        {'thought': '...', 'action': '...', 'observation': '...'},
        {'thought': '...', 'action': 'FINISH', 'observation': '...'},
    ],
    'timing': {...},
    'token_usage': {...},
    'iteration_count': 2,
}

reward_calc = RewardCalculator()
reward, metrics = reward_calc.compute_reward(task_def, result)
print(f'Reward: {reward:.2f}')  # 期望: ~1.5
```

---

## 🧪 测试任务集

来自 `benchmark/tasks.py`，包含 7 个任务：

| ID | 任务 | 类型 | 预期工具 |
|----|------|------|---------|
| `search_01` | 检索最近7天AI论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `search_02` | 获取最近3天ML论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `search_03` | 搜索最近7天NLP论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `download_01` | 下载第1篇论文PDF | 下载 | `download_arxiv_pdf` |
| `translate_01` | 翻译第1篇论文 | 翻译 | `translate_arxiv_pdf` |
| `cache_01` | 查看第1篇论文缓存状态 | 缓存 | `get_paper_cache_status` |
| `composite_01` | 搜索+下载 | 复合 | `get_recently_submitted_cs_papers`, `download_arxiv_pdf` |

---

## 📊 指标监控

### Reward 曲线

使用 TensorBoard 或 wandb 监控：
```bash
tensorboard --logdir ./outputs/grpo/logs
```

### 关键指标

| 指标 | 说明 | 目标 |
|------|------|------|
| `reward` | 平均奖励 | ↑ 上升 |
| `kl_div` | KL 散度（vs reference model） | ↔ 稳定（不过大） |
| `task_completed_rate` | 任务成功率 | ↑ 上升 |
| `tool_call_accurate_rate` | 工具调用准确率 | ↑ 上升 |
| `parse_failures` | 解析失败次数 | ↓ 下降 |
| `tool_exec_failures` | 工具执行失败次数 | ↓ 下降 |

---

## 🛡️ 依赖说明

**核心依赖**（`requirements.txt`）：
```txt
torch>=2.0.0
transformers>=4.35.0
trl>=0.8.0                # TRL (SFT/DPO/GRPO/PPO)
datasets>=2.14.0
accelerate>=0.25.0
arxiv
requests
python-dotenv
loguru
pydantic>=2.0
fire
```

**不再需要**（已去除）：
- `fastapi`、`uvicorn`（无 Web 服务）
- `sqlalchemy`、`pymysql`（改用 JSONL）
- `pdf2zh`（训练时用 mock）

---

## 🔗 相关资源

### 官方文档
- [TRL 文档](https://huggingface.co/docs/trl/)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)
- [GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)

### 论文
- **InstructGPT** (OpenAI, 2022)：RLHF 三阶段（SFT → RM → PPO）
- **DPO** (Stanford, 2023)：直接偏好优化
- **RLVR**：Reinforcement Learning with Verifiable Reward

### 原 AgenticArXiv（Web 应用版）
本项目基于 [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv) 改造，原版包含：
- FastAPI 后端 + Vue3 前端
- 三种 Agent 架构（ReAct/MCP/Skill）
- 实时 SSE 推送、MySQL 存储、PDF 翻译服务

这些功能已归档到 `archive/`。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发建议
1. Fork 本仓库
2. 创建 feature 分支：`git checkout -b feature/your-feature`
3. 提交改动：`git commit -m "feat: add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

---

## 📄 License

MIT License

---

## 🙋 FAQ

### Q: 与原 AgenticArXiv 的区别？

| 维度 | 原版 AgenticArXiv | 本项目 (AgenticArXiv-RL) |
|------|------------------|-------------------------|
| **定位** | 生产级 arXiv 应用 | RL 训练研究环境 |
| **架构** | FastAPI + Vue3 + MySQL | 纯 Python + JSONL |
| **Agent 模式** | 3 种（ReAct/MCP/Skill） | 仅 ReAct（精简） |
| **核心功能** | 实时翻译、SSE、Web UI | SFT/DPO/GRPO 训练 |
| **依赖** | 重（14+ 包） | 轻（8 核心包） |

### Q: 为什么只保留 ReAct，归档 MCP/Skill？

RL 训练专注单一策略（ReAct 正则解析），MCP/Skill 增加复杂度但不改变核心逻辑。

### Q: 为什么改用 JSONL 而非 MySQL？

- **可移植性**：JSONL 无需数据库依赖
- **轻量级**：更适合 RL 训练的离线场景
- **TRL 兼容**：TRL 数据集直接支持 JSONL

### Q: 为什么选 GRPO 不用 PPO？

GRPO 更适合轻量级学习项目：
- ✅ 无需额外 value model（显存/训练开销更小）
- ✅ 适合小模型（如 Qwen2.5-1.5B）
- ✅ 实现简单，调试容易

PPO 更适合生产级大模型训练（7B+），本项目作为学习 demo 不涉及。

---

**开始你的 Agentic RL 训练之旅！** 🚀
