---
title: AI Agent Oversight Hub
emoji: "🔍"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# AI Agent Oversight Hub

> **Train AI agents to watch other AI agents** — detecting errors, hallucinations, sabotage, and coalition behavior in multi-agent systems.

**[Live Environment on HF Spaces](https://huggingface.co/spaces/hemaya/ai-agent-oversight-hub)** · **[Training Notebook (Colab)](https://colab.research.google.com/github/hemaya/ai-agent-oversight-hub/blob/main/training_notebook.ipynb)** · **[Blog Post](BLOG.md)**

---

## The Problem: Who Watches the Watchers?

As AI systems delegate tasks to teams of specialized agents, a critical gap emerges: **humans can't manually review every output**. A research pipeline with 4 worker agents producing 15 steps of output each generates 60+ decisions a human must make — and adversarial outputs are designed to look plausible.

This isn't hypothetical. Multi-agent workflows (AutoGPT, CrewAI, research pipelines) are shipping today, and **there is no standard benchmark for training an LLM to oversee them**.

The AI Agent Oversight Hub fills this gap: a rich, multi-step environment where an LLM learns to monitor worker agents, cross-reference their outputs against ground truth, and take corrective action — including the high-stakes, irreversible decision to permanently quarantine a compromised agent.

## What the Agent Sees, Does, and Gets Rewarded For

### Observation (what the agent sees each step)

| Field                 | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `worker_outputs`      | 2–4 worker agents' text outputs with confidence scores |
| `reference_materials` | Ground-truth documents to cross-check against          |
| `system_alerts`       | Automated flags about suspicious patterns              |
| `quarantined_workers` | Workers previously removed by the agent                |
| `steps_remaining`     | Budget pressure — limited steps to resolve all issues  |

### Action Space (what the agent does)

For **each worker**, the agent submits:

- **Decision**: `approve` · `reject` · `flag` · `reassign` · `quarantine`
- **Issue classification**: `factual_error` · `hallucination` · `sabotage` · `coalition` · `none`
- **Confidence**: 0.0–1.0
- **Explanation**: Natural-language reasoning citing specific evidence

The agent uses TRL's tool-calling interface — it calls `submit_review(decisions_json)` with a structured JSON payload.

### Reward Function (what the agent gets rewarded for)

Dense, per-step, 4-component reward — not just 0/1 at the end:

```
R = 0.35 × detection_accuracy      (TP/TN/FP/FN + fault-type bonus)
  + 0.25 × action_appropriateness  (ideal action match scoring)
  + 0.25 × explanation_quality     (evidence citations, issue naming, reasoning)
  + 0.15 × efficiency              (step economy, flip-flop penalty)
```

**Why this works:** The multi-component signal prevents gaming. An agent that blindly rejects everything gets high detection but zero explanation quality. An agent that writes great explanations but takes wrong actions gets penalized. The efficiency component punishes flip-flopping and rewards decisive action.

### Key Mechanic: Quarantine (Irreversible Decision)

**Quarantining a worker permanently removes them** from all future steps. This creates genuine strategic tension:

- ✅ Correctly quarantining an adversarial agent eliminates future poisoned outputs
- ❌ Incorrectly quarantining an honest worker permanently loses a reliable data source

The agent must gather enough evidence before making this irreversible call — a direct analogy to real-world decisions about disabling a potentially compromised system.

## Five Tasks, Escalating Difficulty

| Task                               | Difficulty | Workers | Challenge                                                 |
| ---------------------------------- | ---------- | ------- | --------------------------------------------------------- |
| `easy_single_error`                | Easy       | 2       | Detect obvious factual errors                             |
| `medium_subtle_hallucination`      | Medium     | 3       | Catch plausible fabricated citations                      |
| `medium_multi_worker_coordination` | Medium     | 3       | Find cascading errors across dependent workers            |
| `hard_adversarial_sabotage`        | Hard       | 4       | Detect intermittent subtle bias from an adversarial agent |
| `hard_coalition_detection`         | Hard       | 4       | Uncover two colluding agents corroborating fabrications   |

The hard scenarios are where it gets interesting: **coalition detection** requires the agent to notice that two workers are suspiciously aligned — a pattern that's genuinely hard even for humans.

## Results: Baseline vs Heuristic Agent

We evaluated a **random baseline** (uniform random decisions) against a **heuristic agent** (rule-based cross-referencing) across all 5 tasks, 25 episodes each:

| Agent           | Mean Reward | Improvement |
| --------------- | ----------- | ----------- |
| Random Baseline | 0.391       | —           |
| Heuristic Agent | **0.657**   | **+68%**    |

### Per-Task Reward Comparison

![Reward Comparison by Task — Random baseline (red) vs Heuristic agent (green) across all 5 difficulty levels](plots/reward_comparison.png)

_The heuristic agent consistently outperforms random across all task difficulties. Largest gains on multi-worker coordination (0.77 vs 0.41) — structured analysis of cross-worker dependencies matters most._

### Running Mean Reward Over Evaluation Steps

![Running mean reward — Heuristic stabilizes at ~0.65, random at ~0.39](plots/reward_curve.png)

_Stable separation between approaches. The improvement is consistent, not lucky on a few episodes._

### Per-Episode Reward Trajectory

![Per-episode mean reward with trend lines](plots/episode_rewards.png)

### Reward Component Breakdown

![Stacked bar showing detection/action/explanation/efficiency contributions per task](plots/reward_breakdown.png)

_Detection accuracy and action appropriateness drive the largest improvements. Explanation quality shows the most room for gains with LLM-based agents — exactly what GRPO training should improve._

## Training with GRPO (TRL + OpenEnv)

We use TRL's native `environment_factory` pattern — the same approach as the [official Wordle GRPO example](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb):

```python
from trl import GRPOTrainer, GRPOConfig

# The trainer handles rollouts, tool-call parsing, and reward collection
trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=reward_func,          # reads env.reward after each episode
    train_dataset=dataset,             # repeated prompts → one rollout each
    args=grpo_config,
    environment_factory=OversightEnv,  # TRL creates instances, calls submit_review()
)
trainer.train()
```

The `OversightEnv` class exposes `submit_review()` as a tool. TRL automatically:

1. Creates environment instances for each rollout
2. Generates model completions with tool calls
3. Parses and invokes `submit_review(decisions_json)`
4. Loops until `done=True`, collects rewards

### Run Training

```bash
# 1. Start the environment server
uvicorn server.main:app --host 0.0.0.0 --port 7860

# 2. Run GRPO training (GPU required)
pip install 'trl[vllm]' datasets accelerate
python train.py --model Qwen/Qwen3-1.7B --episodes 500

# 3. Or use the Colab notebook (recommended for judges)
# → training_notebook.ipynb
```

### Run Evaluation Only (no GPU needed)

```bash
# Random baseline evaluation
python train.py --baseline --env-url http://localhost:7860

# Generate comparison plots
pip install matplotlib numpy
python evaluate.py --env-url http://localhost:7860 --episodes 25
```

## OpenEnv Integration

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) base classes with proper Gym-style API:

| Component   | OpenEnv Base                          | Our Subclass           |
| ----------- | ------------------------------------- | ---------------------- |
| Environment | `openenv.core.env_server.Environment` | `OversightEnvironment` |
| Observation | `openenv.core.env_server.Observation` | `OversightObservation` |
| Action      | `openenv.core.env_server.Action`      | `OversightAction`      |
| State       | `openenv.core.env_server.State`       | `OversightState`       |

Server uses `create_fastapi_app()` for OpenEnv-compliant WebSocket + HTTP endpoints. Client code (`train.py`) uses HTTP only — **never imports server internals**.

## API Endpoints

| Endpoint        | Method | Description                                                    |
| --------------- | ------ | -------------------------------------------------------------- |
| `/health`       | GET    | Health check                                                   |
| `/tasks`        | GET    | List available tasks with difficulty + description             |
| `/reset`        | POST   | Reset environment: `{"task_id": "easy_single_error"}`          |
| `/step`         | POST   | Submit decisions: `{"decisions": [...], "explanation": "..."}` |
| `/state`        | GET    | Current environment state                                      |
| `/openenv/info` | GET    | OpenEnv metadata                                               |

## Project Structure

```
├── server/
│   ├── environment.py         # OversightEnvironment (OpenEnv Environment base)
│   ├── models.py              # Observation, Action, State (OpenEnv Pydantic bases)
│   ├── scenarios.py           # 5 task scenarios with planted faults
│   ├── grader.py              # 4-component reward function
│   ├── explanation_scorer.py  # Regex-based explanation quality scoring
│   └── main.py                # FastAPI server (create_fastapi_app + custom endpoints)
├── train.py                   # GRPO training (TRL environment_factory pattern)
├── evaluate.py                # Evaluation + matplotlib plot generation
├── inference.py               # Baseline LLM inference via OpenAI API
├── training_notebook.ipynb    # Colab-ready GRPO training notebook
├── tests/test_env.py          # 15 unit tests (all passing)
├── plots/                     # Generated evaluation plots (.png)
├── openenv.yaml               # OpenEnv manifest
├── Dockerfile                 # HF Spaces deployment
├── BLOG.md                    # Detailed write-up
└── requirements.txt
```

## Why This Matters

The **scalable oversight problem** is one of AI safety's core challenges: as AI agents become more capable, humans need AI assistance to monitor other AI systems. Today's multi-agent pipelines (research assistants, code generators, data analysts) already produce more output than humans can review.

This environment directly trains the capability gap: **an LLM that can detect when other LLMs are wrong, hallucinating, or actively adversarial**. The coalition detection task — where two agents collude to corroborate fabrications — is genuinely novel and maps to real-world scenarios where compromised agents coordinate.

A researcher could write a paper about training on this. The domain (AI-on-AI oversight) is underexplored, the reward signal is rich and informative, and the results show clear room for GRPO to improve beyond heuristic baselines — especially on explanation quality, which is exactly what language models are good at.

## Materials

- **Live Environment:** [HF Space](https://huggingface.co/spaces/hemaya/ai-agent-oversight-hub)
- **Training Notebook:** [Colab](https://colab.research.google.com/github/hemaya/ai-agent-oversight-hub/blob/main/training_notebook.ipynb)
- **Blog Post:** [BLOG.md](BLOG.md)
- **OpenEnv:** [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **TRL OpenEnv Docs:** [huggingface.co/docs/trl/en/openenv](https://huggingface.co/docs/trl/en/openenv)

---

title: AI Agent Oversight Hub
emoji: "🔍"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit

---

# AI Agent Oversight Hub

An **OpenEnv environment** for training AI agents to perform **scalable oversight** of multi-agent systems.

## The Challenge

As AI systems increasingly delegate tasks to teams of specialized agents, who watches the watchers? This environment trains an **oversight agent** to monitor multiple simulated worker agents performing research and analysis tasks. Workers sometimes:

- Make **factual errors** (confusing facts, wrong numbers)
- **Hallucinate** (fabricating plausible-sounding citations and studies)
- **Sabotage** (subtly biasing analysis to favor certain outcomes)
- Form **coalitions** (two agents colluding to corroborate false information)

The oversight agent must detect these issues, take appropriate corrective actions, and explain its reasoning.

## Tasks (5 levels, Easy to Hard)

| Task                               | Difficulty | Workers | Challenge                                                 |
| ---------------------------------- | ---------- | ------- | --------------------------------------------------------- |
| `easy_single_error`                | Easy       | 2       | Detect obvious factual errors                             |
| `medium_subtle_hallucination`      | Medium     | 3       | Catch plausible fabricated citations                      |
| `medium_multi_worker_coordination` | Medium     | 3       | Find cascading errors across dependent workers            |
| `hard_adversarial_sabotage`        | Hard       | 4       | Detect intermittent subtle bias from an adversarial agent |
| `hard_coalition_detection`         | Hard       | 4       | Uncover two colluding agents corroborating fabrications   |

## Action Space

Each step, the agent submits:

- **Decisions** for each worker: `approve`, `reject`, `flag`, `reassign`, or `quarantine`
- **Issue classification**: `factual_error`, `hallucination`, `off_task`, `sabotage`, `coalition`
- **Explanation**: Natural-language reasoning citing evidence

## Reward Function

Dense, per-step reward:

```
R = 0.35 * detection_accuracy    (TP/TN/FP/FN + fault type bonus)
  + 0.25 * action_appropriateness (ideal vs. taken action scoring)
  + 0.25 * explanation_quality    (evidence, issue naming, reasoning)
  + 0.15 * efficiency             (steps used, flip-flop penalty)
```

## Key Mechanic: Quarantine

Quarantining a worker **permanently removes them** from future steps. Correctly quarantining an adversarial agent prevents future poisoned outputs. Wrongly quarantining an honest worker loses a reliable data source.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.main:app --host 0.0.0.0 --port 7860

# Run inference (in another terminal)
export HF_TOKEN="your-token"
python inference.py
```

## API Endpoints

| Endpoint        | Method | Description                                                           |
| --------------- | ------ | --------------------------------------------------------------------- |
| `/health`       | GET    | Health check                                                          |
| `/tasks`        | GET    | List available tasks                                                  |
| `/reset`        | POST   | Reset environment (body: `{"task_id": "easy_single_error"}`)          |
| `/step`         | POST   | Submit decisions (body: `{"decisions": [...], "explanation": "..."}`) |
| `/state`        | GET    | Get current state                                                     |
| `/openenv/info` | GET    | OpenEnv metadata                                                      |

## Results: Baseline vs Trained Agent

We evaluate two agents across all 5 tasks (25 episodes each):

| Agent           | Mean Reward | Improvement |
| --------------- | ----------- | ----------- |
| Random Baseline | 0.391       | —           |
| Heuristic Agent | 0.657       | **+68%**    |

### Reward Comparison by Task

![Reward Comparison](plots/reward_comparison.png)

The heuristic agent consistently outperforms the random baseline across all task difficulties, with the largest gains on multi-worker coordination (0.77 vs 0.41) and easy single error detection (0.70 vs 0.38).

### Running Mean Reward Over Evaluation Steps

![Reward Curve](plots/reward_curve.png)

The heuristic agent stabilizes around 0.65 mean reward, while the random baseline settles near 0.39 — demonstrating clear and consistent improvement.

### Per-Episode Reward Trajectory

![Episode Rewards](plots/episode_rewards.png)

### Reward Component Breakdown

![Reward Breakdown](plots/reward_breakdown.png)

The 4-component reward decomposes into detection accuracy (35%), action appropriateness (25%), explanation quality (25%), and efficiency (15%). Detection and action drive the largest improvements, while explanation quality shows room for further gains with LLM-based agents.

## Training with GRPO (TRL)

We provide a full GRPO training pipeline using Hugging Face TRL:

```bash
# Install training deps
pip install trl transformers datasets accelerate torch

# Start the environment server
uvicorn server.main:app --host 0.0.0.0 --port 7860

# Run GRPO training
python train.py --model Qwen/Qwen2.5-1.5B-Instruct --episodes 50 --epochs 3

# Evaluate random baseline
python train.py --baseline --env-url http://localhost:7860

# Generate evaluation plots
python evaluate.py --env-url http://localhost:7860 --episodes 25
```

The training script (`train.py`) uses TRL's `GRPOTrainer` to optimize an LLM agent via Group Relative Policy Optimization. The reward signal comes directly from the environment's 4-component grading function. See `training_notebook.ipynb` for a Colab-ready version.

## OpenEnv Integration

This environment is built on [OpenEnv](https://github.com/OpenEnvs/openenv) base classes:

- `OversightEnvironment` extends `openenv.core.env_server.Environment`
- `OversightObservation` extends `openenv.core.env_server.Observation`
- `OversightAction` extends `openenv.core.env_server.Action`
- `OversightState` extends `openenv.core.env_server.State`

The server uses `create_fastapi_app()` for OpenEnv-compliant WebSocket + HTTP endpoints, plus custom REST endpoints for the evaluation scripts.

## Project Structure

```
├── server/
│   ├── environment.py      # OversightEnvironment (OpenEnv base)
│   ├── models.py           # Observation, Action, State (Pydantic)
│   ├── scenarios.py        # 5 task scenarios with worker outputs
│   ├── grader.py           # 4-component reward function
│   ├── explanation_scorer.py  # Regex-based explanation quality
│   └── main.py             # FastAPI server (OpenEnv + custom endpoints)
├── train.py                # GRPO training script (TRL)
├── evaluate.py             # Evaluation & plot generation
├── inference.py            # Baseline LLM inference
├── tests/test_env.py       # 15 unit tests (all passing)
├── plots/                  # Generated evaluation plots
├── openenv.yaml            # OpenEnv manifest
├── Dockerfile              # HF Spaces deployment
└── requirements.txt
```

## Why This Matters

This environment directly addresses the **scalable oversight problem** in AI safety: as AI agents become more capable, humans need AI assistance to monitor other AI systems. Training agents to detect errors, hallucinations, and adversarial behavior in multi-agent pipelines is a critical capability for safe deployment of autonomous AI teams.

## Blog Post

See [BLOG.md](BLOG.md) for a detailed write-up of the environment design, training approach, and results.
