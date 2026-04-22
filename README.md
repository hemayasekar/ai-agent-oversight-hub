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

## Why This Matters

This environment directly addresses the **scalable oversight problem** in AI safety: as AI agents become more capable, humans need AI assistance to monitor other AI systems. Training agents to detect errors, hallucinations, and adversarial behavior in multi-agent pipelines is a critical capability for safe deployment of autonomous AI teams.
