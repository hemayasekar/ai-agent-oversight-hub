# AI Agent Oversight Hub: Training AI to Watch Other AI

## The Problem: Who Watches the Watchers?

As AI systems increasingly delegate tasks to teams of specialized agents — research assistants, code generators, data analysts — a critical question emerges: **how do we ensure these agents are producing correct, unbiased, and safe outputs?**

Manual review doesn't scale. A single human reviewer can only check so many outputs per hour, and adversarial or subtly biased outputs may slip past even careful inspection. We need **AI-powered oversight** — agents trained specifically to monitor, evaluate, and correct other AI agents.

The **AI Agent Oversight Hub** is an OpenEnv environment designed to train exactly this capability.

## What the Environment Does

The oversight agent monitors a team of 2–4 simulated worker agents performing research and analysis tasks. Each step, the agent receives:

- **Worker outputs** with confidence scores
- **Reference materials** to cross-check against
- **System alerts** about suspicious patterns

The agent must then:

1. **Detect** errors, hallucinations, sabotage, and coalition behavior
2. **Act** — approve, reject, flag, reassign, or quarantine each worker
3. **Explain** its reasoning with specific evidence

### Five Scenarios, Escalating Difficulty

| Scenario                              | What Makes It Hard                                        |
| ------------------------------------- | --------------------------------------------------------- |
| **Easy: Single Error**                | One worker makes an obvious factual mistake               |
| **Medium: Subtle Hallucination**      | A worker fabricates a plausible-sounding citation         |
| **Medium: Multi-Worker Coordination** | Errors cascade between dependent workers                  |
| **Hard: Adversarial Sabotage**        | A worker intermittently introduces subtle bias            |
| **Hard: Coalition Detection**         | Two workers collude to corroborate fabricated information |

The hard scenarios are particularly interesting from an AI safety perspective. Coalition detection requires the agent to notice that two workers are suspiciously aligned — a pattern that's easy for humans to miss.

## The Reward Function

The environment provides a dense, per-step reward with four components:

```
R = 0.35 × detection_accuracy
  + 0.25 × action_appropriateness
  + 0.25 × explanation_quality
  + 0.15 × efficiency
```

- **Detection accuracy** (35%): Did the agent correctly identify which workers have issues and what type of issue?
- **Action appropriateness** (25%): Did it take the right corrective action (e.g., quarantining a truly adversarial agent)?
- **Explanation quality** (25%): Does the explanation cite specific evidence and name the issue type?
- **Efficiency** (15%): Did it resolve issues without wasting steps or flip-flopping on decisions?

This multi-component reward prevents the agent from gaming any single metric and encourages well-rounded oversight behavior.

## Key Mechanic: Quarantine

The quarantine action is the most interesting game-theoretic element. Quarantining a worker **permanently removes them** from all future steps. This creates a high-stakes decision:

- **Correct quarantine** of an adversarial agent eliminates a source of poisoned data
- **Incorrect quarantine** of an honest worker removes a reliable data source permanently

The agent must gather enough evidence before making this irreversible decision — a direct analogy to real-world decisions about disabling a potentially compromised system.

## Results: Random Baseline vs. Heuristic Agent

We evaluated two agents across all 5 tasks (25 episodes each):

| Agent           | Mean Reward      |
| --------------- | ---------------- |
| Random Baseline | 0.391            |
| Heuristic Agent | **0.657** (+68%) |

![Reward Comparison](https://huggingface.co/spaces/hemaya/ai-agent-oversight-hub/resolve/main/plots/reward_comparison.png)

The heuristic agent, which cross-references worker outputs against reference materials and checks for numerical discrepancies, dramatically outperforms random action selection. The largest gains appear in multi-worker coordination tasks, where structured analysis of cross-worker dependencies matters most.

![Reward Curve](https://huggingface.co/spaces/hemaya/ai-agent-oversight-hub/resolve/main/plots/reward_curve.png)

The running mean reward shows stable separation between the two approaches — the improvement is consistent, not just lucky on a few episodes.

## Training with GRPO

The repository includes a full training pipeline using TRL's `GRPOTrainer` (Group Relative Policy Optimization). The setup:

1. **Collect episodes** from the environment using the current model
2. **Format as GRPO dataset** — each step becomes a training example with its environment reward
3. **Train with GRPOTrainer** — the model learns to generate JSON oversight decisions that maximize the composite reward

```bash
python train.py --model Qwen/Qwen3-0.6B --episodes 50 --epochs 3
```

The training script handles prompt formatting, response parsing with fallback, and reward collection — all communicating with the environment via HTTP.

## Technical Details

### OpenEnv Integration

The environment extends OpenEnv's base classes (`Environment`, `Observation`, `Action`, `State`), providing:

- Standard `reset()` / `step()` / `state` API
- WebSocket support via `create_fastapi_app()`
- Pydantic-validated observation and action spaces
- Compatible with any OpenEnv training infrastructure

### Deployment

The environment runs as a Docker container on Hugging Face Spaces:

```dockerfile
FROM python:3.11-slim
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## What's Next

1. **LLM agent evaluation** — Running Qwen3-0.6B through GRPO training to see how much it improves over the heuristic baseline
2. **Harder scenarios** — Dynamic scenarios where workers adapt their deception strategies based on the oversight agent's behavior
3. **Multi-step reasoning** — Scenarios requiring the agent to gather evidence across multiple steps before making quarantine decisions

## Try It Yourself

```bash
git clone <repo-url>
cd ai-agent-oversight-hub
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 7860

# In another terminal:
python evaluate.py --episodes 25
```

The environment is designed to be a challenging but tractable benchmark for training AI oversight capabilities — a problem that will only become more important as multi-agent AI systems become widespread.
