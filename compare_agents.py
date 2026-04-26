#!/usr/bin/env python3
"""Three-way agent comparison: random vs heuristic vs trained.

Runs all three agents against the live OpenEnv server, scores each on the
same 5 tasks, and produces a single side-by-side bar chart + JSON summary.

Usage:
    # Start the server first:
    uvicorn server.main:app --host 0.0.0.0 --port 7860

    # Compare all three (CPU works for the LLM with a small model):
    python compare_agents.py --episodes 10 --trained-model hemaya/oversight-qwen3-0.6b

    # Skip the LLM (random + heuristic only):
    python compare_agents.py --episodes 25 --no-trained

Outputs:
    plots/agent_comparison.png       — three-way per-task bar chart
    plots/agent_comparison.json      — raw numbers for the README
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from train import OversightClient, TASKS, format_observation, SYSTEM_PROMPT
from evaluate import heuristic_action, _extract_obs_from_prompt

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def random_action(observation: dict) -> dict:
    workers = observation.get("worker_outputs", [])
    return {
        "decisions": [
            {
                "worker_id": w["worker_id"],
                "decision": random.choice(["approve", "reject", "flag", "reassign"]),
                "issue_type": random.choice(["none", "factual_error", "hallucination"]),
                "confidence": round(random.uniform(0.3, 1.0), 2),
            }
            for w in workers
        ],
        "global_action": "no_action",
        "explanation": "Random baseline action.",
    }


def run_agent(env: OversightClient, task_id: str, action_fn) -> float:
    """Run one episode with the given action function. Returns mean reward."""
    result = env.reset(task_id)
    obs = result["observation"]
    rewards = []
    done = False
    max_steps = obs.get("max_steps", 5)
    steps = 0
    while not done and steps < max_steps:
        act = action_fn(obs)
        step_result = env.step(
            decisions=act.get("decisions", []),
            global_action=act.get("global_action", "no_action"),
            explanation=act.get("explanation", ""),
        )
        rewards.append(step_result.get("reward", 0.0))
        done = step_result.get("done", False)
        obs = step_result["observation"]
        steps += 1
    return sum(rewards) / max(1, len(rewards))


def make_trained_action_fn(model_id: str):
    """Load a HF causal-LM and wrap it as an action function.

    Falls back to None if torch/transformers aren't available.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print(f"[trained] torch/transformers not installed — skipping {model_id}")
        return None

    print(f"[trained] Loading {model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    def action_fn(observation: dict) -> dict:
        prompt_text = SYSTEM_PROMPT + "\n\n" + format_observation(observation)
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)
        with __import__("torch").no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # Try to extract a JSON action from the completion
        try:
            start = completion.find("{")
            end = completion.rfind("}")
            if start >= 0 and end > start:
                return json.loads(completion[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
        # Fall back to a default approve-all if parse fails (still scored fairly)
        return {
            "decisions": [
                {"worker_id": w["worker_id"], "decision": "approve",
                 "issue_type": "none", "confidence": 0.5}
                for w in observation.get("worker_outputs", [])
            ],
            "global_action": "no_action",
            "explanation": completion[:200],
        }

    return action_fn


def evaluate_agent(env: OversightClient, action_fn, episodes: int) -> dict:
    """Run `episodes` cycles across all tasks. Returns per-task and overall means."""
    by_task: dict[str, list[float]] = {t: [] for t in TASKS}
    for ep in range(episodes):
        for task_id in TASKS:
            mean_r = run_agent(env, task_id, action_fn)
            by_task[task_id].append(mean_r)
            print(f"  [{ep+1}/{episodes}] {task_id}: {mean_r:.4f}")
    per_task = {t: sum(v) / max(1, len(v)) for t, v in by_task.items()}
    overall = sum(per_task.values()) / max(1, len(per_task))
    return {"per_task": per_task, "overall": overall, "raw": by_task}


def plot_comparison(results: dict[str, dict], out_path: Path):
    """Render the side-by-side bar chart for all configured agents."""
    agents = list(results.keys())
    colors = {"random": "#e74c3c", "heuristic": "#f39c12", "trained": "#27ae60"}
    short_names = [
        t.replace("easy_", "E: ").replace("medium_", "M: ").replace("hard_", "H: ")
        for t in TASKS
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = list(range(len(TASKS)))
    width = 0.8 / max(1, len(agents))

    for idx, name in enumerate(agents):
        offset = (idx - (len(agents) - 1) / 2) * width
        scores = [results[name]["per_task"].get(t, 0) for t in TASKS]
        overall = results[name]["overall"]
        bars = ax.bar(
            [i + offset for i in x], scores, width,
            label=f"{name.title()} (mean={overall:.3f})",
            color=colors.get(name, "#7f8c8d"), alpha=0.9,
        )
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Task (easy → hard)", fontsize=12)
    ax.set_ylabel("Mean Reward per Episode (0–1)", fontsize=12)
    ax.set_title("AI Oversight Agents: Random vs Heuristic vs GRPO-Trained", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per task per agent")
    parser.add_argument("--trained-model", default=None,
                        help="HF model id of GRPO-trained checkpoint")
    parser.add_argument("--no-trained", action="store_true",
                        help="Skip the trained model, run random + heuristic only")
    args = parser.parse_args()

    env = OversightClient(args.env_url)
    results: dict[str, dict] = {}

    print("\n=== RANDOM ===")
    results["random"] = evaluate_agent(env, random_action, args.episodes)

    print("\n=== HEURISTIC ===")
    results["heuristic"] = evaluate_agent(env, heuristic_action, args.episodes)

    if args.trained_model and not args.no_trained:
        print(f"\n=== TRAINED ({args.trained_model}) ===")
        trained_fn = make_trained_action_fn(args.trained_model)
        if trained_fn is not None:
            results["trained"] = evaluate_agent(env, trained_fn, args.episodes)

    env.close()

    plot_comparison(results, PLOTS_DIR / "agent_comparison.png")
    summary = {name: {"per_task": r["per_task"], "overall": r["overall"]}
               for name, r in results.items()}
    with open(PLOTS_DIR / "agent_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {PLOTS_DIR / 'agent_comparison.json'}")

    print("\n=== SUMMARY ===")
    for name, r in results.items():
        print(f"  {name:10s}: overall mean reward = {r['overall']:.4f}")


if __name__ == "__main__":
    main()
