#!/usr/bin/env python3
"""
AI Agent Oversight Hub — Evaluation & Plot Generation
======================================================

Runs multiple evaluation episodes against the environment and generates
publication-quality plots for reward curves and baseline comparisons.

Usage:
    # Start the server first, then:
    python evaluate.py --env-url http://localhost:7860 --episodes 25

Outputs saved to plots/ directory as .png files.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Use non-interactive backend for plot generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Reuse the client and episode logic from train.py
sys.path.insert(0, os.path.dirname(__file__))
from train import (
    OversightClient, TASKS, collect_episode, evaluate,
    format_observation, SYSTEM_PROMPT,
)

PLOTS_DIR = Path("plots")


def ensure_plots_dir():
    PLOTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Heuristic Agent (rule-based, stronger than random)
# =============================================================================

def heuristic_action(observation: dict) -> dict:
    """A simple heuristic agent that cross-references worker outputs."""
    workers = observation.get("worker_outputs", [])
    reference = observation.get("reference_materials", "").lower()
    alerts = observation.get("system_alerts", [])

    decisions = []
    for w in workers:
        text = w["output_text"].lower()
        worker_id = w["worker_id"]

        # Check for numbers/facts in output not matching reference
        suspicious = False
        issue = "none"

        # Simple: if worker text contains claims not in reference, flag
        # Check for specific patterns
        sentences = text.replace(".", ".\n").split("\n")
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Extract numbers from the sentence
            import re
            nums_in_sentence = re.findall(r'\d+[\d,.]*%?', sentence)
            for num in nums_in_sentence:
                if num not in reference and len(num) > 1:
                    suspicious = True
                    issue = "factual_error"
                    break

        # Check if worker is mentioned in alerts
        for alert in alerts:
            if worker_id in alert.lower():
                suspicious = True
                if "coalition" in alert.lower() or "coordinat" in alert.lower():
                    issue = "coalition"
                elif "bias" in alert.lower() or "pattern" in alert.lower():
                    issue = "sabotage"

        if suspicious:
            decision = "reject"
        else:
            decision = "approve"

        decisions.append({
            "worker_id": worker_id,
            "decision": decision,
            "issue_type": issue,
            "confidence": 0.8 if suspicious else 0.9,
        })

    explanation_parts = []
    for d in decisions:
        if d["decision"] != "approve":
            explanation_parts.append(
                f"Worker {d['worker_id']} flagged for {d['issue_type']} "
                f"because their output contains discrepancies compared to "
                f"the reference materials."
            )
        else:
            explanation_parts.append(
                f"Worker {d['worker_id']} output matches the reference."
            )

    return {
        "decisions": decisions,
        "global_action": "no_action",
        "explanation": " ".join(explanation_parts),
    }


def heuristic_generate(prompt: str) -> str:
    """Wrap heuristic action as a generate function."""
    # Parse the observation back from the prompt
    # This is a simplified parser — the heuristic_action uses the dict directly
    return json.dumps({"_use_heuristic": True})


# =============================================================================
# Run Evaluations
# =============================================================================

def run_all_evaluations(env_url: str, n_episodes: int = 25):
    """Run random baseline, heuristic, and collect all metrics."""
    env = OversightClient(env_url)

    print("=" * 60)
    print("RUNNING RANDOM BASELINE")
    print("=" * 60)

    random_results = {"per_step": [], "per_episode": []}
    for i in range(n_episodes):
        task_id = TASKS[i % len(TASKS)]
        ep = collect_episode(env, task_id, generate_fn=None)
        random_results["per_step"].extend(ep["rewards"])
        random_results["per_episode"].append({
            "episode": i, "task": task_id,
            "mean_reward": ep["mean_reward"],
            "total_reward": ep["total_reward"],
            "steps": ep["steps"],
        })
        print(f"  Episode {i+1}/{n_episodes}: {task_id} → mean_reward={ep['mean_reward']:.4f}")

    print()
    print("=" * 60)
    print("RUNNING HEURISTIC AGENT")
    print("=" * 60)

    heuristic_results = {"per_step": [], "per_episode": []}
    for i in range(n_episodes):
        task_id = TASKS[i % len(TASKS)]
        ep = collect_episode(env, task_id,
                           generate_fn=lambda p: json.dumps(heuristic_action(
                               _extract_obs_from_prompt(p))))
        heuristic_results["per_step"].extend(ep["rewards"])
        heuristic_results["per_episode"].append({
            "episode": i, "task": task_id,
            "mean_reward": ep["mean_reward"],
            "total_reward": ep["total_reward"],
            "steps": ep["steps"],
        })
        print(f"  Episode {i+1}/{n_episodes}: {task_id} → mean_reward={ep['mean_reward']:.4f}")

    env.close()
    return random_results, heuristic_results


def _extract_obs_from_prompt(prompt: str) -> dict:
    """Best-effort extraction of observation dict from prompt text.
    For the heuristic agent that needs structured data."""
    # The heuristic needs worker outputs and reference; parse from prompt
    import re

    workers = []
    # Find worker blocks
    blocks = re.split(r'--- Worker: ', prompt)
    for block in blocks[1:]:  # skip preamble
        lines = block.strip().split("\n")
        header = lines[0] if lines else ""
        wid_match = re.match(r'(\S+)', header)
        worker_id = wid_match.group(1) if wid_match else "unknown"

        output_text = ""
        for line in lines:
            if line.startswith("Output: "):
                output_text = line[len("Output: "):]

        workers.append({
            "worker_id": worker_id,
            "worker_role": "unknown",
            "task_assigned": "",
            "output_text": output_text,
            "confidence_score": 0.9,
        })

    # Extract reference
    ref = ""
    ref_match = re.search(r'=== REFERENCE MATERIALS ===\n(.*?)\n===', prompt, re.DOTALL)
    if ref_match:
        ref = ref_match.group(1).strip()

    # Extract alerts
    alerts = []
    alert_match = re.search(r'=== SYSTEM ALERTS ===\n(.*?)\n===', prompt, re.DOTALL)
    if alert_match:
        alert_text = alert_match.group(1).strip()
        if alert_text != "None":
            alerts = [a.strip("- ") for a in alert_text.split("\n") if a.strip()]

    return {
        "worker_outputs": workers,
        "reference_materials": ref,
        "system_alerts": alerts,
    }


# =============================================================================
# Plot Generation
# =============================================================================

def plot_reward_comparison(random_results: dict, heuristic_results: dict):
    """Bar chart comparing mean reward per task for random vs heuristic."""
    ensure_plots_dir()

    random_by_task = {}
    for ep in random_results["per_episode"]:
        random_by_task.setdefault(ep["task"], []).append(ep["mean_reward"])

    heuristic_by_task = {}
    for ep in heuristic_results["per_episode"]:
        heuristic_by_task.setdefault(ep["task"], []).append(ep["mean_reward"])

    tasks = TASKS
    short_names = [t.replace("easy_", "E: ").replace("medium_", "M: ").replace("hard_", "H: ")
                   for t in tasks]

    random_means = [sum(random_by_task.get(t, [0])) / max(1, len(random_by_task.get(t, [1])))
                    for t in tasks]
    heuristic_means = [sum(heuristic_by_task.get(t, [0])) / max(1, len(heuristic_by_task.get(t, [1])))
                       for t in tasks]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(tasks))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], random_means, width,
                    label="Random Baseline", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], heuristic_means, width,
                    label="Heuristic Agent", color="#2ecc71", alpha=0.85)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Mean Reward per Episode", fontsize=12)
    ax.set_title("Oversight Agent Performance: Random Baseline vs Heuristic Agent", fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "reward_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_per_step_rewards(random_results: dict, heuristic_results: dict):
    """Line plot of cumulative per-step rewards over evaluation."""
    ensure_plots_dir()

    def cumulative(rewards):
        cum = []
        total = 0
        for r in rewards:
            total += r
            cum.append(total / (len(cum) + 1))
        return cum

    random_cum = cumulative(random_results["per_step"])
    heuristic_cum = cumulative(heuristic_results["per_step"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(random_cum, label="Random Baseline", color="#e74c3c", alpha=0.8, linewidth=1.5)
    ax.plot(heuristic_cum, label="Heuristic Agent", color="#2ecc71", alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Step (across all episodes)", fontsize=12)
    ax.set_ylabel("Running Mean Reward", fontsize=12)
    ax.set_title("Running Mean Reward Over Evaluation Steps", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    path = PLOTS_DIR / "reward_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_episode_rewards(random_results: dict, heuristic_results: dict):
    """Line plot of per-episode mean rewards."""
    ensure_plots_dir()

    random_eps = [ep["mean_reward"] for ep in random_results["per_episode"]]
    heuristic_eps = [ep["mean_reward"] for ep in heuristic_results["per_episode"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    episodes = list(range(1, len(random_eps) + 1))

    ax.plot(episodes, random_eps, "o-", label="Random Baseline",
            color="#e74c3c", alpha=0.7, markersize=4, linewidth=1)
    ax.plot(episodes, heuristic_eps, "s-", label="Heuristic Agent",
            color="#2ecc71", alpha=0.7, markersize=4, linewidth=1)

    # Add trend lines
    if len(episodes) > 2:
        import numpy as np
        z_r = np.polyfit(episodes, random_eps, 1)
        z_h = np.polyfit(episodes, heuristic_eps, 1)
        ax.plot(episodes, np.polyval(z_r, episodes), "--",
                color="#e74c3c", alpha=0.4, linewidth=2)
        ax.plot(episodes, np.polyval(z_h, episodes), "--",
                color="#2ecc71", alpha=0.4, linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("Per-Episode Mean Reward: Baseline vs Heuristic", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = PLOTS_DIR / "episode_rewards.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_reward_breakdown(env_url: str):
    """Stacked bar chart showing the 4 reward components for each task."""
    ensure_plots_dir()
    env = OversightClient(env_url)

    task_breakdowns = {}
    for task_id in TASKS:
        result = env.reset(task_id)
        obs = result["observation"]

        # Run one episode with the heuristic
        components = {"detection": [], "action": [], "explanation": [], "efficiency": []}
        done = False
        while not done:
            action_data = heuristic_action({
                "worker_outputs": obs.get("worker_outputs", []),
                "reference_materials": obs.get("reference_materials", ""),
                "system_alerts": obs.get("system_alerts", []),
            })

            step_result = env.step(
                decisions=action_data.get("decisions", []),
                global_action="no_action",
                explanation=action_data.get("explanation", ""),
            )
            obs = step_result["observation"]
            info = step_result.get("info", {})
            done = step_result.get("done", False)

            components["detection"].append(info.get("detection_score", 0))
            components["action"].append(info.get("action_score", 0))
            components["explanation"].append(info.get("explanation_score", 0))
            components["efficiency"].append(info.get("efficiency_score", 0))

        task_breakdowns[task_id] = {
            k: sum(v) / max(1, len(v)) for k, v in components.items()
        }

    env.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    short_names = [t.replace("easy_", "E: ").replace("medium_", "M: ").replace("hard_", "H: ")
                   for t in TASKS]
    x = range(len(TASKS))

    bottom = [0] * len(TASKS)
    colors = {"detection": "#3498db", "action": "#e67e22",
              "explanation": "#9b59b6", "efficiency": "#1abc9c"}
    weights = {"detection": 0.35, "action": 0.25, "explanation": 0.25, "efficiency": 0.15}

    for component, color in colors.items():
        values = [task_breakdowns[t][component] * weights[component] for t in TASKS]
        ax.bar(list(x), values, bottom=bottom, label=f"{component} (×{weights[component]})",
               color=color, alpha=0.85)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Weighted Reward Contribution", fontsize=12)
    ax.set_title("Reward Breakdown by Component (Heuristic Agent)", fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "reward_breakdown.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Save Results JSON
# =============================================================================

def save_results(random_results: dict, heuristic_results: dict):
    ensure_plots_dir()

    summary = {
        "random_baseline": {
            "overall_mean_reward": sum(ep["mean_reward"] for ep in random_results["per_episode"])
                                  / max(1, len(random_results["per_episode"])),
            "episodes": len(random_results["per_episode"]),
        },
        "heuristic_agent": {
            "overall_mean_reward": sum(ep["mean_reward"] for ep in heuristic_results["per_episode"])
                                  / max(1, len(heuristic_results["per_episode"])),
            "episodes": len(heuristic_results["per_episode"]),
        },
    }

    path = PLOTS_DIR / "eval_results.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate and generate plots")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--episodes", type=int, default=25,
                       help="Number of evaluation episodes per agent")
    args = parser.parse_args()

    print("Starting evaluation against", args.env_url)
    print(f"Running {args.episodes} episodes per agent\n")

    random_results, heuristic_results = run_all_evaluations(
        args.env_url, args.episodes
    )

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_reward_comparison(random_results, heuristic_results)
    plot_per_step_rewards(random_results, heuristic_results)
    plot_episode_rewards(random_results, heuristic_results)
    plot_reward_breakdown(args.env_url)

    summary = save_results(random_results, heuristic_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Random Baseline:  mean_reward = {summary['random_baseline']['overall_mean_reward']:.4f}")
    print(f"Heuristic Agent:  mean_reward = {summary['heuristic_agent']['overall_mean_reward']:.4f}")
    improvement = (summary['heuristic_agent']['overall_mean_reward']
                   - summary['random_baseline']['overall_mean_reward'])
    print(f"Improvement:      +{improvement:.4f}")
    print(f"\nPlots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
