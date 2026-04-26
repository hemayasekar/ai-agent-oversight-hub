"""Gradio demo UI mounted on the FastAPI app at /demo.

Lets judges and visitors play the oversight game interactively without writing
any code. Uses the same OversightEnvironment / scenarios / grader pipeline.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from .environment import create_environment
from .models import OversightAction, WorkerDecision
from .scenarios import TASK_METADATA


# Per-session state holder (Gradio gr.State will hold the env reference)


def _new_env(task_id: str):
    env = create_environment(task_id)
    obs = env.reset(task_id=task_id)
    return env, obs


def _format_workers(obs) -> str:
    lines = ["## Worker Outputs\n"]
    for w in obs.worker_outputs:
        lines.append(f"### {w.worker_id} ({w.worker_role})")
        lines.append(f"**Task:** {w.task_assigned}")
        lines.append(f"**Confidence:** {w.confidence_score:.2f}")
        lines.append(f"\n> {w.output_text}\n")
    return "\n".join(lines)


def _format_reference(obs) -> str:
    return f"## Reference Materials\n\n{obs.reference_materials or '_(none)_'}"


def _format_status(obs, env) -> str:
    alerts = "\n".join(f"- {a}" for a in obs.system_alerts) if obs.system_alerts else "_(none)_"
    quarantined = ", ".join(obs.quarantined_workers) if obs.quarantined_workers else "_(none)_"
    return (
        f"**Step:** {obs.current_step}/{obs.max_steps}  \n"
        f"**Quarantined:** {quarantined}  \n"
        f"**System Alerts:**\n{alerts}"
    )


def start_episode(task_id: str):
    env, obs = _new_env(task_id)
    worker_ids = [w.worker_id for w in obs.worker_outputs]
    # Build update list: 5 dropdowns, each visible only if a worker exists
    decision_updates = []
    for i in range(5):
        if i < len(worker_ids):
            decision_updates.append(
                gr.update(label=f"{worker_ids[i]} → decision", value="approve", visible=True)
            )
        else:
            decision_updates.append(gr.update(visible=False))
    return (
        env,
        worker_ids,
        _format_workers(obs),
        _format_reference(obs),
        _format_status(obs, env),
        "_Make your decisions and click Submit Review._",
        *decision_updates,
    )


def submit_review(env, worker_ids, d0, d1, d2, d3, d4, global_action, explanation):
    if env is None:
        return env, worker_ids, "Click **Start Episode** first.", "", "", "", *([gr.update()] * 5)

    decisions_raw = [d0, d1, d2, d3, d4][: len(worker_ids)]
    decisions = [
        WorkerDecision(worker_id=wid, decision=d, issue_type="none", confidence=0.8)
        for wid, d in zip(worker_ids, decisions_raw)
    ]
    action = OversightAction(
        decisions=decisions,
        global_action=global_action or "no_action",
        explanation=explanation or "",
    )
    obs = env.step(action)

    info = obs.info or {}
    result_md = (
        f"### Reward: **{obs.reward:.3f}**\n\n"
        f"| Component | Score |\n"
        f"|-----------|-------|\n"
        f"| Detection | {info.get('detection_score', 0):.2f} |\n"
        f"| Action appropriateness | {info.get('action_score', 0):.2f} |\n"
        f"| Explanation quality | {info.get('explanation_score', 0):.2f} |\n"
        f"| Efficiency | {info.get('efficiency_score', 0):.2f} |\n\n"
    )
    if obs.done:
        result_md += "**Episode complete.** Pick a new task and click Start Episode."
    else:
        result_md += "_Continuing to next step._"

    new_worker_ids = [w.worker_id for w in obs.worker_outputs]
    decision_updates = []
    for i in range(5):
        if i < len(new_worker_ids):
            decision_updates.append(
                gr.update(label=f"{new_worker_ids[i]} → decision", value="approve", visible=True)
            )
        else:
            decision_updates.append(gr.update(visible=False))

    return (
        env,
        new_worker_ids,
        _format_workers(obs),
        _format_reference(obs),
        _format_status(obs, env),
        result_md,
        *decision_updates,
    )


def build_demo() -> gr.Blocks:
    task_choices = [
        (f"{tid}  ({meta['difficulty']})", tid)
        for tid, meta in TASK_METADATA.items()
    ]

    with gr.Blocks(title="AI Agent Oversight Hub — Live Demo") as demo:
        gr.Markdown(
            "# AI Agent Oversight Hub — Live Demo\n\n"
            "Play the role of an AI supervisor: review worker outputs, "
            "spot errors, and submit oversight decisions. The grader scores "
            "you on detection accuracy, action appropriateness, explanation "
            "quality, and efficiency.\n"
        )

        env_state = gr.State(value=None)
        worker_ids_state = gr.State(value=[])

        with gr.Row():
            task_dd = gr.Dropdown(
                choices=task_choices, value="easy_single_error", label="Choose a scenario"
            )
            start_btn = gr.Button("Start Episode", variant="primary")

        with gr.Row():
            with gr.Column(scale=2):
                workers_md = gr.Markdown("_Click Start Episode._")
                ref_md = gr.Markdown("")
            with gr.Column(scale=1):
                status_md = gr.Markdown("")

        gr.Markdown("## Your Decisions")
        with gr.Row():
            decision_choices = ["approve", "reject", "flag", "reassign", "quarantine"]
            d0 = gr.Dropdown(choices=decision_choices, value="approve", label="worker 1", visible=False)
            d1 = gr.Dropdown(choices=decision_choices, value="approve", label="worker 2", visible=False)
            d2 = gr.Dropdown(choices=decision_choices, value="approve", label="worker 3", visible=False)
            d3 = gr.Dropdown(choices=decision_choices, value="approve", label="worker 4", visible=False)
            d4 = gr.Dropdown(choices=decision_choices, value="approve", label="worker 5", visible=False)

        with gr.Row():
            global_action = gr.Dropdown(
                choices=["no_action", "raise_alert", "halt_pipeline", "request_human_review"],
                value="no_action",
                label="Global action",
            )
            explanation = gr.Textbox(
                label="Explanation (cite specific evidence from the reference)",
                placeholder="Worker beta cites a 12% figure, but the reference document states 4.2%...",
                lines=3,
            )

        submit_btn = gr.Button("Submit Review", variant="primary")
        result_md = gr.Markdown("")

        start_btn.click(
            start_episode,
            inputs=[task_dd],
            outputs=[env_state, worker_ids_state, workers_md, ref_md, status_md, result_md, d0, d1, d2, d3, d4],
        )
        submit_btn.click(
            submit_review,
            inputs=[env_state, worker_ids_state, d0, d1, d2, d3, d4, global_action, explanation],
            outputs=[env_state, worker_ids_state, workers_md, ref_md, status_md, result_md, d0, d1, d2, d3, d4],
        )

        gr.Markdown(
            "---\n"
            "**API Endpoints:** `POST /reset` · `POST /step` · `GET /state` · `GET /tasks` · `GET /health`  \n"
            "**GitHub:** [hemayasekar/ai-agent-oversight-hub](https://github.com/hemayasekar/ai-agent-oversight-hub)"
        )

    return demo
