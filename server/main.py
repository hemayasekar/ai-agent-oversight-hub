"""FastAPI server for AI Agent Oversight Hub OpenEnv.

Uses OpenEnv's create_fastapi_app for framework compliance,
with additional custom endpoints for tasks and health.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv.core.env_server import create_fastapi_app

from .environment import (
    OversightEnvironment,
    create_environment,
    get_available_tasks,
)
from .models import (
    OversightAction,
    OversightObservation,
    OversightState,
    WorkerDecision,
)
from .scenarios import TASK_METADATA


# --- Create the OpenEnv-compliant FastAPI app ---
_default_task = "easy_single_error"

app = create_fastapi_app(
    env=lambda: create_environment(_default_task),
    action_cls=OversightAction,
    observation_cls=OversightObservation,
)

app.title = "AI Agent Oversight Hub"
app.description = "An OpenEnv environment for training AI oversight agents to monitor multi-agent systems"
app.version = "1.0.0"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Additional custom endpoints ---

_env: Optional[OversightEnvironment] = None


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    decisions: list[dict]
    global_action: str = "no_action"
    explanation: str = ""


class TaskInfo(BaseModel):
    task_id: str
    difficulty: str
    description: str
    max_steps: int


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="healthy", environment="ai-agent-oversight-hub", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", environment="ai-agent-oversight-hub", version="1.0.0")


@app.get("/tasks", response_model=list[TaskInfo])
async def list_tasks_endpoint():
    return [
        TaskInfo(
            task_id=tid,
            difficulty=meta["difficulty"],
            description=meta["description"],
            max_steps=meta["max_steps"],
        )
        for tid, meta in TASK_METADATA.items()
    ]


@app.post("/reset")
async def reset_endpoint(request: ResetRequest = None):
    global _env

    if request is None:
        request = ResetRequest()

    task_id = request.task_id or "easy_single_error"

    available = get_available_tasks()
    if task_id not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task_id}. Available: {available}",
        )

    if _env:
        _env.close()

    _env = create_environment(task_id)
    obs = _env.reset(task_id=task_id)
    return {"observation": obs.model_dump()}


@app.post("/step")
async def step_endpoint(request: StepRequest):
    global _env

    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    decisions = []
    for d in request.decisions:
        decisions.append(WorkerDecision(
            worker_id=d.get("worker_id", ""),
            decision=d.get("decision", "approve"),
            issue_type=d.get("issue_type", "none"),
            confidence=d.get("confidence", 0.8),
        ))

    action = OversightAction(
        decisions=decisions,
        global_action=request.global_action,
        explanation=request.explanation,
    )

    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": obs.info,
    }


@app.get("/state")
async def state_endpoint():
    global _env

    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    return _env.state.model_dump()


@app.get("/openenv/info")
async def openenv_info():
    return {
        "name": "ai-agent-oversight-hub",
        "version": "1.0.0",
        "description": "Train AI oversight agents to monitor multi-agent systems for errors, hallucinations, and adversarial behavior",
        "author": "OpenEnv Community",
        "tasks": get_available_tasks(),
        "spec_version": "1.0",
    }


@app.get("/openenv/observation_space")
async def observation_space():
    return OversightObservation.model_json_schema()


@app.get("/openenv/action_space")
async def action_space():
    return OversightAction.model_json_schema()


# --- Mount Gradio interactive demo at /demo ---
try:
    import gradio as gr
    from .demo import build_demo

    _demo = build_demo()
    app = gr.mount_gradio_app(app, _demo, path="/demo")
except ImportError:
    pass  # Gradio is optional — server still works without it


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
