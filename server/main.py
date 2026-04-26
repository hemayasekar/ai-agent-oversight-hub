"""FastAPI server for AI Agent Oversight Hub OpenEnv.

Uses OpenEnv's create_fastapi_app for framework compliance,
with additional custom endpoints for tasks and health.

The environment is exposed as a singleton so /reset and /step calls share
state across requests (required for multi-step episodes). Both the
OpenEnv-registered routes (which expect `{"action": {...}}`) and our
legacy flat routes (which accept `{"decisions": [...], ...}`) operate on
the same env instance.
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


# --- Singleton env for stateful multi-step episodes ---
_default_task = "easy_single_error"
_env: OversightEnvironment = create_environment(_default_task)


def _env_singleton() -> OversightEnvironment:
    """Factory that returns the singleton env (and a no-op close)."""
    # OpenEnv calls .close() in finally blocks; we override to be a no-op
    # so the singleton survives across requests.
    _env.close = lambda: None  # type: ignore[method-assign]
    return _env


# --- Create the OpenEnv-compliant FastAPI app ---
app = create_fastapi_app(
    env=_env_singleton,
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


# Legacy flat-payload endpoints (kept for back-compat with train.py / demo.py).
# These provide a flat-payload alternative to OpenEnv's wrapped /reset and
# /step endpoints. The OpenEnv routes are also wired to the singleton env
# above, so both work — choose whichever is easier for your client.

@app.post("/reset_legacy")
async def reset_legacy_endpoint(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()
    task_id = request.task_id or "easy_single_error"
    available = get_available_tasks()
    if task_id not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task_id}. Available: {available}",
        )
    obs = _env.reset(task_id=task_id)
    return {"observation": obs.model_dump()}


@app.post("/step_legacy")
async def step_legacy_endpoint(request: StepRequest):
    decisions = [
        WorkerDecision(
            worker_id=d.get("worker_id", ""),
            decision=d.get("decision", "approve"),
            issue_type=d.get("issue_type", "none"),
            confidence=d.get("confidence", 0.8),
        )
        for d in request.decisions
    ]
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
