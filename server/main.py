"""FastAPI server for AI Agent Oversight Hub OpenEnv."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .environment import (
    OversightEnvironment,
    create_environment,
    get_available_tasks,
)
from .models import (
    OversightAction,
    OversightObservation,
    StepResult,
    ResetResult,
    StateResult,
    WorkerDecision,
)
from .scenarios import TASK_METADATA


_env: Optional[OversightEnvironment] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _env
    _env = create_environment()
    yield
    if _env:
        _env.close()


app = FastAPI(
    title="AI Agent Oversight Hub",
    description="An OpenEnv environment for training AI oversight agents to monitor multi-agent systems",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---

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


# --- Endpoints ---

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


@app.post("/reset", response_model=ResetResult)
async def reset(request: ResetRequest = None):
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
    result = _env.reset(task_id)
    return result


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest):
    global _env

    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    # Parse decisions
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

    result = _env.step(action)
    return result


@app.get("/state", response_model=StateResult)
async def state():
    global _env

    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    return _env.state()


# --- OpenEnv Spec Compliance ---

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


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
