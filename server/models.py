"""Pydantic models for AI Agent Oversight Hub."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class WorkerOutput(BaseModel):
    """A single output from a simulated worker agent."""
    worker_id: str
    worker_role: str  # "summarizer", "fact_checker", "data_extractor"
    task_assigned: str
    output_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = {}


class OversightObservation(BaseModel):
    """What the oversight agent sees each step."""
    task_id: str
    task_description: str
    scenario_id: str
    current_step: int
    max_steps: int
    worker_outputs: list[WorkerOutput]
    reference_materials: str
    worker_history: list[dict[str, Any]] = []
    flagged_items: list[str] = []
    system_alerts: list[str] = []
    quarantined_workers: list[str] = []
    available_actions: list[str] = [
        "approve", "reject", "flag", "reassign", "quarantine"
    ]
    steps_remaining: int


class WorkerDecision(BaseModel):
    """A decision about a single worker's output."""
    worker_id: str
    decision: str  # "approve", "reject", "flag", "reassign", "quarantine"
    issue_type: str = "none"  # "none", "hallucination", "factual_error", "off_task", "sabotage", "coalition"
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class OversightAction(BaseModel):
    """The oversight agent's action each step."""
    decisions: list[WorkerDecision]
    global_action: str = "no_action"  # "no_action", "escalate", "pause_all"
    explanation: str = ""


class StepResult(BaseModel):
    """Result of a step."""
    observation: OversightObservation
    reward: float
    done: bool
    info: dict[str, Any] = {}


class ResetResult(BaseModel):
    """Result of a reset."""
    observation: OversightObservation


class StateResult(BaseModel):
    """Current environment state."""
    task_id: str
    scenario_id: str
    current_step: int
    max_steps: int
    done: bool
    total_reward: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    quarantined_workers: list[str]
