"""Pydantic models for AI Agent Oversight Hub.

Uses OpenEnv base classes (Observation, Action, State) for framework compliance.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict

from openenv.core.env_server import (
    Observation as OpenEnvObservation,
    Action as OpenEnvAction,
    State as OpenEnvState,
)


class WorkerOutput(BaseModel):
    """A single output from a simulated worker agent."""
    worker_id: str
    worker_role: str  # "summarizer", "fact_checker", "data_extractor"
    task_assigned: str
    output_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = {}


class OversightObservation(OpenEnvObservation):
    """What the oversight agent sees each step.

    Extends OpenEnv Observation with done/reward/metadata from base class.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    task_id: str = ""
    task_description: str = ""
    scenario_id: str = ""
    current_step: int = 0
    max_steps: int = 0
    worker_outputs: list[WorkerOutput] = []
    reference_materials: str = ""
    worker_history: list[dict[str, Any]] = []
    flagged_items: list[str] = []
    system_alerts: list[str] = []
    quarantined_workers: list[str] = []
    available_actions: list[str] = [
        "approve", "reject", "flag", "reassign", "quarantine"
    ]
    steps_remaining: int = 0
    info: dict[str, Any] = {}


class WorkerDecision(BaseModel):
    """A decision about a single worker's output."""
    worker_id: str
    decision: str  # "approve", "reject", "flag", "reassign", "quarantine"
    issue_type: str = "none"  # "none", "hallucination", "factual_error", "off_task", "sabotage", "coalition"
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class OversightAction(OpenEnvAction):
    """The oversight agent's action each step.

    Extends OpenEnv Action base class.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    decisions: list[WorkerDecision] = []
    global_action: str = "no_action"  # "no_action", "escalate", "pause_all"
    explanation: str = ""


class OversightState(OpenEnvState):
    """Current environment state.

    Extends OpenEnv State with episode_id and step_count from base class.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    task_id: str = ""
    scenario_id: str = ""
    current_step: int = 0
    max_steps: int = 0
    done: bool = False
    total_reward: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    quarantined_workers: list[str] = []


# Legacy aliases for backward compatibility
StepResult = None  # No longer needed — Observation carries reward+done
ResetResult = None  # No longer needed — reset() returns Observation
StateResult = OversightState  # Alias
