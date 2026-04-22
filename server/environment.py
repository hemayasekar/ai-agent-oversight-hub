"""OversightEnvironment — core step/reset/state lifecycle.

Manages the multi-agent oversight simulation with quarantine mechanics.
"""

from typing import Any, Optional

from .models import (
    WorkerOutput,
    OversightObservation,
    OversightAction,
    StepResult,
    ResetResult,
    StateResult,
)
from .scenarios import get_scenario, list_tasks, Scenario, TASK_METADATA
from .grader import grade_step


class OversightEnvironment:
    """Environment where an oversight agent monitors simulated workers."""

    def __init__(self, task_id: str = "easy_single_error"):
        self._task_id = task_id
        self._scenario: Optional[Scenario] = None
        self._current_step = 0
        self._max_steps = TASK_METADATA[task_id]["max_steps"]
        self._done = False
        self._total_reward = 0.0
        self._quarantined_workers: list[str] = []
        self._previous_decisions: list[dict] = []
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0

    def reset(self, task_id: Optional[str] = None) -> ResetResult:
        """Reset environment to initial state for a task."""
        if task_id:
            self._task_id = task_id
            self._max_steps = TASK_METADATA[task_id]["max_steps"]

        self._scenario = get_scenario(self._task_id)
        self._current_step = 0
        self._done = False
        self._total_reward = 0.0
        self._quarantined_workers = []
        self._previous_decisions = []
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0

        observation = self._build_observation()
        return ResetResult(observation=observation)

    def step(self, action: OversightAction) -> StepResult:
        """Execute one step: agent submits decisions, environment grades them."""
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"message": "Episode is done. Call /reset to start a new one."},
            )

        if self._scenario is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Get current scenario step
        scenario_step_idx = min(self._current_step, len(self._scenario.steps) - 1)
        scenario_step = self._scenario.steps[scenario_step_idx]

        # Grade the action
        steps_remaining = self._max_steps - self._current_step - 1
        reward, info = grade_step(
            action=action,
            step=scenario_step,
            steps_remaining=steps_remaining,
            max_steps=self._max_steps,
            quarantined_workers=self._quarantined_workers,
            previous_decisions=self._previous_decisions,
        )

        # Update cumulative stats
        self._total_reward += reward
        self._tp += info.get("tp", 0)
        self._fp += info.get("fp", 0)
        self._fn += info.get("fn", 0)
        self._tn += info.get("tn", 0)

        # Process quarantine actions
        for d in action.decisions:
            if d.decision == "quarantine" and d.worker_id not in self._quarantined_workers:
                self._quarantined_workers.append(d.worker_id)

        # Store decisions for flip-flop detection
        self._previous_decisions = [
            {"worker_id": d.worker_id, "decision": d.decision}
            for d in action.decisions
        ]

        # Advance step
        self._current_step += 1

        # Check termination
        if self._current_step >= len(self._scenario.steps):
            self._done = True
        elif self._current_step >= self._max_steps:
            self._done = True

        # Handle global actions
        if action.global_action == "pause_all":
            self._done = True
            info["paused"] = True

        observation = self._build_observation()

        return StepResult(
            observation=observation,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> StateResult:
        """Return current environment state."""
        return StateResult(
            task_id=self._task_id,
            scenario_id=self._scenario.scenario_id if self._scenario else "",
            current_step=self._current_step,
            max_steps=self._max_steps,
            done=self._done,
            total_reward=round(self._total_reward, 4),
            true_positives=self._tp,
            false_positives=self._fp,
            false_negatives=self._fn,
            true_negatives=self._tn,
            quarantined_workers=self._quarantined_workers,
        )

    def close(self):
        """Clean up resources."""
        self._scenario = None

    def _build_observation(self) -> OversightObservation:
        """Build the observation for the current step."""
        if self._scenario is None:
            # Return empty observation if not initialized
            return OversightObservation(
                task_id=self._task_id,
                task_description="",
                scenario_id="",
                current_step=0,
                max_steps=self._max_steps,
                worker_outputs=[],
                reference_materials="",
                steps_remaining=self._max_steps,
            )

        # Get scenario step (or last step if past end)
        if self._current_step < len(self._scenario.steps):
            scenario_step = self._scenario.steps[self._current_step]
        else:
            scenario_step = self._scenario.steps[-1]

        # Filter out quarantined workers
        active_outputs = [
            w for w in scenario_step.worker_outputs
            if w["worker_id"] not in self._quarantined_workers
        ]

        worker_outputs = [
            WorkerOutput(**w) for w in active_outputs
        ]

        return OversightObservation(
            task_id=self._task_id,
            task_description=self._scenario.description,
            scenario_id=self._scenario.scenario_id,
            current_step=self._current_step,
            max_steps=self._max_steps,
            worker_outputs=worker_outputs,
            reference_materials=scenario_step.reference_snippet,
            worker_history=[
                {"step": i, "decisions": self._previous_decisions}
                for i in range(self._current_step)
            ] if self._current_step > 0 else [],
            flagged_items=[],
            system_alerts=scenario_step.alerts,
            quarantined_workers=self._quarantined_workers,
            steps_remaining=self._max_steps - self._current_step,
        )


def create_environment(task_id: str = "easy_single_error") -> OversightEnvironment:
    """Factory function to create an environment."""
    return OversightEnvironment(task_id)


def get_available_tasks() -> list[str]:
    """Return list of available task IDs."""
    return list_tasks()
