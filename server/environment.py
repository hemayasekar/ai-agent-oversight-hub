"""OversightEnvironment — OpenEnv-compliant environment.

Subclasses openenv.core.env_server.Environment with proper
reset(), step(), state property lifecycle.
"""

from typing import Any, Optional
import uuid

from openenv.core.env_server import Environment

from .models import (
    WorkerOutput,
    OversightObservation,
    OversightAction,
    OversightState,
)
from .scenarios import get_scenario, list_tasks, Scenario, TASK_METADATA
from .grader import grade_step


class OversightEnvironment(Environment[OversightAction, OversightObservation, OversightState]):
    """Environment where an oversight agent monitors simulated workers.

    Implements the OpenEnv Environment interface (reset, step, state).
    """

    def __init__(self, task_id: str = "easy_single_error"):
        super().__init__()
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
        self._episode_id: Optional[str] = None

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> OversightObservation:
        """Reset environment to initial state for a task.

        Conforms to OpenEnv Environment.reset() signature.
        """
        task_id = kwargs.get("task_id", None)
        if task_id:
            self._task_id = task_id
            self._max_steps = TASK_METADATA[task_id]["max_steps"]

        self._episode_id = episode_id or str(uuid.uuid4())
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

        return self._build_observation(reward=0.0, info={})

    def step(self, action: OversightAction, timeout_s: Optional[float] = None, **kwargs: Any) -> OversightObservation:
        """Execute one step: agent submits decisions, environment grades them.

        Conforms to OpenEnv Environment.step() signature.
        Returns OversightObservation with reward and done set.
        """
        if self._done:
            return self._build_observation(
                reward=0.0,
                done=True,
                info={"message": "Episode is done. Call reset to start a new one."},
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

        return self._build_observation(reward=reward, done=self._done, info=info)

    @property
    def state(self) -> OversightState:
        """Return current environment state.

        Conforms to OpenEnv Environment.state property.
        """
        return OversightState(
            episode_id=self._episode_id,
            step_count=self._current_step,
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

    def _build_observation(
        self,
        reward: float = 0.0,
        done: Optional[bool] = None,
        info: Optional[dict] = None,
    ) -> OversightObservation:
        """Build the observation for the current step."""
        if done is None:
            done = self._done
        if info is None:
            info = {}

        if self._scenario is None:
            return OversightObservation(
                done=done,
                reward=reward,
                task_id=self._task_id,
                task_description="",
                scenario_id="",
                current_step=0,
                max_steps=self._max_steps,
                worker_outputs=[],
                reference_materials="",
                steps_remaining=self._max_steps,
                info=info,
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
            done=done,
            reward=reward,
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
            info=info,
        )


def create_environment(task_id: str = "easy_single_error") -> OversightEnvironment:
    """Factory function to create an environment."""
    return OversightEnvironment(task_id)


def get_available_tasks() -> list[str]:
    """Return list of available task IDs."""
    return list_tasks()
