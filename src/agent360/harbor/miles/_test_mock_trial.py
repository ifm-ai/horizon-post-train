"""Mock Harbor Trial for local testing.

Activated when HARBOR_TEST_MOCK_TRIAL=1 is set. Lets harbor_server.py exercise
every gap-site log emission (semaphore wrap, phase hooks, trial.run timing)
WITHOUT requiring the real harbor-framework package, a real env (Daytona,
managed sandbox services, or any external services.)

Use case: laptop smoke tests and unit tests where we only care about whether
harbor_server emits the right log records, not whether Harbor actually does
anything useful.
"""

from __future__ import annotations

import asyncio
import enum
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


class MockTrialEvent(enum.Enum):
    START = "start"
    AGENT_START = "agent_start"
    VERIFICATION_START = "verification_start"
    END = "end"


class _MockRewards:
    def __init__(self, reward: float) -> None:
        self.rewards = {"reward": reward}


class _MockVerifierResult:
    def __init__(self, reward: float) -> None:
        self.rewards = {"reward": reward}


class _MockAgentResult:
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {"exit_status": "Submitted"}


class _MockResult:
    def __init__(self, reward: float = 1.0) -> None:
        self.exception_info = None
        self.verifier_result = _MockVerifierResult(reward)
        self.agent_result = _MockAgentResult()
        self.reward_total_value = reward


class MockTrial:
    """Mock Trial that fires phase hooks on a configurable schedule.

    Default schedule (in seconds, configurable via HARBOR_TEST_MOCK_*):
      0.00s  trial.run() starts
      0.05s  START hook
      0.10s  AGENT_START hook
      0.20s  VERIFICATION_START hook
      0.25s  END hook + return result
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        # Each MockTrial gets its own scratch dir so the cleanup path in
        # harbor_server.py:_run_trial doesn't tank.
        self.trial_dir = tempfile.mkdtemp(prefix="mock-trial-")
        self._hooks: dict[MockTrialEvent, list[Callable]] = {
            e: [] for e in MockTrialEvent
        }
        # Touch a faux trajectory file so _save_trajectory_final has something
        # to inspect (it's tolerant of missing files but tests prefer a hit).
        agent_dir = Path(self.trial_dir) / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "trajectory.json").write_text('{"messages": []}')

    @classmethod
    async def create(cls, *, config: Any) -> MockTrial:
        # Simulate Trial.create() work (env setup, image pull metadata, etc.)
        await asyncio.sleep(float(os.environ.get("HARBOR_TEST_MOCK_CREATE_SECS", "0.02")))
        return cls(config)

    def add_hook(self, event: MockTrialEvent, callback: Callable) -> None:
        self._hooks[event].append(callback)

    async def _fire(self, event: MockTrialEvent) -> None:
        for cb in self._hooks.get(event, []):
            r = cb(event)
            if asyncio.iscoroutine(r):
                await r

    async def run(self) -> _MockResult:
        # Read configurable timings from env so tests can compress / expand
        # the phase durations.
        delay_start = float(os.environ.get("HARBOR_TEST_MOCK_PHASE_START_SECS", "0.05"))
        delay_agent = float(os.environ.get("HARBOR_TEST_MOCK_PHASE_AGENT_SECS", "0.05"))
        delay_verify = float(os.environ.get("HARBOR_TEST_MOCK_PHASE_VERIFY_SECS", "0.10"))
        delay_end = float(os.environ.get("HARBOR_TEST_MOCK_PHASE_END_SECS", "0.05"))

        await asyncio.sleep(delay_start)
        await self._fire(MockTrialEvent.START)
        await asyncio.sleep(delay_agent)
        await self._fire(MockTrialEvent.AGENT_START)
        await asyncio.sleep(delay_verify)
        await self._fire(MockTrialEvent.VERIFICATION_START)
        await asyncio.sleep(delay_end)
        await self._fire(MockTrialEvent.END)

        reward = float(os.environ.get("HARBOR_TEST_MOCK_REWARD", "1.0"))
        return _MockResult(reward=reward)


def is_mock_enabled() -> bool:
    return os.environ.get("HARBOR_TEST_MOCK_TRIAL", "").lower() in ("1", "true", "yes")
