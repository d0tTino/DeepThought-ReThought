"""Lightweight wrapper for creating temporary CrewAI crews."""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from crewai import Agent, Crew, Task
from langchain.llms.base import LLM


class FunctionLLM(LLM):
    """Simple LLM that returns the result of a Python callable."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    @property
    def _llm_type(self) -> str:  # pragma: no cover - tiny wrapper
        return "function"

    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        return self._fn(prompt)


class TemporaryCrew:
    """Context manager that runs a CrewAI ``Crew`` once when started."""

    def __init__(
        self,
        agents: Iterable[dict[str, Any]],
        tasks: Iterable[dict[str, Any]],
        inputs: dict[str, Any] | None = None,
    ) -> None:
        self._agents = [Agent(**a) for a in agents]
        self._tasks = []
        for cfg in tasks:
            idx = cfg.pop("agent_index", 0)
            agent = self._agents[idx]
            self._tasks.append(Task(agent=agent, **cfg))
        self._crew = Crew(agents=self._agents, tasks=self._tasks)
        self._inputs = inputs or {}

    async def __aenter__(self) -> "TemporaryCrew":
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._crew.kickoff, self._inputs)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


__all__ = ["FunctionLLM", "TemporaryCrew"]

