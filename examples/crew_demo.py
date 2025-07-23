"""Demo showing how to launch a temporary CrewAI crew via the orchestrator."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from deepthought import orchestrator
from deepthought.crew import FunctionLLM, TemporaryCrew


def create_demo_crew() -> TemporaryCrew:
    """Return a crew that echoes the provided topic."""

    def echo(prompt: str) -> str:
        return f"echo: {prompt}"

    agents = [
        {
            "role": "Echoer",
            "goal": "Repeat the input",
            "backstory": "Simple demo agent",
            "llm": FunctionLLM(echo),
        }
    ]
    tasks = [
        {
            "description": "Respond to {topic}",
            "agent_index": 0,
        }
    ]
    return TemporaryCrew(agents, tasks, inputs={"topic": "hello"})


async def main() -> None:
    cfg = Path(tempfile.gettempdir()) / "crew_orch.yaml"
    cfg.write_text(
        "crews:\n  - examples.crew_demo:create_demo_crew\n", encoding="utf-8"
    )
    task = asyncio.create_task(orchestrator.run(str(cfg)))
    await asyncio.sleep(1.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
