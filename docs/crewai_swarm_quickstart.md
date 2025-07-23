# CrewAI Swarm Quick Start

Temporary swarms can be launched via the orchestrator using the `TemporaryCrew`
wrapper.

## Define a crew

```python
from deepthought.crew import FunctionLLM, TemporaryCrew

def echo(prompt: str) -> str:
    return f"echo: {prompt}"

crew = TemporaryCrew(
    agents=[{"role": "Echoer", "goal": "Repeat", "llm": FunctionLLM(echo)}],
    tasks=[{"description": "Respond to {topic}", "agent_index": 0}],
    inputs={"topic": "hello"},
)
```

## Orchestrator configuration

Add the crew factory to `orchestrator.yaml`:

```yaml
crews:
  - examples.crew_demo:create_demo_crew
```

Start the orchestrator:

```bash
dtrt orchestrate orchestrator.yaml
```
