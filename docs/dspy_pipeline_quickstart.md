# DSPy Pipeline Quick Start

This guide demonstrates how to enable the optional DSPy pipeline used for
question answering.

## Install DSPy

```bash
pip install dspy-ai
```

## Basic usage

```python
from deepthought.pipeline import build_qa_pipeline

qa = build_qa_pipeline()
print(qa("What is 2 + 2?"))
```

## Orchestrator integration

`RemoteLLM` activates the pipeline when `USE_DSPY=true` is set:

```bash
export USE_DSPY=true
dtrt orchestrate orchestrator.yaml
```
