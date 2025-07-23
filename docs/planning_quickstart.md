# Planning Quick Start

The planning helpers use L2P and pyperplan to create simple action sequences.

## Translate a goal

```python
from deepthought.planning import L2PTranslator

translator = L2PTranslator()
domain, problem = translator.translate("move obj from loc1 to loc2")
```

## Compute a plan

```python
from deepthought.planning import plan

steps = plan(domain, problem)
print(steps)
```

This prints something like:

```text
['(move obj loc1 loc2)']
```

## Command-line demo

Run the example script:

```bash
python examples/planning_demo.py
```
