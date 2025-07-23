# PSL Integration

The project includes a small wrapper around a simplified Probabilistic Soft Logic (PSL) model.  
A PSL model defines a set of predicates with associated weights. Inference computes a weighted sum
for given evidence and compares it against a threshold.

Configuration files use YAML and must contain a `model` section with `weights` as well as an
optional `threshold` value:

```yaml
model:
  weights:
    lines_added: 0.1
    lines_deleted: 0.2
threshold: 1.0
```

Use :class:`deepthought.psl.risk.RiskScorer` to load the configuration and evaluate commits.
