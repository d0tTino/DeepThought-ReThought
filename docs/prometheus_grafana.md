# Prometheus and Grafana

This snippet runs the metrics server described in [architecture.md](architecture.md) alongside Prometheus and Grafana.

```yaml
version: '3.8'
services:
  metrics:
    image: python:3.11-slim
    command: python -m deepthought.metrics_server
    volumes:
      - .:/app
    working_dir: /app
    ports:
      - "8000:8000"
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  grafana:
    image: grafana/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    ports:
      - "3000:3000"
```

Prometheus scrapes `metrics:8000` using the configuration below:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'deepthought'
    static_configs:
      - targets: ['metrics:8000']
```

Import `grafana/deepthought_metrics.json` to view two panels:

- **Inputs Total per Service** - rate of `inputs_total` events.
- **Average Latency Seconds** - calculated from `input_latency_seconds`.

The metrics server also exposes counters for rule evaluations. Snapshots of the
`rule_evaluations_total` metric can be visualized with the dashboard utility:

```bash
python tools/dashboard_rules.py path/to/metrics --show
```

This plots the evaluation count for each rule over time. When using Grafana you
can create an additional panel with the expression `rate(rule_evaluations_total[1m])`.

These dashboards let you track throughput and latency for each service.
