# Reward Manager

The reward manager is a small utility that records the quality of generated responses. It publishes `RewardEvent` messages through JetStream so other modules can learn from user feedback.

## Publishing Rewards

The `Ledger` class in `deepthought.motivate.ledger` exposes a `publish` method:

```python
ledger = Ledger(nc, js)
await ledger.publish(prompt, response, reward)
```

Each event stores the prompt, the generated response, a numeric reward and a timestamp. Consumers can subscribe to the `motivation` subject to process the data.

## Intended Usage

A future training service will gather these events and periodically fine‑tune the language model with preference based learning techniques.

## Running the Service

Registering ``RewardManager`` under the ``deepthought.services`` entry point
allows the orchestrator to start it alongside other components. Generate a
container skeleton with:

```bash
dtrt bus init service reward
```

This creates ``src/deepthought/services/reward`` containing a Dockerfile and
``nats.env.example``. After building the image you can run the service with the
same environment variables as the other bus templates.

An orchestrator configuration might look like:

```yaml
services:
  - reward_manager
  - memory
```

Launching ``dtrt orchestrate config.yaml`` will now start the reward manager and
memory services in a single process.
