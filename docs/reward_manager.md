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
