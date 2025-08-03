# Interaction Trace Format

This file describes the JSONL format produced by `TraceRecorder` and related tools.
Each line represents a single event emitted by the system.

```json
{
  "event": "INPUT_RECEIVED",
  "payload": {"user_input": "hello", "input_id": "abc"},
  "perception": {"flirtation": 0.2, "avoidance": 0.1, "manipulation": 0.0, "sarcasm": 0.3, "supportiveness": 0.4},
  "affinity": 0.1
}
```

Fields
------
- **event** – name of the event, e.g. `INPUT_RECEIVED`, `RESPONSE_GENERATED` or `CHAT_RAW`.
- **payload** – raw payload sent with the event. The structure depends on the event type and may be
  either a dictionary or a string for raw chat messages.
- **perception** – dictionary of social perception label probabilities. When the event contains
  textual user input the recorder runs the social perception classifier and records the resulting
  probabilities for flirtation, avoidance, manipulation, sarcasm, and supportiveness. If no text
  is available this field is `null`.
- **affinity** – current affinity score for the user after applying the perception delta. The score is
  incremented when a new input is processed.

All records are written on a single line separated by newlines so that the file can be streamed or
processed incrementally.
