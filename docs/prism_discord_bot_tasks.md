# Tasks for Prism Discord Bot Phases

This checklist breaks down the major steps from the full report. Use it to track progress toward the complete Prism-enabled Discord bot.

## Phase 1

- [ ] Initialize a new Discord bot with the required intents.
- [ ] Set up the Python project and virtual environment.
- [ ] Implement basic `bot.py` that logs in and reports readiness.
- [ ] Integrate sentiment analysis using TextBlob or VADER.
- [ ] Create a SQLite database and log user interactions.
- [ ] Monitor idle channels and send prompts when no one is active.
- [ ] Send interaction data to the Prism HTTP endpoint.

## Phase 2

- [ ] Add a `memories` table for notable messages with sentiment scores.
- [ ] Provide a recall function to fetch recent memories for a user.
- [ ] Detect crowded channels and limit bot chatter when multiple bots are present.
- [ ] Generate posts automatically after long idle periods.
- [ ] Implement bullying detection and optional sarcastic replies.
- [ ] Aggregate sentiment trends for thematic modeling.
- [ ] Maintain theories about users and reply cryptically when asked.
- [ ] Queue messages for delayed analysis and thoughtful responses.

