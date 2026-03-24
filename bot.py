import asyncio
import os

from deepthought.config import load_discord_bot_token
from deepthought.runtime.discord_gateway_app import run_discord_gateway


def main() -> None:
    token = load_discord_bot_token()
    if not token:
        raise SystemExit("DISCORD_BOT_TOKEN is required")
    asyncio.run(
        run_discord_gateway(
            token=token,
            nats_url=os.getenv("NATS_URL", "nats://localhost:4222"),
        )
    )


if __name__ == "__main__":
    main()
