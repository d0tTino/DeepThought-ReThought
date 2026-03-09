import asyncio

from deepthought.config import load_bot_env
from examples.social_graph_bot import run


def main() -> None:
    env = load_bot_env()
    asyncio.run(
        run(
            env.DISCORD_BOT_TOKEN,
            env.MONITOR_CHANNEL,
            holiday_locale=env.PROJECT_HOLIDAY_LOCALE,
        )
    )


if __name__ == "__main__":
    main()
