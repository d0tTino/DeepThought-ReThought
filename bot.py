import asyncio

from deepthought.config import load_bot_env
from examples.social_graph_bot import run


def main() -> None:
    env = load_bot_env()
    asyncio.run(run(env.DISCORD_TOKEN, env.MONITOR_CHANNEL))


if __name__ == "__main__":
    main()
