from examples.social_graph_bot import run
from deepthought.config import load_bot_env


def main() -> None:
    env = load_bot_env()
    run(env.DISCORD_TOKEN, env.MONITOR_CHANNEL)


if __name__ == "__main__":
    main()
