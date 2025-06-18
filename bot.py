import os

from examples.social_graph_bot import run


def main() -> None:
    token = os.getenv("DISCORD_TOKEN")
    channel = os.getenv("MONITOR_CHANNEL")
    if not token or not channel:
        raise SystemExit("Please set DISCORD_TOKEN and MONITOR_CHANNEL environment variables")
    run(token, int(channel))


if __name__ == "__main__":
    main()
