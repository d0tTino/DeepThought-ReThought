#!/usr/bin/env python3
"""Record DeepThought events to a JSONL file."""

import argparse
import asyncio
import logging

from nats.aio.client import Client as NATS

from deepthought.harness.trace import TraceRecorder


async def main() -> None:
    parser = argparse.ArgumentParser(description="Record events from NATS")
    parser.add_argument("--output", "-o", default="trace.jsonl", help="Output JSONL file")
    parser.add_argument("--nats", "-n", default="nats://localhost:4222", help="NATS server URL")
    parser.add_argument("--durable", "-d", default="trace_recorder", help="Durable consumer name prefix")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    nc = NATS()
    await nc.connect(servers=[args.nats])
    js = nc.jetstream()

    recorder = TraceRecorder(nc, js, args.output)
    if not await recorder.start(durable_name=args.durable):
        raise SystemExit("Failed to start recorder")

    print("Recording events... Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await recorder.stop()
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())
