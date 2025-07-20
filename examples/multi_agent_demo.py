import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import TypedDict

import nats
from langgraph.graph import StateGraph
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig
from prometheus_client import start_http_server

from deepthought.metrics.prometheus import INPUT_LATENCY_SECONDS, INPUTS_TOTAL
from deepthought.modules import InputHandler, OutputHandler
from deepthought.modules.llm_remote import RemoteLLM
from deepthought.services import MemoryService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STREAM_NAME = "deepthought_events"


async def ensure_stream(js):
    try:
        await js.stream_info(STREAM_NAME)
    except Exception:
        config = StreamConfig(
            name=STREAM_NAME,
            subjects=["dtr.>"],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_msgs_per_subject=100,
            discard=DiscardPolicy.OLD,
        )
        await js.add_stream(config)


async def main() -> None:
    from deepthought.config import get_settings

    settings = get_settings()
    metrics_port = int(os.getenv("METRICS_PORT", "0"))
    if metrics_port > 0:
        start_http_server(metrics_port)
    nc = await nats.connect(settings.nats_url)
    js = nc.jetstream()

    await ensure_stream(js)

    model_proc = None
    container_proc = None
    edge_image = os.getenv("EDGE_IMAGE")
    if edge_image:
        logger.info("Starting edge model container %s", edge_image)
        container_proc = await asyncio.create_subprocess_exec("docker", "run", "--rm", "-p", "8000:8000", edge_image)
        await asyncio.sleep(5.0)
    elif os.getenv("MODEL_PATH"):
        script = Path(__file__).resolve().parents[1] / "tools" / "edge_server.py"
        logger.info("Starting edge model from %s", script)
        model_proc = await asyncio.create_subprocess_exec(sys.executable, str(script))
        await asyncio.sleep(2.0)

    memory_service = MemoryService.from_config(nc, js)
    llm = RemoteLLM(nc, js)

    global input_handlers, output_handlers
    input_handlers = [InputHandler(nc, js) for _ in range(3)]
    output_handlers = [OutputHandler(nc, js) for _ in range(3)]

    async with memory_service:
        await asyncio.gather(
            llm.start_listening(durable_name=f"llm_demo_{uuid.uuid4()}"),
            *(
                oh.start_listening(durable_name=f"out_demo_{i}_{uuid.uuid4()}")
                for i, oh in enumerate(output_handlers)
            ),
        )

        await asyncio.sleep(1.0)

        class AgentState(TypedDict):
            text: str
            idx: int
            count: int

        async def send_receive(state: AgentState) -> AgentState:
            idx = state["idx"]
            event = asyncio.Event()
            start = time.perf_counter()

            def cb(_id: str, text: str) -> None:
                logger.info("Agent %s says: %s", idx + 1, text)
                state["text"] = text
                event.set()

            output_handlers[idx]._output_callback = cb
            await input_handlers[idx].process_input(state["text"])
            await event.wait()
            duration = time.perf_counter() - start
            INPUTS_TOTAL.labels(service="multi_agent_demo").inc()
            INPUT_LATENCY_SECONDS.labels(service="multi_agent_demo").observe(duration)
            state["count"] += 1
            return state

        def rotate(state: AgentState) -> AgentState:
            state["idx"] = (state["idx"] + 1) % 3
            return state

        graph = StateGraph(AgentState)
        graph.add_node("talk", send_receive)
        graph.add_node("next", rotate)
        graph.set_entry_point("talk")
        graph.add_edge("talk", "next")
        graph.add_edge("next", "talk")
        compiled = graph.compile()

        state: AgentState = {"text": "Hello from agent 1!", "idx": 0, "count": 0}
        while state["count"] < 3:
            state = await compiled.ainvoke(state)

        await llm.stop_listening()
        await asyncio.gather(*(oh.stop_listening() for oh in output_handlers))
        await nc.drain()
        if container_proc:
            container_proc.terminate()
            await container_proc.wait()
        elif model_proc:
            model_proc.terminate()
            await model_proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
