from __future__ import annotations

import asyncio
import json
import logging
import ssl
from contextlib import AsyncExitStack
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from .config import get_settings
from .eda import Publisher, Subscriber
from .eda.events import (
    EventSubjects,
    PlanGeneratedPayload,
    PlanRequestedPayload,
)
from .planning import planner, translator

logger = logging.getLogger(__name__)


def discover_services(names: Iterable[str] | None = None) -> list[type]:
    """Return service classes registered under ``deepthought.services``."""
    eps_obj = metadata.entry_points()
    if hasattr(eps_obj, "select"):
        eps = eps_obj.select(group="deepthought.services")
    else:  # pragma: no cover - Python < 3.10
        eps = eps_obj.get("deepthought.services", [])
    selected = []
    if names is None:
        selected = eps
    else:
        wanted = set(names)
        for ep in eps:
            if ep.name in wanted:
                selected.append(ep)
    services = []
    for ep in selected:
        try:
            services.append(ep.load())
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load service %s: %s", ep.name, exc, exc_info=True)
    return services


def _load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError("PyYAML required for YAML config") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


async def _connect_nats():
    from nats.aio.client import Client as NATS

    settings = get_settings()
    ssl_ctx = None
    if settings.nats_tls_cert and settings.nats_tls_key:
        ssl_ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
        if settings.nats_tls_ca:
            ssl_ctx.load_verify_locations(settings.nats_tls_ca)
        ssl_ctx.load_cert_chain(settings.nats_tls_cert, settings.nats_tls_key)
    nc = NATS()
    await nc.connect(
        servers=[settings.nats_url],
        tls=ssl_ctx,
        user=settings.nats_username,
        password=settings.nats_password,
        name="dtrt_orchestrator",
    )
    js = nc.jetstream()
    return nc, js


async def run(config_path: str) -> None:
    """Start services defined in ``config_path``."""
    cfg = _load_config(config_path)
    names = cfg.get("services", [])
    service_classes = discover_services(names)
    if not service_classes:
        logger.warning("No services found for names %s", names)
        return
    nc, js = await _connect_nats()
    pub = Publisher(nc, js)
    sub = Subscriber(nc, js)
    l2p = translator.L2PTranslator()

    async def _handle_plan(msg):
        try:
            data = json.loads(msg.data.decode())
            payload = PlanRequestedPayload.from_dict(data)
            domain, problem = l2p.translate(payload.goal)
            actions = planner.plan(domain, problem)
            out = PlanGeneratedPayload(plan=actions, input_id=payload.input_id)
            await pub.publish(EventSubjects.PLAN_GENERATED, out, use_jetstream=True)
            await msg.ack()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to generate plan: %s", exc, exc_info=True)
            if hasattr(msg, "nak"):
                await msg.nak()

    async with AsyncExitStack() as stack:
        if nc.is_connected:
            stack.push_async_callback(nc.drain)
        stack.push_async_callback(sub.unsubscribe_all)
        instances = []
        for cls in service_classes:
            inst = cls(nc, js)
            await stack.enter_async_context(inst)
            instances.append(inst)
        await sub.subscribe(
            subject=EventSubjects.PLAN_REQUESTED,
            handler=_handle_plan,
            use_jetstream=True,
            durable="planner",
        )
        logger.info("Started %d services", len(instances))
        try:
            await asyncio.Event().wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
