from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, suppress
import json
import logging
import ssl
from importlib import metadata
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from .config import get_settings

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


def _load_callable(path: str) -> Callable[[], Any]:
    """Import and return a callable specified as ``"module:function"``."""
    mod_name, func_name = path.split(":", 1)
    mod = __import__(mod_name, fromlist=[func_name])
    return getattr(mod, func_name)


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
    crew_specs = cfg.get("crews", [])
    graph_specs = cfg.get("graphs", [])
    if not service_classes and not crew_specs and not graph_specs:
        logger.warning("No services, crews or graphs found in config")
        return
    crew_factories = [_load_callable(s) for s in crew_specs]
    graph_factories: list[Callable[[], Awaitable[Any]]] = [
        _load_callable(s) for s in graph_specs
    ]
    nc, js = await _connect_nats()
    async with AsyncExitStack() as stack:
        if nc.is_connected:
            stack.push_async_callback(nc.drain)
        instances = []
        crews = []
        graph_tasks = []
        for cls in service_classes:
            inst = cls(nc, js)
            await stack.enter_async_context(inst)
            instances.append(inst)
        for factory in crew_factories:
            crew = factory()
            await stack.enter_async_context(crew)
            crews.append(crew)
        for factory in graph_factories:
            graph_tasks.append(asyncio.create_task(factory()))
        logger.info(
            "Started %d services, %d crews, %d graphs",
            len(instances),
            len(crews),
            len(graph_tasks),
        )
        try:
            await asyncio.Event().wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            for t in graph_tasks:
                t.cancel()
                with suppress(Exception):
                    await t
