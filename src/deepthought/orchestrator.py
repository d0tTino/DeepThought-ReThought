from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
from contextlib import AsyncExitStack, suppress
from importlib import metadata
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from .config import get_settings
from .eda import Publisher, Subscriber
from .eda.events import EventSubjects, PlanGeneratedPayload, PlanRequestedPayload
from .planning import planner, translator

logger = logging.getLogger(__name__)


def _normalize_event_subject(value: str) -> str:
    return value.rsplit(".", 1)[-1].strip()


def _parse_service_bindings(
    service_bindings: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    publishers: dict[str, set[str]] = {}
    subscribers: dict[str, set[str]] = {}
    for service, spec in service_bindings.items():
        if not isinstance(spec, dict):
            continue
        for binding in spec.get("publish", []) or []:
            if not isinstance(binding, dict):
                continue
            event_subject = binding.get("event_subject")
            if not isinstance(event_subject, str):
                continue
            subject_key = _normalize_event_subject(event_subject)
            publishers.setdefault(subject_key, set()).add(service)
        for binding in spec.get("subscribe", []) or []:
            if not isinstance(binding, dict):
                continue
            event_subject = binding.get("event_subject")
            if not isinstance(event_subject, str):
                continue
            subject_key = _normalize_event_subject(event_subject)
            subscribers.setdefault(subject_key, set()).add(service)
    return publishers, subscribers


def _required_subjects_from_architecture() -> list[dict[str, Any]]:
    arch = Path(__file__).resolve().parents[2] / "docs" / "architecture.md"
    text = arch.read_text(encoding="utf-8")
    required: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not re.match(r"^\d+\.\s+", line):
            continue
        names = re.findall(r"`([^`]+)`", line)
        if not names:
            continue
        consumed = re.findall(r"consumed by `([^`]+)`", line)
        published = re.findall(r"published by `([^`]+)`", line)
        if "either" in line and len(names) >= 3 and consumed:
            required.append(
                {
                    "subjects": [_normalize_event_subject(names[0]), _normalize_event_subject(names[2])],
                    "publishers": {_normalize_event_subject(names[1])},
                    "subscribers": {_normalize_event_subject(consumed[0])},
                    "kind": "any",
                }
            )
            continue
        if " are consumed by " in line and consumed:
            for subject in names[:-1]:
                required.append(
                    {
                        "subjects": [_normalize_event_subject(subject)],
                        "publishers": set(),
                        "subscribers": {_normalize_event_subject(consumed[0])},
                        "kind": "all",
                    }
                )
            continue
        if len(names) >= 3 and published and consumed:
            required.append(
                {
                    "subjects": [_normalize_event_subject(names[0])],
                    "publishers": {_normalize_event_subject(names[1])},
                    "subscribers": {_normalize_event_subject(c) for c in consumed},
                    "kind": "all",
                }
            )
    return required


def _validate_required_bindings(service_bindings: dict[str, Any]) -> None:
    publishers, subscribers = _parse_service_bindings(service_bindings)
    problems: list[str] = []
    for requirement in _required_subjects_from_architecture():
        subjects = requirement["subjects"]
        if requirement["kind"] == "any":
            satisfied = False
            for subject in subjects:
                pub_services = publishers.get(subject, set())
                sub_services = subscribers.get(subject, set())
                if pub_services and requirement["subscribers"].issubset(sub_services):
                    satisfied = True
                    break
            if not satisfied:
                problems.append(
                    "missing alternative edge: expected one of "
                    f"{subjects} to have publisher(s) and subscriber(s) "
                    f"{sorted(requirement['subscribers'])}"
                )
            continue

        for subject in subjects:
            pub_services = publishers.get(subject, set())
            sub_services = subscribers.get(subject, set())
            if not pub_services:
                problems.append(
                    f"missing publisher for {subject}: add a service_bindings.*.publish entry"
                )
            missing_pub = requirement["publishers"] - pub_services
            if missing_pub:
                problems.append(
                    f"{subject} must be published by {sorted(missing_pub)}; found {sorted(pub_services) or 'none'}"
                )
            if not sub_services:
                problems.append(
                    f"missing subscriber for {subject}: add a service_bindings.*.subscribe entry"
                )
            missing_sub = requirement["subscribers"] - sub_services
            if missing_sub:
                problems.append(
                    f"{subject} must be consumed by {sorted(missing_sub)}; found {sorted(sub_services) or 'none'}"
                )

    if problems:
        msg = "Required orchestration edges failed validation:\n - " + "\n - ".join(problems)
        raise ValueError(msg)


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
    service_bindings = cfg.get("service_bindings")
    if service_bindings is None:
        logger.warning(
            "No service_bindings found in config; skipping required orchestration DAG validation"
        )
    elif not isinstance(service_bindings, dict):
        raise ValueError("service_bindings must be a mapping of service name to binding metadata")
    else:
        _validate_required_bindings(service_bindings)
    names = cfg.get("services", [])
    service_classes = discover_services(names)
    crew_specs = cfg.get("crews", [])
    graph_specs = cfg.get("graphs", [])
    if not service_classes and not crew_specs and not graph_specs:
        logger.warning("No services, crews or graphs found in config")
        return
    crew_factories = [_load_callable(s) for s in crew_specs]  # noqa: F841
    graph_factories: list[Callable[[], Awaitable[Any]]] = [  # noqa: F841
        _load_callable(s) for s in graph_specs
    ]
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
        crews = []  # noqa: F841 - reserved for future use
        graph_tasks = []
        desires_file = cfg.get("desires_file", "desires.json")
        for cls in service_classes:
            if cls.__name__ == "PlanningService":
                inst = cls(nc, js, desires_file=desires_file)
            elif cls.__name__ == "ReasoningService":
                inst = cls(nc, js)
            else:
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
        finally:
            for t in graph_tasks:
                t.cancel()
                with suppress(Exception):
                    await t
