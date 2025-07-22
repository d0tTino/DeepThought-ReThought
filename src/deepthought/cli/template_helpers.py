from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

__all__ = ["find_template", "apply_bus_substitutions"]


def find_template(template_name: str) -> Path:
    """Return the path to a bundled template."""
    templ_res = resources.files("deepthought.templates").joinpath(template_name)
    with resources.as_file(templ_res) as path:
        if not path.exists():
            raise SystemExit("Template not found")
        return Path(path)


def apply_bus_substitutions(
    text: str,
    *,
    service_name: str | None = None,
    stream_name: str | None = None,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    tls_ca: str | None = None,
    js_storage: str | None = None,
    max_msgs: int | None = None,
) -> str:
    """Replace placeholders for bus service templates."""

    def _quote(value: str) -> str:
        """Return a Dockerfile/.env-safe quoted value."""
        return json.dumps(value)

    if service_name is not None:
        text = text.replace("${SERVICE_NAME}", service_name)
        text = text.replace("template_service", service_name)
        text = text.replace("template-service", service_name.replace("_", "-"))
    if stream_name is not None:
        text = text.replace("${STREAM_NAME}", stream_name)
        text = text.replace("deepthought_events", stream_name)
    if tls_cert is not None:
        text = text.replace("${TLS_CERT}", tls_cert)
        q_cert = _quote(tls_cert)
        text = text.replace("NATS_TLS_CERT=", f"NATS_TLS_CERT={q_cert}")
        text = text.replace("ENV NATS_TLS_CERT=", f"ENV NATS_TLS_CERT={q_cert}")
    if tls_key is not None:
        text = text.replace("${TLS_KEY}", tls_key)
        q_key = _quote(tls_key)
        text = text.replace("NATS_TLS_KEY=", f"NATS_TLS_KEY={q_key}")
        text = text.replace("ENV NATS_TLS_KEY=", f"ENV NATS_TLS_KEY={q_key}")
    if tls_ca is not None:
        text = text.replace("${TLS_CA}", tls_ca)
        q_ca = _quote(tls_ca)
        text = text.replace("NATS_TLS_CA=", f"NATS_TLS_CA={q_ca}")
        text = text.replace("ENV NATS_TLS_CA=", f"ENV NATS_TLS_CA={q_ca}")
    if js_storage is not None:
        q_js = _quote(js_storage)
        text = text.replace("NATS_JS_STORAGE=memory", f"NATS_JS_STORAGE={q_js}")
        text = text.replace("ENV NATS_JS_STORAGE=memory", f"ENV NATS_JS_STORAGE={q_js}")
    if max_msgs is not None:
        q_msgs = _quote(str(max_msgs))
        text = text.replace("NATS_MAX_MSGS=10000", f"NATS_MAX_MSGS={q_msgs}")
        text = text.replace("ENV NATS_MAX_MSGS=10000", f"ENV NATS_MAX_MSGS={q_msgs}")
    return text
