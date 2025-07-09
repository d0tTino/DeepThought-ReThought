from fastapi import FastAPI, Response

try:  # pragma: no cover - optional dependency may be missing in tests
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
except Exception:  # pragma: no cover - fallback for minimal stubs
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    def generate_latest() -> bytes:
        return b""


app = FastAPI()


@app.get("/metrics")
def metrics() -> Response:
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)
