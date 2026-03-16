from __future__ import annotations

import asyncio
import mimetypes
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass
class IngestedArtifact:
    source_url: str
    local_path: str
    content_type: str
    content_length: int
    modality: str | None


class AttachmentIngestionWorker:
    def __init__(
        self,
        *,
        allowed_schemes: Iterable[str] = ("https",),
        allowed_content_types: Iterable[str] = (
            "image/",
            "audio/",
            "video/",
        ),
        max_content_length: int = 50 * 1024 * 1024,
        timeout_seconds: float = 5.0,
        retries: int = 2,
    ) -> None:
        self.allowed_schemes = {s.lower() for s in allowed_schemes}
        self.allowed_content_types = tuple(allowed_content_types)
        self.max_content_length = int(max_content_length)
        self.timeout_seconds = float(timeout_seconds)
        self.retries = max(1, int(retries))

    @staticmethod
    def _infer_modality(content_type: str) -> str | None:
        ctype = (content_type or "").lower()
        if ctype.startswith("image/"):
            return "image"
        if ctype.startswith("audio/"):
            return "audio"
        if ctype.startswith("video/"):
            return "video"
        return None

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in self.allowed_schemes:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        if not parsed.netloc:
            raise ValueError("Attachment URL must include host")

    def _download_once(self, url: str) -> IngestedArtifact:
        self._validate_url(url)
        req = Request(url, headers={"User-Agent": "DeepThoughtIngestionWorker/1.0"})
        with urlopen(req, timeout=self.timeout_seconds) as resp:  # nosec: B310 - validated HTTPS only
            ctype = (resp.headers.get("Content-Type") or "application/octet-stream").split(";", 1)[0].strip().lower()
            clen_header = resp.headers.get("Content-Length")
            if clen_header is not None:
                try:
                    declared = int(clen_header)
                except ValueError as exc:
                    raise ValueError("Invalid Content-Length header") from exc
                if declared > self.max_content_length:
                    raise ValueError("Attachment exceeds maximum allowed content length")

            if not any(ctype.startswith(prefix) for prefix in self.allowed_content_types):
                raise ValueError(f"Disallowed content type: {ctype}")

            ext = mimetypes.guess_extension(ctype) or ""
            with tempfile.NamedTemporaryFile(prefix="dtr_ingest_", suffix=ext, delete=False) as tmp:
                total = 0
                while True:
                    chunk = resp.read(1024 * 128)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.max_content_length:
                        raise ValueError("Attachment exceeded size limit while streaming")
                    # basic safety guard against executable signatures
                    if total <= 8 and chunk.startswith((b"MZ", b"\x7fELF")):
                        raise ValueError("Blocked potentially executable attachment")
                    tmp.write(chunk)
                path = Path(tmp.name)
        return IngestedArtifact(
            source_url=url,
            local_path=str(path),
            content_type=ctype,
            content_length=total,
            modality=self._infer_modality(ctype),
        )

    async def ingest_attachment(self, attachment: dict[str, Any]) -> IngestedArtifact:
        url = str(attachment.get("url") or "").strip()
        if not url:
            raise ValueError("Attachment missing url")

        last_exc: Exception | None = None
        for _ in range(self.retries):
            try:
                return await asyncio.to_thread(self._download_once, url)
            except Exception as exc:  # pragma: no cover - retry path
                last_exc = exc
        assert last_exc is not None
        raise last_exc
