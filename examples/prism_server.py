"""Minimal Prism server implemented with FastAPI."""

import os

from fastapi import Depends, FastAPI, Header, HTTPException, status


app = FastAPI()


def _get_valid_tokens() -> list[str]:
    """Return the list of valid authorization tokens."""
    tokens = os.getenv("PRISM_TOKENS") or os.getenv("PRISM_TOKEN") or ""
    return [t.strip() for t in tokens.split(",") if t.strip()]


def verify_authorization(authorization: str = Header(...)) -> None:
    """FastAPI dependency that validates the ``Authorization`` header."""
    valid_tokens = _get_valid_tokens()
    token = authorization.replace("Bearer", "").strip()
    if token not in valid_tokens:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


@app.post("/receive_data")
async def receive_data(data: dict, _: None = Depends(verify_authorization)) -> dict:
    """Receive a JSON payload from the Discord bot."""
    print(f"Received from bot: {data}")
    return {"detail": "Data received"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PRISM_PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
