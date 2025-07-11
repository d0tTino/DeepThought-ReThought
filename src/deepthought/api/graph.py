from __future__ import annotations

"""Simple GraphDAL API endpoints."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..config import get_settings
from ..graph import GraphConnector, GraphDAL

router = APIRouter(prefix="/graph")

_dal: GraphDAL | None = None


def _create_dal() -> GraphDAL:
    settings = get_settings()
    connector = GraphConnector(
        host=settings.mg_host,
        port=settings.mg_port,
        username=settings.mg_user,
        password=settings.mg_password,
    )
    return GraphDAL(connector)


def get_dal() -> GraphDAL:
    global _dal
    if _dal is None:
        _dal = _create_dal()
    return _dal


class AddEntityRequest(BaseModel):
    label: str
    props: Dict[str, Any] = {}


class AddRelationshipRequest(BaseModel):
    start_id: str
    end_id: str
    rel_type: str
    props: Dict[str, Any] = {}


class QueryRequest(BaseModel):
    query: str
    params: Dict[str, Any] = {}


@router.post("/entity")
def add_entity(req: AddEntityRequest, dal: GraphDAL = Depends(get_dal)) -> Dict[str, str]:
    try:
        dal.add_entity(req.label, req.props)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok"}


@router.post("/relationship")
def add_relationship(req: AddRelationshipRequest, dal: GraphDAL = Depends(get_dal)) -> Dict[str, str]:
    try:
        dal.add_relationship(req.start_id, req.end_id, req.rel_type, req.props)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok"}


@router.post("/query")
def run_query(req: QueryRequest, dal: GraphDAL = Depends(get_dal)) -> Dict[str, List[Any]]:
    try:
        rows = dal.query_subgraph(req.query, req.params)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc))
    return {"results": rows}
