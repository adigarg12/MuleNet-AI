"""
SAR endpoints.

GET  /sar/                  — list all SAR drafts
GET  /sar/pending           — pending drafts only
GET  /sar/{sar_id}          — single SAR detail
POST /sar/{sar_id}/approve  — compliance officer approves
POST /sar/{sar_id}/dismiss  — compliance officer dismisses
"""

from fastapi import APIRouter, HTTPException
from typing import List

from api.schemas import SARResponse, SARStatusUpdate
from src.sar.sar_store import sar_store

router = APIRouter(prefix="/sar", tags=["sar"])


def _to_response(sar: dict) -> SARResponse:
    return SARResponse(**{k: sar[k] for k in SARResponse.model_fields if k in sar})


@router.get("", response_model=List[SARResponse])
def list_sars():
    """Return all SAR drafts sorted by creation time descending."""
    return [_to_response(s) for s in sorted(
        sar_store.all(), key=lambda s: s["created_at"], reverse=True
    )]


@router.get("/pending", response_model=List[SARResponse])
def list_pending():
    """Return only pending SAR drafts."""
    return [_to_response(s) for s in sorted(
        sar_store.pending(), key=lambda s: s["created_at"], reverse=True
    )]


@router.get("/{sar_id}", response_model=SARResponse)
def get_sar(sar_id: str):
    sar = sar_store.get(sar_id)
    if not sar:
        raise HTTPException(status_code=404, detail=f"SAR {sar_id} not found")
    return _to_response(sar)


@router.post("/{sar_id}/approve", response_model=SARResponse)
def approve_sar(sar_id: str, body: SARStatusUpdate = SARStatusUpdate()):
    sar = sar_store.update_status(sar_id, "approved", body.reviewed_by)
    if not sar:
        raise HTTPException(status_code=404, detail=f"SAR {sar_id} not found")
    return _to_response(sar)


@router.post("/{sar_id}/dismiss", response_model=SARResponse)
def dismiss_sar(sar_id: str, body: SARStatusUpdate = SARStatusUpdate()):
    sar = sar_store.update_status(sar_id, "dismissed", body.reviewed_by)
    if not sar:
        raise HTTPException(status_code=404, detail=f"SAR {sar_id} not found")
    return _to_response(sar)
