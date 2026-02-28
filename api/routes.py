"""
SmartParkTN – FastAPI routes
"""
from __future__ import annotations
import io, base64
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.models import get_db, VehicleCategory
from database.crud import (
    get_vehicle, upsert_vehicle, list_vehicles,
    list_events, check_access, get_tariff,
    list_subscriptions, add_subscription, deactivate_subscription,
    count_currently_parked,
)
from core.pipeline import ALPRPipeline
from core.rag import ParkingAssistant

router = APIRouter()

# Singletons (initialised at startup)
_pipeline:  Optional[ALPRPipeline]   = None
_assistant: Optional[ParkingAssistant] = None


def get_pipeline() -> ALPRPipeline:
    global _pipeline
    if _pipeline is None:
        from database.models import get_db as gdb
        _pipeline = ALPRPipeline(gdb)
    return _pipeline

def get_assistant() -> ParkingAssistant:
    global _assistant
    if _assistant is None:
        _assistant = ParkingAssistant()
    return _assistant


# ── Schemas ───────────────────────────────────────────────────────────────
class VehicleIn(BaseModel):
    plate:      str
    owner_name: Optional[str] = None
    category:   VehicleCategory = VehicleCategory.VISITOR
    notes:      Optional[str] = None

class AskIn(BaseModel):
    question: str

class ExplainIn(BaseModel):
    plate:    str
    decision: str
    reason:   str

class SubscriptionIn(BaseModel):
    plate:      str
    start_date: str   # ISO format YYYY-MM-DD
    end_date:   str   # ISO format YYYY-MM-DD
    zone:       str = "A"


# ── Health ────────────────────────────────────────────────────────────────
@router.get("/health")
def health(db: Session = Depends(get_db)):
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "currently_parked": count_currently_parked(db),
    }


# ── ALPR – process uploaded image ─────────────────────────────────────────
@router.post("/process-image")
async def process_image(
    file:      UploadFile = File(...),
    camera_id: str = Form("CAM_ENTRY_01"),
):
    data  = await file.read()
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image")

    pipeline = get_pipeline()
    pipeline.camera_id = camera_id
    result = pipeline.process_frame(frame)

    # Encode annotated frame as base64 JPEG
    _, buf = cv2.imencode(".jpg", result.pop("annotated_frame"))
    result["annotated_image_b64"] = base64.b64encode(buf).decode()
    return result


# ── ALPR – process base64 frame (for WebSocket / Streamlit) ──────────────
@router.post("/process-frame")
def process_frame(payload: dict):
    b64 = payload.get("frame_b64", "")
    camera_id = payload.get("camera_id", "CAM_ENTRY_01")
    try:
        data  = base64.b64decode(b64)
        arr   = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(400, f"Bad frame: {e}")

    pipeline = get_pipeline()
    pipeline.camera_id = camera_id
    result = pipeline.process_frame(frame)
    _, buf = cv2.imencode(".jpg", result.pop("annotated_frame"))
    result["annotated_image_b64"] = base64.b64encode(buf).decode()
    return result


# ── Vehicles ──────────────────────────────────────────────────────────────
@router.get("/vehicles")
def vehicles(db: Session = Depends(get_db)):
    return [
        {"plate": v.plate, "owner": v.owner_name,
         "category": v.category.value, "active": v.is_active}
        for v in list_vehicles(db)
    ]

@router.get("/vehicles/{plate}")
def vehicle_detail(plate: str, db: Session = Depends(get_db)):
    v = get_vehicle(db, plate.upper())
    if not v:
        raise HTTPException(404, "Vehicle not found")
    access = check_access(db, plate.upper())
    return {
        "plate": v.plate, "owner": v.owner_name,
        "category": v.category.value, "active": v.is_active,
        "notes": v.notes, "access": access["decision"].value,
        "access_reason": access["reason"],
    }

@router.post("/vehicles")
def add_vehicle(payload: VehicleIn, db: Session = Depends(get_db)):
    v = upsert_vehicle(db, payload.plate.upper(), payload.owner_name,
                       payload.category, payload.notes)
    return {"plate": v.plate, "category": v.category.value}

@router.delete("/vehicles/{plate}")
def delete_vehicle(plate: str, db: Session = Depends(get_db)):
    from database.models import Vehicle
    v = get_vehicle(db, plate.upper())
    if not v:
        raise HTTPException(404, "Vehicle not found")
    db.delete(v); db.commit()
    return {"deleted": plate.upper()}


# ── Events ────────────────────────────────────────────────────────────────
@router.get("/events")
def events(limit: int = 100, db: Session = Depends(get_db)):
    evs = list_events(db, limit=limit)
    return [
        {
            "id": e.id, "plate": e.plate, "category": e.category.value,
            "type": e.event_type.value, "decision": e.decision.value,
            "timestamp": e.timestamp.isoformat(),
            "camera": e.camera_id, "duration_min": e.duration_minutes,
            "amount_tnd": e.amount_tnd, "reason": e.decision_reason,
            "ocr_conf": e.confidence, "detect_conf": e.detect_conf,
        }
        for e in evs
    ]


# ── Tariffs ───────────────────────────────────────────────────────────────
@router.get("/tariffs")
def tariffs(db: Session = Depends(get_db)):
    from sqlalchemy import select
    from database.models import Tariff
    ts = db.execute(select(Tariff)).scalars().all()
    return [
        {"category": t.category.value, "price_per_hour": t.price_per_hour,
         "free_minutes": t.free_minutes, "max_daily": t.max_daily,
         "description": t.description}
        for t in ts
    ]


# ── RAG Assistant ─────────────────────────────────────────────────────────
@router.post("/assistant/ask")
def ask(payload: AskIn):
    assistant = get_assistant()
    answer = assistant.query(payload.question)
    return {"question": payload.question, "answer": answer}

@router.post("/assistant/explain")
def explain(payload: ExplainIn):
    assistant = get_assistant()
    explanation = assistant.explain_decision(
        payload.plate, payload.decision, payload.reason
    )
    return {"plate": payload.plate, "explanation": explanation}

@router.post("/assistant/ingest")
def ingest_rules():
    assistant = get_assistant()
    assistant.ingest_documents()
    return {"status": "ingestion complete"}


# ── Subscriptions ─────────────────────────────────────────────────────────
@router.get("/subscriptions")
def get_subscriptions(db: Session = Depends(get_db)):
    subs = list_subscriptions(db)
    return [
        {
            "id": s.id, "plate": s.plate,
            "start_date": s.start_date.isoformat(),
            "end_date": s.end_date.isoformat(),
            "zone": s.zone, "active": s.is_active,
        }
        for s in subs
    ]

@router.post("/subscriptions")
def create_subscription(payload: SubscriptionIn, db: Session = Depends(get_db)):
    try:
        start = datetime.fromisoformat(payload.start_date)
        end   = datetime.fromisoformat(payload.end_date)
    except ValueError as e:
        raise HTTPException(400, f"Invalid date format: {e}")
    if end <= start:
        raise HTTPException(400, "end_date must be after start_date")
    sub = add_subscription(db, payload.plate, start, end, payload.zone)
    return {
        "id": sub.id, "plate": sub.plate,
        "start_date": sub.start_date.isoformat(),
        "end_date": sub.end_date.isoformat(),
        "zone": sub.zone, "active": sub.is_active,
    }

@router.delete("/subscriptions/{sub_id}")
def cancel_subscription(sub_id: int, db: Session = Depends(get_db)):
    ok = deactivate_subscription(db, sub_id)
    if not ok:
        raise HTTPException(404, "Subscription not found")
    return {"status": "cancelled", "id": sub_id}

@router.get("/stats/summary")
def stats_summary(db: Session = Depends(get_db)):
    """Quick summary for dashboard: parked, total today, revenue today."""
    from sqlalchemy import func, cast, Date
    from database.models import ParkingEvent as PE, AccessDecision as AD
    today = datetime.utcnow().date()
    events_today = db.execute(
        select(func.count()).select_from(PE).where(
            cast(PE.timestamp, Date) == today
        )
    ).scalar() or 0
    revenue_today = db.execute(
        select(func.coalesce(func.sum(PE.amount_tnd), 0.0)).where(
            cast(PE.timestamp, Date) == today
        )
    ).scalar() or 0.0
    return {
        "currently_parked": count_currently_parked(db),
        "events_today": events_today,
        "revenue_today": round(float(revenue_today), 2),
    }

