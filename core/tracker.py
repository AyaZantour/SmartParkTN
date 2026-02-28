"""
SmartParkTN – Entry / Exit tracker + billing engine
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from loguru import logger
from database.models import EventType, AccessDecision, VehicleCategory
from database.crud import (
    get_vehicle, create_event, get_last_entry,
    compute_bill, check_access,
)


class ParkingTracker:

    def process_detection(
        self,
        db: Session,
        plate: str,
        camera_id: str,
        confidence: float,
        detect_conf: float,
        raw_ocr: str = None,
        image_path: str = None,
    ) -> dict:
        """
        Main entry point called for every plate detection.
        Determines ENTRY or EXIT, checks access, computes billing on exit.
        Returns a result dict with all relevant info for the UI / API.
        """
        plate = plate.upper().strip()
        now = datetime.now()

        # Determine event type from camera ID
        is_entry = "ENTRY" in camera_id.upper() or "ENTREE" in camera_id.upper()
        is_exit  = "EXIT"  in camera_id.upper() or "SORTIE" in camera_id.upper()

        # If camera is neither explicitly entry nor exit, infer from state
        if not is_entry and not is_exit:
            last = get_last_entry(db, plate)
            is_entry = last is None
            is_exit  = not is_entry

        event_type = EventType.ENTRY if is_entry else EventType.EXIT

        # Access control
        access = check_access(db, plate, now)
        decision = access["decision"]
        category = access["category"]
        reason   = access["reason"]

        duration_minutes = None
        amount_tnd = None

        if is_exit:
            last_entry = get_last_entry(db, plate)
            if last_entry:
                delta = (now - last_entry.timestamp).total_seconds() / 60.0
                duration_minutes = round(delta, 2)
                amount_tnd = compute_bill(db, category, duration_minutes)
                reason = (f"Durée: {int(duration_minutes)}min | "
                          f"Montant: {amount_tnd:.2f} TND")

        ev = create_event(
            db,
            plate=plate,
            category=category,
            event_type=event_type,
            timestamp=now,
            camera_id=camera_id,
            confidence=confidence,
            detect_conf=detect_conf,
            raw_ocr_text=raw_ocr,
            decision=decision,
            decision_reason=reason,
            image_path=image_path,
            duration_minutes=duration_minutes,
            amount_tnd=amount_tnd,
        )

        logger.info(
            f"[{camera_id}] {plate} | {event_type.value.upper()} | "
            f"{category.value} | {decision.value} | {reason[:60]}"
        )

        return {
            "event_id":       ev.id,
            "plate":          plate,
            "category":       category.value,
            "event_type":     event_type.value,
            "decision":       decision.value,
            "reason":         reason,
            "duration_min":   duration_minutes,
            "amount_tnd":     amount_tnd,
            "timestamp":      now.isoformat(),
            "confidence":     confidence,
            "detect_conf":    detect_conf,
        }
