"""
SmartParkTN – Full ALPR Pipeline
Combines detector + OCR + tracker for one-call processing.
"""
from __future__ import annotations
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
import os, time
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

_SAVE_DIR = "./assets/captures"
os.makedirs(_SAVE_DIR, exist_ok=True)


# Category colors (BGR)
CATEGORY_COLORS = {
    "visitor":    (0, 200, 255),
    "subscriber": (0, 255, 100),
    "vip":        (255, 215, 0),
    "blacklist":  (0, 0, 255),
    "employee":   (255, 165, 0),
    "emergency":  (0, 165, 255),
}

DECISION_COLORS = {
    "allowed": (0, 220, 0),
    "denied":  (0, 0, 220),
    "pending": (0, 165, 255),
}


class ALPRPipeline:
    def __init__(self, db_session_factory, camera_id: str = "CAM_ENTRY_01"):
        from core.detector import PlateDetector
        from core.ocr     import PlateOCR
        from core.tracker import ParkingTracker

        self.detector   = PlateDetector()
        self.ocr        = PlateOCR()
        self.tracker    = ParkingTracker()
        self.db_factory = db_session_factory
        self.camera_id  = camera_id
        logger.info(f"ALPRPipeline ready (camera: {camera_id})")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Full pipeline for one frame.
        Returns result dict; annotated frame added under 'annotated_frame'.
        """
        annotated = frame.copy()
        result = {
            "plate": None, "confidence": 0.0, "detect_conf": 0.0,
            "decision": "pending", "category": "visitor",
            "reason": "No plate detected", "duration_min": None,
            "amount_tnd": None, "timestamp": datetime.now().isoformat(),
            "annotated_frame": annotated,
        }

        detections = self.detector.detect(frame)
        if not detections:
            return result

        # Take highest-confidence detection
        detections.sort(key=lambda x: x[1], reverse=True)
        crop, det_conf, bbox = detections[0]

        plate_text, ocr_conf = self.ocr.read(crop)
        if not plate_text:
            return result

        # Save crop
        img_path = self._save_crop(crop, plate_text)

        db = next(self.db_factory())
        try:
            res = self.tracker.process_detection(
                db, plate_text, self.camera_id,
                confidence=ocr_conf, detect_conf=det_conf,
                raw_ocr=plate_text, image_path=img_path,
            )
        finally:
            db.close()

        result.update(res)

        # Annotate frame
        cat   = res.get("category", "visitor")
        dec   = res.get("decision", "pending")
        color = DECISION_COLORS.get(dec, (255, 255, 255))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"{plate_text} | {cat.upper()} | {dec.upper()}"
        cv2.putText(annotated, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)

        if res.get("amount_tnd") is not None:
            bill = f"{res['duration_min']:.0f}min | {res['amount_tnd']:.2f} TND"
            cv2.putText(annotated, bill, (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        result["annotated_frame"] = annotated
        return result

    @staticmethod
    def _save_crop(img: np.ndarray, plate: str) -> str:
        ts   = int(time.time())
        safe = plate.replace(" ", "_")
        path = os.path.join(_SAVE_DIR, f"{safe}_{ts}.jpg")
        cv2.imwrite(path, img)
        return path
