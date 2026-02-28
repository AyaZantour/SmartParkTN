"""
SmartParkTN – License-plate detector (YOLOv8)
Uses ultralytics YOLOv8n fine-tuned on license plates.
Falls back to full-frame crop if no YOLO model is found.
"""
from __future__ import annotations
import os, cv2
import numpy as np
from typing import List, Tuple
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Public YOLO LP model (license-plate fine-tuned) – auto-downloads ~6 MB
_DEFAULT_MODEL = "keremberke/yolov8n-license-plate-extraction"
_LOCAL_WEIGHTS = os.getenv("YOLO_WEIGHTS", "./models/plate_detector.pt")
_CONF = float(os.getenv("PLATE_DETECT_CONF", "0.40"))


class PlateDetector:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if os.path.isfile(_LOCAL_WEIGHTS):
                self.model = YOLO(_LOCAL_WEIGHTS)
                logger.info(f"Loaded local YOLO: {_LOCAL_WEIGHTS}")
            else:
                # Use a HuggingFace hosted plate-detection model
                self.model = YOLO(_DEFAULT_MODEL)
                logger.info("Loaded HuggingFace plate detector")
        except Exception as e:
            logger.warning(f"YOLO load failed ({e}). Using fallback detector.")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float, Tuple]]:
        """
        Returns list of (cropped_plate_img, confidence, bbox_xyxy).
        If YOLO unavailable, returns the full frame as fallback.
        """
        if self.model is None:
            return self._fallback(frame)

        results = self.model.predict(
            frame, conf=_CONF, verbose=False, device="0"
        )
        crops = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append((crop, conf, (x1, y1, x2, y2)))
        return crops if crops else self._fallback(frame)

    @staticmethod
    def _fallback(frame: np.ndarray):
        """Return full frame with conf=0 as a fallback crop."""
        h, w = frame.shape[:2]
        return [(frame.copy(), 0.0, (0, 0, w, h))]

    def draw(self, frame: np.ndarray, detections, label: str = "",
             color=(0, 255, 0)) -> np.ndarray:
        out = frame.copy()
        for _, conf, (x1, y1, x2, y2) in detections:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            txt = f"{label}  [{conf:.2f}]" if label else f"{conf:.2f}"
            cv2.putText(out, txt, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return out
