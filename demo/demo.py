"""
SmartParkTN – Video Demo Script
================================
Processes a video file (or webcam) frame-by-frame through the full ALPR
pipeline and saves an annotated output video.

Usage:
    python demo/demo.py --input demo/test_video.mp4 --output demo/output.mp4
    python demo/demo.py --input 0          # webcam
    python demo/demo.py --simulate         # synthetic simulation (no real video needed)
"""
import sys, os, argparse, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from datetime import datetime
from loguru import logger

from database.models import init_db, get_db
from database.crud import upsert_vehicle
from database.models import VehicleCategory

# CATEGORY COLOR MAP (BGR)
COLORS = {
    "visitor":    (0, 200, 255),
    "subscriber": (0, 255, 100),
    "vip":        (255, 215, 0),
    "blacklist":  (0, 0, 255),
    "employee":   (255, 165, 0),
    "emergency":  (0, 165, 255),
}
DEC_COLORS = {"allowed": (0, 220, 0), "denied": (0, 0, 220), "pending": (200, 200, 0)}

SAMPLE_PLATES = [
    ("100 TN 1234", VehicleCategory.VISITOR),
    ("200 TN 5678", VehicleCategory.SUBSCRIBER),
    ("300 TN 9012", VehicleCategory.VIP),
    ("400 TN 3456", VehicleCategory.EMPLOYEE),
    ("500 TN 7890", VehicleCategory.BLACKLIST),
    ("777 TN 8888", VehicleCategory.VISITOR),
    ("111 TN 2222", VehicleCategory.EMERGENCY),
]


def draw_overlay(frame, result: dict, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    plate    = result.get("plate") or "—"
    category = result.get("category", "visitor")
    decision = result.get("decision", "pending")
    reason   = result.get("reason", "")
    dur      = result.get("duration_min")
    amt      = result.get("amount_tnd")
    det_conf = result.get("detect_conf", 0.0)
    ocr_conf = result.get("confidence", 0.0)

    panel_h = 160
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    dec_color = DEC_COLORS.get(decision, (200, 200, 0))
    cat_color = COLORS.get(category, (200, 200, 200))

    cv2.putText(out, plate, (20, h - panel_h + 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)

    cv2.putText(out, f"Cat: {category.upper()}", (20, h - panel_h + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, cat_color, 2)
    cv2.putText(out, f"Decision: {decision.upper()}", (20, h - panel_h + 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, dec_color, 2)

    if reason:
        cv2.putText(out, reason[:60], (20, h - panel_h + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Right side
    rt = w - 300
    cv2.putText(out, f"FPS: {fps:.1f}", (rt, h - panel_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)
    cv2.putText(out, f"Det: {det_conf:.0%}  OCR: {ocr_conf:.0%}",
                (rt, h - panel_h + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    if amt is not None:
        cv2.putText(out, f"Duree: {dur:.0f}min  {amt:.2f} TND",
                    (rt, h - panel_h + 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 230, 100), 2)

    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(out, f"SmartParkTN  |  {ts}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Horizontal line
    cv2.line(out, (0, h - panel_h), (w, h - panel_h), (50, 50, 150), 2)
    return out


def run_simulation(output_path: str, duration_sec: int = 30):
    """Generate a synthetic demo video with fake plate detections."""
    logger.info("Running SIMULATION mode …")
    init_db()
    db = next(get_db())

    # Seed plates
    for plate, cat in SAMPLE_PLATES:
        upsert_vehicle(db, plate, category=cat)

    from core.tracker import ParkingTracker
    tracker = ParkingTracker()

    W, H = 1280, 720
    fps  = 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    total_frames = fps * duration_sec
    result = {}

    for f_idx in range(total_frames):
        # Simulate frame
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Gradient background
        for row in range(H):
            b = int(20 + (row / H) * 40)
            frame[row, :] = (b, b + 10, b + 20)

        # Every 3 seconds: new detection
        if f_idx % (fps * 3) == 0:
            plate, cat = random.choice(SAMPLE_PLATES)
            camera = random.choice(["CAM_ENTRY_01", "CAM_EXIT_01"])
            det_conf = round(random.uniform(0.70, 0.99), 2)
            ocr_conf = round(random.uniform(0.72, 0.98), 2)

            try:
                result = tracker.process_detection(
                    db, plate, camera, ocr_conf, det_conf, plate
                )
            except Exception as e:
                result = {"plate": plate, "category": cat.value,
                          "decision": "pending", "reason": str(e),
                          "confidence": ocr_conf, "detect_conf": det_conf}

            # Draw fake bounding box
            bx, by = random.randint(300, 600), random.randint(H // 3, H // 2)
            bw, bh = 350, 120
            dec_color = DEC_COLORS.get(result.get("decision", "pending"), (200, 200, 0))
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), dec_color, 3)
            cv2.putText(frame, plate, (bx + 10, by + 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        # Draw overlay with last result
        if result:
            annotated = draw_overlay(frame, result, fps)
        else:
            annotated = frame.copy()
            cv2.putText(annotated, "SmartParkTN – En attente …",
                        (W // 4, H // 2), cv2.FONT_HERSHEY_DUPLEX,
                        1.0, (200, 200, 200), 2)

        writer.write(annotated)

        if f_idx % fps == 0:
            logger.info(f"  Frame {f_idx}/{total_frames}  "
                        f"({f_idx // fps}s / {duration_sec}s)")

    writer.release()
    db.close()
    logger.info(f"Demo saved → {output_path}")


def run_video(input_path, output_path: str, camera_id: str):
    """Run ALPR pipeline on a real video file."""
    init_db()
    from core.pipeline import ALPRPipeline
    pipeline = ALPRPipeline(get_db, camera_id=camera_id)

    cap = cv2.VideoCapture(int(input_path) if input_path.isdigit() else input_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_path}"); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed = time.time() - t0
        current_fps = frame_count / elapsed if elapsed > 0 else fps

        result = pipeline.process_frame(frame)
        annotated = draw_overlay(
            result.get("annotated_frame", frame), result, current_fps
        )
        writer.write(annotated)

        if frame_count % 50 == 0:
            logger.info(f"  Processed {frame_count} frames  ({current_fps:.1f} FPS)")

    cap.release()
    writer.release()
    logger.info(f"Output saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartParkTN Video Demo")
    parser.add_argument("--input",    default="0",           help="Video path or 0 for webcam")
    parser.add_argument("--output",   default="demo/output_demo.mp4", help="Output video path")
    parser.add_argument("--camera",   default="CAM_ENTRY_01")
    parser.add_argument("--simulate", action="store_true",   help="Run synthetic simulation")
    parser.add_argument("--duration", type=int, default=30,  help="Simulation duration (sec)")
    args = parser.parse_args()

    if args.simulate:
        run_simulation(args.output, args.duration)
    else:
        run_video(args.input, args.output, args.camera)
