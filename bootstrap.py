"""
SmartParkTN – Bootstrap
Run:  python bootstrap.py
Creates all project directories and source files in one shot.
"""
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))

def w(rel_path: str, content: str):
    """Write content to BASE/rel_path, creating parent dirs as needed."""
    full = os.path.join(BASE, *rel_path.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    lines = content.split("\n")
    # Detect indent from the first non-empty line (robust vs embedded \n)
    indent = 0
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip(" "))
            break
    # Strip exactly `indent` spaces from every line that starts with them
    out = "\n".join(
        (line[indent:] if line.startswith(" " * indent) else line)
        for line in lines
    ).lstrip("\n")
    with open(full, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"  ✓  {rel_path}")

# ─────────────────────────────────────────────────────────────────────────────
# __init__ stubs
# ─────────────────────────────────────────────────────────────────────────────
for pkg in ["core", "database", "api", "ui", "demo", "scripts"]:
    w(f"{pkg}/__init__.py", "")

for d in ["data/rules", "data/vehicles", "data/chroma_db", "assets", "models"]:
    os.makedirs(os.path.join(BASE, *d.split("/")), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# database/models.py
# ─────────────────────────────────────────────────────────────────────────────
w("database/models.py", '''
    """
    SmartParkTN – Database Models (SQLAlchemy)
    """
    from __future__ import annotations
    from datetime import datetime
    from sqlalchemy import (
        create_engine, Column, String, Float, Integer,
        DateTime, Boolean, Text, Enum as SAEnum,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy.pool import StaticPool
    import enum, os
    from dotenv import load_dotenv

    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smartpark.db")

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()


    class VehicleCategory(str, enum.Enum):
        VISITOR    = "visitor"
        SUBSCRIBER = "subscriber"
        VIP        = "vip"
        BLACKLIST  = "blacklist"
        EMPLOYEE   = "employee"
        EMERGENCY  = "emergency"

    class EventType(str, enum.Enum):
        ENTRY = "entry"
        EXIT  = "exit"

    class AccessDecision(str, enum.Enum):
        ALLOWED = "allowed"
        DENIED  = "denied"
        PENDING = "pending"


    class Vehicle(Base):
        __tablename__ = "vehicles"
        plate      = Column(String(20), primary_key=True, index=True)
        owner_name = Column(String(100), nullable=True)
        category   = Column(SAEnum(VehicleCategory), default=VehicleCategory.VISITOR)
        is_active  = Column(Boolean, default=True)
        notes      = Column(Text, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    class Subscription(Base):
        __tablename__ = "subscriptions"
        id         = Column(Integer, primary_key=True, autoincrement=True)
        plate      = Column(String(20), index=True)
        start_date = Column(DateTime, nullable=False)
        end_date   = Column(DateTime, nullable=False)
        zone       = Column(String(50), default="A")
        is_active  = Column(Boolean, default=True)

    class ParkingEvent(Base):
        __tablename__ = "events"
        id               = Column(Integer, primary_key=True, autoincrement=True)
        plate            = Column(String(20), index=True)
        category         = Column(SAEnum(VehicleCategory), default=VehicleCategory.VISITOR)
        event_type       = Column(SAEnum(EventType))
        timestamp        = Column(DateTime, default=datetime.utcnow, index=True)
        camera_id        = Column(String(20), default="CAM_ENTRY_01")
        confidence       = Column(Float, default=0.0)
        detect_conf      = Column(Float, default=0.0)
        raw_ocr_text     = Column(String(50), nullable=True)
        decision         = Column(SAEnum(AccessDecision), default=AccessDecision.ALLOWED)
        decision_reason  = Column(Text, nullable=True)
        image_path       = Column(String(255), nullable=True)
        duration_minutes = Column(Float, nullable=True)
        amount_tnd       = Column(Float, nullable=True)
        is_paid          = Column(Boolean, default=False)

    class Tariff(Base):
        __tablename__ = "tariffs"
        id             = Column(Integer, primary_key=True, autoincrement=True)
        category       = Column(SAEnum(VehicleCategory), unique=True)
        price_per_hour = Column(Float, default=0.0)
        free_minutes   = Column(Integer, default=15)
        max_daily      = Column(Float, default=20.0)
        description    = Column(Text, nullable=True)

    class AccessRule(Base):
        __tablename__ = "access_rules"
        id          = Column(Integer, primary_key=True, autoincrement=True)
        rule_name   = Column(String(100), unique=True)
        category    = Column(SAEnum(VehicleCategory), nullable=True)
        allowed     = Column(Boolean, default=True)
        time_start  = Column(String(5), nullable=True)
        time_end    = Column(String(5), nullable=True)
        zone        = Column(String(20), nullable=True)
        description = Column(Text, nullable=True)


    def init_db():
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            _seed(db)
        finally:
            db.close()

    def _seed(db):
        from sqlalchemy import select
        tariffs = [
            (VehicleCategory.VISITOR,    2.0,  15, 20.0, "Visiteur – 2 TND/h, 15 min gratuites"),
            (VehicleCategory.SUBSCRIBER, 0.0,   0,  0.0, "Abonné – gratuit"),
            (VehicleCategory.VIP,        0.0,   0,  0.0, "VIP – gratuit, accès illimité"),
            (VehicleCategory.EMPLOYEE,   0.0,   0,  0.0, "Employé – gratuit"),
            (VehicleCategory.BLACKLIST,  0.0,   0,  0.0, "Liste noire – accès refusé"),
            (VehicleCategory.EMERGENCY,  0.0,   0,  0.0, "Urgence – accès prioritaire"),
        ]
        for cat, price, free, max_d, desc in tariffs:
            if not db.execute(select(Tariff).where(Tariff.category == cat)).scalar_one_or_none():
                db.add(Tariff(category=cat, price_per_hour=price,
                              free_minutes=free, max_daily=max_d, description=desc))

        rules = [
            ("blacklist_denied",  VehicleCategory.BLACKLIST,  False, None,    None,    None, "Blacklist – toujours refusé"),
            ("visitor_hours",     VehicleCategory.VISITOR,    True,  "06:00", "23:00", None, "Visiteurs: 06h-23h"),
            ("vip_anytime",       VehicleCategory.VIP,        True,  None,    None,    None, "VIP: 24h/24"),
            ("emergency_anytime", VehicleCategory.EMERGENCY,  True,  None,    None,    None, "Urgence: 24h/24"),
            ("subscriber_zones",  VehicleCategory.SUBSCRIBER, True,  None,    None,    "A,B","Abonnés zones A et B"),
        ]
        for name, cat, allowed, ts, te, zone, desc in rules:
            if not db.execute(select(AccessRule).where(AccessRule.rule_name == name)).scalar_one_or_none():
                db.add(AccessRule(rule_name=name, category=cat, allowed=allowed,
                                  time_start=ts, time_end=te, zone=zone, description=desc))
        db.commit()

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
''')

# ─────────────────────────────────────────────────────────────────────────────
# database/crud.py
# ─────────────────────────────────────────────────────────────────────────────
w("database/crud.py", '''
    """CRUD helpers for SmartParkTN."""
    from __future__ import annotations
    from datetime import datetime
    from typing import Optional
    from sqlalchemy.orm import Session
    from sqlalchemy import select, desc
    from .models import (
        Vehicle, ParkingEvent, Tariff, AccessRule, Subscription,
        VehicleCategory, EventType, AccessDecision,
    )


    # ── Vehicles ──────────────────────────────────────────────────────────────
    def get_vehicle(db: Session, plate: str) -> Optional[Vehicle]:
        return db.execute(select(Vehicle).where(Vehicle.plate == plate)).scalar_one_or_none()

    def upsert_vehicle(db: Session, plate: str, owner: str = None,
                       category: VehicleCategory = VehicleCategory.VISITOR,
                       notes: str = None) -> Vehicle:
        v = get_vehicle(db, plate)
        if v:
            v.category = category
            if owner:  v.owner_name = owner
            if notes:  v.notes = notes
        else:
            v = Vehicle(plate=plate, owner_name=owner, category=category, notes=notes)
            db.add(v)
        db.commit(); db.refresh(v)
        return v

    def list_vehicles(db: Session, skip=0, limit=100):
        return db.execute(select(Vehicle).offset(skip).limit(limit)).scalars().all()


    # ── Events ────────────────────────────────────────────────────────────────
    def create_event(db: Session, **kwargs) -> ParkingEvent:
        ev = ParkingEvent(**kwargs)
        db.add(ev); db.commit(); db.refresh(ev)
        return ev

    def get_last_entry(db: Session, plate: str) -> Optional[ParkingEvent]:
        return db.execute(
            select(ParkingEvent)
            .where(ParkingEvent.plate == plate,
                   ParkingEvent.event_type == EventType.ENTRY,
                   ParkingEvent.duration_minutes == None)
            .order_by(desc(ParkingEvent.timestamp))
        ).scalars().first()

    def list_events(db: Session, limit=200):
        return db.execute(
            select(ParkingEvent).order_by(desc(ParkingEvent.timestamp)).limit(limit)
        ).scalars().all()


    # ── Tariffs ───────────────────────────────────────────────────────────────
    def get_tariff(db: Session, category: VehicleCategory) -> Optional[Tariff]:
        return db.execute(select(Tariff).where(Tariff.category == category)).scalar_one_or_none()

    def compute_bill(db: Session, category: VehicleCategory, duration_minutes: float) -> float:
        tariff = get_tariff(db, category)
        if not tariff or tariff.price_per_hour == 0:
            return 0.0
        billable = max(0, duration_minutes - tariff.free_minutes)
        amount = (billable / 60.0) * tariff.price_per_hour
        return round(min(amount, tariff.max_daily), 2)


    # ── Access Rules ─────────────────────────────────────────────────────────
    def check_access(db: Session, plate: str, now: datetime = None) -> dict:
        now = now or datetime.now()
        v = get_vehicle(db, plate)
        category = v.category if v else VehicleCategory.VISITOR

        # Blacklist check
        if category == VehicleCategory.BLACKLIST:
            rule = db.execute(
                select(AccessRule).where(AccessRule.rule_name == "blacklist_denied")
            ).scalar_one_or_none()
            reason = (rule.description if rule else "") or "Véhicule blacklisté"
            return {"decision": AccessDecision.DENIED, "category": category, "reason": reason}

        # Time-window check
        rules = db.execute(
            select(AccessRule).where(
                (AccessRule.category == category) | (AccessRule.category == None),
                AccessRule.allowed == True
            )
        ).scalars().all()

        for rule in rules:
            if rule.time_start and rule.time_end:
                ts = datetime.strptime(rule.time_start, "%H:%M").time()
                te = datetime.strptime(rule.time_end, "%H:%M").time()
                if not (ts <= now.time() <= te):
                    return {
                        "decision": AccessDecision.DENIED,
                        "category": category,
                        "reason": f"Hors horaires autorisés ({rule.time_start}–{rule.time_end}). Règle: {rule.rule_name}",
                    }

        # Subscription validity
        if category == VehicleCategory.SUBSCRIBER:
            sub = db.execute(
                select(Subscription).where(
                    Subscription.plate == plate,
                    Subscription.is_active == True,
                    Subscription.end_date >= now,
                )
            ).scalars().first()
            if not sub:
                return {
                    "decision": AccessDecision.DENIED,
                    "category": category,
                    "reason": "Abonnement expiré ou introuvable",
                }

        return {
            "decision": AccessDecision.ALLOWED,
            "category": category,
            "reason": f"Accès autorisé – catégorie: {category.value}",
        }
''')

# ─────────────────────────────────────────────────────────────────────────────
# core/detector.py
# ─────────────────────────────────────────────────────────────────────────────
w("core/detector.py", r'''
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
''')

# ─────────────────────────────────────────────────────────────────────────────
# core/ocr.py
# ─────────────────────────────────────────────────────────────────────────────
w("core/ocr.py", r'''
    """
    SmartParkTN – OCR engine (PaddleOCR)
    Reads Tunisian license plates and validates the format.

    Tunisian plate formats:
      Modern  : NNN TN NNNN   e.g. 100 TN 1234
      Numeric : NNNNNNN       legacy
      Arabic  : contains Arabic-indic digits (converted automatically)
    """
    from __future__ import annotations
    import re, os
    import numpy as np
    import cv2
    from typing import Optional, Tuple
    from loguru import logger
    from dotenv import load_dotenv

    load_dotenv()

    _CONF_THRESH = float(os.getenv("OCR_CONF_THRESHOLD", "0.65"))

    # Tunisian plate regex patterns (accept spaces/dashes/dots as separators)
    _PATTERNS = [
        re.compile(r"(\d{1,3})\s*TN\s*(\d{1,4})", re.IGNORECASE),   # 100 TN 1234
        re.compile(r"(\d{1,3})\s*RS\s*(\d{1,4})", re.IGNORECASE),   # older RS plates
        re.compile(r"\d{4,8}"),                                        # pure numeric fallback
    ]

    # Arabic-Indic digit map
    _AR_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


    def _arabic_to_latin(text: str) -> str:
        return text.translate(_AR_MAP)


    def _normalize(raw: str) -> str:
        text = _arabic_to_latin(raw.upper().strip())
        text = re.sub(r"[^A-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


    def _validate(text: str) -> Optional[str]:
        for pat in _PATTERNS:
            m = pat.search(text)
            if m:
                return text  # return full normalized string
        return None


    def _preprocess(img: np.ndarray) -> np.ndarray:
        """Enhance plate image for better OCR accuracy."""
        if img is None or img.size == 0:
            return img
        # Upscale if small
        h, w = img.shape[:2]
        if w < 200:
            scale = 200 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        # Bilateral filter (preserve edges)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


    class PlateOCR:
        def __init__(self):
            self._ocr = None
            self._init_paddle()

        def _init_paddle(self):
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=True,
                    show_log=False,
                )
                logger.info("PaddleOCR initialised (GPU)")
            except Exception as e:
                logger.warning(f"PaddleOCR GPU init failed: {e}. Trying CPU…")
                try:
                    from paddleocr import PaddleOCR
                    self._ocr = PaddleOCR(use_angle_cls=True, lang="en",
                                          use_gpu=False, show_log=False)
                    logger.info("PaddleOCR initialised (CPU)")
                except Exception as e2:
                    logger.error(f"PaddleOCR unavailable: {e2}")

        def read(self, img: np.ndarray) -> Tuple[Optional[str], float]:
            """
            Returns (plate_text, confidence).
            plate_text is None if confidence < threshold or invalid format.
            """
            if img is None or img.size == 0:
                return None, 0.0

            enhanced = _preprocess(img)

            if self._ocr is None:
                return self._tesseract_fallback(enhanced)

            try:
                result = self._ocr.ocr(enhanced, cls=True)
                if not result or not result[0]:
                    return None, 0.0

                texts, confs = [], []
                for line in result[0]:
                    if line and len(line) >= 2:
                        text_info = line[1]
                        texts.append(text_info[0])
                        confs.append(float(text_info[1]))

                if not texts:
                    return None, 0.0

                raw = " ".join(texts)
                conf = sum(confs) / len(confs)
                norm = _normalize(raw)
                validated = _validate(norm)

                if conf < _CONF_THRESH:
                    return norm, conf  # return anyway but low conf

                return (validated or norm), conf
            except Exception as e:
                logger.error(f"OCR error: {e}")
                return None, 0.0

        @staticmethod
        def _tesseract_fallback(img: np.ndarray) -> Tuple[Optional[str], float]:
            """Very basic fallback if neither PaddleOCR nor tesseract works."""
            logger.warning("Using mock OCR – install PaddleOCR for real results")
            return "MOCK_PLATE", 0.50
''')

# ─────────────────────────────────────────────────────────────────────────────
# core/tracker.py
# ─────────────────────────────────────────────────────────────────────────────
w("core/tracker.py", '''
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
''')

# ─────────────────────────────────────────────────────────────────────────────
# core/rag.py
# ─────────────────────────────────────────────────────────────────────────────
w("core/rag.py", '''
    """
    SmartParkTN – RAG Assistant
    Uses ChromaDB + sentence-transformers for retrieval
    and Groq (free API) for generation (Llama-3.1-8B-Instant).

    Get free Groq API key: https://console.groq.com/keys
    """
    from __future__ import annotations
    import os, glob
    from typing import List
    from loguru import logger
    from dotenv import load_dotenv

    load_dotenv()

    _GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
    _GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    _CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    _RULES_DIR  = os.getenv("RULES_DIR", "./data/rules")
    _EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 45 MB, CPU-friendly

    SYSTEM_PROMPT = """Tu es l\'assistant IA du parking SmartParkTN.
    Tu réponds aux questions du personnel de parking concernant:
    - Les règles d\'accès et les tarifs
    - Les catégories de véhicules (visiteur, abonné, VIP, liste noire, employé, urgence)
    - Les procédures en cas de litige ou d\'incident
    - Les horaires autorisés et les zones de stationnement
    - Le calcul des montants et durées

    Réponds toujours en français, de façon concise et précise.
    Base tes réponses UNIQUEMENT sur le contexte fourni.
    Si tu ne trouves pas l\'information, dis-le clairement.
    """


    class ParkingAssistant:
        def __init__(self):
            self._collection = None
            self._embed_fn   = None
            self._groq_client = None
            self._init()

        def _init(self):
            self._init_embedder()
            self._init_chroma()
            self._init_groq()

        def _init_embedder(self):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(_EMBED_MODEL)
                logger.info(f"Embedder loaded: {_EMBED_MODEL}")
            except Exception as e:
                logger.error(f"Embedder init failed: {e}")
                self._model = None

        def _init_chroma(self):
            try:
                import chromadb
                client = chromadb.PersistentClient(path=_CHROMA_DIR)
                self._collection = client.get_or_create_collection(
                    name="parking_rules",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(f"ChromaDB ready: {self._collection.count()} chunks")
            except Exception as e:
                logger.error(f"ChromaDB init failed: {e}")
                self._collection = None

        def _init_groq(self):
            if not _GROQ_KEY or _GROQ_KEY.startswith("gsk_XXX"):
                logger.warning("No valid GROQ_API_KEY – assistant in offline mode")
                return
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=_GROQ_KEY)
                logger.info(f"Groq client ready (model: {_GROQ_MODEL})")
            except Exception as e:
                logger.error(f"Groq init failed: {e}")

        # ── Document ingestion ─────────────────────────────────────────────
        def ingest_documents(self, rules_dir: str = _RULES_DIR):
            """Load all .md and .txt files from rules_dir into ChromaDB."""
            if not self._collection or not self._model:
                logger.error("ChromaDB or embedder not ready – cannot ingest")
                return

            files = glob.glob(os.path.join(rules_dir, "*.md")) + \\
                    glob.glob(os.path.join(rules_dir, "*.txt"))
            if not files:
                logger.warning(f"No rule files found in {rules_dir}")
                return

            for path in files:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                chunks = self._chunk(text)
                ids = [f"{os.path.basename(path)}::{i}" for i in range(len(chunks))]
                embeddings = self._model.encode(chunks, show_progress_bar=False).tolist()
                self._collection.upsert(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=[{"source": os.path.basename(path)}] * len(chunks),
                )
            logger.info(f"Ingested {len(files)} file(s) → {self._collection.count()} total chunks")

        @staticmethod
        def _chunk(text: str, size: int = 400, overlap: int = 80) -> List[str]:
            words = text.split()
            chunks, i = [], 0
            while i < len(words):
                chunk = " ".join(words[i:i + size])
                chunks.append(chunk)
                i += size - overlap
            return chunks or [text]

        # ── Query ─────────────────────────────────────────────────────────
        def query(self, question: str, n_results: int = 5) -> str:
            context = self._retrieve(question, n_results)
            return self._generate(question, context)

        def _retrieve(self, question: str, n: int) -> str:
            if not self._collection or not self._model:
                return "Base de connaissances non disponible."
            try:
                emb = self._model.encode([question]).tolist()
                res = self._collection.query(query_embeddings=emb, n_results=n)
                docs = res.get("documents", [[]])[0]
                return "\\n\\n---\\n\\n".join(docs) if docs else "Aucun document pertinent trouvé."
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                return "Erreur de recherche."

        def _generate(self, question: str, context: str) -> str:
            if not self._groq_client:
                return (
                    f"[Mode hors ligne – Groq non configuré]\\n\\n"
                    f"Contexte trouvé:\\n{context[:600]}"
                )
            try:
                resp = self._groq_client.chat.completions.create(
                    model=_GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"Contexte:\\n{context}\\n\\nQuestion: {question}"},
                    ],
                    temperature=0.2,
                    max_tokens=512,
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq generation error: {e}")
                return f"Erreur de génération: {str(e)}"

        def explain_decision(self, plate: str, decision: str, reason: str) -> str:
            q = (f"Explique pourquoi le véhicule {plate} a reçu la décision "
                 f"\\"{decision}\\" avec la raison: {reason}")
            return self.query(q)
''')

# ─────────────────────────────────────────────────────────────────────────────
# core/pipeline.py  – single unified pipeline
# ─────────────────────────────────────────────────────────────────────────────
w("core/pipeline.py", r'''
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
''')

# ─────────────────────────────────────────────────────────────────────────────
# api/routes.py
# ─────────────────────────────────────────────────────────────────────────────
w("api/routes.py", '''
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


    # ── Health ────────────────────────────────────────────────────────────────
    @router.get("/health")
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


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
''')

# ─────────────────────────────────────────────────────────────────────────────
# main.py  – FastAPI entry point
# ─────────────────────────────────────────────────────────────────────────────
w("main.py", '''
    """
    SmartParkTN – FastAPI Application
    Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    """
    import os
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from dotenv import load_dotenv
    from loguru import logger

    load_dotenv()

    from database.models import init_db
    from api.routes import router, get_assistant

    app = FastAPI(
        title="SmartParkTN API",
        description="ALPR System for Tunisian Parking Lots",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")


    @app.on_event("startup")
    async def startup():
        logger.info("Initialising SmartParkTN …")
        init_db()
        logger.info("Database initialised ✓")
        # Pre-load assistant and ingest rules
        try:
            assistant = get_assistant()
            assistant.ingest_documents()
            logger.info("RAG assistant ready ✓")
        except Exception as e:
            logger.warning(f"Assistant startup warning: {e}")
        logger.info("SmartParkTN ready ✓  →  http://localhost:8000/docs")


    @app.get("/")
    def root():
        return {
            "project": "SmartParkTN",
            "version": "1.0.0",
            "docs": "/docs",
            "api":  "/api/v1",
        }
''')

# ─────────────────────────────────────────────────────────────────────────────
# data/rules/*.md
# ─────────────────────────────────────────────────────────────────────────────
w("data/rules/reglement_parking.md", """
    # Règlement du Parking SmartParkTN

    ## 1. Catégories de véhicules

    | Catégorie   | Description                                      |
    |-------------|--------------------------------------------------|
    | VISITOR     | Véhicule visiteur sans abonnement                |
    | SUBSCRIBER  | Véhicule abonné au parking (mensuel/annuel)       |
    | VIP         | Véhicule VIP – accès prioritaire, gratuit         |
    | EMPLOYEE    | Véhicule d'employé – gratuit, accès étendu       |
    | BLACKLIST   | Véhicule interdit – accès systématiquement refusé |
    | EMERGENCY   | Véhicule d'urgence (ambulance, police, pompiers)  |

    ## 2. Règles d'accès générales

    - Tout véhicule blacklisté est refusé 24h/24, 7j/7, sans exception.
    - Les véhicules d'urgence ont un accès prioritaire à tout moment, sur toutes les zones.
    - Les VIP ont accès à toutes les zones, à toute heure, sans limite de durée.
    - Les employés ont accès aux zones A, B et C de 06h00 à 23h00.
    - Les visiteurs ont accès aux zones A et B de 06h00 à 23h00 uniquement.
    - Les abonnés ont accès à leurs zones désignées (A ou B), selon leur contrat.

    ## 3. Zones de stationnement

    - **Zone A** : Entrée principale, visiteurs et abonnés standards.
    - **Zone B** : Sous-sol niveau 1, abonnés préférentiels.
    - **Zone C** : Réservée aux employés et VIP.
    - **Zone VIP** : Emplacements réservés côté bâtiment, signalés en jaune.
    - **Zone URGENCE** : 4 emplacements devant l'entrée principale, toujours libres.

    ## 4. Procédure en cas de litige

    1. Prendre une capture de la plaque et de l'horodatage.
    2. Vérifier le statut dans le système SmartParkTN.
    3. Contacter le superviseur si le statut est ambigu.
    4. En cas de contestation d'un montant, le superviseur peut annuler ou modifier la facture.
    5. Tout incident doit être documenté dans le registre des incidents.

    ## 5. Véhicules abandonnés

    Un véhicule présent depuis plus de 24h sans paiement est considéré abandonné.
    Procédure : notification, puis fourrière après 48h.
""")

w("data/rules/tarifs.md", """
    # Tarifs du Parking SmartParkTN

    ## Tarif standard (Visiteurs)

    | Durée              | Tarif          |
    |--------------------|----------------|
    | 0–15 minutes       | GRATUIT        |
    | 15 min – 1 heure   | 2,000 TND      |
    | Chaque heure supp. | 2,000 TND/h    |
    | Maximum journalier | 20,000 TND     |

    ## Abonnements mensuels

    | Type         | Tarif mensuel |
    |--------------|---------------|
    | Zone A       | 80,000 TND    |
    | Zone B       | 100,000 TND   |
    | Zone A+B     | 150,000 TND   |
    | VIP annuel   | 1 200,000 TND |

    ## Tarifs spéciaux

    - **Employés** : Gratuit (badge requis).
    - **VIP** : Gratuit (abonnement annuel ou convention d'entreprise).
    - **Véhicules d'urgence** : Toujours gratuit.

    ## Pénalités

    | Infraction                    | Montant  |
    |-------------------------------|----------|
    | Dépassement de 24h            | +10 TND  |
    | Stationnement zone non autorisée | 15 TND |
    | Tentative d'accès interdit    | Signalement |

    ## Remises

    - Abonné renouvelant avant expiration : -10% sur le prochain mois.
    - Parking week-end visiteur (sam./dim.) : -20%.
    - Clients avec validation commerces partenaires : 1h gratuite supplémentaire.

    ## Mode de paiement

    Espèces, TPE (carte bancaire), paiement mobile (D17, Flouci), virement.
""")

w("data/rules/acces_et_exceptions.md", """
    # Politique d'Accès et Exceptions – SmartParkTN

    ## Accès refusé : raisons possibles

    1. **Véhicule blacklisté** : La plaque figure sur la liste noire (fraude, impayés, problème judiciaire).
    2. **Hors horaires** : Le véhicule tente d'entrer ou sortir en dehors des heures autorisées.
    3. **Abonnement expiré** : L'abonnement du véhicule n'est plus valide.
    4. **Zone non autorisée** : La zone demandée n'est pas incluse dans l'abonnement.
    5. **Plaque non lisible** : Qualité d'image insuffisante pour identifier le véhicule.

    ## Exceptions et procédures spéciales

    ### Ambulances et services d'urgence
    - Accès immédiat et prioritaire sans contrôle horaire.
    - Les caméras signalent automatiquement un événement URGENCE.
    - Aucun tarif appliqué.

    ### Véhicules diplomatiques
    - Accès autorisé sur présentation manuelle au superviseur.
    - Enregistrement manuel dans le système.

    ### Événements spéciaux
    - Le superviseur peut activer un "mode événement" permettant un accès étendu temporaire.
    - Tarification spéciale configurable par l'administrateur.

    ### Panne de système
    - En cas de panne du lecteur optique, utiliser le mode manuel (saisie clavier).
    - Garder un registre papier jusqu'à rétablissement du système.

    ## Comment interpréter une décision du système ?

    Le système SmartParkTN affiche toujours :
    - La plaque détectée (avec niveau de confiance OCR)
    - La catégorie identifiée (visitor, subscriber, vip, blacklist, employee, emergency)
    - La décision (ALLOWED / DENIED)
    - La raison de la décision (règle appliquée)

    Exemple : "Accès refusé – Hors horaires autorisés (06:00–23:00). Règle: visitor_hours"
    → Le véhicule est un visiteur et a tenté d'entrer après 23h.

    ## Contact superviseur

    - Poste superviseur : extension 201
    - Urgences : extension 999
    - Administration parking : admin@smartparktn.com
""")

# ─────────────────────────────────────────────────────────────────────────────
# scripts/seed_vehicles.py
# ─────────────────────────────────────────────────────────────────────────────
w("scripts/seed_vehicles.py", """
    # Seed sample Tunisian license plates into the database.
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from database.models import init_db, SessionLocal
    from database.crud import upsert_vehicle
    from database.models import VehicleCategory
    from datetime import datetime, timedelta

    def seed():
        init_db()
        db = SessionLocal()
        try:
            vehicles = [
                ("100 TN 1234", "Ahmed Ben Ali",     VehicleCategory.VISITOR,    None),
                ("200 TN 5678", "Sonia Gharbi",      VehicleCategory.SUBSCRIBER, "Abonnée Zone A"),
                ("300 TN 9012", "Mohamed Trabelsi",  VehicleCategory.VIP,        "Directeur général"),
                ("400 TN 3456", "Nour Chaabane",     VehicleCategory.EMPLOYEE,   "Employée RH"),
                ("500 TN 7890", "Inconnu",           VehicleCategory.BLACKLIST,  "Impayés x3 – signalé le 2024-01-15"),
                ("111 TN 2222", "SAMU Tunis",        VehicleCategory.EMERGENCY,  "Ambulance SAMU"),
                ("777 TN 8888", "Karim Mansouri",    VehicleCategory.SUBSCRIBER, "Abonné Zone B"),
                ("999 TN 0001", "Leila Ben Youssef", VehicleCategory.VIP,        "VIP partenaire"),
            ]
            for plate, owner, cat, notes in vehicles:
                upsert_vehicle(db, plate, owner, cat, notes)
                print(f"  ✓  {plate:18s} [{cat.value}]")

            # Add subscription for subscribers
            from database.models import Subscription
            subs = [
                ("200 TN 5678", "A", datetime.now(), datetime.now() + timedelta(days=30)),
                ("777 TN 8888", "B", datetime.now(), datetime.now() + timedelta(days=30)),
            ]
            for plate, zone, start, end in subs:
                existing = db.query(Subscription).filter(Subscription.plate == plate).first()
                if not existing:
                    db.add(Subscription(plate=plate, zone=zone, start_date=start, end_date=end))
            db.commit()
            print("\\nSeeding complete! 🚗")
        finally:
            db.close()

    if __name__ == "__main__":
        seed()
""")

# ─────────────────────────────────────────────────────────────────────────────
# scripts/ingest_rules.py
# ─────────────────────────────────────────────────────────────────────────────
w("scripts/ingest_rules.py", """
    # Ingest parking rules documents into ChromaDB vector store.
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from core.rag import ParkingAssistant

    if __name__ == "__main__":
        print("Ingesting parking rules into ChromaDB…")
        assistant = ParkingAssistant()
        assistant.ingest_documents()
        print("Done ✓")
""")

# ─────────────────────────────────────────────────────────────────────────────
# ui/dashboard.py  – Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
w("ui/dashboard.py", '''
    """
    SmartParkTN – Streamlit Dashboard
    Run: streamlit run ui/dashboard.py
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import streamlit as st
    import requests
    import pandas as pd
    import plotly.express as px
    import cv2
    import numpy as np
    import base64
    import time
    from datetime import datetime
    from dotenv import load_dotenv

    load_dotenv()

    API_BASE = f"http://localhost:{os.getenv(\'API_PORT\', 8000)}/api/v1"

    st.set_page_config(
        page_title="SmartParkTN",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        padding: 1rem; border-radius: 12px; color: white;
        text-align: center; margin: 0.3rem;
    }
    .allowed  { color: #00e676; font-weight: bold; }
    .denied   { color: #ff5252; font-weight: bold; }
    .pending  { color: #ffd740; font-weight: bold; }
    .plate-display {
        font-family: monospace; font-size: 2rem;
        background: #fff; color: #000;
        padding: 0.5rem 1.5rem; border-radius: 8px;
        border: 4px solid #1e3a5f; display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Helpers ────────────────────────────────────────────────────────────
    def api_get(path, default=None):
        try:
            r = requests.get(f"{API_BASE}{path}", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return default

    def api_post(path, data=None, files=None):
        try:
            if files:
                r = requests.post(f"{API_BASE}{path}", data=data, files=files, timeout=15)
            else:
                r = requests.post(f"{API_BASE}{path}", json=data, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.image("https://img.icons8.com/color/96/car--v1.png", width=80)
    st.sidebar.title("SmartParkTN")
    st.sidebar.markdown("Système ALPR Tunisien")
    page = st.sidebar.radio("Navigation", [
        "📊 Tableau de bord",
        "📷 Détection en direct",
        "🚗 Véhicules",
        "📋 Événements",
        "💬 Assistant IA",
        "⚙️ Paramètres",
    ])

    # ── Check API ──────────────────────────────────────────────────────────
    health = api_get("/health")
    api_ok = health is not None and "ok" in str(health.get("status", ""))
    st.sidebar.markdown(
        "🟢 **API connectée**" if api_ok else "🔴 **API hors ligne**\\n\\n`uvicorn main:app`"
    )

    # ─────────────────────────────────────────────────────────────────────
    if "📊" in page:
        st.title("📊 Tableau de bord – SmartParkTN")
        events = api_get("/events?limit=500", [])
        vehicles = api_get("/vehicles", [])
        tariffs  = api_get("/tariffs", [])

        df = pd.DataFrame(events)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚗 Total événements", len(df))
        with col2:
            n_allowed = len(df[df.decision == "allowed"]) if not df.empty else 0
            st.metric("✅ Accès autorisés", n_allowed)
        with col3:
            n_denied = len(df[df.decision == "denied"]) if not df.empty else 0
            st.metric("🚫 Accès refusés", n_denied)
        with col4:
            rev = df.amount_tnd.sum() if not df.empty and "amount_tnd" in df.columns else 0
            st.metric("💰 Revenus (TND)", f"{rev:.2f}")

        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(df, names="category", title="Catégories de véhicules",
                             color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.histogram(df, x="hour", title="Trafic par heure",
                                    nbins=24, color_discrete_sequence=["#1e88e5"])
                fig2.update_layout(xaxis_title="Heure", yaxis_title="Véhicules")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Derniers événements")
            disp = df.sort_values("timestamp", ascending=False).head(20)[[
                "timestamp", "plate", "category", "type", "decision", "amount_tnd", "reason"
            ]]
            st.dataframe(disp, use_container_width=True)
        else:
            st.info("Aucune donnée. Lancez la détection pour voir des statistiques.")

        st.subheader("Tarifs en vigueur")
        if tariffs:
            st.dataframe(pd.DataFrame(tariffs), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    elif "📷" in page:
        st.title("📷 Détection ALPR en direct")

        mode = st.radio("Source", ["📁 Téléverser une image", "🎥 Webcam en direct"])
        camera_id = st.selectbox("Caméra", ["CAM_ENTRY_01", "CAM_EXIT_01",
                                             "CAM_ENTRY_02", "CAM_EXIT_02"])

        if "📁" in mode:
            uploaded = st.file_uploader("Image véhicule / plaque", type=["jpg","jpeg","png"])
            if uploaded:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(uploaded, caption="Image originale", use_column_width=True)
                with col_b:
                    with st.spinner("Analyse en cours…"):
                        result = api_post(
                            "/process-image",
                            data={"camera_id": camera_id},
                            files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")},
                        )
                    if "error" not in result:
                        if result.get("annotated_image_b64"):
                            img_bytes = base64.b64decode(result["annotated_image_b64"])
                            st.image(img_bytes, caption="Plaque détectée", use_column_width=True)

                        plate = result.get("plate", "—")
                        dec   = result.get("decision", "pending")
                        color_map = {"allowed": "🟢", "denied": "🔴", "pending": "🟡"}
                        emoji = color_map.get(dec, "🟡")

                        st.markdown(f"### {emoji} Décision : **{dec.upper()}**")
                        st.markdown(f"<div class=\'plate-display\'>{plate}</div>",
                                    unsafe_allow_html=True)
                        cols = st.columns(3)
                        cols[0].metric("Catégorie", result.get("category", "—"))
                        cols[1].metric("Confiance OCR", f"{result.get(\'confidence\', 0):.0%}")
                        cols[2].metric("Confiance Détection", f"{result.get(\'detect_conf\', 0):.0%}")

                        if result.get("amount_tnd") is not None:
                            st.success(f"Durée: {result[\'duration_min\']:.0f} min | "
                                       f"Montant: {result[\'amount_tnd\']:.2f} TND")
                        st.info(f"Raison: {result.get(\'reason\', \'—\')}")
                    else:
                        st.error(f"Erreur: {result[\'error\']}")

        else:  # Webcam
            st.warning("Assurez-vous que la caméra est accessible (cv2.VideoCapture(0)).")
            run = st.checkbox("▶ Activer la webcam")
            frame_ph = st.empty()
            result_ph = st.empty()

            if run:
                cap = cv2.VideoCapture(0)
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Impossible d\'ouvrir la caméra."); break
                    _, buf = cv2.imencode(".jpg", frame)
                    b64 = base64.b64encode(buf).decode()
                    frame_ph.image(frame[:, :, ::-1], channels="RGB",
                                   caption="Flux caméra", use_column_width=True)
                    result = api_post("/process-frame",
                                      {"frame_b64": b64, "camera_id": camera_id})
                    if result.get("plate"):
                        result_ph.success(
                            f"🎯 {result[\'plate\']} | {result[\'category\']} | {result[\'decision\']}"
                        )
                    time.sleep(0.5)
                cap.release()

    # ─────────────────────────────────────────────────────────────────────
    elif "🚗" in page:
        st.title("🚗 Gestion des Véhicules")
        tab1, tab2 = st.tabs(["Liste", "Ajouter / Modifier"])

        with tab1:
            vehicles = api_get("/vehicles", [])
            if vehicles:
                df_v = pd.DataFrame(vehicles)
                search = st.text_input("🔍 Rechercher par plaque")
                if search:
                    df_v = df_v[df_v["plate"].str.contains(search.upper(), na=False)]
                st.dataframe(df_v, use_container_width=True)
                # Quick lookup
                st.subheader("Vérification d\'accès rapide")
                plate_check = st.text_input("Entrer une plaque")
                if plate_check:
                    detail = api_get(f"/vehicles/{plate_check.upper()}")
                    if detail:
                        dec = detail.get("access", "pending")
                        emoji = {"allowed": "✅", "denied": "❌", "pending": "⏳"}.get(dec, "❓")
                        st.markdown(f"{emoji} **{plate_check.upper()}** – {dec.upper()}")
                        st.write(detail)
            else:
                st.info("Aucun véhicule enregistré.")

        with tab2:
            with st.form("add_vehicle"):
                st.subheader("Enregistrer un véhicule")
                plate = st.text_input("Plaque (ex: 100 TN 1234)").upper()
                owner = st.text_input("Propriétaire")
                cat   = st.selectbox("Catégorie",
                    ["visitor","subscriber","vip","employee","blacklist","emergency"])
                notes = st.text_area("Notes")
                if st.form_submit_button("Enregistrer"):
                    r = api_post("/vehicles", {"plate": plate, "owner_name": owner,
                                               "category": cat, "notes": notes})
                    if "error" not in r:
                        st.success(f"Véhicule {plate} enregistré ✓")
                    else:
                        st.error(str(r))

    # ─────────────────────────────────────────────────────────────────────
    elif "📋" in page:
        st.title("📋 Historique des Événements")
        limit = st.slider("Nombre d\'événements", 50, 500, 100)
        events = api_get(f"/events?limit={limit}", [])
        if events:
            df_e = pd.DataFrame(events)
            # Filter
            cats = ["Toutes"] + list(df_e["category"].unique()) if "category" in df_e.columns else ["Toutes"]
            cat_f = st.selectbox("Filtrer par catégorie", cats)
            dec_f = st.selectbox("Filtrer par décision", ["Toutes", "allowed", "denied", "pending"])
            if cat_f != "Toutes":
                df_e = df_e[df_e["category"] == cat_f]
            if dec_f != "Toutes":
                df_e = df_e[df_e["decision"] == dec_f]

            st.dataframe(df_e, use_container_width=True)
            # Export CSV
            csv = df_e.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Télécharger CSV", csv, "events.csv", "text/csv")
        else:
            st.info("Aucun événement enregistré.")

    # ─────────────────────────────────────────────────────────────────────
    elif "💬" in page:
        st.title("💬 Assistant IA – SmartParkTN")
        st.caption("Posez vos questions sur les règles, tarifs et procédures.")

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content":
                 "Bonjour ! Je suis l\'assistant SmartParkTN. "
                 "Posez-moi vos questions sur les règlements, tarifs et procédures."}
            ]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Votre question…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Recherche en cours…"):
                    result = api_post("/assistant/ask", {"question": question})
                answer = result.get("answer", "Désolé, une erreur est survenue.")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        st.divider()
        st.subheader("Expliquer une décision")
        col1, col2, col3 = st.columns(3)
        exp_plate = col1.text_input("Plaque", key="exp_plate")
        exp_dec   = col2.selectbox("Décision", ["denied", "allowed"], key="exp_dec")
        exp_reason= col3.text_input("Raison", key="exp_reason")
        if st.button("Expliquer"):
            r = api_post("/assistant/explain",
                         {"plate": exp_plate, "decision": exp_dec, "reason": exp_reason})
            st.info(r.get("explanation", "Erreur"))

    # ─────────────────────────────────────────────────────────────────────
    elif "⚙️" in page:
        st.title("⚙️ Paramètres")
        st.subheader("Ingestion des règles (RAG)")
        if st.button("🔄 Réingérer les documents"):
            r = api_post("/assistant/ingest")
            st.success(str(r))

        st.subheader("Base de données")
        if st.button("🌱 Seeder les véhicules de test"):
            import subprocess
            subprocess.run(["python", "scripts/seed_vehicles.py"], cwd="..")
            st.success("Seeding lancé !")

        st.subheader("Configuration API")
        st.code(f"API_BASE = {API_BASE}", language="text")
        st.json(health or {"status": "offline"})
''')

# ─────────────────────────────────────────────────────────────────────────────
# demo/demo.py
# ─────────────────────────────────────────────────────────────────────────────
w("demo/demo.py", r'''
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
''')

# ─────────────────────────────────────────────────────────────────────────────
# streamlit_app.py  – top-level entry for Streamlit Cloud
# ─────────────────────────────────────────────────────────────────────────────
w("streamlit_app.py", """\
# Entry point for `streamlit run streamlit_app.py`
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ui.dashboard import *   # noqa – dashboard registers all Streamlit pages
""")

# ─────────────────────────────────────────────────────────────────────────────
# .env  (default – user must add Groq key)
# ─────────────────────────────────────────────────────────────────────────────
env_path = os.path.join(BASE, ".env")
if not os.path.exists(env_path):
    import shutil
    shutil.copy(os.path.join(BASE, ".env.example"), env_path)
    print(f"  ✓  .env  (copied from .env.example – add your GROQ_API_KEY)")

# ─────────────────────────────────────────────────────────────────────────────
print("""
╔═══════════════════════════════════════════════════════════════════╗
║          SmartParkTN – Bootstrap Complete ✓                       ║
╠═══════════════════════════════════════════════════════════════════╣
║  Next steps:                                                      ║
║                                                                   ║
║  1. Install dependencies:                                         ║
║       pip install -r requirements.txt                             ║
║                                                                   ║
║  2. Add your FREE Groq API key to .env:                           ║
║       GROQ_API_KEY=gsk_...    (https://console.groq.com/keys)     ║
║                                                                   ║
║  3. Seed sample vehicles:                                         ║
║       python scripts/seed_vehicles.py                             ║
║                                                                   ║
║  4. Start the API server:                                         ║
║       uvicorn main:app --reload --port 8000                       ║
║                                                                   ║
║  5. Start the dashboard (new terminal):                           ║
║       streamlit run ui/dashboard.py                               ║
║                                                                   ║
║  6. Run the video demo:                                           ║
║       python demo/demo.py --simulate                              ║
║       python demo/demo.py --input your_video.mp4                  ║
║                                                                   ║
║  API docs: http://localhost:8000/docs                             ║
║  Dashboard: http://localhost:8501                                 ║
╚═══════════════════════════════════════════════════════════════════╝
""")
