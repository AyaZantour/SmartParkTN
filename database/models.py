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
