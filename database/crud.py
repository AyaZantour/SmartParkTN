"""CRUD helpers for SmartParkTN."""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List
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


def count_currently_parked(db: Session) -> int:
    """Count vehicles currently inside (entry without matching exit)."""
    from sqlalchemy import func
    # Entries that have no subsequent exit logged (duration_minutes is NULL)
    result = db.execute(
        select(func.count()).select_from(ParkingEvent).where(
            ParkingEvent.event_type == EventType.ENTRY,
            ParkingEvent.duration_minutes == None,
        )
    ).scalar()
    return result or 0


# ── Subscriptions ─────────────────────────────────────────────────────────
def list_subscriptions(db: Session) -> List[Subscription]:
    return db.execute(
        select(Subscription).order_by(desc(Subscription.start_date))
    ).scalars().all()


def get_subscription(db: Session, sub_id: int) -> Optional[Subscription]:
    return db.execute(
        select(Subscription).where(Subscription.id == sub_id)
    ).scalar_one_or_none()


def add_subscription(
    db: Session,
    plate: str,
    start_date: datetime,
    end_date: datetime,
    zone: str = "A",
) -> Subscription:
    sub = Subscription(
        plate=plate.upper(),
        start_date=start_date,
        end_date=end_date,
        zone=zone,
        is_active=True,
    )
    db.add(sub); db.commit(); db.refresh(sub)
    # Ensure vehicle is marked as subscriber
    upsert_vehicle(db, plate.upper(), category=VehicleCategory.SUBSCRIBER)
    return sub


def deactivate_subscription(db: Session, sub_id: int) -> bool:
    sub = get_subscription(db, sub_id)
    if not sub:
        return False
    sub.is_active = False
    db.commit()
    return True


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
