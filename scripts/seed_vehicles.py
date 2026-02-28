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
        print("\nSeeding complete! 🚗")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
