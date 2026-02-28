"""Run this once to bootstrap the project folder structure."""
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))

dirs = [
    "core", "database", "api",
    os.path.join("data", "rules"),
    os.path.join("data", "vehicles"),
    os.path.join("data", "chroma_db"),
    "ui", "demo", "models", "scripts", "assets",
]

for d in dirs:
    path = os.path.join(BASE, d)
    os.makedirs(path, exist_ok=True)
    # create __init__.py for Python packages
    if d not in ("models", "assets", os.path.join("data","rules"),
                 os.path.join("data","vehicles"), os.path.join("data","chroma_db")):
        init_file = os.path.join(path, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, "w").close()
    print(f"  ✓  {path}")

print("\nDone. Run: pip install -r requirements.txt")
