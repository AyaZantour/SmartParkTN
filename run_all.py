"""
SmartParkTN – run_all.py
Convenience launcher: starts FastAPI + opens dashboard in browser.
Run: python run_all.py
"""
import subprocess, sys, time, webbrowser, os

def main():
    print("\n SmartParkTN – Starting services …\n")

    # 1. Bootstrap if needed
    if not os.path.exists("core"):
        print(" Running bootstrap …")
        subprocess.run([sys.executable, "bootstrap.py"], check=True)

    # 2. Seed if DB missing
    if not os.path.exists("smartpark.db"):
        print(" Seeding database …")
        subprocess.run([sys.executable, "scripts/seed_vehicles.py"], check=True)

    # 3. Start FastAPI
    api = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0", "--port", "8000", "--reload"
    ])
    print(" ✓ API starting on http://localhost:8000")
    time.sleep(3)

    # 4. Start Streamlit
    ui = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "ui/dashboard.py",
        "--server.port", "8501", "--server.headless", "true"
    ])
    print(" ✓ Dashboard starting on http://localhost:8501")
    time.sleep(3)

    webbrowser.open("http://localhost:8501")
    print("\n Press Ctrl+C to stop all services.\n")

    try:
        api.wait()
    except KeyboardInterrupt:
        print("\n Shutting down …")
        api.terminate()
        ui.terminate()

if __name__ == "__main__":
    main()
