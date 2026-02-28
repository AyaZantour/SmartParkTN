# Ingest parking rules documents into ChromaDB vector store.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.rag import ParkingAssistant

if __name__ == "__main__":
    print("Ingesting parking rules into ChromaDB…")
    assistant = ParkingAssistant()
    assistant.ingest_documents()
    print("Done ✓")
