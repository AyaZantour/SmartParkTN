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
