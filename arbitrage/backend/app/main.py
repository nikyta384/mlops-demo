"""
Crypto Arbitrage MVP – FastAPI application entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import market as market_router
from app.api import trades as trades_router
from app.core.config import settings
from app.db.session import Base, engine
from app.services.execution import execution_service
from app.services.market_data import market_data_service
from app.ws import market as ws_market

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    log.info("Starting up – creating database tables…")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database ready.")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────
    log.info("Shutting down services…")
    await market_data_service.close()
    await execution_service.close()
    log.info("Shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routers
app.include_router(market_router.router, prefix="/api/v1")
app.include_router(trades_router.router, prefix="/api/v1")

# WebSocket
app.include_router(ws_market.router)

# Serve static frontend files
import os

_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="static")
