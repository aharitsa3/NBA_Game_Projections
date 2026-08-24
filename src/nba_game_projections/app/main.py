"""FastAPI application entrypoint.

Run locally with: uvicorn nba_game_projections.app.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from nba_game_projections.app.db import init_db
from nba_game_projections.app.routers import admin, predictions

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="NBA Game Projections", lifespan=lifespan)
app.include_router(admin.router)
app.include_router(predictions.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Mounted last so it never shadows the routes above - StaticFiles(html=True)
# serves static/index.html at "/" as the app's UI (PRD §13/§20's deferred
# frontend, built as a lightweight static page rather than Angular).
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
