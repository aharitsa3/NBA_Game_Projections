"""SQLAlchemy engine, session, and declarative base for app-managed state.

Roster overrides and model version metadata (Wave 5+) are defined as models
against `Base` in their own repository modules; this module only owns the
engine/session machinery so every table shares one connection setup.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from nba_game_projections import config

engine = create_engine(
    config.APP_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create any tables that don't exist yet, so a fresh clone needs no manual setup."""
    config.MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped DB session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
