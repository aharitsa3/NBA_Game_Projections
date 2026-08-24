import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections.app.db import Base
from nba_game_projections.app.repositories.predictions import build_prediction_repository


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine)() as db_session:
        yield db_session


@pytest.fixture
def repo(session):
    return build_prediction_repository(session)


def _upsert(repo, **overrides):
    fields = {
        "game_id": 1,
        "date": "2027-01-15",
        "home_team_id": 1,
        "home_team_name": "Team One",
        "away_team_id": 2,
        "away_team_name": "Team Two",
        "predicted_margin": 3.5,
        "predicted_winner": "home",
        "home_win_probability": 0.6,
        "model_version_id": 10,
    }
    fields.update(overrides)
    return repo.upsert_prediction(**fields)


def test_upsert_inserts_new_row(repo):
    record = _upsert(repo)

    assert record.game_id == 1
    assert record.predicted_margin == 3.5
    assert record.reconciled_at is None
    assert repo.list() == [record]


def test_upsert_refreshes_unreconciled_row(repo):
    first = _upsert(repo, predicted_margin=3.5, model_version_id=10)

    second = _upsert(repo, predicted_margin=7.0, model_version_id=11)

    assert second.id == first.id
    assert second.predicted_margin == 7.0
    assert second.model_version_id == 11
    assert len(repo.list()) == 1


def test_upsert_never_overwrites_reconciled_row(repo):
    original = _upsert(repo, predicted_margin=3.5)
    repo.reconcile(1, actual_margin=5.0, actual_winner="home")

    unchanged = _upsert(repo, predicted_margin=99.0)

    assert unchanged.predicted_margin == 3.5
    assert unchanged.id == original.id


def test_reconcile_locks_row_and_sets_actual_result(repo):
    _upsert(repo)

    reconciled = repo.reconcile(1, actual_margin=5.0, actual_winner="home")

    assert reconciled.actual_margin == 5.0
    assert reconciled.actual_winner == "home"
    assert reconciled.reconciled_at is not None


def test_reconcile_unknown_game_id_returns_none(repo):
    assert repo.reconcile(999, actual_margin=1.0, actual_winner="home") is None


def test_reconcile_already_reconciled_row_returns_none(repo):
    _upsert(repo)
    repo.reconcile(1, actual_margin=5.0, actual_winner="home")

    assert repo.reconcile(1, actual_margin=99.0, actual_winner="away") is None


def test_list_pending_excludes_reconciled_rows(repo):
    _upsert(repo, game_id=1)
    _upsert(repo, game_id=2)
    repo.reconcile(1, actual_margin=5.0, actual_winner="home")

    pending = repo.list_pending()

    assert [record.game_id for record in pending] == [2]


def test_get_by_game_id_returns_none_when_missing(repo):
    assert repo.get_by_game_id(999) is None
