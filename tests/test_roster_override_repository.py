import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections.app.db import Base
from nba_game_projections.app.repositories.roster_overrides import (
    build_roster_override_repository,
)


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine)() as db_session:
        yield db_session


@pytest.fixture
def repo(session):
    return build_roster_override_repository(session)


def test_add_then_list_reflects_current_state(repo):
    repo.add(player_id=1, player_name="Player One", team_id=100, status="out", note="ankle")

    overrides = repo.list()

    assert len(overrides) == 1
    assert overrides[0].player_id == 1
    assert overrides[0].status == "out"
    assert overrides[0].note == "ankle"


def test_update_existing_player_status(repo):
    added = repo.add(player_id=2, player_name="Player Two", team_id=100, status="out")

    updated = repo.update(added.id, status="in", note="cleared")

    assert updated.status == "in"
    assert updated.note == "cleared"
    assert repo.list()[0].status == "in"


def test_update_missing_id_returns_none(repo):
    assert repo.update(999, status="in") is None


def test_remove_deletes_row(repo):
    added = repo.add(player_id=3, player_name="Player Three", team_id=100, status="reassigned",
                      new_team_id=200)

    removed = repo.remove(added.id)

    assert removed is True
    assert repo.list() == []


def test_remove_missing_id_returns_false(repo):
    assert repo.remove(999) is False


def test_add_rejects_invalid_status(repo):
    with pytest.raises(ValueError):
        repo.add(player_id=4, player_name="Player Four", team_id=100, status="benched")


def test_update_rejects_invalid_status(repo):
    added = repo.add(player_id=5, player_name="Player Five", team_id=100, status="out")
    with pytest.raises(ValueError):
        repo.update(added.id, status="benched")


def test_get_by_player_id_finds_existing_override(repo):
    added = repo.add(player_id=6, player_name="Player Six", team_id=100, status="out")

    found = repo.get_by_player_id(6)

    assert found.id == added.id


def test_get_by_player_id_returns_none_when_missing(repo):
    assert repo.get_by_player_id(999) is None
