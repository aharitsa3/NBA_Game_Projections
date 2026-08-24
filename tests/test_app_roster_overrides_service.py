import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections.app.db import Base
from nba_game_projections.app.repositories.roster_overrides import (
    build_roster_override_repository,
)
from nba_game_projections.app.services.roster_overrides import load_live_roster_overrides
from nba_game_projections.features import roster_overrides as feature_overrides


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine)() as db_session:
        yield db_session


def _seed_in_out_reassigned_mix(session):
    repository = build_roster_override_repository(session)
    repository.add(
        player_id=3945274,
        player_name="Example Player A",
        team_id=1610612738,
        status="out",
        note="same-day injury, not yet reflected in trailing rolling stats",
    )
    repository.add(
        player_id=4066648,
        player_name="Example Player B",
        team_id=1610612747,
        status="in",
        note="returning from injury",
    )
    repository.add(
        player_id=3136776,
        player_name="Example Player C",
        team_id=1610612738,
        status="reassigned",
        new_team_id=1610612744,
        note="trade completed, not yet reflected in player_box team_id",
    )


def test_load_live_roster_overrides_matches_sqlite_state(session):
    _seed_in_out_reassigned_mix(session)

    overrides = load_live_roster_overrides(session)

    assert len(overrides) == 3
    out_entry, in_entry, reassigned_entry = overrides

    assert isinstance(out_entry, feature_overrides.RosterOverride)
    assert out_entry.player_id == 3945274
    assert out_entry.status == "out"
    assert out_entry.new_team_id is None

    assert in_entry.player_id == 4066648
    assert in_entry.status == "in"

    assert reassigned_entry.player_id == 3136776
    assert reassigned_entry.status == "reassigned"
    assert reassigned_entry.new_team_id == 1610612744


def test_produced_overrides_behave_identically_under_apply_roster_overrides(session):
    """Same in/out/reassigned mix and assertions as T2.4's own
    `test_roster_overrides.py` - only the override source (SQLite vs. JSON
    file) differs, per PRD §15.
    """
    _seed_in_out_reassigned_mix(session)
    overrides = load_live_roster_overrides(session)

    out_result = feature_overrides.apply_roster_overrides(
        1610612738, {3945274, 111, 222}, overrides
    )
    assert 3945274 not in out_result
    assert out_result == {111, 222}

    in_result = feature_overrides.apply_roster_overrides(1610612747, {555, 666}, overrides)
    assert 4066648 in in_result
    assert in_result == {555, 666, 4066648}

    old_team_id, new_team_id = 1610612738, 1610612744
    old_roster = feature_overrides.apply_roster_overrides(old_team_id, {3136776, 999}, overrides)
    new_roster = feature_overrides.apply_roster_overrides(new_team_id, {888}, overrides)
    assert 3136776 not in old_roster
    assert 3136776 in new_roster


def test_load_live_roster_overrides_empty_when_no_rows(session):
    assert load_live_roster_overrides(session) == []
