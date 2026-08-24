import json
from datetime import date

import pytest

from nba_game_projections.features import roster_overrides


def _sample_payload(**overrides):
    payload = {
        "_description": "Example roster override file for tests.",
        "prediction_date": "2026-01-15",
        "overrides": [
            {
                "player_id": 3945274,
                "player_name": "Example Player A",
                "team_id": 1610612738,
                "status": "out",
                "note": "same-day injury, not yet reflected in trailing rolling stats",
            },
            {
                "player_id": 4066648,
                "player_name": "Example Player B",
                "team_id": 1610612747,
                "status": "in",
                "note": "returning from injury",
            },
            {
                "player_id": 3136776,
                "player_name": "Example Player C",
                "team_id": 1610612738,
                "status": "reassigned",
                "new_team_id": 1610612744,
                "note": "trade completed, not yet reflected in player_box team_id",
            },
        ],
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, payload, name="roster_overrides.json"):
    file_path = tmp_path / name
    file_path.write_text(json.dumps(payload))
    return file_path


def test_load_roster_overrides_parses_all_status_types(tmp_path):
    path = _write(tmp_path, _sample_payload())

    prediction_date, overrides = roster_overrides.load_roster_overrides(path)

    assert prediction_date == date(2026, 1, 15)
    assert len(overrides) == 3

    out_entry, in_entry, reassigned_entry = overrides

    assert out_entry.player_id == 3945274
    assert out_entry.player_name == "Example Player A"
    assert out_entry.team_id == 1610612738
    assert out_entry.status == "out"
    assert out_entry.new_team_id is None

    assert in_entry.player_id == 4066648
    assert in_entry.team_id == 1610612747
    assert in_entry.status == "in"
    assert in_entry.new_team_id is None

    assert reassigned_entry.player_id == 3136776
    assert reassigned_entry.team_id == 1610612738
    assert reassigned_entry.status == "reassigned"
    assert reassigned_entry.new_team_id == 1610612744
    assert reassigned_entry.note == "trade completed, not yet reflected in player_box team_id"


def test_out_status_removes_player_even_if_present(tmp_path):
    path = _write(tmp_path, _sample_payload())
    _, overrides = roster_overrides.load_roster_overrides(path)

    team_id = 1610612738
    automatic_roster = {3945274, 111, 222}

    result = roster_overrides.apply_roster_overrides(team_id, automatic_roster, overrides)

    assert 3945274 not in result
    assert result == {111, 222}


def test_in_status_adds_player_even_if_absent(tmp_path):
    path = _write(tmp_path, _sample_payload())
    _, overrides = roster_overrides.load_roster_overrides(path)

    team_id = 1610612747
    automatic_roster = {555, 666}

    result = roster_overrides.apply_roster_overrides(team_id, automatic_roster, overrides)

    assert 4066648 in result
    assert result == {555, 666, 4066648}


def test_reassigned_status_queried_from_both_sides(tmp_path):
    path = _write(tmp_path, _sample_payload())
    _, overrides = roster_overrides.load_roster_overrides(path)

    old_team_id = 1610612738
    new_team_id = 1610612744

    old_additions, old_removals = roster_overrides.roster_override_deltas(
        old_team_id, overrides
    )
    new_additions, new_removals = roster_overrides.roster_override_deltas(
        new_team_id, overrides
    )

    assert 3136776 in old_removals
    assert 3136776 not in old_additions
    assert 3136776 in new_additions
    assert 3136776 not in new_removals

    old_roster = roster_overrides.apply_roster_overrides(
        old_team_id, {3136776, 999}, overrides
    )
    new_roster = roster_overrides.apply_roster_overrides(
        new_team_id, {888}, overrides
    )

    assert 3136776 not in old_roster
    assert 3136776 in new_roster


def test_missing_override_file_returns_empty_list(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"

    _, overrides = roster_overrides.load_roster_overrides(missing_path)

    assert overrides == []


def test_unknown_status_raises_value_error(tmp_path):
    payload = _sample_payload()
    payload["overrides"][0]["status"] = "benched"
    path = _write(tmp_path, payload)

    with pytest.raises(ValueError, match="benched"):
        roster_overrides.load_roster_overrides(path)
