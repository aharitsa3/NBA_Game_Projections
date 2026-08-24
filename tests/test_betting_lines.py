import json

import pandas as pd
import pytest

from nba_game_projections.data import betting_lines


def _raw_record(**overrides):
    record = {
        "season": "2017",
        "game_date": "2017-10-17 00:00:00",
        "game_time": "20",
        "home_team_id": "11",
        "home_team_stats_id": "CLE",
        "home_team_abbrev": "CLE",
        "visit_team_id": "2",
        "visit_team_stats_id": "BOS",
        "visit_team_abbrev": "BOS",
        "home_team_score": 102,
        "visit_team_score": 99,
        "game_over_under": "216.0",
        "line": -4.5,
        "tipoff": "Oct 17 8:00 PM",
        "month": "October",
        "start": "Night",
        "favorite": "CLE",
        "score": "99-102",
        "total": 201,
        "spread": 4.5,
        "over_hit": 0,
        "under_hit": 1,
        "favorite_covered": 0,
        "underdog_covered": 1,
        "name": (
            '<span style="color:#1da561 !important">BOS</span> @ '
            '<span style="font-weight:700;">CLE</span>'
        ),
    }
    record.update(overrides)
    return record


def _write_archive(tmp_path, records):
    dest_dir = tmp_path / "betting_lines"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "games-archive.json").write_text(json.dumps(records))


def test_load_betting_lines_schema_and_dtypes(monkeypatch, tmp_path):
    monkeypatch.setattr(betting_lines.config, "DATA_RAW_DIR", tmp_path)
    _write_archive(tmp_path, [_raw_record(), _raw_record(home_team_id="7", visit_team_id="9")])

    result = betting_lines.load_betting_lines()

    assert len(result) == 2
    assert set(result.columns) == set(betting_lines._COLUMNS)
    assert "name" not in result.columns

    assert result["season"].dtype.kind == "i"
    assert result["home_team_id"].dtype.kind == "i"
    assert result["visit_team_id"].dtype.kind == "i"
    assert result["home_team_score"].dtype.kind == "i"
    assert result["visit_team_score"].dtype.kind == "i"
    assert result["game_over_under"].dtype.kind == "f"
    assert result["line"].dtype.kind == "f"
    assert result["spread"].dtype.kind == "f"
    assert result["favorite_covered"].dtype == bool
    assert result["underdog_covered"].dtype == bool
    assert result["over_hit"].dtype == bool
    assert result["under_hit"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(result["game_date"])

    row = result.iloc[0]
    assert row["season"] == 2017
    assert row["home_team_id"] == 11
    assert row["visit_team_id"] == 2
    assert row["game_over_under"] == 216.0
    assert bool(row["favorite_covered"]) is False
    assert bool(row["underdog_covered"]) is True


def test_load_betting_lines_filters_by_season(monkeypatch, tmp_path):
    monkeypatch.setattr(betting_lines.config, "DATA_RAW_DIR", tmp_path)
    _write_archive(
        tmp_path,
        [_raw_record(season="2017"), _raw_record(season="2018")],
    )

    result = betting_lines.load_betting_lines(seasons=[2018])

    assert len(result) == 1
    assert result.iloc[0]["season"] == 2018


def test_load_betting_lines_coerces_empty_string_numeric_fields_to_nan(monkeypatch, tmp_path):
    monkeypatch.setattr(betting_lines.config, "DATA_RAW_DIR", tmp_path)
    _write_archive(tmp_path, [_raw_record(game_over_under="", favorite="")])

    result = betting_lines.load_betting_lines()

    assert pd.isna(result.iloc[0]["game_over_under"])
    assert result.iloc[0]["favorite"] == ""


def test_missing_archive_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(betting_lines.config, "DATA_RAW_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="betting_lines"):
        betting_lines.load_betting_lines()
