import datetime

from nba_game_projections import config


def test_paths_are_under_project_root():
    assert config.DATA_RAW_DIR == config.PROJECT_ROOT / "data" / "raw"
    assert config.DATA_PROCESSED_DIR == config.PROJECT_ROOT / "data" / "processed"
    assert config.ROSTER_OVERRIDES_PATH == config.PROJECT_ROOT / "config" / "roster_overrides.json"


def test_current_season_before_october_is_current_year(monkeypatch):
    class FixedDate(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 3, 1)

    monkeypatch.setattr(config.datetime, "date", FixedDate)
    assert config.current_season() == 2026


def test_current_season_from_october_is_next_year(monkeypatch):
    class FixedDate(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 15)

    monkeypatch.setattr(config.datetime, "date", FixedDate)
    assert config.current_season() == 2027


def test_season_range_starts_at_2017(monkeypatch):
    class FixedDate(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 3, 1)

    monkeypatch.setattr(config.datetime, "date", FixedDate)
    seasons = list(config.season_range())
    assert seasons[0] == 2017
    assert seasons[-1] == 2026
