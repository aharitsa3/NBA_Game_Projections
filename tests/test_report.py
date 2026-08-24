import math

import pandas as pd
import pytest

from nba_game_projections.backtest import report


def test_accuracy_table_computes_per_season_and_pooled_with_na_exclusion():
    df = pd.DataFrame(
        {
            "test_season": [2018, 2018, 2018, 2018, 2019, 2019],
            "predicted_home_win": [True, True, False, False, True, True],
            "baseline_x": pd.array(
                [True, pd.NA, False, True, pd.NA, True], dtype="boolean"
            ),
            "actual_home_win": [True, False, False, True, True, True],
        }
    )

    result = report.accuracy_table(
        df, {"model": "predicted_home_win", "baseline_x": "baseline_x"}
    )

    def row(predictor, season):
        match = result[(result["predictor"] == predictor) & (result["test_season"] == season)]
        return match.iloc[0]

    model_2018 = row("model", 2018)
    assert model_2018["n_games"] == 4
    assert model_2018["n_correct"] == 2
    assert model_2018["accuracy"] == pytest.approx(0.5)

    model_2019 = row("model", 2019)
    assert model_2019["n_games"] == 2
    assert model_2019["accuracy"] == pytest.approx(1.0)

    model_pooled = row("model", "pooled")
    assert model_pooled["n_games"] == 6
    assert model_pooled["accuracy"] == pytest.approx(4 / 6)

    baseline_2018 = row("baseline_x", 2018)
    assert baseline_2018["n_games"] == 3  # one NA excluded
    assert baseline_2018["accuracy"] == pytest.approx(1.0)

    baseline_pooled = row("baseline_x", "pooled")
    assert baseline_pooled["n_games"] == 4  # two NA excluded across both seasons
    assert baseline_pooled["accuracy"] == pytest.approx(1.0)


def test_margin_error_table_computes_mae_and_rmse_per_season_and_pooled():
    df = pd.DataFrame(
        {
            "test_season": [2018, 2018, 2019, 2019],
            "predicted_margin": [5.0, -3.0, 10.0, 0.0],
            "actual_margin": [3.0, -3.0, 8.0, 5.0],
        }
    )

    result = report.margin_error_table(df)

    def row(season):
        return result[result["test_season"] == season].iloc[0]

    r2018 = row(2018)
    assert r2018["mae"] == pytest.approx(1.0)
    assert r2018["rmse"] == pytest.approx(math.sqrt(2.0))

    r2019 = row(2019)
    assert r2019["mae"] == pytest.approx(3.5)
    assert r2019["rmse"] == pytest.approx(math.sqrt(14.5))

    pooled = row("pooled")
    assert pooled["n_games"] == 4
    assert pooled["mae"] == pytest.approx(2.25)
    assert pooled["rmse"] == pytest.approx(math.sqrt(8.25))


def test_brier_score_known_values():
    predicted = pd.Series([1.0, 0.0, 0.5, 0.8])
    actual = pd.Series([True, False, True, False])

    assert report.brier_score(predicted, actual) == pytest.approx(0.2225)
    assert report.brier_score(pd.Series([1.0, 0.0]), pd.Series([True, False])) == pytest.approx(0.0)
    assert report.brier_score(pd.Series([1.0, 0.0]), pd.Series([False, True])) == pytest.approx(1.0)


def test_calibration_table_bins_and_computes_realized_win_rate():
    predicted = pd.Series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    actual = pd.Series([False, False, True, True, True, False])

    table = report.calibration_table(predicted, actual, n_bins=2).sort_values(
        "predicted_prob_mean"
    )

    assert len(table) == 2
    low_bin, high_bin = table.iloc[0], table.iloc[1]
    assert low_bin["n_games"] == 3
    assert low_bin["predicted_prob_mean"] == pytest.approx(0.2)
    assert low_bin["actual_win_rate"] == pytest.approx(1 / 3)

    assert high_bin["n_games"] == 3
    assert high_bin["predicted_prob_mean"] == pytest.approx(0.8)
    assert high_bin["actual_win_rate"] == pytest.approx(2 / 3)


def _schedules_fixture() -> pd.DataFrame:
    rows = [
        (1, "2019-11-01", 1, 2, 100, 90, True, False),
        (2, "2019-11-03", 2, 1, 95, 105, False, True),
        (3, "2019-11-05", 1, 2, 110, 100, True, False),
        (4, "2019-11-07", 2, 1, 90, 95, False, True),
    ]
    return pd.DataFrame(
        [
            {
                "game_id": gid,
                "season": 2019,
                "date": pd.Timestamp(date, tz="America/New_York"),
                "home_team_id": home_id,
                "home_team_abbreviation": "BOS" if home_id == 1 else "LAL",
                "away_team_id": away_id,
                "away_team_abbreviation": "LAL" if home_id == 1 else "BOS",
                "home_score": hs,
                "away_score": as_,
                "home_winner": hw,
                "away_winner": aw,
            }
            for gid, date, home_id, away_id, hs, as_, hw, aw in rows
        ]
    )


def _betting_lines_fixture() -> pd.DataFrame:
    rows = [
        ("2019-11-01", "BOS", "LAL", -5.0),
        ("2019-11-03", "LAL", "BOS", 3.0),
    ]
    return pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp(date),
                "home_team_abbrev": home,
                "visit_team_abbrev": away,
                "line": line,
            }
            for date, home, away, line in rows
        ]
    )


def _predictions_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "test_season": [2019, 2019, 2019, 2019],
            "game_id": [1, 2, 3, 4],
            "home_team_id": [1, 2, 1, 2],
            "away_team_id": [2, 1, 2, 1],
            "home_score": [100, 95, 110, 90],
            "away_score": [90, 105, 100, 95],
            "actual_margin": [10, -10, 10, -5],
            "actual_home_win": [True, False, True, False],
            "predicted_margin": [8.0, -8.0, 9.0, -4.0],
            "predicted_home_win": [True, False, True, False],
            "predicted_win_probability": [0.75, 0.2, 0.8, 0.35],
        }
    )


def test_attach_baselines_aligns_by_game_id():
    predictions = _predictions_fixture()
    schedules = _schedules_fixture()
    betting_lines = _betting_lines_fixture()

    result = report.attach_baselines(predictions, schedules, betting_lines)

    assert len(result) == 4
    assert set(result["game_id"]) == {1, 2, 3, 4}
    assert bool(result.loc[result["game_id"] == 1, "baseline_always_home"].iloc[0]) is True
    assert bool(result.loc[result["game_id"] == 1, "baseline_vegas_favorite"].iloc[0]) is True


def test_build_report_end_to_end():
    predictions = _predictions_fixture()
    schedules = _schedules_fixture()
    betting_lines = _betting_lines_fixture()

    result = report.build_report(predictions, schedules, betting_lines)

    assert set(result.keys()) == {"accuracy", "margin_error", "brier_score", "calibration"}
    assert set(result["accuracy"]["predictor"]) == {
        "model",
        "always_home",
        "better_record",
        "elo_alone",
        "vegas_favorite",
    }
    assert (result["accuracy"]["test_season"] == "pooled").sum() == 5
    assert result["margin_error"]["test_season"].tolist() == [2019, "pooled"]
    assert isinstance(result["brier_score"], float)
    assert result["calibration"]["n_games"].sum() == 4
