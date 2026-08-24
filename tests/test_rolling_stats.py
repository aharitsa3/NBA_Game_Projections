import pandas as pd

from nba_game_projections.features import rolling_stats

_WINDOW = 3


def _row(**overrides):
    row = {
        "game_id": 1,
        "season": 2021,
        "game_date": pd.Timestamp("2021-01-01"),
        "athlete_id": 100,
        "athlete_display_name": "Test Player",
        "team_id": 1,
        "home_away": "home",
        "opponent_team_id": 2,
        "starter": True,
        "did_not_play": False,
        "dnp_reason": None,
        "minutes": 30.0,
        "points": 20.0,
        "rebounds": 5.0,
        "offensive_rebounds": 1.0,
        "defensive_rebounds": 4.0,
        "assists": 3.0,
        "steals": 1.0,
        "blocks": 1.0,
        "turnovers": 2.0,
        "fouls": 2.0,
        "field_goals_made": 8.0,
        "field_goals_attempted": 15.0,
        "three_point_field_goals_made": 2.0,
        "three_point_field_goals_attempted": 5.0,
        "free_throws_made": 2.0,
        "free_throws_attempted": 2.0,
    }
    row.update(overrides)
    return row


def _player_games(athlete_id, team_id, game_ids, dates, **stat_overrides):
    return [
        _row(
            game_id=gid,
            athlete_id=athlete_id,
            team_id=team_id,
            game_date=pd.Timestamp(date),
            **stat_overrides,
        )
        for gid, date in zip(game_ids, dates, strict=True)
    ]


def _trade_dataset():
    """Team 1 (A) and team 2 (B), player 100 (X) traded from A to B after 2021-01-04.

    Player 200 (Y) stays on team A for all 8 of its games; player 300 (Z)
    stays on team B for all 8 of its games. Team A's games are A1..A8
    (2021-01-01..08); team B's games are B1..B8 on the same dates but a
    disjoint game_id range, so the two teams' trailing windows can be
    reasoned about independently.
    """
    dates = [f"2021-01-{day:02d}" for day in range(1, 9)]
    a_game_ids = list(range(1, 9))  # A1..A8
    b_game_ids = list(range(101, 109))  # B1..B8

    x_rows = _player_games(
        100, 1, a_game_ids[:4], dates[:4], points=20.0, rebounds=5.0, assists=3.0,
        minutes=30.0, field_goals_made=8.0, field_goals_attempted=15.0,
        three_point_field_goals_made=2.0,
    ) + _player_games(
        100, 2, b_game_ids[4:8], dates[4:8], points=20.0, rebounds=5.0, assists=3.0,
        minutes=30.0, field_goals_made=8.0, field_goals_attempted=15.0,
        three_point_field_goals_made=2.0,
    )
    y_rows = _player_games(
        200, 1, a_game_ids, dates, points=10.0, rebounds=6.0, assists=2.0,
        minutes=25.0, field_goals_made=4.0, field_goals_attempted=10.0,
        three_point_field_goals_made=1.0,
    )
    z_rows = _player_games(
        300, 2, b_game_ids, dates, points=15.0, rebounds=4.0, assists=5.0,
        minutes=28.0, field_goals_made=6.0, field_goals_attempted=12.0,
        three_point_field_goals_made=2.0,
    )
    return pd.DataFrame(x_rows + y_rows + z_rows)


def test_shortly_after_trade_player_gradually_phases_between_teams():
    player_box = _trade_dataset()

    team_a = rolling_stats.team_rolling_features(
        player_box, team_id=1, as_of_date="2021-01-06", window=_WINDOW
    )
    team_b = rolling_stats.team_rolling_features(
        player_box, team_id=2, as_of_date="2021-01-06", window=_WINDOW
    )

    # Team A's last 3 games before 01-06 are A3, A4, A5: X appears in A3/A4
    # (not yet fully rolled off), so his production is still reflected.
    assert team_a["roster_size"] == 2
    assert team_a["points"] == 30.0  # X (20) + Y (10)

    # Team B's last 3 games before 01-06 are B3, B4, B5: X appears in B5
    # only (already phasing in), alongside Z who's been there all along.
    assert team_b["roster_size"] == 2
    assert team_b["points"] == 35.0  # Z (15) + X (20)


def test_long_after_trade_player_fully_moves_to_new_team():
    player_box = _trade_dataset()

    team_a = rolling_stats.team_rolling_features(
        player_box, team_id=1, as_of_date="2021-01-08", window=_WINDOW
    )
    team_b = rolling_stats.team_rolling_features(
        player_box, team_id=2, as_of_date="2021-01-08", window=_WINDOW
    )

    # Team A's last 3 games before 01-08 are A5, A6, A7: X never appears
    # (traded away), so he's fully dropped out.
    assert team_a["roster_size"] == 1
    assert team_a["points"] == 10.0  # Y only

    # Team B's last 3 games before 01-08 are B5, B6, B7: X appears in all
    # three, fully phased in alongside Z.
    assert team_b["roster_size"] == 2
    assert team_b["points"] == 35.0  # Z (15) + X (20)


def test_active_player_ids_overrides_automatic_roster():
    player_box = _trade_dataset()
    # Player 999 has never played for team 1 (or team 2) at all — their own
    # trailing history is on an unrelated team 3.
    other_team_rows = _player_games(
        999, 3, [201, 202, 203], ["2021-01-01", "2021-01-02", "2021-01-03"],
        points=50.0, rebounds=9.0, assists=1.0, minutes=40.0,
        field_goals_made=10.0, field_goals_attempted=15.0,
        three_point_field_goals_made=3.0,
    )
    player_box = pd.concat([player_box, pd.DataFrame(other_team_rows)], ignore_index=True)

    result = rolling_stats.team_rolling_features(
        player_box,
        team_id=1,
        as_of_date="2021-01-06",
        window=_WINDOW,
        active_player_ids=[999],
    )

    assert result["roster_size"] == 1
    assert result["points"] == 50.0
    assert result["minutes_per_game"] == 40.0


def test_game_on_as_of_date_is_excluded_no_leakage():
    rows = _player_games(
        100, 1, [1, 2], ["2021-01-01", "2021-01-05"], points=10.0,
    )
    # A game dated exactly on the as_of_date, with wildly different stats —
    # must never influence the result.
    rows.append(
        _row(
            game_id=3,
            athlete_id=100,
            team_id=1,
            game_date=pd.Timestamp("2021-01-10"),
            points=999.0,
        )
    )
    player_box = pd.DataFrame(rows)

    result = rolling_stats.team_rolling_features(
        player_box, team_id=1, as_of_date="2021-01-10", window=_WINDOW
    )

    assert result["points"] == 10.0
    assert result["roster_size"] == 1


def test_zero_history_team_returns_zero_not_error():
    player_box = _trade_dataset()

    result = rolling_stats.team_rolling_features(
        player_box, team_id=999, as_of_date="2021-01-01", window=_WINDOW
    )

    assert result["roster_size"] == 0
    assert result["points"] == 0.0
    assert result["minutes_per_game"] == 0.0
    assert pd.isna(result["efg_pct"])
