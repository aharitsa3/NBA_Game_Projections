from fastapi.testclient import TestClient

from nba_game_projections.app.routers import admin


def test_refresh_data_calls_downloader_for_current_season_and_returns_success(monkeypatch):
    monkeypatch.setattr(admin.config, "current_season", lambda: 2027)

    calls = []

    def fake_download_season_files(folder, seasons, *, force=False):
        calls.append((folder, list(seasons)))
        return [f"data/raw/{folder}/file_2027.parquet"]

    monkeypatch.setattr(admin.download, "download_season_files", fake_download_season_files)

    from nba_game_projections.app.main import app

    with TestClient(app) as client:
        response = client.post("/admin/refresh-data")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["season"] == 2027
    assert set(body["downloaded"]) == {"player_box", "team_box", "schedules"}

    called_folders = {folder for folder, _ in calls}
    assert called_folders == {"player_box", "team_box", "schedules"}
    assert all(seasons == [2027] for _, seasons in calls)
