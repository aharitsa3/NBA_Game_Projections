from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections.app import db, main


def _use_isolated_db(monkeypatch, tmp_path):
    test_engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setattr(db, "engine", test_engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=test_engine))
    monkeypatch.setattr(db.config, "MODEL_ARTIFACTS_DIR", tmp_path / "models")


def test_add_list_update_delete_round_trip(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)

    with TestClient(main.app) as client:
        add_response = client.post(
            "/admin/roster-overrides",
            json={
                "player_id": 1,
                "player_name": "Player One",
                "team_id": 100,
                "status": "out",
                "note": "ankle",
            },
        )
        assert add_response.status_code == 200
        override_id = add_response.json()["id"]

        list_response = client.get("/admin/roster-overrides")
        assert list_response.status_code == 200
        overrides = list_response.json()
        assert len(overrides) == 1
        assert overrides[0]["player_id"] == 1
        assert overrides[0]["status"] == "out"

        update_response = client.post(
            "/admin/roster-overrides",
            json={
                "player_id": 1,
                "player_name": "Player One",
                "team_id": 100,
                "status": "in",
                "note": "cleared",
            },
        )
        assert update_response.status_code == 200
        assert update_response.json()["id"] == override_id
        assert update_response.json()["status"] == "in"

        list_after_update = client.get("/admin/roster-overrides").json()
        assert len(list_after_update) == 1
        assert list_after_update[0]["status"] == "in"

        delete_response = client.delete(f"/admin/roster-overrides/{override_id}")
        assert delete_response.status_code == 200

        list_after_delete = client.get("/admin/roster-overrides").json()
        assert list_after_delete == []


def test_add_rejects_invalid_status(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)

    with TestClient(main.app) as client:
        response = client.post(
            "/admin/roster-overrides",
            json={
                "player_id": 2,
                "player_name": "Player Two",
                "team_id": 100,
                "status": "benched",
            },
        )

    assert response.status_code == 422


def test_delete_unknown_id_returns_404(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)

    with TestClient(main.app) as client:
        response = client.delete("/admin/roster-overrides/999")

    assert response.status_code == 404
