from fastapi.testclient import TestClient

from nba_game_projections.app.main import app
from nba_game_projections.app.routers import admin


def test_get_retrain_status_for_unknown_job_returns_404():
    with TestClient(app) as client:
        response = client.get("/admin/retrain/does-not-exist")
    assert response.status_code == 404


def test_start_retrain_returns_job_id_and_running_status(monkeypatch):
    # Prevent the real background task from running at all so the endpoint's
    # own "create job, return running" behavior can be checked in isolation
    # from whatever the job eventually does.
    scheduled = []
    monkeypatch.setattr(
        admin.BackgroundTasks, "add_task", lambda self, func, *a, **kw: scheduled.append((func, a))
    )

    with TestClient(app) as client:
        response = client.post("/admin/retrain")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    job_id = body["job_id"]

    with TestClient(app) as client:
        status_response = client.get(f"/admin/retrain/{job_id}")
    assert status_response.json() == {"job_id": job_id, "status": "running", "error": None}

    # confirm the actual job function was scheduled (not run inline)
    assert len(scheduled) == 1
    assert scheduled[0][0] is admin._execute_retrain_job
    assert scheduled[0][1] == (job_id,)


def test_execute_retrain_job_marks_job_done_on_success(monkeypatch):
    monkeypatch.setattr(admin, "run_production_training", lambda session: object())
    admin._retrain_jobs["job-success"] = {"status": "running", "error": None}

    admin._execute_retrain_job("job-success")

    assert admin._retrain_jobs["job-success"] == {"status": "done", "error": None}


def test_execute_retrain_job_marks_job_failed_on_exception(monkeypatch):
    def failing_train(session):
        raise RuntimeError("boom")

    monkeypatch.setattr(admin, "run_production_training", failing_train)
    admin._retrain_jobs["job-failure"] = {"status": "running", "error": None}

    admin._execute_retrain_job("job-failure")

    assert admin._retrain_jobs["job-failure"] == {"status": "failed", "error": "boom"}


def test_status_endpoint_reflects_job_state(monkeypatch):
    monkeypatch.setattr(admin, "run_production_training", lambda session: object())
    admin._retrain_jobs["job-status-check"] = {"status": "running", "error": None}
    admin._execute_retrain_job("job-status-check")

    with TestClient(app) as client:
        response = client.get("/admin/retrain/job-status-check")

    assert response.json() == {"job_id": "job-status-check", "status": "done", "error": None}
