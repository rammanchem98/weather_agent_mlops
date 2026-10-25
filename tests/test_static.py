from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.static import mount_frontend


def _make_app(dist_dir) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    mount_frontend(app, dist_dir=str(dist_dir))
    return app


def test_serves_index_html_at_root(tmp_path):
    (tmp_path / "index.html").write_text("<div id='root'></div>")
    client = TestClient(_make_app(tmp_path))

    response = client.get("/")

    assert response.status_code == 200
    assert "<div id='root'></div>" in response.text


def test_api_routes_still_win_over_the_frontend_mount(tmp_path):
    (tmp_path / "index.html").write_text("<div id='root'></div>")
    client = TestClient(_make_app(tmp_path))

    response = client.get("/health")

    assert response.json() == {"status": "ok"}


def test_missing_build_serves_api_only(tmp_path):
    client = TestClient(_make_app(tmp_path / "does-not-exist"))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 404
