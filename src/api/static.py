# src/api/static.py
import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.common.config import ROOT_DIR

logger = logging.getLogger(__name__)

FRONTEND_DIST = os.path.join(ROOT_DIR, "frontend", "dist")


def mount_frontend(app: FastAPI, dist_dir: str = FRONTEND_DIST) -> None:
    """Serves the built React app (frontend/dist) at "/".

    Must be called AFTER all API routes are registered: a mount at "/" matches
    every path, and Starlette checks routes in registration order, so anything
    added after it would be unreachable.

    If there is no build (local dev uses the Vite dev server, unit tests don't
    build the front end), the API is served alone."""
    if not os.path.isdir(dist_dir):
        logger.info(f"No front end build found at {dist_dir} — serving the API only")
        return
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")
