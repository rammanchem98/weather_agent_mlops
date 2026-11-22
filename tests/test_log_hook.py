"""The redaction log hook must keep log records in the shape formatters expect
(uvicorn's access log unpacks record.args as a 5-tuple) and never raise."""
import logging
import os
import socket
import subprocess
import sys

import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest
from uvicorn.logging import AccessFormatter

import src.common.redact as redact
from src.common.redact import REDACTED

OW_KEY = "SECRET123"
GEMINI_KEY = "DUMMY-GEMINI-KEY-789"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def secrets_env(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", OW_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)


def make_record(msg, args, name="uvicorn.access", exc_info=None):
    # Logger.makeRecord goes through the installed record factory (the hook).
    return logging.getLogger(name).makeRecord(name, logging.INFO, __file__, 1, msg, args, exc_info)


# --- unit tests --------------------------------------------------------------

def test_uvicorn_access_formatter_works_and_query_secret_is_redacted(secrets_env):
    record = make_record('%s - "%s %s HTTP/%s" %d',
                         ("127.0.0.1:5000", "GET", f"/health?appid={OW_KEY}", "1.1", 200))

    assert isinstance(record.args, tuple) and len(record.args) == 5   # shape preserved
    line = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False).format(record)

    assert line == f'127.0.0.1:5000 - "GET /health?appid={REDACTED} HTTP/1.1" 200 OK'
    assert OW_KEY not in line


def test_dict_args_are_redacted_and_keep_their_shape(secrets_env):
    record = make_record("key=%(key)s n=%(n)d", {"key": GEMINI_KEY, "n": 3}, name="test")

    assert record.args == {"key": REDACTED, "n": 3}
    assert record.getMessage() == f"key={REDACTED} n=3"


def test_non_string_args_and_non_string_msg_are_left_as_they_are(secrets_env):
    marker = object()
    record = make_record("%s %d %s", (marker, 7, None), name="test")
    assert record.args == (marker, 7, None)

    obj_msg = {"not": "a string"}
    assert make_record(obj_msg, (), name="test").msg is obj_msg


def test_string_msg_and_traceback_are_redacted(secrets_env):
    try:
        raise RuntimeError(f"failed with {GEMINI_KEY}")
    except RuntimeError:
        record = make_record(f"plain message {OW_KEY}", (), name="test", exc_info=sys.exc_info())

    assert OW_KEY not in record.msg
    assert GEMINI_KEY not in record.exc_text and "RuntimeError" in record.exc_text


def test_hook_leaves_the_record_untouched_on_an_unexpected_error(secrets_env):
    args = ("a", OW_KEY)
    with patch.object(redact, "redact_secrets", side_effect=ValueError("boom")):
        record = make_record("%s %s", args, name="test")

    assert record.msg == "%s %s" and record.args == args   # unchanged, nothing raised


# --- real server smoke test -----------------------------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status


def _read(path):
    # errors="replace": a stray byte in server output must not crash the checks
    return path.read_text(encoding="utf-8", errors="replace")


def test_real_uvicorn_server_logs_requests_without_errors_or_secrets(tmp_path):
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "OPENWEATHER_API_KEY": OW_KEY, "GEMINI_API_KEY": GEMINI_KEY,
           "PYTHONUNBUFFERED": "1", "env": "dev",
           # Windows writes redirected output in the console code page unless told otherwise
           "PYTHONIOENCODING": "utf-8"}
    log_path = tmp_path / "server.log"

    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                try:
                    assert _get(f"{base}/health") == 200
                    break
                except (OSError, AssertionError):
                    if proc.poll() is not None or time.monotonic() > deadline:
                        pytest.fail(f"server did not start:\n{_read(log_path)}")
                    time.sleep(0.25)

            assert _get(f"{base}/health?appid={OW_KEY}") == 200

            deadline = time.monotonic() + 10   # wait for the access-log line to be written
            while "/health?appid=" not in _read(log_path):
                assert time.monotonic() < deadline, "access-log line never appeared"
                time.sleep(0.1)
        finally:
            proc.terminate()
            proc.wait(timeout=15)

    output = _read(log_path)
    assert "Logging error" not in output
    assert '"GET /health HTTP/1.1" 200' in output
    assert f'"GET /health?appid={REDACTED} HTTP/1.1" 200' in output
    assert OW_KEY not in output
    assert GEMINI_KEY not in output
