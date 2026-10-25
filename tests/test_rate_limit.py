from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.rate_limit import RateLimiter, client_ip

def test_allows_up_to_the_limit_then_blocks():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    results = [limiter.allow("1.2.3.4", now=t) for t in (0, 1, 2, 3)]
    assert results == [True, True, True, False]

def test_allows_again_once_old_requests_leave_the_window():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.allow("1.2.3.4", now=0)
    limiter.allow("1.2.3.4", now=1)
    assert not limiter.allow("1.2.3.4", now=59)
    assert limiter.allow("1.2.3.4", now=60)  # the request at t=0 has expired

def test_clients_are_limited_separately():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("1.1.1.1", now=0)
    assert limiter.allow("2.2.2.2", now=0)
    assert not limiter.allow("1.1.1.1", now=1)

def test_client_ip_trusts_only_the_last_forwarded_entry():
    request = MagicMock()
    request.headers = {"x-forwarded-for": "6.6.6.6, 203.0.113.7"}  # first entry is client-supplied
    assert client_ip(request) == "203.0.113.7"

def test_client_ip_falls_back_to_the_socket_address():
    request = MagicMock()
    request.headers = {}
    request.client.host = "127.0.0.1"
    assert client_ip(request) == "127.0.0.1"

@patch("server.check_input")
@patch("server.chat_rate_limiter", RateLimiter(max_requests=0, window_seconds=60))
def test_chat_returns_429_before_running_guardrails(mock_check_input):
    import server
    client = TestClient(server.app)

    response = client.post("/chat", json={"message": "What is the weather in Tokyo?"})

    assert response.status_code == 429
    mock_check_input.assert_not_called()  # no classifier (LLM) call was spent
