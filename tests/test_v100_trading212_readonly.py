import base64
import io
import json
import os

import pytest

from rlm.v100 import trading212_readonly as trading


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("SYNTA_TRADING212_API_KEY", "synthetic-key")
    monkeypatch.setenv("SYNTA_TRADING212_API_SECRET", "synthetic-secret")
    calls = []

    class Connection:
        status = 200
        payload = b'{"id":123,"cash":{"free":100},"accountId":456}'
        sock = None

        def __init__(self, host, timeout):
            calls.append((host, timeout))
            self.body = io.BytesIO(self.payload)

        def request(self, method, path, *, headers):
            calls.append((method, path, headers))

        def getresponse(self):
            return self

        def read1(self, size):
            return self.body.read(size)

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(trading.http.client, "HTTPSConnection", Connection)
    return Connection, calls


def test_demo_default_only_get_and_private_ids_removed(transport):
    _, calls = transport
    client = trading.Trading212ReadOnly()
    assert calls == []
    assert client.account_summary() == {"cash": {"free": 100}}
    assert calls[0] == ("demo.trading212.com", 10)
    method, path, headers = calls[1]
    assert (method, path) == ("GET", "/api/v0/equity/account/summary")
    expected = base64.b64encode(b"synthetic-key:synthetic-secret").decode()
    assert headers["Authorization"] == "Basic " + expected
    assert calls[-1] == "closed"


def test_explicit_live_positions_and_no_proxy_use(transport, monkeypatch):
    connection, calls = transport
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid")
    connection.payload = b'[{"ticker":"TEST_EQ","quantity":1}]'
    assert trading.Trading212ReadOnly(environment="live").positions() == [
        {"ticker": "TEST_EQ", "quantity": 1}
    ]
    assert calls[0][0] == "live.trading212.com"
    assert calls[1][0:2] == ("GET", "/api/v0/equity/positions")


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_redirect_never_followed_or_auth_leaked(transport, status):
    connection, calls = transport
    connection.status = status
    connection.payload = b"https://attacker.invalid/synthetic-secret"
    with pytest.raises(trading.Trading212ReadError, match="redirect refused") as error:
        trading.Trading212ReadOnly().account_summary()
    assert "synthetic" not in str(error.value)
    assert len([call for call in calls if isinstance(call, tuple) and call[0] == "GET"]) == 1
    assert calls[-1] == "closed"


def test_response_size_bounded(transport, monkeypatch):
    connection, calls = transport
    monkeypatch.setattr(trading, "MAX_RESPONSE_BYTES", 16)
    connection.payload = b"x" * 32
    with pytest.raises(trading.Trading212ReadError, match="size limit"):
        trading.Trading212ReadOnly().account_summary()
    assert calls[-1] == "closed"


@pytest.mark.parametrize("payload", [b"not-json synthetic-secret", b"\xff", b"[]"])
def test_invalid_summary_sanitized(transport, payload):
    connection, _ = transport
    connection.payload = payload
    with pytest.raises(trading.Trading212ReadError) as error:
        trading.Trading212ReadOnly().account_summary()
    assert "synthetic-secret" not in str(error.value)


def test_arbitrary_endpoint_refused_before_credentials_or_network(transport, monkeypatch):
    _, calls = transport
    monkeypatch.delenv("SYNTA_TRADING212_API_KEY")
    with pytest.raises(ValueError, match="Unsupported"):
        trading.Trading212ReadOnly().read("https://attacker.invalid")
    assert calls == []


def test_missing_credentials_no_network(transport, monkeypatch):
    _, calls = transport
    monkeypatch.delenv("SYNTA_TRADING212_API_SECRET")
    with pytest.raises(trading.Trading212ReadError, match="credentials"):
        trading.Trading212ReadOnly().positions()
    assert calls == []


@pytest.mark.parametrize("timeout", [0, -1, 31, float("inf"), float("nan")])
def test_timeout_limits(timeout):
    with pytest.raises(ValueError, match="Timeout"):
        trading.Trading212ReadOnly(timeout=timeout)


def test_no_retry_on_http_error(transport):
    connection, calls = transport
    connection.status = 429
    with pytest.raises(trading.Trading212ReadError, match="HTTP 429"):
        trading.Trading212ReadOnly().positions()
    assert len(calls) == 3


def test_transport_exception_does_not_expose_credentials(transport, monkeypatch):
    connection, calls = transport

    def fail(_self):
        raise OSError("synthetic-secret leaked by external transport")

    monkeypatch.setattr(connection, "getresponse", fail)
    with pytest.raises(trading.Trading212ReadError) as error:
        trading.Trading212ReadOnly().positions()
    assert "synthetic-secret" not in str(error.value)
    assert error.value.__suppress_context__
    assert calls[-1] == "closed"


def test_body_deadline_stops_reading(transport, monkeypatch):
    _, calls = transport
    ticks = iter([0, 11])
    monkeypatch.setattr(trading.time, "monotonic", lambda: next(ticks))
    with pytest.raises(trading.Trading212ReadError, match="timed out"):
        trading.Trading212ReadOnly().positions()
    assert calls[-1] == "closed"


@pytest.mark.parametrize("payload", [b"{}", b"[1]"])
def test_positions_shape_required(transport, payload):
    connection, _ = transport
    connection.payload = payload
    with pytest.raises(trading.Trading212ReadError, match="list of objects"):
        trading.Trading212ReadOnly().positions()


def test_cli_existing_env_no_prompt_and_explicit_live(transport, monkeypatch, capsys):
    connection, calls = transport
    connection.payload = b"[]"
    monkeypatch.setattr(trading.Trading212ReadOnly, "account_summary", lambda _self: {"cash": 100})
    monkeypatch.setattr(
        trading.getpass, "getpass", lambda _prompt: pytest.fail("Unexpected prompt")
    )
    assert trading.main(["--environment", "live"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["environment"] == "live"
    assert result["scope"] == "read-only; orders disabled"
    assert calls[0][0] == "live.trading212.com"
    assert os.environ["SYNTA_TRADING212_API_SECRET"] == "synthetic-secret"


def test_cli_private_prompts_restores_environment(transport, monkeypatch, capsys):
    for name in ("SYNTA_TRADING212_API_KEY", "SYNTA_TRADING212_API_SECRET"):
        monkeypatch.delenv(name)
    answers = iter(["prompt-key", "prompt-secret"])
    monkeypatch.setattr(trading.getpass, "getpass", lambda _prompt: next(answers))
    monkeypatch.setattr(trading.Trading212ReadOnly, "positions", lambda _self: [])
    assert trading.main([]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["environment"] == "demo"
    assert "prompt-secret" not in output
    assert "SYNTA_TRADING212_API_KEY" not in os.environ
    assert "SYNTA_TRADING212_API_SECRET" not in os.environ


def test_cli_error_sanitized_restores_partial_env(transport, monkeypatch, capsys):
    connection, _ = transport
    connection.status = 500
    monkeypatch.delenv("SYNTA_TRADING212_API_SECRET")
    monkeypatch.setattr(trading.getpass, "getpass", lambda _prompt: "prompt-secret")
    assert trading.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Trading 212 read failed; no orders submitted.\n"
    assert os.environ["SYNTA_TRADING212_API_KEY"] == "synthetic-key"
    assert "SYNTA_TRADING212_API_SECRET" not in os.environ
