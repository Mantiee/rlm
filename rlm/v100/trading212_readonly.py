"""Opt-in, read-only Trading 212 Invest transport; no automatic network activity."""

from __future__ import annotations

import argparse
import base64
import getpass
import http.client
import json
import math
import os
import sys
import time
import warnings
from typing import Any, Literal

HOSTS = {"demo": "demo.trading212.com", "live": "live.trading212.com"}
MAX_RESPONSE_BYTES = 1_048_576


class Trading212ReadError(RuntimeError):
    """A sanitized transport or response failure without credentials/body content."""


def without_account_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_account_ids(item)
            for key, item in value.items()
            if key.lower().replace("_", "") not in {"accountid", "accountnumber", "id"}
        }
    if isinstance(value, list):
        return [without_account_ids(item) for item in value]
    return value


class Trading212ReadOnly:
    """Only account summary and positions; environment credentials are read per request.

    Constructing this object performs no requests. Live is an explicit read-only opt-in.
    Response objects remain in memory and must be treated as private financial data.
    """

    def __init__(self, *, environment: Literal["demo", "live"] = "demo", timeout: float = 10):
        if environment not in HOSTS:
            raise ValueError("Trading 212 environment must be demo or live")
        if not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ValueError("Timeout must be greater than zero and at most 30 seconds")
        self.environment = environment
        self.timeout = timeout

    def account_summary(self) -> dict[str, Any]:
        result = self.read("account/summary")
        if not isinstance(result, dict):
            raise Trading212ReadError("Account summary response must be an object")
        return result

    def positions(self) -> list[dict[str, Any]]:
        result = self.read("positions")
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise Trading212ReadError("Positions response must be a list of objects")
        return result

    def read(self, resource: str) -> Any:
        # Allowlist before constructing auth: callers cannot supply an arbitrary URL/path.
        if resource not in {"account/summary", "positions"}:
            raise ValueError("Unsupported read-only resource")
        key = os.environ.get("SYNTA_TRADING212_API_KEY", "")
        secret = os.environ.get("SYNTA_TRADING212_API_SECRET", "")
        if not key or not secret or ":" in key or any(c in key + secret for c in "\r\n"):
            raise Trading212ReadError("Missing or invalid Trading 212 environment credentials")
        authorization = "Basic " + base64.b64encode(f"{key}:{secret}".encode()).decode("ascii")
        # HTTPSConnection does not consult proxy environment variables or follow redirects.
        connection = http.client.HTTPSConnection(HOSTS[self.environment], timeout=self.timeout)
        deadline = time.monotonic() + self.timeout
        try:
            connection.request(
                "GET",
                "/api/v0/equity/" + resource,
                headers={"Authorization": authorization, "Accept": "application/json"},
            )
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise Trading212ReadError("Trading 212 redirect refused")
            if response.status != 200:
                raise Trading212ReadError(f"Trading 212 read failed (HTTP {response.status})")
            body = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Trading212ReadError("Trading 212 read timed out")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(body)))
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise Trading212ReadError("Trading 212 response exceeds size limit")
                if not chunk:
                    break
            return without_account_ids(json.loads(body.decode("utf-8")))
        except Trading212ReadError:
            raise
        except Exception:
            # Network exceptions may include remote data; never propagate their text.
            raise Trading212ReadError("Trading 212 read failed; response unavailable") from None
        finally:
            connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private Trading 212 portfolio read; no orders")
    parser.add_argument("--environment", choices=("demo", "live"), default="demo")
    args = parser.parse_args(argv)
    names = ("SYNTA_TRADING212_API_KEY", "SYNTA_TRADING212_API_SECRET")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name, label in zip(names, ("API key", "API secret"), strict=True):
            if not os.environ.get(name):
                with warnings.catch_warnings():
                    # Refuse getpass's echoing fallback when no private terminal is available.
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    os.environ[name] = getpass.getpass(f"Trading 212 {args.environment} {label}: ")
        client = Trading212ReadOnly(environment=args.environment)
        summary = client.account_summary()
        positions = client.positions()
        print(
            json.dumps(
                {
                    "environment": args.environment,
                    "scope": "read-only; orders disabled",
                    "account_summary": summary,
                    "positions": positions,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (Exception, KeyboardInterrupt):
        # Includes prompt/encoding errors; never show remote data or credentials in tracebacks.
        print("Trading 212 read failed; no orders submitted.", file=sys.stderr)
        return 2
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
