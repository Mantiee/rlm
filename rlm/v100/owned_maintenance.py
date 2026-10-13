"""Bounded local drain for the dedicated node, including its legacy LAN-only listener."""

import http.client
import json
import time


def request_drain() -> tuple[int, dict]:
    errors = []
    for host in ("127.0.0.1", "192.168.0.9"):
        # Both addresses belong to this host. The legacy handler admits maintenance
        # only from loopback; bind the connection locally instead of spoofing headers.
        connection = http.client.HTTPConnection(
            host, 11435, timeout=3, source_address=("127.0.0.1", 0)
        )
        try:
            connection.request(
                "POST",
                "/api/synta-maintenance",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read(16385)
            if len(body) > 16384:
                raise ValueError("Owned maintenance reply exceeds byte budget")
            value = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError("Owned maintenance reply must be an object")
            return response.status, value
        except OSError as error:
            errors.append(str(error))
        finally:
            connection.close()
    raise OSError("Owned local maintenance endpoint unavailable: " + "; ".join(errors))


def drain(budget: float = 60) -> bool:
    started = time.monotonic()
    deadline = started + budget
    while True:
        status, value = request_drain()
        if status == 200 and value.get("state") == "drained":
            print(
                json.dumps({"stage": "owned-jobs-drained", "seconds": time.monotonic() - started}),
                flush=True,
            )
            return True
        if status != 503:
            raise ValueError("Owned maintenance rejected: " + str(status) + " " + str(value))
        remaining = deadline - time.monotonic()
        print(
            json.dumps(
                {
                    "stage": "waiting-for-owned-jobs",
                    "reason": str(value.get("error", "unknown"))[:300],
                    "remaining_seconds": max(0, round(remaining)),
                }
            ),
            flush=True,
        )
        if remaining <= 0:
            return False
        time.sleep(min(1, remaining))


if __name__ == "__main__":
    raise SystemExit(0 if drain() else 75)
