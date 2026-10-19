"""Bounded, unauthenticated GitHub PR/check-run snapshots for dashboard status."""

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from rlm.v100.common import atomic_json

VERIFIED_REPOSITORY = "Mantiee/rlm"
DEFAULT_PULL_REQUESTS = (1, 18)
API_ROOT = "https://api.github.com/repos/Mantiee/rlm"
CACHE_RELATIVE = Path("research/github-dashboard/latest.json")
MAX_CACHE_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_PULL_REQUESTS = 5
REFRESH_SECONDS = 10 * 60
REQUEST_TIMEOUT = (0.75, 1.5)
REQUEST_DEADLINE_SECONDS = 3.0
SHA_RE = re.compile(r"[0-9a-f]{40}")
CHECK_STATUSES = {"queued", "in_progress", "completed"}
CHECK_CONCLUSIONS = {"success", "neutral", "skipped"}
SNAPSHOT_STATES = {"available", "partial", "unavailable"}
CHECK_STATES = {"success", "failure", "pending", "unknown"}


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _valid_timestamp(value) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.utcoffset() is not None


def _valid_error(value) -> bool:
    if not isinstance(value, dict) or set(value) - {"kind", "detail", "http_status"}:
        return False
    if not isinstance(value.get("kind"), str) or not 1 <= len(value["kind"]) <= 64:
        return False
    if "detail" in value and (not isinstance(value["detail"], str) or len(value["detail"]) > 200):
        return False
    status = value.get("http_status")
    return status is None or (type(status) is int and 100 <= status <= 599)


def _valid_cached_check(row) -> bool:
    if not isinstance(row, dict) or set(row) - {
        "name",
        "status",
        "conclusion",
        "started_at",
        "completed_at",
        "details_url",
    }:
        return False
    if (
        not isinstance(row.get("name"), str)
        or len(row["name"]) > 120
        or row.get("status") not in CHECK_STATUSES
    ):
        return False
    conclusion = row.get("conclusion")
    if conclusion is not None and (not isinstance(conclusion, str) or len(conclusion) > 100):
        return False
    for key in ("started_at", "completed_at"):
        timestamp = row.get(key)
        if timestamp is not None and not _valid_timestamp(timestamp):
            return False
    details_url = row.get("details_url")
    return details_url is None or (
        isinstance(details_url, str)
        and len(details_url) <= 500
        and details_url.startswith("https://github.com/Mantiee/rlm/actions/")
    )


def _cached_check_state(rows: list[dict]) -> str:
    if any(
        row["status"] == "completed" and row.get("conclusion") not in CHECK_CONCLUSIONS
        for row in rows
    ):
        return "failure"
    if all(
        row["status"] == "completed" and row.get("conclusion") in CHECK_CONCLUSIONS for row in rows
    ):
        return "success"
    return "pending"


def _validate_cached_snapshot(value: dict) -> None:
    allowed_fields = {
        "schema",
        "repository",
        "source",
        "state",
        "observed_at",
        "pull_requests",
        "error",
    }
    if set(value) - allowed_fields:
        raise ValueError("Cached snapshot has unsupported fields")
    if (
        value.get("source") != "unauthenticated GitHub REST GET; read-only status evidence"
        or value.get("state") not in SNAPSHOT_STATES
        or not _valid_timestamp(value.get("observed_at"))
    ):
        raise ValueError("Cached snapshot status or timestamp is invalid")

    prs = value["pull_requests"]
    if not isinstance(prs, list) or len(prs) > MAX_PULL_REQUESTS:
        raise ValueError("Cached pull request list is invalid")
    seen_numbers = set()
    for row in prs:
        if not isinstance(row, dict) or set(row) - {
            "number",
            "title",
            "state",
            "draft",
            "head_sha",
            "head_ref",
            "requested_at",
            "observed_at",
            "check_state",
            "check_run_count",
            "check_runs",
            "error",
        }:
            raise ValueError("Cached pull request row is invalid")
        required = {
            "number",
            "state",
            "check_state",
            "check_run_count",
            "requested_at",
            "observed_at",
        }
        if row.get("state") == "unavailable":
            required.add("error")
        else:
            required.update({"title", "draft", "head_sha", "head_ref", "check_runs"})
        if not required <= row.keys():
            raise ValueError("Cached pull request row is incomplete")
        number = row.get("number")
        if type(number) is not int or not 1 <= number <= 1_000_000 or number in seen_numbers:
            raise ValueError("Cached pull request identity is invalid")
        seen_numbers.add(number)
        if (
            row.get("state") not in ("open", "closed", "unavailable")
            or row.get("check_state") not in CHECK_STATES
            or not _valid_timestamp(row.get("requested_at"))
            or not _valid_timestamp(row.get("observed_at"))
        ):
            raise ValueError("Cached pull request state or timestamp is invalid")

        check_state = row["check_state"]
        runs = row.get("check_runs")
        if row["state"] == "unavailable":
            if (
                check_state != "unknown"
                or row.get("check_run_count") is not None
                or runs is not None
                or not _valid_error(row.get("error"))
            ):
                raise ValueError("Unavailable pull request row is inconsistent")
            continue

        if (
            not isinstance(row.get("title"), str)
            or len(row["title"]) > 180
            or row.get("draft") is not None
            and not isinstance(row.get("draft"), bool)
            or not isinstance(row.get("head_sha"), str)
            or not SHA_RE.fullmatch(row["head_sha"])
            or not isinstance(row.get("head_ref"), str)
            or len(row["head_ref"]) > 120
            or not isinstance(runs, list)
            or len(runs) > 100
            or (row.get("check_run_count") is not None and type(row["check_run_count"]) is not int)
            or any(not _valid_cached_check(check) for check in runs)
        ):
            raise ValueError("Cached pull request metadata or check runs are invalid")

        has_error = "error" in row
        if has_error and not _valid_error(row["error"]):
            raise ValueError("Cached pull request error is invalid")
        if check_state == "unknown":
            if runs or row.get("check_run_count") is not None:
                raise ValueError("Unknown check state must not expose check rows")
        elif (
            has_error
            or not 1 <= len(runs) <= 100
            or type(row.get("check_run_count")) is not int
            or row["check_run_count"] != len(runs)
            or _cached_check_state(runs) != check_state
        ):
            raise ValueError("Cached check state does not match its complete check rows")

    state = value["state"]
    top_error = value.get("error")
    if top_error is not None and not _valid_error(top_error):
        raise ValueError("Cached snapshot error is invalid")
    available_rows = [row for row in prs if row.get("state") != "unavailable"]
    has_unknown = any(row.get("check_state") == "unknown" or "error" in row for row in prs)
    expected_state = (
        "unavailable"
        if not available_rows
        else "partial"
        if has_unknown or len(available_rows) != len(prs)
        else "available"
    )
    if state != expected_state or (not prs and top_error is None):
        raise ValueError("Cached snapshot state does not match its rows")


def _config(root: Path) -> tuple[list[int], dict | None]:
    """Read optional repo/PR selection; the network origin is intentionally fixed."""
    path = root / "research/github-dashboard/config.json"
    if not path.exists():
        return list(DEFAULT_PULL_REQUESTS), None
    try:
        if path.is_symlink() or path.stat().st_size > 8192:
            raise ValueError("GitHub dashboard config is linked or exceeds 8 KiB")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) - {"repository", "pull_requests"}:
            raise ValueError("GitHub dashboard config has unsupported fields")
        if value.get("repository") != VERIFIED_REPOSITORY:
            raise ValueError("Repository does not match the pinned Mantiee/rlm identity")
        prs = value.get("pull_requests", list(DEFAULT_PULL_REQUESTS))
        if (
            not isinstance(prs, list)
            or not 1 <= len(prs) <= MAX_PULL_REQUESTS
            or any(type(number) is not int or not 1 <= number <= 1_000_000 for number in prs)
            or len(set(prs)) != len(prs)
        ):
            raise ValueError("Select 1-5 distinct positive pull request numbers")
        return prs, None
    except (OSError, ValueError, TypeError) as error:
        return [], {"kind": "invalid_configuration", "detail": str(error)[:200]}


def _request_json(session, url: str) -> tuple[dict | None, dict | None]:
    """One unauthenticated same-origin GET, with bounded time and response size."""
    if not url.startswith(API_ROOT + "/"):
        return None, {"kind": "unsafe_endpoint", "detail": "Endpoint is outside pinned repository"}
    try:
        deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
        with session.get(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "SyntaReadOnlyDashboard/1.0",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=REQUEST_TIMEOUT,
            stream=True,
            allow_redirects=False,
            auth=None,
        ) as response:
            status = response.status_code
            if status != 200:
                if status == 401:
                    kind = "credentials_required"
                elif status == 403:
                    remaining = response.headers.get("X-RateLimit-Remaining")
                    kind = "rate_limited" if remaining == "0" else "forbidden_or_rate_limited"
                elif status == 429:
                    kind = "rate_limited"
                elif status == 404:
                    kind = "not_found_or_private"
                else:
                    kind = "http_error"
                return None, {"kind": kind, "http_status": status}
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_RESPONSE_BYTES:
                raise ValueError("GitHub response exceeds 512 KiB")
            chunks, total = [], 0
            for chunk in response.iter_content(8192):
                if time.monotonic() > deadline:
                    raise TimeoutError("GitHub request exceeded its 3 second total budget")
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise ValueError("GitHub response exceeds 512 KiB")
                chunks.append(chunk)
        value = json.loads(b"".join(chunks))
        if not isinstance(value, dict):
            raise ValueError("GitHub response must be a JSON object")
        return value, None
    except (requests.RequestException, OSError, ValueError, TypeError, TimeoutError) as error:
        return None, {"kind": "request_failed", "detail": str(error)[:200]}


def _check_state(checks: dict) -> str:
    if not isinstance(checks, dict):
        return "unknown"
    rows = checks.get("check_runs")
    total_count = checks.get("total_count")
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 100
        or type(total_count) is not int
        or not 1 <= total_count <= 100
        or total_count != len(rows)
        or any(not isinstance(row, dict) for row in rows)
    ):
        return "unknown"
    failed = any(
        row.get("status") == "completed"
        and row.get("conclusion") not in ("success", "neutral", "skipped")
        for row in rows
        if isinstance(row, dict)
    )
    if failed:
        return "failure"
    if all(
        isinstance(row, dict)
        and row.get("status") == "completed"
        and row.get("conclusion") in ("success", "neutral", "skipped")
        for row in rows
    ):
        return "success"
    return "pending"


def _safe_check(row: dict) -> dict:
    result = {
        "name": str(row.get("name", "unknown"))[:120],
        "status": row.get("status")
        if row.get("status") in ("queued", "in_progress", "completed")
        else "unknown",
        "conclusion": row.get("conclusion") if isinstance(row.get("conclusion"), str) else None,
        "started_at": row.get("started_at") if isinstance(row.get("started_at"), str) else None,
        "completed_at": row.get("completed_at")
        if isinstance(row.get("completed_at"), str)
        else None,
    }
    url = row.get("details_url")
    if isinstance(url, str) and url.startswith("https://github.com/Mantiee/rlm/actions/"):
        result["details_url"] = url[:500]
    return result


def _pull_request(session, number: int, now: float) -> dict:
    requested_at = _iso(now)
    pr, error = _request_json(session, f"{API_ROOT}/pulls/{number}")
    if error:
        return {
            "number": number,
            "state": "unavailable",
            "check_state": "unknown",
            "check_run_count": None,
            "requested_at": requested_at,
            "observed_at": _iso(time.time()),
            "error": error,
        }
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    sha = head.get("sha")
    if (
        type(pr.get("number")) is not int
        or pr["number"] != number
        or pr.get("state") not in ("open", "closed")
        or not isinstance(sha, str)
        or not SHA_RE.fullmatch(sha)
    ):
        return {
            "number": number,
            "state": "unavailable",
            "check_state": "unknown",
            "check_run_count": None,
            "requested_at": requested_at,
            "observed_at": _iso(time.time()),
            "error": {"kind": "invalid_pull_request_response"},
        }
    checks, error = _request_json(session, f"{API_ROOT}/commits/{sha}/check-runs?per_page=100")
    common = {
        "number": number,
        "title": str(pr.get("title", ""))[:180],
        "state": pr["state"],
        "draft": pr.get("draft") if isinstance(pr.get("draft"), bool) else None,
        "head_sha": sha,
        "head_ref": str(head.get("ref", ""))[:120],
        "requested_at": requested_at,
        "observed_at": _iso(time.time()),
    }
    if error:
        return {
            **common,
            "check_state": "unknown",
            "check_run_count": None,
            "check_runs": [],
            "error": error,
        }
    rows = checks.get("check_runs")
    if not isinstance(rows, list):
        return {
            **common,
            "check_state": "unknown",
            "check_run_count": None,
            "check_runs": [],
            "error": {"kind": "invalid_check_runs_response"},
        }
    safe_rows = [_safe_check(row) for row in rows[:100] if isinstance(row, dict)]
    check_state = _check_state(checks)
    total_count = checks.get("total_count")
    check_run_count = (
        total_count
        if check_state != "unknown" and type(total_count) is int and total_count == len(rows)
        else None
    )
    if check_state == "unknown":
        # Hide apparently successful rows when the endpoint did not prove completeness.
        safe_rows = []
    return {
        **common,
        "check_state": check_state,
        "check_run_count": check_run_count,
        "check_runs": safe_rows,
    }


def refresh(root: Path, *, now: float | None = None, session=None) -> dict:
    """Fetch configured PR/check snapshots and cache them; never mutates GitHub."""
    now = time.time() if now is None else now
    prs, config_error = _config(root)
    result = {
        "schema": "synta-github-dashboard-v1",
        "repository": VERIFIED_REPOSITORY,
        "source": "unauthenticated GitHub REST GET; read-only status evidence",
        "state": "unavailable",
        "observed_at": _iso(now),
        "pull_requests": [],
    }
    if config_error:
        result["error"] = config_error
    else:
        own_session = session is None
        session = session or requests.Session()
        if own_session:
            # Never import credentials from environment, .netrc, or a configured proxy auth.
            session.trust_env = False
        try:
            result["pull_requests"] = [_pull_request(session, number, now) for number in prs]
        finally:
            if own_session:
                session.close()
        statuses = result["pull_requests"]
        available = sum(row["state"] != "unavailable" for row in statuses)
        errors = sum("error" in row or row["check_state"] == "unknown" for row in statuses)
        result["state"] = (
            "unavailable"
            if available == 0
            else "partial"
            if errors or available < len(prs)
            else "available"
        )
    result["observed_at"] = _iso(time.time())
    path = root / CACHE_RELATIVE
    if path.is_symlink() or path.parent.is_symlink():
        result["state"] = "unavailable"
        result["error"] = {"kind": "unsafe_cache_path"}
        return result
    atomic_json(path, result)
    return result


def cached(root: Path, *, now: float | None = None) -> dict:
    """Load bounded cache without network access; stale data never stays green."""
    now = time.time() if now is None else now
    path = root / CACHE_RELATIVE
    if not path.is_file() or path.is_symlink():
        return {
            "schema": "synta-github-dashboard-v1",
            "repository": VERIFIED_REPOSITORY,
            "source": "unauthenticated GitHub REST GET; read-only status evidence",
            "state": "unavailable",
            "observed_at": _iso(now),
            "pull_requests": [],
            "error": {"kind": "no_cached_snapshot"},
        }
    try:
        if path.stat().st_size > MAX_CACHE_BYTES:
            raise ValueError("Cached snapshot exceeds 256 KiB")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema") != "synta-github-dashboard-v1"
            or value.get("repository") != VERIFIED_REPOSITORY
            or not isinstance(value.get("pull_requests"), list)
        ):
            raise ValueError("Cached GitHub snapshot identity is invalid")
        _validate_cached_snapshot(value)
        try:
            observed = datetime.fromisoformat(value["observed_at"])
            if observed.utcoffset() is None:
                raise ValueError("Cached timestamp must include a timezone")
            age = now - observed.timestamp()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Cached snapshot timestamp is invalid") from error
        value = {**value, "age_seconds": round(age, 3)}
        if age < 0:
            value.update(state="unknown", error={"kind": "future_snapshot_timestamp"})
        elif age >= REFRESH_SECONDS:
            value.update(state="stale", error={"kind": "cached_snapshot_expired"})
        if age < 0 or age >= REFRESH_SECONDS:
            value["pull_requests"] = [
                {
                    **row,
                    "previous_check_state": row.get("check_state", "unknown"),
                    "check_state": "unknown",
                    "check_run_count": None,
                    "check_runs": [],
                }
                if isinstance(row, dict)
                else row
                for row in value["pull_requests"]
            ]
        return value
    except (OSError, ValueError, TypeError) as error:
        return {
            "schema": "synta-github-dashboard-v1",
            "repository": VERIFIED_REPOSITORY,
            "source": "unauthenticated GitHub REST GET; read-only status evidence",
            "state": "unavailable",
            "observed_at": _iso(now),
            "pull_requests": [],
            "error": {"kind": "invalid_cache", "detail": str(error)[:200]},
        }
