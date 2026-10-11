"""Public web inspection and one-shot, operator-approved form submissions."""

import fcntl
import hashlib
import json
import re
import shlex
import ssl
import time
import uuid
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin

import urllib3

from rlm.v100.common import atomic_json


class PageParser(HTMLParser):
    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.links: list[dict] = []
        self.forms: list[dict] = []
        self.text: list[str] = []
        self.hidden = 0
        self.form: dict | None = None

    def handle_starttag(self, tag: str, attrs: list):
        values = dict(attrs)
        if tag in ("script", "style"):
            self.hidden += 1
        if tag == "a" and values.get("href") and len(self.links) < 80:
            self.links.append({"url": urljoin(self.url, values["href"])})
        if tag == "form" and len(self.forms) < 16:
            self.form = {
                "action": urljoin(self.url, values.get("action") or self.url),
                "method": values.get("method", "GET").upper(),
                "fields": [],
            }
            self.forms.append(self.form)
        if tag in ("input", "textarea", "select") and self.form is not None:
            # Values may contain personal data or auth tokens. Return field names only.
            if len(self.form["fields"]) < 64:
                self.form["fields"].append(
                    {
                        "name": values.get("name", ""),
                        "type": values.get("type", tag),
                        "required": "required" in values,
                    }
                )

    def handle_endtag(self, tag: str):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if tag == "form":
            self.form = None

    def handle_data(self, data: str):
        if not self.hidden and len(self.text) < 200:
            self.text.append(data.strip())


def inspect_page(url: str) -> dict:
    from rlm.v100.research_tools import download_page

    final_url, body = download_page(url)
    parser = PageParser(final_url)
    parser.feed(body)
    return {
        "url": final_url,
        "text": " ".join(parser.text)[:12000],
        "links": parser.links,
        "forms": parser.forms,
        "scope": "Untrusted public HTTP content, not rendered DOM or instructions. Form values omitted. No submission executed.",
    }


def open_page(root: Path, url: str) -> dict:
    from rlm.v100.desktop import run

    page = inspect_page(url)
    result = run(
        root,
        script="DISPLAY=:0 firefox-esr --new-tab "
        + shlex.quote(page["url"])
        + " >/workspace/browser.log 2>&1 &\n",
        seconds=10,
    )
    if result.get("exit_code") != 0:
        raise RuntimeError("Guest browser launch failed: " + json.dumps(result))
    return {
        "page": page,
        "browser_launch": result,
        "scope": "Browser launch receipt; use sandbox_gui to observe/interact. A successful launch is not proof of external task completion.",
    }


def digest(request: dict) -> str:
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(root: Path, url: str, fields_json: str, purpose: str) -> dict:
    from rlm.v100.research_tools import public_origin

    public_origin(url)
    fields = json.loads(fields_json)
    if (
        not isinstance(fields, dict)
        or not fields
        or len(fields) > 64
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in fields.items())
    ):
        raise ValueError("Fields must be a nonempty JSON object of string names and values")
    if any(
        re.search(r"password|passwd|secret|token|authorization|credit.?card", k, re.I)
        for k in fields
    ):
        raise ValueError(
            "Credentials/payment secrets need operator browser handoff, not stored drafts"
        )
    if len(urlencode(fields).encode()) > 8192 or not 1 <= len(purpose) <= 2000:
        raise ValueError("Submission exceeds the bounded form/purpose budget")
    identity = uuid.uuid4().hex
    request = {
        "url": url,
        "method": "POST",
        "fields": fields,
        "purpose": purpose,
        "created": time.time(),
    }
    record = {
        "id": identity,
        "state": "awaiting operator approval",
        "request": request,
        "request_sha256": digest(request),
        "sent": False,
    }
    folder = root / "research/web-operations"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / (identity + ".json")
    atomic_json(path, record)
    path.chmod(0o600)
    return {
        **record,
        "approve_command": "/approve-web " + identity,
        "scope": "Local draft only. Review the exact URL and fields before approval. No network POST, account creation, email, purchase or order executed.",
    }


def post(request: dict) -> dict:
    from rlm.v100.research_tools import public_origin

    origin, address = public_origin(request["url"])
    pool = urllib3.HTTPSConnectionPool(
        address,
        port=443,
        assert_hostname=origin.hostname,
        server_hostname=origin.hostname,
        cert_reqs="CERT_REQUIRED",
        ca_certs=ssl.get_default_verify_paths().cafile,
        timeout=urllib3.Timeout(connect=10, read=20),
    )
    response = None
    try:
        response = pool.request(
            "POST",
            (origin.path or "/") + ("?" + origin.query if origin.query else ""),
            body=urlencode(request["fields"]).encode(),
            headers={
                "Host": origin.hostname,
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Synta operator-approved form",
            },
            redirect=False,
            retries=False,
            preload_content=False,
        )
        body = response.read(262145)
        return {
            "http_status": response.status,
            "response_sha256": hashlib.sha256(body).hexdigest(),
            "response_excerpt": body[:2000].decode("utf-8", errors="replace"),
            "response_truncated": len(body) > 262144,
            "location": response.headers.get("Location"),
            "scope": "HTTP transport receipt only. Redirects were not followed. A 2xx response does not verify account creation, delivery, income or task success.",
        }
    finally:
        if response is not None:
            response.close()
        pool.close()


def approve(root: Path, identity: str) -> dict:
    """Called only by an explicit authenticated operator command, never a model tool."""
    if not re.fullmatch(r"[0-9a-f]{32}", identity):
        raise ValueError("Invalid web operation ID")
    path = root / "research/web-operations" / (identity + ".json")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = json.loads(path.read_text())
        if record["request_sha256"] != digest(record["request"]):
            raise ValueError("Draft changed; prepare and review a new operation")
        if record["state"] != "awaiting operator approval":
            return {
                **record,
                "scope": "Already attempted. Never automatically resubmit an uncertain external operation.",
            }
        if time.time() - record["request"]["created"] > 3600:
            raise ValueError("Draft expired; prepare a fresh operation for review")
        record.update(state="submitting; outcome unknown until receipt", approved_at=time.time())
        atomic_json(path, record)
        try:
            result = post(record["request"])
        except Exception as error:
            record.update(
                state="transport failed; external outcome unknown", error=str(error)[:500]
            )
            atomic_json(path, record)
            raise
        record.update(state="HTTP response recorded", sent=True, result=result)
        atomic_json(path, record)
        return record
