"""Opt-in private GitHub telemetry and source mirror. Never a remote shell."""

import argparse
import base64
import io
import json
import os
import re
import runpy
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import quote

REPOSITORY = "Mantiee/synta-control"
SOURCE_REPOSITORY = "Mantiee/rlm"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("GitHub authentication redirect refused")


class GitHub:
    def __init__(self):
        reply = subprocess.run(
            ["gh", "auth", "token", "--hostname", "github.com"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if reply.returncode or not reply.stdout.strip():
            raise RuntimeError("GitHub CLI login required on the controller")
        self.token = reply.stdout.strip()
        self.opener = urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({}))

    def request(self, path: str, method: str = "GET", data: dict | None = None):
        if not any(
            path == "/repos/" + repo or path.startswith("/repos/" + repo + "/")
            for repo in (REPOSITORY, SOURCE_REPOSITORY)
        ):
            raise ValueError("GitHub request outside the owned repositories")
        request = urllib.request.Request(
            "https://api.github.com" + path,
            method=method,
            data=json.dumps(data).encode() if data is not None else None,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "SyntaPrivateControl/1.0",
            },
        )
        try:
            with self.opener.open(request, timeout=30) as reply:
                body = reply.read(4 * 2**20 + 1)
        except urllib.error.HTTPError as error:
            if error.code == 404 and method == "GET":
                return None
            raise RuntimeError(
                "GitHub API rejected private sync: HTTP " + str(error.code)
            ) from None
        if len(body) > 4 * 2**20:
            raise ValueError("GitHub response too large")
        return json.loads(body)

    def private(self):
        value = self.request("/repos/" + REPOSITORY)
        if not value or value.get("private") is not True or value.get("full_name") != REPOSITORY:
            raise ValueError("Private repository not available; no telemetry published")
        return value["default_branch"]

    def content(self, path: str, branch: str):
        return self.request(
            "/repos/" + REPOSITORY + "/contents/" + path + "?ref=" + quote(branch, safe="")
        )

    def write(self, path: str, value: dict, branch: str):
        previous = self.content(path, branch)
        encoded = json.dumps(value, indent=2, ensure_ascii=False).encode()
        if previous and base64.b64decode(previous["content"]) == encoded:
            return
        payload = {
            "message": "Refresh Synta controller receipts",
            "content": base64.b64encode(encoded).decode(),
            "branch": branch,
        }
        if previous:
            payload["sha"] = previous["sha"]
        self.private()
        self.request("/repos/" + REPOSITORY + "/contents/" + path, "PUT", payload)


def clean(value):
    """Bounded receipts, with token-shaped values and sensitive keys removed."""
    if isinstance(value, dict):
        return {
            k: clean(v)
            for k, v in value.items()
            if not re.search(
                r"(?:^|_)(password|secret|authorization|credential|token|cookie)(?:$|_)", k, re.I
            )
        }
    if isinstance(value, list):
        return [clean(v) for v in value[:64]]
    if isinstance(value, str):
        text = re.sub(r"(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", "[redacted]", value)
        text = re.sub(r"(?i)(?:bearer|token)\s+[A-Za-z0-9._-]+", "[redacted]", text)
        return text[:1000]
    return value


def source_mirror(root: Path, revision: str):
    if not re.fullmatch("[a-f0-9]{40}", revision):
        raise ValueError("Invalid source revision")
    folder = root / "research/private-sync/source.git"
    folder.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "credential.https://github.com.helper",
        "GIT_CONFIG_VALUE_0": "!gh auth git-credential",
        "GIT_CONFIG_KEY_1": "core.hooksPath",
        "GIT_CONFIG_VALUE_1": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    commands = []
    if not folder.exists():
        commands.append(["git", "init", "--bare", str(folder)])
    commands += [
        [
            "git",
            "-C",
            str(folder),
            "fetch",
            "https://github.com/" + SOURCE_REPOSITORY + ".git",
            revision,
        ],
        [
            "git",
            "-C",
            str(folder),
            "push",
            "https://github.com/" + REPOSITORY + ".git",
            revision + ":refs/heads/source",
        ],
    ]
    for command in commands:
        reply = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=180
        )
        if reply.returncode:
            raise RuntimeError(
                "Source mirror failed; private source changes are never force-overwritten"
            )


def audit(root: Path) -> dict:
    script = root / "tools/private-sync/audit-synta-hosts.py"
    if not script.is_file():
        raise ValueError("Owned audit script not installed")
    with redirect_stdout(io.StringIO()):
        runpy.run_path(str(script), run_name="__main__")
    path = root / "research/host-audit.json"
    if path.is_symlink() or path.stat().st_size > 8 * 2**20:
        raise ValueError("Host audit is linked or oversized")
    return json.loads(path.read_text())


def run(root: Path) -> dict:
    if __package__:
        from rlm.v100.release_selector import validate
    else:
        from release_selector import validate
    github = GitHub()
    branch = github.private()  # Check before reading host data or mirroring code.
    control = github.content("control/approved-release.json", branch)
    choice = (
        validate(json.loads(base64.b64decode(control["content"])))
        if control
        else {
            "schema": "synta-approved-release-v1",
            "repository": SOURCE_REPOSITORY,
            "mode": "follow-published",
        }
    )
    if not control:
        github.write("control/approved-release.json", choice, branch)
    revision = (
        choice.get("revision")
        or github.request("/repos/" + SOURCE_REPOSITORY + "/git/ref/heads/v100-rtx-reconnect")[
            "object"
        ]["sha"]
    )
    status_path = root / "research/private-sync/status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    source_state = "current"
    if previous.get("source_revision") != revision:
        try:
            source_mirror(root, revision)
            source_state = "mirrored"
        except (OSError, RuntimeError, subprocess.SubprocessError):
            source_state = "blocked; existing private code retained"
    value = audit(root)
    payload = clean(
        {
            k: value.get(k)
            for k in (
                "collected_at",
                "installed_package",
                "mission",
                "receipts",
                "qualification_details",
                "hosts",
                "services",
            )
        }
    )
    github.write("hosts/latest.json", payload, branch)
    target = root / "research/private-sync/approved-release.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".new")
    temporary.write_text(json.dumps(choice))
    temporary.replace(target)
    result = {
        "stage": "synced",
        "repository": REPOSITORY,
        "source_revision": revision
        if source_state in ("current", "mirrored")
        else previous.get("source_revision"),
        "desired_revision": revision,
        "source_state": source_state,
        "updated": time.time(),
        "scope": "Private source copy and host receipts; approved public code release selection, not remote shell or proof of learning",
    }
    status_path.write_text(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    try:
        result = run(root)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result = {
            "stage": "blocked",
            "error": type(error).__name__,
            "detail": clean(str(error)),
            "updated": time.time(),
            "scope": "Mission continues independently; inspect private sync configuration",
        }
        path = root / "research/private-sync/status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
