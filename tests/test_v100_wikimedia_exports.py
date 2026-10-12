import hashlib
import io
import urllib.error
from datetime import UTC, datetime

import pytest

from rlm.v100 import wikimedia_exports as exports


def test_identifiable_user_agent_and_no_retry_on_forbidden(monkeypatch):
    requests = []

    def reject(request, **kwargs):
        requests.append(request)
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(b"policy"))

    monkeypatch.setattr(exports.urllib.request, "urlopen", reject)
    with pytest.raises(RuntimeError, match="identifiable User-Agent sent"):
        exports.open_export("https://dumps.wikimedia.org/test")
    assert len(requests) == 1
    assert requests[0].get_header("User-agent") == exports.USER_AGENT


def test_missing_current_manifest_uses_previous_completed_month(monkeypatch):
    urls = []
    manifest = "a" * 64 + "  enwiki-2026-09-01-p1p20.xml.bz2\n"

    def reply(url, **kwargs):
        urls.append(url)
        if "/2026-10-01/" in url:
            raise urllib.error.HTTPError(url, 404, "Not ready", {}, None)
        return io.BytesIO(manifest.encode())

    monkeypatch.setattr(exports, "open_export", reply)
    date, _, _, rows = exports.select_export("en", datetime(2026, 10, 12, tzinfo=UTC))
    assert date == "2026-09-01"
    assert len(rows) == 1 and len(urls) == 2


@pytest.mark.parametrize(
    "name", ["../escape.xml.bz2", "/escape.xml.bz2", "otherwiki-2026-09-01-p1p2.xml.bz2"]
)
def test_manifest_cannot_escape_dataset(name):
    with pytest.raises(ValueError):
        exports.manifest_rows("a" * 64 + "  " + name, "enwiki", "2026-09-01")


def test_current_part_is_checksum_verified_before_admission(tmp_path, monkeypatch):
    data = b"export-part"

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(data))}

    monkeypatch.setattr(exports, "open_export", lambda *args, **kwargs: Response(data))
    name = "enwiki-2026-09-01-p1p20.xml.bz2"
    with pytest.raises(ValueError, match="incorrect Wikipedia part"):
        exports.acquire(tmp_path, "https://dumps.wikimedia.org/test", name, "a" * 64)
    assert not (tmp_path / name).exists()
    path = exports.acquire(
        tmp_path, "https://dumps.wikimedia.org/test", name, hashlib.sha256(data).hexdigest()
    )
    assert path.read_bytes() == data
