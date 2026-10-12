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


def test_resume_preserves_prefix_and_requires_exact_range(tmp_path, monkeypatch):
    data = b"prefix-remainder"
    name = "enwiki-2026-09-01-p1p20.xml.bz2"
    partial = (tmp_path / name).with_suffix(".partial")
    partial.write_bytes(b"prefix-")
    requested = []

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 7-{len(data) - 1}/{len(data)}"}

    def reply(*args, **kwargs):
        requested.append(kwargs["headers"])
        return Response(data[7:])

    monkeypatch.setattr(exports, "open_export", reply)
    result = exports.acquire(
        tmp_path, "https://dumps.wikimedia.org/test", name, hashlib.sha256(data).hexdigest()
    )
    assert result.read_bytes() == data
    assert requested == [{"Range": "bytes=7-"}]


def test_ignored_range_never_truncates_existing_prefix(tmp_path, monkeypatch):
    name = "enwiki-2026-09-01-p1p20.xml.bz2"
    partial = (tmp_path / name).with_suffix(".partial")
    partial.write_bytes(b"keep-this-prefix")

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "20"}

    monkeypatch.setattr(exports, "open_export", lambda *args, **kwargs: Response(b"new"))
    with pytest.raises(ValueError, match="prefix preserved"):
        exports.acquire(tmp_path, "https://dumps.wikimedia.org/test", name, "a" * 64)
    assert partial.read_bytes() == b"keep-this-prefix"


def test_downloading_logs_measured_progress(tmp_path, monkeypatch, capsys):
    data = b"part"
    clock = iter([0, 6])

    class Response(io.BytesIO):
        headers = {"Content-Length": "4"}

    monkeypatch.setattr(exports, "open_export", lambda *args, **kwargs: Response(data))
    monkeypatch.setattr(exports.time, "monotonic", lambda: next(clock))
    exports.acquire(
        tmp_path,
        "https://dumps.wikimedia.org/test",
        "part.xml.bz2",
        hashlib.sha256(data).hexdigest(),
    )
    assert "download-progress" in capsys.readouterr().out


def test_prefetch_uses_two_transfers_and_serial_indexing(tmp_path, monkeypatch):
    import threading

    rows = [("first.xml.bz2", "a" * 64), ("second.xml.bz2", "b" * 64)]
    monkeypatch.setattr(
        exports,
        "select_export",
        lambda language: ("2026-10-01", "https://dumps.wikimedia.org/test", "manifest", rows),
    )
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    maximum = 0

    def acquire(folder, origin, name, digest, cancel):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=3)
        with lock:
            active -= 1
        return folder / name

    indexed = []
    monkeypatch.setattr(exports, "acquire", acquire)
    monkeypatch.setattr(
        exports,
        "import_wiki",
        lambda root, path, language, budget: indexed.append(path.name) or {"pages_imported": 1},
    )
    receipt = exports.run(tmp_path, "en", 2)
    assert maximum == 2
    assert indexed == ["first.xml.bz2", "second.xml.bz2"]
    assert receipt["parts_indexed"] == 2 and receipt["stage"] == "complete"
