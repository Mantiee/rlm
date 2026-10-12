import io
import zipfile

import pytest

from rlm.v100 import source_sync


def archive(entries):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as output:
        for path, value in entries.items():
            output.writestr(path, value)
    return data.getvalue()


def test_pinned_source_reuse_checks_bytes_and_never_runs_code(tmp_path, monkeypatch):
    code = "raise RuntimeError('must never execute')"
    data = archive({"snapshot/rlm/v100/mission_chat.py": code})
    calls = []

    def fetch(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(data)

    monkeypatch.setattr(source_sync.urllib.request, "urlopen", fetch)
    result = source_sync.sync(tmp_path, "a" * 40)
    assert source_sync.sync(tmp_path, "a" * 40) == result
    assert len(calls) == 1 and calls[0].endswith("a" * 40)
    path = tmp_path / "research/managed-source" / ("a" * 40) / "rlm/v100/mission_chat.py"
    path.write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        source_sync.sync(tmp_path, "a" * 40)


@pytest.mark.parametrize("path", ["snapshot/../escape.py", "/absolute.py"])
def test_source_archive_rejects_escape_before_committing(tmp_path, monkeypatch, path):
    data = archive({path: "unsafe"})
    monkeypatch.setattr(
        source_sync.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(data)
    )
    with pytest.raises(ValueError, match="escapes"):
        source_sync.sync(tmp_path, "a" * 40)
    assert not (tmp_path / "research/managed-source" / ("a" * 40)).exists()
