import hashlib
import io
import json

import pytest

from rlm.v100 import wiki_ingest


def test_official_export_checksum_and_cached_receipt(tmp_path, monkeypatch):
    data = b"export"
    status = {
        "jobs": {
            "articlesmultistreamdump": {
                "status": "done",
                "files": {
                    "plwiki-20261001-pages-articles-multistream.xml.bz2": {
                        "size": len(data),
                        "sha1": hashlib.sha1(data).hexdigest(),
                    }
                },
            }
        }
    }
    monkeypatch.setattr(
        wiki_ingest.urllib.request,
        "urlopen",
        lambda url, **kw: io.BytesIO(json.dumps(status).encode() if url.endswith("json") else data),
    )
    path = wiki_ingest.fetch(tmp_path, "pl")
    assert path.read_bytes() == data
    assert path.with_suffix(".receipt.json").exists()


def test_incomplete_export_not_downloaded(tmp_path, monkeypatch):
    status = {"jobs": {"articlesmultistreamdump": {"status": "waiting"}}}
    monkeypatch.setattr(
        wiki_ingest.urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(status).encode()),
    )
    with pytest.raises(RuntimeError, match="not complete"):
        wiki_ingest.fetch(tmp_path, "en")
