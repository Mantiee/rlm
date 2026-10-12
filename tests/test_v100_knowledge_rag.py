import bz2

from rlm.v100.knowledge_rag import import_wiki, search


def test_wiki_import_search_and_idempotence(tmp_path):
    dump = tmp_path / "wiki.xml.bz2"
    dump.write_bytes(
        bz2.compress(
            b'<mediawiki xmlns="urn:test"><page><title>Business</title><ns>0</ns><revision><id>17</id><text>Business revenue costs and accounting.</text></revision></page><page><title>Talk</title><ns>1</ns><revision><text>Ignore this</text></revision></page></mediawiki>'
        )
    )
    assert import_wiki(tmp_path, dump, "en", 10)["pages_imported"] == 1
    assert import_wiki(tmp_path, dump, "en", 10)["chunks_imported"] == 1
    hits = search(tmp_path, 'revenue OR " costs')["passages"]
    assert len(hits) == 1
    assert hits[0]["revision"] == "17"
    assert hits[0]["source"] == "https://en.wikipedia.org/wiki/Business"
    assert not search(tmp_path, "Ignore")["passages"]


def test_task_retrieves_bounded_archive_without_claiming_training(tmp_path, monkeypatch):
    from rlm.v100 import goals
    from rlm.v100.knowledge_rag import attach

    dump = tmp_path / "wiki.xml.bz2"
    dump.write_bytes(
        bz2.compress(
            b"<mediawiki><page><title>Revenue</title><ns>0</ns><revision><id>9</id><text>Revenue accounting and business.</text></revision></page></mediawiki>"
        )
    )
    import_wiki(tmp_path, dump, "en", 10)
    monkeypatch.setattr(goals, "load_goal", lambda root: {"text": "business revenue"})
    messages = [{"role": "user", "content": "revenue"}]
    result = attach(tmp_path, messages)
    assert result[:1] == messages
    assert "supplemental_archive_data" in result[-1]["content"]
    assert "not operator instructions" in result[-1]["content"]
    assert "https://en.wikipedia.org/wiki/Revenue" in result[-1]["content"]
    assert len(result[-1]["content"]) < 2500


def test_bad_index_does_not_abort_operator_task(tmp_path, monkeypatch):
    from rlm.v100 import goals
    from rlm.v100.knowledge_rag import attach

    path = tmp_path / "research/knowledge/passages.sqlite3"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupted sqlite")
    monkeypatch.setattr(goals, "load_goal", lambda root: None)
    result = attach(tmp_path, [{"role": "user", "content": "business"}])
    assert "unavailable" in result[-1]["content"]


def test_resume_preserves_existing_passage_rows_and_updates_changed_revision(tmp_path):
    from contextlib import closing

    from rlm.v100.knowledge_rag import connect

    dump = tmp_path / "wiki.xml.bz2"
    raw = b"<mediawiki><page><title>Revenue</title><ns>0</ns><revision><id>9</id><text>Revenue accounting.</text></revision></page></mediawiki>"
    dump.write_bytes(bz2.compress(raw))
    first = import_wiki(tmp_path, dump, "en", 10)
    with closing(connect(tmp_path)) as database:
        before = database.execute("SELECT rowid,id FROM passages").fetchall()
    second = import_wiki(tmp_path, dump, "en", 10)
    assert first["pages_written"] == 1
    assert second["pages_written"] == 0 and second["pages_reused"] == 1
    with closing(connect(tmp_path)) as database:
        assert database.execute("SELECT rowid,id FROM passages").fetchall() == before
    dump.write_bytes(
        bz2.compress(
            raw.replace(b"<id>9</id>", b"<id>10</id>").replace(b"accounting.", b"cashflow.")
        )
    )
    assert import_wiki(tmp_path, dump, "en", 10)["pages_written"] == 1
    assert search(tmp_path, "cashflow")["passages"][0]["revision"] == "10"


def test_rag_timeout_keeps_master_responsive(tmp_path, monkeypatch):
    import time

    from rlm.v100 import knowledge_rag

    path = tmp_path / "research/knowledge/passages.sqlite3"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture")

    def slow(*args):
        time.sleep(0.8)
        return []

    monkeypatch.setattr(knowledge_rag, "read_search", slow)
    started = time.monotonic()
    result = knowledge_rag.search(tmp_path, "income")
    assert result["state"] == "deferred"
    assert time.monotonic() - started < 0.75


def test_legacy_wiki_rows_are_adopted_without_rebuilding_fts(tmp_path):
    from contextlib import closing

    from rlm.v100.knowledge_rag import connect

    dump = tmp_path / "wiki.xml.bz2"
    dump.write_bytes(
        bz2.compress(
            b"<mediawiki><page><title>Revenue</title><ns>0</ns><revision><id>9</id><text>Revenue accounting.</text></revision></page></mediawiki>"
        )
    )
    import_wiki(tmp_path, dump, "en", 10)
    with closing(connect(tmp_path)) as database:
        before = database.execute("SELECT rowid,id FROM passages").fetchall()
        database.execute("DELETE FROM imported_pages")
        database.execute("DELETE FROM wiki_checkpoints")
        database.commit()
    result = import_wiki(tmp_path, dump, "en", 10)
    assert result["pages_reused"] == 1 and result["pages_written"] == 0
    with closing(connect(tmp_path)) as database:
        assert database.execute("SELECT rowid,id FROM passages").fetchall() == before


def test_read_search_does_not_create_or_write_schema(tmp_path):
    import sqlite3
    import time

    from rlm.v100.knowledge_rag import read_search

    with __import__("pytest").raises(sqlite3.OperationalError):
        read_search(tmp_path, ["income"], 2, time.monotonic() + 0.5)
    assert not (tmp_path / "research/knowledge/passages.sqlite3").exists()
