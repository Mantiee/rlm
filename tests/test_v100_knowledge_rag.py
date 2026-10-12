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
