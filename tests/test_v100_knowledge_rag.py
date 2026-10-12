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
