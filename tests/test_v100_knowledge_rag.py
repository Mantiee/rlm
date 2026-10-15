import bz2
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

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


def test_task_receipt_binds_source_revision_and_attached_excerpt(tmp_path, monkeypatch):
    from rlm.v100 import goals
    from rlm.v100.knowledge_rag import attach

    dump = tmp_path / "wiki.xml.bz2"
    text = "Revenue accounting and business."
    dump.write_bytes(
        bz2.compress(
            b"<mediawiki><page><title>Revenue</title><ns>0</ns><revision><id>9</id><text>"
            + text.encode()
            + b"</text></revision></page></mediawiki>"
        )
    )
    import_wiki(tmp_path, dump, "en", 10)
    monkeypatch.setattr(goals, "load_goal", lambda root: {"id": "goal-1", "text": "revenue"})
    attach(
        tmp_path,
        [{"role": "user", "content": "Find revenue accounting"}],
        task_id="task-1",
        run_id="run-1",
    )
    receipt = json.loads(
        (tmp_path / "research/knowledge/tasks/task-1.json").read_text(encoding="utf-8")
    )
    passage = receipt["passages"][0]
    assert receipt["schema"] == "v100-knowledge-task-v1"
    assert (receipt["task_id"], receipt["run_id"], receipt["goal_id"]) == (
        "task-1",
        "run-1",
        "goal-1",
    )
    assert receipt["retrieval_state"] == "retrieved" and receipt["context_prepared"]
    assert receipt["query"] == "Find revenue accounting revenue"
    assert passage["url"] == "https://en.wikipedia.org/wiki/Revenue"
    assert passage["revision"] == "9"
    assert passage["content_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert passage["excerpt_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert "does not prove" in receipt["scope"]
    latest = json.loads((tmp_path / "research/knowledge/goal-use.json").read_text(encoding="utf-8"))
    from pathlib import Path

    assert Path(latest["task_receipt"]).name == "task-1.json"


def test_task_receipts_distinguish_no_results_busy_and_timeout(tmp_path, monkeypatch):
    from rlm.v100 import goals, knowledge_rag

    database = tmp_path / "research/knowledge/passages.sqlite3"
    database.parent.mkdir(parents=True)
    database.touch()
    monkeypatch.setattr(goals, "load_goal", lambda root: None)
    cases = (
        ("empty", "retrieved", [], "no-results"),
        ("busy", "busy", [], "busy"),
        ("timeout", "deferred", [], "deferred"),
    )
    for task_id, result_state, passages, expected_state in cases:
        monkeypatch.setattr(
            knowledge_rag,
            "search",
            lambda *args, state=result_state, rows=passages: {
                "passages": rows,
                "seconds": 0.1,
                "state": state,
                "scope": "test state",
            },
        )
        attached = knowledge_rag.attach(
            tmp_path, [{"role": "user", "content": "Find a useful source"}], task_id=task_id
        )
        receipt = json.loads(
            (tmp_path / f"research/knowledge/tasks/{task_id}.json").read_text(encoding="utf-8")
        )
        assert receipt["retrieval_state"] == expected_state
        assert receipt["context_prepared"] is False
        assert receipt["passages"] == []
        assert attached == [{"role": "user", "content": "Find a useful source"}]
        latest = json.loads(
            (tmp_path / "research/knowledge/goal-use.json").read_text(encoding="utf-8")
        )
        assert latest["state"] == expected_state
    original_receipt = (tmp_path / "research/knowledge/tasks/empty.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        knowledge_rag.attach(
            tmp_path,
            [{"role": "user", "content": "changed query"}],
            task_id="empty",
        )
    assert (tmp_path / "research/knowledge/tasks/empty.json").read_bytes() == original_receipt


def test_broken_goal_snapshot_does_not_stop_archive_research(tmp_path, monkeypatch):
    from rlm.v100 import goals, knowledge_rag

    database = tmp_path / "research/knowledge/passages.sqlite3"
    database.parent.mkdir(parents=True)
    database.touch()

    def broken_goal(root):
        raise ValueError("invalid goal snapshot")

    monkeypatch.setattr(goals, "load_goal", broken_goal)
    monkeypatch.setattr(
        knowledge_rag,
        "search",
        lambda root, query, limit: {
            "passages": [],
            "seconds": 0.01,
            "state": "retrieved",
            "scope": "test retrieval",
        },
    )
    messages = [{"role": "user", "content": "Find a useful source"}]
    assert knowledge_rag.attach(tmp_path, messages, task_id="broken-goal") == messages
    receipt = json.loads(
        (tmp_path / "research/knowledge/tasks/broken-goal.json").read_text(encoding="utf-8")
    )
    assert receipt["goal_state"] == "unavailable"
    assert receipt["goal_error"] == "invalid goal snapshot"
    assert receipt["retrieval_state"] == "no-results"


def test_receipt_write_oserror_omits_passages_and_continues_without_context(tmp_path, monkeypatch):
    from rlm.v100 import goals, knowledge_rag

    database = tmp_path / "research/knowledge/passages.sqlite3"
    database.parent.mkdir(parents=True)
    database.touch()
    monkeypatch.setattr(goals, "load_goal", lambda root: {"id": "goal", "text": "revenue"})
    monkeypatch.setattr(
        knowledge_rag,
        "search",
        lambda root, query, limit: {
            "passages": [
                {
                    "id": "revenue",
                    "title": "Revenue",
                    "text": "Revenue accounting passage",
                    "source": "https://en.wikipedia.org/wiki/Revenue",
                    "revision": "9",
                }
            ],
            "seconds": 0.01,
            "state": "retrieved",
            "scope": "test retrieval",
        },
    )

    def fail_receipt(path, value):
        raise OSError("read-only receipt directory")

    monkeypatch.setattr(knowledge_rag, "save_task_receipt", fail_receipt)
    messages = [{"role": "user", "content": "Find revenue accounting"}]
    attached = knowledge_rag.attach(tmp_path, messages, task_id="receipt-oserror")

    assert attached == messages
    assert not (tmp_path / "research/knowledge/tasks/receipt-oserror.json").exists()
    latest = json.loads((tmp_path / "research/knowledge/goal-use.json").read_text())
    assert latest["state"] == "unavailable"
    assert latest["sources"] == []
    assert latest["task_receipt"] is None
    assert "receipt persistence failed" in latest["scope"]


def test_task_receipt_is_published_only_after_complete_temp_write(tmp_path, monkeypatch):
    from rlm.v100 import knowledge_rag

    real_link = knowledge_rag.os.link

    def inspect_then_link(source, destination):
        assert not destination.exists()
        assert json.loads(source.read_text(encoding="utf-8")) == {"complete": True}
        return real_link(source, destination)

    monkeypatch.setattr(knowledge_rag.os, "link", inspect_then_link)
    target = tmp_path / "tasks/task.json"
    knowledge_rag.save_task_receipt(target, {"complete": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}
    assert list(target.parent.iterdir()) == [target]


def test_parallel_tasks_keep_separate_immutable_receipt_paths(tmp_path, monkeypatch):
    from rlm.v100 import goals, knowledge_rag

    database = tmp_path / "research/knowledge/passages.sqlite3"
    database.parent.mkdir(parents=True)
    database.touch()
    monkeypatch.setattr(goals, "load_goal", lambda root: {"id": "goal", "text": ""})
    overlap = Barrier(2)

    def search(root, query, limit):
        overlap.wait(timeout=2)
        marker = "one" if "task one" in query else "two"
        return {
            "passages": [
                {
                    "id": marker,
                    "title": marker,
                    "text": marker + " source passage",
                    "source": f"https://en.wikipedia.org/wiki/{marker}",
                    "revision": marker + "-revision",
                }
            ],
            "seconds": 0.01,
            "state": "retrieved",
            "scope": "test retrieval",
        }

    monkeypatch.setattr(knowledge_rag, "search", search)
    jobs = (
        ("task-one", "run-one", "task one"),
        ("task-two", "run-two", "task two"),
    )
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(
                knowledge_rag.attach,
                tmp_path,
                [{"role": "user", "content": task}],
                task_id=task_id,
                run_id=run_id,
            )
            for task_id, run_id, task in jobs
        ]
        for future in futures:
            future.result(timeout=5)
    receipts = [
        json.loads(
            (tmp_path / f"research/knowledge/tasks/{task_id}.json").read_text(encoding="utf-8")
        )
        for task_id, _, _ in jobs
    ]
    assert [row["task_id"] for row in receipts] == ["task-one", "task-two"]
    assert [row["run_id"] for row in receipts] == ["run-one", "run-two"]
    assert [row["passages"][0]["id"] for row in receipts] == ["one", "two"]


def test_bad_index_does_not_abort_operator_task(tmp_path, monkeypatch):
    from rlm.v100 import goals
    from rlm.v100.knowledge_rag import attach

    path = tmp_path / "research/knowledge/passages.sqlite3"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupted sqlite")
    monkeypatch.setattr(goals, "load_goal", lambda root: None)
    result = attach(tmp_path, [{"role": "user", "content": "business"}])
    assert "unavailable" in result[-1]["content"]


def test_repeated_retrieval_replaces_archive_and_preserves_operator_task(tmp_path, monkeypatch):
    from rlm.v100 import goals, knowledge_rag

    path = tmp_path / "research/knowledge/passages.sqlite3"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(goals, "load_goal", lambda root: {"text": "income"})
    monkeypatch.setattr(
        knowledge_rag,
        "search",
        lambda *args: {
            "passages": [
                {
                    "text": "x" * 100000,
                    "title": "y" * 10000,
                    "source": "https://en.wikipedia.org/wiki/Revenue",
                    "revision": "9",
                }
            ]
            * 8,
            "seconds": 0.1,
            "state": "retrieved",
            "scope": "untrusted archive",
        },
    )
    original = [
        {"role": "user", "content": "Compare legal income experiments"},
        {"role": "assistant", "content": "Keep this previous conclusion"},
    ]
    result = original
    for _ in range(10):
        result = knowledge_rag.attach(tmp_path, result)
    assert result[:-1] == original
    supplement = json.loads(result[-1]["content"])
    assert len(supplement["supplemental_archive_data"]["passages"]) == 2
    assert sum(len(p["text"]) for p in supplement["supplemental_archive_data"]["passages"]) == 1600
    receipt = json.loads((path.parent / "goal-use.json").read_text())
    assert receipt["excerpt_characters"] == 1600
    assert len(result[-1]["content"]) < 4500


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


def test_complete_archive_checkpoint_skips_xml_but_partial_checkpoint_resumes(
    tmp_path, monkeypatch
):
    from rlm.v100 import knowledge_rag

    dump = tmp_path / "wiki.xml.bz2"
    dump.write_bytes(
        bz2.compress(
            b"<mediawiki><page><title>Revenue</title><ns>0</ns><revision><id>9</id><text>Revenue costs.</text></revision></page></mediawiki>"
        )
    )
    first = import_wiki(tmp_path, dump, "en", 1)
    assert first["pages_written"] == 1
    assert "state" not in import_wiki(tmp_path, dump, "en", 10)
    monkeypatch.setattr(
        knowledge_rag.ET,
        "iterparse",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("complete archive should not be rescanned")
        ),
    )
    result = import_wiki(tmp_path, dump, "en", 10)
    assert result["state"] == "verified-complete-checkpoint"
    assert result["pages_reused"] == 1 and result["pages_written"] == 0
    dump.write_bytes(bz2.compress(b"<mediawiki/>"))
    with __import__("pytest").raises(AssertionError, match="should not be rescanned"):
        import_wiki(tmp_path, dump, "en", 10)


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


def test_structured_task_queries_ignore_json_keys_and_rank_topic_titles(tmp_path, monkeypatch):
    from rlm.v100 import goals, knowledge_rag

    dump = tmp_path / "wiki.xml.bz2"
    dump.write_bytes(
        bz2.compress(
            b"<mediawiki><page><title>Business revenue</title><ns>0</ns><revision><id>9</id><text>Revenue accounting costs.</text></revision></page><page><title>Branch</title><ns>0</ns><revision><id>10</id><text>branch user_goal json message text research source.</text></revision></page></mediawiki>"
        )
    )
    import_wiki(tmp_path, dump, "en", 10)
    monkeypatch.setattr(goals, "load_goal", lambda root: {"text": "business revenue"})
    message = {
        "role": "user",
        "content": json.dumps(
            {
                "branch": "A",
                "user_goal": {"text": "business revenue accounting"},
                "task": "Compare business revenue costs",
            }
        ),
    }
    result = knowledge_rag.attach(tmp_path, [message])
    assert result[0] == message
    passages = json.loads(result[-1]["content"])["supplemental_archive_data"]["passages"]
    assert [p["title"] for p in passages] == ["Business revenue"]
    receipt = json.loads((tmp_path / "research/knowledge/goal-use.json").read_text())
    assert "user_goal" not in receipt["query"] and "branch" not in receipt["query"]
    assert receipt["state"] == "retrieved"


def test_long_goal_prefers_matching_article_title_over_unrelated_body_words(tmp_path):
    from contextlib import closing

    from rlm.v100.knowledge_rag import connect

    with closing(connect(tmp_path)) as database:
        database.executemany(
            "INSERT INTO passages VALUES (?,?,?,?,?)",
            [
                (
                    "other",
                    "Unrelated biography",
                    "net income deposits paid apis sports",
                    "fixture:other",
                    "1",
                ),
                ("income", "Net income", "Revenue minus expenses.", "fixture:income", "1"),
            ],
        )
        database.commit()
    result = search(tmp_path, "net income deposits paid apis sports", 2)
    assert result["state"] == "retrieved"
    assert [row["id"] for row in result["passages"]] == ["income"]
    # A body-only query still retrieves content if no article title matches.
    assert search(tmp_path, "revenue expenses")["passages"][0]["id"] == "income"
