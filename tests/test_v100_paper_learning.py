import copy
import json
import threading

import pytest

from rlm.v100 import paper_feeds, paper_learning
from rlm.v100.common import atomic_json
from rlm.v100.paper import PaperBook


def settings():
    return {
        "schema": "v100-paper-learning-v1",
        "objective": "Maximize fast lawful repeatable net income, without deposits",
        "crypto": False,
        "ciks": [],
        "sec_contact": "",
        "observer_interval": 30,
        "research_rounds": 1,
        "other_income_rnd": True,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"objective": ""},
        {"crypto": 1},
        {"observer_interval": 1},
        {"research_rounds": 7},
        {"ciks": ["not-cik"]},
        {"ciks": ["1", "1"]},
        {"ciks": ["1"], "sec_contact": ""},
        {"paid_api": True},
    ],
)
def test_learning_settings_reject_invalid_or_unbounded_inputs(tmp_path, change):
    path = tmp_path / "settings.json"
    atomic_json(path, {**settings(), **change})
    with pytest.raises(ValueError):
        paper_learning.load_settings(path)


def test_bridge_preserves_bankroll_uses_one_lease_and_detects_goal_changes(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    atomic_json(path, settings())
    with pytest.raises(ValueError, match="paper-init"):
        paper_learning.PaperLearning(tmp_path, path)
    book = PaperBook(tmp_path)
    initial = book.initialize()
    book.close()
    observed = threading.Event()
    monkeypatch.setattr(paper_learning.PaperLearning, "tick", lambda self: observed.set())
    with paper_learning.PaperLearning(tmp_path, path) as session:
        assert observed.wait(5)
        with pytest.raises(BlockingIOError):
            with paper_learning.PaperLearning(tmp_path, path):
                pytest.fail("Duplicate owner allowed")
        original = copy.deepcopy(settings())
        atomic_json(path, {**original, "objective": "Changed goal"})
        with pytest.raises(ValueError, match="changed"):
            session.check()
        atomic_json(path, original)
    assert session.worker is not None and not session.worker.is_alive()
    book = PaperBook(tmp_path)
    assert book.state()["branches"] == initial["branches"]
    book.close()


def test_crypto_poll_requires_registered_sources_and_fees(tmp_path):
    book = PaperBook(tmp_path)
    book.initialize()
    book.close()
    path = tmp_path / "settings.json"
    atomic_json(path, {**settings(), "crypto": True})
    with pytest.raises(ValueError, match="Register"):
        paper_learning.PaperLearning(tmp_path, path)


def test_observer_failure_cannot_silently_continue_learning(tmp_path, monkeypatch):
    book = PaperBook(tmp_path)
    book.initialize()
    book.close()
    path = tmp_path / "settings.json"
    atomic_json(path, settings())
    failure = threading.Event()

    def fail(session):
        failure.set()
        raise ValueError("Synthetic invalid source")

    monkeypatch.setattr(paper_learning.PaperLearning, "tick", fail)
    with pytest.raises(RuntimeError, match="observer failed"):
        with paper_learning.PaperLearning(tmp_path, path) as session:
            assert failure.wait(5)
            session.worker.join(timeout=5)
            session.check()


def test_cancelled_filing_poll_never_imports_a_late_response(tmp_path, monkeypatch):
    book = PaperBook(tmp_path)
    book.initialize()
    before = book.sequence()
    monkeypatch.setattr(paper_feeds, "public_json", lambda *a: ({}, "a" * 64, "unused"))
    assert paper_feeds.poll_filings(book, "1", "test@example.org", lambda: True) == []
    assert book.sequence() == before
    book.close()


def test_settings_are_copied_to_an_immutable_run_snapshot(tmp_path, monkeypatch):
    book = PaperBook(tmp_path)
    book.initialize()
    book.close()
    path = tmp_path / "settings.json"
    atomic_json(path, settings())
    monkeypatch.setattr(paper_learning.PaperLearning, "tick", lambda self: None)
    with paper_learning.PaperLearning(tmp_path, path) as session:
        digest = session.settings_sha
    snapshot = tmp_path / "research/paper/learning-settings" / (digest + ".json")
    assert json.loads(snapshot.read_text()) == settings()
    atomic_json(snapshot, {**settings(), "crypto": True})
    with pytest.raises(ValueError, match="snapshot changed"):
        with paper_learning.PaperLearning(tmp_path, path):
            pytest.fail("Changed archived configuration was accepted")


def test_readiness_lists_dataset_paths_without_reading_or_resetting_them(tmp_path):
    book = PaperBook(tmp_path)
    original = book.initialize()
    sequence = book.sequence()
    book.close()
    directory = tmp_path / "datasets"
    directory.mkdir()
    dataset = directory / "training.gz"
    dataset.write_bytes(b"Not decoded or used as a valid training file")
    value = paper_learning.readiness(
        tmp_path, {"training": {"base_model": str(tmp_path / "model")}}, directory
    )
    assert value["paper"]["initialized"]
    assert value["available_files"] == [{"path": str(dataset), "bytes": dataset.stat().st_size}]
    assert not value["cpu_helper"]["profile_exists"]
    book = PaperBook(tmp_path)
    assert book.state() == original
    assert book.sequence() == sequence
    book.close()


def test_dead_observer_without_a_captured_exception_still_blocks_learning(tmp_path):
    book = PaperBook(tmp_path)
    book.initialize()
    book.close()
    path = tmp_path / "settings.json"
    atomic_json(path, settings())
    session = paper_learning.PaperLearning(tmp_path, path)
    session.worker = threading.Thread(target=lambda: None)
    session.worker.start()
    session.worker.join(timeout=5)
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        session.check()
