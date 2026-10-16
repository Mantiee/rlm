"""One coherent, fully verified canonical ledger audit per aggregate report."""

import json
import sqlite3

import pytest

from rlm.v100 import progress
from rlm.v100.paper import PaperBook


def initialized(root):
    book = PaperBook(root)
    book.initialize("100")
    book.close()


def test_report_paper_snapshot_hashes_full_chain_once(tmp_path, monkeypatch):
    initialized(tmp_path)
    original = PaperBook.events
    calls = []

    def counted(book):
        calls.append(book)
        return original(book)

    monkeypatch.setattr(PaperBook, "events", counted)
    state, summary, count, ready = progress.paper_snapshot(tmp_path)
    assert ready and count == 0
    assert state["initial_capital"] == "100"
    assert summary["branches"]["A"]["cash"] == "100"
    assert len(calls) == 1
    progress.paper_snapshot(tmp_path)
    assert len(calls) == 2  # No cross-report cache hiding later corruption.


def test_snapshot_does_not_mix_a_concurrent_appended_event(tmp_path, monkeypatch):
    initialized(tmp_path)
    original = progress.summarize_snapshot

    def append_then_summarize(book, period):
        writer = PaperBook(tmp_path)
        try:
            writer.note("A", {"test": "concurrent append"})
        finally:
            writer.close()
        return original(book, period)

    monkeypatch.setattr(progress, "summarize_snapshot", append_then_summarize)
    _, summary, _, _ = progress.paper_snapshot(tmp_path)
    assert summary["ledger_sequence"] == 1
    with sqlite3.connect(tmp_path / "research/paper/ledger.sqlite3") as db:
        assert db.execute("SELECT MAX(sequence) FROM events").fetchone()[0] == 2


@pytest.mark.parametrize("tamper", ["chain", "state", "sequence"])
def test_snapshot_still_rejects_tampering(tmp_path, tamper):
    initialized(tmp_path)
    path = tmp_path / "research/paper/ledger.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER immutable_events_update")
        if tamper == "chain":
            db.execute("UPDATE events SET event_hash=?", ("a" * 64,))
        elif tamper == "sequence":
            db.execute("UPDATE events SET sequence=2")
        else:
            state = json.loads(db.execute("SELECT payload FROM state").fetchone()[0])
            state["initial_capital"] = "999"
            db.execute("UPDATE state SET payload=?", (json.dumps(state),))
    with pytest.raises(ValueError, match="chain changed|sequence|state changed"):
        progress.paper_snapshot(tmp_path)


def test_missing_ledger_does_not_create_a_database(tmp_path):
    _, summary, count, ready = progress.paper_snapshot(tmp_path)
    assert not ready and count == 0 and summary["currency"] == "unknown"
    assert not (tmp_path / "research/paper/ledger.sqlite3").exists()


def test_count_preserves_preceding_decision_and_pnl_checks():
    decision = {"kind": "decision", "sequence": 1, "payload": {"proposal": {"id": "x"}}}
    trade = {"action": "close", "order_id": "x", "payout": "9", "allocation": "10", "net_pnl": "-1"}
    outcome = {"kind": "observation", "sequence": 2, "payload": {"trades": [trade]}}
    assert progress.audited_outcome_count([decision, outcome]) == 1
    with pytest.raises(ValueError, match="preceding decision"):
        progress.audited_outcome_count([outcome])
    trade["net_pnl"] = "1"
    with pytest.raises(ValueError, match="payout minus allocation"):
        progress.audited_outcome_count([decision, outcome])


def test_complete_report_uses_one_full_ledger_audit(tmp_path, monkeypatch):
    initialized(tmp_path)
    original = PaperBook.events
    calls = []

    def counted(book):
        calls.append(book)
        return original(book)

    monkeypatch.setattr(PaperBook, "events", counted)
    result = progress.report(tmp_path)
    assert len(calls) == 1
    assert result["available_audited_paper_outcomes"] == 0
    assert result["paper"]["branches"]["A"]["cash"] == "100"
    assert json.loads((tmp_path / "research/mission/latest-report.json").read_text()) == result
