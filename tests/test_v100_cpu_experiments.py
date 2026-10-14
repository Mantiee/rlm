"""Regression tests for CPU experiments independent from remote inference."""

import json

import pytest

from rlm.v100.cpu_experiments import run_integrity_experiment


def _write(pool, rows):
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_cpu_receipt_is_independent_and_persistent(tmp_path):
    pool = tmp_path / "pool.jsonl"
    output = tmp_path / "receipts"
    _write(pool, [{"id": "a"}, {"id": "b"}])
    report = run_integrity_experiment(pool, output, 1)
    assert report["status"] == "ACCEPTED"
    assert report["evaluation"]["independent_recalculation_matches"]
    assert report["measurement"]["records"] == 2
    assert report["transitions"][-1]["state"] == "ACCEPTED"
    assert run_integrity_experiment(pool, output, 1) == report
    assert (output / "cpu-pool-integrity-0001.json").exists()


def test_duplicate_records_are_rejected(tmp_path):
    pool = tmp_path / "pool.jsonl"
    _write(pool, [{"a": 1, "b": 2}, {"b": 2, "a": 1}])
    report = run_integrity_experiment(pool, tmp_path / "receipts", 2)
    assert report["status"] == "REJECTED"
    assert report["measurement"]["duplicate_records"] == 1
    assert report["evaluation"]["passed"] is False


def test_empty_pool_is_inconclusive(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text("")
    report = run_integrity_experiment(pool, tmp_path / "receipts", 1)
    assert report["status"] == "INCONCLUSIVE"


def test_invalid_pool_is_failed_not_success(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text("{invalid json}\n")
    report = run_integrity_experiment(pool, tmp_path / "receipts", 1)
    assert report["status"] == "FAILED"
    assert report["error"]["type"] == "JSONDecodeError"


def test_reused_receipt_rejects_changed_input(tmp_path):
    pool = tmp_path / "pool.jsonl"
    _write(pool, [{"a": 1}])
    run_integrity_experiment(pool, tmp_path / "receipts", 1)
    _write(pool, [{"a": 2}])
    with pytest.raises(ValueError, match="changed"):
        run_integrity_experiment(pool, tmp_path / "receipts", 1)
