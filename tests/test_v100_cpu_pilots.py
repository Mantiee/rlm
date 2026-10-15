"""CPU dispatch is independent of GPU qualification and remote inference."""

import json

from rlm.v100 import cpu_pilots


def test_cpu_pilots_queue_once_and_recover_receipts(tmp_path, monkeypatch):
    config = tmp_path / "research/compute-config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    calls = []
    monkeypatch.setattr(cpu_pilots, "inspect", lambda root: {"workers": [], "jobs": []})

    def queue(root, branch, architecture, steps, purpose, domain, device="cpu"):
        calls.append((branch, architecture, steps, domain, device))
        return {"id": branch + "-job", "state": "queued"}

    monkeypatch.setattr(cpu_pilots, "propose", queue)
    folder = tmp_path / "learning/cpu-pilots"
    first = cpu_pilots.schedule_cpu_pilots(tmp_path, folder)
    assert [row["state"] for row in first] == ["queued", "queued"]
    assert calls == [
        ("A", "gru", 40, "arithmetic", "cpu"),
        ("B", "transformer", 40, "arithmetic", "cpu"),
    ]
    assert cpu_pilots.schedule_cpu_pilots(tmp_path, folder) == first
    assert len(calls) == 2
    assert json.loads((folder / "cpu-pilot-A.json").read_text())["id"] == "A-job"


def test_unconfigured_compute_blocks_without_faking_job(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu_pilots, "inspect", lambda root: {"workers": [], "jobs": []})
    monkeypatch.setattr(
        cpu_pilots, "propose", lambda *args: (_ for _ in ()).throw(AssertionError())
    )
    result = cpu_pilots.schedule_cpu_pilots(tmp_path, tmp_path / "receipts")
    assert all(row["state"] == "blocked" for row in result)
    assert all("not configured" in row["reason"] for row in result)


def test_dispatch_failure_is_recorded_and_rate_limited(tmp_path, monkeypatch):
    config = tmp_path / "research/compute-config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    attempts = []
    monkeypatch.setattr(cpu_pilots, "inspect", lambda root: {"workers": [], "jobs": []})

    def fail(*args, **kwargs):
        attempts.append(1)
        raise ValueError("queue saturated")

    monkeypatch.setattr(cpu_pilots, "propose", fail)
    first = cpu_pilots.schedule_cpu_pilots(tmp_path, tmp_path / "receipts")
    assert first[0]["state"] == "blocked"
    assert first[0]["reason"] == "ValueError: queue saturated"
    assert cpu_pilots.schedule_cpu_pilots(tmp_path, tmp_path / "receipts") == first
    assert len(attempts) == 2


def test_cuda_dispatch(tmp_path, monkeypatch):
    config = tmp_path / "research/compute-config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    worker = {"phase": "idle", "stale": False, "device": "cuda"}
    snapshot = {"workers": [worker], "jobs": []}
    monkeypatch.setattr(cpu_pilots, "inspect", lambda root: snapshot)
    devices = []

    def queue(root, branch, architecture, steps, purpose, domain, device="cpu"):
        devices.append(device)
        return {"id": branch + "-job", "state": "queued"}

    monkeypatch.setattr(cpu_pilots, "propose", queue)
    result = cpu_pilots.schedule_cpu_pilots(tmp_path, tmp_path / "receipts")
    assert devices == ["cpu", "cuda"]
    assert [row["device"] for row in result] == ["cpu", "cuda"]


def test_cuda_terminal_failure_requires_valid_receipt_and_persisted_cooldown(tmp_path, monkeypatch):
    config = tmp_path / "research/compute-config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    folder = tmp_path / "learning/cpu-pilots"
    folder.mkdir(parents=True)
    pilot = folder / "cpu-pilot-B.json"
    pilot.write_text(
        json.dumps(
            {
                "branch": "B",
                "architecture": "transformer",
                "state": "queued",
                "id": "cuda-job",
                "device": "cuda",
                "cuda_dispatches": 1,
            }
        )
    )
    diagnostic = {
        "device": "cuda",
        "reasons": ["worker-time-budget"],
        "host_rss_limit_bytes": 768 * 2**20,
        "host_available_minimum_bytes": 2**30,
        "elapsed_limit_seconds": 150,
        "gpu_ready": True,
        "elapsed_seconds": 151,
        "host_rss_bytes": 100,
        "host_available_bytes": 2**31,
    }
    snapshot = {
        "workers": [{"phase": "idle", "stale": False, "device": "cuda"}],
        "jobs": [
            {
                "id": "cuda-job",
                "state": "failed",
                "attempts": 1,
                "last_failure": {"resource_failure": diagnostic},
            }
        ],
    }
    now = [10_000.0]
    monkeypatch.setattr(cpu_pilots, "time", type("Clock", (), {"time": lambda: now[0]}))
    monkeypatch.setattr(cpu_pilots, "inspect", lambda root: snapshot)
    queued = []

    def propose(root, branch, architecture, steps, purpose, domain, device="cpu"):
        queued.append((branch, device))
        return {"id": "retry-job", "state": "queued"}

    monkeypatch.setattr(cpu_pilots, "propose", propose)
    first = cpu_pilots.schedule_cpu_pilots(tmp_path, folder)
    b = next(row for row in first if row["branch"] == "B")
    assert b["state"] == "failed"
    assert b["retry_policy"] == "diagnosed-resource-cooldown"
    assert b["retry_at"] == now[0] + cpu_pilots.CUDA_RETRY_SECONDS
    assert "retry_at" not in queued

    now[0] += cpu_pilots.CUDA_RETRY_SECONDS - 1
    second = cpu_pilots.schedule_cpu_pilots(tmp_path, folder)
    assert next(row for row in second if row["branch"] == "B")["state"] == "failed"
    assert not any(branch == "B" for branch, _ in queued)

    now[0] += 2
    third = cpu_pilots.schedule_cpu_pilots(tmp_path, folder)
    b = next(row for row in third if row["branch"] == "B")
    assert b["state"] == "queued"
    assert b["cuda_dispatches"] == 2
    assert queued == [("A", "cpu"), ("B", "cuda")]


def test_cuda_cancel_and_missing_resource_diagnostics_fail_closed(tmp_path, monkeypatch):
    config = tmp_path / "research/compute-config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    folder = tmp_path / "learning/cpu-pilots"
    folder.mkdir(parents=True)
    pilot = folder / "cpu-pilot-B.json"
    pilot.write_text(
        json.dumps(
            {
                "branch": "B",
                "architecture": "transformer",
                "state": "queued",
                "id": "cuda-job",
                "device": "cuda",
                "cuda_dispatches": 1,
            }
        )
    )
    snapshot = {
        "workers": [],
        "jobs": [{"id": "cuda-job", "state": "cancelled", "attempts": 1}],
    }
    monkeypatch.setattr(cpu_pilots, "inspect", lambda root: snapshot)
    calls = []
    monkeypatch.setattr(
        cpu_pilots,
        "propose",
        lambda *args, **kwargs: calls.append(args[1]) or {"id": "unexpected", "state": "queued"},
    )
    for _ in range(2):
        result = cpu_pilots.schedule_cpu_pilots(tmp_path, folder)
        assert (
            next(row for row in result if row["branch"] == "B")["retry_policy"]
            == "terminal-cancelled"
        )
    assert calls == ["A"]

    snapshot["jobs"][0].update(state="failed", last_failure={"detail": "generic reserve failure"})
    pilot.write_text(
        json.dumps(
            {
                "branch": "B",
                "architecture": "transformer",
                "state": "queued",
                "id": "cuda-job",
                "device": "cuda",
                "cuda_dispatches": 1,
            }
        )
    )
    result = cpu_pilots.schedule_cpu_pilots(tmp_path, folder)
    b = next(row for row in result if row["branch"] == "B")
    assert b["retry_policy"] == "blocked-unverified-cuda-failure"
    assert b["retry_at"] is None
    assert calls == ["A"]
