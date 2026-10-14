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


def test_cuda_pilot_uses_idle_gpu_worker_without_inference_qualification(
    tmp_path, monkeypatch
):
    config = tmp_path / "research/compute-config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    monkeypatch.setattr(
        cpu_pilots,
        "inspect",
        lambda root: {
            "workers": [{"phase": "idle", "stale": False, "device": "cuda"}],
            "jobs": []
        },
    )
    devices = []

    def queue(root, branch, architecture, steps, purpose, domain, device="cpu"):
        devices.append(device)
        return {"id": branch + "-job", "state": "queued"}

    monkeypatch.setattr(cpu_pilots, "propose", queue)
    result = cpu_pilots.schedule_cpu_pilots(tmp_path, tmp_path / "receipts")
    assert devices == ["cpu", "cuda"]
    assert [row["device"] for row in result] == ["cpu", "cuda"]
