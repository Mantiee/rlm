import json
from types import SimpleNamespace

import pytest

from rlm.v100 import distributed_compute, goal_compute, goal_learning, mission, research_tools
from rlm.v100.common import atomic_json
from rlm.v100.goals import set_goal


@pytest.fixture
def outcomes(tmp_path, monkeypatch):
    clock, values = [1000.0], {}
    monkeypatch.setattr(goal_learning, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(
        goal_learning.shutil, "disk_usage", lambda root: SimpleNamespace(free=8 * 2**30)
    )
    monkeypatch.setattr(
        research_tools,
        "download_page",
        lambda url, max_bytes: (url, json.dumps({"value": values[url]})),
    )
    monkeypatch.setattr(mission, "status", lambda root: {"running": False})
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "id": "fixed",
                "skill": "retention",
                "match": "exact",
                "expected": "4",
                "messages": [{"role": "user", "content": "2+2"}],
            }
        )
        + "\n"
    )
    goal = set_goal(tmp_path, "Find useful predictive associations for my goal", suite)
    mailbox = tmp_path / "owned-mailbox"
    distributed_compute.configure(tmp_path, mailbox)
    for index in range(24):
        clock[0] = 1000 + index * 120
        url = f"https://example.org/metric{index}"
        values[url] = 100 + index
        before = goal_learning.observe(tmp_path, url, "value")
        prediction = goal_learning.predict(
            tmp_path,
            "A",
            {
                "question": "Does the observed metric predict a change?",
                "rationale": "Test a precommitted association",
                "evidence": [before["id"]],
                "target": before["id"],
                "horizon_seconds": 60,
                "threshold": 1,
                "probabilities": [0.2, 0.2, 0.6],
            },
        )
        clock[0] += 60
        values[url] += 3 if index % 2 else -3
        after = goal_learning.observe(tmp_path, url, "value")
        goal_learning.settle(tmp_path, prediction["id"], after["id"])
    return tmp_path, mailbox, goal, suite


def test_real_observation_pipeline_queues_goal_jobs_with_frozen_disjoint_labels(outcomes):
    root, mailbox, goal, _ = outcomes
    a = goal_compute.propose(root, "A")
    b = goal_compute.propose(root, "B")
    assert a["state"] == b["state"] == "queued"
    assert a["goal_id"] == goal["id"] and not a["weights_changed"]
    job = json.loads((mailbox / "jobs" / (a["id"] + ".json")).read_text())
    assert job["architecture"] == "gru"
    assert {r["group"] for r in job["train"]}.isdisjoint(r["group"] for r in job["validation"])
    assert {r["answer"] for r in job["train"] + job["validation"]} == {"up", "down"}
    assert all(
        "actual" not in r["prompt"] and "probabilities" not in r["prompt"] for r in job["train"]
    )
    goal_compute.verify_job(root, job)
    assert goal_compute.propose(root, "A")["reused"] is True
    assert len(list((mailbox / "jobs").glob("*.json"))) == 2
    job["train"][0]["answer"] = "flat"
    with pytest.raises(ValueError, match="projection or labels"):
        goal_compute.verify_job(root, job)


def test_goal_change_does_not_relabel_old_jobs_and_dispatch_blocks_without_data(outcomes):
    root, _, _, suite = outcomes
    goal_compute.propose(root)
    set_goal(root, "A new operator goal", suite)
    result = goal_compute.propose(root)
    assert result["state"] == "blocked" and result["available"] == 0


def test_automatic_goal_dispatch_uses_idle_owned_worker_and_preserves_pending_priority(outcomes):
    root, mailbox, _, _ = outcomes
    atomic_json(root / "research/user-preferences.json", {"automatic_goal_compute": True})
    atomic_json(mailbox / "workers/windows-cpu.json", {"phase": "idle", "threads": 2})
    goal_compute.tick(root)
    status = json.loads((root / "research/goal-compute/status.json").read_text())
    assert status["state"] == "dispatched"
    assert len(list((mailbox / "jobs").glob("*.json"))) == 2
    status["next_poll"] = 0
    atomic_json(root / "research/goal-compute/status.json", status)
    goal_compute.tick(root)
    assert (
        json.loads((root / "research/goal-compute/status.json").read_text())["state"] == "waiting"
    )


def test_dispatch_without_fresh_workers_reports_blocker_without_dummy_jobs(tmp_path):
    atomic_json(tmp_path / "research/user-preferences.json", {"automatic_goal_compute": True})
    goal_compute.tick(tmp_path)
    status = json.loads((tmp_path / "research/goal-compute/status.json").read_text())
    assert status["state"] == "blocked" and "idle owned CPU" in status["reason"]
    assert not (tmp_path / "research/compute-jobs").exists()


def test_goal_projection_really_trains_and_reloads_cpu_weights(outcomes):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from rlm.v100 import compute_kernel

    root, mailbox, _, _ = outcomes
    result = goal_compute.propose(root)
    job = json.loads((mailbox / "jobs" / (result["id"] + ".json")).read_text())
    job.update(steps=10, width=32, threads=1, seconds=30)
    trained = compute_kernel.run(job, root / "cpu-trained")
    assert trained["metrics"] and trained["metrics"][-1]["step"] > 0
    restored = compute_kernel.run(
        job, root / "cpu-reloaded", root / "cpu-trained/weights.safetensors"
    )
    assert restored["heldout_loss"] == pytest.approx(trained["heldout_loss"], rel=1e-5)
    assert restored["weights_promoted"] is False


def test_goal_outcomes_execute_owned_child_and_receive_independent_host_decision(
    outcomes, monkeypatch
):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from pathlib import Path

    import psutil

    from rlm.v100 import compute_kernel, compute_worker

    root, mailbox, goal, _ = outcomes
    # Real child training/reload; sensor readings simulated because this sandbox
    # does not expose /proc. No Windows/CUDA resource-cap assertion is made.
    monkeypatch.setattr(compute_worker, "available", lambda: (True, "Test CPU ready"))
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval: 10)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=12 * 2**30))
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=100 * 2**20), children=lambda recursive: []
        ),
    )
    proposed = goal_compute.propose(root, "A")
    path, lease = compute_worker.claim(mailbox, "goal-cpu")
    assert lease["job_id"] == proposed["id"]
    compute_worker.execute(mailbox, path, lease, Path(compute_kernel.__file__))
    report = json.loads(
        (mailbox / "results" / (proposed["id"] + "-" + lease["nonce"]) / "report.json").read_text()
    )
    assert report["metrics"][-1]["step"] > 0
    distributed_compute.tick(root)
    checked = distributed_compute.validate_locally(root, proposed["id"])
    assert checked["state"] in ("locally-validated", "rejected")
    assert checked["weights_promoted"] is False
    assert Path(checked["local_report"]).is_file()
    events = [
        json.loads(line)
        for p in root.glob("research/logs/activity/*/timeline.jsonl")
        for line in p.read_text().splitlines()
    ]
    transitions = [e["payload"]["state"] for e in events if e["kind"] == "compute-state-changed"]
    assert transitions == ["queued", "imported", checked["state"]]
    # Historical experiments retain their original goal, but their labels
    # cannot enter a newly selected goal training pool.
    set_goal(root, "Different goal", outcomes[3])
    job = json.loads(path.read_text())
    goal_compute.verify_job(root, job)
    assert job["provenance"]["goal_id"] == goal["id"]
    assert goal_compute.propose(root)["state"] == "blocked"
