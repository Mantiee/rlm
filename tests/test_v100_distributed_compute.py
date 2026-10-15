import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rlm.v100 import (
    colab_jobs,
    compute_kernel,
    compute_worker,
    distributed_compute,
    drones,
    mission,
    remote_nodes,
)
from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash


def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(mission, "status", lambda root: {"running": False})
    mailbox = tmp_path / "owned-mailbox"
    distributed_compute.configure(tmp_path, mailbox)
    proposal = distributed_compute.propose(
        tmp_path, "A", "gru", 10, "Test fresh small model", "arithmetic"
    )
    return mailbox, proposal["id"]


def test_owned_job_validation_and_new_data_each_trial(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    first = compute_worker.read_json(mailbox / "jobs" / (identity + ".json"))
    second_id = distributed_compute.propose(
        tmp_path, "B", "transformer", 10, "Different trial", "arithmetic"
    )["id"]
    second = compute_worker.read_json(mailbox / "jobs" / (second_id + ".json"))
    compute_kernel.validate(first)
    assert {r["group"] for r in first["train"] + first["validation"]}.isdisjoint(
        r["group"] for r in second["train"] + second["validation"]
    )
    first["device"] = "cuda"
    compute_kernel.validate(first)
    first["device"] = "malicious-device"
    with pytest.raises(ValueError, match="CPU/CUDA"):
        compute_kernel.validate(first)


def test_two_workers_cannot_claim_same_job(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)

    def take(name):
        barrier.wait(timeout=2)
        return compute_worker.claim(mailbox, name)

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(take, ["one", "two"]))
    assert sum(r is not None for r in result) == 1
    assert next(r for r in result if r)[1]["job_id"] == identity


def test_lost_worker_lease_requeued_and_cancelled_job_not_reclaimed(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    _, lease = compute_worker.claim(mailbox, "lost")
    lease["heartbeat"] = time.time() - 301
    atomic_json(mailbox / "claims" / identity / "lease.json", lease)
    distributed_compute.tick(tmp_path)
    state = distributed_compute.inspect(tmp_path)["jobs"][0]
    assert state["state"] == "queued" and state["attempts"] == 1
    assert compute_worker.claim(mailbox, "new")
    distributed_compute.cancel(tmp_path, identity)
    assert compute_worker.claim(mailbox, "third") is None


def failed_cuda_receipt(tmp_path, mailbox, identity, diagnostic=True):
    job_path = tmp_path / "research/compute-jobs" / identity / "job.json"
    job = compute_worker.read_json(job_path)
    job["device"] = "cuda"
    atomic_json(job_path, job)
    atomic_json(mailbox / "jobs" / (identity + ".json"), job)
    state_path = job_path.with_name("state.json")
    state = compute_worker.read_json(state_path)
    state["job_sha256"] = file_hash(job_path)
    atomic_json(state_path, state)
    nonce = "c" * 32
    lease = {"job_id": identity, "worker": "debian-3060ti-gpu", "nonce": nonce}
    atomic_json(mailbox / "claims" / identity / "lease.json", lease)
    result = {
        **lease,
        "state": "failed",
        "job_sha256": file_hash(job_path),
        "detail": "Owned compute reserve stopped: worker-host-rss-budget",
    }
    if diagnostic:
        result["resource_failure"] = {
            "device": "cuda",
            "reasons": ["worker-host-rss-budget"],
            "gpu_ready": True,
            "elapsed_seconds": 4.0,
            "host_rss_bytes": 768 * 2**20 + 1,
            "host_available_bytes": 4 * 2**30,
            "host_rss_limit_bytes": 768 * 2**20,
            "host_available_minimum_bytes": 2**30,
            "elapsed_limit_seconds": 150,
        }
    destination = mailbox / "results" / (identity + "-" + nonce)
    atomic_json(destination / "receipt.json", result)


def test_cuda_failure_persists_diagnostic_and_cooldown_before_retry(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    failed_cuda_receipt(tmp_path, mailbox, identity)
    before = time.time()

    distributed_compute.tick(tmp_path)

    state = compute_worker.read_json(
        tmp_path / "research/compute-jobs" / identity / "state.json"
    )
    assert state["state"] == "queued"
    assert state["attempts"] == 1
    assert state["retry_not_before"] >= before + distributed_compute.CUDA_RETRY_COOLDOWN_SECONDS
    assert state["last_failure"]["resource_failure"]["reasons"] == [
        "worker-host-rss-budget"
    ]

    monkeypatch.setattr(remote_nodes, "request", lambda *args, **kwargs: pytest.fail("retried"))
    remote_nodes.cpu_bridge(
        tmp_path,
        {
            "cpu_ready": True,
            "cpu_slots": 1,
            "cpu_active": 0,
            "ready": True,
            "free_vram_gib": 3.0,
            "temperature_c": 40,
            "generating": False,
            "gpu_compute_claimed": False,
            "kernel_sha256": "unused",
            "worker_sha256": "unused",
        },
    )
    assert not (mailbox / "claims" / identity / "lease.json").exists()


def test_cuda_failure_without_structured_diagnostic_is_terminal(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    failed_cuda_receipt(tmp_path, mailbox, identity, diagnostic=False)

    distributed_compute.tick(tmp_path)

    state = compute_worker.read_json(
        tmp_path / "research/compute-jobs" / identity / "state.json"
    )
    assert state["state"] == "failed"
    assert state["last_failure"]["resource_failure"] is None
    assert state["retry_not_before"] is None
    assert (mailbox / "closed" / (identity + ".json")).is_file()


def test_cancelled_cuda_job_is_not_reclaimed_and_remote_lease_is_deleted(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    job_path = tmp_path / "research/compute-jobs" / identity / "job.json"
    job = compute_worker.read_json(job_path)
    job["device"] = "cuda"
    atomic_json(job_path, job)
    atomic_json(mailbox / "jobs" / (identity + ".json"), job)
    nonce = "d" * 32
    atomic_json(
        mailbox / "claims" / identity / "lease.json",
        {"job_id": identity, "worker": "debian-3060ti-gpu", "nonce": nonce},
    )
    distributed_compute.cancel(tmp_path, identity)
    calls = []
    monkeypatch.setattr(
        remote_nodes,
        "request",
        lambda method, path, value=None: calls.append((method, path)) or {"state": "cancelled"},
    )

    remote_nodes.cpu_bridge(
        tmp_path,
        {
            "cpu_ready": True,
            "cpu_slots": 1,
            "cpu_active": 0,
            "ready": True,
            "free_vram_gib": 3.0,
            "temperature_c": 40,
            "generating": False,
            "gpu_compute_claimed": False,
            "kernel_sha256": "unused",
            "worker_sha256": "unused",
        },
    )
    distributed_compute.tick(tmp_path)

    state = compute_worker.read_json(
        tmp_path / "research/compute-jobs" / identity / "state.json"
    )
    assert state["state"] == "cancelled"
    assert state["retry_not_before"] is None
    assert calls == [("DELETE", "/api/synta-cpu/" + identity + "-" + nonce)]


def received(tmp_path, mailbox, identity, bad_hash=False):
    path, lease = compute_worker.claim(mailbox, "remote")
    target = mailbox / "results" / (identity + "-" + lease["nonce"])
    atomic_json(target / "report.json", {"initial_loss": 5, "heldout_loss": 4, "seconds": 2})
    (target / "weights.safetensors").write_bytes(b"not deserialized during import")
    atomic_json(
        target / "receipt.json",
        {
            **lease,
            "state": "complete",
            "job_sha256": "forged" if bad_hash else file_hash(path),
            "files": {
                name: file_hash(target / name) for name in ("report.json", "weights.safetensors")
            },
        },
    )
    return target


def test_remote_result_is_untrusted_import_then_independent_audit_job(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    received(tmp_path, mailbox, identity)
    distributed_compute.tick(tmp_path)
    state = distributed_compute.inspect(tmp_path)["jobs"][0]
    assert state["state"] == "imported" and not state["weights_promoted"]
    assert drones.inspect(tmp_path) == []
    distributed_compute.tick(tmp_path)
    job = drones.inspect(tmp_path)[0]
    assert job["kind"] == "compute-audit" and job["state"] == "queued"
    assert distributed_compute.inspect(tmp_path)["jobs"][0]["validation_job"] == job["id"]


def test_remote_forged_input_hash_rejected(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    received(tmp_path, mailbox, identity, bad_hash=True)
    distributed_compute.tick(tmp_path)
    state = distributed_compute.inspect(tmp_path)["jobs"][0]
    assert state["state"] == "queued" and "changed" in state["detail"]
    assert not drones.inspect(tmp_path)


def test_received_tensor_tampering_rejected_before_loading(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    received(tmp_path, mailbox, identity)
    distributed_compute.tick(tmp_path)
    state = distributed_compute.inspect(tmp_path)["jobs"][0]
    (Path(state["result"]) / "weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="weights changed"):
        distributed_compute.validate_locally(tmp_path, identity)


def test_each_notebook_package_has_fresh_data_and_pinned_safe_kernel(tmp_path):
    atomic_json(tmp_path / "research/self-code-source.json", {"revision": "a" * 40})
    a = colab_jobs.propose(tmp_path, "A", 10, "Notebook A")
    b = colab_jobs.propose(tmp_path, "B", 10, "Notebook B")
    assert a["pool_sha256"] != b["pool_sha256"]
    compute_kernel.validate(a["experiment"])
    assert (
        file_hash(Path(a["proposal"]).with_suffix("") / "kernel.py")
        == a["experiment"]["kernel_sha256"]
    )


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_small_model_real_training_safetensors_and_local_forward(tmp_path, architecture):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    train, validation = distributed_compute.fresh_examples("arithmetic", 32)
    job = {
        "schema": "v100-compute-job-v1",
        "architecture": architecture,
        "width": 32,
        "layers": 1,
        "steps": 10,
        "context": 256,
        "threads": 1,
        "device": "cpu",
        "seconds": 30,
        "learning_rate": 0.001,
        "train": train,
        "validation": validation,
    }
    result = compute_kernel.run(job, tmp_path / "trained")
    assert torch.isfinite(torch.tensor(result["heldout_loss"]))
    local = compute_kernel.run(job, tmp_path / "host", tmp_path / "trained/weights.safetensors")
    assert local["heldout_loss"] == pytest.approx(result["heldout_loss"], rel=1e-5)
    assert not local["weights_promoted"]


def test_owned_worker_executes_pinned_child_and_host_validates_weights(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import psutil

    mailbox, identity = configured(tmp_path, monkeypatch)
    monkeypatch.setattr(compute_worker, "available", lambda: (True, "Test CPU host ready"))
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval: 10)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("Memory", (), {"available": 12 * 2**30})(),
    )
    # The sandbox has no /proc. Training and validation are real subprocesses;
    # resource sensor readings are simulated, not asserted as hardware validation.
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: type(
            "Process",
            (),
            {
                "memory_info": lambda self: type("RSS", (), {"rss": 100 * 2**20})(),
                "children": lambda self, recursive: [],
            },
        )(),
    )
    path, lease = compute_worker.claim(mailbox, "actual-cpu")
    compute_worker.execute(mailbox, path, lease, Path(compute_kernel.__file__))
    distributed_compute.tick(tmp_path)
    state = distributed_compute.validate_locally(tmp_path, identity)
    assert state["state"] == "locally-validated"
    assert not state["weights_promoted"]
    assert Path(state["local_report"]).is_file()


def test_oversized_compute_dataset_and_symlink_json_rejected(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    path = mailbox / "jobs" / (identity + ".json")
    job = json.loads(path.read_text())
    job["train"] *= 4
    with pytest.raises(ValueError, match="dataset"):
        compute_kernel.validate(job)
    linked = tmp_path / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(ValueError, match="Unsafe"):
        compute_worker.read_json(linked)


def test_full_drone_queue_defers_import_without_discarding_weights(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    received(tmp_path, mailbox, identity)
    distributed_compute.tick(tmp_path)
    actual = drones.schedule

    def full(*args):
        raise ValueError("Queue full")

    monkeypatch.setattr(drones, "schedule", full)
    distributed_compute.tick(tmp_path)
    assert distributed_compute.inspect(tmp_path)["jobs"][0]["state"] == "imported"
    monkeypatch.setattr(drones, "schedule", actual)
    distributed_compute.tick(tmp_path)
    assert drones.inspect(tmp_path)[0]["kind"] == "compute-audit"


def test_compute_transition_journal_is_complete_and_does_not_repeat_polling(tmp_path):
    path = tmp_path / "research/compute-jobs" / ("a" * 24) / "state.json"
    state = {"state": "queued", "branch": "A", "attempts": 0}
    distributed_compute.record_state(tmp_path, path, state)
    distributed_compute.record_state(tmp_path, path, state)
    state.update(state="running", worker="windows-cpu")
    distributed_compute.record_state(tmp_path, path, state)
    state.update(state="failed", attempts=3, detail="Worker failed")
    distributed_compute.record_state(tmp_path, path, state)
    events = [
        json.loads(line)
        for p in tmp_path.glob("research/logs/activity/*/timeline.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert [e["payload"]["state"] for e in events] == ["queued", "running", "failed"]
    assert all(e["payload"]["job_id"] == "a" * 24 for e in events)
    assert events[-1]["payload"]["detail"] == "Worker failed"


def test_generic_cpu_worker_does_not_claim_cuda_job(tmp_path, monkeypatch):
    mailbox, identity = configured(tmp_path, monkeypatch)
    path = mailbox / "jobs" / (identity + ".json")
    job = json.loads(path.read_text())
    job["device"] = "cuda"
    atomic_json(path, job)
    assert compute_worker.claim(mailbox, "windows-gaming-cpu") is None
    assert not (mailbox / "claims" / identity).exists()
