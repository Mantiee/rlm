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
    with pytest.raises(ValueError, match="CPU"):
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


def test_small_model_real_training_safetensors_and_local_forward(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    train, validation = distributed_compute.fresh_examples("arithmetic", 32)
    job = {
        "schema": "v100-compute-job-v1",
        "architecture": "gru",
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
