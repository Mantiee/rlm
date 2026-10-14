import base64
import hashlib
import json
import shutil
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from rlm.v100 import owned_node, remote_helper, remote_nodes
from rlm.v100.common import atomic_json


@pytest.mark.parametrize(
    "ram,temp,free,ready",
    [
        (3, 55, 2, True),
        (3, 79.9, 1, True),
        (0.9, 55, 2, False),
        (3, 80, 2, False),
        (3, 55, 0.2, False),
    ],
)
def test_dedicated_budget_uses_gpu_without_windows_cpu_idle_limit(ram, temp, free, ready):
    assert owned_node.resource_budget(ram, temp, free)["ready"] is ready


def test_invalid_telemetry_never_admits_gpu():
    with pytest.raises(ValueError):
        owned_node.resource_budget(3, float("nan"), 2)


def test_full_duty_is_scoped_to_the_explicit_owned_node():
    with pytest.raises(ValueError):
        remote_helper.validate_workload(64, 100)
    remote_helper.validate_workload(64, 100, True)
    with pytest.raises(ValueError, match="registered"):
        remote_helper.OllamaResearchClient(
            base_url="http://192.168.0.61:11435",
            model_digest="a" * 64,
            dedicated_host=True,
            helper_duty_percent=100,
        )


def test_independent_nodes_can_hold_workload_leases_together(monkeypatch):
    monkeypatch.setattr(remote_helper, "helper_boot_id", lambda: "owned-test")
    legacy = remote_helper.OllamaResearchClient(
        base_url="http://192.168.0.61:11435", model_digest="a" * 64
    )
    dedicated = remote_helper.OllamaResearchClient(
        base_url=remote_nodes.ENDPOINT,
        model_digest="a" * 64,
        dedicated_host=True,
        helper_duty_percent=100,
    )
    with legacy.workload_slot(wait=False):
        with dedicated.workload_slot(wait=False) as quota:
            dedicated.reserve_helper_idle(quota, 10)
            assert quota["planned_idle_seconds"] == 0


def test_stale_or_changed_node_never_routes_a_job(tmp_path):
    atomic_json(tmp_path / "research/owned-node.json", {})
    folder = tmp_path / "research/owned-node"
    profile = {"runtime": {"base_url": remote_nodes.ENDPOINT, "model_name": "gemma4:e2b-it-q4_K_M"}}
    path = folder / "profile.json"
    atomic_json(path, profile)
    state = {
        "admitted": True,
        "resources": {"ready": True},
        "updated": time.time(),
        "profile": str(path),
        "profile_sha256": remote_nodes.json_hash(profile),
    }
    atomic_json(folder / "status.json", state)
    assert remote_nodes.selected_node(tmp_path) == path.resolve()
    atomic_json(folder / "status.json", {**state, "updated": time.time() - 100})
    assert remote_nodes.selected_node(tmp_path) is None
    atomic_json(folder / "status.json", state)
    atomic_json(path, {"runtime": {"base_url": "http://192.168.0.10:11435"}})
    with pytest.raises(ValueError, match="changed"):
        remote_nodes.selected_node(tmp_path)


def test_other_lan_peer_cannot_reach_service():
    handler = owned_node.Handler.__new__(owned_node.Handler)
    handler.connection = Mock()
    handler.client_address = ("192.168.0.20", 1)
    handler.server = SimpleNamespace(owner="192.168.0.68", server_address=("192.168.0.9", 11435))
    handler.send_error = Mock()
    handler.forward()
    handler.send_error.assert_called_once_with(403)


def test_http_exposes_only_fixed_paths_and_models():
    node = SimpleNamespace(state={"ready": True}, generating=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), owned_node.Handler)
    server.owner, server.node, server.upstream_port = "192.168.0.68", node, 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert (
            requests.post(url + "/api/pull", json={"model": "anything"}, timeout=2).status_code
            == 404
        )
        assert (
            requests.post(url + "/api/chat", json={"model": "gpt-oss:120b"}, timeout=2).status_code
            == 400
        )
        assert requests.get(url + "/api/synta-resources", timeout=2).json()["ready"] is True
        assert requests.get(url + "/api/synta-cpu/../../etc/passwd", timeout=2).status_code == 404
    finally:
        server.shutdown()
        server.server_close()


def job(identity):
    return {
        "schema": "v100-compute-job-v1",
        "id": identity,
        "architecture": "gru",
        "steps": 10,
        "width": 32,
        "layers": 1,
        "threads": 2,
        "context": 128,
        "seconds": 10,
        "device": "cpu",
        "learning_rate": 0.001,
        "train": [{"group": "training", "prompt": "1+1=", "answer": "2"}],
        "validation": [{"group": "validation", "prompt": "2+1=", "answer": "3"}],
    }


def test_cpu_job_is_hash_pinned_idempotent_and_does_not_execute_received_code(
    tmp_path, monkeypatch
):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 2**30))
    source = tmp_path / "source"
    source.mkdir()
    for name in ("compute_kernel.py", "compute_worker.py"):
        shutil.copyfile(Path(owned_node.__file__).with_name(name), source / name)
    node = owned_node.Node(tmp_path / "work", source)
    monkeypatch.setattr(node.cpu_pool, "submit", lambda *args: None)
    node.state = {
        "cpu_ready": True,
        **{
            name + "_sha256": hashlib.sha256(
                (source / ("compute_" + name + ".py")).read_bytes()
            ).hexdigest()
            for name in ("kernel", "worker")
        },
    }
    value = job("a" * 24) | {
        "kernel_sha256": node.state["kernel_sha256"],
        "worker_sha256": node.state["worker_sha256"],
        "command": "touch /etc/not-allowed",
    }
    raw = json.dumps(value).encode()
    request = {
        "identity": "a" * 24 + "-" + "b" * 32,
        "job": base64.b64encode(raw).decode(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    try:
        assert node.submit_cpu(request)["state"] == "queued"
        assert node.submit_cpu(request)["state"] == "queued"
        assert len(node.processes) == 1
        with pytest.raises(ValueError, match="hash"):
            node.submit_cpu({**request, "sha256": "f" * 64})
        value["architecture"] = "shell"
        raw = json.dumps(value).encode()
        with pytest.raises(ValueError, match="built-in"):
            node.submit_cpu(
                {
                    **request,
                    "job": base64.b64encode(raw).decode(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    finally:
        node.cpu_pool.shutdown(wait=True)


def test_bridge_imports_only_hash_checked_artifacts_and_retains_local_validation(
    tmp_path, monkeypatch
):
    mailbox = tmp_path / "mailbox"
    atomic_json(tmp_path / "research/compute-config.json", {"mailbox": str(mailbox)})
    identity, nonce = "a" * 24, "b" * 32
    atomic_json(mailbox / "jobs" / (identity + ".json"), job(identity))
    atomic_json(tmp_path / "research/compute-jobs" / identity / "job.json", job(identity))
    raw = (mailbox / "jobs" / (identity + ".json")).read_bytes()
    atomic_json(
        mailbox / "claims" / identity / "lease.json",
        {"worker": "debian-3060ti-cpu", "nonce": nonce, "job_id": identity},
    )
    files = {
        "report.json": b'{"heldout_loss":1}',
        "weights.safetensors": b"untrusted-safe-tensor-fixture",
    }
    result = {
        "state": "complete",
        "job_sha256": hashlib.sha256(raw).hexdigest(),
        "files": {p: hashlib.sha256(b).hexdigest() for p, b in files.items()},
        "artifacts": {p: base64.b64encode(b).decode() for p, b in files.items()},
    }
    monkeypatch.setattr(remote_nodes, "request", lambda *args, **kwargs: result)
    remote_nodes.cpu_bridge(tmp_path, {"cpu_ready": True, "cpu_slots": 1})
    receipt = json.loads(
        (mailbox / "results" / (identity + "-" + nonce) / "receipt.json").read_text()
    )
    assert "untrusted" in receipt["scope"]
    assert "weights_promoted" not in receipt
    assert (
        mailbox / "results" / (identity + "-" + nonce) / "weights.safetensors"
    ).read_bytes() == files["weights.safetensors"]


def test_cpu_work_does_not_wait_for_gpu_qualification(tmp_path, monkeypatch):
    from rlm.v100 import distributed_compute, goal_compute

    events = []
    monkeypatch.setattr(
        remote_nodes,
        "request",
        lambda *args: {
            "contract": remote_nodes.CONTRACT,
            "updated": time.time(),
            "ready": False,
            "cpu_ready": True,
            "reason": "GPU absent",
        },
    )
    monkeypatch.setattr(remote_nodes, "cpu_bridge", lambda *args: events.append("cpu"))
    monkeypatch.setattr(distributed_compute, "tick", lambda *args: events.append("audit"))
    monkeypatch.setattr(goal_compute, "tick", lambda *args: events.append("goal"))
    assert not remote_nodes.tick(tmp_path)["admitted"]
    assert events == ["cpu", "audit", "goal"]


def test_cpu_error_does_not_disable_a_qualified_gpu(tmp_path, monkeypatch):
    from rlm.v100 import distributed_compute, goal_compute

    atomic_json(
        tmp_path / "research/owned-node/qualification-state.json",
        {"admitted": True, "phase": "qualified", "resources_revision": "a" * 40},
    )
    monkeypatch.setattr(
        remote_nodes,
        "request",
        lambda *args: {
            "contract": remote_nodes.CONTRACT,
            "updated": time.time(),
            "ready": True,
            "revision": "a" * 40,
        },
    )
    monkeypatch.setattr(
        remote_nodes,
        "cpu_bridge",
        lambda *args: (_ for _ in ()).throw(ValueError("bad CPU artifact")),
    )
    monkeypatch.setattr(distributed_compute, "tick", lambda *args: None)
    monkeypatch.setattr(goal_compute, "tick", lambda *args: None)
    value = remote_nodes.tick(tmp_path)
    assert value["admitted"] is True
    assert value["cpu_errors"] == {"transport": "bad CPU artifact"}


@pytest.mark.parametrize(
    "installed,returncode,expected",
    [(True, 0, "current"), (False, 0, "installed"), (False, 75, "deferred-busy")],
)
def test_owned_updater_retries_incomplete_install_and_defers_busy_node(
    tmp_path, monkeypatch, installed, returncode, expected
):
    from rlm.v100 import owned_node_update, release_selector

    monkeypatch.setattr(release_selector, "selection", lambda *a, **kw: None)

    base, state = tmp_path / "base", tmp_path / "state"
    revision = "a" * 40
    source = base / "releases" / revision
    source.mkdir(parents=True)
    (base / "current").symlink_to(source)
    if installed:
        atomic_json(state / "installation.json", {"stage": "prepared", "revision": revision})
    body = b"#!/bin/bash\nexit 0\n"
    manifest = {
        "node3060_installer": "tools/bootstrap-synta-3060.sh",
        "files": {"tools/bootstrap-synta-3060.sh": hashlib.sha256(body).hexdigest()},
    }

    def read(url, maximum=1):
        if "git/ref" in url:
            return json.dumps({"object": {"sha": revision}}).encode()
        return json.dumps(manifest).encode() if url.endswith("managed-release.json") else body

    monkeypatch.setattr(owned_node_update, "read", read)
    calls = []
    monkeypatch.setattr(
        owned_node_update.subprocess,
        "run",
        lambda args, **kwargs: calls.append(args) or SimpleNamespace(returncode=returncode),
    )
    assert owned_node_update.run(base, state)["stage"] == expected
    assert bool(calls) is not installed
    assert (base / "current").resolve() == source


def test_income_advisor_reuses_qualified_node_without_launching_master(tmp_path, monkeypatch):
    from rlm.v100 import competition, income_work, mission_chat

    path = tmp_path / "node.json"
    atomic_json(path, {})
    client = SimpleNamespace(timeout=5, identity=lambda: None, sampling_args={})
    monkeypatch.setattr(remote_nodes, "selected_node", lambda root: path)
    monkeypatch.setattr(mission_chat, "preferences", lambda root: {})
    monkeypatch.setattr(
        income_work, "load_profile", lambda *args: {"resources": {"device": "remote"}}
    )
    monkeypatch.setattr(competition, "helper_client", lambda *args: client)
    assert income_work.owned_advisor(tmp_path, "A", "training") is client
    assert client.activity_actor == "income-owned-advisor"
    assert client.timeout == 120
    assert client.research_token_ceiling == 2048



def test_cuda_jobs_require_dedicated_resource_budget_not_model_qualification(tmp_path, monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 2**30))
    source = tmp_path / "source"
    source.mkdir()
    for name in ("compute_kernel.py", "compute_worker.py"):
        shutil.copyfile(Path(owned_node.__file__).with_name(name), source / name)
    node = owned_node.Node(tmp_path / "gpu-work", source)
    monkeypatch.setattr(node.cpu_pool, "submit", lambda *args: None)
    node.state = {
        "ready": True,
        "cpu_ready": True,
        "free_vram_gib": 2.75,
        "temperature_c": 40,
        "kernel_sha256": hashlib.sha256((source / "compute_kernel.py").read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256((source / "compute_worker.py").read_bytes()).hexdigest(),
    }
    value = job("c" * 24)
    value.update(
        device="cuda",
        kernel_sha256=node.state["kernel_sha256"],
        worker_sha256=node.state["worker_sha256"],
    )
    raw = json.dumps(value).encode()
    request = {
        "identity": "c" * 24 + "-" + "d" * 32,
        "job": base64.b64encode(raw).decode(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    try:
        node.state["free_vram_gib"] = 1.0
        with pytest.raises(BlockingIOError, match="CUDA reserve"):
            node.submit_cpu(request)
        node.state["free_vram_gib"] = 2.75
        assert node.submit_cpu(request)["state"] == "queued"
        assert node.gpu_compute_claimed
        assert node.submit_cpu(request)["state"] == "queued"
    finally:
        node.cpu_pool.shutdown(wait=True)


def test_cuda_lane_dispatched_independently_of_finance_model_qualification(tmp_path):
    mailbox = tmp_path / "mailbox"
    atomic_json(tmp_path / "research/compute-config.json", {"mailbox": str(mailbox)})
    remote_nodes.cpu_bridge(
        tmp_path,
        {
            "ready": True,
            "cpu_ready": True,
            "cpu_slots": 8,
            "cpu_active": 0,
            "free_vram_gib": 2.8,
            "temperature_c": 35,
            "generating": False,
            "gpu_compute_claimed": False,
        },
    )
    gpu = json.loads((mailbox / "workers/debian-3060ti-gpu.json").read_text())
    assert gpu["device"] == "cuda" and gpu["phase"] == "idle"
    assert gpu["gpu_enabled"]
    assert json.loads((mailbox / "workers/debian-3060ti-cpu.json").read_text())["device"] == "cpu"
