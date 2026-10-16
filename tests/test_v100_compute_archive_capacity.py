"""Retained evidence cannot starve explicitly admitted new owned-compute work."""

import json
from types import SimpleNamespace

import pytest

from rlm.v100 import compute_worker, remote_nodes
from rlm.v100 import distributed_compute as lab
from rlm.v100.common import atomic_json


@pytest.fixture
def world(tmp_path, monkeypatch):
    mailbox = tmp_path / "mailbox"
    for name in ("jobs", "claims", "results", "workers", "closed"):
        (mailbox / name).mkdir(parents=True)
    atomic_json(tmp_path / "research/compute-config.json", {"mailbox": str(mailbox)})
    monkeypatch.setattr(lab.shutil, "disk_usage", lambda root: SimpleNamespace(free=100 * 2**30))
    monkeypatch.setattr(lab, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="f" * 32)))
    return tmp_path, mailbox


def history(root, mailbox, count=128, state="locally-validated"):
    for index in range(count):
        identity = f"{index:024x}"
        atomic_json(
            root / "research/compute-jobs" / identity / "state.json",
            {"state": state, "attempts": 1, "branch": "A"},
        )
        atomic_json(mailbox / "jobs" / (identity + ".json"), {"device": "cpu"})
        if state not in ("queued", "running", "imported"):
            atomic_json(mailbox / "closed" / (identity + ".json"), {"state": state})


def configure(root, value):
    path = root / "research/compute-jobs/capacity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def queue(root, mailbox):
    return lab.queue_job(
        root,
        mailbox,
        "A",
        "gru",
        10,
        "New verified bounded trial",
        "arithmetic",
        [{"group": "train", "prompt": "1+1=", "answer": "2"}],
        [{"group": "validation", "prompt": "2+2=", "answer": "4"}],
    )


def test_default128_archive_ceiling_is_retained(world):
    root, mailbox = world
    history(root, mailbox)
    assert lab.queue_capacity(root) == {"max_jobs": 128, "minimum_free_disk_gib": 2}
    with pytest.raises(ValueError, match="128/128 retained"):
        queue(root, mailbox)
    assert len(lab.job_states(root)) == 128


def test_128_terminal_history_new_job_is_visible_and_dedicated_bridge_progresses(
    world, monkeypatch
):
    root, mailbox = world
    history(root, mailbox)
    configure(root, {"max_jobs": 256, "minimum_free_disk_gib": 4})
    original = {path: path.read_bytes() for path, _ in lab.job_states(root)}
    proposal = queue(root, mailbox)
    assert proposal["id"] == "f" * 24
    view = lab.inspect(root)
    assert view["retained_jobs"] == 129 and view["jobs_omitted"] == 1
    assert view["jobs"][0]["id"] == proposal["id"]
    import requests

    job = compute_worker.read_json(mailbox / "jobs" / (proposal["id"] + ".json"))
    submitted = []

    def request(method, path, value=None):
        if method == "GET":
            raise requests.HTTPError(response=SimpleNamespace(status_code=400))
        submitted.append(value)
        return {"state": "running"}

    monkeypatch.setattr(remote_nodes, "request", request)
    remote_nodes.cpu_bridge(
        root,
        {
            "cpu_ready": True,
            "cpu_slots": 1,
            "kernel_sha256": job["kernel_sha256"],
            "worker_sha256": job["worker_sha256"],
        },
    )
    assert len(submitted) == 1 and submitted[0]["identity"].startswith(proposal["id"])
    lab.tick(root)
    state = compute_worker.read_json(root / "research/compute-jobs" / proposal["id"] / "state.json")
    assert state["state"] == "running" and state["worker"] == "debian-3060ti-cpu"
    assert all(path.read_bytes() == saved for path, saved in original.items())


def test_active16_limit_is_not_expanded_by_archive_configuration(world, monkeypatch):
    root, mailbox = world
    configure(root, {"max_jobs": 2048})
    history(root, mailbox, 16, "queued")
    with pytest.raises(ValueError, match="16 active"):
        queue(root, mailbox)
    monkeypatch.setattr(
        lab,
        "fresh_examples",
        lambda *a, **k: pytest.fail("No dataset allocation when active queue full"),
    )
    with pytest.raises(ValueError, match="16 active"):
        lab.propose(root, "A", "gru", 10, "Purpose", "arithmetic")


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"unknown": 1},
        {"max_jobs": True},
        {"max_jobs": 127},
        {"max_jobs": 2049},
        {"max_jobs": 256.0},
        {"minimum_free_disk_gib": True},
        {"minimum_free_disk_gib": 1},
        {"minimum_free_disk_gib": 65},
        {"minimum_free_disk_gib": float("nan")},
    ],
)
def test_invalid_capacity_fails_closed_before_publication(world, value):
    root, mailbox = world
    configure(root, value)
    with pytest.raises(ValueError):
        queue(root, mailbox)
    assert not list((mailbox / "jobs").glob("*.json"))


def test_disk_reserve_is_checked_before_publication(world, monkeypatch):
    root, mailbox = world
    configure(root, {"max_jobs": 256, "minimum_free_disk_gib": 4})
    monkeypatch.setattr(lab.shutil, "disk_usage", lambda root: SimpleNamespace(free=3 * 2**30))
    with pytest.raises(ValueError, match="4 GiB"):
        queue(root, mailbox)
    assert not list((mailbox / "jobs").glob("*.json"))


def test_capacity_file_read_is_bounded(world):
    root, _ = world
    path = configure(root, {})
    path.write_bytes(b" " * 4097)
    with pytest.raises(ValueError, match="oversized"):
        lab.queue_capacity(root)


def test_linked_capacity_is_rejected(world):
    root, _ = world
    path = configure(root, {})
    target = root / "other.json"
    target.write_text("{}")
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    with pytest.raises(ValueError, match="escaped"):
        lab.queue_capacity(root)


def test_inspect_preserves_active_after_128_older_terminal_records(world):
    root, mailbox = world
    history(root, mailbox)
    configure(root, {"max_jobs": 256})
    atomic_json(
        root / "research/compute-jobs" / ("f" * 24) / "state.json",
        {"state": "imported", "branch": "B", "attempts": 0},
    )
    assert lab.inspect(root)["jobs"][0]["state"] == "imported"


def test_concurrent_publication_cannot_exceed_active16(world, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from itertools import count

    root, mailbox = world
    configure(root, {"max_jobs": 256})
    history(root, mailbox, 15, "queued")
    identities = count(1000)
    monkeypatch.setattr(
        lab,
        "uuid",
        SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=f"{next(identities):024x}" + "0" * 8)),
    )

    def attempt():
        try:
            return queue(root, mailbox)
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))
    assert sum(isinstance(value, dict) for value in results) == 1
    assert any(isinstance(value, str) and "16 active" in value for value in results)
    assert sum(state["state"] == "queued" for _, state in lab.job_states(root)) == 16
