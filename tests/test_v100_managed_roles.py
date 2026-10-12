import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from rlm.v100 import managed_bootstrap as updater
from rlm.v100.common import atomic_json
from rlm.v100.model_roles import better, json_hash, selected_profile


def test_role_admission_keeps_quality_and_requires_material_speed():
    baseline = dict(passed=20, critical_passed=True, seconds=10)
    assert not better("fast", baseline, dict(passed=19, critical_passed=True, seconds=1))
    assert not better("fast", baseline, dict(passed=20, critical_passed=False, seconds=1))
    assert not better("fast", baseline, dict(passed=20, critical_passed=True, seconds=9.5))
    assert better("fast", baseline, dict(passed=20, critical_passed=True, seconds=8))
    assert better("planner", dict(baseline, passed=19), dict(baseline, seconds=12))


def test_role_registry_rejects_changed_identity_and_escape(tmp_path):
    folder = tmp_path / "research/model-roles"
    profile = {"resources": {"model_digest": "a" * 64}}
    report = dict(admitted=True, role="fast", model_digest="a" * 64)
    atomic_json(folder / "profile.json", profile)
    atomic_json(folder / "report.json", report)
    entry = dict(
        profile=str(folder / "profile.json"),
        report=str(folder / "report.json"),
        profile_sha256=json_hash(profile),
        report_sha256=json_hash(report),
    )
    atomic_json(folder / "active.json", {"roles": {"fast": entry}})
    assert selected_profile(tmp_path, "fast") == (folder / "profile.json").resolve()
    atomic_json(folder / "profile.json", {"resources": {"model_digest": "b" * 64}})
    with pytest.raises(ValueError, match="changed"):
        selected_profile(tmp_path, "fast")
    entry["profile"] = str(tmp_path / "outside.json")
    atomic_json(folder / "active.json", {"roles": {"fast": entry}})
    with pytest.raises(ValueError, match="escaped"):
        selected_profile(tmp_path, "fast")


def test_manifest_entrypoint_must_be_declared(monkeypatch):
    payload = dict(
        schema="synta-managed-release-v1",
        version=83,
        files={},
        linux_installer="../../foreign.sh",
        windows_prepare="tools/a.ps1",
    )
    monkeypatch.setattr(updater, "fetch", lambda url: json.dumps(payload).encode())
    with pytest.raises(ValueError, match="declared"):
        updater.release("a" * 40)


def test_verified_materialization_rejects_wrong_bytes(tmp_path, monkeypatch):
    data = b"owned release"
    manifest = {"files": {"tools/update.sh": hashlib.sha256(data).hexdigest()}}
    monkeypatch.setattr(updater, "fetch", lambda url: data)
    updater.materialize(tmp_path, "a" * 40, manifest)
    assert (tmp_path / "tools/update.sh").read_bytes() == data
    monkeypatch.setattr(updater, "fetch", lambda url: b"changed")
    with pytest.raises(ValueError, match="checksum"):
        updater.materialize(tmp_path, "a" * 40, manifest)
    assert (tmp_path / "tools/update.sh").read_bytes() == data


def test_training_defers_package_without_stopping_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        updater.subprocess,
        "check_output",
        lambda args, **kw: "python -m rlm.v100.cli train dataset.jsonl"
        if args[0] == "ps"
        else "0.1.3+v100.82",
    )
    monkeypatch.setattr(updater.subprocess, "run", lambda *a, **kw: calls.append(a))
    result = updater.linux(tmp_path, tmp_path, "a" * 40, {"version": 83})
    assert result["stage"] == "deferred-training"
    assert not calls


def test_downgrade_is_rejected_before_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        updater.subprocess,
        "check_output",
        lambda args, **kw: "" if args[0] == "ps" else "0.1.3+v100.99",
    )
    with pytest.raises(ValueError, match="downgrade"):
        updater.linux(tmp_path, tmp_path, "a" * 40, {"version": 83})


def test_v83_bash_syntax():
    base = Path(__file__).parents[1]
    for name in ("bootstrap-synta-debian.sh", "update-synta-v83.sh"):
        subprocess.run(["/bin/bash", "-n", str(base / "tools" / name)], check=True)


def test_native_maintenance_never_interrupts_train(tmp_path, monkeypatch):
    from rlm.v100 import managed_native

    monkeypatch.setattr(
        managed_native.subprocess,
        "check_output",
        lambda *a, **k: "python -m rlm.v100.cli train dataset.jsonl",
    )
    assert managed_native.run(tmp_path, allow_maintenance=True)["state"] == "deferred-training"


def test_native_maintenance_restores_profile_after_failed_benchmark(tmp_path, monkeypatch):
    from rlm.v100 import inference_benchmark, managed_native, mission, supervisor

    folder = tmp_path / "research/mission/run-current"
    folder.mkdir(parents=True)
    model = folder / "existing.gguf"
    model.write_bytes(b"owned model fixture")
    profile = {"runtime": {}, "server": {"model": str(model)}}
    atomic_json(folder / "input-profile.json", profile)
    monkeypatch.setattr(managed_native, "load_profile", lambda *a: profile)
    monkeypatch.setattr(managed_native.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(managed_native.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        mission,
        "status",
        lambda *a: {"running": True, "run": str(folder)} if not stopped else {"running": False},
    )
    stopped = []
    monkeypatch.setattr(mission, "stop", lambda *a: stopped.append(True))
    monkeypatch.setattr(
        inference_benchmark,
        "run",
        lambda *a: (_ for _ in ()).throw(RuntimeError("failed native probe")),
    )
    resumed = []
    monkeypatch.setattr(
        supervisor,
        "install",
        lambda root, path: resumed.append(json.loads(path.read_text())) or {"submitted": True},
    )
    result = managed_native.run(tmp_path, allow_maintenance=True)
    assert result["state"] == "blocked"
    assert resumed == [profile]
    assert "failed native probe" in result["error"]


@pytest.mark.parametrize("regression", [False, True])
def test_native_promotion_requires_retained_case_quality(tmp_path, monkeypatch, regression):
    from contextlib import contextmanager

    from rlm.v100 import competition, evaluation, goals, lineages, managed_native

    folder = tmp_path / "qualification"
    folder.mkdir()
    profile = {"server": {"model": "owned.gguf"}, "runtime": {}}
    baseline = {
        "schema": "v100-quality-v1",
        "suite_sha256": "frozen",
        "generation": {},
        "memory_mode": "fixed",
        "cases": [{"id": "retained", "passed": True}],
    }
    candidate = {**baseline, "cases": [{"id": "retained", "passed": not regression}]}
    reports = iter((baseline, candidate))

    @contextmanager
    def server(*args):
        yield None

    monkeypatch.setattr(competition, "managed_server", server)
    monkeypatch.setattr(competition, "helper_client", lambda *a: None)
    monkeypatch.setattr(evaluation, "evaluate_suite", lambda *a: next(reports))
    monkeypatch.setattr(lineages, "read", lambda *a: ({}, [baseline]))
    monkeypatch.setattr(goals, "load_goal", lambda *a: {})
    path, proof = managed_native.qualify_settings(
        tmp_path,
        profile,
        {
            "state": "measured",
            "winner": "ngram",
            "server": {"spec_type": "ngram-simple"},
            "speedup": 1.2,
        },
        folder,
    )
    assert proof["activated"] is not regression
    assert (path is None) is regression


def test_release_manifest_matches_every_deployment_file():
    base = Path(__file__).parents[1]
    manifest = json.loads((base / "tools/managed-release.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected
    assert manifest["linux_installer"] in manifest["files"]
    assert manifest["windows_prepare"] in manifest["files"]


def test_only_recorded_owned_windows_tailnet_peer_is_considered(tmp_path, monkeypatch):
    import requests

    from rlm.v100 import managed_prepare

    monkeypatch.setattr(
        managed_prepare.subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            {"Peer": {"other": {"HostName": "windows", "TailscaleIPs": ["100.1.2.3"]}}}
        ),
    )
    queried = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **kw):
            queried.append(url)
            raise requests.ConnectionError("offline")

    monkeypatch.setattr(managed_prepare.requests, "Session", Session)
    assert managed_prepare.helper_endpoint(tmp_path) is None
    assert queried == ["http://192.168.0.61:11435/api/tags"]


def test_wikipedia_start_failure_keeps_model_preparation_running(tmp_path, monkeypatch):
    monkeypatch.setattr(
        updater.subprocess,
        "check_output",
        lambda args, **kw: "" if args[0] == "ps" else "0.1.3+v100.84",
    )
    calls = []

    def launch(args, **kwargs):
        calls.append(args)
        if args[0] == "systemd-run":
            raise subprocess.CalledProcessError(1, args)
        atomic_json(tmp_path / "research/managed-update/preparation.json", {"stage": "prepared"})

    monkeypatch.setattr(updater.subprocess, "run", launch)
    result = updater.linux(tmp_path, tmp_path, "a" * 40, {"version": 84})
    assert result["wiki"]["stage"] == "blocked"
    assert any("rlm.v100.managed_prepare" in args for args in calls)
    assert result["stage"] == "package-ready-preparation-partial"


def test_malformed_helper_inventory_is_unavailable_not_task_crash(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rlm.v100 import managed_prepare

    monkeypatch.setattr(managed_prepare.subprocess, "check_output", lambda *a, **k: "{}")

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [])

    monkeypatch.setattr(managed_prepare.requests, "Session", Session)
    assert managed_prepare.helper_endpoint(tmp_path) is None


def test_helper_is_prepared_before_bulk_assets_and_failed_cpu_pilot_is_isolated(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import huggingface_hub

    from rlm.v100 import foundation, managed_native, managed_prepare, model_roles

    ordered = []

    def endpoint(root):
        ordered.append("helper")
        return "http://192.168.0.61:11435"

    def asset(root, native, role, context):
        ordered.append("asset-" + role)
        return {"server": {}}, {"bytes": 1}

    monkeypatch.setattr(managed_prepare, "helper_endpoint", endpoint)
    monkeypatch.setattr(managed_prepare, "prepare", asset)
    monkeypatch.setattr(model_roles, "qualify", lambda root: {"roles": {"fast": {}}})
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(model_info=lambda *a: SimpleNamespace(sha="a" * 40)),
    )
    monkeypatch.setattr(foundation, "propose", lambda *a: {"state": "queued"})
    monkeypatch.setattr(
        managed_prepare,
        "memory_pilot",
        lambda root: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["owned-pilot"])),
    )
    monkeypatch.setattr(managed_native, "run", lambda root: {"state": "measured"})
    result = managed_prepare.run(tmp_path)
    assert ordered == ["helper", "asset-fast", "asset-planner"]
    assert result["memory_pilot"]["state"] == "blocked"
    assert result["native_optimization"]["state"] == "measured"
    assert result["stage"] == "partially-blocked"


def test_planner_identity_probe_does_not_leave_ten_second_generation_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rlm.v100 import competition, income_work, mission_chat, model_roles

    observed = []
    client = SimpleNamespace(
        timeout=900, sampling_args={}, identity=lambda: observed.append(client.timeout)
    )
    monkeypatch.setattr(model_roles, "selected_profile", lambda *a: tmp_path / "admitted.json")
    monkeypatch.setattr(mission_chat, "preferences", lambda root: {"remote_helper_enabled": True})
    monkeypatch.setattr(income_work, "load_profile", lambda *a: {})
    monkeypatch.setattr(competition, "helper_client", lambda *a: client)
    assert income_work.accepted_master(tmp_path, "A") is client
    assert observed == [10]
    assert client.timeout == 120
    assert client.research_token_ceiling == 2048


def test_automatic_native_preparation_does_not_pause_live_master(tmp_path, monkeypatch):
    from rlm.v100 import managed_native

    monkeypatch.setattr(
        managed_native,
        "maintenance",
        lambda root: (_ for _ in ()).throw(AssertionError("mission must not pause")),
    )
    assert managed_native.run(tmp_path)["state"] == "deferred-live-priority"


@pytest.mark.parametrize("model", ["gpt-oss:20b", "gpt-oss:120b"])
def test_operator_exclusion_overrides_previously_admitted_role(tmp_path, model):
    folder = tmp_path / "research/model-roles"
    profile = {"runtime": {"model_name": model}, "resources": {"model_digest": "a" * 64}}
    report = dict(admitted=True, role="planner", model_digest="a" * 64)
    atomic_json(folder / "profile.json", profile)
    atomic_json(folder / "report.json", report)
    entry = dict(
        profile=str(folder / "profile.json"),
        report=str(folder / "report.json"),
        profile_sha256=json_hash(profile),
        report_sha256=json_hash(report),
    )
    atomic_json(folder / "active.json", {"roles": {"planner": entry}})
    assert selected_profile(tmp_path, "planner") is None
    assert json.loads((folder / "report.json").read_text())["admitted"] is True
