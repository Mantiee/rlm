from contextlib import contextmanager

import pytest

from rlm.v100 import cli, inference_benchmark, speculative
from rlm.v100.common import atomic_json
from rlm.v100.serving import receipt_path


def test_cli_serve_uses_accepted_manager_and_waits_in_foreground(tmp_path, monkeypatch):
    profile = {"runtime": {"base_url": "http://127.0.0.1:8088"}, "server": {}}
    events = []
    monkeypatch.setattr(cli, "load_profile", lambda *args: profile)
    monkeypatch.setattr(cli, "server_command", lambda selected: ["native-server"])

    @contextmanager
    def server(path, root, log, *, lifecycle_role):
        events.append(("enter", lifecycle_role, log.name))
        yield profile
        events.append(("exit",))

    monkeypatch.setattr("rlm.v100.competition.managed_server", server)
    monkeypatch.setattr(
        cli,
        "wait_for_managed_server",
        lambda actual, root: events.append(("wait",)),
    )
    cli.serve_foreground(tmp_path / "profile.json", tmp_path)
    assert events == [
        ("enter", "accepted", "serve-8088.log"),
        ("wait",),
        ("exit",),
    ]


def test_cli_ctrl_c_unwinds_managed_server_context(tmp_path, monkeypatch):
    profile = {"runtime": {"base_url": "http://127.0.0.1:8088"}, "server": {}}
    events = []
    monkeypatch.setattr(cli, "load_profile", lambda *args: profile)
    monkeypatch.setattr(cli, "server_command", lambda selected: ["native-server"])

    @contextmanager
    def server(path, root, log, *, lifecycle_role):
        events.append(("enter", lifecycle_role))
        try:
            yield profile
        finally:
            events.append(("cleanup-owned-child",))

    monkeypatch.setattr("rlm.v100.competition.managed_server", server)
    monkeypatch.setattr(
        cli,
        "wait_for_managed_server",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        cli.serve_foreground(tmp_path / "profile.json", tmp_path)
    assert events == [("enter", "accepted"), ("cleanup-owned-child",)]


def test_wait_for_managed_server_uses_receipt_process_identity(tmp_path, monkeypatch):
    profile = {"runtime": {"base_url": "http://127.0.0.1:8765"}}
    atomic_json(receipt_path(profile, tmp_path), {"pid": 1234, "process_start": "owned-start"})
    seen = []
    identities = iter(("owned-start", ProcessLookupError()))

    def process_identity(pid):
        seen.append(pid)
        value = next(identities)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("rlm.v100.serving.process_identity", process_identity)
    pauses = []
    cli.wait_for_managed_server(profile, tmp_path, pause=pauses.append)
    assert seen == [1234, 1234]
    assert pauses == [0.25]


def test_mtp_writer_holds_8089_candidate_lease_for_full_run(tmp_path, monkeypatch):
    events = []

    @contextmanager
    def lease(root, profile, *, exclusive, timeout=None):
        events.append(("enter", root, profile["runtime"]["base_url"], exclusive))
        yield True
        events.append(("exit",))

    monkeypatch.setattr("rlm.v100.serving.endpoint_lease", lease)
    monkeypatch.setattr(
        speculative,
        "_test_mtp_unleased",
        lambda root, repeats, draft_tokens: (events.append(("run", repeats, draft_tokens)) or root),
    )
    assert speculative.test_mtp(tmp_path, 2, (2,)) == tmp_path
    assert events == [
        ("enter", tmp_path, "http://127.0.0.1:8089", True),
        ("run", 2, (2,)),
        ("exit",),
    ]


def test_inference_benchmark_writer_leases_selected_port(tmp_path, monkeypatch):
    events = []

    @contextmanager
    def lease(root, profile, *, exclusive, timeout=None):
        events.append(("enter", root, profile["runtime"]["base_url"], exclusive))
        yield True
        events.append(("exit",))

    monkeypatch.setattr("rlm.v100.serving.endpoint_lease", lease)
    monkeypatch.setattr(
        inference_benchmark,
        "_run_unleased",
        lambda profile, output, port: (
            events.append(("run", output, port)) or {"state": "measured"}
        ),
    )
    result = inference_benchmark.run({}, tmp_path / "out", 8097, root=tmp_path)
    assert result["state"] == "measured"
    assert events == [
        ("enter", tmp_path, "http://127.0.0.1:8097", True),
        ("run", tmp_path / "out", 8097),
        ("exit",),
    ]


def test_inference_benchmark_cli_passes_configured_root(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    root = tmp_path / "custom-root"
    seen = []
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference-benchmark",
            "--profile",
            str(profile),
            "--output",
            str(tmp_path / "out"),
            "--port",
            "8097",
            "--root",
            str(root),
        ],
    )
    monkeypatch.setattr(
        inference_benchmark,
        "run",
        lambda selected, output, port, *, root: (
            seen.append((selected, output, port, root)) or {"selection": {"state": "measured"}}
        ),
    )
    inference_benchmark.main()
    assert seen == [({}, tmp_path / "out", 8097, root)]


def test_inference_benchmark_legacy_call_uses_ai_v100_root(tmp_path, monkeypatch):
    events = []

    @contextmanager
    def lease(root, profile, *, exclusive, timeout=None):
        events.append(("enter", root, exclusive))
        yield True
        events.append(("exit",))

    monkeypatch.setenv("AI_V100_ROOT", str(tmp_path))
    monkeypatch.setattr("rlm.v100.serving.endpoint_lease", lease)
    monkeypatch.setattr(
        inference_benchmark,
        "_run_unleased",
        lambda profile, output, port: events.append(("run", port)) or {"state": "measured"},
    )
    assert inference_benchmark.run({}, tmp_path / "out", 8097) == {"state": "measured"}
    assert events == [("enter", tmp_path, True), ("run", 8097), ("exit",)]
