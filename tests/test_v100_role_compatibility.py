import copy
import json
from http.client import RemoteDisconnected
from urllib.error import HTTPError

import pytest
import requests

from rlm.v100 import competition, managed_bootstrap, model_roles
from rlm.v100.common import atomic_json


@pytest.mark.parametrize("metadata_change", ["none", "ordering", "template"])
def test_dense_planner_migration_keeps_foundation_and_routes_old_workers(
    tmp_path, monkeypatch, metadata_change
):
    info = {
        "template": "pinned template",
        "tokenizer": "pinned tokenizer",
        "parameters": "temperature 0\nnum_ctx 32768",
    }
    metadata = model_roles.metadata_sha(info, model_roles.STABLE_METADATA)
    profile = {
        "runtime": {"model_name": "qwen3.8:27b-q4_K_M", "max_output_tokens": 4096},
        "server": {},
        "resources": {
            "allow_cpu_offload": True,
            "model_digest": "a" * 64,
            "metadata_sha256": metadata,
        },
        "memory": {"chunk_tokens": 1024, "fanout": 2, "retrieve_count": 2},
    }
    original = tmp_path / "research/researcher-rtx3090.json"
    baseline = copy.deepcopy(profile)
    baseline["runtime"]["model_name"] = "qwen3.5:9b-q8_0"
    atomic_json(original, baseline)
    pinned = tmp_path / "research/model-roles/pinned/profile.json"
    atomic_json(pinned, profile)
    atomic_json(
        tmp_path / "research/model-roles/active.json",
        {"roles": {"planner": {"profile": str(pinned)}}},
    )
    if metadata_change == "ordering":
        info["parameters"] = "num_ctx 32768\ntemperature 0"
    elif metadata_change == "template":
        info["template"] = "changed template"
    monkeypatch.setattr(
        model_roles, "selected_profile", lambda root, role: pinned if role == "planner" else None
    )
    monkeypatch.setattr(
        model_roles, "load_profile", lambda path, root: json.loads(path.read_text())
    )

    class Client:
        timeout = 120

        def remote_request(self, endpoint):
            return {"models": [{"name": profile["runtime"]["model_name"], "digest": "a" * 64}]}

        def identity(self):
            return info

    def factory(candidate, root):
        if candidate["runtime"]["model_name"] == "qwen3.8:27b-q4_K_M":
            assert candidate["resources"]["allow_cpu_offload"] is False
        return Client()

    monkeypatch.setattr(competition, "helper_client", factory)
    monkeypatch.setattr(
        model_roles,
        "evaluate",
        lambda client: {"passed": 20, "total": 20, "critical_passed": True, "seconds": 10},
    )
    result = model_roles.qualify(tmp_path)
    if metadata_change == "template":
        assert result["roles"]["planner"]["profile"] == str(pinned)
        assert result["unresolved_compatibility"] == ["planner"]
        assert json.loads(original.read_text()) == baseline
        assert json.loads(pinned.read_text()) == profile
        return
    migrated = json.loads(open(result["roles"]["planner"]["profile"]).read())
    assert migrated["runtime"]["model_name"] == profile["runtime"]["model_name"]
    assert migrated["resources"]["model_digest"] == "a" * 64
    assert migrated["resources"]["metadata_sha256"] == model_roles.metadata_sha(
        info, model_roles.ORDERED_METADATA
    )
    assert json.loads((tmp_path / migrated["resources"]["metadata_snapshot"]).read_text()) == info
    assert migrated["resources"]["allow_cpu_offload"] is False
    assert migrated["runtime"]["context_window"] == 65536
    assert migrated["server"]["context_per_slot"] == 65536
    assert migrated["memory"]["retrieve_count"] <= 4
    assert json.loads(original.read_text()) == migrated
    assert result["legacy_route"]["role"] == "planner"
    assert "fast" not in result["roles"]
    assert json.loads(pinned.read_text()) == profile


def test_legacy_role_pin_recovers_only_parameter_order(tmp_path):
    info = {
        "template": "pinned tokenizer template",
        "parameters": 'temperature 0\nstop "one"\nstop "two"\nnum_ctx 32768',
        "modelfile": 'FROM pinned\nPARAMETER temperature 0\nPARAMETER num_ctx 32768\nTEMPLATE """\nPARAMETER this is template text\n"""\n',
        "modified_at": "before",
    }
    pin = {
        "metadata_hash_scheme": model_roles.STABLE_METADATA,
        "metadata_sha256": model_roles.metadata_sha(info, model_roles.STABLE_METADATA),
    }
    current = dict(info, modified_at="later")
    current["parameters"] = 'num_ctx 32768\nstop "one"\nstop "two"\ntemperature 0'
    current["modelfile"] = info["modelfile"].replace(
        "PARAMETER temperature 0\nPARAMETER num_ctx 32768",
        "PARAMETER num_ctx 32768\nPARAMETER temperature 0",
    )
    assert (
        model_roles.compatible_metadata(tmp_path, pin, current)["state"] == "legacy-pin-reproduced"
    )
    for changed in (
        dict(current, template="different tokenizer template"),
        dict(
            current,
            parameters=current["parameters"].replace(
                'stop "one"\nstop "two"', 'stop "two"\nstop "one"'
            ),
        ),
        dict(
            current,
            modelfile=current["modelfile"].replace("this is template text", "new template text"),
        ),
    ):
        with pytest.raises(ValueError, match="cannot change pinned"):
            model_roles.compatible_metadata(tmp_path, pin, changed)


def test_metadata_migration_requires_verified_snapshot_and_rejects_escape(tmp_path):
    info = {"template": "unchanged", "parameters": "num_ctx 32768\ntemperature 0"}
    previous = dict(info, parameters="temperature 0\nnum_ctx 32768")
    path = tmp_path / "research/model-roles/pinned.show.json"
    atomic_json(path, previous)
    pin = {
        "metadata_snapshot": str(path),
        "metadata_hash_scheme": model_roles.STABLE_METADATA,
        "metadata_sha256": model_roles.metadata_sha(previous, model_roles.STABLE_METADATA),
    }
    assert model_roles.compatible_metadata(tmp_path, pin, info)["state"] == "parameter-order-proven"
    with pytest.raises(ValueError, match="fields=.*template"):
        model_roles.compatible_metadata(tmp_path, pin, dict(info, template="changed"))
    with pytest.raises(ValueError, match="escaped"):
        model_roles.compatible_metadata(
            tmp_path, dict(pin, metadata_snapshot="../foreign.json"), info
        )


def test_role_qualification_has_one_owned_admission_lease(tmp_path, monkeypatch):
    import fcntl

    folder = tmp_path / "research/model-roles"
    folder.mkdir(parents=True)
    monkeypatch.setattr(model_roles, "qualify_locked", lambda root: {"stage": "measured"})
    with (folder / "qualification.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert model_roles.qualify(tmp_path)["stage"] == "already-running"
    assert model_roles.qualify(tmp_path)["stage"] == "measured"


def test_offload_permission_is_architecture_specific():
    assert not model_roles.offload_allowed("qwen3.8:27b-q4_K_M")
    assert model_roles.offload_allowed("gemma4:26b-a4b-it-q4_K_M")


def test_qualification_retries_transport_admission_but_not_wrong_answers(monkeypatch):
    from rlm.v100 import agent

    calls = []
    delays = []
    prompts = {row["prompt"]: row["expected"] for row in model_roles.cases()}

    def reply(client, messages, **kwargs):
        calls.append(messages[0]["content"])
        if len(calls) <= 2:
            response = requests.Response()
            response.status_code = 503
            raise requests.HTTPError("model loading", response=response)
        return {"content": json.dumps({"answer": prompts[messages[0]["content"]]})}

    monkeypatch.setattr(agent, "native_turn", reply)
    monkeypatch.setattr(model_roles.time, "sleep", delays.append)
    result = model_roles.evaluate(type("Client", (), {"timeout": 120})())
    assert result["passed"] == 20
    assert result["cases"][0]["transport_attempts"] == 3
    assert len(calls) == 22 and delays == [1, 2]

    calls.clear()

    def wrong(client, messages, **kwargs):
        calls.append(1)
        return {"content": '{"answer":"wrong"}'}

    monkeypatch.setattr(agent, "native_turn", wrong)
    assert model_roles.evaluate(type("Client", (), {"timeout": 120})())["passed"] == 0
    assert len(calls) == 20


def test_interrupted_metadata_download_retries_without_weakening_validation(monkeypatch):
    calls = []
    delays = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, maximum):
            return b"verified later by manifest"

    def open_url(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise RemoteDisconnected("Remote end closed connection without response")
        return Response()

    monkeypatch.setattr(managed_bootstrap.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(managed_bootstrap.time, "sleep", delays.append)
    assert (
        managed_bootstrap.fetch("https://raw.githubusercontent.com/Mantiee/rlm/test")
        == b"verified later by manifest"
    )
    assert delays == [1, 2]
    assert len(calls) == 3


def test_forbidden_metadata_request_is_not_retried(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.org", 403, "Forbidden", {}, None)

    monkeypatch.setattr(managed_bootstrap.urllib.request, "urlopen", forbidden)
    with pytest.raises(HTTPError):
        managed_bootstrap.fetch("https://raw.githubusercontent.com/Mantiee/rlm/test")
    assert len(calls) == 1
