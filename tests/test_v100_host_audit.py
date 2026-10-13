import importlib.util
import json
import os
import sqlite3
from pathlib import Path


def test_planner_snapshot_selection_is_independent_of_newer_node_reports(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/audit-synta-hosts.py"
    spec = importlib.util.spec_from_file_location("host_audit", source)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    role = tmp_path / "research/model-roles/old/qualification.json"
    role.parent.mkdir(parents=True)
    role.write_text(json.dumps({"reports": [{"role": "planner", "metadata_snapshot": "owned"}]}))
    os.utime(role, (1, 1))
    node = tmp_path / "research/owned-node/new.qualification.json"
    node.parent.mkdir(parents=True)
    node.write_text(json.dumps({"admitted": False}))
    assert audit.latest_role_report(tmp_path)["reports"][0]["role"] == "planner"


def test_audit_observes_only_the_fixed_probe_without_mutating_the_queue(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/audit-synta-hosts.py"
    spec = importlib.util.spec_from_file_location("host_audit", source)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    path = tmp_path / "research/state/user-chat.sqlite3"
    path.parent.mkdir(parents=True)
    identity, revision = "a" * 32, "b" * 40
    message = f"What is 19 times 23? Reply with only 437. Do not use tools or change anything. Read-only verification ID: {revision}"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE requests(id TEXT PRIMARY KEY,message TEXT,state TEXT,attempts INTEGER,error TEXT,updated REAL,response TEXT)"
        )
        db.execute(
            "INSERT INTO requests VALUES(?,?,?,0,NULL,1,?)",
            (
                identity,
                message,
                "completed",
                json.dumps({"answer": "437", "actions": [], "responder": {"model": "accepted"}}),
            ),
        )
    before = path.read_bytes()
    result = audit.chat_probe_queue(tmp_path, {"request_id": identity, "revision": revision})
    assert result["state"] == "completed" and result["response"]["answer"] == "437"
    assert path.read_bytes() == before
    with sqlite3.connect(path) as db:
        db.execute("UPDATE requests SET message='private unrelated message'")
    result = audit.chat_probe_queue(tmp_path, {"request_id": identity, "revision": revision})
    assert result["state"] == "rejected" and "response" not in result
