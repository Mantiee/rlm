import importlib.util
import json
import os
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
