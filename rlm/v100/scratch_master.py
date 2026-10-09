"""Custom full-weight masters run only in the resource-limited architecture sandbox.

This is a byte-token backend, not a GGUF converter. Loading a candidate never
imports its Python into the controller. Model selection still uses host gates.
"""

import copy
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from rlm.clients.llamacpp import LlamaCppClient
from rlm.v100.architectures import (
    candidate_path,
    phase,
    run_candidate,
    validate_budget,
    verify_candidate,
)
from rlm.v100.common import atomic_json
from rlm.v100.protection import ExpertRegistry, compare_reports, file_hash


def is_scratch(profile: dict) -> bool:
    return profile["runtime"].get("backend") == "isolated-architecture"


def verify(profile: dict) -> dict:
    resources = profile["resources"]
    for name, path in (
        ("source", resources["scratch_source"]),
        ("budget", resources["scratch_budget"]),
        ("weights", profile["server"]["model"]),
        ("runner", profile["server"]["binary"]),
    ):
        artifact = Path(path)
        if artifact.is_symlink() or file_hash(artifact) != resources["scratch_hashes"][name]:
            raise ValueError("Frozen scratch " + name + " changed")
    runner = Path(__file__).with_name("architecture_worker.py")
    if file_hash(runner) != resources["scratch_hashes"]["runner"]:
        raise ValueError("Scratch runner requires revalidation after a controller update")
    budget = json.loads(Path(resources["scratch_budget"]).read_text())
    validate_budget(budget)
    if budget["context_window"] < profile["runtime"]["context_window"]:
        raise ValueError("Scratch context cannot satisfy the selected profile")
    return budget


class ScratchClient(LlamaCppClient):
    """Reuse controller chat/usage protocol with an isolated local transport."""

    def __init__(self, *, scratch_profile: dict, **kwargs):
        self.scratch_profile = copy.deepcopy(scratch_profile)
        self.scratch_root = Path(scratch_profile["resources"]["scratch_root"])
        super().__init__(**kwargs)

    def http_request(self, path: str, payload=None):
        from rlm.v100.architecture_worker import prompt_bytes

        if path == "/health":
            verify(self.scratch_profile)
            return {"status": "ok"}
        if path == "/props":
            return {"model_path": self.scratch_profile["server"]["model"]}
        if path == "/tokenize":
            return {"tokens": list(payload["content"].encode("utf-8"))}
        if path == "/apply-template":
            return {"prompt": bytes(prompt_bytes(payload["messages"])[1:]).decode()}
        if path != "/v1/chat/completions":
            raise ValueError("Unsupported scratch transport operation")
        if payload.get("tools"):
            raise ValueError("Scratch tools require the controller JSON tool protocol")
        budget = verify(self.scratch_profile)
        from rlm.v100.researchers import available_ram_gib

        if available_ram_gib() < budget["ram_gib"] + 2:
            raise RuntimeError("Scratch turn deferred: insufficient free host RAM")
        budget["max_new_tokens"] = min(payload["max_tokens"], budget["max_new_tokens"])
        budget["timeout"] = min(budget["timeout"], int(self.timeout))
        messages = payload["messages"]
        if not isinstance(messages, list) or any(
            set(m) - {"role", "content"} or not isinstance(m.get("content"), str) for m in messages
        ):
            raise ValueError("Scratch backend supports bounded text messages")
        prompt_count = len(prompt_bytes(messages))
        if prompt_count + budget["max_new_tokens"] > budget["context_window"]:
            raise ValueError("Scratch byte prompt and output exceed context")
        folder = self.scratch_root / "research/scratch-runtime" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        try:
            (folder / "source").mkdir()
            (folder / "checks").mkdir()
            shutil.copyfile(
                self.scratch_profile["resources"]["scratch_source"], folder / "source/model.py"
            )
            shutil.copyfile(self.scratch_profile["server"]["binary"], folder / "checks/runner.py")
            atomic_json(
                folder / "architecture.json",
                {
                    "schema": "v100-architecture-v1",
                    "code_sha256": file_hash(folder / "source/model.py"),
                    "runner_sha256": file_hash(folder / "checks/runner.py"),
                },
            )
            run = folder / "run"
            (run / "weights").mkdir(parents=True)
            # Read-only mount and immutable hash guard; no huge copy on every turn.
            os.link(self.scratch_profile["server"]["model"], run / "weights/weights.safetensors")
            with tempfile.TemporaryDirectory(prefix="scratch-input-") as temporary:
                inputs = Path(temporary)
                atomic_json(inputs / "config.json", budget)
                atomic_json(
                    inputs / "data.json",
                    [
                        {
                            "id": "turn",
                            "messages": messages,
                            "sampling": {
                                k: payload[k]
                                for k in ("temperature", "top_p", "top_k", "seed")
                                if k in payload
                            },
                        }
                    ],
                )
                from rlm.v100.architectures import resource_lease

                with resource_lease(self.scratch_root, budget["device"]):
                    phase(folder, run, inputs, budget, "infer")
            verify(self.scratch_profile)
            log = run / "infer.log"
            if log.stat().st_size > 2**20:
                raise ValueError("Scratch response exceeds protocol budget")
            rows = [
                json.loads(line)
                for line in log.read_text().splitlines()
                if line.startswith('{"id":')
            ]
            if len(rows) != 1 or rows[0].get("id") != "turn":
                raise ValueError("Scratch worker returned an invalid turn")
            row = rows[0]
            if (
                set(row) != {"id", "answer", "completion_tokens", "finish_reason"}
                or not isinstance(row["answer"], str)
                or len(row["answer"].encode()) > budget["max_new_tokens"] * 3
                or type(row["completion_tokens"]) is not int
                or not 0 <= row["completion_tokens"] <= budget["max_new_tokens"]
                or row["finish_reason"] not in ("stop", "length")
            ):
                raise ValueError("Scratch worker output failed host validation")
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": row["answer"]},
                        "finish_reason": row["finish_reason"],
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_count,
                    "completion_tokens": row["completion_tokens"],
                    "total_tokens": prompt_count + row["completion_tokens"],
                },
            }
        finally:
            shutil.rmtree(folder)


def propose(root: Path, branch: str, candidate_id: str, budget: dict) -> dict:
    if branch not in ("A", "B"):
        raise ValueError("Scratch master proposal needs branch A or B")
    validate_budget(budget)
    manifest = verify_candidate(candidate_path(root, candidate_id))
    if manifest["owner"] not in (branch, "shared"):
        raise ValueError("Scratch candidate belongs to another branch")
    folder = root / "research/scratch-master-trials"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (candidate_id + ".json")
    value = {
        "candidate_id": candidate_id,
        "branch": branch,
        "budget": budget,
        "state": "queued",
        "created_at": time.time(),
    }
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["budget"] != budget or previous["branch"] != branch:
            raise ValueError("Scratch proposal already frozen with different settings")
        return previous
    if len(pending(root)) >= 4:
        raise ValueError("At most four queued scratch master trials")
    atomic_json(path, value)
    return value


def pending(root: Path) -> list[Path]:
    from rlm.v100.serving import process_identity

    result = []
    for path in sorted((root / "research/scratch-master-trials").glob("*.json")):
        proposal = json.loads(path.read_text())
        if proposal["state"] == "running":
            try:
                owned = process_identity(proposal["pid"]) == proposal["process_start"]
            except (OSError, KeyError):
                owned = False
            if not owned:
                # A partially exposed audit must not become another selection attempt.
                proposal.update(state="interrupted; predecessor retained", eligible=False)
                atomic_json(path, proposal)
        if proposal["state"] == "queued":
            result.append(path)
    return result


def profile_for(root: Path, parent: dict, candidate_id: str, budget: dict) -> dict:
    folder = candidate_path(root, candidate_id)
    verify_candidate(folder)
    candidate = copy.deepcopy(parent)
    candidate["runtime"]["backend"] = "isolated-architecture"
    candidate["runtime"]["model_version"] = "scratch-" + candidate_id
    # Exact controller prompts and sampling remain comparable. Custom models
    # use UTF-8 bytes; throughput is never reported as comparable BPE tokens.
    if parent["runtime"].get("tool_protocol") != "json":
        raise ValueError("Scratch promotion requires a JSON-tool parent baseline")
    candidate["server"].update(
        model=str(folder / "trial/weights/weights.safetensors"),
        binary=str(folder / "checks/runner.py"),
        draft_model="",
        draft_tokens=0,
    )
    candidate["training"].update(
        init_adapter="", teacher_adapter="", output=str(folder / "training")
    )
    resources = candidate.setdefault("resources", {})
    resources.pop("branch_lineages", None)
    resources.pop("mtp_validation", None)
    resources.update(
        device=budget["device"],
        scratch_root=str(root.resolve()),
        scratch_source=str(folder / "source/model.py"),
        scratch_budget=str(folder / "trial/budget.json"),
        scratch_hashes={
            "source": file_hash(folder / "source/model.py"),
            "runner": file_hash(folder / "checks/runner.py"),
            "budget": file_hash(folder / "trial/budget.json"),
            "weights": file_hash(folder / "trial/weights/weights.safetensors"),
        },
        token_unit="UTF-8 bytes; not directly comparable to native BPE throughput",
    )
    verify(candidate)
    return candidate


def trial(root: Path, parent: dict, pool: Path, suite: Path, gates: list[dict], path: Path) -> dict:
    from rlm.v100 import fresh_audit
    from rlm.v100.architecture_promotion import complete
    from rlm.v100.competition import helper_client, managed_server
    from rlm.v100.evaluation import evaluate_suite

    proposal = json.loads(path.read_text())
    from rlm.v100.serving import process_identity

    proposal.update(state="running", pid=os.getpid(), process_start=process_identity(os.getpid()))
    atomic_json(path, proposal)
    folder = candidate_path(root, proposal["candidate_id"])
    evidence = folder / "master-gates"
    try:
        evidence.mkdir(exist_ok=False)
        budget = proposal["budget"]
        if (
            budget["context_window"] < parent["runtime"]["context_window"]
            or budget["max_new_tokens"] < parent["runtime"]["max_output_tokens"]
        ):
            raise ValueError(
                "Scratch budget cannot preserve the parent's context/output conditions"
            )
        if not (folder / "trial").exists():
            initial = None
            if (
                is_scratch(parent)
                and file_hash(folder / "source/model.py")
                == parent["resources"]["scratch_hashes"]["source"]
            ):
                verify(parent)
                initial = Path(parent["server"]["model"])
            run_candidate(root, proposal["candidate_id"], pool, suite, budget, initial)
        stored = json.loads((folder / "trial/budget.json").read_text())
        if stored != budget:
            raise ValueError("Trained scratch budget differs from its frozen proposal")
        candidate = profile_for(root, parent, proposal["candidate_id"], budget)
        serving = evidence / "serving.json"
        atomic_json(serving, candidate)
        audit = fresh_audit.create(
            root, parent, Path(candidate["server"]["model"]), evidence / "fresh-audit"
        )
        fresh_audit.parent_report(root, parent, audit)
        from rlm.v100 import architecture_goal_gate

        goal_suite = architecture_goal_gate.prepare(root, parent, pool, evidence)
        with managed_server(serving, root, evidence / "server.log"):
            quality = evaluate_suite(
                helper_client(candidate), candidate, suite, evidence / "quality.json"
            )
            judgments = [compare_reports(g, quality) for g in gates]
            eligible = complete(quality) and all(g["passed"] for g in judgments)
            eligible &= architecture_goal_gate.evaluate(
                root, candidate, goal_suite, evidence, parent
            )
            public = None
            if parent.get("resources", {}).get("public_benchmarks"):
                from rlm.v100.public_benchmarks import compare, evaluate

                public = evaluate(root, candidate, evidence / "public-quality.json")
                baseline = Path(parent["resources"]["public_baseline"])
                if file_hash(baseline) != parent["resources"]["public_baseline_sha256"]:
                    raise ValueError("Official baseline changed")
                eligible &= compare(json.loads(baseline.read_text()), public)["passed"]
            fresh_audit.candidate_report(helper_client(candidate), candidate, audit)
            eligible &= fresh_audit.verified_gate(audit, Path(candidate["server"]["model"]))[
                "passed"
            ]
        expert_id = "scratch-" + proposal["candidate_id"]
        if eligible:
            ExpertRegistry(root / "research/experts").register(
                expert_id, candidate, "Isolated full-weight scratch master", gates, quality
            )
        proposal.update(
            state="completed",
            eligible=eligible,
            goal_gate=str(evidence) if goal_suite is not None else None,
            expert_id=expert_id if eligible else None,
            fresh_audit=str(audit),
            public_quality=str(evidence / "public-quality.json") if public else None,
            public_quality_sha256=file_hash(evidence / "public-quality.json") if public else None,
            quality=str(evidence / "quality.json"),
            judgments=judgments,
        )
    except (ValueError, RuntimeError, OSError, TimeoutError) as error:
        proposal.update(
            state="failed; predecessor retained",
            eligible=False,
            error=type(error).__name__,
            detail=str(error)[:600],
        )
    atomic_json(path, proposal)
    return proposal


def continuations(root: Path, profile: dict) -> None:
    """Queue A/B full-weight children only when genuinely new examples arrive."""
    from rlm.v100.architectures import create_candidate

    budget = verify(profile)
    code = Path(profile["resources"]["scratch_source"]).read_text()
    for branch in ("A", "B"):
        identity = "continue-" + branch + "-" + uuid.uuid4().hex[:20]
        create_candidate(
            root,
            branch,
            identity,
            code,
            "Full-weight continuation on new verified replay, followed by independent retention gates",
        )
        settings = {**budget, "seed": int(uuid.uuid4().hex[:7], 16)}
        propose(root, branch, identity, settings)
