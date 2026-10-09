"""Pinned alternative pretrained bases, library-only loaders and exclusive trials."""

import copy
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash


def model_class(directory: Path):
    from transformers import AutoModelForCausalLM, Gemma4UnifiedForConditionalGeneration

    configuration = json.loads((directory / "config.json").read_text())
    if configuration.get("auto_map") or configuration.get("text_config", {}).get("auto_map"):
        raise ValueError("Custom remote model code is not imported into the host trainer")
    return (
        Gemma4UnifiedForConditionalGeneration
        if "Gemma4UnifiedForConditionalGeneration" in configuration.get("architectures", [])
        else AutoModelForCausalLM
    )


def propose(root: Path, branch: str, model_id: str, revision: str, rationale: str) -> dict:
    if (
        branch not in ("A", "B")
        or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError(
            "Alternative foundation needs A/B, an HF repository and exact 40-character revision"
        )
    if not isinstance(rationale, str) or not 1 <= len(rationale) <= 1200:
        raise ValueError("Foundation trial needs a bounded falsifiable rationale")
    from rlm.v100.paper import sha

    identity = sha([branch, model_id, revision])[:24]
    folder = root / "research/foundation-trials" / identity
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "proposal.json"
    value = {
        "id": identity,
        "branch": branch,
        "model_id": model_id,
        "revision": revision,
        "rationale": rationale,
        "state": "queued",
        "scope": "Alternate library-supported architecture; isolated LoRA pilot, never arbitrary remote host code or deleting the predecessor",
    }
    if path.exists():
        return json.loads(path.read_text())
    if len(list((root / "research/foundation-trials").glob("*/proposal.json"))) >= 4:
        raise ValueError("Four alternative-base trials already registered")
    atomic_json(path, value)
    return value


def prepare(root: Path, proposal: dict) -> Path:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(
        proposal["model_id"], revision=proposal["revision"], files_metadata=True
    )
    if info.sha != proposal["revision"]:
        raise ValueError("Alternative base revision changed")
    files = [
        f
        for f in info.siblings
        if f.rfilename.endswith((".json", ".safetensors", ".model", ".jinja"))
    ]
    if any(Path(f.rfilename).is_absolute() or ".." in Path(f.rfilename).parts for f in files):
        raise ValueError("Alternative base contains an unsafe file path")
    weights = [f for f in files if f.rfilename.endswith(".safetensors")]
    if (
        not weights
        or any(f.size is None for f in files)
        or sum(f.size for f in weights) > 28 * 2**30
        or sum(f.size for f in files) > 30 * 2**30
    ):
        raise ValueError("Alternative base exceeds 28 GiB safe weight/30 GiB total download budget")
    if shutil.disk_usage(root).free < 160 * 2**30:
        raise ValueError("Alternative trials need 160 GiB free for snapshots and exports")
    destination = root / "models/foundation-trials" / proposal["id"]
    snapshot_download(
        proposal["model_id"],
        revision=proposal["revision"],
        local_dir=destination,
        allow_patterns=[f.rfilename for f in files],
        max_workers=2,
    )
    configuration = json.loads((destination / "config.json").read_text())
    if configuration.get("auto_map") or configuration.get("text_config", {}).get("auto_map"):
        raise ValueError("Alternate base requires custom host code; use the private VM instead")
    atomic_json(
        destination / "v100-source.json",
        {
            "model_id": proposal["model_id"],
            "revision": proposal["revision"],
            "files": {f.rfilename: file_hash(destination / f.rfilename) for f in files},
        },
    )
    return destination


def pending(root: Path) -> list[Path]:
    return [
        p
        for p in sorted((root / "research/foundation-trials").glob("*/proposal.json"))
        if json.loads(p.read_text())["state"] == "queued"
    ]


def trial(
    root: Path, parent: dict, pool: Path, suite: Path, gates: list[dict], proposal_path: Path
) -> dict:
    from rlm.v100.competition import helper_client, managed_server, require_idle_gpu
    from rlm.v100.evaluation import evaluate_suite
    from rlm.v100.protection import compare_reports
    from rlm.v100.serving import assert_served_expert
    from rlm.v100.training import export_candidate

    require_idle_gpu()
    proposal = json.loads(proposal_path.read_text())
    folder = proposal_path.parent
    proposal.update(state="running", started_at=time.time())
    atomic_json(proposal_path, proposal)
    try:
        base = prepare(root, proposal)
        configuration = json.loads((base / "config.json").read_text())
        text = configuration.get("text_config", configuration)
        native_context = text.get("max_position_embeddings", text.get("n_positions", 0))
        if native_context < parent["runtime"]["context_window"]:
            raise ValueError(
                "Alternate base has a shorter native context; existing generation conditions cannot be preserved"
            )
        candidate = copy.deepcopy(parent)
        candidate["training"].update(
            base_model=str(base),
            init_adapter="",
            teacher_adapter="",
            retention_mode=0,
            retention_rank_growth=1,
            retention_reference_groups=[],
            output=str(folder / "training"),
            max_steps=25,
            eval_steps=5,
            save_steps=5,
            microbatch=1,
            gradient_accumulation=4,
            max_length=512,
            precision="nf4",
            rank=8,
        )
        candidate.setdefault("resources", {}).pop("branch_lineages", None)
        candidate["server"]["draft_model"] = ""
        candidate["resources"].pop("mtp_validation", None)
        profile_path = folder / "pilot-profile.json"
        atomic_json(profile_path, candidate)
        with (folder / "training.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "rlm.v100.cli",
                    "--root",
                    str(root),
                    "--profile",
                    str(profile_path),
                    "train",
                    str(pool),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
                check=True,
            )
        export_candidate(candidate, root)
        candidate["server"]["model"] = str(folder / "training/export-Q6_K.gguf")
        candidate["runtime"]["model_version"] = "foundation-" + proposal["id"]
        candidate["training"].update(
            init_adapter=str(folder / "training/candidate"),
            teacher_adapter=str(folder / "training/candidate"),
        )
        # Export names come from the existing native converter, checked before launch.
        if not Path(candidate["server"]["model"]).is_file():
            raise FileNotFoundError(candidate["server"]["model"])
        serving = folder / "serving.json"
        atomic_json(serving, candidate)
        # Architecture promotion always needs a post-freeze fresh gate.
        from rlm.v100 import fresh_audit

        audit = fresh_audit.create(
            root, parent, Path(candidate["server"]["model"]), folder / "fresh-audit"
        )
        fresh_audit.parent_report(root, parent, audit)
        from rlm.v100 import architecture_goal_gate

        goal_suite = architecture_goal_gate.prepare(root, parent, pool, folder)
        with managed_server(serving, root, folder / "server.log"):
            client = helper_client(candidate, root)
            assert_served_expert(client, candidate, root)
            quality = evaluate_suite(client, candidate, suite, folder / "quality.json")
            judgments = [compare_reports(gate, quality) for gate in gates]
            eligible = all(j.get("passed", False) for j in judgments)
            eligible &= architecture_goal_gate.evaluate(root, candidate, goal_suite, folder, parent)
            public = None
            if parent.get("resources", {}).get("public_benchmarks"):
                from rlm.v100.public_benchmarks import compare, evaluate

                public = evaluate(root, candidate, folder / "public-quality.json")
                baseline = Path(parent["resources"]["public_baseline"])
                if file_hash(baseline) != parent["resources"]["public_baseline_sha256"]:
                    raise ValueError("Foundation public baseline changed")
                eligible &= compare(json.loads(baseline.read_text()), public)["passed"]
            if audit is not None:
                fresh_audit.candidate_report(helper_client(candidate), candidate, audit)
                eligible &= fresh_audit.verified_gate(audit, Path(candidate["server"]["model"]))[
                    "passed"
                ]
        if eligible:
            from rlm.v100.protection import ExpertRegistry

            ExpertRegistry(root / "research/experts").register(
                "foundation-" + proposal["id"], candidate, proposal["rationale"], gates, quality
            )
        proposal.update(
            state="completed",
            eligible=eligible,
            goal_gate=str(folder) if goal_suite is not None else None,
            quality=str(folder / "quality.json"),
            serving=str(serving),
            judgments=judgments,
            expert_id="foundation-" + proposal["id"] if eligible else None,
            fresh_audit=str(audit) if audit is not None else None,
            public_quality=str(folder / "public-quality.json") if public is not None else None,
            public_quality_sha256=file_hash(folder / "public-quality.json")
            if public is not None
            else None,
            status="registered independent expert; predecessor retained"
            if eligible
            else "rejected; predecessor retained",
        )
    except Exception as error:
        proposal.update(
            state="failed; predecessor retained",
            error=type(error).__name__,
            detail=str(error)[:600],
        )
    atomic_json(proposal_path, proposal)
    return proposal
