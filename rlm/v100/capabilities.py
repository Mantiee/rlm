"""Host evidence audit, separating mechanisms from observed improvements."""

import json
from pathlib import Path

from rlm.v100.common import atomic_json


def audit(root: Path) -> dict:
    from rlm.v100.mission import status
    from rlm.v100.progress import report
    from rlm.v100.research_policy import settings

    mission, evidence = status(root), report(root)
    run = Path(mission["run"]) if mission.get("run") else None
    paths = (
        [Path(mission["learning"]["live_profile"])]
        if mission.get("learning", {}).get("live_profile")
        else []
    )
    if run:
        paths.append(run / "input-profile.json")
    profile = next((json.loads(p.read_text()) for p in paths if p.exists()), {})
    helper = root / "research/researcher-rtx3090.json"
    helper = json.loads(helper.read_text()) if helper.exists() else {}
    result = {
        "schema": "v100-audit-v1",
        "mission_running": mission["running"],
        "last_cycle": evidence["last_learning_cycle"],
        "accepted_updates_this_run": evidence["accepted_weight_updates_this_run"],
        "training_metrics": evidence["training_metrics"],
        "paper_blockers": evidence["paper_blockers"],
        "audited_paper_outcomes": evidence["available_audited_paper_outcomes"],
        "research_policy": settings(root),
        "user_controls": {
            "chat": "persistent queue; during exclusive training messages wait",
            "directives": "next R&D request",
            "alerts": "local paper proposals, including rejection flag",
        },
        "helper_endpoint": helper.get("runtime", {}).get("base_url"),
        "requested_context": profile.get("runtime", {}).get("context_window"),
        "flash_attention_requested": profile.get("server", {}).get("flash_attention"),
        "kv_cache_requested": profile.get("server", {}).get("cache_type"),
        "mtp_configured": bool(profile.get("server", {}).get("draft_model")),
        "mtp_evidence_attached": bool(profile.get("resources", {}).get("mtp_validation")),
        "persistent_branch_lineages": list(profile.get("resources", {}).get("branch_lineages", {})),
        "hybrid_memory_prepared": (root / "research/mission-semantic.json").exists(),
        "own_code_prepared": (root / "research/self-code-source.json").exists(),
        "implemented": "LoRA SFT, optimizer checkpoints, replay/KL, independent finite gates, bounded CPU architecture pilots, accepted-parent crossbreeding, source drones and sandbox code candidates",
        "not_established": [
            "repeatable income edge",
            "universal zero forgetting",
            "full-context quality at 131072",
            "best possible hardware configuration",
            "Windows crash cause or prevention",
        ],
        "not_implemented": [
            "profit-maximizing policy RL",
            "automatic pretrained Gemma architecture replacement",
            "automatic sports/equities provider integration",
            "automatic account creation/browser model consultation",
            "distributed free Colab workers",
        ],
        "outcome_learning_scope": "Audited retrospective trade accounting including losses; not a proof that the decision policy learned to make money",
        "helper_limit_scope": "30% request active-time target in R&D, not a hard per-process GPU utilization/power cap; game guard pauses only the isolated helper",
        "runtime_scope": "Requested configuration; actual execution requires native logs and on-device measurements",
    }
    atomic_json(root / "research/mission/capabilities.json", result)
    return result
