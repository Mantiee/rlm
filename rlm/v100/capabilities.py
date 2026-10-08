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
        "mission_evidence": evidence["mission_evidence"],
        "paper_blockers": evidence["paper_blockers"],
        "audited_paper_outcomes": evidence["available_audited_paper_outcomes"],
        "research_policy": settings(root),
        "user_controls": {
            "chat": "persistent queue; accepted V100 while available, explicitly labeled RTX delegate during exclusive training",
            "long_term_goal": "clear current operator natural-language request or /goal or /cel; no autonomous model changes",
            "short_mid_plans": "versioned plans, model or operator can revise",
            "directives": "next R&D request",
            "alerts": "local paper proposals, including rejection flag",
        },
        "helper_endpoint": helper.get("runtime", {}).get("base_url"),
        "helper_active_time_target_percent": helper.get("resources", {}).get("helper_duty_percent"),
        "requested_context": profile.get("runtime", {}).get("context_window"),
        "flash_attention_requested": profile.get("server", {}).get("flash_attention"),
        "kv_cache_requested": profile.get("server", {}).get("cache_type"),
        "mtp_configured": bool(profile.get("server", {}).get("draft_model")),
        "mtp_evidence_attached": bool(profile.get("resources", {}).get("mtp_validation")),
        "persistent_branch_lineages": list(profile.get("resources", {}).get("branch_lineages", {})),
        "hybrid_memory_prepared": (root / "research/mission-semantic.json").exists(),
        "own_code_prepared": (root / "research/self-code-source.json").exists(),
        "desktop_prepared": (root / "research/desktop/manifest.json").exists(),
        "public_benchmarks_prepared": (root / "research/public-benchmarks/current.json").exists(),
        "resident_workers": evidence["drones"],
        "implemented": "LoRA SFT and outcome-derived DPO preferences, optimizer checkpoints, replay/KL, fixed/public plus one-use fresh gates, new verified curriculum tool, bounded reward shadow head, automatic library-supported or isolated custom full-weight master architecture activation/rollback, owned distributed CPU training workers, source drones and sandbox/guest browser code candidates",
        "fresh_audit_required": profile.get("resources", {}).get("fresh_audit_required", False),
        "paper_research_enabled": profile.get("resources", {}).get("paper_research_enabled", False),
        "foundation_trials": [
            json.loads(p.read_text())
            for p in sorted((root / "research/foundation-trials").glob("*/proposal.json"))[:4]
        ],
        "normalized_provider_mappings": len(list((root / "research/providers").glob("*.json"))),
        "free_market_adapters": [
            json.loads(p.read_text())
            for p in sorted((root / "research/market-adapters").glob("*.json"))
            if p.name != "quota.json"
        ],
        "master_architecture": {
            "automatic_activation": True,
            "backend": profile.get("runtime", {}).get("backend", "llama.cpp"),
            "custom_scratch_scope": "Arbitrary build(config) network, full-weight warm-start children, isolated byte-token inference and the same fixed/fresh/public gates. Not an automatic universal GGUF converter; inference/training may fail their resource/time limits.",
            "active_expert": profile.get("resources", {}).get("active_foundation_expert"),
            "history": profile.get("resources", {}).get("architecture_history", []),
            "gate": "Task improvement, no retained regressions, fresh audit, public gate when configured and actual inference probe; predecessor kept",
        },
        "owned_external_compute": evidence["external_compute"],
        "colab": {
            "interactive_trials": "Fresh datasets, bounded training, safe tensors and independent local validation",
            "distributed_free_workers": "Not permitted by free managed Colab rules",
            "source": "https://research.google.com/colaboratory/faq.html",
        },
        "not_established": [
            "repeatable income edge",
            "universal zero forgetting",
            "full-context quality at 131072",
            "best possible hardware configuration",
            "Windows crash cause or prevention",
        ],
        "not_implemented": [
            "profit-maximizing policy RL",
            "hard per-helper GPU utilization or power cap on Windows WDDM",
            "accepted-master chat inference during exclusive master training on the single V100; RTX delegation is used instead",
            "autonomous distributed free managed Colab workers",
            "automatic certification of sports/equities provider fees, account rights and settlement semantics",
            "automatic account creation or bypassing browser authentication/quotas",
        ],
        "outcome_learning_scope": "Causal earlier-input preferences from fully resolved audited paper trades enable LoRA DPO, including losses; observational hindsight labels and CPU shadow reward optimization are not unbiased policy RL or proof of profit",
        "helper_limit_scope": "Configured request active-time target in R&D, not a hard per-process GPU utilization/power cap; game guard pauses only the isolated helper",
        "runtime_scope": "Requested configuration; actual execution requires native logs and on-device measurements",
    }
    atomic_json(root / "research/mission/capabilities.json", result)
    return result
