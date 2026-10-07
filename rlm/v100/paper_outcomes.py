"""Audit-checked paper postmortems, including losses, never future decision inputs."""

import json
from decimal import Decimal
from pathlib import Path

from rlm.v100.paper import PaperBook, sha


def records(root: Path) -> list[dict]:
    if not (root / "research/paper/ledger.sqlite3").exists():
        return []
    book = PaperBook(root)
    try:
        events = book.events()  # Verify the immutable hash chain before using labels.
        if not events:
            return []
        decisions = {}
        result = []
        for event in events:
            if event["kind"] == "decision":
                decision = event["payload"]["proposal"]
                decisions[decision["id"]] = (event, decision)
            if event["kind"] != "observation":
                continue
            for index, trade in enumerate(event["payload"].get("trades", [])):
                if trade["action"] not in ("close", "liquidation", "settle"):
                    continue
                origin = decisions.get(trade["order_id"])
                if origin is None or origin[0]["sequence"] >= event["sequence"]:
                    raise ValueError("Paper outcome has no preceding decision")
                pnl = Decimal(trade["payout"]) - Decimal(trade["allocation"])
                if pnl != Decimal(trade["net_pnl"]):
                    raise ValueError("Paper outcome disagrees with payout minus allocation")
                identity = "paper-outcome-" + sha([event["event_hash"], index])
                label = {
                    "net_pnl_before_personal_tax": str(pnl),
                    "result": "profit" if pnl > 0 else "loss" if pnl < 0 else "flat",
                    "scope": "Realized paper outcome; one result does not establish a repeatable edge",
                }
                # Include the later result only in an explicitly retrospective task.
                # Never teach an earlier decision prompt using future quote features.
                decision_inputs = origin[0]["payload"].get("decision_inputs")
                context = {"snapshot": "not captured in this older ledger event"}
                if decision_inputs:
                    quote = decision_inputs.get("quote") or {}
                    context = {
                        "ledger_context_sha256": sha(decision_inputs),
                        "quote": {
                            key: quote[key]
                            for key in ("bid", "ask", "observed_at", "available_at")
                            if key in quote
                        },
                        "fee_profile_sha256": sha(decision_inputs.get("fee_profile")),
                        "instrument": {
                            key: decision_inputs["instrument"][key]
                            for key in ("market", "product", "price_basis")
                            if key in decision_inputs["instrument"]
                        },
                    }
                inputs = {
                    "decision": {
                        key: origin[1][key]
                        for key in ("action", "symbol", "side", "budget", "leverage")
                        if key in origin[1]
                    },
                    "rationale": origin[1]["rationale"][:240],
                    "decision_inputs": context,
                    "decision_time": origin[0]["time"],
                    "outcome_time": event["time"],
                    "resolved_fill": {
                        key: value for key, value in trade.items() if key != "net_pnl"
                    },
                }
                result.append(
                    {
                        "group": identity,
                        "document_ids": [
                            "paper-order-" + trade["order_id"],
                            "paper-session-" + event["time"][:10] + "-" + trade["symbol"],
                        ],
                        "messages": [
                            {
                                "role": "user",
                                "content": "Retrospective paper trade review, NOT a prediction. Calculate the realized net outcome including allocation and all modeled costs. Return JSON.\n"
                                + json.dumps(inputs, sort_keys=True),
                            },
                            {"role": "assistant", "content": json.dumps(label, sort_keys=True)},
                        ],
                        "verification": {
                            "kind": "paper_outcome",
                            "accepted": True,
                            "event_hash": event["event_hash"],
                            "fill_index": index,
                        },
                    }
                )
        return result
    finally:
        book.close()


def verify(record: dict, root: Path) -> None:
    matching = next((row for row in records(root) if row["group"] == record["group"]), None)
    if matching is None or record != matching:
        raise ValueError("Paper training label differs from the audited realized outcome")
