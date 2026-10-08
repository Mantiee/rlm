"""Audited, outcome-derived action preferences and bounded LoRA DPO loss."""

import json
from decimal import Decimal
from pathlib import Path

from rlm.v100.paper import PaperBook, sha, timestamp


def records(root: Path) -> list[dict]:
    if not (root / "research/paper/ledger.sqlite3").exists():
        return []
    book = PaperBook(root)
    try:
        decisions, rewards = {}, {}
        for event in book.events():
            if event["kind"] == "decision":
                proposal = event["payload"]["proposal"]
                decisions[proposal["id"]] = event
            if event["kind"] == "observation":
                for trade in event["payload"].get("trades", []):
                    if trade["action"] in ("close", "liquidation", "settle"):
                        origin = decisions.get(trade["order_id"])
                        if not origin or origin["sequence"] >= event["sequence"]:
                            raise ValueError("Reward preference lacks a causal ledger decision")
                        pnl = Decimal(trade["payout"]) - Decimal(trade["allocation"])
                        if pnl != Decimal(trade["net_pnl"]):
                            raise ValueError("Reward preference payout disagrees with ledger")
                        entry = rewards.setdefault(
                            trade["order_id"],
                            {
                                "pnl": Decimal(0),
                                "allocation": Decimal(0),
                                "hashes": [],
                                "resolved_at": event["time"],
                            },
                        )
                        entry["pnl"] += pnl
                        entry["allocation"] += Decimal(trade["allocation"])
                        entry["hashes"].append(event["event_hash"])
                        entry["resolved_at"] = event["time"]
        result = []
        state = book.state()
        pending_orders = {
            position["order_id"]
            for portfolio in state["branches"].values()
            for position in portfolio["positions"].values()
        }
        for identity, reward in rewards.items():
            if identity in pending_orders or reward["pnl"] == 0 or reward["allocation"] <= 0:
                continue
            event = decisions[identity]
            proposal = event["payload"]["proposal"]
            inputs = event["payload"].get("decision_inputs")
            if proposal["action"] != "open" or not inputs or not inputs.get("quote"):
                continue
            if timestamp(inputs["quote"]["available_at"]) > timestamp(event["time"]):
                raise ValueError("Future quote in reward preference")
            # The earlier prompt contains ONLY the immutable earlier context, never
            # the later price, settlement, profit or postmortem. Reward labels are
            # hindsight preferences for this sample, not a claim of predictive edge.
            context = {
                "decision_at": event["time"],
                "quote": {
                    key: inputs["quote"][key]
                    for key in ("bid", "ask", "bid_size", "ask_size", "available_at")
                    if key in inputs["quote"]
                },
                "instrument": {
                    key: inputs["instrument"][key]
                    for key in ("market", "product", "cluster", "quantity_step")
                    if key in inputs["instrument"]
                },
                "fees": {
                    key: value
                    for key, value in inputs["fee_profile"].items()
                    if key.endswith("_bps")
                    or key.endswith("commission")
                    or key
                    in (
                        "winnings_tax_threshold",
                        "winnings_tax_basis",
                        "winnings_tax_base",
                        "currency",
                    )
                },
            }
            action = {
                key: proposal[key] for key in ("action", "symbol", "side", "budget", "leverage")
            }
            cash = {"action": "hold", "symbol": "CASH", "side": "long", "budget": 0, "leverage": 1}
            chosen, rejected = (action, cash) if reward["pnl"] > 0 else (cash, action)
            prompt = [
                {
                    "role": "user",
                    "content": "Choose PAPER action versus keeping cash=0 modeled return. Return only the action JSON. Source assumptions and future profitability remain uncertain.\n"
                    + json.dumps(context, sort_keys=True),
                }
            ]
            result.append(
                {
                    "group": "paper-policy-" + sha([identity, reward["hashes"]]),
                    "document_ids": [
                        "paper-order-" + identity,
                        "paper-session-" + reward["resolved_at"][:10] + "-" + proposal["symbol"],
                    ],
                    "messages": prompt
                    + [{"role": "assistant", "content": json.dumps(chosen, sort_keys=True)}],
                    "rejected_messages": prompt
                    + [{"role": "assistant", "content": json.dumps(rejected, sort_keys=True)}],
                    "preference_weight": max(
                        0.25, min(2.0, abs(float(reward["pnl"] / reward["allocation"])) * 10)
                    ),
                    "verification": {
                        "kind": "paper_policy_preference",
                        "accepted": True,
                        "order_id": identity,
                        "outcome_hashes": reward["hashes"],
                        "net_pnl": str(reward["pnl"]),
                        "scope": "Hindsight conditional preference, cash counterfactual=0. Observational bias; must pass independent quality and forward paper tests.",
                    },
                }
            )
        return result
    finally:
        book.close()


def encode(record, tokenizer, max_length):
    from rlm.v100.training import encode_record

    chosen = encode_record(record, tokenizer, max_length)
    preference = record.get("verification", {}).get("kind") == "paper_policy_preference"
    rejected = (
        encode_record({"messages": record["rejected_messages"]}, tokenizer, max_length)
        if preference
        else chosen
    )
    return {
        **chosen,
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "rejected_labels": rejected["labels"],
        "preference_weight": record["preference_weight"] if preference else 0.0,
    }


def collator(tokenizer, base_collator):
    def collate(rows):
        import torch

        selected = [
            {
                key: value
                for key, value in row.items()
                if not key.startswith("rejected_") and key != "preference_weight"
            }
            for row in rows
        ]
        rejected = [
            {
                key.removeprefix("rejected_"): value
                for key, value in row.items()
                if key.startswith("rejected_")
            }
            for row in rows
        ]
        batch = base_collator(selected)
        batch.update({"rejected_" + key: value for key, value in base_collator(rejected).items()})
        batch["preference_weight"] = torch.tensor(
            [row["preference_weight"] for row in rows], dtype=torch.float32
        )
        return batch

    return collate


def log_probabilities(logits, labels):
    import torch

    shifted = labels[:, 1:]
    valid = shifted != -100
    if not valid.any(dim=1).all():
        raise ValueError("Preference has no supervised action tokens")
    # Token chunks avoid a second full sequence x vocabulary FP32 allocation.
    values = []
    for start in range(0, shifted.shape[1], 16):
        target = shifted[:, start : start + 16]
        mask = valid[:, start : start + 16]
        probabilities = torch.nn.functional.log_softmax(
            logits[:, start : start + 16].float(), dim=-1
        )
        values.append(
            probabilities.gather(-1, target.masked_fill(~mask, 0).unsqueeze(-1))
            .squeeze(-1)
            .masked_fill(~mask, 0)
            .sum(-1)
        )
    return sum(values)


def preference_loss(chosen, rejected, reference_chosen, reference_rejected, weights, beta=0.1):
    import torch

    if not 0 < beta <= 1 or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid bounded DPO parameters")
    advantage = (chosen - rejected) - (reference_chosen - reference_rejected)
    return (
        -torch.nn.functional.logsigmoid(beta * advantage) * weights
    ).sum() / weights.count_nonzero().clamp_min(1)


def trainer(base_trainer, predecessor_adapter: bool):
    class OutcomeTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch

            from rlm.v100.distillation import teacher_mode

            inputs = dict(inputs)
            weights = inputs.pop("preference_weight")
            rejected = {
                key.removeprefix("rejected_"): inputs.pop(key)
                for key in list(inputs)
                if key.startswith("rejected_")
            }
            mask = weights > 0
            if not mask.any():
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            chosen_inputs = {key: value[mask] for key, value in inputs.items()}
            rejected_inputs = {key: value[mask] for key, value in rejected.items()}
            with teacher_mode(model, predecessor_adapter), torch.no_grad():
                reference_chosen = log_probabilities(
                    model(**chosen_inputs).logits, chosen_inputs["labels"]
                )
                reference_rejected = log_probabilities(
                    model(**rejected_inputs).logits, rejected_inputs["labels"]
                )
            # Restore the candidate adapter before checkpointed forwards/backward.
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            chosen = log_probabilities(outputs.logits[mask], chosen_inputs["labels"])
            rejected_logits = model(**rejected_inputs).logits
            rejected_logp = log_probabilities(rejected_logits, rejected_inputs["labels"])
            dpo = preference_loss(
                chosen,
                rejected_logp,
                reference_chosen,
                reference_rejected,
                weights[mask].to(chosen.device),
            )
            loss = loss + dpo
            if model.training:
                self.log(
                    {"outcome_dpo_loss": dpo.detach().item(), "preference_pairs": mask.sum().item()}
                )
            return (loss, outputs) if return_outputs else loss

    return OutcomeTrainer
