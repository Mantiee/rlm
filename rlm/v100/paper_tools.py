"""Read-only financial tester tools. Hypothetical calculations are never fills."""

import copy
from decimal import Decimal

from rlm.v100.paper import PaperBook, fee_total, number, timestamp, winning_payout
from rlm.v100.paper_reports import summarize


def execute(root, branch: str, name: str, arguments: dict) -> dict:
    book = PaperBook(root)
    try:
        if name == "paper_status":
            if arguments:
                raise ValueError("paper_status takes no arguments")
            state = book.state()
            return {
                "goal": state["goal"],
                "sequence": book.sequence(),
                "portfolio": state["branches"][branch],
                "risk": state["risk"],
                "currency": state["currency"],
            }
        if name == "paper_observed_results":
            if arguments:
                raise ValueError("paper_observed_results takes no arguments")
            report = summarize(book)
            result = {
                key: report[key]
                for key in (
                    "time",
                    "elapsed_days",
                    "branches",
                    "stale_symbols",
                    "expired_fee_profiles",
                    "real_money_ready",
                    "personal_income_tax",
                    "notes",
                )
            }
            result["branches"] = {
                owner: {key: value for key, value in values.items() if key != "positions"}
                for owner, values in report["branches"].items()
            }
            result["notes"] = report["notes"][:2]
            return result
        if name != "paper_test_position" or set(arguments) != {
            "symbol",
            "budget",
            "leverage",
            "side",
            "adverse_bps",
        }:
            raise ValueError("Invalid financial tester arguments")
        state = copy.deepcopy(book.state())
        symbol, side = arguments["symbol"], arguments["side"]
        if symbol not in state["instruments"] or symbol not in state["quotes"]:
            raise ValueError("Unknown observed paper instrument")
        instrument, quote = state["instruments"][symbol], state["quotes"][symbol]
        profile = state["fee_profiles"][instrument["fee_profile"]]
        budget, leverage, adverse = (
            number(arguments[field]) for field in ("budget", "leverage", "adverse_bps")
        )
        if (
            budget <= 0
            or not 1 <= leverage <= number(state["risk"]["max_leverage"])
            or adverse > 10000
            or side not in ("long", "short")
        ):
            raise ValueError("Invalid hypothetical exposure")
        if (
            timestamp(profile["valid_until"]) <= book.clock()
            or (book.clock() - timestamp(quote["observed_at"])).total_seconds()
            > state["risk"]["quote_age_seconds"]
        ):
            raise ValueError("Fee evidence or quote is stale")
        if symbol in state["branches"][branch]["positions"]:
            raise ValueError("Test a new position, not an existing one")
        if instrument["product"] != "isolated-linear" and (leverage != 1 or side != "long"):
            raise ValueError("Product has no isolated short/leverage")
        book.check_budget(state, branch, symbol, budget)
        order = {
            "id": "hypothetical-not-an-order",
            "side": side,
            "leverage": str(leverage),
            "remaining_budget": str(budget),
        }
        trade = book.open_position(
            state,
            branch,
            symbol,
            quote,
            order,
            budget,
            number(quote["ask_size"] if side == "long" else quote["bid_size"]),
        )
        if trade is None:
            return {"status": "not-fillable", "hypothetical": True}
        position = state["branches"][branch]["positions"][symbol]
        if instrument["product"] == "bet":
            effective = number(position["margin"])
            payout = winning_payout(profile, effective, number(position["odds"]))
            return {
                "hypothetical": True,
                "entry": trade,
                "win_net": str(payout - number(position["allocation"])),
                "loss_net": str(-number(position["allocation"])),
                "note": "A hypothetical win/loss, not a probability estimate or settled outcome",
            }
        price = (
            number(quote["bid"] if side == "long" else quote["ask"])
            * (1 - adverse / 10000 if side == "long" else 1 + adverse / 10000)
            * (1 - number(profile["slippage_bps"]) / 10000 * (1 if side == "long" else -1))
        )
        quantity = number(position["quantity"])
        gross = (price - number(position["entry"])) * quantity * (1 if side == "long" else -1)
        maintenance = (
            quantity * price * number(instrument["maintenance_fraction"])
            if instrument["product"] == "isolated-linear"
            else Decimal("0")
        )
        liquidated = instrument["product"] == "isolated-linear" and number(
            position["margin"]
        ) + gross <= maintenance + fee_total(profile, quantity * price, liquidation=True)
        payout = (
            Decimal("0")
            if liquidated
            else max(
                Decimal("0"),
                number(position["margin"]) + gross - fee_total(profile, quantity * price),
            )
        )
        return {
            "hypothetical": True,
            "entry": trade,
            "adverse_bps": str(adverse),
            "liquidated": liquidated,
            "stress_net": str(payout - number(position["allocation"])),
            "note": "Synthetic instantaneous adverse-price scenario; future funding, financing and execution uncertainty are not certified. No ledger mutation or training approval.",
        }
    finally:
        book.close()
