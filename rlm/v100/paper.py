"""Forward-only paper portfolios. No broker credentials or real order endpoints."""

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

from rlm.v100.activity import ActivityLog

D = Decimal
ZERO = D("0")
GOAL = (
    "Compare sports research, crypto and equities using forward-only paper decisions. "
    "Maximize repeatable net return within fixed exposure/drawdown limits, including all "
    "documented costs. Research public social media, quarterly filings, news, fundamentals "
    "and counterarguments. No future data, edited past decisions, paid services or real orders. "
    "Remaining in cash is an acceptable decision; one profitable trial is not proof of an edge."
)
RISK = {
    "position_fraction": "0.05",
    "cluster_fraction": "0.15",
    "market_fraction": "0.20",
    "total_fraction": "0.30",
    "drawdown_pause": "0.10",
    "max_leverage": "3",
    "quote_age_seconds": 300,
    "order_ttl_seconds": 300,
}
FEE_FIELDS = (
    "entry_commission_bps",
    "entry_minimum_commission",
    "entry_venue_fee_bps",
    "entry_fx_bps",
    "exit_commission_bps",
    "exit_minimum_commission",
    "exit_venue_fee_bps",
    "exit_fx_bps",
    "slippage_bps",
    "financing_annual_bps",
    "liquidation_bps",
    "stake_tax_bps",
    "winnings_tax_bps",
    "winnings_tax_threshold",
    "winnings_commission_bps",
)


def utcnow() -> datetime:
    return datetime.now(UTC)


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps require an explicit timezone")
    return result.astimezone(UTC)


def number(value, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("Missing or invalid numeric value; unknown costs are not zero")
    try:
        result = D(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid numeric value") from error
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError("Expected a finite nonnegative number")
    return result


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def sha(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", value):
        raise ValueError("Invalid paper identifier")
    return value


def source_fields(value: dict, now: datetime) -> None:
    url = urlparse(value.get("source_url", ""))
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("Evidence requires a public HTTPS source URL")
    if not re.fullmatch(r"[a-f0-9]{64}", value.get("source_sha256", "")):
        raise ValueError("Evidence requires a source snapshot SHA256")
    if timestamp(value["available_at"]) > now:
        raise ValueError("Future information cannot enter the paper lab")


def fees(
    profile: dict, notional: Decimal, *, liquidation: bool = False, closing: bool = True
) -> dict[str, Decimal]:
    prefix = "exit_" if closing else "entry_"
    return {
        "commission": max(
            notional * number(profile[prefix + "commission_bps"]) / 10000,
            number(profile[prefix + "minimum_commission"]),
        ),
        "venue": notional * number(profile[prefix + "venue_fee_bps"]) / 10000,
        "fx": notional * number(profile[prefix + "fx_bps"]) / 10000,
        "liquidation": notional * number(profile["liquidation_bps"]) / 10000
        if liquidation
        else ZERO,
    }


def fee_total(
    profile: dict, notional: Decimal, *, liquidation: bool = False, closing: bool = True
) -> Decimal:
    return sum(fees(profile, notional, liquidation=liquidation, closing=closing).values(), ZERO)


def winning_payout(profile: dict, effective: Decimal, odds: Decimal) -> Decimal:
    gross = effective * odds
    commission = (gross - effective) * number(profile["winnings_commission_bps"]) / 10000
    payout = gross - commission
    base = gross if profile["winnings_tax_base"] == "gross-payout" else payout
    threshold = number(profile["winnings_tax_threshold"])
    if base > threshold:
        taxable = base if profile["winnings_tax_basis"] == "whole-payout" else base - threshold
        payout -= taxable * number(profile["winnings_tax_bps"]) / 10000
    return max(ZERO, payout)


class PaperBook:
    def __init__(self, root: Path, clock=utcnow):
        self.root, self.clock = root, clock
        directory = root / "research/paper"
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "ledger.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT);
            CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY, time TEXT,
                kind TEXT, branch TEXT, payload TEXT, previous_hash TEXT, event_hash TEXT);
            CREATE TRIGGER IF NOT EXISTS immutable_events_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT,'Paper events are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_events_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT,'Paper events are immutable'); END;
        """)

    def close(self) -> None:
        self.db.close()

    def state(self) -> dict:
        row = self.db.execute(
            "SELECT payload,(SELECT payload FROM events ORDER BY sequence DESC LIMIT 1) AS last_payload FROM state WHERE id=1"
        ).fetchone()
        if row is None:
            raise ValueError("Initialize the paper lab first")
        value = json.loads(row["payload"])
        if row["last_payload"] is None or json.loads(row["last_payload"])["state_sha256"] != sha(
            value
        ):
            raise ValueError("Paper state changed outside its ledger")
        self.events()
        return value

    def sequence(self) -> int:
        return self.db.execute("SELECT COALESCE(MAX(sequence),0) FROM events").fetchone()[0]

    def record(self, state: dict, kind: str, payload: dict, branch: str = "controller") -> dict:
        now = self.clock().isoformat()
        last = self.db.execute(
            "SELECT event_hash,time FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if last is not None and timestamp(now) < timestamp(last["time"]):
            raise ValueError("Controller clock moved backwards")
        previous = last["event_hash"] if last else "0" * 64
        value = {**payload, "state_sha256": sha(state)}
        digest = sha(
            {
                "time": now,
                "kind": kind,
                "branch": branch,
                "payload": value,
                "previous_hash": previous,
            }
        )
        cursor = self.db.execute(
            "INSERT INTO events(time,kind,branch,payload,previous_hash,event_hash) VALUES(?,?,?,?,?,?)",
            (now, kind, branch, canonical(value), previous, digest),
        )
        self.db.execute(
            "INSERT INTO state VALUES(1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
            (canonical(state),),
        )
        return {
            "sequence": cursor.lastrowid,
            "time": now,
            "kind": kind,
            "branch": branch,
            "payload": value,
            "event_hash": digest,
        }

    def events(self) -> list[dict]:
        result, previous = [], "0" * 64
        for row in self.db.execute("SELECT * FROM events ORDER BY sequence"):
            event = dict(row)
            event["payload"] = json.loads(event["payload"])
            body = {
                key: event[key] for key in ("time", "kind", "branch", "payload", "previous_hash")
            }
            if event["previous_hash"] != previous or sha(body) != event["event_hash"]:
                raise ValueError("Paper audit chain changed")
            previous = event["event_hash"]
            result.append(event)
        return result

    def note(self, branch: str, payload: dict) -> dict:
        if branch not in ("A", "B", "controller"):
            raise ValueError("Unknown paper branch")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            return self.record(self.state(), "research", payload, branch)

    def initialize(self, capital="10000", currency="PLN") -> dict:
        capital = number(capital, positive=True)
        if not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError("Use an ISO currency code")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.sequence():
                raise FileExistsError("Paper experiment already exists; no bankroll reset")
            state = {
                "schema": "v100-paper-v1",
                "goal": GOAL,
                "risk": RISK,
                "currency": currency,
                "initial_capital": str(capital),
                "fee_profiles": {},
                "instruments": {},
                "quotes": {},
                "news": [],
                "orders": [],
                "outcomes": {},
                "branches": {
                    branch: {
                        "cash": str(capital),
                        "positions": {},
                        "peak": str(capital),
                        "paused": False,
                        "costs": "0",
                    }
                    for branch in ("A", "B")
                },
            }
            self.record(
                state, "initialize", {"capital": str(capital), "currency": currency, "goal": GOAL}
            )
        return state

    def configure(self, configuration: dict) -> dict:
        """Operator-owned inputs. Models cannot register fees or executable instruments."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            state = self.state()
            for profile in configuration.get("fee_profiles", []):
                key = identifier(profile["id"])
                if key in state["fee_profiles"]:
                    raise FileExistsError("Fee profiles are versioned; use a new ID")
                source_fields(profile, self.clock())
                if (
                    profile.get("operator_verified") is not True
                    or profile["currency"] != state["currency"]
                ):
                    raise ValueError("Fees need operator verification and the portfolio currency")
                if timestamp(profile["valid_until"]) <= self.clock():
                    raise ValueError("Expired fee evidence")
                for field in FEE_FIELDS:
                    number(profile.get(field))
                if number(profile["slippage_bps"]) >= 10000:
                    raise ValueError("Execution slippage must be below 100%")
                for prefix in ("entry_", "exit_"):
                    if (
                        sum(
                            (
                                number(profile[prefix + field])
                                for field in ("commission_bps", "venue_fee_bps", "fx_bps")
                            ),
                            ZERO,
                        )
                        >= 10000
                    ):
                        raise ValueError("Combined proportional trading charges must be below 100%")
                if any(
                    number(profile[field]) >= 10000
                    for field in ("stake_tax_bps", "winnings_tax_bps", "winnings_commission_bps")
                ):
                    raise ValueError("Invalid sports deductions")
                if type(profile.get("void_refunds_stake_tax")) is not bool:
                    raise ValueError("Specify the documented void-refund rule")
                if type(profile.get("void_refunds_entry_fees")) is not bool:
                    raise ValueError("Specify whether voids refund entry fees")
                if profile.get("winnings_tax_basis") not in ("whole-payout", "excess-only"):
                    raise ValueError("Specify the thresholded winnings-tax basis")
                if profile.get("winnings_tax_base") not in ("gross-payout", "after-commission"):
                    raise ValueError("Specify the winnings-tax base before/after commission")
                state["fee_profiles"][key] = profile
            for instrument in configuration.get("instruments", []):
                symbol = identifier(instrument["symbol"])
                if symbol in state["instruments"]:
                    raise FileExistsError("Instrument definitions cannot be silently rewritten")
                if instrument["market"] not in ("equities", "crypto", "sports") or instrument[
                    "product"
                ] not in ("spot", "isolated-linear", "bet"):
                    raise ValueError("Unsupported paper product")
                if (instrument["market"] == "sports") != (instrument["product"] == "bet"):
                    raise ValueError("Sports require a fixed-odds bet product")
                if instrument["fee_profile"] not in state["fee_profiles"]:
                    raise ValueError("Unknown fee profile")
                identifier(instrument["cluster"])
                identifier(instrument["feed_id"])
                number(instrument["quantity_step"], positive=True)
                source_fields(instrument, self.clock())
                if instrument.get("operator_verified") is not True:
                    raise ValueError("Instrument execution rules need operator verification")
                if instrument.get("price_basis") != "raw-unadjusted":
                    raise ValueError(
                        "Use raw quotes and explicit corporate actions, not future-adjusted prices"
                    )
                if instrument["product"] == "spot" and number(
                    state["fee_profiles"][instrument["fee_profile"]]["financing_annual_bps"]
                ):
                    raise ValueError("Fully paid spot positions have no modeled margin borrowing")
                if instrument["product"] == "isolated-linear":
                    if (
                        instrument.get("loss_cap") != "position"
                        or instrument.get("auto_add_margin") is not False
                    ):
                        raise ValueError(
                            "Only documented position-isolated loss caps without auto margin are supported"
                        )
                    if not ZERO < number(instrument["maintenance_fraction"]) < 1:
                        raise ValueError("Missing maintenance margin rule")
                if instrument["product"] == "bet":
                    timestamp(instrument["starts_at"])
                state["instruments"][symbol] = instrument
            if len(state["instruments"]) > 30:
                raise ValueError("Paper watchlist exceeds the 30-instrument budget")
            for update in configuration.get("fee_updates", []):
                symbol, key = update["symbol"], update["fee_profile"]
                if symbol not in state["instruments"] or key not in state["fee_profiles"]:
                    raise ValueError("Fee update must reference registered versions")
                if timestamp(state["fee_profiles"][key]["valid_until"]) <= self.clock():
                    raise ValueError("New fee version is already expired")
                old = state["fee_profiles"][state["instruments"][symbol]["fee_profile"]]
                if number(state["fee_profiles"][key]["exit_minimum_commission"]) > number(
                    old["exit_minimum_commission"]
                ) and any(symbol in branch["positions"] for branch in state["branches"].values()):
                    raise ValueError(
                        "Close positions before increasing their reserved minimum exit fee"
                    )
                if old["financing_annual_bps"] != state["fee_profiles"][key][
                    "financing_annual_bps"
                ] and any(symbol in branch["positions"] for branch in state["branches"].values()):
                    raise ValueError(
                        "Close fixed-rate positions before changing their financing basis"
                    )
                state["instruments"][symbol]["fee_profile"] = key
            self.record(state, "configure", configuration)
        return state

    def equity(self, state: dict, branch: str) -> Decimal:
        value = number(state["branches"][branch]["cash"])
        for symbol, position in state["branches"][branch]["positions"].items():
            instrument = state["instruments"][symbol]
            if instrument["product"] == "bet":
                value += number(position["margin"])
                continue
            quote = state["quotes"][symbol]
            price = number(quote["bid"] if position["side"] == "long" else quote["ask"])
            quantity, entry = number(position["quantity"]), number(position["entry"])
            gross = (price - entry) * quantity * (1 if position["side"] == "long" else -1)
            profile = state["fee_profiles"][instrument["fee_profile"]]
            value += max(
                ZERO,
                number(position["margin"])
                + gross
                - D(position["carrying_costs"])
                - fee_total(profile, quantity * price),
            )
        return value

    def exposure(self, state: dict, branch: str) -> list[dict]:
        values = [
            {"symbol": symbol, "amount": number(position["allocation"])}
            for symbol, position in state["branches"][branch]["positions"].items()
        ]
        values.extend(
            {"symbol": order["symbol"], "amount": number(order["remaining_budget"])}
            for order in state["orders"]
            if order["branch"] == branch and order["action"] == "open"
        )
        return values

    def check_budget(self, state: dict, branch: str, symbol: str, budget: Decimal) -> None:
        equity = self.equity(state, branch)
        risk, instruments = state["risk"], state["instruments"]
        exposure = self.exposure(state, branch)
        target = instruments[symbol]
        for field, selector in (
            ("position_fraction", lambda item: item["symbol"] == symbol),
            (
                "cluster_fraction",
                lambda item: instruments[item["symbol"]]["cluster"] == target["cluster"],
            ),
            (
                "market_fraction",
                lambda item: instruments[item["symbol"]]["market"] == target["market"],
            ),
            ("total_fraction", lambda item: True),
        ):
            used = sum((item["amount"] for item in exposure if selector(item)), ZERO)
            if used + budget > equity * number(risk[field]):
                raise ValueError(f"Paper diversification limit: {field}")
        reserved = sum(
            (
                number(order["remaining_budget"])
                for order in state["orders"]
                if order["branch"] == branch and order["action"] == "open"
            ),
            ZERO,
        )
        if budget + reserved > number(state["branches"][branch]["cash"]):
            raise ValueError("Insufficient unreserved paper cash")

    def decide(self, branch: str, proposal: dict, asof_sequence: int) -> dict:
        if branch not in ("A", "B"):
            raise ValueError("Unknown paper branch")
        if (
            proposal.get("action") not in ("hold", "open", "close", "cancel")
            or not isinstance(proposal.get("rationale"), str)
            or not 1 <= len(proposal["rationale"]) <= 1600
        ):
            raise ValueError("Invalid paper decision")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            state = self.state()
            if asof_sequence != self.sequence():
                raise ValueError("Decision context changed; recompute instead of using future data")
            order = {
                **proposal,
                "id": uuid.uuid4().hex,
                "branch": branch,
                "created_at": self.clock().isoformat(),
                "asof_sequence": asof_sequence,
            }
            if proposal["action"] == "cancel":
                symbol = identifier(proposal["symbol"])
                if not any(
                    item["branch"] == branch and item["symbol"] == symbol
                    for item in state["orders"]
                ):
                    raise ValueError("No pending paper order to cancel")
                state["orders"] = [
                    item
                    for item in state["orders"]
                    if item["branch"] != branch or item["symbol"] != symbol
                ]
            if proposal["action"] not in ("hold", "cancel"):
                symbol = identifier(proposal["symbol"])
                if symbol not in state["quotes"] or symbol in state["outcomes"]:
                    raise ValueError("No active observed quote")
                instrument = state["instruments"][symbol]
                if any(
                    item["symbol"] == symbol and item["branch"] == branch
                    for item in state["orders"]
                ):
                    raise ValueError("One pending order per branch/instrument")
                profile = state["fee_profiles"][instrument["fee_profile"]]
                if timestamp(profile["valid_until"]) <= self.clock():
                    raise ValueError("Fee evidence expired; new decisions cannot assume zero costs")
                if proposal["action"] == "open":
                    if (
                        self.clock() - timestamp(state["quotes"][symbol]["observed_at"])
                    ).total_seconds() > state["risk"]["quote_age_seconds"]:
                        raise ValueError("Cannot plan new exposure from a stale quote")
                    budget = number(proposal["budget"], positive=True)
                    leverage = number(proposal["leverage"], positive=True)
                    if leverage > number(state["risk"]["max_leverage"]) or leverage < 1:
                        raise ValueError("Leverage outside the fixed experiment budget")
                    if instrument["product"] != "isolated-linear" and (
                        leverage != 1 or proposal["side"] != "long"
                    ):
                        raise ValueError("Spot/bets have no margin short or leverage")
                    if (
                        proposal["side"] not in ("long", "short")
                        or state["branches"][branch]["paused"]
                    ):
                        raise ValueError("New exposure is paused or side is invalid")
                    if symbol in state["branches"][branch]["positions"]:
                        raise ValueError("Close the existing position before reopening")
                    self.check_budget(state, branch, symbol, budget)
                    if (
                        instrument["product"] == "bet"
                        and timestamp(instrument["starts_at"]) <= self.clock()
                    ):
                        raise ValueError("Cannot bet after the event starts")
                    order["remaining_budget"] = str(budget)
                elif (
                    symbol not in state["branches"][branch]["positions"]
                    or instrument["product"] == "bet"
                ):
                    raise ValueError("No closeable position; sports require independent settlement")
                state["orders"].append(order)
            instrument = state["instruments"].get(proposal.get("symbol"), {})
            decision_inputs = {
                "quote": state["quotes"].get(proposal.get("symbol")),
                "instrument": instrument,
                "fee_profile": state["fee_profiles"].get(instrument.get("fee_profile")),
                "risk": state["risk"],
            }
            event = self.record(
                state, "decision", {"proposal": order, "decision_inputs": decision_inputs}, branch
            )
        ActivityLog(self.root, branch, "model").write("decisions", "paper-decision", event)
        from rlm.v100.experiments import SharedLab

        shared = SharedLab(self.root / "research/state/competition.sqlite3")
        try:
            shared.append(
                branch,
                "message",
                {"domain": "paper", "decision": order, "paper_sequence": event["sequence"]},
            )
        finally:
            shared.close()
        return event

    def ingest(self, snapshot: dict) -> dict:
        """Controller imports independent feed observations, not model-proposed prices."""
        now = self.clock()
        source_fields(snapshot, now)
        kind = snapshot["kind"]
        if kind not in ("quote", "outcome", "news", "corporate"):
            raise ValueError("Unsupported observation")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            state = self.state()
            observation = {**snapshot, "observed_at": now.isoformat()}
            trades = []
            if kind == "news":
                if (
                    snapshot.get("category") not in ("social", "quarterly", "news", "research")
                    or not isinstance(snapshot.get("excerpt"), str)
                    or len(snapshot["excerpt"]) > 6000
                ):
                    raise ValueError("Invalid timestamped research observation")
                state["news"] = [*state["news"][-19:], observation]
            else:
                symbol = identifier(snapshot["symbol"])
                instrument = state["instruments"][symbol]
                if snapshot["feed_id"] != instrument["feed_id"]:
                    raise ValueError("Observation from an unapproved execution feed")
                if kind == "corporate":
                    if (
                        instrument["market"] != "equities"
                        or instrument["product"] != "spot"
                        or snapshot["action"] not in ("split", "dividend")
                    ):
                        raise ValueError("Corporate actions require registered spot equities")
                    effective = timestamp(snapshot["effective_at"])
                    previous = state["quotes"].get(symbol)
                    if not previous or not timestamp(
                        previous["available_at"]
                    ) < effective <= timestamp(snapshot["available_at"]):
                        raise ValueError(
                            "Apply a corporate action between the pre-event and post-event raw quotes"
                        )
                    if any(
                        event["kind"] == "observation"
                        and event["payload"]["observation"].get("corporate_id")
                        == snapshot["corporate_id"]
                        for event in self.events()
                    ):
                        raise ValueError("Corporate action already applied")
                    trades.extend(self.corporate_action(state, symbol, snapshot))
                elif kind == "outcome":
                    if (
                        instrument["product"] != "bet"
                        or symbol in state["outcomes"]
                        or snapshot["result"] not in ("won", "lost", "void")
                        or timestamp(snapshot["available_at"]) < timestamp(instrument["starts_at"])
                    ):
                        raise ValueError("Invalid independent sports settlement")
                    state["outcomes"][symbol] = observation
                    trades.extend(self.settle_bet(state, symbol, snapshot["result"]))
                    state["orders"] = [
                        order for order in state["orders"] if order["symbol"] != symbol
                    ]
                else:
                    age = (now - timestamp(snapshot["available_at"])).total_seconds()
                    if age > state["risk"]["quote_age_seconds"]:
                        raise ValueError(
                            "Stale quote; historical imports are not forward paper evidence"
                        )
                    previous = state["quotes"].get(symbol)
                    if previous and timestamp(snapshot["available_at"]) <= timestamp(
                        previous["available_at"]
                    ):
                        raise ValueError("Quotes must advance their availability time")
                    if snapshot["currency"] != state["currency"]:
                        raise ValueError(
                            "Normalize prices to the portfolio currency with recorded FX evidence"
                        )
                    for field in ("bid", "ask", "bid_size", "ask_size"):
                        number(snapshot.get(field), positive=field in ("bid", "ask"))
                    if number(snapshot["bid"]) > number(snapshot["ask"]):
                        raise ValueError("Crossed quote")
                    if instrument["product"] == "isolated-linear":
                        low, high, mark = (
                            number(snapshot.get(key), positive=True)
                            for key in ("mark_low", "mark_high", "mark")
                        )
                        if not low <= mark <= high or not isinstance(
                            snapshot.get("funding_events"), list
                        ):
                            raise ValueError(
                                "Perps require adverse interval marks and explicit funding costs"
                            )
                        if timestamp(snapshot["interval_start"]) != timestamp(
                            previous["available_at"] if previous else snapshot["available_at"]
                        ):
                            raise ValueError(
                                "Perp funding/adverse-mark interval must cover exactly the previous quote interval"
                            )
                        seen = set()
                        for payment in snapshot["funding_events"]:
                            at = timestamp(payment["at"])
                            rate = D(str(payment["rate_bps"]))
                            if (
                                not rate.is_finite()
                                or not timestamp(snapshot["interval_start"])
                                < at
                                <= timestamp(snapshot["available_at"])
                                or payment["at"] in seen
                            ):
                                raise ValueError(
                                    "Funding settlements must be unique and inside the covered interval"
                                )
                            number(payment["mark"], positive=True)
                            seen.add(payment["at"])
                    if instrument["product"] == "bet" and number(snapshot["ask"]) <= 1:
                        raise ValueError("Decimal odds must exceed one")
                    state["quotes"][symbol] = observation
                    trades.extend(self.mark_positions(state, symbol, previous, observation))
                    self.refresh_risk(state)
                    trades.extend(self.fill_orders(state, symbol, observation))
            self.refresh_risk(state)
            event = self.record(
                state,
                "observation",
                {
                    "observation": observation,
                    "trades": trades,
                    "equity": {branch: str(self.equity(state, branch)) for branch in ("A", "B")},
                },
            )
        ActivityLog(self.root, "controller", "paper").write("metrics", "paper-observation", event)
        return event

    def corporate_action(self, state: dict, symbol: str, snapshot: dict) -> list[dict]:
        result = []
        if snapshot["action"] == "split":
            ratio = number(snapshot["ratio"], positive=True)
            for field in ("bid", "ask"):
                state["quotes"][symbol][field] = str(number(state["quotes"][symbol][field]) / ratio)
            for field in ("bid_size", "ask_size"):
                state["quotes"][symbol][field] = str(number(state["quotes"][symbol][field]) * ratio)
            state["quotes"][symbol]["derived_after_split"] = snapshot["corporate_id"]
        else:
            if (
                snapshot["currency"] != state["currency"]
                or number(snapshot["withholding_bps"]) >= 10000
            ):
                raise ValueError("Dividend needs currency conversion and documented withholding")
            number(snapshot["amount_per_unit"])
            if timestamp(snapshot["entitlement_at"]) > timestamp(snapshot["effective_at"]):
                raise ValueError("Dividend entitlement must precede payment")
        for branch, portfolio in state["branches"].items():
            position = portfolio["positions"].get(symbol)
            if snapshot["action"] == "split":
                if not position:
                    continue
                position["quantity"] = str(number(position["quantity"]) * ratio)
                position["entry"] = str(number(position["entry"]) / ratio)
                result.append(
                    {"branch": branch, "symbol": symbol, "action": "split", "ratio": str(ratio)}
                )
            else:
                held = ZERO
                for event in self.events():
                    if event["kind"] != "observation" or timestamp(
                        event["payload"]["observation"]["available_at"]
                    ) >= timestamp(snapshot["entitlement_at"]):
                        continue
                    for trade in event["payload"]["trades"]:
                        if trade["branch"] != branch or trade["symbol"] != symbol:
                            continue
                        if trade["action"] == "buy":
                            held += number(trade["quantity"])
                        elif trade["action"] in ("close", "liquidation"):
                            held -= number(trade["quantity"])
                        elif trade["action"] == "split":
                            held *= number(trade["ratio"])
                gross = number(snapshot["amount_per_unit"]) * max(ZERO, held)
                tax = gross * number(snapshot["withholding_bps"]) / 10000
                portfolio["cash"] = str(number(portfolio["cash"]) + gross - tax)
                portfolio["costs"] = str(D(portfolio["costs"]) + tax)
                result.append(
                    {
                        "branch": branch,
                        "symbol": symbol,
                        "action": "dividend",
                        "payout": str(gross - tax),
                        "costs": str(tax),
                    }
                )
        return result

    def refresh_risk(self, state: dict) -> None:
        for branch, portfolio in state["branches"].items():
            equity = self.equity(state, branch)
            peak = max(number(portfolio["peak"]), equity)
            portfolio["peak"] = str(peak)
            if peak and 1 - equity / peak >= number(state["risk"]["drawdown_pause"]):
                portfolio["paused"] = True
            if portfolio["paused"]:
                state["orders"] = [
                    order
                    for order in state["orders"]
                    if order["branch"] != branch or order["action"] == "close"
                ]

    def mark_positions(
        self, state: dict, symbol: str, previous: dict | None, quote: dict
    ) -> list[dict]:
        result = []
        instrument = state["instruments"][symbol]
        if instrument["product"] == "bet":
            return result
        profile = state["fee_profiles"][instrument["fee_profile"]]
        for branch, portfolio in state["branches"].items():
            position = portfolio["positions"].get(symbol)
            if not position:
                continue
            hours = (
                D(
                    str(
                        (
                            timestamp(quote["available_at"]) - timestamp(position["last_mark_at"])
                        ).total_seconds()
                    )
                )
                / 3600
            )
            quantity = number(position["quantity"])
            notional = quantity * number(position["entry"])
            financing = (
                notional * number(profile["financing_annual_bps"]) / 10000 * hours / (365 * 24)
            )
            funding = sum(
                (
                    quantity
                    * number(payment["mark"])
                    * D(str(payment["rate_bps"]))
                    / 10000
                    * (1 if position["side"] == "long" else -1)
                    for payment in quote.get("funding_events", [])
                    if timestamp(payment["at"]) >= timestamp(position["opened_at"])
                ),
                ZERO,
            )
            charge = financing + funding
            position["carrying_costs"] = str(D(position["carrying_costs"]) + charge)
            portfolio["costs"] = str(D(portfolio["costs"]) + charge)
            position["last_mark_at"] = quote["available_at"]
            if instrument["product"] != "isolated-linear":
                continue
            price = number(quote["mark_low"] if position["side"] == "long" else quote["mark_high"])
            gross = (
                quantity
                * (price - number(position["entry"]))
                * (1 if position["side"] == "long" else -1)
            )
            maintenance = quantity * price * number(instrument["maintenance_fraction"])
            if number(position["margin"]) + gross - D(
                position["carrying_costs"]
            ) <= maintenance + fee_total(profile, quantity * price, liquidation=True):
                result.append(
                    self.close_position(
                        state, branch, symbol, quote, "liquidation", quantity, price
                    )
                )
                state["orders"] = [
                    order
                    for order in state["orders"]
                    if order["branch"] != branch or order["symbol"] != symbol
                ]
        return result

    def fill_orders(self, state: dict, symbol: str, quote: dict) -> list[dict]:
        result, retained = [], []
        instrument = state["instruments"][symbol]
        profile = state["fee_profiles"][instrument["fee_profile"]]
        liquidity = {
            branch: {"long": number(quote["ask_size"]), "short": number(quote["bid_size"])}
            for branch in ("A", "B")
        }
        for order in state["orders"]:
            expired = (self.clock() - timestamp(order["created_at"])).total_seconds() > state[
                "risk"
            ]["order_ttl_seconds"]
            if expired:
                result.append(
                    {
                        "branch": order["branch"],
                        "symbol": order["symbol"],
                        "action": "cancel",
                        "reason": "expired",
                        "order_id": order["id"],
                    }
                )
                continue
            if order["symbol"] != symbol or timestamp(quote["available_at"]) <= timestamp(
                order["created_at"]
            ):
                retained.append(order)
                continue
            branch = order["branch"]
            if timestamp(profile["valid_until"]) <= self.clock():
                result.append(
                    {
                        "branch": branch,
                        "symbol": symbol,
                        "action": "cancel",
                        "reason": "expired-fees",
                        "order_id": order["id"],
                    }
                )
                continue
            if order["action"] == "close":
                position = state["branches"][branch]["positions"][symbol]
                available = number(
                    quote["bid_size"] if position["side"] == "long" else quote["ask_size"]
                )
                quantity = min(number(position["quantity"]), available)
                if quantity:
                    result.append(
                        self.close_position(state, branch, symbol, quote, "close", quantity)
                    )
                if symbol in state["branches"][branch]["positions"]:
                    retained.append(order)
                continue
            if (
                instrument["product"] == "bet"
                and timestamp(instrument["starts_at"]) <= self.clock()
            ):
                result.append(
                    {
                        "branch": branch,
                        "symbol": symbol,
                        "action": "cancel",
                        "reason": "event-started",
                    }
                )
                continue
            # Each partially filled order remains a reserved budget. Recheck at actual fill.
            budget = min(
                number(order["remaining_budget"]), number(state["branches"][branch]["cash"])
            )
            others = [item for item in state["orders"] if item["id"] != order["id"]]
            saved = state["orders"]
            state["orders"] = others
            try:
                self.check_budget(state, branch, symbol, budget)
            except ValueError:
                result.append(
                    {
                        "branch": branch,
                        "symbol": symbol,
                        "action": "cancel",
                        "reason": "risk-limit-at-fill",
                    }
                )
                continue
            finally:
                state["orders"] = saved
            trade = self.open_position(
                state, branch, symbol, quote, order, budget, liquidity[branch][order["side"]]
            )
            if trade:
                result.append(trade)
                liquidity[branch][order["side"]] -= number(trade["quantity"])
            if number(order["remaining_budget"]) > number(profile["entry_minimum_commission"]) and (
                instrument["product"] != "bet" or trade is None
            ):
                retained.append(order)
        state["orders"] = retained
        return result

    def open_position(
        self,
        state: dict,
        branch: str,
        symbol: str,
        quote: dict,
        order: dict,
        budget: Decimal,
        available: Decimal,
    ) -> dict | None:
        instrument = state["instruments"][symbol]
        profile = state["fee_profiles"][instrument["fee_profile"]]
        portfolio = state["branches"][branch]
        if instrument["product"] == "bet":
            step = number(instrument["quantity_step"], positive=True)
            stake = (min(budget, available) / step).to_integral_value(rounding=ROUND_DOWN) * step
            if not stake:
                return None
            entry_fees = fee_total(profile, stake, closing=False)
            effective = max(ZERO, stake - entry_fees) * (
                1 - number(profile["stake_tax_bps"]) / 10000
            )
            if not effective:
                return None
            position = {
                "margin": str(effective),
                "allocation": str(stake),
                "odds": quote["ask"],
                "opened_at": quote["available_at"],
                "order_id": order["id"],
                "entry_costs": str(stake - effective),
                "entry_fees": str(entry_fees),
                "stake_tax": str(stake - entry_fees - effective),
            }
            portfolio["positions"][symbol] = position
            portfolio["cash"] = str(number(portfolio["cash"]) - stake)
            portfolio["costs"] = str(D(portfolio["costs"]) + stake - effective)
            order["remaining_budget"] = "0"
            return {
                "branch": branch,
                "symbol": symbol,
                "action": "bet",
                "quantity": str(stake),
                "allocation": str(stake),
                "costs": str(stake - effective),
                "odds": quote["ask"],
                "order_id": order["id"],
            }
        side = order["side"]
        price = number(quote["ask"] if side == "long" else quote["bid"]) * (
            1 + number(profile["slippage_bps"]) / 10000 * (1 if side == "long" else -1)
        )
        if price <= 0:
            raise ValueError("Invalid execution slippage")
        leverage = number(order["leverage"])
        rate = (
            sum(
                (
                    number(profile[field])
                    for field in ("entry_commission_bps", "entry_venue_fee_bps", "entry_fx_bps")
                ),
                ZERO,
            )
            / 10000
        )
        exit_reserve = number(profile["exit_minimum_commission"])
        quantity = max(
            ZERO,
            (budget - number(profile["entry_minimum_commission"]) - exit_reserve)
            / (price * (1 / leverage + rate)),
        )
        step = number(instrument["quantity_step"], positive=True)
        quantity = (min(quantity, available) / step).to_integral_value(rounding=ROUND_DOWN) * step
        if not quantity:
            return None
        notional = quantity * price
        margin, entry_fees = (
            notional / leverage + exit_reserve,
            fee_total(profile, notional, closing=False),
        )
        cost = margin + entry_fees
        if cost > budget:
            raise ValueError("Execution exceeds the reserved allocation")
        existing = portfolio["positions"].get(symbol)
        if existing:
            old_quantity = number(existing["quantity"])
            existing["entry"] = str(
                (old_quantity * number(existing["entry"]) + notional) / (old_quantity + quantity)
            )
            existing["quantity"] = str(old_quantity + quantity)
            for field, value in (
                ("margin", margin),
                ("allocation", cost),
                ("entry_costs", entry_fees),
            ):
                existing[field] = str(D(existing[field]) + value)
        else:
            portfolio["positions"][symbol] = {
                "side": side,
                "quantity": str(quantity),
                "entry": str(price),
                "margin": str(margin),
                "allocation": str(cost),
                "entry_costs": str(entry_fees),
                "carrying_costs": "0",
                "opened_at": quote["available_at"],
                "last_mark_at": quote["available_at"],
                "order_id": order["id"],
            }
        portfolio["cash"] = str(number(portfolio["cash"]) - cost)
        portfolio["costs"] = str(D(portfolio["costs"]) + entry_fees)
        order["remaining_budget"] = str(budget - cost)
        return {
            "branch": branch,
            "symbol": symbol,
            "action": "buy" if side == "long" else "short",
            "quantity": str(quantity),
            "price": str(price),
            "allocation": str(cost),
            "costs": str(entry_fees),
            "order_id": order["id"],
        }

    def close_position(
        self,
        state: dict,
        branch: str,
        symbol: str,
        quote: dict,
        action: str,
        quantity: Decimal,
        override_price: Decimal | None = None,
    ) -> dict:
        portfolio = state["branches"][branch]
        position = portfolio["positions"][symbol]
        profile = state["fee_profiles"][state["instruments"][symbol]["fee_profile"]]
        side = position["side"]
        price = (
            override_price
            if override_price is not None
            else number(quote["bid"] if side == "long" else quote["ask"])
            * (1 - number(profile["slippage_bps"]) / 10000 * (1 if side == "long" else -1))
        )
        fraction = quantity / number(position["quantity"])
        margin = number(position["margin"]) * fraction
        gross = (price - number(position["entry"])) * quantity * (1 if side == "long" else -1)
        exit_fees = fee_total(profile, quantity * price, liquidation=action == "liquidation")
        # Simulator cap applies only to registered position-isolated contracts.
        carrying = D(position["carrying_costs"]) * fraction
        payout = (
            ZERO if action == "liquidation" else max(ZERO, margin + gross - carrying - exit_fees)
        )
        allocation = number(position["allocation"]) * fraction
        entry_costs = number(position["entry_costs"]) * fraction
        realized = payout - allocation
        portfolio["cash"] = str(number(portfolio["cash"]) + payout)
        portfolio["costs"] = str(D(portfolio["costs"]) + exit_fees)
        if fraction == 1:
            del portfolio["positions"][symbol]
        else:
            position["quantity"] = str(number(position["quantity"]) - quantity)
            for field in ("margin", "allocation", "entry_costs", "carrying_costs"):
                position[field] = str(D(position[field]) * (1 - fraction))
        return {
            "branch": branch,
            "symbol": symbol,
            "action": action,
            "quantity": str(quantity),
            "price": str(price),
            "payout": str(payout),
            "allocation": str(allocation),
            "net_pnl": str(realized),
            "gross_pnl": str(gross),
            "costs": str(exit_fees),
            "carrying_costs": str(carrying),
            "entry_costs": str(entry_costs),
            "order_id": position["order_id"],
            "loss_cap_adjustment": str(payout - (margin + gross - carrying - exit_fees)),
        }

    def settle_bet(self, state: dict, symbol: str, outcome: str) -> list[dict]:
        result = []
        profile = state["fee_profiles"][state["instruments"][symbol]["fee_profile"]]
        for branch, portfolio in state["branches"].items():
            position = portfolio["positions"].pop(symbol, None)
            if not position:
                continue
            effective, allocation = number(position["margin"]), number(position["allocation"])
            payout = effective * number(position["odds"]) if outcome == "won" else ZERO
            gross = payout
            if outcome == "won":
                payout = winning_payout(profile, effective, number(position["odds"]))
            if outcome == "void":
                refund = (
                    number(position["stake_tax"]) if profile["void_refunds_stake_tax"] else ZERO
                ) + (number(position["entry_fees"]) if profile["void_refunds_entry_fees"] else ZERO)
                payout = effective + refund
                portfolio["costs"] = str(D(portfolio["costs"]) - refund)
                gross = payout
            portfolio["cash"] = str(number(portfolio["cash"]) + payout)
            portfolio["costs"] = str(D(portfolio["costs"]) + gross - payout)
            result.append(
                {
                    "branch": branch,
                    "symbol": symbol,
                    "action": "settle",
                    "result": outcome,
                    "allocation": str(allocation),
                    "payout": str(payout),
                    "net_pnl": str(payout - allocation),
                    "costs": str(gross - payout),
                    "order_id": position["order_id"],
                }
            )
        return result

    def context(self, branch: str) -> dict:
        with self.db:
            self.db.execute("BEGIN")
            return self.context_snapshot(branch)

    def context_snapshot(self, branch: str) -> dict:
        state = self.state()
        return {
            "goal": state["goal"],
            "sequence": self.sequence(),
            "currency": state["currency"],
            "risk": state["risk"],
            "branch": branch,
            "portfolio": state["branches"][branch],
            "equity": str(self.equity(state, branch)),
            "instruments": state["instruments"],
            "quotes": state["quotes"],
            "news": state["news"][-4:],
            "fee_profiles": state["fee_profiles"],
            "pending": [order for order in state["orders"] if order["branch"] == branch],
        }
