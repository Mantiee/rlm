"""Primary-source paper spot configuration; no account entitlement or real orders."""

import hashlib
import re
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.paper import FEE_FIELDS, PaperBook
from rlm.v100.paper_feeds import poll_crypto, public_json
from rlm.v100.research_tools import TextOnly, download_page

FEES_URL = "https://www.kraken.com/features/fee-schedule"
PAIRS_URL = "https://api.kraken.com/0/public/AssetPairs?pair=XBTUSD,ETHUSD"


def fee_bps(body: str) -> str:
    parser = TextOnly()
    parser.feed(body)
    text = " ".join(parser.parts)
    if "Spot Crypto" not in text or "Spot Maker Rebate" not in text:
        raise ValueError("Could not identify the spot-crypto fee table")
    sections = [part.split("Spot Maker Rebate", 1)[0] for part in text.split("Spot Crypto")[1:]]
    section = " ".join(part for part in sections if "Tier 1" in part)
    rows = re.findall(
        r"Tier\s*1\s*\$0\+\s*N/A\s*(\d+(?:\.\d+)?)\s*%\s*(\d+(?:\.\d+)?)\s*%", section
    )
    if len(rows) != 1 or not Decimal("0") <= Decimal(rows[0][1]) < Decimal("10"):
        raise ValueError("Lowest-volume taker fee missing or ambiguous; no guessed fee")
    return str(Decimal(rows[0][1]) * 100)


def discover(root: Path, limit: int = 40) -> dict:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Market discovery limit must be 1-100")
    url = "https://api.kraken.com/0/public/AssetPairs"
    payload, digest, observed = public_json(root, url)
    if payload.get("error") or not isinstance(payload.get("result"), dict):
        raise ValueError("Kraken pair discovery failed")
    pairs = [
        item
        for item in payload["result"].values()
        if item.get("quote") == "ZUSD"
        and item.get("status") == "online"
        and re.fullmatch(r"[A-Z0-9]{2,20}USD", item.get("altname", ""))
    ]
    pairs.sort(key=lambda item: item["altname"])
    return {
        "pairs": [
            {k: p.get(k) for k in ("altname", "wsname", "ordermin", "costmin")}
            for p in pairs[:limit]
        ],
        "total": len(pairs),
        "source_url": url,
        "source_sha256": digest,
        "available_at": observed,
        "scope": "Public spot candidates; listing is not a profitability or account-entitlement claim",
    }


def prepare(root: Path, refresh: bool = False, pair_codes: list[str] | None = None) -> dict:
    from rlm.v100.mission import status

    if not refresh and status(root)["running"]:
        raise ValueError("Stop the mission before preparing its paper instruments")
    book = PaperBook(root)
    try:
        state = book.state()
        if not state.get("currency"):
            raise ValueError("Initialize the paper ledger first")
        if pair_codes is None:
            pair_codes = sorted(
                {
                    item["feed_id"].split(":", 1)[1]
                    for item in state["instruments"].values()
                    if item["feed_id"].startswith("kraken:")
                }
            ) or ["XBTUSD", "ETHUSD"]
        if (
            not isinstance(pair_codes, list)
            or not 1 <= len(pair_codes) <= 8
            or len(set(pair_codes)) != len(pair_codes)
            or any(
                not isinstance(p, str) or not re.fullmatch(r"[A-Z0-9]{2,20}USD", p)
                for p in pair_codes
            )
        ):
            raise ValueError("Choose 1-8 unique public USD spot pair codes")

        registered = {
            item["feed_id"].split(":", 1)[1]
            for item in state["instruments"].values()
            if item["feed_id"].startswith("kraken:")
        }
        if len(registered | set(pair_codes)) > 8:
            raise ValueError("At most eight registered spot pairs in this resource budget")

        def symbol_for(code):
            return ("BTC" if code == "XBTUSD" else code[:-3]) + "-KRAKEN"

        owned = [
            symbol_for(code) for code in pair_codes if symbol_for(code) in state["instruments"]
        ]
        if len(owned) == len(pair_codes) and all(
            datetime.fromisoformat(
                state["fee_profiles"][state["instruments"][s]["fee_profile"]]["valid_until"]
            )
            > datetime.now(UTC) + timedelta(hours=1)
            for s in owned
        ):
            poll_crypto(book)
            return {
                "status": "existing paper spot configuration retained",
                "weights_changed": False,
            }
        url, body = download_page(FEES_URL, max_bytes=2 * 2**20)
        if url != FEES_URL:
            raise ValueError("Unexpected fee-source redirect")
        taker = fee_bps(body)
        pairs_url = PAIRS_URL.split("?", 1)[0] + "?pair=" + ",".join(pair_codes)
        rules, rule_hash, observed = public_json(root, pairs_url)
        if (
            rules.get("error")
            or not isinstance(rules.get("result"), dict)
            or len(rules["result"]) != len(pair_codes)
            or {item.get("altname") for item in rules["result"].values()} != set(pair_codes)
        ):
            raise ValueError("Kraken instrument rules unavailable")
        digest = hashlib.sha256(body.encode()).hexdigest()
        snapshot = root / "research/paper/sources" / (digest + ".html")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            snapshot.write_text(body)
        identity = "kraken-tier1-" + str(time.time_ns())
        evidence = {"source_url": FEES_URL, "source_sha256": digest, "available_at": observed}
        profile = {
            **dict.fromkeys(FEE_FIELDS, "0"),
            **evidence,
            "id": identity,
            "currency": state["currency"],
            "operator_verified": True,
            "valid_until": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "entry_venue_fee_bps": taker,
            "exit_venue_fee_bps": taker,
            "slippage_bps": "10",
            "entry_fx_bps": "0" if state["currency"] == "USD" else "20",
            "exit_fx_bps": "0" if state["currency"] == "USD" else "20",
            "void_refunds_stake_tax": True,
            "void_refunds_entry_fees": False,
            "winnings_tax_basis": "whole-payout",
            "winnings_tax_base": "after-commission",
            "cost_evidence": "Published lowest-volume taker fee. Slippage 10 bps, FX 20 bps and zero minimum commissions are explicit paper assumptions. No account or regional entitlement, personal tax, deposit, withdrawal or gas certification. Fully paid exchange spot only. Never infer volume discounts from paper trading.",
        }
        instruments, updates = [], []
        for item in rules["result"].values():
            symbol = symbol_for(item["altname"])
            if not symbol or item.get("quote") != "ZUSD" or item.get("status") != "online":
                raise ValueError("Unexpected or unavailable USD spot pair")
            minimum = str(Decimal(item["ordermin"]))
            notional = str(Decimal(item["costmin"]))
            instrument = {
                "symbol": symbol,
                "market": "crypto",
                "product": "spot",
                "cluster": "crypto",
                "feed_id": "kraken:" + item["altname"],
                "quantity_step": str(Decimal(10) ** -int(item["lot_decimals"])),
                "minimum_quantity": minimum,
                "minimum_notional_quote": notional,
                "quote_currency": "USD",
                "fee_profile": identity,
                "price_basis": "raw-unadjusted",
                "operator_verified": True,
                "source_url": pairs_url,
                "source_sha256": rule_hash,
                "available_at": observed,
            }
            if symbol in state["instruments"]:
                original = state["instruments"][symbol]
                for key in (
                    "feed_id",
                    "quantity_step",
                    "minimum_quantity",
                    "minimum_notional_quote",
                    "quote_currency",
                ):
                    if original.get(key) != instrument[key]:
                        raise ValueError(
                            "Spot execution rules changed; retain positions and inspect before updating"
                        )
                updates.append({"symbol": symbol, "fee_profile": identity})
            else:
                instruments.append(instrument)
        configuration = {
            "fee_profiles": [profile],
            "instruments": instruments,
            "fee_updates": updates,
        }
        atomic_json(root / "research/paper/sources" / (identity + ".json"), configuration)
        book.configure(configuration)
        poll_crypto(book)
        return {
            "status": "paper spot sources configured",
            "symbols": [symbol_for(code) for code in pair_codes],
            "taker_fee_bps": taker,
            "simulation_assumptions": profile["cost_evidence"],
            "weights_changed": False,
            "real_money_ready": False,
        }
    finally:
        book.close()
