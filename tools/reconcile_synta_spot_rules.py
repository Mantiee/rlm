"""Inspect and explicitly record stricter Kraken minima for flat paper instruments."""

import argparse
import hashlib
import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from rlm.v100.paper import PaperBook, number, sha, source_fields, timestamp
from rlm.v100.paper_feeds import public_json

FIELDS = (
    "feed_id",
    "quantity_step",
    "minimum_quantity",
    "minimum_notional_quote",
    "quote_currency",
)
BASES = {"XBTUSD": "XXBT", "ETHUSD": "XETH"}


def propose(state, publication):
    if publication.get("error") or not isinstance(publication.get("result"), dict):
        raise ValueError("Invalid Kraken publication")
    changes = []
    for symbol, original in state["instruments"].items():
        if not original.get("feed_id", "").startswith("kraken:"):
            continue
        code = original["feed_id"].split(":", 1)[1]
        # Other registered instruments remain untouched and retain their gates.
        if code not in BASES:
            continue
        rows = [row for row in publication["result"].values() if row.get("altname") == code]
        if len(rows) != 1:
            raise ValueError("Unknown or missing registered Kraken pair")
        row = rows[0]
        if (
            row.get("base") != BASES[code]
            or row.get("quote") != "ZUSD"
            or row.get("status") != "online"
        ):
            raise ValueError("Pair identity or availability changed")
        if original.get("product") != "spot" or original.get("quote_currency") != "USD":
            raise ValueError("Only registered USD spot instruments can be reconciled")
        if number(original["quantity_step"], positive=True) != Decimal(10) ** -int(
            row["lot_decimals"]
        ):
            raise ValueError("Quantity step changed; manual migration required")
        if number(original["minimum_notional_quote"]) != number(row["costmin"]):
            raise ValueError("Minimum notional changed; manual migration required")
        old, new = (
            number(original["minimum_quantity"], positive=True),
            number(row["ordermin"], positive=True),
        )
        if new < old:
            raise ValueError("This recovery cannot relax a quantity minimum")
        if new > old:
            changes.append(
                {
                    "symbol": symbol,
                    "before": {k: original.get(k) for k in FIELDS},
                    "minimum_quantity": str(new),
                }
            )
    return changes


def reconcile(book, publication, evidence, expected_digest):
    source_fields(evidence, book.clock())
    if book.clock() - timestamp(evidence["available_at"]) > timedelta(minutes=5):
        raise ValueError("Rule publication is stale")
    if not evidence["source_url"].startswith("https://api.kraken.com/0/public/AssetPairs?pair="):
        raise ValueError("Unexpected rule publication URL")
    snapshot = book.root / "research/paper/sources" / (evidence["source_sha256"] + ".json")
    body = snapshot.read_bytes()
    if (
        hashlib.sha256(body).hexdigest() != evidence["source_sha256"]
        or json.loads(body) != publication
    ):
        raise ValueError("Rule publication snapshot does not match")
    with book.db:
        book.db.execute("BEGIN IMMEDIATE")
        state = book.state()
        changes = propose(state, publication)
        if not changes:
            return {"state": "unchanged", "changes": []}
        if sha(changes) != expected_digest:
            raise ValueError("Rules changed after inspection")
        affected = {item["symbol"] for item in changes}
        if any(affected.intersection(branch["positions"]) for branch in state["branches"].values()):
            raise ValueError("Retain open positions; manual migration required")
        if any(order["symbol"] in affected for order in state["orders"]):
            raise ValueError("Retain pending orders; manual migration required")
        for change in changes:
            instrument = state["instruments"][change["symbol"]]
            instrument.update(evidence, minimum_quantity=change["minimum_quantity"])
        event = book.record(
            state, "spot-rule-reconciliation", {"changes": changes, "evidence": evidence}
        )
    return {"state": "reconciled", "changes": changes, "event_hash": event["event_hash"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply-digest", help="Apply only the exact inspected proposal SHA256")
    args = parser.parse_args()
    book = PaperBook(args.root)
    try:
        codes = sorted(
            {
                item["feed_id"].split(":", 1)[1]
                for item in book.state()["instruments"].values()
                if item.get("feed_id", "").startswith("kraken:")
                and item["feed_id"].split(":", 1)[1] in BASES
            }
        )
        if not codes or not set(codes).issubset(BASES):
            raise ValueError("Unknown Kraken watchlist")
        url = "https://api.kraken.com/0/public/AssetPairs?pair=" + ",".join(codes)
        publication, digest, observed = public_json(args.root, url)
        proposal = propose(book.state(), publication)
        evidence = {"source_url": url, "source_sha256": digest, "available_at": observed}
        result = {
            "state": "inspection-only",
            "changes": proposal,
            "apply_digest": sha(proposal),
            "evidence": evidence,
        }
        if args.apply_digest:
            result = reconcile(book, publication, evidence, args.apply_digest)
        print(json.dumps(result, indent=2))
    finally:
        book.close()


if __name__ == "__main__":
    main()
