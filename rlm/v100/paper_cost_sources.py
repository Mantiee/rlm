"""Read-only Kraken L2 collection for isolated shadow cost receipts.

REST schema reference: https://docs.kraken.com/api-reference/market-data/get-order-book
No ledger ingestion, FX reference substitution, fills or promotion.
"""

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from rlm.v100.execution_costs import amount, book_levels, estimate, timestamp

PAIRS = {"BTC": "BTCUSD", "ETH": "ETHUSD"}


def depth_url(base, count=100):
    if base not in PAIRS or isinstance(count, bool) or count not in (25, 100):
        raise ValueError("Shadow collector supports BTC/ETH USD and depth 25/100")
    return f"https://api.kraken.com/0/public/Depth?pair={PAIRS[base]}&count={count}&assetVersion=1"


def normalize_depth(raw, *, base, count, source_url, source_sha256, available_at):
    """Bind normalized levels to exact immutable bytes and the requested pair."""
    if source_url != depth_url(base, count):
        raise ValueError("Depth URL does not match fixed requested source")
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 2 * 2**20:
        raise ValueError("Invalid bounded source body")
    if hashlib.sha256(raw).hexdigest() != source_sha256:
        raise ValueError("Depth source hash mismatch")
    observed = timestamp(available_at)
    data = json.loads(raw, parse_float=Decimal)
    if not isinstance(data, dict) or data.get("error") != []:
        raise ValueError("Kraken returned errors or malformed response")
    result = data.get("result")
    pair = f"{base}/USD"
    if not isinstance(result, dict) or list(result) != [pair]:
        raise ValueError("Kraken returned a different or ambiguous pair")
    snapshot = {
        "base_currency": base,
        "quote_currency": "USD",
        "source_url": source_url,
        "source_sha256": source_sha256,
        "available_at": available_at,
        "venue": "Kraken",
        "depth_kind": "aggregated-L2",
        "requested_levels": count,
        "scope": "Visible resting liquidity; level timestamps are not snapshot availability",
    }
    for name in ("bids", "asks"):
        rows = result[pair][name]
        if not isinstance(rows, list) or not 1 <= len(rows) <= count:
            raise ValueError("Unexpected depth level count")
        levels, times = [], []
        for row in rows:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError("Kraken L2 row requires price, volume and timestamp")
            if not isinstance(row[0], str) or not isinstance(row[1], str):
                raise ValueError("Kraken prices and volumes must preserve decimal strings")
            price, size = amount(row[0], positive=True), amount(row[1], positive=True)
            at = amount(row[2], positive=True)
            if at > Decimal(str(observed.timestamp())):
                raise ValueError("Level timestamp is future relative to observation")
            levels.append([str(price), str(size)])
            times.append(str(at))
        snapshot[name] = levels
        snapshot[name + "_level_times"] = times
    book_levels(snapshot)
    return snapshot


def fetch_public(root, url):
    # Existing bounded allowlisted reader preserves exact raw bytes. Its root is
    # deliberately the shadow subtree; production paper sources are untouched.
    from rlm.v100.paper_feeds import public_json

    return public_json(root, url)


def immutable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(body)
    except FileExistsError:
        if path.read_bytes() != body:
            raise ValueError("Immutable shadow artifact changed") from None


def collect(
    root: Path,
    fee: dict,
    *,
    base="BTC",
    side="buy",
    quantity="0.001",
    reporting_currency="PLN",
    count=100,
    decision_at=None,
):
    """Collect one shadow estimate; fee is a caller-reviewed evidenced schedule.

    No FX feed is used. USD is a hypothetical native-quote estimate only; PLN
    necessarily returns UNKNOWN, irrespective of any available NBP reference.
    Receipt records the fee input; it never certifies account fee eligibility.
    """
    folder = root / "research/paper-cost-shadow"
    url = depth_url(base, count)
    _, digest, observed = fetch_public(folder, url)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Invalid raw snapshot identity")
    raw_path = folder / "research/paper/sources" / f"{digest}.json"
    if raw_path.is_symlink():
        raise ValueError("Shadow raw source must not be a symlink")
    if raw_path.stat().st_size > 2 * 2**20:
        raise ValueError("Raw source exceeds bound")
    raw = raw_path.read_bytes()
    book = normalize_depth(
        raw, base=base, count=count, source_url=url, source_sha256=digest, available_at=observed
    )
    decision = decision_at or datetime.now(UTC).isoformat()
    receipt = estimate(
        book,
        fee,
        side=side,
        quantity=quantity,
        decision_at=decision,
        reporting_currency=reporting_currency,
    )
    record = {
        "schema": "synta-shadow-cost-collection-v1",
        "receipt": receipt,
        "normalized_book": book,
        "fee_input": fee,
        "scope": "Isolated research; no canonical wallet, execution or promotion changes",
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > 256 * 1024:
        raise ValueError("Shadow receipt exceeds bound")
    identity = hashlib.sha256(encoded).hexdigest()
    immutable(folder / "receipts" / f"{identity}.json", encoded)
    return {"record_sha256": identity, **record}
