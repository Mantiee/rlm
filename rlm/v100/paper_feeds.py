"""Free read-only observations. No exchange/broker account or order endpoint."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import requests

from rlm.v100.paper import PaperBook, number


def public_json(
    root: Path, url: str, user_agent: str = "V100-Paper-Research/1.0"
) -> tuple[dict, str, str]:
    # All callers construct URLs on these fixed origins; no model-selected endpoint.
    if not url.startswith(
        (
            "https://api.exchange.coinbase.com/products/",
            "https://api.nbp.pl/api/exchangerates/",
            "https://data.sec.gov/submissions/",
        )
    ):
        raise ValueError("Unsupported public data origin")
    with requests.Session() as session:
        session.trust_env = False
        with session.get(
            url,
            headers={"User-Agent": user_agent},
            allow_redirects=False,
            timeout=(10, 20),
            stream=True,
        ) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("Public feed redirected or returned no observation")
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 2 * 2**20:
                    raise ValueError("Public data snapshot exceeds 2 MiB")
                chunks.append(chunk)
    body = b"".join(chunks)
    value = json.loads(body)
    digest = hashlib.sha256(body).hexdigest()
    directory = root / "research/paper/sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(body)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Public source snapshot changed")
    return value, digest, datetime.now(UTC).isoformat()


def poll_crypto(book: PaperBook) -> list[dict]:
    state = book.state()
    selected = [
        instrument
        for instrument in state["instruments"].values()
        if instrument["market"] == "crypto"
        and instrument["product"] == "spot"
        and instrument["feed_id"].startswith("coinbase:")
    ]
    if not selected:
        raise ValueError("Register operator-verified Coinbase spot instruments and fees first")
    conversion = {}
    result = []
    for instrument in selected:
        product = instrument["feed_id"].removeprefix("coinbase:")
        if not re.fullmatch(r"[A-Z0-9]{2,12}-[A-Z]{3}", product):
            raise ValueError("Unsupported Coinbase product identifier")
        quote_currency = product.rsplit("-", 1)[1]
        rate, fx = number("1"), None
        if quote_currency != state["currency"]:
            if state["currency"] != "PLN" or quote_currency not in ("USD", "EUR", "GBP"):
                raise ValueError("Automatic FX supports USD/EUR/GBP reference conversion to PLN")
            if quote_currency not in conversion:
                url = f"https://api.nbp.pl/api/exchangerates/rates/a/{quote_currency.lower()}/?format=json"
                data, digest, observed = public_json(book.root, url)
                conversion[quote_currency] = {
                    "rate": data["rates"][-1]["mid"],
                    "effective_date": data["rates"][-1]["effectiveDate"],
                    "source_url": url,
                    "source_sha256": digest,
                    "available_at": observed,
                    "note": "NBP reference rate, not an executable broker FX quote",
                }
            fx = conversion[quote_currency]
            rate = number(fx["rate"], positive=True)
        url = f"https://api.exchange.coinbase.com/products/{product}/book?level=1"
        data, digest, observed = public_json(book.root, url)
        if not data["bids"] or not data["asks"]:
            raise ValueError("No executable top-of-book quote")
        snapshot = {
            "kind": "quote",
            "symbol": instrument["symbol"],
            "feed_id": instrument["feed_id"],
            "bid": str(number(data["bids"][0][0], positive=True) * rate),
            "ask": str(number(data["asks"][0][0], positive=True) * rate),
            "bid_size": data["bids"][0][1],
            "ask_size": data["asks"][0][1],
            "currency": state["currency"],
            "price_source_currency": quote_currency,
            "source_url": url,
            "source_sha256": digest,
            "available_at": observed,
            "fx": fx,
            "feed_sequence": data.get("sequence"),
        }
        result.append(book.ingest(snapshot))
    return result


def poll_filings(book: PaperBook, cik: str, contact: str) -> list[dict]:
    if not re.fullmatch(r"\d{1,10}", cik) or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", contact):
        raise ValueError(
            "SEC access needs a numeric CIK and your real contact email for User-Agent"
        )
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    data, digest, observed = public_json(book.root, url, f"V100 paper research {contact}")
    recent = data["filings"]["recent"]
    seen = {
        event["payload"]["observation"].get("accession")
        for event in book.events()
        if event["kind"] == "observation" and event["payload"]["observation"]["kind"] == "news"
    }
    result = []
    for index, form in enumerate(recent["form"]):
        accession = recent["accessionNumber"][index]
        if form not in ("10-Q", "10-K", "8-K") or accession in seen:
            continue
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{recent['primaryDocument'][index]}"
        snapshot = {
            "kind": "news",
            "category": "quarterly" if form != "8-K" else "news",
            "available_at": observed,
            "source_url": url,
            "source_sha256": digest,
            "accession": accession,
            "excerpt": json.dumps(
                {
                    "company": data["name"],
                    "cik": cik,
                    "form": form,
                    "filing_date": recent["filingDate"][index],
                    "acceptance_timestamp_as_reported": recent.get(
                        "acceptanceDateTime", [None] * len(recent["form"])
                    )[index],
                    "filing_url": filing_url,
                    "note": "Newly observed filing metadata; financial contents require reading the filing. Do not substitute fiscal period end for public availability.",
                }
            ),
        }
        result.append(book.ingest(snapshot))
        if len(result) >= 3:
            break
    return result
