"""Public normalized JSON feed adapters for any registered paper market/product."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json
from rlm.v100.paper import PaperBook, identifier, timestamp


def select(value, dotted: str):
    if not isinstance(dotted, str) or len(dotted) > 200:
        raise ValueError("JSON selector exceeds budget")
    for key in dotted.split(".") if dotted else []:
        value = value[int(key)] if isinstance(value, list) and key.isdigit() else value[key]
    return value


def register(root: Path, configuration: dict) -> dict:
    from rlm.v100.research_tools import public_origin

    required = {"id", "url", "documentation_url", "symbol", "feed_id", "rows", "fields", "kind"}
    if not isinstance(configuration, dict) or set(configuration) != required:
        raise ValueError("Provider needs the exact documented JSON mapping")
    identity = identifier(configuration["id"])
    public_origin(configuration["url"])
    public_origin(configuration["documentation_url"])
    if configuration["kind"] not in ("quote", "outcome", "news", "corporate"):
        raise ValueError("Unknown normalized feed kind")
    fields = configuration["fields"]
    if not isinstance(fields, dict) or not 1 <= len(fields) <= 24:
        raise ValueError("Provider mapping needs 1..24 fields")
    allowed = {
        "available_at",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "currency",
        "result",
        "category",
        "excerpt",
        "action",
        "effective_at",
        "corporate_id",
        "ratio",
        "amount_per_share",
        "mark",
        "mark_low",
        "mark_high",
        "funding_events",
        "interval_start",
    }
    if (
        set(fields) - allowed
        or "available_at" not in fields
        or any(not isinstance(v, str) or len(v) > 200 for v in fields.values())
    ):
        raise ValueError("Map explicit source timestamps and supported observation fields")
    if not isinstance(configuration["rows"], str) or len(configuration["rows"]) > 200:
        raise ValueError("Invalid JSON row selector")
    book = PaperBook(root)
    try:
        if configuration["kind"] != "news":
            instrument = book.state()["instruments"].get(configuration["symbol"])
            if not instrument or instrument["feed_id"] != configuration["feed_id"]:
                raise ValueError(
                    "Execution feeds require an already operator-registered instrument/feed; research cannot certify broker rules"
                )
    finally:
        book.close()
    folder = root / "research/providers"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identity + ".json")
    if path.exists():
        if json.loads(path.read_text()) != configuration:
            raise ValueError("Provider mappings are immutable; use a new ID")
    elif len(list(folder.glob("*.json"))) >= 16:
        raise ValueError("Sixteen feed mappings already registered")
    else:
        from rlm.v100.research_tools import download_page

        final_url, body = download_page(configuration["documentation_url"], max_bytes=2**20)
        if final_url != configuration["documentation_url"] or not body.strip():
            raise ValueError("Documented source changed or returned no evidence")
        evidence = folder / "documentation"
        evidence.mkdir(exist_ok=True)
        document = evidence / (identity + ".html")
        document.write_text(body)
        atomic_json(
            evidence / (identity + ".json"),
            {
                "url": final_url,
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
                "retrieved_at": datetime.now(UTC).isoformat(),
                "scope": "Public provider documentation archived; mapping correctness and broker eligibility still require independent verification",
            },
        )
        atomic_json(path, configuration)
    return {"id": identity, "status": "registered read-only mapping", "fees_unchanged": True}


def normalize(configuration: dict, data, digest: str, retrieved: str) -> list[dict]:
    rows = select(data, configuration["rows"])
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or len(rows) > 100:
        raise ValueError("Provider returns at most 100 objects per tick")
    result = []
    for row in rows:
        mapped = {name: select(row, selector) for name, selector in configuration["fields"].items()}
        # Retrieval does NOT turn yesterday's quote into fresh evidence.
        if timestamp(mapped["available_at"]) > timestamp(retrieved):
            raise ValueError("Provider timestamp is in the future")
        result.append(
            {
                **mapped,
                "kind": configuration["kind"],
                "symbol": configuration["symbol"],
                "feed_id": configuration["feed_id"],
                "source_url": configuration["url"],
                "source_sha256": digest,
                "retrieved_at": retrieved,
            }
        )
    return result


def poll(book: PaperBook, cancelled=None) -> list[dict]:
    from rlm.v100.research_tools import download_page

    results = []
    paths = sorted((book.root / "research/providers").glob("*.json"))[:16]
    cursor = book.root / "research/providers/cursor/state.json"
    offset = json.loads(cursor.read_text())["next"] if cursor.exists() else 0
    selected = [paths[(offset + index) % len(paths)] for index in range(min(4, len(paths)))]
    if paths:
        atomic_json(cursor, {"next": (offset + len(selected)) % len(paths)})
    # At most four bounded reads per observer tick; one flaky provider does not
    # stall an entire registry of sixteen or disable unrelated observations.
    for path in selected:
        if cancelled and cancelled():
            break
        configuration = json.loads(path.read_text())
        try:
            url, body = download_page(configuration["url"], max_bytes=2**20)
            if url != configuration["url"]:
                raise ValueError("Provider redirected; register exact documented final endpoint")
            digest = hashlib.sha256(body.encode()).hexdigest()
            folder = book.root / "research/paper/sources"
            folder.mkdir(parents=True, exist_ok=True)
            archive = folder / (digest + ".json")
            if not archive.exists():
                archive.write_text(body)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
                raise ValueError("Provider archive changed")
            rows = normalize(configuration, json.loads(body), digest, datetime.now(UTC).isoformat())
            for row in rows:
                if cancelled and cancelled():
                    break
                state = book.state()
                old = state["quotes"].get(row["symbol"]) if row["kind"] == "quote" else None
                if old and timestamp(row["available_at"]) <= timestamp(old["available_at"]):
                    continue
                if row["kind"] == "outcome" and row["symbol"] in state["outcomes"]:
                    continue
                results.append(book.ingest(row))
        except Exception as error:
            # Independent provider failure must not stop healthy observers or fabricate data.
            ActivityLog(book.root, "controller", "provider").write(
                "errors",
                "provider-unavailable",
                {
                    "id": configuration["id"],
                    "error": type(error).__name__,
                    "detail": str(error)[:400],
                },
            )
    return results


def catalog(root: Path) -> dict:
    return {
        "providers": [
            json.loads(p.read_text())
            for p in sorted((root / "research/providers").glob("*.json"))[:16]
        ],
        "scope": "Generic source-timestamped JSON feeds: equity quotes/corporate actions, sports odds/settlements, isolated marks/funding, and news. No guessed fees or real order endpoints.",
    }
