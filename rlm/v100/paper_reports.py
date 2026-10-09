"""Audited paper reports, static SVG charts and a local daily/weekly schedule."""

import csv
import html
import io
import json
import uuid
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from rlm.v100.common import atomic_json
from rlm.v100.paper import PaperBook, number, timestamp


def chart(series: list[tuple[str, list[tuple[str, float]]]], title: str, markers=()) -> str:
    points = [point for _, values in series for point in values]
    if not points:
        return "<p>No observations yet.</p>"
    epochs = [timestamp(point[0]).timestamp() for point in points]
    values = [point[1] for point in points]
    start, end, low, high = min(epochs), max(epochs), min(values), max(values)
    pad = max((high - low) * 0.05, abs(high) * 0.001, 0.01)
    low, high = low - pad, high + pad

    def xy(at, value):
        return (
            55 + (timestamp(at).timestamp() - start) / max(1, end - start) * 820,
            245 - (value - low) / (high - low) * 205,
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 930 290" role="img" aria-label="{html.escape(title, quote=True)}">',
        f'<text x="55" y="20">{html.escape(title)}</text>',
        '<path d="M55 35V245H875" stroke="#999" fill="none"/>',
    ]
    for label, y in ((f"{high:.2f}", 45), (f"{low:.2f}", 245)):
        parts.append(f'<text x="3" y="{y}">{label}</text>')
    colors = ("#2364aa", "#c7522a", "#32854b")
    for index, (label, observations) in enumerate(series):
        coordinates = " ".join(
            f"{x:.2f},{y:.2f}" for at, value in observations for x, y in [xy(at, value)]
        )
        color = colors[index % len(colors)]
        parts.append(
            f'<polyline points="{coordinates}" stroke="{color}" stroke-width="2" fill="none"/><text x="{55 + index * 210}" y="282" fill="{color}">{html.escape(label)}</text>'
        )
    for at, value, label in markers:
        x, y = xy(at, value)
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#7b3294"><title>{html.escape(label)}</title></circle>'
        )
    parts.extend(
        [
            f'<text x="55" y="264">{html.escape(points[epochs.index(start)][0][:19])} UTC</text>',
            f'<text x="650" y="264">{html.escape(points[epochs.index(end)][0][:19])} UTC</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def summarize(book: PaperBook, period: str = "all") -> dict:
    with book.db:
        book.db.execute("BEGIN")
        return summarize_snapshot(book, period)


def summarize_snapshot(book: PaperBook, period: str) -> dict:
    if period not in ("all", "daily", "weekly"):
        raise ValueError("Unknown report period")
    state, events = book.state(), book.events()
    now = book.clock()
    days = (now - timestamp(events[0]["time"])).total_seconds() / 86400
    cutoff = (
        now - timedelta(days=1 if period == "daily" else 7)
        if period != "all"
        else timestamp(events[0]["time"])
    )
    points = {
        branch: [(events[0]["time"], float(state["initial_capital"]))] for branch in ("A", "B")
    }
    trades = []
    price_series = {}
    for event in events:
        if event["kind"] != "observation":
            continue
        payload = event["payload"]
        for branch in points:
            points[branch].append((event["time"], float(payload["equity"][branch])))
        trades.extend(
            {**trade, "time": event["time"], "sequence": event["sequence"]}
            for trade in payload["trades"]
        )
        observation = payload["observation"]
        if observation["kind"] == "quote":
            price_series.setdefault(observation["symbol"], []).append(
                (event["time"], (float(observation["bid"]) + float(observation["ask"])) / 2)
            )
    stale_symbols = [
        symbol
        for symbol, quote in state["quotes"].items()
        if (now - timestamp(quote["observed_at"])).total_seconds()
        > state["risk"]["quote_age_seconds"]
        and symbol not in state["outcomes"]
    ]
    active_fee_ids = {instrument["fee_profile"] for instrument in state["instruments"].values()}
    expired_fees = [
        key
        for key, profile in state["fee_profiles"].items()
        if key in active_fee_ids and timestamp(profile["valid_until"]) <= now
    ]
    expired_archived_fees = [
        key
        for key, profile in state["fee_profiles"].items()
        if key not in active_fee_ids and timestamp(profile["valid_until"]) <= now
    ]
    branches = {}
    for branch in ("A", "B"):
        equity = book.equity(state, branch)
        peak, drawdown = number(state["initial_capital"]), Decimal("0")
        for _, value in points[branch]:
            amount = Decimal(str(value))
            peak = max(peak, amount)
            drawdown = max(drawdown, 1 - amount / peak)
        before = [Decimal(str(value)) for at, value in points[branch] if timestamp(at) < cutoff]
        starting = before[-1] if before else number(state["initial_capital"])
        resolved = [
            trade
            for trade in trades
            if trade["branch"] == branch and trade["action"] in ("close", "liquidation", "settle")
        ]
        period_trades = [trade for trade in resolved if timestamp(trade["time"]) >= cutoff]
        branches[branch] = {
            "equity": str(equity),
            "cash": state["branches"][branch]["cash"],
            "net_pnl_before_personal_tax": str(equity - number(state["initial_capital"])),
            "period_marked_pnl": str(equity - starting),
            "modeled_costs": state["branches"][branch]["costs"],
            "max_observed_drawdown": str(drawdown),
            "resolved_fills": len(resolved),
            "closed_order_ids": len({trade["order_id"] for trade in resolved}),
            "period_realized_pnl": str(
                sum((Decimal(trade["net_pnl"]) for trade in period_trades), Decimal("0"))
            ),
            "paused": state["branches"][branch]["paused"],
            "positions": state["branches"][branch]["positions"],
        }
    return {
        "schema": "v100-paper-report-v1",
        "time": now.isoformat(),
        "period": period,
        "currency": state["currency"],
        "initial_capital": state["initial_capital"],
        "elapsed_days": days,
        "branches": branches,
        "trades": trades,
        "equity_series": points,
        "price_series": price_series,
        "stale_symbols": stale_symbols,
        "expired_fee_profiles": expired_fees,
        "expired_archived_fee_profiles": expired_archived_fees,
        "audit_tail_sha256": events[-1]["event_hash"],
        "ledger_sequence": events[-1]["sequence"],
        "descriptive_leader": max(branches, key=lambda branch: Decimal(branches[branch]["equity"])),
        "tie": branches["A"]["equity"] == branches["B"]["equity"],
        "real_money_ready": False,
        "personal_income_tax": "not configured; not deducted",
        "notes": [
            "Forward paper decisions, not real executions or a guarantee of future profit.",
            "Fees/instrument rules are declared and verified by the operator, not certified by this simulator.",
            "Observed drawdown is sampled; market gaps between observations may be larger.",
            "Cash sports bets are valued conservatively at effective stake until independent settlement.",
            "Each partial fill charges its own minimum commission; this may conservatively overestimate costs.",
            "Isolated liquidation forfeits the entire position allocation; theoretical costs and loss-cap adjustments are logged separately.",
            "Open-position equity includes estimated exit commission; personal income tax is not estimated without jurisdiction/account rules.",
            "Read-only book snapshots do not prove the ability to fill later at the same price/size.",
        ],
    }


def write_report(book: PaperBook, period: str = "all") -> Path:
    report = summarize(book, period)
    directory = (
        book.root
        / "research/paper/reports"
        / (book.clock().strftime("%Y%m%d-%H%M%S") + f"-{period}-" + uuid.uuid4().hex[:8])
    )
    directory.mkdir(parents=True)
    atomic_json(directory / "report.json", report)
    rows = [
        "| Branch | Equity | Net P&L before personal tax | Costs | Max observed drawdown | Closed orders |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch, values in report["branches"].items():
        rows.append(
            f"| {branch} | {float(values['equity']):.2f} | {float(values['net_pnl_before_personal_tax']):.2f} | {float(values['modeled_costs']):.2f} | {float(values['max_observed_drawdown']) * 100:.2f}% | {values['closed_order_ids']} |"
        )
    text = (
        f"# Paper report, {report['currency']}\n\n{report['time']} | {period}\n\n"
        + "\n".join(rows)
        + "\n\n"
        + "\n".join(f"- {note}" for note in report["notes"])
    )
    text += f"\n\nStale quotes: {report['stale_symbols']}\n\nExpired costs: {report['expired_fee_profiles']}\n\nAudit: {report['audit_tail_sha256']}\n"
    (directory / "report.md").write_text(text, encoding="utf-8")
    plots = [chart(list(report["equity_series"].items()), f"Paper equity ({report['currency']})")]
    for symbol, prices in report["price_series"].items():
        markers = [
            (
                trade["time"],
                float(trade["price"]),
                f"{trade['branch']} {trade['action']} qty={trade.get('quantity', '')}",
            )
            for trade in report["trades"]
            if trade["symbol"] == symbol and "price" in trade
        ]
        plots.append(
            chart(
                [(symbol, prices)],
                f"{symbol}: observed price / odds and executed paper entries/exits",
                markers,
            )
        )
    (directory / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Paper research report</title><style>body{font:15px system-ui;max-width:1100px;margin:30px auto;padding:15px}svg{width:100%;background:#fafafa}svg text{font:12px system-ui}pre{white-space:pre-wrap}</style><h1>Paper research report</h1>'
        + "".join(plots)
        + "<pre>"
        + html.escape(text)
        + "</pre></html>",
        encoding="utf-8",
    )
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "time",
            "branch",
            "symbol",
            "action",
            "quantity",
            "price",
            "allocation",
            "payout",
            "costs",
            "net_pnl",
            "order_id",
        ],
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(report["trades"])
    (directory / "trades.csv").write_text(output.getvalue(), encoding="utf-8")
    atomic_json(
        book.root / "research/paper/latest-report.json",
        {"directory": str(directory), "time": report["time"], "period": period},
    )
    return directory


def scheduled_reports(book: PaperBook) -> list[Path]:
    """Called by the running local paper loop, not an external chat scheduler."""
    now = book.clock().astimezone(ZoneInfo("Europe/Warsaw"))
    path = book.root / "research/paper/report-schedule.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    due = []
    for period, key, eligible in (
        ("daily", now.date().isoformat(), now.hour >= 20),
        (
            "weekly",
            f"{now.isocalendar().year}-{now.isocalendar().week:02d}",
            now.weekday() == 6 and now.hour >= 20,
        ),
    ):
        if eligible and saved.get(period) != key:
            directory = write_report(book, period)
            saved[period] = key
            atomic_json(path, saved)
            due.append(directory)
    return due
