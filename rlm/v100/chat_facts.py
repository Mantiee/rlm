"""Local factual answers do not depend on model availability or narratives."""

import re
from pathlib import Path

from rlm.v100.income_policy import normalized


def language(message: str) -> str:
    return (
        "Polish"
        if re.search(
            r"\b(?:co|czego|czemu|jak|czy|kiedy|gdzie|uczy|naucz|zarab|zarob|znalaz|zrob|zmien|przetestuj|poszukaj|zbieraj|sprawdz|zbadaj|odchylen|hej|czesc|odpowiedz)\w*\b",
            normalized(message),
        )
        else "English"
    )


def requested(message: str) -> bool:
    return bool(
        re.search(
            r"zarab|zarob|strateg|income|profit|backtest|minus|paper|opportunit|candidate|brier|kandydat|drawdown|obsunie|odchylen|glitch",
            normalized(message),
        )
    )


def response(root: Path, message: str) -> dict:
    from rlm.v100.income_opportunities import status
    from rlm.v100.progress import snapshot

    value = snapshot(root)
    board = status(root)
    pl = language(message) == "Polish"
    paper = value.get("paper", {})
    lines = [
        "Oceniam wyniki symulacji i strategii, przy założeniu hipotetycznego kapitału. Brak realnych zleceń nie wyjaśnia strat w paper."
        if pl
        else "I assess simulated capital growth and strategy results. No real orders does not explain paper losses.",
        ("Faza: " if pl else "Phase: ") + str(value.get("phase")),
        ("Wykonania paper: " if pl else "Paper fills: ")
        + str(paper.get("executed_fills", "unknown")),
        "Zero wykonań nie potwierdza strategii. Prognozy nie są transakcjami."
        if pl
        else "Zero fills do not validate a strategy. Forecasts are not trades.",
        ("Blokady: " if pl else "Blockers: ")
        + (", ".join(value["paper_blockers"]) if "paper_blockers" in value else "unknown"),
        ("Czas zapisanej kontroli: " if pl else "Stored audit timestamp: ")
        + str(value.get("updated_at", "unavailable")),
        ("Zarejestrowane hipotezy dla celu: " if pl else "Registered current-goal hypotheses: ")
        + str(len(board["candidates"])),
        "Ujemny zwrot backtestu jest stratą symulacji historycznej. Nie jest wynikiem Brier ani benchmarku."
        if pl
        else "A negative backtest return is a historical simulated loss, not a Brier or benchmark score.",
    ]
    lines.append(
        "Obsunięcie to największy spadek kapitału od wcześniejszego szczytu, nie odchylenie standardowe. Wynik netto przy podwójnych kosztach to oddzielny zwrot. Sama strata nie dowodzi błędu danych."
        if pl
        else "Drawdown is the largest equity decline from a prior peak, not standard deviation. Double-cost net return is a separate return. A loss alone does not prove corrupt data."
    )
    audit_path = root / "research/backtests/audit-status.json"
    if audit_path.exists() and audit_path.stat().st_size <= 2 * 2**20:
        import json

        audit = json.loads(audit_path.read_text())
        for row in audit.get("reports", [])[:4]:
            lines.append(
                "Audit: "
                + row["state"]
                + " | "
                + row["report"]
                + " | "
                + row.get("reason", "archived simulator replay only")
            )
    else:
        lines.append("Audit: unavailable; no source integrity claim accepted")
    for row in value.get("recent_exploratory_backtests", []):
        net = row.get("development_test", {}).get("net_return")
        lines.append(
            str(row.get("parameters", {}).get("rule"))
            + ": "
            + (f"{net:.2%}" if type(net) in (int, float) else "unknown")
            + " | "
            + str(row.get("report"))
        )
    return {
        "answer": "\n".join(lines),
        "actions": [],
        "applied": [],
        "facts": value,
        "responder": {"model": "controller-financial-evidence"},
        "scope": "Read-only host evidence; no plan, weight, order or spending changes",
    }


def test_response(root: Path, message: str) -> dict:
    from rlm.v100.market_research import compare

    value = compare(root)
    lines = [value["state"], value.get("error", value["scope"])]
    for row in value["comparisons"]:
        net = row["development_test"].get("net_return")
        lines.append(
            row["parameters"]["rule"]
            + ": "
            + (f"{net:.2%}" if type(net) in (int, float) else "unknown")
            + " | "
            + row["triage"]["state"]
            + " | "
            + row["report"]
        )
    return {
        "answer": "\n".join(lines),
        "actions": [],
        "applied": [value],
        "responder": {"model": "controller-historical-comparison"},
        "scope": "Historical research executed; no future edge, verified income or orders claimed",
    }


def audit_response(root: Path, message: str) -> dict:
    from rlm.v100.backtest_audit import run

    value = run(root)
    rows = [
        "Audit: "
        + row["state"]
        + " | "
        + row["report"]
        + " | "
        + row.get("reason", "archived replay passed")
        for row in value["reports"]
    ]
    return {
        "answer": (
            "Obsunięcie to największy spadek kapitału od wcześniejszego szczytu, nie odchylenie standardowe. Zwrot przy podwójnych kosztach jest osobną metryką. Sama strata nie dowodzi błędu cen. Audyt poniżej odtwarza zapisane źródła i obliczenia; nie potwierdza ich u niezależnego dostawcy."
            if language(message) == "Polish"
            else value["metric_correction"]
        )
        + "\n"
        + ("\n".join(rows) or "No archived reports to audit"),
        "actions": [],
        "applied": [value],
        "responder": {"model": "controller-backtest-audit"},
        "scope": value["scope"],
    }
