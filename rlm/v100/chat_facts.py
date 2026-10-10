"""Local factual answers do not depend on model availability or narratives."""

import re
from pathlib import Path

from rlm.v100.income_policy import normalized


def language(message: str) -> str:
    return (
        "Polish"
        if re.search(
            r"\b(?:co|czemu|jak|czy|kiedy|gdzie|zarab|zarabianiem|zrob|zmien|przetestuj|poszukaj|zbieraj|hej|czesc|odpowiedz)\w*\b",
            normalized(message),
        )
        else "English"
    )


def requested(message: str) -> bool:
    return bool(
        re.search(
            r"zarab|income|profit|backtest|minus|paper|opportunit|candidate|brier|kandydat",
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
        "Brak potwierdzonego rzeczywistego dochodu. Nie ma wiarygodnej daty pierwszego zarobku."
        if pl
        else "No actual income verified. There is no verified date of first income.",
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
