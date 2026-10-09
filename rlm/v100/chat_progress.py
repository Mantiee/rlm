"""Direct read-only progress replies, independent of model conversation drift."""

import re
from pathlib import Path

from rlm.v100.chat_goals import normalized


def requested(message: str) -> bool:
    text = normalized(message)
    return bool(
        re.search(
            r"co (?:teraz |aktualnie )?robisz|jaki (?:jest )?(?:postep|progres)|czego sie nauczyl",
            text,
        )
    ) and not re.search(r"\b(?:zmien|dodaj|napraw|zrob|wdroz|zaimplementuj)\b", text)


def continue_requested(message: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:pracuj dalej(?:,? ucz sie)?|ucz sie(?: dalej)?|kontynuuj(?: prace| nauke)?)\s*[!.]*",
            normalized(message),
        )
    )


def respond(root: Path) -> dict:
    from rlm.v100.mission_evidence import collect

    evidence = collect(root)
    learning = evidence["learning"]
    official = evidence.get("official_benchmark", {})

    def metric(value):
        return str(value) if value is not None else "brak potwierdzonych danych"

    lines = [
        "Status z plików kontrolera, bez zmiany planów:",
        "Misja: " + ("działa" if evidence["mission_running"] else "zatrzymana"),
        "Faza: " + str(evidence.get("phase")),
        "Cykle uczenia: " + metric(learning.get("completed_cycles")),
        "Potwierdzone kroki optymalizatora: " + metric(evidence["optimizer_updates_observed"]),
        "Zaakceptowane aktualizacje wag: " + metric(evidence["accepted_weight_updates_this_run"]),
    ]
    if official:
        lines += [
            f"Benchmark oficjalny: {official.get('completed', '?')}/{official.get('total', '?')} | {official.get('state', 'stan nieznany')}",
            "Raport: " + official["report"],
        ]
        if official.get("current_case"):
            lines.append("Bieżący przypadek: " + official["current_case"])
    else:
        lines.append("Benchmark oficjalny: brak licznika potwierdzonego dla bieżącego przebiegu.")
    for report in evidence["reports"]:
        if isinstance(report, dict) and report.get("cases"):
            lines.append(f"Test bazowy: {report['passed']}/{report['cases']} | {report['path']}")
    goal = evidence.get("goal_learning", {})
    lines.append("Zapisane prognozy pod cel: " + str(len(goal.get("forecasts", []))))
    lines.append("Przebieg: " + str(evidence["run"]))
    lines.append(
        "Dashboard odświeża renderer hosta; skrypty w edytowalnym HTML mastera nie są potrzebne. Brak licznika nie dowodzi bezczynności."
    )
    return {
        "answer": "\n".join(lines),
        "actions": [],
        "applied": [],
        "evidence": evidence,
        "responder": {"model": "controller-status", "delegated_while_master_busy": False},
    }
