"""Direct read-only progress replies, independent of model conversation drift."""

import json
import re
from pathlib import Path

from rlm.v100.chat_goals import normalized


def requested(message: str) -> bool:
    text = normalized(message)
    return bool(
        re.search(
            r"co (?:teraz |aktualnie )?robisz|jaki (?:jest )?(?:postep|progres)|czego sie nauczyl|czy .*uczysz sie|czy uczysz sie|learn.*continuously|what have you learned|what are you doing",
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


def respond(root: Path, message: str | None = None) -> dict:
    from rlm.v100.chat_facts import language
    from rlm.v100.mission_evidence import collect

    evidence = collect(root)
    learning = evidence["learning"]
    official = evidence.get("official_benchmark", {})
    pl = message is None or language(message) == "Polish"

    def wording(polish: str, english: str) -> str:
        return polish if pl else english

    def metric(value):
        return (
            str(value)
            if value is not None
            else wording("brak potwierdzonych danych", "no verified data")
        )

    lines = [
        wording(
            "Pętla badań działa osobno od treningu wag. Trening wymaga nowych zweryfikowanych danych; nie odbywa się przy każdej odpowiedzi.",
            "Research runs separately from weight training. Training requires new verified data; answering a message is not an optimizer update.",
        ),
        wording("Misja: ", "Mission: ")
        + (
            wording("działa", "running")
            if evidence["mission_running"]
            else wording("zatrzymana", "stopped")
        ),
        wording("Faza: ", "Phase: ") + str(evidence.get("phase")),
        wording("Cykle pętli zakończone: ", "Completed loop cycles: ")
        + metric(learning.get("completed_cycles")),
        wording("Potwierdzone kroki optymalizatora: ", "Verified optimizer updates: ")
        + metric(evidence["optimizer_updates_observed"]),
        wording("Zaakceptowane aktualizacje wag: ", "Accepted weight updates: ")
        + metric(evidence["accepted_weight_updates_this_run"]),
    ]
    if official:
        lines += [
            wording("Benchmark oficjalny: ", "Official benchmark: ")
            + f"{official.get('completed', '?')}/{official.get('total', '?')} | {official.get('state', 'unknown')}",
            wording("Raport: ", "Report: ") + official["report"],
        ]
        if official.get("current_case"):
            lines.append(
                wording("Bieżący przypadek: ", "Current case: ") + official["current_case"]
            )
    else:
        lines.append(
            wording(
                "Benchmark oficjalny: brak licznika potwierdzonego dla bieżącego przebiegu.",
                "Official benchmark: no verified current-run counter.",
            )
        )
    for report in evidence["reports"]:
        if isinstance(report, dict) and report.get("cases"):
            lines.append(
                wording("Test bazowy: ", "Baseline: ")
                + f"{report['passed']}/{report['cases']} | {report['path']}"
            )
    goal = evidence.get("goal_learning", {})
    lines.append(
        wording("Prognozy w ostatnim oknie statusu: ", "Forecasts in the recent status window: ")
        + str(len(goal.get("forecasts", [])))
    )
    if not evidence["accepted_weight_updates_this_run"]:
        lines.append(
            wording(
                "Nie mam potwierdzenia nowych zaakceptowanych wag w tym przebiegu. Zapisanie prognozy lub planu nie jest treningiem.",
                "No newly accepted weights verified in this run. Recording a forecast or plan is not training.",
            )
        )
    if evidence.get("run"):
        path = Path(evidence["run"]) / "learning/heartbeat.json"
        if path.exists() and path.stat().st_size <= 16384:
            try:
                activity = json.loads(path.read_text())
                lines.append(
                    wording("Ostatni etap pętli: ", "Last loop stage: ")
                    + str(activity.get("stage"))
                )
                if activity.get("reason"):
                    lines.append(
                        wording("Powód oczekiwania: ", "Wait reason: ") + activity["reason"]
                    )
            except (OSError, ValueError) as error:
                lines.append(
                    wording("Status etapu niedostępny: ", "Stage status unavailable: ")
                    + str(error)[:200]
                )
    lines.append(wording("Przebieg: ", "Run: ") + str(evidence["run"]))
    return {
        "answer": "\n".join(lines),
        "actions": [],
        "applied": [],
        "evidence": evidence,
        "responder": {"model": "controller-status", "delegated_while_master_busy": False},
    }
