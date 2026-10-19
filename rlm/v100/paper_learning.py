"""Bridge paper/income R&D to verified learning, with read-only background observations."""

import json
import re
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import requests

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json
from rlm.v100.paper import PaperBook, sha
from rlm.v100.paper_agents import loop_lease, paper_round
from rlm.v100.paper_feeds import poll_crypto, poll_filings
from rlm.v100.paper_reports import scheduled_reports, write_report

FIELDS = {
    "schema",
    "objective",
    "crypto",
    "ciks",
    "sec_contact",
    "observer_interval",
    "research_rounds",
    "other_income_rnd",
}


def readiness(root: Path, profile: dict, datasets: Path | None = None) -> dict:
    """Inspect paths and bounded filenames, never dataset contents or weights."""
    base = Path(profile["training"]["base_model"])
    helper = root / "research/researcher-cpu.toml"
    paper = {"initialized": False, "instruments": 0, "fee_profiles": 0}
    if (root / "research/paper/ledger.sqlite3").exists():
        book = PaperBook(root)
        try:
            state = book.state()
            paper = {
                "initialized": True,
                "instruments": len(state["instruments"]),
                "fee_profiles": len(state["fee_profiles"]),
            }
        finally:
            book.close()
    locations = [root / "research"]
    if datasets is not None:
        locations.append(datasets.expanduser().resolve())
    files = []
    for location in locations:
        if not location.is_dir():
            continue
        for path in sorted(location.iterdir())[:200]:
            if path.is_file() and path.suffix.lower() in (".gz", ".jsonl", ".toml", ".json"):
                files.append({"path": str(path), "bytes": path.stat().st_size})
    return {
        "paper": paper,
        "training_model": {"path": str(base), "config_exists": (base / "config.json").is_file()},
        "cpu_helper": {"path": str(helper), "profile_exists": helper.is_file()},
        "goal_exists": (root / "research/goal.json").is_file(),
        "available_files": files,
        "note": "Inspection only. Need a verified training pool, fixed development suite and baseline reports before learn-loop. Does not start inference, learning or trading.",
    }


def load_settings(path: Path) -> dict:
    if path.stat().st_size > 16384:
        raise ValueError("Paper learning configuration exceeds 16 KiB")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Incomplete or unknown paper learning settings")
    if value["schema"] != "v100-paper-learning-v1":
        raise ValueError("Unsupported paper learning schema")
    if not isinstance(value["objective"], str) or not 1 <= len(value["objective"].strip()) <= 2000:
        raise ValueError("Income objective needs 1-2000 characters")
    if any(type(value[key]) is not bool for key in ("crypto", "other_income_rnd")):
        raise ValueError("Feed and income-R&D flags must be boolean")
    if (
        type(value["observer_interval"]) is not int
        or not 30 <= value["observer_interval"] <= 300
        or type(value["research_rounds"]) is not int
        or not 1 <= value["research_rounds"] <= 6
    ):
        raise ValueError("Use 30-300 seconds for observations and 1-6 research rounds")
    if (
        not isinstance(value["ciks"], list)
        or len(value["ciks"]) > 10
        or any(
            not isinstance(cik, str) or not re.fullmatch(r"\d{1,10}", cik) for cik in value["ciks"]
        )
        or len(set(value["ciks"])) != len(value["ciks"])
    ):
        raise ValueError("Use up to ten distinct numeric SEC CIKs")
    contact = value["sec_contact"]
    if not isinstance(contact, str) or (
        value["ciks"] and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", contact)
    ):
        raise ValueError("SEC polling requires your real contact email")
    return value


class PaperLearning:
    def __init__(self, root: Path, path: Path):
        self.root, self.path = root, path.resolve()
        self.settings = load_settings(self.path)
        self.settings_sha = sha(self.settings)
        self.stop_event = threading.Event()
        self.fee_retry_after = 0.0
        self.spot_rules_status_path = root / "research/paper/spot-rules-status.json"
        spot_status = (
            json.loads(self.spot_rules_status_path.read_text())
            if self.spot_rules_status_path.exists()
            else {}
        )
        self.spot_rules_blocked = spot_status.get("state") == "blocked"
        self.failure: Exception | None = None
        self.worker: threading.Thread | None = None
        self.scope = ExitStack()
        if not (root / "research/paper/ledger.sqlite3").exists():
            raise ValueError("Run paper-init before enabling paper learning")
        book = PaperBook(root)
        try:
            state = book.state()
        finally:
            book.close()
        if self.settings["crypto"] and not any(
            item["market"] == "crypto"
            and item["product"] == "spot"
            and item["feed_id"].startswith(("coinbase:", "kraken:"))
            for item in state["instruments"].values()
        ):
            raise ValueError("Register verified Coinbase spot instruments and fees before polling")

    def check(self) -> None:
        if sha(load_settings(self.path)) != self.settings_sha:
            raise ValueError("Income objective or observer settings changed; start a separate run")
        if self.failure is not None:
            raise RuntimeError(
                "Paper observer failed; inspect activity logs before continuing"
            ) from self.failure
        if self.worker is not None and not self.worker.is_alive() and not self.stop_event.is_set():
            raise RuntimeError("Paper observer stopped unexpectedly; learning cannot continue")

    def note(self, kind: str, payload: dict) -> None:
        book = PaperBook(self.root)
        try:
            book.note("controller", {"status": kind, **payload})
        finally:
            book.close()
        ActivityLog(self.root, "controller", "paper-learning").write("steps", kind, payload)

    def decision_round(self, profile: dict, cycle: int) -> dict:
        """Run and receipt one bounded A/B model decision pass on the serving endpoint."""
        from rlm.v100.paper_agents import paper_decision_round

        started = time.time()
        model = profile.get("runtime", {}).get("model_version", "unknown")
        self.note(
            "decision-round-started",
            {
                "cycle": cycle,
                "started_at": started,
                "model_version": model,
                "branches": ["A", "B"],
                "scope": "Paper proposals only; host cost and risk gates still apply; no real orders.",
            },
        )
        try:
            self.check()
            book = PaperBook(self.root)
            try:
                events = paper_decision_round(book, profile)
            finally:
                book.close()
            outcome = {
                "status": "decision-round-completed",
                "cycle": cycle,
                "started_at": started,
                "completed_at": time.time(),
                "model_version": model,
                "decisions_recorded": len(events),
                "branch_actions": [
                    {
                        "branch": event.get("branch"),
                        "action": event.get("payload", {}).get("proposal", {}).get("action"),
                        "sequence": event.get("sequence"),
                    }
                    for event in events
                ],
                "scope": "Paper decisions only. A HOLD or proposal is not a fill, profit, model update, or real order.",
            }
        except Exception as error:
            outcome = {
                "status": "decision-round-unavailable",
                "cycle": cycle,
                "started_at": started,
                "completed_at": time.time(),
                "model_version": model,
                "error": type(error).__name__,
                "detail": str(error)[:400],
                "scope": "Decision attempt failed; no trade is inferred or fabricated.",
            }
        self.note(outcome["status"], outcome)
        return outcome

    def tick(self) -> None:
        self.check()
        book = PaperBook(self.root)
        try:
            if self.settings["crypto"]:
                state = book.state()
                kraken = [
                    item
                    for item in state["instruments"].values()
                    if item["feed_id"].startswith("kraken:")
                ]
                if kraken:
                    from datetime import UTC, datetime, timedelta

                    from rlm.v100.spot_bootstrap import FeeSourceUnavailable, prepare

                    if time.time() >= self.fee_retry_after and (
                        self.spot_rules_blocked
                        or any(
                            datetime.fromisoformat(
                                state["fee_profiles"][item["fee_profile"]]["valid_until"]
                            )
                            <= datetime.now(UTC) + timedelta(hours=1)
                            for item in kraken
                        )
                    ):
                        try:
                            prepare(self.root, refresh=True)
                        except FeeSourceUnavailable as error:
                            self.fee_retry_after = time.time() + 3600
                            self.note(
                                "fee-refresh-blocked",
                                {
                                    "error": type(error).__name__,
                                    "detail": str(error)[:500],
                                    "retry_at": self.fee_retry_after,
                                    "scope": "Existing fee expiry and fill gates unchanged. Quotes, research and independent verified learning may continue; no guessed fees or expiry extension.",
                                },
                            )
                        except ValueError as error:
                            if str(error) != (
                                "Spot execution rules changed; retain positions and inspect "
                                "before updating"
                            ):
                                raise
                            # Preserve all positions and fee gates. Existing spot
                            # rules cannot authorize another simulated fill.
                            self.spot_rules_blocked = True
                            self.fee_retry_after = time.time() + 3600
                            atomic_json(
                                self.spot_rules_status_path,
                                {
                                    "state": "blocked",
                                    "reason": str(error),
                                    "retry_at": self.fee_retry_after,
                                },
                            )
                            self.note(
                                "spot-rules-blocked",
                                {
                                    "detail": str(error),
                                    "retry_at": self.fee_retry_after,
                                    "scope": "Kraken/crypto paper polling and fills suspended; ledger, balances and positions unchanged; independent learning continues.",
                                },
                            )
                        else:
                            self.spot_rules_blocked = False
                            atomic_json(
                                self.spot_rules_status_path,
                                {"state": "cleared", "updated": time.time()},
                            )
                if not self.spot_rules_blocked:
                    poll_crypto(book, self.stop_event.is_set)
            from rlm.v100.provider_registry import poll as poll_registered
            from rlm.v100.reward_policy import train as train_reward

            poll_registered(book, self.stop_event.is_set)
            from rlm.v100.market_adapters import poll as poll_market_adapters

            poll_market_adapters(book, self.stop_event.is_set)
            for branch in ("A", "B"):
                if self.stop_event.is_set():
                    return
                result = train_reward(self.root, branch)
                atomic_json(self.root / "research/reward-policies" / branch / "latest.json", result)
            for cik in self.settings["ciks"]:
                if self.stop_event.is_set():
                    return
                poll_filings(book, cik, self.settings["sec_contact"], self.stop_event.is_set)
            if not self.stop_event.is_set():
                for directory in scheduled_reports(book):
                    print("Scheduled paper report:", directory, flush=True)
        finally:
            book.close()

    def observe(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.tick()
            except requests.RequestException as error:
                # A transport outage is logged, never filled with synthetic prices.
                self.note("feed-unavailable", {"error": type(error).__name__})
            except Exception as error:
                self.failure = error
                ActivityLog(self.root, "controller", "paper-learning").write(
                    "errors",
                    "observer-failed",
                    {"error": type(error).__name__, "detail": str(error)[:400]},
                )
                return
            self.stop_event.wait(self.settings["observer_interval"])

    def __enter__(self):
        self.scope.enter_context(loop_lease(self.root))
        try:
            directory = self.root / "research/paper/learning-settings"
            snapshot = directory / (self.settings_sha + ".json")
            if snapshot.exists():
                if json.loads(snapshot.read_text()) != self.settings:
                    raise ValueError("Paper learning settings snapshot changed")
            else:
                atomic_json(snapshot, self.settings)
            self.note(
                "paper-learning-started",
                {
                    "objective": self.settings["objective"],
                    "settings_sha256": self.settings_sha,
                    "scope": "Verified formal learning and paper/income hypotheses; no real orders",
                },
            )
            # One data-observer thread, no additional model or GPU context.
            self.worker = threading.Thread(target=self.observe, name="paper-observer", daemon=True)
            self.worker.start()
            return self
        except BaseException:
            self.scope.close()
            raise

    def research(self, profile: dict, helper: dict, branches: tuple[str, ...] = ("A", "B")) -> None:
        self.check()
        from rlm.v100.goals import load_goal

        objective = (load_goal(self.root) or {}).get("text", self.settings["objective"])
        book = PaperBook(self.root)
        try:
            book.note(
                "controller",
                {
                    "status": "income-objective",
                    "objective": objective,
                    "model_version": profile["runtime"]["model_version"],
                    "profile_sha256": sha(profile),
                },
            )
            paper_round(
                book,
                profile,
                helper,
                self.settings["research_rounds"],
                income_research=self.settings["other_income_rnd"],
                objective=objective,
                **({"branches": branches} if branches != ("A", "B") else {}),
            )
            from rlm.v100.progress import report

            report(self.root)
        finally:
            book.close()

    def phase(self, phase: str, cycle: int, profile: dict) -> None:
        self.check()
        self.note(
            phase,
            {
                "cycle": cycle,
                "model_version": profile["runtime"]["model_version"],
                "profile_sha256": sha(profile),
                "settings_sha256": self.settings_sha,
            },
        )

    def __exit__(self, error_type, error, traceback):
        self.stop_event.set()
        try:
            if self.worker is not None:
                self.worker.join(timeout=45)
                if self.worker.is_alive():
                    raise TimeoutError(
                        "Observer did not stop; no further learning run should start"
                    )
            if error_type is None:
                self.check()
                book = PaperBook(self.root)
                try:
                    print("Final paper report:", write_report(book), flush=True)
                finally:
                    book.close()
        finally:
            self.scope.close()
        return False
