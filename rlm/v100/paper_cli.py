"""Paper lab CLI, separate from weight training and real financial accounts."""

import hashlib
import json
from pathlib import Path

from rlm.v100.common import load_profile
from rlm.v100.paper import PaperBook


def add_commands(sub) -> None:
    initialize = sub.add_parser("paper-init", help="Create equal A/B paper portfolios; no deposits")
    initialize.add_argument("--capital", default="10000")
    initialize.add_argument("--currency", default="PLN")
    configure = sub.add_parser(
        "paper-configure", help="Register operator-verified fee/instrument rules"
    )
    configure.add_argument("file", type=Path)
    ingest = sub.add_parser(
        "paper-ingest", help="Import timestamped independent observations as JSONL"
    )
    ingest.add_argument("file", type=Path)
    source = sub.add_parser(
        "paper-source", help="Archive a public documentation source for fee research"
    )
    source.add_argument("url")
    report = sub.add_parser(
        "paper-report", help="Write audited HTML/SVG, Markdown, JSON and trades CSV"
    )
    report.add_argument("--period", choices=("all", "daily", "weekly"), default="all")
    status = sub.add_parser(
        "paper-learning-status",
        help="Inspect learning inputs without starting models or reading datasets",
    )
    status.add_argument("--datasets", type=Path)
    for name in ("paper-round", "paper-loop"):
        command = sub.add_parser(
            name, help="Financial A/B research and forward-only paper proposals"
        )
        command.add_argument("--researcher-profile", type=Path)
        command.add_argument("--research-rounds", type=int, default=2)
        if name == "paper-loop":
            command.add_argument("--interval", type=int, default=300)
            command.add_argument("--cycles", type=int, default=0)
            feed_arguments(command)
    poll = sub.add_parser(
        "paper-poll", help="Fetch free read-only crypto books / SEC filing metadata"
    )
    feed_arguments(poll)


def feed_arguments(parser) -> None:
    parser.add_argument("--crypto", action="store_true")
    parser.add_argument("--cik", action="append", default=[])
    parser.add_argument(
        "--sec-contact", help="Your real contact email, required by SEC access policy"
    )


def handle(args, root: Path) -> None:
    if args.command == "paper-learning-status":
        from rlm.v100.paper_learning import readiness

        profile = load_profile(args.profile or root / "research/v100.toml", root)
        print(json.dumps(readiness(root, profile, args.datasets), ensure_ascii=False, indent=2))
        return
    if args.command == "paper-source":
        from rlm.v100.research_tools import ResearchTools

        print(
            json.dumps(
                ResearchTools(root, {}).execute("read_public_page", {"url": args.url}),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "paper-loop":
        from rlm.v100.paper_agents import paper_loop

        profile = load_profile(args.profile or root / "research/v100.toml", root)
        paper_loop(
            root,
            profile,
            args.researcher_profile,
            args.interval,
            args.cycles,
            args.research_rounds,
            args.crypto,
            tuple(args.cik),
            args.sec_contact,
        )
        return
    book = PaperBook(root)
    try:
        if args.command == "paper-init":
            book.initialize(args.capital, args.currency)
            print("PAPER LAB READY: equal A/B portfolios,", args.capital, args.currency)
            print(
                "No trades enabled until documented fee profiles, instruments and fresh quotes exist."
            )
            print(
                "Read PAPER_RESEARCH.md before configuring feeds; no real order endpoints installed."
            )
        elif args.command == "paper-configure":
            if args.file.stat().st_size > 1024 * 1024:
                raise ValueError("Configuration exceeds 1 MiB")
            book.configure(json.loads(args.file.read_text()))
            print("Operator-owned execution rules registered")
        elif args.command == "paper-ingest":
            if args.file.stat().st_size > 8 * 2**20:
                raise ValueError("Import exceeds 8 MiB")
            raw = args.file.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            directory = root / "research/paper/imports"
            directory.mkdir(parents=True, exist_ok=True)
            snapshot = directory / f"{digest}.jsonl"
            if not snapshot.exists():
                with snapshot.open("xb") as file:
                    file.write(raw)
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                raise ValueError("Imported source changed")
            for index, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
                if line.strip():
                    event = book.ingest(json.loads(line))
                    print("Accepted row", index, "ledger sequence", event["sequence"])
        elif args.command == "paper-report":
            from rlm.v100.paper_reports import write_report

            print("Paper report:", write_report(book, args.period))
        elif args.command == "paper-round":
            from rlm.v100.paper_agents import paper_round

            profile = load_profile(args.profile or root / "research/v100.toml", root)
            helper = (
                load_profile(args.researcher_profile, root) if args.researcher_profile else None
            )
            if helper and helper["server"].get("gpu_layers") != 0:
                raise ValueError("Financial helper profile must use CPU")
            print(
                json.dumps(
                    paper_round(book, profile, helper, args.research_rounds),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "paper-poll":
            from rlm.v100.paper_feeds import poll_crypto, poll_filings

            if not args.crypto and not args.cik:
                raise ValueError("Choose a configured crypto feed or CIK")
            if args.crypto:
                print("Crypto observations:", len(poll_crypto(book)))
            for cik in args.cik:
                print(
                    "New filing observations:", len(poll_filings(book, cik, args.sec_contact or ""))
                )
    finally:
        book.close()
