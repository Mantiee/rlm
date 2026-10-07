"""Bounded public-source reads and isolated code checks; no paid API clients."""

import copy
import hashlib
import ipaddress
import json
import re
import socket
import ssl
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import urllib3

from rlm.v100.activity import ActivityLog
from rlm.v100.agent import native_turn, research_output_limit, tool_schema, tool_turn
from rlm.v100.tool_protocol import json_object

COMPACT_CPU_TOOLS = {
    "search_memory",
    "read_source",
    "read_public_page",
    "paper_status",
    "paper_observed_results",
    "paper_test_position",
    "check_code_candidate",
    "backtest_prices",
}

TOOLS = [
    tool_schema(
        "discover_spot_markets",
        "Discover public USD spot pair codes and venue minimums from primary Kraken data; choose what to research rather than a fixed crypto list.",
        {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
    ),
    tool_schema(
        "register_paper_spot",
        "Register 1-8 selected USD spot pairs using fresh primary rules and lowest-volume fee evidence. Fully paid PAPER only, existing risk limits and quote checks apply. Never certify income or account entitlement.",
        {
            "pair_codes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {"type": "string"},
            }
        },
    ),
    tool_schema(
        "propose_colab_trial",
        "Queue a short optional interactive Colab tiny-model training pilot. User opens notebook; this does not create accounts, remote workers, bypass quotas or replace the main model.",
        {
            "steps": {"type": "integer", "minimum": 10, "maximum": 100},
            "purpose": {"type": "string", "maxLength": 600},
        },
    ),
    tool_schema(
        "list_algorithm_files",
        "List editable own algorithm source; protected system prompts, host controllers and evaluators stay read-only.",
        {},
    ),
    tool_schema(
        "read_algorithm_file",
        "Read one tracked editable algorithm file from the pinned experimental source.",
        {"filename": {"type": "string"}},
    ),
    tool_schema(
        "create_code_candidate",
        "Propose one exact algorithm edit in an isolated copy. Run check_code_candidate afterward; independent training and task-quality gates are still required.",
        {
            "filename": {"type": "string"},
            "find": {"type": "string", "maxLength": 6000},
            "replace": {"type": "string", "maxLength": 6000},
            "hypothesis": {"type": "string", "maxLength": 1200},
        },
    ),
    tool_schema(
        "calculate",
        "Calculate explicit costs and percentages using decimal arithmetic and + - * / parentheses. Inputs are assumptions, not market predictions.",
        {"expression": {"type": "string", "maxLength": 512}},
    ),
    tool_schema(
        "set_research_budget",
        "Choose thinking and output budget for master V100 or helper RTX R&D. Helper batch is 16 and active-time target at most 30% after crashes. Master batch is a proposal requiring a stopped-server benchmark; evaluations stay fixed. Use thinking for difficult analysis and disable for fast extraction.",
        {
            "target": {"type": "string", "enum": ["master", "helper"]},
            "thinking": {"type": "boolean"},
            "max_tokens": {"type": "integer", "minimum": 256, "maximum": 8192},
            "batch_tokens": {"type": "integer", "enum": [16, 128, 256, 512]},
        },
    ),
    tool_schema(
        "parallel_source_research",
        "Send up to eight public URLs to lightweight source drones, at most two concurrent fetches. Returns bounded source excerpts, not verified conclusions. No arbitrary installed code, paid services or browser accounts.",
        {"urls": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8}},
    ),
    tool_schema(
        "backtest_prices",
        "Run chronological historical spot-price research on public JSON OHLC data. Choose instrument/source and rule. Official Coinbase candles work, e.g. https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600. Other sources need open_time, available_at, open, close fields per row. Optional event_url must return published_at and available_at per event; empty string disables event filter. Select lookback using only prior data in three disjoint walk-forward windows, compare cash/buy-hold and doubled costs. Duplicate data/parameters reuse the pinned report. Costs are assumptions, not certified fees. Not for sports odds or leverage. Results are exploratory hypotheses, never automatic profit labels.",
        {
            "price_url": {"type": "string"},
            "event_url": {"type": "string"},
            "rule": {"type": "string", "enum": ["momentum", "mean_reversion", "buy_hold"]},
            "fee_bps": {"type": "number", "minimum": 0, "maximum": 1000},
            "slippage_bps": {"type": "number", "minimum": 0, "maximum": 1000},
        },
    ),
    tool_schema(
        "search_memory",
        "Search preserved original research sources and public A/B findings.",
        {"query": {"type": "string"}},
    ),
    tool_schema(
        "read_source",
        "Read an original research passage previously returned by search_memory.",
        {"source_id": {"type": "string"}},
    ),
    tool_schema(
        "paper_status",
        "Inspect your current forward paper portfolio and fixed risk budget. Read-only, no real money.",
        {},
    ),
    tool_schema(
        "paper_observed_results",
        "Inspect audited past paper results for A/B, costs, data freshness and limitations. Past results do not certify a repeatable edge.",
        {},
    ),
    tool_schema(
        "paper_test_position",
        "Independently calculate an estimated position and an instantaneous adverse-price scenario using registered fees and the latest quote. Hypothetical only, no order, no future outcome or automatic training label.",
        {
            "symbol": {"type": "string"},
            "budget": {"type": "number"},
            "leverage": {"type": "number"},
            "side": {"type": "string", "enum": ["long", "short"]},
            "adverse_bps": {"type": "number", "minimum": 0, "maximum": 10000},
        },
    ),
    tool_schema(
        "list_free_models",
        "Search the live catalog of zero-price OpenRouter text models. Provide an empty query or a model/name substring. No intelligence ranking. Consultation requires a locally configured free-tier account key.",
        {"query": {"type": "string"}},
    ),
    tool_schema(
        "consult_free_model",
        "Ask a selected explicit :free model for advice. Only verified zero-price catalog entries and a free-tier account are accepted; provider max prices are zero, no paid fallback. Answer remains unverified research data.",
        {"model_id": {"type": "string"}, "question": {"type": "string"}},
    ),
    tool_schema(
        "list_free_services",
        "Inspect researched browser-service proposals, discovery documentation and peer selections. Browser login is not installed; dedicated free API models are listed separately with list_free_models.",
        {},
    ),
    tool_schema(
        "propose_free_service",
        "Research any potentially useful free LLM service. Read its documentation with read_public_page first. This proposes a service; it cannot authorize cost or log in.",
        {
            "name": {"type": "string"},
            "url": {"type": "string"},
            "documentation": {"type": "string"},
            "rationale": {"type": "string"},
        },
    ),
    tool_schema(
        "choose_free_service",
        "Choose a previously researched service proposal for this task; may change later. State evidence and avoid claiming unmeasured quality. Records preference only, does not consult the service or create accounts.",
        {
            "candidate_id": {"type": "string"},
            "purpose": {"type": "string"},
            "rationale": {"type": "string"},
        },
    ),
    tool_schema(
        "read_public_page",
        "Read an HTTPS public source URL; no accounts or credentials.",
        {"url": {"type": "string"}},
    ),
    tool_schema(
        "check_code_candidate",
        "Run fixed tests for an existing lab code candidate in bubblewrap; no arbitrary host execution.",
        {"candidate_id": {"type": "string"}},
    ),
    tool_schema(
        "create_submodel",
        "Create isolated Python architecture code: build(config) returns a causal torch.nn.Module, input integer byte tokens [B,T], output logits [B,T,257]. No execution on the host. Full weights can learn in a small CPU pilot.",
        {
            "candidate_id": {"type": "string"},
            "code": {"type": "string"},
            "hypothesis": {"type": "string"},
            "joint": {"type": "boolean"},
        },
    ),
    tool_schema(
        "support_submodel",
        "Agree to a peer's joint CPU submodel trial with the fixed pilot budget. Both A and B must agree before a shared trial runs.",
        {"candidate_id": {"type": "string"}},
    ),
    tool_schema(
        "test_submodel",
        "Train a previously created submodel for 40 steps on CPU with 2 threads and evaluate against the fixed lab goal. Maximum 2M parameters, 256 byte context, 120s per phase, 8GiB virtual RAM. Requires a prepared workload and 12GiB available host RAM.",
        {"candidate_id": {"type": "string"}},
    ),
]


def public_origin(url: str) -> tuple[object, str]:
    origin = urlparse(url)
    if (
        origin.scheme != "https"
        or not origin.hostname
        or origin.username
        or origin.password
        or origin.port not in (None, 443)
        or len(url) > 2048
    ):
        raise ValueError("Research URLs must be public HTTPS origins without credentials")
    addresses = {
        item[4][0] for item in socket.getaddrinfo(origin.hostname, 443, type=socket.SOCK_STREAM)
    }
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("Research cannot read local, private or reserved network addresses")
    return origin, sorted(addresses)[0]


def download_page(url: str, max_bytes: int = 262144) -> tuple[str, str]:
    if type(max_bytes) is not int or not 262144 <= max_bytes <= 2 * 2**20:
        raise ValueError("Invalid bounded source read budget")
    for _ in range(3):
        origin, address = public_origin(url)
        # Connect to the already checked IP, retaining hostname verification and
        # SNI. DNS rebinding cannot reroute this request to a private address.
        pool = urllib3.HTTPSConnectionPool(
            address,
            port=443,
            assert_hostname=origin.hostname,
            server_hostname=origin.hostname,
            cert_reqs="CERT_REQUIRED",
            ca_certs=ssl.get_default_verify_paths().cafile,
            timeout=urllib3.Timeout(connect=10, read=20),
        )
        response = None
        try:
            target = origin.path or "/"
            if origin.query:
                target += "?" + origin.query
            response = pool.request(
                "GET",
                target,
                headers={
                    "Host": origin.hostname,
                    "User-Agent": "V100-Research/0.1",
                    "Accept-Encoding": "identity",
                },
                redirect=False,
                retries=False,
                preload_content=False,
            )
            if response.status in (301, 302, 303, 307, 308):
                url = urljoin(url, response.headers["Location"])
                continue
            if response.status != 200:
                raise ValueError(
                    f"Source returned HTTP {response.status}; no login or block bypass"
                )
            if (
                not response.headers.get("Content-Type", "")
                .lower()
                .startswith(("text/", "application/json"))
            ):
                raise ValueError(
                    "Research reader currently supports text/HTML/JSON, not PDF or binaries"
                )
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError(f"Source exceeds the {max_bytes} byte read budget")
            return url, body.decode("utf-8", errors="replace")
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise ValueError("Source redirect budget exceeded")


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ignored = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.ignored += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.ignored:
            self.ignored -= 1

    def handle_data(self, data):
        if not self.ignored and data.strip():
            self.parts.append(data.strip())


class ResearchTools:
    def __init__(self, root: Path, settings: dict, branch: str = "A"):
        self.root, self.settings, self.sources = root, settings, []
        self.branch = branch
        self.known_memory_sources = set()

    def execute(self, name: str, arguments: dict) -> dict:
        if name in ("discover_spot_markets", "register_paper_spot"):
            from rlm.v100.spot_bootstrap import discover, prepare

            if name == "discover_spot_markets" and set(arguments) == {"limit"}:
                return discover(self.root, **arguments)
            if name == "register_paper_spot" and set(arguments) == {"pair_codes"}:
                return prepare(self.root, refresh=True, **arguments)
            raise ValueError("Invalid market discovery/registration arguments")
        if name == "propose_colab_trial":
            from rlm.v100.colab_jobs import propose

            if set(arguments) != {"steps", "purpose"}:
                raise ValueError("Invalid Colab proposal")
            return propose(self.root, self.branch, **arguments)
        if name in ("list_algorithm_files", "read_algorithm_file", "create_code_candidate"):
            from rlm.v100.self_code import execute

            return execute(self.root, self.branch, name, arguments)
        if name == "calculate":
            from rlm.v100.calculator import calculate

            if set(arguments) != {"expression"}:
                raise ValueError("Invalid calculator input")
            return calculate(arguments["expression"])
        if name == "set_research_budget":
            from rlm.v100.research_policy import choose

            if set(arguments) != {"target", "thinking", "max_tokens", "batch_tokens"}:
                raise ValueError("Invalid research budget")
            return choose(self.root, **arguments)
        if name == "parallel_source_research":
            from concurrent.futures import ThreadPoolExecutor

            urls = arguments.get("urls")
            if (
                set(arguments) != {"urls"}
                or not isinstance(urls, list)
                or not 1 <= len(urls) <= 8
                or any(not isinstance(url, str) for url in urls)
            ):
                raise ValueError("Source drones require 1-8 public URLs")

            def fetch(url: str) -> dict:
                try:
                    reader = ResearchTools(self.root, self.settings, self.branch)
                    return reader.execute("read_public_page", {"url": url})
                except (ValueError, OSError, requests.RequestException) as error:
                    return {"url": url, "error": str(error)[:300], "status": "failed"}

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(fetch, dict.fromkeys(urls)))
            return {
                "sources": results,
                "scope": "Unverified source excerpts; original documents preserved",
                "workers": 2,
            }
        if name == "backtest_prices":
            from rlm.v100.backtesting import run

            try:
                return run(self.root, self.branch, arguments)
            except (KeyError, TypeError, IndexError) as error:
                raise ValueError(
                    "Backtest source does not match the required timestamped schema"
                ) from error
        if name in ("search_memory", "read_source"):
            from rlm.v100.mission_memory import store

            memory = store(self.root)
            try:
                if name == "search_memory" and set(arguments) == {"query"}:
                    from rlm.v100.mission_semantic import retrieve

                    hits = retrieve(self.root, memory, arguments["query"], 4)
                    self.known_memory_sources.update(hit["id"] for hit in hits)
                    return {
                        "passages": [
                            {
                                "id": hit["id"],
                                "preview": hit["text"][:300],
                                "source": hit["source"],
                                "provenance": hit["provenance"],
                            }
                            for hit in hits
                        ]
                    }
                if (
                    name == "read_source"
                    and set(arguments) == {"source_id"}
                    and arguments["source_id"] in self.known_memory_sources
                ):
                    row = memory.node(arguments["source_id"])
                    return {
                        key: row[key]
                        for key in ("id", "text", "document_id", "source", "provenance")
                    }
                raise ValueError("Memory read must reference a previously retrieved passage")
            finally:
                memory.close()
        if name in ("paper_status", "paper_observed_results", "paper_test_position"):
            from rlm.v100.paper_tools import execute

            return execute(self.root, self.branch, name, arguments)
        if name in ("list_free_models", "consult_free_model"):
            from rlm.v100.free_router import consult, public_catalog

            if name == "list_free_models" and set(arguments) == {"query"}:
                return public_catalog(arguments["query"])
            if name == "consult_free_model" and set(arguments) == {"model_id", "question"}:
                return consult(self.root, self.branch, **arguments)
            raise ValueError("Invalid free-model request")
        if name in ("list_free_services", "propose_free_service", "choose_free_service"):
            from rlm.v100.free_services import ServiceBook

            expected = {
                "list_free_services": set(),
                "propose_free_service": {"name", "url", "documentation", "rationale"},
                "choose_free_service": {"candidate_id", "purpose", "rationale"},
            }
            if set(arguments) != expected[name]:
                raise ValueError("Invalid service research arguments")
            book = ServiceBook(self.root)
            try:
                if name == "list_free_services":
                    return {**book.catalog(), "recent_choices": book.recent()}
                if name == "propose_free_service":
                    public_origin(arguments["url"])
                    return book.propose(**arguments)
                return book.choose(self.branch, **arguments)
            finally:
                book.close()
        if name == "create_submodel":
            from rlm.v100.architectures import create_candidate

            if set(arguments) != {"candidate_id", "code", "hypothesis", "joint"}:
                raise ValueError("Invalid submodel proposal")
            return create_candidate(self.root, self.branch, **arguments)
        if name == "support_submodel":
            from rlm.v100.architectures import support_candidate

            if set(arguments) != {"candidate_id"}:
                raise ValueError("Invalid shared experiment agreement")
            return support_candidate(self.root, self.branch, arguments["candidate_id"])
        if name == "test_submodel":
            from rlm.v100.architectures import prepared_inputs, run_candidate
            from rlm.v100.researchers import available_ram_gib

            if set(arguments) != {"candidate_id"}:
                raise ValueError("Invalid submodel test")
            if available_ram_gib() < 12:
                raise RuntimeError(
                    "Submodel trial deferred: need 12 GiB available RAM alongside the main learner"
                )
            pool, suite = prepared_inputs(self.root)
            report = run_candidate(self.root, arguments["candidate_id"], pool, suite)
            return {
                key: report[key]
                for key in ("candidate_id", "owner", "passed_cases", "training_seconds", "status")
            }
        if name == "read_public_page":
            if set(arguments) != {"url"} or not isinstance(arguments["url"], str):
                raise ValueError("Invalid public-source request")
            url, body = download_page(arguments["url"])
            parser = TextOnly()
            parser.feed(body)
            text = "\n".join(parser.parts) if parser.parts else body
            sha = hashlib.sha256(text.encode()).hexdigest()
            directory = self.root / "research/web-sources"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (sha + ".txt")
            if not path.exists():
                with path.open("x") as file:
                    file.write(text)
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError("Cached source differs from its recorded digest")
            source = {
                "url": url,
                "sha256": sha,
                "excerpt": text[:1200],
                "status": "source text, not independently verified truth",
                "fetched_at": datetime.now(UTC).isoformat(),
            }
            self.sources.append(source)
            from rlm.v100.mission_memory import archive

            source["memory_document_id"] = archive(
                self.root,
                url + "#" + sha,
                "Public research source. URL: "
                + url
                + "\nFetched at: "
                + source["fetched_at"]
                + "\n"
                + text,
            )
            from rlm.v100.free_services import ServiceBook

            book = ServiceBook(self.root)
            try:
                book.record_source(source)
            finally:
                book.close()
            return source
        if name == "check_code_candidate":
            from rlm.v100.code_lab import check_code

            if (
                set(arguments) != {"candidate_id"}
                or not isinstance(arguments["candidate_id"], str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", arguments["candidate_id"])
            ):
                raise ValueError("Invalid code-candidate identifier")
            parent = (self.root / "research/code-candidates").resolve()
            candidate = (parent / arguments["candidate_id"]).resolve()
            if not candidate.is_relative_to(parent):
                raise ValueError("Code candidate escaped its lab directory")
            verdict = check_code(candidate, timeout=60, cpu_threads=4)
            return {
                "candidate_id": arguments["candidate_id"],
                "passed": verdict["passed"],
                "scope": verdict["scope"],
            }
        raise ValueError("Unknown research capability")


def research_turn(client, messages: list[dict], schema: dict, root: Path) -> dict:
    from rlm.v100.research_policy import apply

    client = apply(client, root)
    owner = getattr(client, "research_owner", "A")
    journal = ActivityLog(root, owner, getattr(client, "activity_actor", "tester"))
    tools = ResearchTools(
        root, getattr(client, "research_config", {}), getattr(client, "research_owner", "A")
    )
    trace = []
    selected_names = getattr(client, "research_tool_names", None)
    selected_tools = [
        tool
        for tool in TOOLS
        if selected_names is None or tool["function"]["name"] in selected_names
    ]
    # Routing is a bounded read-only choice. Keep reasoning for the final analysis.
    router = copy.copy(client)
    router.sampling_args = dict(client.sampling_args)
    router.sampling_args["max_tokens"] = min(1024, client.sampling_args.get("max_tokens", 512))
    router.enable_thinking = False
    for _ in range(3):
        available_tools = copy.deepcopy(selected_tools)
        for tool in available_tools:
            if tool["function"]["name"] == "read_source":
                tool["function"]["parameters"]["properties"]["source_id"]["enum"] = sorted(
                    tools.known_memory_sources
                )
        available_tools = [
            tool
            for tool in available_tools
            if tool["function"]["name"] != "read_source" or tools.known_memory_sources
        ]
        turn = (
            tool_turn(
                router,
                messages,
                tools=available_tools,
                retry_output_limit=2048,
            )
            if getattr(client, "tool_protocol", "native") == "json"
            else native_turn(
                router,
                messages,
                tools=available_tools,
                retry_output_limit=2048,
            )
        )
        calls = turn.get("tool_calls") or []
        if not calls:
            break
        if len(calls) != 1:
            raise ValueError("Research tools run sequentially within their budget")
        call = calls[0]
        if call["function"]["name"] not in {tool["function"]["name"] for tool in selected_tools}:
            raise ValueError("Researcher selected a tool outside its task scope")
        step_id = journal.write(
            "tools",
            "tool-start",
            {
                "tool": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            },
            tool_call_id=call["id"],
        )
        try:
            result = tools.execute(
                call["function"]["name"], json.loads(call["function"]["arguments"])
            )
        except (
            ValueError,
            RuntimeError,
            OSError,
            urllib3.exceptions.HTTPError,
            requests.RequestException,
        ) as error:
            result = {"status": "failed", "error": str(error)[:300]}
        trace.append(
            {
                "tool": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
                "result": result,
            }
        )
        if call["function"]["name"] == "set_research_budget" and result.get("scope"):
            client = apply(client, root)
        journal.write("tools", "tool-result", trace[-1], step_id=step_id)
        messages = [
            *messages,
            turn,
            {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)},
        ]
    result = (
        tool_turn(
            client,
            messages,
            response_format={"type": "json_object", "schema": schema},
            retry_output_limit=research_output_limit(client),
        )
        if getattr(client, "tool_protocol", "native") == "json"
        else native_turn(
            client,
            messages,
            response_format={"type": "json_object", "schema": schema},
            retry_output_limit=research_output_limit(client),
        )
    )
    result["content"] = json.dumps(json_object(result.get("content"), "Research final answer"))
    result["research_trace"] = trace
    return result
