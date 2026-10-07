"""Bounded public-source reads and isolated code checks; no paid API clients."""

import hashlib
import ipaddress
import json
import re
import socket
import ssl
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import urllib3

from rlm.v100.activity import ActivityLog
from rlm.v100.agent import native_turn, tool_schema

TOOLS = [
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


def download_page(url: str) -> tuple[str, str]:
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
            body = response.read(262145)
            if len(body) > 262144:
                raise ValueError("Source exceeds the 256 KiB read budget")
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

    def execute(self, name: str, arguments: dict) -> dict:
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
            text = text[:40000]
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
            }
            self.sources.append(source)
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
    owner = getattr(client, "research_owner", "A")
    journal = ActivityLog(root, owner, getattr(client, "activity_actor", "tester"))
    tools = ResearchTools(
        root, getattr(client, "research_config", {}), getattr(client, "research_owner", "A")
    )
    trace = []
    for _ in range(2):
        turn = native_turn(client, messages, tools=TOOLS)
        calls = turn.get("tool_calls") or []
        if not calls:
            break
        if len(calls) != 1:
            raise ValueError("Research tools run sequentially within their budget")
        call = calls[0]
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
        journal.write("tools", "tool-result", trace[-1], step_id=step_id)
        messages = [
            *messages,
            turn,
            {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)},
        ]
    result = native_turn(
        client, messages, response_format={"type": "json_object", "schema": schema}
    )
    result["research_trace"] = trace
    return result
