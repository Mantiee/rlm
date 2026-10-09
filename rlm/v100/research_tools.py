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
from rlm.v100.insights import PROOF_DOMAINS
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
        "compute_resources",
        "Read actual pinned helper load, owned CPU worker heartbeats and queued/running jobs. Worker RAM/disk stay worker-local; never infer pooled host RAM or actual execution from readiness.",
        {},
    ),
    tool_schema(
        "dashboard_status",
        "Verify the guest HTML and exact host publication hash. A model declaration is not publication.",
        {},
    ),
    tool_schema(
        "read_dashboard",
        "Read the editable passive dashboard HTML. Live bindings are supplied by the host; no scripts or event handlers.",
        {},
    ),
    tool_schema(
        "write_dashboard",
        "Validate and atomically save passive dashboard HTML/CSS inside the private guest; preserves backup. Keep required IDs. Use dashboard_status to verify publication.",
        {"html": {"type": "string", "maxLength": 11000}},
    ),
    tool_schema(
        "goal_learning_status",
        "Read precommitted goal-linked forecasts and observed scores; scores are not causal/profit proof.",
        {},
    ),
    tool_schema(
        "observe_goal_source",
        "Archive a free public source now. Optional dotted JSON field selects a numeric outcome in any domain. No operator credentials or synthetic labels.",
        {"url": {"type": "string"}, "field": {"type": "string", "maxLength": 200}},
    ),
    tool_schema(
        "predict_goal_pattern",
        "Commit a goal-relevant hypothesis BEFORE its outcome. specification: question, rationale, evidence (1-8 host observation IDs), target (fresh numeric observation ID), horizon_seconds (60-604800), threshold (positive absolute change in target units), probabilities ([down,flat,up] sum=1). Host observer later resolves the outcome. Search domains and signals yourself or follow the user's direction; do not change the long-term goal.",
        {"specification": {"type": "object"}},
    ),
    tool_schema(
        "read_tool_result",
        "Read the next 512 UTF-8 bytes of a complete archived tool result using its SHA256 and byte offset. Pages fit the small helper tool-response budget even with JSON escaping. Validate every page against the archive hash; continue until next_offset is null. An excerpt is incomplete evidence.",
        {
            "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "offset": {"type": "integer", "minimum": 0},
        },
    ),
    tool_schema(
        "mission_evidence",
        "Read actual mission training health, successful optimizer counters, accepted updates, complete baseline paths and recorded errors. Missing evidence is unknown; GUI boot and arithmetic training do not block unrelated work. Does not expose hidden audit answers or run an experiment.",
        {},
    ),
    tool_schema(
        "configure_free_market_adapter",
        "Configure current public Bybit research or free IEX/sports data by exact documented mapping. No paid histories, trading endpoints or model-supplied secrets. Free keys must already exist in operator environment. Sports execution requires separately registered fees/instrument and an explicit bookmaker settlement rule; provider odds have no certified liquidity.",
        {"configuration": {"type": "object"}},
    ),
    tool_schema(
        "propose_scratch_master",
        "Queue a previously created custom architecture for full-weight training and master selection in the next exclusive window. Supply all resource budget fields. Context/output must cover the current master. Source runs only in bubblewrap; immutable parent, complete fixed/official and fresh gates precede activation. Tiny pilot quality does not prove master quality.",
        {"candidate_id": {"type": "string"}, "budget": {"type": "object"}},
    ),
    tool_schema(
        "scratch_master_status",
        "Inspect queued full-weight master candidates, independent selection evidence and a complete budget template to adapt to the current master.",
        {},
    ),
    tool_schema(
        "support_scratch_master",
        "Agree to a joint custom master trial using the exact source and full resource budget. Both A and B must independently agree to this budget before a shared candidate can train.",
        {"candidate_id": {"type": "string"}, "budget": {"type": "object"}},
    ),
    tool_schema(
        "request_fresh_curriculum",
        "Request 1-32 NEW independently verified arithmetic, decimal, equation, sequence or structured-extraction examples. Only when relevant to the operator goal or a demonstrated weakness. Never draws from benchmark/audit cases. Admission is not an optimizer update.",
        {
            "domain": {
                "type": "string",
                "enum": list(PROOF_DOMAINS),
            },
            "count": {"type": "integer", "minimum": 1, "maximum": 32},
        },
    ),
    tool_schema(
        "propose_foundation_trial",
        "Queue an exclusive alternate pretrained architecture trial. Choose a free HF model, exact commit and falsifiable rationale. Safe weights <=28GiB, library models only, native context cannot shrink. While queued the predecessor keeps serving; optimizer, fixed/public gates run in the next exclusive window. Successful models become independent immutable experts; original Gemma stays available.",
        {
            "model_id": {"type": "string"},
            "revision": {"type": "string"},
            "rationale": {"type": "string", "maxLength": 1200},
        },
    ),
    tool_schema(
        "foundation_trial_status",
        "Inspect proposed architecture changes, their failures, fixed/public quality and expert registration. Proposal is not deployment evidence.",
        {},
    ),
    tool_schema(
        "public_feed_status",
        "List normalized public read-only equity/sports/crypto feed mappings and their limitations. Existing verified instrument rules are mandatory for prices/settlements.",
        {},
    ),
    tool_schema(
        "register_public_feed",
        "Register a public JSON feed for an already verified instrument/feed, or timestamped research news. Does not certify fees, bookmaker rules, source reliability or live account eligibility. No private/paid endpoints. New feeds are picked up by the observer next tick.",
        {"configuration": {"type": "object"}},
    ),
    tool_schema(
        "reward_policy_status",
        "Inspect the branch's reward-trained CPU shadow policy. Actual realized paper outcomes include losses and modeled costs. Observational reward learning is not unbiased RL or proof of profit; no orders are authorized by the head.",
        {},
    ),
    tool_schema(
        "train_reward_policy",
        "Train a bounded CPU shadow policy from the immutable audited ledger, with chronological outcome purging and immutable weight versions. Requires at least 40 resolved orders. Does not replace Gemma or change trading authority.",
        {},
    ),
    tool_schema(
        "consult_browser_model",
        "Open a previously researched free model service inside the private guest browser. Returns page observation; use sandbox_gui to enter the question/read replies. Authentication/captcha needs operator handoff; respect quotas/payment walls, never rotate accounts/cookies to bypass them.",
        {"url": {"type": "string"}, "question": {"type": "string", "maxLength": 3000}},
    ),
    tool_schema(
        "read_master_code",
        "Read the full pinned master source, including readonly controller files. Empty filename lists files. Copies inside the guest can change; the running controller remains protected.",
        {"filename": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}},
    ),
    tool_schema(
        "sandbox_state",
        "Read the private desktop resource and readiness status; no host shell access.",
        {},
    ),
    tool_schema(
        "sandbox_run",
        "Execute a Linux shell script only inside the private Debian VM, with guest-root permissions and public HTTP(S). /workspace persists; /opt/master-source is readonly. No host files, credentials, LAN or GPU access. Prefer schedule_drone kind desktop for background work.",
        {
            "script": {"type": "string", "maxLength": 16000},
            "seconds": {"type": "integer", "minimum": 1, "maximum": 120},
        },
    ),
    tool_schema(
        "sandbox_gui",
        "Observe the private Linux GUI through the RTX vision helper, or click/type/key inside it. Coordinates use 1280x800. No control of host desktop. Supply empty text and x=y=0 for observe.",
        {
            "action": {"type": "string", "enum": ["observe", "click", "type", "key"]},
            "text": {"type": "string", "maxLength": 2000},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
    ),
    tool_schema("get_plan", "Read the user's long-term goal and current short/mid plans.", {}),
    tool_schema(
        "set_plan",
        "Plan or replan the next tasks against the user's goal. Record concrete success criteria. User-owned long-term goal is changed through chat only.",
        {
            "horizon": {"type": "string", "enum": ["short", "mid"]},
            "text": {"type": "string", "maxLength": 2000},
        },
    ),
    tool_schema(
        "schedule_drone",
        "Queue a persistent source watcher, isolated Python experiment, RTX researcher or critic. Jobs continue during V100 evaluation/training. Source payload is a public URL; Python gets fetch(url) for public internet and /work for files. Other payloads are short task briefs. interval=0 runs once; 300..86400 repeats. At most two CPU/source jobs and one RTX job execute simultaneously.",
        {
            "kind": {
                "type": "string",
                "enum": ["source", "python", "researcher", "critic", "desktop"],
            },
            "payload": {"type": "string", "maxLength": 12000},
            "interval": {"type": "integer", "minimum": 0, "maximum": 86400},
        },
    ),
    tool_schema("drone_status", "Read bounded worker status, failures and recent results.", {}),
    tool_schema(
        "cancel_drone",
        "Stop scheduling an unnecessary worker; an in-flight bounded job may finish.",
        {"identity": {"type": "string"}},
    ),
    tool_schema(
        "run_research_python",
        "Run a CPU Python experiment in a private namespace: two CPUs, 3GiB address space, 60s. fetch(url) reads public internet through an audited GET broker; write artifacts in /work. No host home, credentials, GPU, system prompts or direct LAN network. Results remain unverified experiments.",
        {"code": {"type": "string", "maxLength": 12000}},
    ),
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
        "propose_compute_trial",
        "Queue bounded built-in all-weight GRU/transformer pilots on authenticated operator-owned CPU computers. Fresh proof-domain data, independent local tensor validation. Does not replace master or use free managed Colab workers.",
        {
            "architecture": {"type": "string", "enum": ["gru", "transformer"]},
            "steps": {"type": "integer", "minimum": 10, "maximum": 100},
            "width": {"type": "integer", "minimum": 32, "maximum": 128},
            "layers": {"type": "integer", "minimum": 1, "maximum": 2},
            "learning_rate": {"type": "number", "minimum": 0.00001, "maximum": 0.01},
            "purpose": {"type": "string", "maxLength": 600},
            "domain": {
                "type": "string",
                "enum": list(PROOF_DOMAINS),
            },
        },
    ),
    tool_schema(
        "compute_trial_status",
        "Read owned worker readiness and locally validated experiment results.",
        {},
    ),
    tool_schema(
        "cancel_compute_trial",
        "Cancel a queued or active bounded owned-compute experiment.",
        {"identity": {"type": "string"}},
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
        if name == "goal_learning_status" and not arguments:
            from rlm.v100.goal_learning import status

            return status(self.root)
        if name == "observe_goal_source" and set(arguments) == {"url", "field"}:
            from rlm.v100.goal_learning import observe

            return observe(self.root, **arguments)
        if name == "predict_goal_pattern" and set(arguments) == {"specification"}:
            from rlm.v100.goal_learning import predict

            return predict(self.root, self.branch, arguments["specification"])
        if name == "read_tool_result" and set(arguments) == {"sha256", "offset"}:
            digest, offset = arguments["sha256"], arguments["offset"]
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or type(offset) is not int
                or offset < 0
            ):
                raise ValueError("Invalid archive identity or byte offset")
            path = self.root / "research/tool-results" / f"{digest}.json"
            if path.is_symlink() or path.stat().st_size > 8 * 2**20:
                raise ValueError("Archive exceeds its read budget or is a symlink")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Archived tool result changed")
            if offset > len(raw):
                raise ValueError("Byte offset exceeds the archive")
            end = min(offset + 512, len(raw))
            while end < len(raw) and raw[end] & 0xC0 == 0x80:
                end -= 1
            try:
                page = raw[offset:end].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("Byte offset splits a UTF-8 character") from error
            return {
                "sha256": digest,
                "path": str(path),
                "offset": offset,
                "next_offset": end if end < len(raw) else None,
                "total_bytes": len(raw),
                "content": page,
                "scope": "One verified archive excerpt; inspect remaining pages before inferring absent facts.",
            }
        if name == "mission_evidence" and not arguments:
            from rlm.v100.mission_evidence import collect

            return collect(self.root)
        if name == "propose_foundation_trial" and set(arguments) == {
            "model_id",
            "revision",
            "rationale",
        }:
            from rlm.v100.foundation import propose

            return propose(self.root, self.branch, **arguments)
        if name == "foundation_trial_status" and not arguments:
            return {
                "trials": [
                    json.loads(p.read_text())
                    for p in sorted(
                        (self.root / "research/foundation-trials").glob("*/proposal.json")
                    )[:4]
                ]
            }
        if name == "request_fresh_curriculum" and set(arguments) == {"domain", "count"}:
            from rlm.v100.curriculum import request

            return request(self.root, self.branch, **arguments)
        if name == "public_feed_status" and not arguments:
            from rlm.v100.provider_registry import catalog

            return catalog(self.root)
        if name == "register_public_feed" and set(arguments) == {"configuration"}:
            from rlm.v100.provider_registry import register

            return register(self.root, arguments["configuration"])
        if name in ("reward_policy_status", "train_reward_policy") and not arguments:
            from rlm.v100.reward_policy import inspect, train

            return (inspect if name == "reward_policy_status" else train)(self.root, self.branch)
        if name == "consult_browser_model" and set(arguments) == {"url", "question"}:
            from rlm.v100.browser_research import consult

            return consult(self.root, self.branch, **arguments)
        if name == "read_master_code" and set(arguments) == {"filename", "offset"}:
            from rlm.v100.code_lab import source_files

            source = Path(
                json.loads((self.root / "research/self-code-source.json").read_text())["source"]
            )
            files = source_files(source)
            filename, offset = arguments["filename"], arguments["offset"]
            if type(offset) is not int or offset < 0:
                raise ValueError("Invalid source offset")
            if not filename:
                return {"files": files, "scope": "Full pinned own code, read only"}
            if filename not in files:
                raise ValueError("Only tracked own-source files can be read")
            path = (source / filename).resolve()
            if not path.is_relative_to(source.resolve()) or path.stat().st_size > 2 * 2**20:
                raise ValueError("Source file is outside the read budget")
            content = path.read_text()
            return {
                "filename": filename,
                "offset": offset,
                "text": content[offset : offset + 6000],
                "remaining": max(0, len(content) - offset - 6000),
            }
        if name == "compute_resources" and not arguments:
            from rlm.v100.chat_resources import status

            return status(self.root)
        if name in ("dashboard_status", "read_dashboard") and not arguments:
            from rlm.v100.dashboard_editor import status

            return status(self.root, include_html=name == "read_dashboard")
        if name == "write_dashboard" and set(arguments) == {"html"}:
            from rlm.v100.dashboard_editor import write

            return write(self.root, arguments["html"])
        if name == "sandbox_state" and not arguments:
            path = self.root / "research/desktop/status.json"
            return (
                json.loads(path.read_text())
                if path.exists()
                else {"running": False, "state": "not started or not prepared"}
            )
        if name == "sandbox_run" and set(arguments) == {"script", "seconds"}:
            from rlm.v100.desktop import run

            return run(self.root, **arguments)
        if name == "sandbox_gui" and set(arguments) == {"action", "text", "x", "y"}:
            from rlm.v100.desktop import gui

            return gui(self.root, **arguments)
        if name in ("get_plan", "set_plan"):
            from rlm.v100.planning import read, update

            if name == "get_plan" and not arguments:
                return read(self.root)
            if name == "set_plan" and set(arguments) == {"horizon", "text"}:
                return update(self.root, arguments["horizon"], arguments["text"], self.branch)
            raise ValueError("Invalid plan request")
        if name in ("schedule_drone", "drone_status", "cancel_drone"):
            from rlm.v100.drones import cancel, inspect, schedule

            if name == "drone_status" and not arguments:
                return {"jobs": inspect(self.root)}
            if name == "cancel_drone" and set(arguments) == {"identity"}:
                return cancel(self.root, arguments["identity"])
            if name == "schedule_drone" and set(arguments) == {"kind", "payload", "interval"}:
                return schedule(self.root, self.branch, **arguments)
            raise ValueError("Invalid drone request")
        if name == "run_research_python" and set(arguments) == {"code"}:
            from rlm.v100.research_sandbox import run

            return run(self.root, self.branch, arguments["code"])
        if name in ("discover_spot_markets", "register_paper_spot"):
            from rlm.v100.spot_bootstrap import discover, prepare

            if name == "discover_spot_markets" and set(arguments) == {"limit"}:
                return discover(self.root, **arguments)
            if name == "register_paper_spot" and set(arguments) == {"pair_codes"}:
                return prepare(self.root, refresh=True, **arguments)
            raise ValueError("Invalid market discovery/registration arguments")
        if name in ("propose_compute_trial", "compute_trial_status", "cancel_compute_trial"):
            from rlm.v100.distributed_compute import cancel, inspect, propose

            if name == "propose_compute_trial" and set(arguments) == {
                "architecture",
                "steps",
                "purpose",
                "domain",
                "width",
                "layers",
                "learning_rate",
            }:
                return propose(self.root, self.branch, **arguments)
            if name == "compute_trial_status" and not arguments:
                return inspect(self.root)
            if name == "cancel_compute_trial" and set(arguments) == {"identity"}:
                return cancel(self.root, **arguments)
            raise ValueError("Invalid owned compute arguments")
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
        if name == "configure_free_market_adapter" and set(arguments) == {"configuration"}:
            from rlm.v100.market_adapters import register

            return register(self.root, arguments["configuration"])
        if name == "propose_scratch_master":
            from rlm.v100.scratch_master import propose

            if set(arguments) != {"candidate_id", "budget"}:
                raise ValueError("Invalid scratch master proposal")
            return propose(self.root, self.branch, **arguments)
        if name == "scratch_master_status" and not arguments:
            from rlm.v100.architectures import DEFAULT_BUDGET

            return {
                "budget_template": {
                    **DEFAULT_BUDGET,
                    "context_window": self.settings.get("runtime", {}).get("context_window", 8192),
                    "max_new_tokens": self.settings.get("runtime", {}).get(
                        "max_output_tokens", 2048
                    ),
                },
                "limits": "1B parameters, 24 GiB RAM, 30 GiB configured VRAM, 7200 seconds; context up to 262144 bytes and output up to 16384 bytes. Exclusive GPU experiments; measured quality precedes promotion.",
                "trials": [
                    json.loads(p.read_text())
                    for p in sorted((self.root / "research/scratch-master-trials").glob("*.json"))[
                        -16:
                    ]
                ],
            }
        if name == "support_scratch_master" and set(arguments) == {"candidate_id", "budget"}:
            from rlm.v100.architectures import support_candidate

            return support_candidate(self.root, self.branch, **arguments)
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


def bounded_tool_result(root: Path, result: dict, limit: int = 4096) -> str:
    """Keep complete evidence on disk, exposing explicitly incomplete previews."""
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    raw = encoded.encode("utf-8")
    if len(raw) <= limit:
        return encoded
    from rlm.v100.common import atomic_json

    ordered = json.loads(encoded)
    raw = (json.dumps(ordered, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(raw).hexdigest()
    folder = root / "research/tool-results"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}.json"
    if not path.exists():
        atomic_json(path, ordered)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Archived tool result changed")
    return json.dumps(
        {
            "truncated": True,
            "full_result_sha256": digest,
            "full_result_path": str(path),
            "original_bytes": len(raw),
            "preview": raw[: max(128, limit - 1024)].decode("utf-8", errors="ignore"),
            "scope": "Incomplete untrusted excerpt; do not infer absent facts or certify a result from this preview.",
        },
        ensure_ascii=False,
    )


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
            if (
                getattr(client, "resource_only_chat", False)
                and call["function"]["name"] == "schedule_drone"
                and json.loads(call["function"]["arguments"]).get("kind") == "desktop"
            ):
                raise ValueError("Resource request cannot schedule an unrelated GUI/desktop script")
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
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": bounded_tool_result(
                    root, result, max(1024, min(4096, client.context_window // 16))
                ),
            },
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
