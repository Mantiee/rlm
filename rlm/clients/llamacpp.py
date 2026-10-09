"""Local llama.cpp backend with exact token budgeting and thread-local usage."""

import asyncio
import ipaddress
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class LlamaCppClient(BaseLM):
    def __init__(
        self,
        model_name: str = "v100",
        base_url: str = "http://127.0.0.1:8088",
        context_window: int = 8192,
        sampling_args: dict[str, Any] | None = None,
        timeout: float = 600,
        metrics_path: str | None = None,
        enable_thinking: bool | None = None,
        activity_root: str | None = None,
        activity_branch: str = "controller",
        activity_actor: str = "model",
        activity_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, sampling_args=sampling_args, timeout=timeout)
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or not ipaddress.ip_address(parsed.hostname).is_loopback:
            raise ValueError("llamacpp backend requires an HTTP loopback IP address")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("base_url must be a server origin, without /v1 or credentials")
        self.base_url = base_url.rstrip("/")
        self.context_window = context_window
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError("enable_thinking must be boolean or None")
        self.enable_thinking = enable_thinking
        self.activity_root = Path(activity_root) if activity_root else None
        self.research_owner = activity_branch
        self.research_tool_names: set[str] | None = None
        self.research_config: dict[str, Any] = {}
        self.tool_protocol = "native"
        self.activity_actor = activity_actor
        self.activity_context = activity_context or {}
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.lock = threading.Lock()
        self.thread_state = threading.local()
        self.totals: dict[str, list[int]] = {}
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def request(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint != "/v1/chat/completions" or not self.activity_root:
            return self.http_request(endpoint, data)
        from rlm.v100.activity import ActivityLog

        journal = ActivityLog(self.activity_root, self.research_owner, self.activity_actor)
        request_id = journal.write(
            "steps",
            "inference-start",
            {
                "model": (data or {}).get("model"),
                "max_tokens": (data or {}).get("max_tokens"),
                "message_count": len((data or {}).get("messages", [])),
                "tools": [item["function"]["name"] for item in (data or {}).get("tools", [])],
                "enable_thinking": self.enable_thinking,
                "device": self.activity_context.get("device"),
                "endpoint": self.base_url,
            },
            **self.activity_context,
        )
        started = time.perf_counter()
        try:
            result = self.http_request(endpoint, data)
        except Exception as error:
            journal.write(
                "errors", "inference-failed", {"error": type(error).__name__}, request_id=request_id
            )
            raise
        message = result["choices"][0]["message"]
        # Do not log request prompts or raw reasoning_content. Structured rationale in
        # final content and observable tool choices are the reviewable decision record.
        journal.write(
            "decisions",
            "model-output",
            {
                "model": self.model_name,
                "device": self.activity_context.get("device"),
                "endpoint": self.base_url,
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls", []),
                "finish_reason": result["choices"][0].get("finish_reason"),
            },
            request_id=request_id,
        )
        journal.write(
            "metrics",
            "inference-finished",
            {
                "model": self.model_name,
                "device": self.activity_context.get("device"),
                "seconds": time.perf_counter() - started,
                "usage": result.get("usage"),
                "timings": result.get("timings"),
            },
            request_id=request_id,
        )
        return result

    def remaining_timeout(self) -> float:
        deadline = getattr(self, "request_deadline", None)
        if deadline is None:
            return self.timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise requests.Timeout("Chat request exceeded its total time budget")
        return min(self.timeout, remaining)

    def http_request(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        # No environment proxy routing for local model traffic.
        with requests.Session() as session:
            session.trust_env = False
            response = (
                session.get(self.base_url + endpoint, timeout=self.remaining_timeout())
                if data is None
                else session.post(
                    self.base_url + endpoint, json=data, timeout=self.remaining_timeout()
                )
            )
            response.raise_for_status()
            return response.json()

    def count_text(self, text: str, parse_special: bool = False) -> int:
        return len(
            self.request(
                "/tokenize", {"content": text, "add_special": False, "parse_special": parse_special}
            )["tokens"]
        )

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        prompt = self.request("/apply-template", {"messages": messages, **self.template_args()})[
            "prompt"
        ]
        return self.count_text(prompt, parse_special=True) + 1  # reserve possible BOS

    def template_args(self) -> dict[str, Any]:
        if self.enable_thinking is None:
            return {}
        args = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if not self.enable_thinking:
            args["reasoning_effort"] = "none"
        return args

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        self.thread_state.response_info = {}
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        model_name = model or self.model_name
        args = {"temperature": 0.0, "max_tokens": 512, "seed": 42, **self.sampling_args}
        if {
            "messages",
            "model",
            "stream",
            "chat_template_kwargs",
            "reasoning_effort",
        } & args.keys():
            raise ValueError(
                "Use enable_thinking for template controls; sampling_args cannot override routing"
            )
        max_tokens = args["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        prompt_tokens = self.count_messages(messages)
        if prompt_tokens + max_tokens > self.context_window:
            raise ValueError(
                f"Context overflow: {prompt_tokens} + {max_tokens} > {self.context_window}"
            )
        started = time.perf_counter()
        data = self.request(
            "/v1/chat/completions",
            {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "cache_prompt": True,
                **args,
                **self.template_args(),
            },
        )
        choice = data["choices"][0]
        text = choice["message"].get("content")
        usage = data["usage"]
        summary = ModelUsageSummary(1, usage["prompt_tokens"], usage["completion_tokens"])
        elapsed = time.perf_counter() - started
        self.thread_state.last = summary
        response_info = {
            "finish_reason": choice.get("finish_reason"),
            "answer_chars": len(text) if isinstance(text, str) else 0,
            "reasoning_chars": len(choice["message"].get("reasoning_content") or ""),
            "enable_thinking": self.enable_thinking,
            "timings": data.get("timings"),
        }
        self.thread_state.response_info = response_info
        with self.lock:
            totals = self.totals.setdefault(model_name, [0, 0, 0])
            totals[0] += 1
            totals[1] += summary.total_input_tokens
            totals[2] += summary.total_output_tokens
            if self.metrics_path:
                event = {
                    "model": model_name,
                    "seconds": elapsed,
                    "input_tokens": summary.total_input_tokens,
                    "output_tokens": summary.total_output_tokens,
                    "output_tokens_per_second": summary.total_output_tokens / elapsed,
                    **response_info,
                }
                with self.metrics_path.open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"No final answer: finish_reason={response_info['finish_reason']}, "
                f"output_tokens={summary.total_output_tokens}, "
                f"reasoning_chars={response_info['reasoning_chars']}. "
                "Use enable_thinking=False or increase max_tokens; reasoning is never treated as a final answer."
            )
        return text

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        def complete():
            answer = self.completion(prompt, model)
            return answer, self.get_last_usage()

        answer, usage = await asyncio.to_thread(complete)
        self.thread_state.last = usage
        return answer

    def get_usage_summary(self) -> UsageSummary:
        with self.lock:
            return UsageSummary(
                {name: ModelUsageSummary(*counts) for name, counts in self.totals.items()}
            )

    def get_last_usage(self) -> ModelUsageSummary:
        return self.thread_state.last

    def get_response_info(self) -> dict[str, Any]:
        return dict(getattr(self.thread_state, "response_info", {}))
