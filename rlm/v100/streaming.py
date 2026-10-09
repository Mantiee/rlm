"""Bounded local SSE and owned Ollama NDJSON assembly with observable deltas."""

import json


def local_reply(lines, emit) -> dict:
    content, reasoning, calls, usage = "", "", {}, {}
    finish, done, consumed = None, False, 0
    for raw in lines:
        consumed += len(raw)
        if consumed > 2 * 1024**2:
            raise ValueError("Inference stream exceeds its byte budget")
        if not raw.startswith(b"data:"):
            continue
        value = raw[5:].strip()
        if value == b"[DONE]":
            done = True
            break
        packet = json.loads(value)
        if packet.get("error"):
            raise ValueError("Inference stream returned an error")
        usage = packet.get("usage") or usage
        for choice in packet.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Only one inference choice is supported")
            delta = choice.get("delta", {})
            for key in ("content", "reasoning_content"):
                text = delta.get(key) or ""
                if not isinstance(text, str):
                    raise ValueError("Invalid inference text delta")
                if text:
                    emit("reasoning" if key == "reasoning_content" else "output", text)
                if key == "content":
                    content += text
                else:
                    reasoning += text
            for item in delta.get("tool_calls", []):
                index = item["index"]
                target = calls.setdefault(
                    index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                target["id"] += item.get("id", "")
                for key in ("name", "arguments"):
                    target["function"][key] += item.get("function", {}).get(key, "")
            finish = choice.get("finish_reason") or finish
    if not done or not finish or not usage:
        raise ValueError("Incomplete inference stream; partial output is not accepted")
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        message["tool_calls"] = [calls[key] for key in sorted(calls)]
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


def ollama_reply(lines, emit) -> dict:
    content, reasoning, consumed = "", "", 0
    for raw in lines:
        consumed += len(raw)
        if consumed > 2 * 1024**2:
            raise ValueError("Remote inference stream exceeds its byte budget")
        if not raw:
            continue
        packet = json.loads(raw)
        if not isinstance(packet, dict) or packet.get("error"):
            raise ValueError("Invalid remote inference stream")
        message = packet.get("message", {})
        for key in ("content", "thinking"):
            text = message.get(key) or ""
            if not isinstance(text, str):
                raise ValueError("Invalid remote text delta")
            if text:
                emit("reasoning" if key == "thinking" else "output", text)
            if key == "content":
                content += text
            else:
                reasoning += text
        if packet.get("done") is True:
            packet["message"] = {"role": "assistant", "content": content, "thinking": reasoning}
            return packet
    raise ValueError("Incomplete remote inference stream; partial output is not accepted")
