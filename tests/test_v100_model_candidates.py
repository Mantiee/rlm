import hashlib

import pytest

from rlm.v100 import model_candidates as candidates
from rlm.v100.inference_benchmark import command, variants


def test_catalog_resolves_immutable_hash(monkeypatch):
    monkeypatch.setattr(
        candidates,
        "json_url",
        lambda url: {
            "sha": "a" * 40,
            "siblings": [
                {"rfilename": "gemma-4-E2B-it-Q4_K_M.gguf", "lfs": {"sha256": "b" * 64, "size": 1024}}
            ],
        },
    )
    assert candidates.artifact("fast")["revision"] == "a" * 40


def test_existing_candidate_mismatch_is_rejected(tmp_path):
    record = dict(
        role="fast",
        revision="a" * 40,
        filename="model.gguf",
        sha256=hashlib.sha256(b"correct").hexdigest(),
        bytes=7,
    )
    folder = tmp_path / "models/candidates/fast" / record["revision"]
    folder.mkdir(parents=True)
    (folder / record["filename"]).write_bytes(b"incorrect")
    with pytest.raises(ValueError, match="checksum"):
        candidates.download(tmp_path, record)


def test_long_context_flags_and_cpu_moe_are_explicit():
    server = dict(
        binary="llama",
        model="Qwen3.5-35B-A3B.gguf",
        context_per_slot=524288,
        slots=1,
        threads=4,
        batch_size=128,
        ubatch_size=64,
        flash_attention="on",
        cache_type="q8_0",
        jinja=True,
        gpu_layers=999,
        rope_scaling="yarn",
        rope_scale=2,
        yarn_orig_ctx=262144,
        cpu_moe=True,
    )
    args = command(server, 8097)
    assert args[args.index("--rope-scale") + 1] == "2"
    assert "--cpu-moe" in args
    values = variants({"server": server}, "--cpu-moe")
    assert any(row["name"] == "cpu-moe" for row in values)


def test_long_context_requires_measured_prefill(tmp_path, monkeypatch):
    from rlm.v100 import inference_benchmark as benchmark

    def response(port, path, data=None, **kwargs):
        if path == "/tokenize":
            return {"tokens": [17] * 1000}
        if path == "/apply-template":
            return {"prompt": "query"}
        return {
            "content": "synta_first_7391 synta_middle_4826 synta_last_9153",
            "timings": {"prompt_n": 1},
        }

    monkeypatch.setattr(benchmark, "request", response)
    record = benchmark.long_context_probe(8097, 524288, tmp_path / "evidence.json")
    assert record["input_tokens_sent"] >= 500000
    assert record["passed"] is False


@pytest.mark.parametrize("honors_range", [True, False])
def test_resume_uses_pinned_prefix_and_rejects_ignored_range(tmp_path, monkeypatch, honors_range):
    import io
    from types import SimpleNamespace

    payload = b"verified-model-bytes"
    record = dict(
        role="fast",
        revision="a" * 40,
        filename="model.gguf",
        repo="owner/model",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )
    folder = tmp_path / "models/candidates/fast" / record["revision"]
    folder.mkdir(parents=True)
    partial = folder / "model.partial"
    partial.write_bytes(payload[:7])
    monkeypatch.setattr(
        candidates.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30)
    )

    class Reply(io.BytesIO):
        status = 206 if honors_range else 200
        headers = {"Content-Range": f"bytes 7-{len(payload) - 1}/{len(payload)}"}

    def urlopen(request, **kw):
        assert request.get_header("Range") == "bytes=7-"
        assert "a" * 40 in request.full_url
        return Reply(payload[7:])

    monkeypatch.setattr(candidates.urllib.request, "urlopen", urlopen)
    if honors_range:
        assert candidates.download(tmp_path, record).read_bytes() == payload
        assert not partial.exists()
    else:
        with pytest.raises(ValueError, match="partial retained"):
            candidates.download(tmp_path, record)
        assert partial.read_bytes() == payload[:7]
