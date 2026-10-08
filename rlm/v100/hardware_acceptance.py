"""Bounded on-device infrastructure acceptance; never claims financial quality."""

import json
import subprocess
import sys
import time
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile


def run(root: Path, profile_path: Path, desktop: bool = False) -> Path:
    from rlm.v100.competition import helper_client, managed_server, require_idle_gpu
    from rlm.v100.mission import status
    from rlm.v100.serving import assert_served_expert

    if status(root)["running"]:
        raise ValueError("Hardware acceptance requires the owned mission to be stopped")
    require_idle_gpu()
    folder = root / "research/hardware-acceptance" / ("run-" + str(time.time_ns()))
    folder.mkdir(parents=True)
    result = {
        "schema": "v100-hardware-acceptance-v1",
        "profile": str(profile_path),
        "checks": {},
        "scope": "Infrastructure only; short inputs do not prove full-context quality, profit, universal retention or RTX crash prevention",
    }

    def check(name, action, required=False):
        print(json.dumps({"acceptance": name, "state": "running"}), flush=True)
        try:
            value = action()
            result["checks"][name] = {"state": "passed", "result": value}
        except Exception as error:
            result["checks"][name] = {
                "state": "failed",
                "error": type(error).__name__,
                "detail": str(error)[:500],
            }
            if required:
                result["required_passed"] = False
                atomic_json(folder / "report.json", result)
                atomic_json(
                    root / "research/hardware-acceptance/latest.json",
                    {"report": str(folder / "report.json")},
                )
                raise
        atomic_json(folder / "report.json", result)
        print(json.dumps({"acceptance": name, **result["checks"][name]}), flush=True)

    def cuda():
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
            raise ValueError("Expected the configured Volta sm70 CUDA device")
        value = torch.ones((64, 64), dtype=torch.float16, device="cuda", requires_grad=True)
        loss = (value @ value).float().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(value.grad).all():
            raise ValueError("CUDA backward produced nonfinite gradients")
        measured = {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "loss": float(loss.detach()),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        del loss, value
        torch.cuda.empty_cache()
        return measured

    check("v100-cuda-backward", cuda, required=True)
    profile = load_profile(profile_path, root)

    def master():
        with managed_server(profile_path, root, folder / "native-server.log"):
            client = helper_client(profile, root)
            assert_served_expert(client, profile, root)
            client.sampling_args["max_tokens"] = 64
            client.enable_thinking = False
            text = client.completion("Return only the integer result of 2+2.")
            if not text.strip() or client.get_response_info()["finish_reason"] == "length":
                raise ValueError("Master did not finish the infrastructure response")
            return {
                "answer": text[:200],
                "finish_reason": client.get_response_info()["finish_reason"],
                "native_log": str(folder / "native-server.log"),
                "requested_context": profile["runtime"]["context_window"],
                "requested_flash_attention": profile["server"]["flash_attention"],
                "requested_kv_type": profile["server"]["cache_type"],
                "draft_model": profile["server"]["draft_model"],
                "requested_configuration_is_not_kernel_proof": True,
            }

    check("master-inference", master, required=True)

    def sandbox():
        from rlm.v100.code_lab import sandbox_command, sandbox_environment

        (folder / "source").mkdir()
        (folder / "checks").mkdir()
        child = subprocess.run(
            sandbox_command(folder, [sys.executable, "-I", "-c", "print('sandbox-ok')"]),
            env=sandbox_environment(),
            timeout=30,
            capture_output=True,
            text=True,
            check=True,
        )
        if child.stdout.strip() != "sandbox-ok":
            raise ValueError("Sandbox probe returned another result")
        return {"namespace_execution": True, "network_unshared": True}

    check("private-code-sandbox", sandbox)

    def rtx():
        path = root / "research/researcher-rtx3090.json"
        selected = load_profile(path, root)
        client = helper_client(selected, root)
        client.identity()
        return client.loaded()

    check("rtx-helper-identity-and-loaded-model", rtx)
    if desktop:
        from rlm.v100.desktop import health

        def guest():
            result = health(root)
            if not result["ready"]:
                raise ValueError(
                    "Private GUI guest has not passed its SSH/service probe: " + result["state"]
                )
            return result

        check("private-desktop", guest)
    result["required_passed"] = all(
        result["checks"][name]["state"] == "passed"
        for name in ("v100-cuda-backward", "master-inference")
    )
    atomic_json(folder / "report.json", result)
    atomic_json(
        root / "research/hardware-acceptance/latest.json", {"report": str(folder / "report.json")}
    )
    return folder / "report.json"
