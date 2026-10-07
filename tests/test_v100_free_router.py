import pytest

from rlm.v100 import free_router


def row(model="lab/teacher:free"):
    return {
        "id": model,
        "name": "Teacher",
        "pricing": {"prompt": "0", "completion": "0", "request": "0"},
        "architecture": {"output_modalities": ["text"]},
        "context_length": 8192,
    }


@pytest.mark.parametrize(
    "change",
    ["paid_name", "paid_prompt", "paid_request", "missing_completion", "nan_price", "image_only"],
)
def test_price_and_modality_filters_reject_uncertain_or_paid_routes(change):
    model = row()
    if change == "paid_name":
        model["id"] = "lab/paid"
    elif change == "paid_prompt":
        model["pricing"]["prompt"] = "0.001"
    elif change == "paid_request":
        model["pricing"]["request"] = "0.1"
    elif change == "missing_completion":
        del model["pricing"]["completion"]
    elif change == "nan_price":
        model["pricing"]["completion"] = "NaN"
    else:
        model["architecture"]["output_modalities"] = ["image"]
    assert not free_router.free_model(model)


def test_consultation_checks_live_price_and_free_account_before_request(tmp_path, monkeypatch):
    monkeypatch.setenv(free_router.KEY_VARIABLE, "test-credential-not-for-logging")
    calls = []

    def api(method, path, key="", payload=None):
        calls.append((method, path, payload))
        if path == "/key":
            return {"data": {"is_free_tier": True}}
        if path == "/models":
            return {"data": [row()]}
        assert payload["model"] == "lab/teacher:free"
        assert payload["provider"]["max_price"] == {"prompt": 0, "completion": 0, "request": 0}
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["enforce_distillable_text"] is True
        assert payload["plugins"] == [] and "models" not in payload
        return {
            "model": "lab/teacher",
            "usage": {"cost": 0},
            "choices": [{"message": {"content": "Check your counterexample."}}],
        }

    monkeypatch.setattr(free_router, "api", api)
    result = free_router.consult(
        tmp_path, "B", "lab/teacher:free", "Find a flaw in this hypothesis"
    )
    assert "advisory" in result["status"]
    assert [path for _, path, _ in calls] == ["/key", "/models", "/chat/completions"]
    assert all(
        "test-credential" not in path.read_text(errors="ignore")
        for path in tmp_path.rglob("*")
        if path.is_file() and path.suffix == ".json"
    )


@pytest.mark.parametrize("paid_account", [True, False])
def test_account_billing_or_changed_prices_block_completion(tmp_path, monkeypatch, paid_account):
    monkeypatch.setenv(free_router.KEY_VARIABLE, "credential")
    calls = []
    priced = row()
    priced["pricing"]["completion"] = "0.01"

    def api(method, path, key="", payload=None):
        calls.append(path)
        return (
            {"data": {"is_free_tier": not paid_account}} if path == "/key" else {"data": [priced]}
        )

    monkeypatch.setattr(free_router, "api", api)
    with pytest.raises(ValueError):
        free_router.consult(tmp_path, "A", "lab/teacher:free", "Research")
    assert "/chat/completions" not in calls


def test_missing_account_never_attempts_a_consultation(tmp_path, monkeypatch):
    monkeypatch.delenv(free_router.KEY_VARIABLE, raising=False)
    monkeypatch.setattr(
        free_router, "api", lambda *a, **k: pytest.fail("No network before account setup")
    )
    with pytest.raises(RuntimeError, match="configured"):
        free_router.consult(tmp_path, "A", "lab/teacher:free", "Help")
    with pytest.raises(ValueError, match=":free"):
        free_router.consult(tmp_path, "A", "openrouter/auto", "Help")


def test_global_quota_and_cooldown_apply_to_all_agents(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(free_router.time, "time", lambda: clock[0])
    free_router.reserve_request(tmp_path, "same-account")
    with pytest.raises(RuntimeError, match="cooldown"):
        free_router.reserve_request(tmp_path, "same-account")
    for _ in range(39):
        clock[0] += 10
        free_router.reserve_request(tmp_path, "same-account")
    clock[0] += 10
    with pytest.raises(RuntimeError, match="exhausted"):
        free_router.reserve_request(tmp_path, "same-account")


def test_no_provider_can_silently_substitute_another_model(tmp_path, monkeypatch):
    monkeypatch.setenv(free_router.KEY_VARIABLE, "credential")

    def api(method, path, key="", payload=None):
        if path == "/key":
            return {"data": {"is_free_tier": True}}
        if path == "/models":
            return {"data": [row()]}
        return {"model": "lab/other-model", "choices": []}

    monkeypatch.setattr(free_router, "api", api)
    with pytest.raises(ValueError, match="different model"):
        free_router.consult(tmp_path, "A", "lab/teacher:free", "Research")


def test_unexpected_provider_cost_disables_future_requests(tmp_path, monkeypatch):
    monkeypatch.setenv(free_router.KEY_VARIABLE, "credential")

    def api(method, path, key="", payload=None):
        if path == "/key":
            return {"data": {"is_free_tier": True}}
        if path == "/models":
            return {"data": [row()]}
        return {"model": "lab/teacher:free", "usage": {"cost": 0.01}, "choices": []}

    monkeypatch.setattr(free_router, "api", api)
    with pytest.raises(RuntimeError, match="disabled"):
        free_router.consult(tmp_path, "A", "lab/teacher:free", "Research")
    monkeypatch.setattr(
        free_router, "api", lambda *a, **k: pytest.fail("Guard must stop future network calls")
    )
    with pytest.raises(RuntimeError, match="operator review"):
        free_router.consult(tmp_path, "B", "lab/teacher:free", "Research again")
