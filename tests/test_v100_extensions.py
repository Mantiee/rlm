import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path

import pytest

from rlm.v100 import curriculum, fresh_audit, provider_registry, reward_training
from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash, fixed_split


def profiles(tmp_path):
    parent, candidate = tmp_path / "parent.gguf", tmp_path / "candidate.gguf"
    parent.write_bytes(b"parent weights")
    candidate.write_bytes(b"candidate weights")
    return {"server": {"model": str(parent)}, "training": {}}, candidate


def report(profile, suite, output):
    rows = [json.loads(line) for line in suite.read_text().splitlines()]
    value = {
        "schema": "v100-quality-v1",
        "model_sha256": file_hash(Path(profile["server"]["model"])),
        "suite_sha256": file_hash(suite),
        "generation": {},
        "memory_mode": "fixed",
        "cases": [
            {"id": row["id"], "passed": True, "error": None, "finish_reason": "stop"}
            for row in rows
        ],
    }
    atomic_json(output, value)
    return value


def ready_audit(tmp_path, monkeypatch):
    from rlm.v100 import competition, evaluation

    parent, candidate = profiles(tmp_path)
    monkeypatch.setattr(competition, "managed_server", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(competition, "helper_client", lambda *args: object())
    monkeypatch.setattr(
        evaluation,
        "evaluate_suite",
        lambda client, profile, suite, output: report(profile, suite, output),
    )
    folder = fresh_audit.create(tmp_path, parent, candidate, tmp_path / "audit", 8)
    fresh_audit.parent_report(tmp_path, parent, folder)
    child = {**parent, "server": {"model": str(candidate)}}
    return folder, child, candidate


def test_fresh_audits_disjoint_postfreeze_and_training_protected(tmp_path):
    parent, candidate = profiles(tmp_path)
    first = fresh_audit.create(tmp_path, parent, candidate, tmp_path / "first", 12)
    second = fresh_audit.create(tmp_path, parent, candidate, tmp_path / "second", 12)
    rows = [json.loads(line) for line in (first / "suite.jsonl").read_text().splitlines()]
    next_rows = [json.loads(line) for line in (second / "suite.jsonl").read_text().splitlines()]
    assert {r["id"] for r in rows}.isdisjoint(r["id"] for r in next_rows)
    assert all(all(m["role"] != "assistant" for m in row["messages"]) for row in rows)
    assert json.loads((first / "manifest.json").read_text())["candidate_sha256"] == file_hash(
        candidate
    )
    with pytest.raises(ValueError, match="[Aa]udit"):
        fixed_split(
            [
                {"group": rows[0]["id"], "document_ids": [rows[0]["id"]]},
                {"group": "independent", "document_ids": ["independent"]},
            ],
            tmp_path / "research/state/splits.sqlite3",
        )


def test_fresh_audit_consumed_and_artifacts_bound(tmp_path, monkeypatch):
    folder, profile, candidate = ready_audit(tmp_path, monkeypatch)
    assert fresh_audit.candidate_report(object(), profile, folder)["passed"]
    assert fresh_audit.verified_gate(folder, candidate)["passed"]
    with pytest.raises(ValueError, match="single-use"):
        fresh_audit.candidate_report(object(), profile, folder)
    candidate.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="frozen"):
        fresh_audit.verified_gate(folder, candidate)


def test_interrupted_fresh_attempt_cannot_be_retried(tmp_path, monkeypatch):
    from rlm.v100 import evaluation

    folder, profile, _ = ready_audit(tmp_path, monkeypatch)

    def failed(*args):
        raise OSError("connection lost")

    monkeypatch.setattr(evaluation, "evaluate_suite", failed)
    with pytest.raises(OSError):
        fresh_audit.candidate_report(object(), profile, folder)
    assert json.loads((folder / "manifest.json").read_text())["state"] == "consumed"
    with pytest.raises(ValueError, match="single-use"):
        fresh_audit.candidate_report(object(), profile, folder)


def test_changed_audit_report_and_regression_fail(tmp_path, monkeypatch):
    folder, profile, candidate = ready_audit(tmp_path, monkeypatch)
    fresh_audit.candidate_report(object(), profile, folder)
    value = json.loads((folder / "candidate.json").read_text())
    value["cases"][0]["passed"] = False
    atomic_json(folder / "candidate.json", value)
    with pytest.raises(ValueError, match="changed"):
        fresh_audit.verified_gate(folder, candidate)


def test_new_curriculum_admitted_but_never_changes_weights(tmp_path):
    result = curriculum.request(tmp_path, "A", "linear_equation", 8)
    assert result["new_verified_examples"] == 8 and not result["weights_changed"]
    with sqlite3.connect(tmp_path / "research/state/verified-insights.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM insights WHERE admitted=1").fetchone()[0] == 8
    with pytest.raises(ValueError):
        curriculum.request(tmp_path, "A", "financial_prediction", 8)


def test_dpo_gradient_prefers_chosen_and_ignores_zero_weight():
    torch = pytest.importorskip("torch")
    chosen = torch.tensor([0.0, 0.0], requires_grad=True)
    rejected = torch.tensor([0.0, 0.0], requires_grad=True)
    zero = torch.zeros(2)
    loss = reward_training.preference_loss(chosen, rejected, zero, zero, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert chosen.grad[0] < 0 and rejected.grad[0] > 0
    assert chosen.grad[1] == rejected.grad[1] == 0
    assert torch.isfinite(loss)
    with pytest.raises(ValueError):
        reward_training.preference_loss(
            chosen, rejected, zero, zero, torch.tensor([float("nan"), 1.0])
        )


def test_dpo_logprob_single_token_multi_batch_and_prompt_mask():
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], [[3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]], requires_grad=True
    )
    labels = torch.tensor([[-100, 2], [-100, 0]])
    result = reward_training.log_probabilities(logits, labels)
    assert result.shape == (2,)
    expected = torch.log_softmax(logits[:, 0], dim=-1)[[0, 1], [2, 0]]
    assert torch.allclose(result, expected)
    result.sum().backward()
    assert torch.count_nonzero(logits.grad[:, 1]) == 0
    with pytest.raises(ValueError, match="supervised"):
        reward_training.log_probabilities(logits, torch.full_like(labels, -100))


def test_provider_never_relabels_stale_data_as_current():
    config = {
        "rows": "quotes",
        "fields": {"available_at": "time", "bid": "price"},
        "kind": "quote",
        "symbol": "ANY",
        "feed_id": "documented",
        "url": "https://example.org/feed",
    }
    stale = "2026-01-01T00:00:00+00:00"
    rows = provider_registry.normalize(
        config, {"quotes": [{"time": stale, "price": 5}]}, "digest", "2026-01-02T00:00:00+00:00"
    )
    assert rows[0]["available_at"] == stale
    with pytest.raises(ValueError, match="future"):
        provider_registry.normalize(
            config, {"quotes": [{"time": "2027-01-01T00:00:00+00:00", "price": 5}]}, "digest", stale
        )


def test_alternate_base_proposals_are_bounded_and_immutable(tmp_path):
    from rlm.v100.foundation import propose

    result = propose(tmp_path, "A", "owner/model", "a" * 40, "Test a different architecture")
    assert result["state"] == "queued"
    assert propose(tmp_path, "A", "owner/model", "a" * 40, "Another rationale") == result
    with pytest.raises(ValueError):
        propose(tmp_path, "A", "../model", "main", "bad")


def test_default_goal_is_self_upgrade_and_keeps_user_goal(tmp_path):
    from rlm.v100.goals import load_goal, set_goal
    from rlm.v100.mission import OBJECTIVE

    assert "self-upgrades" in OBJECTIVE and "net income" not in OBJECTIVE
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "id": "one",
                "skill": "math",
                "match": "exact",
                "expected": "4",
                "messages": [{"role": "user", "content": "2+2"}],
            }
        )
        + "\n"
    )
    set_goal(tmp_path, "Operator chooses a completely different goal", suite)
    assert load_goal(tmp_path)["text"] == "Operator chooses a completely different goal"


@pytest.mark.parametrize("exit_price,profitable", [("120", True), ("80", False)])
def test_outcome_preferences_use_causal_inputs_and_include_losses(tmp_path, exit_price, profitable):
    from rlm.v100 import reward_policy
    from tests.test_v100_paper import decide, quote, setup

    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    decide(book)
    clock.advance()
    book.ingest(quote(clock))
    # An unresolved/open position is never an outcome training label.
    assert reward_training.records(tmp_path) == []
    decide(book, "close")
    clock.advance()
    book.ingest(quote(clock, exit_price))
    book.close()
    rows = reward_training.records(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    chosen = json.loads(row["messages"][-1]["content"])
    rejected = json.loads(row["rejected_messages"][-1]["content"])
    assert chosen["action"] == ("open" if profitable else "hold")
    assert rejected["action"] == ("hold" if profitable else "open")
    prompt = row["messages"][0]["content"]
    assert "net_pnl" not in prompt and "resolved_at" not in prompt
    assert row["messages"][:-1] == row["rejected_messages"][:-1]
    samples = reward_policy.samples(tmp_path, "A")
    assert len(samples) == 1 and (samples[0]["reward"] > 0) == profitable
    assert samples[0]["decision_at"] < samples[0]["resolved_at"]


def test_shadow_reward_gradient_direction_and_no_fake_eligibility(tmp_path):
    from rlm.v100 import reward_policy

    feature = [1.0] + [0.0] * 7
    before = [0.0] * 8
    win = reward_policy.fit([{"features": feature, "reward": 0.8}], before, steps=20)
    lose = reward_policy.fit([{"features": feature, "reward": -0.8}], before, steps=20)
    assert win[0] > 0 > lose[0]
    result = reward_policy.train(tmp_path, "A")
    assert not result["weights_changed"] and result["samples"] == 0
