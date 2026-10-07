import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import architectures, continuous, efficiency, insights, research_tools, researchers
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.protection import file_hash
from rlm.v100.training import encode_record, load_records


@pytest.mark.parametrize(
    "task,answer",
    [
        ({"kind": "arithmetic", "expression": "(13-7)*3%5"}, "3"),
        ({"kind": "arithmetic", "expression": "-7//3"}, "-3"),
        ({"kind": "linear_equation", "expression": "3*x+4=9"}, "5/3"),
        ({"kind": "linear_equation", "expression": "-2*x-7=3"}, "-5"),
    ],
)
def test_reference_computes_answers_without_trusting_models(task, answer):
    record = insights.verified_record(task)
    assert record["messages"][-1]["content"] == answer
    insights.verify_record(record)
    record["messages"][-1]["content"] = "model insists it is 123"
    with pytest.raises(ValueError, match="reference"):
        insights.verify_record(record)


@pytest.mark.parametrize(
    "expression",
    [
        "open('/etc/passwd').read()",
        "__import__('os').system('id')",
        "2**1000000",
        "1/2",
        "1//0",
        "True+1",
    ],
)
def test_formal_checker_never_executes_proposed_python(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        insights.reference({"kind": "arithmetic", "expression": expression})


def test_queue_requires_both_reference_check_and_parent_admission(tmp_path, monkeypatch):
    task = {"kind": "arithmetic", "expression": "13+19"}
    queue = insights.InsightQueue(tmp_path)
    assert queue.add("A", task)
    assert not queue.add("B", task)
    assert queue.records() == []
    queue.close()
    worker = {"exercise_checks": [{"task": task, "verified": True}]}
    monkeypatch.setattr(
        researchers,
        "native_turn",
        lambda *a, **k: {"content": json.dumps({"useful_indices": [], "conclusion": "Reject"})},
    )
    researchers.review_research(SimpleNamespace(), "A", [worker], tmp_path)
    queue = insights.InsightQueue(tmp_path)
    assert queue.records() == []
    queue.close()
    monkeypatch.setattr(
        researchers,
        "native_turn",
        lambda *a, **k: {
            "content": json.dumps(
                {"useful_indices": [0], "conclusion": "Useful verified challenge"}
            )
        },
    )
    researchers.review_research(SimpleNamespace(), "A", [worker], tmp_path)
    queue = insights.InsightQueue(tmp_path)
    assert len(queue.records()) == 1
    queue.close()
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(insights.verified_record({"kind": "arithmetic", "expression": "2+3"})) + "\n"
    )
    extended = tmp_path / "extended.jsonl"
    assert insights.extend_pool(pool, tmp_path, extended)
    train, validation = load_records(extended)
    assert len(train) == len(validation) == 1
    assert not insights.extend_pool(extended, tmp_path, tmp_path / "no-change")


def test_automatic_labels_really_change_lora_weights_but_not_base(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

    torch.set_num_threads(1)

    class BytesTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            prefix = "".join(
                message["role"] + ":" + message["content"] + "\n" for message in messages
            )
            if add_generation_prompt:
                prefix += "assistant:"
            return list(prefix.encode())

    # encode_record needs the exact assistant prefix in the full template.
    class Template(BytesTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            text = "".join(
                message["role"] + ":" + message["content"] + "\n" for message in messages
            )
            if add_generation_prompt:
                text += "assistant:"
            return list(text.encode())

    task = {"kind": "arithmetic", "expression": "13+19"}
    data = encode_record(insights.verified_record(task), Template(), 256)
    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=256,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
            )
        ),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            max_steps=2,
            per_device_train_batch_size=1,
            learning_rate=0.01,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
        ),
        train_dataset=[data] * 4,
    )
    trainer.train()
    changed = [
        name for name, value in model.named_parameters() if not torch.equal(value, before[name])
    ]
    assert changed and all("lora_" in name for name in changed)


def test_runtime_never_overrides_quality_and_small_differences_remain_ties():
    measurements = {"A": {"total_wall_seconds": 100}, "B": {"total_wall_seconds": 70}}
    assert efficiency.continuation("A", measurements)[0] == "A"
    assert efficiency.continuation(None, measurements)[0] is None
    assert efficiency.continuation("tie", measurements)[0] == "B"
    measurements["B"]["total_wall_seconds"] = 97
    assert efficiency.continuation("tie", measurements)[0] == "A"


def test_performance_must_match_actual_model_and_report(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    report = {"model_sha256": file_hash(model)}
    atomic_json(tmp_path / "development-quality.json", report)
    atomic_json(
        tmp_path / "performance.json",
        {
            "schema": "v100-performance-v1",
            "model_sha256": file_hash(model),
            "quality_report_sha256": file_hash(tmp_path / "development-quality.json"),
            "training_wall_seconds": 10,
            "export_wall_seconds": 20,
            "evaluation_wall_seconds": 5,
            "total_wall_seconds": 35,
        },
    )
    assert efficiency.load_performance(tmp_path, model, report)["total_wall_seconds"] == 35
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="different"):
        efficiency.load_performance(tmp_path, model, report)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "169.254.169.254", "::1"])
def test_internet_reader_rejects_local_and_private_destinations(monkeypatch, address):
    monkeypatch.setattr(
        research_tools.socket,
        "getaddrinfo",
        lambda *a, **k: [(None, None, None, None, (address, 443))],
    )
    with pytest.raises(ValueError, match="private"):
        research_tools.public_origin("https://example.org/page")


def test_reader_connects_to_checked_ip_and_excludes_scripts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        research_tools.socket,
        "getaddrinfo",
        lambda *a, **k: [(None, None, None, None, ("93.184.216.34", 443))],
    )
    seen = {}

    class Pool:
        def __init__(self, address, **kwargs):
            seen.update(
                address=address, hostname=kwargs["assert_hostname"], sni=kwargs["server_hostname"]
            )

        def request(self, *a, **k):
            return SimpleNamespace(
                status=200,
                headers={"Content-Type": "text/html"},
                read=lambda size: b"<p>Source fact</p><script>steal secrets</script>",
                close=lambda: None,
            )

        def close(self):
            pass

    monkeypatch.setattr(research_tools.urllib3, "HTTPSConnectionPool", Pool)
    source = research_tools.ResearchTools(tmp_path, {}).execute(
        "read_public_page", {"url": "https://example.org/page"}
    )
    assert seen == {"address": "93.184.216.34", "hostname": "example.org", "sni": "example.org"}
    assert "Source fact" in source["excerpt"] and "steal secrets" not in source["excerpt"]
    assert (tmp_path / "research/web-sources" / (source["sha256"] + ".txt")).exists()


def test_paid_api_tools_are_unavailable_and_cannot_run_arbitrary_code(tmp_path, monkeypatch):
    tools = research_tools.ResearchTools(tmp_path, {})
    with pytest.raises(ValueError, match="Unknown"):
        tools.execute("consult_teacher", {"question": "Review this"})
    with pytest.raises(ValueError, match="identifier"):
        tools.execute("check_code_candidate", {"candidate_id": "../../outside"})
    with pytest.raises(ValueError, match="Unknown"):
        tools.execute("shell", {"cmd": "anything"})


@pytest.mark.parametrize("paper_mode", [False, True])
def test_continuous_loop_admits_new_examples_trains_and_retains_prior_versions(
    tmp_path, monkeypatch, paper_mode
):
    data = load_profile(Path(__file__).parents[1] / "profiles/v100.toml", tmp_path)
    pool = tmp_path / "initial.jsonl"
    seed = [
        insights.verified_record({"kind": "arithmetic", "expression": expr})
        for expr in ("2+3", "7*4")
    ]
    pool.write_text("".join(json.dumps(row) + "\n" for row in seed))
    untouched = json.dumps(data, sort_keys=True)
    suite = tmp_path / "suite.jsonl"
    suite.write_text("frozen suite")
    baseline = {
        "schema": "v100-quality-v1",
        "suite_sha256": file_hash(suite),
        "generation": {},
        "memory_mode": "fixed",
        "cases": [{"id": "old", "passed": True}],
    }
    helper = tmp_path / "helper.json"
    cpu_profile = json.loads(json.dumps(data))
    cpu_profile["server"]["gpu_layers"] = 0
    cpu_profile["resources"] = {"device": "cpu"}
    helper.write_text(json.dumps(cpu_profile))
    active, trials = [], []

    @contextmanager
    def server(path, *args):
        active.append(path)
        yield data
        active.remove(path)

    monkeypatch.setattr(continuous, "managed_server", server)
    monkeypatch.setattr(continuous, "require_idle_gpu", lambda: None)
    monkeypatch.setattr(architectures, "prepare_inputs", lambda *a: None)
    monkeypatch.setattr(continuous, "helper_client", lambda p, *a: None)
    monkeypatch.setattr(
        continuous.shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 2**30)
    )
    task = {"kind": "arithmetic", "expression": "13+19"}

    def researcher(*args):
        if paper_mode:
            return {"exercise_checks": []}
        queue = insights.InsightQueue(tmp_path)
        queue.add(args[1], task)
        queue.close()
        return {"exercise_checks": [{"task": task, "verified": True}]}

    def review(*args):
        queue = insights.InsightQueue(tmp_path)
        queue.admit(task)
        queue.close()

    monkeypatch.setattr(continuous, "research_task", researcher)
    monkeypatch.setattr(continuous, "review_research", review)
    paper_config = None
    if paper_mode:
        import threading

        from rlm.v100 import paper_learning
        from rlm.v100.paper import PaperBook

        book = PaperBook(tmp_path)
        original_paper = book.initialize()
        book.close()
        paper_config = tmp_path / "paper-learning.json"
        atomic_json(
            paper_config,
            {
                "schema": "v100-paper-learning-v1",
                "objective": "Fast lawful repeatable net income",
                "crypto": False,
                "ciks": [],
                "sec_contact": "",
                "observer_interval": 30,
                "research_rounds": 1,
                "other_income_rnd": True,
            },
        )
        training_started, observation_saved = threading.Event(), threading.Event()

        def financial_research(book, profile, helper, rounds, **kwargs):
            assert len(active) == 2
            assert kwargs["income_research"] is True
            queue = insights.InsightQueue(tmp_path)
            queue.add("A", task)
            queue.admit(task)
            queue.close()

        def observe_during_training(session):
            assert training_started.wait(5)
            session.note("observation-during-training", {"synthetic_test": True})
            observation_saved.set()

        monkeypatch.setattr(paper_learning, "paper_round", financial_research)
        monkeypatch.setattr(paper_learning.PaperLearning, "tick", observe_during_training)

    def evolve(settings, root, expanded, trial, suite, gates, helper, **kwargs):
        assert active == []  # owned inference stopped before exclusive GPU training
        if paper_mode:
            training_started.set()
            assert observation_saved.wait(5)
        assert len(expanded.read_text().splitlines()) == 3
        trials.append(trial)
        directory = trial / "generation-01"
        directory.mkdir(parents=True)
        (directory / "verified-pool.jsonl").write_bytes(expanded.read_bytes())
        branches = {}
        for branch in ("A", "B"):
            chosen = json.loads(json.dumps(settings))
            chosen["training"]["output"] = str(directory / branch / "training")
            chosen["server"]["model"] = str(directory / branch / "new.gguf")
            path = directory / branch / "profile.json"
            atomic_json(path, chosen)
            atomic_json(directory / branch / "serving.json", chosen)
            atomic_json(directory / branch / "development-quality.json", baseline)
            branches[branch] = {"profile": str(path)}
        monkeypatch.setattr(continuous, "load_duel", lambda _: {"branches": branches})
        return {
            "rounds": [
                {
                    "directory": str(directory),
                    "judgment": {
                        "winner": "tie",
                        "continuation_branch": "B",
                        "branches": {"A": {"eligible": True}, "B": {"eligible": True}},
                    },
                }
            ]
        }

    monkeypatch.setattr(continuous, "evolve", evolve)
    result = continuous.learn_loop(
        data,
        tmp_path,
        pool,
        tmp_path / "loop",
        suite,
        [baseline],
        helper,
        cycles=2,
        paper_config=paper_config,
    )
    assert len(trials) == 1  # repeated already learned exercises do not trigger churn
    assert len(result["cycles"]) == 2
    chosen = json.loads(Path(result["live_profile"]).read_text())
    assert chosen["training"]["init_adapter"].endswith("B/training/candidate")
    assert json.dumps(data, sort_keys=True) == untouched
    assert pool.read_text() == "".join(json.dumps(row) + "\n" for row in seed)
    if paper_mode:
        book = PaperBook(tmp_path)
        assert book.state()["branches"] == original_paper["branches"]
        statuses = [event["payload"].get("status") for event in book.events()]
        assert "training-started" in statuses
        assert "selected for next serving phase after finite quality gates" in statuses
        assert "observation-during-training" in statuses
        book.close()
