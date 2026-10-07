import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from rlm.v100.agent import answer_with_tools, select_expert
from rlm.v100.common import load_profile
from rlm.v100.evaluation import evaluate_suite
from rlm.v100.memory import Memory
from rlm.v100.protection import (
    ExpertRegistry,
    assert_candidate_output,
    compare_reports,
    fixed_split,
    reserve_audit_sources,
)


def sample(group):
    return {"group": group, "document_ids": [group]}


def test_split_is_fixed_across_rounds_and_bridges_fail_atomically(tmp_path):
    ledger = tmp_path / "roles.sqlite"
    rows = [sample(str(i)) for i in range(10)]
    train, validation = fixed_split(rows, ledger)
    previous_train = {row["group"] for row in train}
    previous_validation = {row["group"] for row in validation}
    new_train, new_validation = fixed_split(rows + [sample("new-a"), sample("new-b")], ledger)
    assert previous_train <= {row["group"] for row in new_train}
    assert previous_validation <= {row["group"] for row in new_validation}
    bridge = {"group": "bridge", "document_ids": [train[0]["group"], validation[0]["group"]]}
    with sqlite3.connect(ledger) as db:
        before = list(db.execute("SELECT * FROM roles ORDER BY source"))
    with pytest.raises(ValueError, match="connects"):
        fixed_split(rows + [bridge, sample("not-committed")], ledger)
    with sqlite3.connect(ledger) as db:
        assert list(db.execute("SELECT * FROM roles ORDER BY source")) == before


def test_audit_is_never_reclassified_as_train_or_validation(tmp_path):
    ledger = tmp_path / "split.sqlite"
    reserve_audit_sources(ledger, ["reserved"])
    with pytest.raises(ValueError, match="Audit"):
        fixed_split([sample("reserved"), sample("other")], ledger)
    with sqlite3.connect(ledger) as db:
        assert list(db.execute("SELECT * FROM roles")) == [("reserved", "audit")]
    fixed_split([sample(str(i)) for i in range(20)], ledger)
    with pytest.raises(ValueError, match="Previously"):
        reserve_audit_sources(ledger, ["0"])


def test_candidate_cannot_write_into_base_adapter_or_experts(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    for output in (base, base / "child", tmp_path, adapter, tmp_path / "research/experts/old"):
        with pytest.raises(ValueError, match="overlaps"):
            assert_candidate_output(output, base, adapter, tmp_path)
    assert_candidate_output(tmp_path / "round-2", base, adapter, tmp_path)


def profile(tmp_path):
    data = load_profile(Path(__file__).parents[1] / "profiles/v100.toml", tmp_path)
    model, binary = tmp_path / "base.gguf", tmp_path / "llama-server"
    model.write_bytes(b"original weights")
    binary.write_bytes(b"native binary")
    data["server"].update(model=str(model), binary=str(binary))
    return data


def test_expert_copy_survives_original_mutation_and_never_overwrites(tmp_path):
    data = profile(tmp_path)
    library = tmp_path / "libggml.so"
    library.write_bytes(b"library")
    registry = ExpertRegistry(tmp_path / "research/experts")
    first = registry.register("original", data, "original model")
    Path(data["server"]["model"]).write_bytes(b"different weights")
    assert Path(first["profile"]["server"]["model"]).read_bytes() == b"original weights"
    assert (registry.directory / "original/libggml.so").read_bytes() == b"library"
    assert registry.get("original")["id"] == "original"
    with pytest.raises(FileExistsError):
        registry.register("original", data, "replacement")
    target = Path(first["profile"]["server"]["model"])
    target.chmod(0o644)
    target.write_bytes(b"tampered snapshot")
    with pytest.raises(ValueError, match="changed"):
        registry.get("original")


def quality(cases):
    return {
        "schema": "v100-quality-v1",
        "suite_sha256": "fixed",
        "generation": {"seed": 42},
        "memory_mode": "fixtures",
        "cases": [{"id": name, "passed": result} for name, result in cases],
    }


def test_gate_rejects_rare_regression_even_when_average_improves():
    old = quality([("rare", True), ("new1", False), ("new2", False)])
    new = quality([("rare", False), ("new1", True), ("new2", True)])
    assert compare_reports(old, new)["regressions"] == ["rare"]
    new["cases"][0]["passed"] = True
    assert compare_reports(old, new)["passed"] is True
    new["suite_sha256"] = "changed"
    with pytest.raises(ValueError, match="suites"):
        compare_reports(old, new)
    with pytest.raises(ValueError, match="quality reports"):
        compare_reports({"median_tokens_per_second": 99}, new)


def test_quality_evaluation_and_registration_require_same_artifact(tmp_path):
    data = profile(tmp_path)
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "id": "math",
                "skill": "arithmetic",
                "match": "exact",
                "expected": "4",
                "messages": [{"role": "user", "content": "2+2?"}],
            }
        )
    )

    class Client:
        def completion(self, messages):
            return "4"

    result = evaluate_suite(Client(), data, suite, tmp_path / "old.json")
    registry = ExpertRegistry(tmp_path / "experts")
    assert registry.register("tested", data, "tested model", result, result)["gate"]["passed"]
    Path(data["server"]["model"]).write_bytes(b"unmeasured candidate")
    with pytest.raises(ValueError, match="different model"):
        registry.register("changed", data, "changed", result, result)
    with pytest.raises(FileExistsError):
        evaluate_suite(Client(), data, suite, tmp_path / "old.json")


def test_memory_backup_includes_committed_wal_and_retains_originals(tmp_path):
    memory = Memory(tmp_path / "memory.sqlite", "v1")
    memory.ingest("source", "original source", len, 32)
    memory.backup(tmp_path / "backup.sqlite")
    with sqlite3.connect(tmp_path / "backup.sqlite") as db:
        assert db.execute("SELECT text FROM documents").fetchone()[0] == "original source"
    with pytest.raises(FileExistsError):
        memory.backup(tmp_path / "backup.sqlite")
    memory.close()


class ToolClient:
    model_name = "local"
    sampling_args = {"max_tokens": 32}
    context_window = 4096

    def __init__(self, turns):
        self.turns = iter(turns)
        self.requests = []

    def template_args(self):
        return {}

    def count_text(self, text, parse_special=False):
        return len(text.split())

    def request(self, endpoint, data):
        self.requests.append((endpoint, deepcopy(data)))
        if endpoint == "/apply-template":
            return {"prompt": json.dumps(data)}
        return {"choices": [{"message": next(self.turns), "finish_reason": "stop"}]}


def call(name, arguments, identifier="tool-1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": identifier,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def test_model_selects_memory_tools_without_executing_generated_code():
    client = ToolClient(
        [
            call("search_memory", {"query": "cat"}),
            call("read_source", {"source_id": "source-1"}, "tool-2"),
            {"content": "Warszawa [source-1]"},
        ]
    )
    result = answer_with_tools(
        client, "Where?", lambda query: [{"id": "source-1", "text": "Warszawa"}]
    )
    assert result["answer"] == "Warszawa [source-1]"
    assert [row["tool"] for row in result["trace"]] == ["search_memory", "read_source"]
    template = client.requests[0][1]
    assert template["tools"] == client.requests[1][1]["tools"]
    assert "tool_calls" in result["messages"][2]


@pytest.mark.parametrize(
    "message",
    [
        call("shell", {"command": "anything"}),
        call("read_source", {"source_id": "never retrieved"}),
        call("search_memory", {"query": "cat", "extra": True}),
    ],
)
def test_unapproved_tools_and_invalid_arguments_fail(message):
    with pytest.raises(ValueError):
        answer_with_tools(ToolClient([message]), "Q", lambda query: [])


def test_model_expert_choice_is_constrained_to_registered_ids():
    experts = [{"id": "code", "description": "coding"}]
    assert (
        select_expert(ToolClient([{"content": '{"expert_id":"code"}'}]), "Q", experts)["expert_id"]
        == "code"
    )
    with pytest.raises(ValueError, match="unregistered"):
        select_expert(ToolClient([{"content": '{"expert_id":"made-up"}'}]), "Q", experts)


def test_tool_budget_rejects_infinite_loop_and_overflow():
    client = ToolClient([call("search_memory", {"query": "cat"})] * 2)
    with pytest.raises(ValueError, match="budget"):
        answer_with_tools(client, "Q", lambda query: [], max_turns=2)
    client = ToolClient([])
    client.context_window = 10
    with pytest.raises(ValueError, match="context"):
        answer_with_tools(client, "Q", lambda query: [])


def test_distillation_shift_mask_and_gradient_are_correct():
    torch = pytest.importorskip("torch")
    from rlm.v100.distillation import masked_kl

    teacher = torch.randn(1, 4, 7)
    student = teacher.clone().requires_grad_()
    labels = torch.tensor([[-100, -100, 3, 4]])
    assert masked_kl(student, teacher, labels).item() == pytest.approx(0, abs=1e-6)
    altered = teacher.clone()
    altered[:, 0] += torch.arange(7) * 10
    assert masked_kl(student, altered, labels).item() == pytest.approx(0, abs=1e-6)
    altered[:, 1, 0] += 5
    loss = masked_kl(student, altered, labels)
    assert loss > 0
    loss.backward()
    assert student.grad[:, 0].abs().sum() == 0
    assert student.grad[:, 1].abs().sum() > 0
    assert student.grad[:, -1].abs().sum() == 0


def test_shared_teacher_restores_candidate_even_on_failure(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from rlm.v100.distillation import teacher_mode

    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=32,
            )
        ),
        LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    model.save_pretrained(tmp_path / "teacher")
    model.load_adapter(tmp_path / "teacher", adapter_name="teacher", is_trainable=False)
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        if ".teacher." in name:
            parameter.requires_grad_(False)
    before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    model.train()
    with pytest.raises(RuntimeError):
        with teacher_mode(model, True):
            assert not model.training
            assert all(not parameter.requires_grad for parameter in model.parameters())
            raise RuntimeError("interrupted teacher forward")
    assert model.active_adapter == "default" and model.training
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == before


def test_semantic_retrieval_finds_synonyms_and_survives_restarts(tmp_path):
    pytest.importorskip("torch")
    from rlm.v100.semantic import hybrid_retrieve, index_memory

    class Encoder:
        identity = "encoder-v1"

        def encode(self, text):
            return (
                [[1.0, 0.0]] if any(word in text for word in ("samochód", "auto")) else [[0.0, 1.0]]
            )

    memory = Memory(tmp_path / "memory.sqlite", "v1")
    memory.ingest("cars", "samochód jeździ", len, 32)
    memory.ingest("animals", "kot śpi", len, 32)
    assert index_memory(memory, Encoder()) == 2
    assert index_memory(memory, Encoder()) == 0
    memory.close()
    memory = Memory(tmp_path / "memory.sqlite", "v2")
    assert hybrid_retrieve(memory, Encoder(), "auto", 1)[0]["text"] == "samochód jeździ"
    encoder = Encoder()
    encoder.identity = "new-version"
    with pytest.raises(ValueError, match="incomplete"):
        hybrid_retrieve(memory, encoder, "auto", 1)
    memory.close()


def test_semantic_transaction_rolls_back_interrupted_chunk(tmp_path):
    from rlm.v100.semantic import index_memory

    class BrokenEncoder:
        identity = "encoder"

        def encode(self, text):
            return [[1.0, 0.0], [float("nan"), 0.0]]

    memory = Memory(tmp_path / "memory.sqlite", "v1")
    memory.ingest("doc", "text", len, 32)
    with pytest.raises(ValueError, match="invalid"):
        index_memory(memory, BrokenEncoder())
    assert memory.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    memory.close()


def test_bred_child_must_preserve_successes_of_both_parents(tmp_path):
    from rlm.v100.protection import execution_hash, file_hash

    data = profile(tmp_path)
    registry = ExpertRegistry(tmp_path / "experts")
    first = quality([("a", True), ("b", False)])
    second = quality([("a", False), ("b", True)])
    first["model_sha256"], second["model_sha256"] = "parent-a", "parent-b"
    child = quality([("a", True), ("b", False)])
    child.update(
        model_sha256=file_hash(Path(data["server"]["model"])), execution_sha256=execution_hash(data)
    )
    provenance = Path(data["server"]["model"]).with_suffix(".provenance.json")
    provenance.write_text(
        json.dumps(
            {
                "model_sha256": child["model_sha256"],
                "adapter": {"parents": [{"id": "a"}, {"id": "b"}]},
                "parent_model_sha256": ["parent-a", "parent-b"],
            }
        )
    )
    with pytest.raises(ValueError, match="regressions"):
        registry.register("bad-child", data, "child", [first, second], child)
    child["cases"][1]["passed"] = True
    with pytest.raises(ValueError, match="both parents"):
        registry.register("single-parent", data, "child", [first], child)
    assert registry.register("good-child", data, "child", [first, second], child)["gate"]["passed"]


def test_local_server_receipt_rejects_wrong_model_or_changed_runtime(tmp_path, monkeypatch):
    from importlib import import_module

    from rlm.v100.serving import assert_served_expert, write_receipt

    data = profile(tmp_path)
    monkeypatch.setattr(
        import_module("rlm.v100.serving"), "process_identity", lambda pid: "fixed-start"
    )
    write_receipt(data, tmp_path)

    class Client:
        def request(self, endpoint):
            return {"model_path": data["server"]["model"]}

    assert_served_expert(Client(), data, tmp_path)
    data["server"]["threads"] += 1
    with pytest.raises(ValueError, match="not the model"):
        assert_served_expert(Client(), data, tmp_path)


def test_real_encoder_uses_batched_windows_without_truncation(tmp_path):
    pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    from rlm.v100.protection import file_hash
    from rlm.v100.semantic import ENCODER_ID, Encoder

    core = Tokenizer(models.WordLevel({"[UNK]": 0, "foo": 1, "bar": 2}, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="[UNK]", pad_token="[UNK]", model_max_length=32
    )
    tokenizer.save_pretrained(tmp_path)
    model = BertModel(
        BertConfig(
            vocab_size=3,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
        )
    )
    model.save_pretrained(tmp_path)
    (tmp_path / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 16}))
    files = {path.name: file_hash(path) for path in tmp_path.iterdir() if path.is_file()}
    (tmp_path / "encoder.json").write_text(json.dumps({"model": ENCODER_ID, "files": files}))
    encoder = Encoder(tmp_path)
    vectors = encoder.encode(" ".join(["foo"] * 40))
    assert len(vectors) >= 3
    assert all(len(vector) == 8 for vector in vectors)
    assert all(abs(sum(value * value for value in vector) - 1) < 1e-5 for vector in vectors)
