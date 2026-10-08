"""Mirror the pinned LiveBench CLI's old/new instruction-following dispatch."""

import json
import sys
from types import ModuleType

import pytest

from rlm.v100 import benchmark_worker


@pytest.mark.parametrize("date, legacy", [("2025-11-24", True), ("2025-11-25", False)])
def test_official_instruction_formats_use_their_own_grader(tmp_path, monkeypatch, date, legacy):
    calls = []
    monkeypatch.chdir(tmp_path)
    common = ModuleType("livebench.common")
    common.MatchSingle = lambda *args: args
    judge = ModuleType("livebench.gen_ground_truth_judgment")
    judge.play_a_match_gt = lambda match: calls.append("new") or {"score": 0.75}
    utils = ModuleType("livebench.process_results.instruction_following.utils")

    def old(questions, answers, task, model, debug):
        calls.append("legacy")
        assert task == "simplify"
        assert (tmp_path / "data/live_bench/instruction_following/simplify/model_judgment").is_dir()
        assert answers[model]["q"]["question_id"] == "q"
        assert answers[model]["q"]["choices"][0]["turns"][0] == "Title"
        return [{"question_id": "q", "score": 0.5}]

    utils.instruction_following_process_results = old
    for name, module in [
        ("livebench.common", common),
        ("livebench.gen_ground_truth_judgment", judge),
        ("livebench.process_results.instruction_following.utils", utils),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(benchmark_worker, "offline_nltk", lambda: None)
    data = {
        "question": {
            "category": "instruction_following",
            "question_id": "q",
            "task": "simplify",
            "livebench_release_date": date,
        },
        "model": "candidate",
        "answer": {"choices": [{"turns": ["<think>hidden</think>Title"]}]},
    }
    request, output = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(json.dumps(data))
    benchmark_worker.grade(request, output)
    assert calls == ["legacy" if legacy else "new"]
    assert json.loads(output.read_text())["score"] == (0.5 if legacy else 0.75)
    assert json.loads(request.read_text()) == data
