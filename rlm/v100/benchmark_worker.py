"""Runs only in the separate pinned official-benchmark environment."""

import copy
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path

NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "stopwords": "corpora/stopwords",
}


def key(row):
    return "/".join(str(row[k]) for k in ("category", "task", "question_id"))


def prepare_nltk(folder):
    import nltk

    # Authorize an existing private data root, rather than asking NLTK to trust
    # the snapshot's possibly group-writable parent before creating it.
    resources = folder / "nltk"
    resources.mkdir(mode=0o700, exist_ok=True)
    info = resources.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise ValueError("Official grader resources must use an owned private directory")
    for package in NLTK_RESOURCES:
        if not nltk.download(package, download_dir=str(resources), quiet=True):
            raise RuntimeError("Official grader resource download failed: " + package)


def offline_nltk():
    import nltk

    def present(package, *args, **kwargs):
        # IFBench calls nltk.download('stopwords') even while grading. Check
        # the frozen local resource instead of fetching an index without network.
        if package not in NLTK_RESOURCES:
            raise ValueError("Unprepared official grader resource: " + str(package))
        nltk.data.find(NLTK_RESOURCES[package])
        return True

    nltk.download = present


def prepare(folder, limit, coding=False):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from livebench.common import LIVE_BENCH_RELEASES, load_questions

    release = max(LIVE_BENCH_RELEASES)
    categories = ["reasoning", "math", "data_analysis", "language", "instruction_following"]
    if coding:
        categories.append("coding")
    questions, revisions = [], {}
    for category in categories:
        name = "livebench/" + category
        revision = HfApi().dataset_info(name).sha
        revisions[name] = revision
        data = load_dataset(name, revision=revision, split="test")
        for task in sorted(set(data["task"])):
            loaded = load_questions(
                data, {r for r in LIVE_BENCH_RELEASES if r <= release}, release, task
            )
            for question in loaded:
                question["_source_category"] = category
            questions.extend(loaded)
    if not questions or len({key(q) for q in questions}) != len(questions):
        raise ValueError("Empty or ambiguous official question snapshot")
    groups = {}
    for question in questions:
        groups.setdefault(question["_source_category"], []).append(question)
    for rows in groups.values():
        rows.sort(key=lambda q: hashlib.sha256(key(q).encode()).hexdigest())
    selected = []
    while any(groups.values()) and (not limit or len(selected) < limit):
        for category in categories:
            if groups.get(category) and (not limit or len(selected) < limit):
                selected.append(groups[category].pop(0))
    path = folder / "questions.json"
    path.write_text(json.dumps(selected, ensure_ascii=False, default=str))
    reference_error = None
    references = []
    try:
        name = "livebench/model_judgment"
        revisions[name] = HfApi().dataset_info(name).sha
        rows = load_dataset(name, revision=revisions[name], split="leaderboard")
        wanted = {key(q) for q in selected}
        # Namespaced IDs: question_id alone is NOT unique across official tasks.
        for row in rows:
            if (
                all(k in row for k in ("category", "task", "question_id", "model", "score"))
                and key(row) in wanted
            ):
                references.append(
                    {
                        k: row.get(k)
                        for k in (
                            "category",
                            "task",
                            "question_id",
                            "model",
                            "score",
                            "tstamp",
                            "eval_status",
                        )
                    }
                )
    except Exception as error:
        reference_error = str(error)[:400]
    (folder / "references.json").write_text(json.dumps(references, default=str))
    (folder / "snapshot.json").write_text(
        json.dumps(
            {
                "release": release,
                "revisions": revisions,
                "selected": len(selected),
                "available": len(questions),
                "categories": categories,
                "reference_error": reference_error,
                "coding_included": coding,
                "full_snapshot_requested": limit == 0,
            },
            indent=2,
        )
    )
    prepare_nltk(folder)
    # Import during preparation so missing official dependencies fail before GPU work.
    from livebench.gen_ground_truth_judgment import play_a_match_gt  # noqa: F401


def grade(request, output):
    offline_nltk()
    from livebench.common import MatchSingle
    from livebench.gen_ground_truth_judgment import play_a_match_gt

    data = json.loads(request.read_text())
    question, model, answer = data["question"], data["model"], data["answer"]
    # Same split as the pinned official CLI; old IFEval never enters IFBench.
    if (
        question.get("category") == "instruction_following"
        and question.get("livebench_release_date", "") < "2025-11-25"
        and answer.get("eval_status") not in {"api_error", "token_exhaustion"}
    ):
        from livebench.process_results.instruction_following.utils import (
            instruction_following_process_results,
        )

        answer = copy.deepcopy(answer)
        answer.setdefault("question_id", question["question_id"])
        turns = answer["choices"][0]["turns"]
        turns[0] = re.sub(r"<think>.*?</think>", "", turns[0], flags=re.DOTALL).strip()
        # The official legacy writer assumes the CLI already created this directory.
        (Path("data/live_bench/instruction_following") / question["task"] / "model_judgment").mkdir(
            parents=True, exist_ok=True
        )
        scores = instruction_following_process_results(
            [question], {model: {question["question_id"]: answer}}, question["task"], model, False
        )
        if len(scores) != 1 or scores[0]["question_id"] != question["question_id"]:
            raise ValueError("Official legacy scorer returned another question")
        result = {
            **scores[0],
            "task": question["task"],
            "model": model,
            "category": "instruction_following",
            "turn": 1,
            "tstamp": time.time(),
            "grader_route": "official-legacy-instruction-following",
        }
    else:
        result = play_a_match_gt(MatchSingle(question, model, answer))
    output.write_text(json.dumps(result, default=str, allow_nan=False))


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(Path(sys.argv[2]), int(sys.argv[3]), "--coding" in sys.argv[4:])
    elif sys.argv[1] == "grade":
        grade(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise ValueError("Unknown official benchmark operation")
