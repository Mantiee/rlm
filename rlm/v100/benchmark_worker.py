"""Runs only in the separate pinned official-benchmark environment."""

import hashlib
import json
import os
import stat
import sys
from pathlib import Path


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
    for package in ("punkt", "punkt_tab", "averaged_perceptron_tagger_eng"):
        if not nltk.download(package, download_dir=str(resources), quiet=True):
            raise RuntimeError("Official grader resource download failed: " + package)


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
    from livebench.common import MatchSingle
    from livebench.gen_ground_truth_judgment import play_a_match_gt

    data = json.loads(request.read_text())
    result = play_a_match_gt(MatchSingle(data["question"], data["model"], data["answer"]))
    output.write_text(json.dumps(result, default=str, allow_nan=False))


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(Path(sys.argv[2]), int(sys.argv[3]), "--coding" in sys.argv[4:])
    elif sys.argv[1] == "grade":
        grade(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise ValueError("Unknown official benchmark operation")
