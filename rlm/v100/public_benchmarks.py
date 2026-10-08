"""Pinned public LiveBench tasks, official graders and clearly scoped comparisons."""

import json
import math
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.inference import generation_conditions
from rlm.v100.protection import file_hash

SCORER_REVISION = "24364d65076429adfcba7be18af4d44fddc43dce"


def question_key(row):
    return "/".join(str(row[k]) for k in ("category", "task", "question_id"))


def prepare(root: Path, limit: int = 20, coding: bool = False) -> Path:
    if type(limit) is not int or limit != 0 and not 5 <= limit <= 500:
        raise ValueError("Use 5..500 public cases, or 0 for the entire pinned release")
    from rlm.v100.mission import status

    if status(root)["running"]:
        raise ValueError("Prepare pinned benchmark snapshots while the mission is stopped")
    folder = root / "research/public-benchmarks" / ("snapshot-" + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    environment = root / "venvs/official-benchmarks"
    python = environment / "bin/python"
    if not python.exists():
        subprocess.run(
            ["uv", "--no-config", "venv", "--python", sys.executable, str(environment)], check=True
        )
    subprocess.run(
        [
            "uv",
            "--no-config",
            "pip",
            "install",
            "--python",
            str(python),
            "--index-url",
            "https://pypi.org/simple",
            "livebench @ git+https://github.com/LiveBench/LiveBench.git@" + SCORER_REVISION,
        ],
        check=True,
    )
    worker = Path(__file__).with_name("benchmark_worker.py")
    subprocess.run(
        [str(python), str(worker), "prepare", str(folder), str(limit)]
        + (["--coding"] if coding else []),
        check=True,
    )
    frozen = subprocess.check_output(
        ["uv", "--no-config", "pip", "freeze", "--python", str(python)], text=True
    )
    (folder / "requirements.txt").write_text(frozen)
    manifest = json.loads((folder / "snapshot.json").read_text())
    manifest.update(
        python=str(python),
        scorer_revision=SCORER_REVISION,
        questions_sha256=file_hash(folder / "questions.json"),
        references_sha256=file_hash(folder / "references.json"),
        worker_sha256=file_hash(worker),
        created_at=time.time(),
        scope="Official tasks and grader; full six-category pinned release"
        if coding and limit == 0
        else "Official tasks and grader; selected panel is NOT the overall LiveBench leaderboard score",
    )
    atomic_json(folder / "manifest.json", manifest)
    atomic_json(root / "research/public-benchmarks/current.json", {"snapshot": str(folder)})
    return folder


def current(root: Path) -> Path | None:
    path = root / "research/public-benchmarks/current.json"
    return Path(json.loads(path.read_text())["snapshot"]) if path.exists() else None


def grade(root: Path, snapshot: Path, question: dict, answer: dict, folder: Path) -> dict:
    if question.get("category") == "coding":
        from rlm.v100.guest_benchmarks import grade as guest_grade

        return guest_grade(root, snapshot, question, answer, folder)
    from rlm.v100.research_sandbox import runtime

    pin = json.loads((snapshot / "manifest.json").read_text())
    python = Path(pin["python"])
    worker = Path(__file__).with_name("benchmark_worker.py")
    if file_hash(worker) != pin["worker_sha256"]:
        raise ValueError("Official grading adapter changed; prepare a new snapshot")
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(
        folder / "request.json",
        {"question": question, "model": "local-candidate", "answer": answer},
    )
    # Graders may parse model-produced mathematical expressions. They have no
    # network, credentials, other datasets or write access to the host installation.
    args = [
        runtime(root),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--cap-drop",
        "ALL",
    ]
    for path in dict.fromkeys(
        [Path("/usr"), Path("/lib"), Path("/lib64"), Path(sys.base_prefix), python.parent.parent]
    ):
        if path.exists():
            args += ["--ro-bind", str(path), str(path)]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(worker),
        "/worker.py",
        "--bind",
        str(folder),
        "/work",
        "--ro-bind",
        str(snapshot / "nltk"),
        "/nltk",
        "--setenv",
        "NLTK_DATA",
        "/nltk",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "OMP_NUM_THREADS",
        "1",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--chdir",
        "/work",
        str(python),
        "-I",
        "-c",
        "import resource,runpy,sys; resource.setrlimit(resource.RLIMIT_AS,(6*2**30,6*2**30)); "
        "resource.setrlimit(resource.RLIMIT_CPU,(90,90)); resource.setrlimit(resource.RLIMIT_FSIZE,(8*2**20,8*2**20)); "
        "sys.argv=['/worker.py','grade','/work/request.json','/work/result.json']; runpy.run_path('/worker.py',run_name='__main__')",
    ]
    with (folder / "grader.log").open("w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=100, check=True)
    result = json.loads((folder / "result.json").read_text())
    if result.get("eval_status") == "eval_error" or result.get("error_msg"):
        raise ValueError("Official scorer error: " + str(result)[:300])
    score = result.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Official grader returned an invalid score")
    return result


def summary(rows: list[dict]) -> dict:
    by_category = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["score"])
    means = {category: statistics.mean(scores) for category, scores in by_category.items()}
    return {
        "cases": len(rows),
        "category_means": means,
        "panel_macro_mean": statistics.mean(means.values()) if means else None,
    }


def published(snapshot: Path, cases: list[dict]) -> list[dict]:
    wanted = {row["key"] for row in cases}
    models = {}
    for row in json.loads((snapshot / "references.json").read_text()):
        key = question_key(row)
        score = row.get("score")
        if (
            key not in wanted
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            continue
        if row.get("eval_status") == "eval_error":
            continue
        entries = models.setdefault(row["model"], {})
        # Repeated official judgments select the latest dated row, never the best score.
        old = entries.get(key)
        if old is None or str(row.get("tstamp", "")) > str(old.get("tstamp", "")):
            entries[key] = row
    output = []
    for model, entries in models.items():
        if set(entries) == wanted:
            output.append(
                {
                    "model": model,
                    **summary(list(entries.values())),
                    "comparison": "Published same-question reference; provider generation budgets/settings may differ",
                }
            )
    return sorted(output, key=lambda row: row["panel_macro_mean"], reverse=True)[:20]


def evaluate(root: Path, profile: dict, output: Path, snapshot: Path | None = None) -> dict:
    from rlm.v100.competition import helper_client
    from rlm.v100.remote_helper import remote_profile

    snapshot = snapshot or current(root)
    if snapshot is None:
        raise ValueError("Prepare official benchmarks first")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if (
        file_hash(snapshot / "questions.json") != manifest["questions_sha256"]
        or file_hash(snapshot / "references.json") != manifest["references_sha256"]
    ):
        raise ValueError("Pinned public benchmark inputs changed")
    conditions = generation_conditions(profile)
    model_sha = (
        profile["resources"]["model_digest"]
        if remote_profile(profile)
        else file_hash(Path(profile["server"]["model"]))
    )
    identity = {
        "snapshot_sha256": file_hash(snapshot / "manifest.json"),
        "model_sha256": model_sha,
        "generation": conditions,
    }
    partial = output.with_suffix(".partial.json")
    if output.exists():
        result = json.loads(output.read_text())
        if result["identity"] != identity:
            raise ValueError("Existing official report has different conditions")
        return result
    report = (
        json.loads(partial.read_text())
        if partial.exists()
        else {
            "identity": identity,
            "model_version": profile["runtime"]["model_version"],
            "cases": [],
            "scope": manifest["scope"],
            "snapshot": str(snapshot),
        }
    )
    if report["identity"] != identity:
        raise ValueError("Partial official evaluation differs; choose a new report path")
    client = helper_client(
        profile
    )  # No research memory/activity: benchmark answers never become training examples.
    if remote_profile(profile):
        client.identity()
    else:
        from rlm.v100.serving import assert_served_expert

        assert_served_expert(client, profile, root)
    client.pacing_root = str(root)
    rows = json.loads((snapshot / "questions.json").read_text())
    done = {row["key"] for row in report["cases"]}
    for question in rows:
        key = question_key(question)
        if key in done:
            continue
        started, turns, messages = time.monotonic(), [], []
        error, finish = None, None
        try:
            for prompt in question["turns"]:
                messages.append({"role": "user", "content": prompt})
                text = client.completion(messages)
                finish = client.get_response_info().get("finish_reason")
                turns.append(text)
                messages.append({"role": "assistant", "content": text})
                if finish == "length":
                    break
        except (ValueError, OSError) as failure:
            error = str(failure)[:400]
        answer = {"choices": [{"turns": turns or [""]}]}
        if error or finish == "length":
            answer["eval_status"] = "api_error" if error else "token_exhaustion"
        grading = output.parent / (output.stem + "-grading") / str(len(report["cases"]))
        try:
            result = grade(root, snapshot, question, answer, grading)
        except (ValueError, OSError, subprocess.SubprocessError) as failure:
            # Grader errors block promotion instead of masquerading as zero model ability.
            atomic_json(
                output.with_suffix(".error.json"), {"key": key, "error": str(failure)[:500]}
            )
            raise RuntimeError("Official grading failed; report remains partial") from failure
        report["cases"].append(
            {
                "key": key,
                "category": question["category"],
                "score": result["score"],
                "error": error,
                "finish_reason": finish,
                "seconds": time.monotonic() - started,
            }
        )
        atomic_json(partial, report)
        atomic_json(
            root / "research/public-benchmarks/progress.json",
            {
                "model": report["model_version"],
                "completed": len(report["cases"]),
                "total": len(rows),
                "report": str(output),
            },
        )
        print(
            json.dumps(
                {
                    "official_benchmark": len(report["cases"]),
                    "total": len(rows),
                    "model": report["model_version"],
                }
            ),
            flush=True,
        )
    report.update(summary(report["cases"]))
    report["published_same_cases"] = published(snapshot, report["cases"])
    report["finished_at"] = time.time()
    report["complete"] = len(report["cases"]) == len(rows)
    atomic_json(output, report)
    return report


def reusable(root: Path, profile: dict) -> Path | None:
    """Reuse only the pinned complete baseline for this exact model and generation."""
    resources = profile.get("resources", {})
    pin = resources.get("public_baseline_sha256")
    value = resources.get("public_baseline")
    snapshot = current(root)
    if not value or not pin or snapshot is None:
        return None
    path = Path(value).resolve()
    if not path.is_relative_to((root / "research").resolve()):
        raise ValueError("Public baseline must be inside the owned research directory")
    if not path.exists():
        return None
    if file_hash(path) != pin:
        raise ValueError("Pinned public baseline changed")
    result = json.loads(path.read_text())
    from rlm.v100.remote_helper import remote_profile

    model_sha = (
        resources["model_digest"]
        if remote_profile(profile)
        else file_hash(Path(profile["server"]["model"]))
    )
    expected = {
        "snapshot_sha256": file_hash(snapshot / "manifest.json"),
        "model_sha256": model_sha,
        "generation": generation_conditions(profile),
    }
    rows = json.loads((snapshot / "questions.json").read_text())
    keys = [row["key"] for row in result.get("cases", [])]
    if (
        result.get("complete")
        and result.get("identity") == expected
        and len(keys) == len(set(keys)) == len(rows)
        and set(keys) == {question_key(row) for row in rows}
    ):
        return path
    return None


def compare(parent: dict, child: dict) -> dict:
    if not parent.get("complete") or not child.get("complete"):
        raise ValueError("Both official reports must be complete")
    for field in ("snapshot_sha256", "generation"):
        if parent["identity"][field] != child["identity"][field]:
            raise ValueError("Official benchmark dataset or generation settings differ")
    old = {r["key"]: r for r in parent["cases"]}
    new = {r["key"]: r for r in child["cases"]}
    if old.keys() != new.keys():
        raise ValueError("Official cases differ")
    regressions = [key for key in old if new[key]["score"] + 1e-9 < old[key]["score"]]
    return {
        "passed": not regressions
        and not any(
            row.get("error") or row.get("finish_reason") == "length" for row in new.values()
        ),
        "regressed_cases": len(regressions),
        "cases": len(old),
        "scope": "Finite public panel; repeated development use is not an independent hidden audit",
    }
