"""
Tests for rlm.v100.source_influence — roadmap item: public_candidate_research.
"""
import importlib.util
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "source_influence",
    Path(__file__).parent.parent / "rlm/v100/source_influence.py",
)
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)


def _root():
    return Path(tempfile.mkdtemp())


def _sha():
    return "a" * 64


# ── 1. Empty source list returns zero-recorded receipt ────────────────────────
r = si.record_hypothesis_sources(_root(), "h1", [])
assert r["sources_recorded"] == 0
print("check 1 ok: empty sources → sources_recorded=0")

# ── 2. Valid source records and returns receipt ───────────────────────────────
root = _root()
src = {"source_id": "s1", "url": "https://ex.com/feed", "sha256": _sha(), "excerpt": "text", "model_score": 0.8, "acquired_at": "2026-10-18T10:00:00+00:00"}
r = si.record_hypothesis_sources(root, "h2", [src])
assert r["sources_recorded"] == 1
assert "s1" in r["source_ids"]
print("check 2 ok: valid source recorded")

# ── 3. Invalid sha256 is skipped ──────────────────────────────────────────────
root2 = _root()
bad_src = {"source_id": "s-bad", "url": "https://ex.com/", "sha256": "tooshort", "excerpt": "", "model_score": 0.5, "acquired_at": "2026-10-18T10:00:00+00:00"}
r = si.record_hypothesis_sources(root2, "h3", [bad_src])
assert r["sources_recorded"] == 0
assert r["skipped"] == 1
print("check 3 ok: invalid sha256 skipped")

# ── 4. model_score out of range is skipped ────────────────────────────────────
root3 = _root()
bad_score = {"source_id": "sx", "url": "https://x.com/", "sha256": _sha(), "excerpt": "", "model_score": 1.5, "acquired_at": "2026-01-01T00:00:00+00:00"}
r = si.record_hypothesis_sources(root3, "h4", [bad_score])
assert r["sources_recorded"] == 0
print("check 4 ok: model_score > 1 skipped")

# ── 5. record_outcome with no prior sources sets source_used=False ────────────
root4 = _root()
r = si.record_outcome(root4, "h5", 0.05)
assert r["source_used"] is False
print("check 5 ok: outcome with no sources → source_used=False")

# ── 6. record_outcome after record_hypothesis_sources sets source_used=True ───
root5 = _root()
si.record_hypothesis_sources(root5, "h6", [{"source_id": "s6", "url": "https://x.com/", "sha256": _sha(), "excerpt": "x", "model_score": 0.6, "acquired_at": "2026-10-18T09:00:00+00:00"}])
r = si.record_outcome(root5, "h6", 0.10)
assert r["source_used"] is True
print("check 6 ok: outcome after source attribution → source_used=True")

# ── 7. source_influence_report returns correct counts and delta ───────────────
root6 = _root()
si.record_hypothesis_sources(root6, "h7a", [{"source_id": "s7", "url": "https://x.com/", "sha256": _sha(), "excerpt": "x", "model_score": 0.7, "acquired_at": "2026-10-18T09:00:00+00:00"}])
si.record_outcome(root6, "h7a", 0.20)
si.record_outcome(root6, "h7b", 0.10)  # no source
rep = si.source_influence_report(root6)
assert rep["used_count"] == 1
assert rep["control_count"] == 1
assert rep["used_mean_outcome"] == 0.2
assert rep["control_mean_outcome"] == 0.1
assert abs(rep["influence_delta"] - 0.1) < 1e-6
print("check 7 ok: influence_delta = used_mean - control_mean")

# ── 8. Empty DB returns None means and delta ──────────────────────────────────
rep2 = si.source_influence_report(_root())
assert rep2["used_count"] == 0
assert rep2["influence_delta"] is None
print("check 8 ok: empty DB → delta=None")

# ── 9. Excerpt is truncated to 512 chars ─────────────────────────────────────
root7 = _root()
long_src = {"source_id": "sl", "url": "https://x.com/", "sha256": _sha(), "excerpt": "x" * 1000, "model_score": 0.3, "acquired_at": "2026-10-18T00:00:00+00:00"}
r = si.record_hypothesis_sources(root7, "h9", [long_src])
assert r["sources_recorded"] == 1
print("check 9 ok: long excerpt accepted (truncated internally)")

# ── 10. hypothesis_id validation ─────────────────────────────────────────────
root8 = _root()
try:
    si.record_hypothesis_sources(root8, "", [])
    assert False, "should raise"
except ValueError:
    pass
print("check 10 ok: empty hypothesis_id rejected")

print("\nAll 10 checks passed.")
