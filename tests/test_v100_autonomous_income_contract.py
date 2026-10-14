from pathlib import Path


def test_income_research_requires_autonomous_test_and_independent_evidence():
    source = (
        Path(__file__).resolve().parents[1] / "rlm/v100/income_work.py"
    ).read_text()
    assert "No predetermined asset" in source
    assert "No predetermined asset, " in source
    assert "autonomy_contract" in source
    assert "independent_promotion_checks" in source
    assert "promotion_gate({})" in source
    assert "Each candidate needs a testable failure condition" in source
    assert "A losing or " in source
    assert "inconclusive trial cannot be labelled successful" in source
