from rlm.v100.research_curriculum import curriculum, promotion_gate


def test_research_catalog_has_distinct_hypotheses_and_multiscale_horizons():
    entries = curriculum()
    assert len(entries) >= 5
    assert len({entry["name"] for entry in entries}) == len(entries)
    horizons = {h for entry in entries for h in entry["horizons"]}
    assert {"1h", "1d", "1w"}.issubset(horizons)
    for entry in entries:
        assert entry["controls"]
        assert entry["failure_condition"]
        assert entry["required_data"]


def test_evidence_gate_blocks_empty_unverified_or_real_money_promotion():
    empty = promotion_gate({})
    assert not empty["paper_research_qualified"]
    assert not empty["real_money_authorized"]
    assert "independent_forward_paper" in empty["missing_evidence"]
    all_evidence = {name: True for name in empty["missing_evidence"]}
    qualified = promotion_gate(all_evidence)
    assert qualified["paper_research_qualified"]
    assert qualified["missing_evidence"] == []
    assert not qualified["real_money_authorized"]


def test_missing_single_condition_fails_closed():
    checks = promotion_gate({})["missing_evidence"]
    for omitted in checks:
        evidence = {name: name != omitted for name in checks}
        assert omitted in promotion_gate(evidence)["missing_evidence"]
