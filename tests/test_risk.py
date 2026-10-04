"""Règles de risque LOW/MED/HIGH — seuils Q33/Q66 et confiances."""
import pytest

from agent.schemas import Biomarkers, RiskLevel
from tools.risk import classify_risk


def _bio(wt: float, et: float, n_comp: int) -> Biomarkers:
    """Biomarqueurs synthétiques cohérents (TC ≥ ET, WT ≥ TC)."""
    tc = max(et, wt * 0.4)
    return Biomarkers(
        volume_wt_cm3=wt,
        volume_tc_cm3=tc,
        volume_et_cm3=et,
        volume_edema_cm3=max(wt - tc, 0.0),
        volume_necrosis_cm3=max(tc - et, 0.0),
        ratio_et_wt=et / wt if wt else 0.0,
        ratio_et_tc=et / tc if tc else 0.0,
        ratio_tc_wt=tc / wt if wt else 0.0,
        ratio_edema_wt=0.0,
        sphericity_wt=0.9,
        n_components=n_comp,
    )


def test_low_case():
    r = classify_risk(_bio(wt=50, et=5, n_comp=1))
    assert r.level == RiskLevel.LOW
    assert r.confidence == pytest.approx(0.85)
    assert len(r.criteria) == 3


def test_low_fails_without_single_component():
    # petit volume mais 2 composantes → pas LOW, pas HIGH → MED
    r = classify_risk(_bio(wt=50, et=5, n_comp=2))
    assert r.level == RiskLevel.MED
    assert r.confidence == pytest.approx(0.70)


def test_high_by_wt():
    r = classify_risk(_bio(wt=120, et=5, n_comp=1))
    assert r.level == RiskLevel.HIGH
    assert r.confidence == pytest.approx(0.60 + 0.15)  # 1 critère


def test_high_by_et_only():
    r = classify_risk(_bio(wt=80, et=30, n_comp=1))
    assert r.level == RiskLevel.HIGH
    assert r.confidence == pytest.approx(0.75)


def test_high_by_three_components():
    r = classify_risk(_bio(wt=80, et=15, n_comp=3))
    assert r.level == RiskLevel.HIGH
    assert len(r.criteria) >= 1


def test_high_confidence_clamped_at_one():
    # les 3 critères : 0.60 + 0.45 = 1.05 → plafonné à 1.0
    r = classify_risk(_bio(wt=120, et=30, n_comp=4))
    assert r.level == RiskLevel.HIGH
    assert r.confidence == pytest.approx(1.0)
    assert r.confidence <= 1.0


def test_med_default():
    r = classify_risk(_bio(wt=80, et=15, n_comp=2))
    assert r.level == RiskLevel.MED
    assert r.confidence == pytest.approx(0.70)


def test_boundaries_are_exclusive_like_spec():
    # WT = 62 exact → pas < 62 → pas LOW
    r = classify_risk(_bio(wt=62.0, et=5, n_comp=1))
    assert r.level != RiskLevel.LOW
    # WT = 115 exact → pas > 115 → pas HIGH (MED ici)
    r2 = classify_risk(_bio(wt=115.0, et=5, n_comp=1))
    assert r2.level != RiskLevel.HIGH


def test_note_mentions_leakage():
    r = classify_risk(_bio(wt=50, et=5, n_comp=1))
    note = r.note.lower()
    assert "q33" in note and "recommandation clinique" in note
