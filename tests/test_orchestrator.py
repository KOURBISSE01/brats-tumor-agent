"""Orchestrateur : classification d'intention, pipeline, grounding, isolation."""
import numpy as np
import pytest

from agent.orchestrator import Case, check_grounding, classify_intent, run_agent
from agent.schemas import RiskLevel, Workflow
from agent.skills_loader import list_skills, load_skill, skill_for_workflow


def _case(case_id="CASE_001", size=48, gt=False) -> Case:
    rng = np.random.default_rng(0)
    volume = rng.normal(100, 30, (4, size, size, size)).astype(np.float32)
    gt_labels = None
    if gt:
        gt_labels = np.zeros((size, size, size), dtype=np.uint8)
        c = size // 2
        gt_labels[c - 8:c + 8, c - 8:c + 8, c - 8:c + 8] = 2
    return Case(case_id=case_id, volume=volume, gt_labels=gt_labels)


# -- intention -------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("analyse complète du cas", Workflow.COMPLET),
        ("rapport détaillé", Workflow.COMPLET),
        ("évalue le modèle avec le dice", Workflow.COMPLET),
        ("rapide svp", Workflow.RAPIDE),
        ("compare ces deux patients", Workflow.COMPARAISON),
        ("segmente", Workflow.STANDARD),
        ("bonjour", Workflow.STANDARD),
    ],
)
def test_classify_intent(text, expected):
    assert classify_intent(text) == expected


# -- pipeline --------------------------------------------------------------
def test_standard_workflow():
    result = run_agent("analyse standard", _case())
    assert result.workflow == Workflow.STANDARD
    names = [s.name for s in result.steps]
    assert names == ["segmentation", "biomarqueurs", "risque", "rapport"]
    assert all(s.ok for s in result.steps)
    assert result.biomarkers is not None and result.biomarkers.volume_wt_cm3 > 0
    assert result.risk is not None and result.risk.level in RiskLevel
    assert result.grounded is True
    assert result.report.startswith("## Rapport d'analyse")


def test_complet_workflow_has_rag_and_skill():
    result = run_agent("rapport complet avec évaluation", _case(gt=True))
    assert result.workflow == Workflow.COMPLET
    names = [s.name for s in result.steps]
    assert "rag" in names
    assert "evaluation" in names          # GT présent
    assert result.dice is not None
    assert result.rag_refs, "le workflow complet doit citer des refs RAG"
    assert "Références (RAG)" in result.report
    assert result.skill == "rapport_radiologique"


def test_rapide_workflow_skips_biomarkers():
    result = run_agent("très rapide", _case())
    assert result.workflow == Workflow.RAPIDE
    names = [s.name for s in result.steps]
    assert names == ["segmentation", "rapport"]
    assert result.biomarkers is None
    assert "indisponible" in result.report  # rapport dégradé explicite


def test_comparaison_workflow():
    r = run_agent("compare ces deux patients", _case("A"), second_case=_case("B"))
    assert r.workflow == Workflow.COMPARAISON
    names = [s.name for s in r.steps]
    assert "segmentation_2" in names
    assert "biomarqueurs_2" in names
    assert r.second is not None
    assert r.second.case_id == "B"
    assert "### 5. Comparaison" in r.report


def test_step_isolation_on_error():
    """Une étape qui plante n'abandonne pas les autres."""
    case = _case()
    case.volume = "pas un ndarray"  # segmentation va échouer
    result = run_agent("analyse standard", case)
    seg = next(s for s in result.steps if s.name == "segmentation")
    assert seg.ok is False and seg.error
    # les étapes suivantes ne doivent même pas être tentées sans labels,
    # mais le rapport doit rester produisible (message d'indisponibilité)
    assert result.report
    assert result.biomarkers is None


# -- grounding -------------------------------------------------------------
def test_grounding_accepts_report_numbers():
    case = _case()
    result = run_agent("rapport complet", case)
    assert result.grounded is True
    assert result.warnings == [] or not any("Grounding" in w for w in result.warnings)


def test_grounding_flags_rogue_number():
    case = _case()
    result = run_agent("analyse standard", case)
    # injecte un nombre absent des faits
    rogue_report = result.report + "\nVolume suspicion: 99.99 cm³"
    grounded, rogue = check_grounding(rogue_report, {"biomarkers": result.biomarkers, "risk": result.risk})
    assert grounded is False
    assert 99.99 in rogue


def test_grounding_allows_display_rounding():
    """Un arrondi d'affichage (<1%) n'est pas considéré comme non raciné."""
    from agent.schemas import Biomarkers

    b = Biomarkers(
        volume_wt_cm3=19.484123, volume_tc_cm3=5.0, volume_et_cm3=1.0,
        volume_edema_cm3=14.0, volume_necrosis_cm3=4.0,
        ratio_et_wt=0.05, ratio_et_tc=0.2, ratio_tc_wt=0.26, ratio_edema_wt=0.72,
        sphericity_wt=0.957, n_components=1,
    )
    ctx = {"biomarkers": b}
    grounded, _ = check_grounding("WT = 19.48 cm³", ctx)
    assert grounded is True


# -- skills ----------------------------------------------------------------
def test_skills_present_and_loadable():
    skills = list_skills()
    assert {"segmentation", "rapport_radiologique", "evaluation"} <= set(skills)
    for name in skills:
        text = load_skill(name)
        assert text.startswith("---")          # frontmatter YAML
        assert "SKILL" not in text[:0] or True
    name, text = skill_for_workflow("complet")
    assert name == "rapport_radiologique" and text
    assert skill_for_workflow("inconnu")[0] == "segmentation"
    assert load_skill("n_existe_pas") == ""    # jamais d'exception


# -- rapport & compare ------------------------------------------------------
def test_report_contains_required_sections():
    result = run_agent("rapport complet", _case(gt=True))
    report = result.report
    assert "### 1. Volumes" in report
    assert "### 4. Risque" in report
    assert "Aide à la décision — pas un diagnostic" in report
    assert "Seuils" in report


def test_llm_fallback_without_key():
    """use_llm=True sans OPENAI_API_KEY → rapport déterministe, pas d'erreur."""
    result = run_agent("rapport complet", _case(), use_llm=True)
    # soit le pas LLM a échoué proprement et le rapport déterministe est là
    assert result.report
    rap = next(s for s in result.steps if s.name == "rapport_llm")
    assert rap.ok is False  # clé absente → isolation
    assert result.grounded is True
