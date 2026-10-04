"""RAG BM25 : pertinence des requêtes, robustesse tokenisation."""
from rag.knowledge_base import KnowledgeBase, query, reset


def test_query_risk_thresholds_finds_risk_doc():
    hits = query("seuils classification risque HIGH WT ET volumes", k=3)
    assert hits, "aucun hit pour une requête pourtant centrale"
    assert hits[0]["source"] == "regles_risque.md"
    assert hits[0]["score"] > 0


def test_query_dice_finds_evaluation_doc():
    hits = query("dice variance ET inter-observateurs interpretation", k=3)
    assert hits[0]["source"] == "interpretation_dice.md"


def test_query_regions_finds_labels_doc():
    hits = query("labels segmentation WT TC ET hiérarchie", k=3)
    assert hits[0]["source"] == "brats_regions.md"


def test_query_training_finds_protocol_doc():
    hits = query("DynUNet entraînement deep supervision perte dice focal", k=3)
    assert hits[0]["source"] == "protocole_entrainement.md"


def test_accent_insensitive():
    """« sphéricité » et « sphericite » doivent remonter les biomarqueurs."""
    a = query("sphéricité formule biomarqueurs", k=1)
    b = query("sphericite formule biomarqueurs", k=1)
    assert a[0]["source"] == b[0]["source"] == "biomarqueurs.md"


def test_empty_and_stopword_queries():
    assert query("") == []
    assert query("le la de des") == []   # que des stopwords → vide


def test_k_limit_respected():
    hits = query("risque volume tumeur segmentation dice", k=2)
    assert len(hits) <= 2
    # scores décroissants
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_kb_loads_from_alternate_dir(tmp_path):
    (tmp_path / "doc.md").write_text("# Titre\n## Section A\ntexte brazos uniquement ici\n", encoding="utf-8")
    kb = KnowledgeBase(docs_dir=tmp_path)
    hits = kb.query("brazos", k=1)
    assert len(hits) == 1
    assert hits[0]["heading"] == "Section A"


def test_reset_reloads_singleton():
    reset()
    first = query("risque")
    reset()
    second = query("risque")
    assert [h["source"] for h in first] == [h["source"] for h in second]
