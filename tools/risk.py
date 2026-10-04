"""Classification de risque LOW / MED / HIGH — règles du projet BraTS.

Seuils d'origine (calibrés Q33/Q66 des volumes — voir rag/docs/regles_risque.md) :
- LOW  : WT < 62 cm³  ET < 11 cm³  ET  1 seule composante connexe        → conf 0.85
- HIGH : WT > 115 cm³  OU  ET > 25 cm³  OU  ≥ 3 composantes connexes
         conf = 0.60 + 0.15 × n_critères_déclenchés  (plafonné à 1.0)
- MED  : sinon                                                      → conf 0.70

Priorité LOW avant HIGH (les conditions sont mutuellement exclusives).
"""
from __future__ import annotations

from agent.schemas import Biomarkers, RiskAssessment, RiskLevel

LOW_WT_CM3 = 62.0
LOW_ET_CM3 = 11.0
HIGH_WT_CM3 = 115.0
HIGH_ET_CM3 = 25.0
HIGH_MIN_COMPONENTS = 3

CONF_LOW = 0.85
CONF_MED = 0.70
CONF_HIGH_BASE = 0.60
CONF_HIGH_STEP = 0.15


def classify_risk(b: Biomarkers) -> RiskAssessment:
    criteria: list[str] = []

    # --- LOW : les trois conditions simultanément
    low_ok = (
        b.volume_wt_cm3 < LOW_WT_CM3
        and b.volume_et_cm3 < LOW_ET_CM3
        and b.n_components == 1
    )
    if low_ok:
        criteria.append(f"WT {b.volume_wt_cm3:.2f} < {LOW_WT_CM3:.0f} cm³")
        criteria.append(f"ET {b.volume_et_cm3:.2f} < {LOW_ET_CM3:.0f} cm³")
        criteria.append("1 composante connexe unique")
        return RiskAssessment(level=RiskLevel.LOW, confidence=CONF_LOW, criteria=criteria)

    # --- HIGH : n'importe quel critère
    high_hits = 0
    if b.volume_wt_cm3 > HIGH_WT_CM3:
        high_hits += 1
        criteria.append(f"WT {b.volume_wt_cm3:.2f} > {HIGH_WT_CM3:.0f} cm³")
    if b.volume_et_cm3 > HIGH_ET_CM3:
        high_hits += 1
        criteria.append(f"ET {b.volume_et_cm3:.2f} > {HIGH_ET_CM3:.0f} cm³")
    if b.n_components >= HIGH_MIN_COMPONENTS:
        high_hits += 1
        criteria.append(f"{b.n_components} composantes connexes ≥ {HIGH_MIN_COMPONENTS}")

    if high_hits > 0:
        conf = min(CONF_HIGH_BASE + CONF_HIGH_STEP * high_hits, 1.0)
        return RiskAssessment(level=RiskLevel.HIGH, confidence=round(conf, 4), criteria=criteria)

    # --- MED : défaut
    criteria.append("aucun seuil LOW ni HIGH franchi")
    return RiskAssessment(level=RiskLevel.MED, confidence=CONF_MED, criteria=criteria)
