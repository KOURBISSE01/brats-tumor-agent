"""Schémas Pydantic du pipeline agent BraTS — validation des échanges inter-étapes."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Workflow(str, Enum):
    """Workflows de l'agent (planification par règles, fallback LLM)."""

    RAPIDE = "rapide"            # segmentation + résumé
    STANDARD = "standard"        # + biomarqueurs + risque
    COMPLET = "complet"          # + rapport rédigé + RAG + évaluation si GT
    COMPARAISON = "comparaison"  # deux patients, table comparatif


class RiskLevel(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


class Biomarkers(BaseModel):
    """Les 11 biomarqueurs BraTS (5 volumes + 4 ratios + 2 forme/division)."""

    # volumes (cm³)
    volume_wt_cm3: float = Field(ge=0.0, description="Whole Tumor = labels 1+2+3")
    volume_tc_cm3: float = Field(ge=0.0, description="Tumor Core = labels 1+3")
    volume_et_cm3: float = Field(ge=0.0, description="Enhancing Tumor = label 3")
    volume_edema_cm3: float = Field(ge=0.0, description="Edema = label 2 (région péritumorale)")
    volume_necrosis_cm3: float = Field(ge=0.0, description="Nécrose = label 1 (non-contrasté)")
    # ratios (0..1,0 si dénominateur nul)
    ratio_et_wt: float = Field(ge=0.0, le=1.0)
    ratio_et_tc: float = Field(ge=0.0, le=1.0)
    ratio_tc_wt: float = Field(ge=0.0, le=1.0)
    ratio_edema_wt: float = Field(ge=0.0, le=1.0)
    # forme / division
    sphericity_wt: float = Field(ge=0.0, description="Sphéricité WT : 1 = sphère parfaite, 0 = masque vide")
    n_components: int = Field(ge=0, description="Composantes connexes WT (volume min configurable)")


class RiskAssessment(BaseModel):
    level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    criteria: list[str] = Field(default_factory=list)
    note: str = (
        "Seuils calibrés sur les quantiles Q33/Q66 du jeu de test BraTS — "
        "heuristique d'aide à la décision, PAS une recommandation clinique."
    )


class DiceReport(BaseModel):
    dice_wt: float = Field(ge=0.0, le=1.0)
    dice_tc: float = Field(ge=0.0, le=1.0)
    dice_et: float = Field(ge=0.0, le=1.0)
    mean_dice: float = Field(ge=0.0, le=1.0)


class StepLog(BaseModel):
    name: str
    ms: float = Field(ge=0.0)
    ok: bool
    error: Optional[str] = None


class PatientSummary(BaseModel):
    """Résumé affichable pour le dashboard / l'API."""

    case_id: str
    risk: Optional[RiskAssessment] = None
    biomarkers: Optional[Biomarkers] = None
    dice: Optional[DiceReport] = None
    backend: str = "mock"


class AgentResult(BaseModel):
    workflow: Workflow
    steps: list[StepLog] = Field(default_factory=list)
    biomarkers: Optional[Biomarkers] = None
    risk: Optional[RiskAssessment] = None
    dice: Optional[DiceReport] = None
    second: Optional[PatientSummary] = None   # workflow comparaison
    report: str = ""
    grounded: bool = True
    rag_refs: list[str] = Field(default_factory=list)
    skill: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
