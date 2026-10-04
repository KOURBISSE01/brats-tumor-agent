"""Cœur agent BraTS : planification par règles (+ fallback LLM), pipeline à
contexte avec isolation d'erreur par étape, RAG, skills et grounding des nombres."""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from agent.prompts import build_report, llm_explain
from agent.schemas import (
    AgentResult,
    Biomarkers,
    DiceReport,
    RiskAssessment,
    StepLog,
    Workflow,
)
from agent.skills_loader import skill_for_workflow
from rag.knowledge_base import query as rag_query
from tools.biomarkers import compute_biomarkers
from tools.evaluation import evaluate_dice
from tools.risk import classify_risk
from tools.segmentation import segment_volume

log = logging.getLogger("brats.agent")

# ---------------------------------------------------------------------------
# Planning : règles d'abord (déterministe, testable), LLM en secours
# ---------------------------------------------------------------------------
# Texte normalisé (accents retirés) avant matching : « complète » == « complete »
INTENT_RULES: list[tuple[Workflow, re.Pattern]] = [
    (Workflow.COMPARAISON, re.compile(r"\b(compar\w*|versus|vs\.?|deux patients?|cote a cote|contre)\b", re.I)),
    (Workflow.COMPLET, re.compile(r"\b(complet\w*|approfondi|detail\w*|rapport|report|evalu\w*|evaluate|dice|ground.?truth)\b", re.I)),
    (Workflow.RAPIDE, re.compile(r"\b(rapide|vite|quick|synth\w*|resume|simplifi\w*)\b", re.I)),
]


def _normalize(text: str) -> str:
    """Minuscules + suppression des accents (complète → complete)."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def classify_intent(text: str) -> Workflow:
    """Workflow demandé par l'utilisateur. Défaut : STANDARD (règles, pas de LLM)."""
    normalized = _normalize(text or "")
    for workflow, pattern in INTENT_RULES:
        if pattern.search(normalized):
            return workflow
    return Workflow.STANDARD


# ---------------------------------------------------------------------------
# Case in-memory (l'API et le MCP passent par des fichiers, ici on a des arrays)
# ---------------------------------------------------------------------------
@dataclass
class Case:
    case_id: str
    volume: np.ndarray                    # (4, D, H, W) — T1n, T1c, T2w, FLAIR
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gt_labels: Optional[np.ndarray] = None  # (D, H, W) labels 1/2/3
    model_path: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grounding : tout nombre à décimale du rapport doit exister dans les faits
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+\.\d+")


def extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(text)]


def collect_allowed(ctx: dict) -> list[float]:
    allowed: list[float] = []
    for key in ("biomarkers", "risk", "dice"):
        obj = ctx.get(key)
        if obj is None:
            continue
        for v in obj.model_dump().values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                allowed.append(float(v))
    # les scores RAG cités dans le rapport sont des sorties d'outil → racines
    for ref in ctx.get("rag_refs", []) or []:
        allowed.extend(extract_numbers(str(ref)))
    return allowed


def check_grounding(report: str, ctx: dict) -> tuple[bool, list[str]]:
    """Retourne (grounded, nombres_non_racinés). Tolérance 1 % (arrondis d'affichage)."""
    allowed = collect_allowed(ctx)
    rogue: list[float] = []
    for n in extract_numbers(report):
        if not any(abs(n - a) <= max(0.01 * abs(a), 0.01) for a in allowed):
            rogue.append(n)
    return (not rogue), rogue


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _facts(ctx: dict) -> dict:
    """Faits sérialisables pour le prompt LLM / le rapport."""
    facts = {}
    for key in ("biomarkers", "risk", "dice", "workflow", "backend", "case_id"):
        if ctx.get(key) is not None:
            v = ctx[key]
            facts[key] = v.model_dump() if hasattr(v, "model_dump") else v
    return facts


def run_agent(
    user_text: str,
    case: Case,
    second_case: Optional[Case] = None,
    use_llm: Optional[bool] = None,
) -> AgentResult:
    """Exécute le pipeline complet. Chaque étape est isolée : une erreur ne casse
    que cette étape (visible dans steps[] + warnings)."""
    workflow = classify_intent(user_text)
    skill_name, skill_text = skill_for_workflow(workflow.value)
    steps: list[StepLog] = []
    warnings: list[str] = list(case.warnings)
    ctx: dict = {
        "case_id": case.case_id,
        "workflow": workflow.value,
        "workflow_enum": workflow,
    }

    def step(name: str, fn: Callable, *args, **kw):
        t0 = time.perf_counter()
        try:
            out = fn(*args, **kw)
            ok, err = True, None
        except Exception as exc:  # isolation par étape
            out, ok, err = None, False, f"{type(exc).__name__}: {exc}"
            log.warning("étape %s en échec : %s", name, err)
        steps.append(StepLog(name=name, ms=(time.perf_counter() - t0) * 1000, ok=ok, error=err))
        return out

    # 1. Segmentation (toujours requise)
    seg = step("segmentation", segment_volume, case.volume, spacing=case.spacing, model_path=case.model_path)
    labels = None
    if seg is not None:
        labels, meta = seg
        ctx["labels"] = labels
        ctx["backend"] = meta.get("backend", "mock")
        warnings.extend(meta.get("warnings", []))
    else:
        ctx["backend"] = "unavailable"

    # 2. Biomarqueurs + risque (workflow standard et au-delà)
    biomarkers: Biomarkers | None = None
    risk: RiskAssessment | None = None
    if labels is not None and workflow != Workflow.RAPIDE:
        biomarkers = step("biomarqueurs", compute_biomarkers, labels, spacing=case.spacing)
        ctx["biomarkers"] = biomarkers
        if biomarkers is not None:
            risk = step("risque", classify_risk, biomarkers)
            ctx["risk"] = risk

    # 3. Évaluation si ground truth (workflow complet)
    dice: DiceReport | None = None
    if labels is not None and case.gt_labels is not None and workflow == Workflow.COMPLET:
        dice = step("evaluation", evaluate_dice, labels, case.gt_labels)
        ctx["dice"] = dice

    # 4. Comparaison (2e patient)
    second_summary = None
    if workflow == Workflow.COMPARAISON and second_case is not None:
        seg2 = step("segmentation_2", segment_volume, second_case.volume,
                    spacing=second_case.spacing, model_path=second_case.model_path)
        if seg2 is not None:
            labels2, meta2 = seg2
            bio2 = step("biomarqueurs_2", compute_biomarkers, labels2, spacing=second_case.spacing)
            risk2 = step("risque_2", classify_risk, bio2) if bio2 is not None else None
            if bio2 is not None:
                from agent.schemas import PatientSummary
                second_summary = PatientSummary(
                    case_id=second_case.case_id, risk=risk2, biomarkers=bio2,
                    backend=meta2.get("backend", "mock"),
                )
                if biomarkers is not None and risk is not None:
                    ctx["comparison"] = {
                        "second_id": second_case.case_id,
                        "wt1": biomarkers.volume_wt_cm3, "wt2": bio2.volume_wt_cm3,
                        "et1": biomarkers.volume_et_cm3, "et2": bio2.volume_et_cm3,
                        "risk1": risk.level.value, "risk2": risk2.level.value if risk2 else "?",
                    }

    # 5. RAG — requêtes contextuelles selon le résultat
    rag_refs: list[str] = []
    if workflow in (Workflow.COMPLET, Workflow.COMPARAISON) and risk is not None:
        hits = step(
            "rag",
            lambda: rag_query(
                f"interprétation {risk.level.value} biomarqueurs volumes risque",
                k=2,
            ),
        )
        if hits:
            for h in hits:
                rag_refs.append(f"{h['source']} · {h['heading']} (score {h['score']:.3f})")
                ctx.setdefault("rag_texts", []).append(h["text"][:600])
    ctx["rag_refs"] = rag_refs

    # 6. Rapport : déterministe d'abord, LLM optionnel si use_llm=True + clé
    report = ""
    grounded = True
    want_llm = use_llm  # None = auto (LLM seulement si explicitement demandé)
    if want_llm:
        try:
            report = step("rapport_llm", llm_explain, user_text, _facts(ctx), skill_text)
        except Exception:
            report = None
    if not report:
        report = step("rapport", lambda: build_report(ctx)) or ""
    grounded, rogue = check_grounding(report, ctx)
    if not grounded:
        warnings.append(f"Grounding échoué — nombres non racinés : {rogue}")

    return AgentResult(
        workflow=workflow,
        steps=steps,
        biomarkers=biomarkers,
        risk=risk,
        dice=dice,
        second=second_summary,
        report=report,
        grounded=grounded,
        rag_refs=rag_refs,
        skill=skill_name or None,
        warnings=warnings,
    )
