"""Prompts et génération de rapport — toute affirmation quantitative doit être
racinée dans la sortie des outils (check de grounding dans orchestrator)."""
from __future__ import annotations

SYSTEM_PROMPT = """\
Tu es un assistant d'analyse IRM cérébrale BraTS (tumeurs gliomes).
Règles strictes :
1. Tu ne cites QUE des nombres présents dans les faits JSON fournis par les outils.
2. Tu ne produis AUCUN diagnostic : tu formules des observations quantitatives.
3. Tu mentionnes systématiquement : « Aide à la décision — pas un diagnostic ».
4. Si une métrique est absente, dis « non disponible », ne l'invente pas.
5. Style : radiologue, phrases courtes, français clinique neutre.
"""

REPORT_TEMPLATE = """\
## Rapport d'analyse — {case_id}
**Workflow exécuté :** {workflow} · **Backend modèle :** {backend}

### 1. Volumes (cm³)
| Région | Volume |
|---|---|
| WT (tumeur entière) | {wt:.3f} |
| TC (cœur tumoral) | {tc:.3f} |
| ET (rehaussement) | {et:.3f} |
| Œdème | {edema:.3f} |
| Nécrose | {necrosis:.3f} |

### 2. Ratios clés
- ET/WT = {r_et_wt:.4f} · ET/TC = {r_et_tc:.4f}
- TC/WT = {r_tc_wt:.4f} · Œdème/WT = {r_edema_wt:.4f}

### 3. Forme & division
- Sphéricité WT = {sphericity:.4f} (1 = sphère parfaite) · Composantes connexes = {n_components}

### 4. Risque
**Niveau : {risk_level}** (confiance {risk_conf:.2f})
Critères : {risk_criteria}

{dice_section}
{rag_section}
---
*Aide à la décision — pas un diagnostic. Seuils heuristiques (Q33/Q66) — valider avec un clinicien.*
"""

COMPARISON_SECTION = """\
### 5. Comparaison — {second_id}
- WT : {wt1:.3f} vs {wt2:.3f} cm³ · ET : {et1:.3f} vs {et2:.3f} cm³
- Risque : {risk1} vs {risk2}
"""

DICE_SECTION = """\
### {dice_title}
- Dice WT = {dice_wt:.4f} · Dice TC = {dice_tc:.4f} · Dice ET = {dice_et:.4f} · Moyenne = {mean:.4f}
"""


def build_report(ctx: dict) -> str:
    """Rapport déterministe (fallback toujours disponible, aucun LLM requis)."""
    b = ctx.get("biomarkers")
    r = ctx.get("risk")
    if b is None or r is None:
        return "## Rapport indisponible\nÉtapes cœur (segmentation/biomarqueurs/risque) en échec — voir steps."
    dice_section = ""
    if ctx.get("dice") is not None:
        d = ctx["dice"]
        dice_section = DICE_SECTION.format(
            dice_title="5. Évaluation (vs ground truth)",
            dice_wt=d.dice_wt, dice_tc=d.dice_tc, dice_et=d.dice_et, mean=d.mean_dice,
        )
    rag_section = ""
    if ctx.get("rag_refs"):
        rag_section = "### Références (RAG)\n" + "\n".join(f"- {ref}" for ref in ctx["rag_refs"]) + "\n"
    if ctx.get("comparison"):
        c = ctx["comparison"]
        rag_section = COMPARISON_SECTION.format(**c) + rag_section
    return REPORT_TEMPLATE.format(
        case_id=ctx.get("case_id", "?"),
        workflow=ctx.get("workflow", "?"),
        backend=ctx.get("backend", "?"),
        wt=b.volume_wt_cm3, tc=b.volume_tc_cm3, et=b.volume_et_cm3,
        edema=b.volume_edema_cm3, necrosis=b.volume_necrosis_cm3,
        r_et_wt=b.ratio_et_wt, r_et_tc=b.ratio_et_tc,
        r_tc_wt=b.ratio_tc_wt, r_edema_wt=b.ratio_edema_wt,
        sphericity=b.sphericity_wt, n_components=b.n_components,
        risk_level=r.level.value, risk_conf=r.confidence,
        risk_criteria="; ".join(r.criteria) or "aucun critère déclenché",
        dice_section=dice_section, rag_section=rag_section,
    )


def llm_explain(user_text: str, facts: dict, skill_text: str = "") -> str:
    """Fallback LLM optionnel (OpenAI-compatible). Lève si non configuré."""
    import json
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY absent — utilisation du rapport déterministe")
    from openai import OpenAI  # import retardé, optionnel

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    user = (
        f"Question : {user_text}\n\n"
        f"Faits des outils (seule source de chiffres) :\n{json.dumps(facts, ensure_ascii=False, default=str)}\n"
        + (f"\nProcédure à suivre (skill) :\n{skill_text}" if skill_text else "")
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""
