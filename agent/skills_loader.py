"""Chargement des skills (format SKILL.md façon Claude Code) par le planner."""
from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# workflow -> skill principal (chargé dans le contexte de génération)
WORKFLOW_SKILLS = {
    "rapide": "segmentation",
    "standard": "segmentation",
    "complet": "rapport_radiologique",
    "comparaison": "evaluation",
}


def list_skills() -> list[str]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p.parent.name for p in SKILLS_DIR.glob("*/SKILL.md"))


def load_skill(name: str) -> str:
    """Retourne le texte du skill ; chaîne vide si absent (jamais d'erreur bloquante)."""
    path = SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def skill_for_workflow(workflow_value: str) -> tuple[str, str]:
    """(nom, texte) du skill associé au workflow."""
    name = WORKFLOW_SKILLS.get(workflow_value, "segmentation")
    return name, load_skill(name)
