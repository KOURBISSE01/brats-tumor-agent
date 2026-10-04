"""Les 11 biomarqueurs BraTS — 5 volumes + 4 ratios + sphéricité + n composantes.

Formules :
- Volume  = n_voxels × spacing_z × spacing_y × spacing_x  (mm³ → cm³ /1000)
- Ratios  = rapport des volumes (0.0 si dénominateur nul)
- Sphéricité ψ = π^(1/3) × (6V)^(2/3) / A — voir tools/surface.py
  (lissage + marching tets ; sphère ≈ 0.96, cube ≈ 0.89).
"""
from __future__ import annotations

import numpy as np

from agent.schemas import Biomarkers
from tools.postprocess import (
    LABEL_ED,
    LABEL_ET,
    LABEL_NCR,
    count_components,
    region_mask,
)
from tools.surface import sphericity as _sphericity_mask


def _volume_cm3(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    voxel_mm3 = float(np.prod(spacing))
    return float(mask.sum()) * voxel_mm3 / 1000.0


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def sphericity(labels: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """ψ de la région WT — délègue à tools.surface (lissage gaussien + marching tets).

    Voir tools/surface.py : sphère ≈ 0.96, cube ≈ 0.89, ordre sphère > cube garanti
    (le comptage de faces brut, lui, inversait l'ordre : sphère ≈ 0.67 < cube).
    """
    mask = region_mask(labels, "wt")
    if not mask.any():
        return 0.0
    return _sphericity_mask(mask, spacing)


def compute_biomarkers(
    labels: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_component_volume_mm3: float = 0.0,
) -> Biomarkers:
    """Calcule les 11 biomarqueurs à partir d'une carte de labels {0,1,2,3}."""
    labels = np.asarray(labels)
    vol_wt = _volume_cm3(labels > 0, spacing)
    vol_tc = _volume_cm3(np.isin(labels, (LABEL_NCR, LABEL_ET)), spacing)
    vol_et = _volume_cm3(labels == LABEL_ET, spacing)
    vol_ed = _volume_cm3(labels == LABEL_ED, spacing)
    vol_nec = _volume_cm3(labels == LABEL_NCR, spacing)

    n_components, _ = count_components(
        labels, region="wt",
        min_volume_mm3=min_component_volume_mm3, spacing=spacing,
    )

    return Biomarkers(
        volume_wt_cm3=round(vol_wt, 6),
        volume_tc_cm3=round(vol_tc, 6),
        volume_et_cm3=round(vol_et, 6),
        volume_edema_cm3=round(vol_ed, 6),
        volume_necrosis_cm3=round(vol_nec, 6),
        ratio_et_wt=round(_ratio(vol_et, vol_wt), 6),
        ratio_et_tc=round(_ratio(vol_et, vol_tc), 6),
        ratio_tc_wt=round(_ratio(vol_tc, vol_wt), 6),
        ratio_edema_wt=round(_ratio(vol_ed, vol_wt), 6),
        sphericity_wt=round(sphericity(labels, spacing), 6),
        n_components=n_components,
    )
