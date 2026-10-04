"""Post-traitement : hiérarchie WT ⊇ TC ⊇ ET + composantes connexes 3D."""
from __future__ import annotations

import numpy as np

LABEL_NCR = 1   # nécrose / tumeur non-contrastée  → dans TC
LABEL_ED = 2    # œdème                           → WT seulement
LABEL_ET = 3    # rehaussement                     → dans TC et WT


def enforce_hierarchy(labels: np.ndarray) -> np.ndarray:
    """Garantit WT ⊇ TC ⊇ ET sur une carte de labels : un voxel ET est TC,
    un voxel TC (1 ou 3) est WT. Un voxel 'flottant' hors {0,1,2,3} est mis à 0."""
    out = np.zeros_like(labels)
    valid = np.isin(labels, (LABEL_NCR, LABEL_ED, LABEL_ET))
    out[valid] = labels[valid]
    # (les 3 labels impliquent déjà WT ; la contrainte sert pour les maps venant
    # d'un modèle dont les canaux ont été fusionnés séparément)
    return out


def enforce_probability_hierarchy(p_wt: np.ndarray, p_tc: np.ndarray, p_et: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """p_et ≤ p_tc ≤ p_wt pointwise (min cumulé, style nnU-Net post-hoc)."""
    p_tc = np.minimum(p_tc, p_wt)
    p_et = np.minimum(p_et, p_tc)
    return p_wt, p_tc, p_et


def _connected_components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """Composantes connexes 6-voisins. scipy si dispo, sinon flood-fill pur Python."""
    mask = mask.astype(bool)
    if not mask.any():
        return 0, np.zeros(mask.shape, dtype=np.int32)
    try:
        from scipy import ndimage  # type: ignore

        cc, n = ndimage.label(mask, structure=_structure())
        return int(n), cc.astype(np.int32)
    except ImportError:
        return _fallback_cc(mask)


def _structure() -> np.ndarray:
    st = np.zeros((3, 3, 3), dtype=int)
    st[1, 1, 1] = 1
    for ax in range(3):
        sl = [1, 1, 1]
        sl[ax] = 0
        st[tuple(sl)] = 1
        sl[ax] = 2
        st[tuple(sl)] = 1
    return st


def _fallback_cc(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """Flood-fill itératif (BFS) — lent mais sans dépendance, correct pour tests/CI."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    # indexation par face-avant pour accélérer le balayage
    idx = np.flatnonzero(mask.ravel())
    n = 0
    flat = mask.ravel()
    flab = labels.ravel()
    neighbors_of = _build_neighbors(mask.shape)
    for start in idx:
        if flab[start] != 0:
            continue
        n += 1
        stack = [int(start)]
        flab[start] = n
        while stack:
            v = stack.pop()
            for u in neighbors_of[v]:
                if flat[u] and flab[u] == 0:
                    flab[u] = n
                    stack.append(u)
    return n, labels


def _build_neighbors(shape):
    """Liste de voisins valides (6-voisins, hors bornes = ignoré) par index plat."""
    d, h, w = shape
    total = d * h * w
    nbrs = [[] for _ in range(total)]
    for z in range(d):
        for y in range(h):
            for x in range(w):
                i = (z * h + y) * w + x
                if z > 0:
                    nbrs[i].append(i - h * w)
                if z < d - 1:
                    nbrs[i].append(i + h * w)
                if y > 0:
                    nbrs[i].append(i - w)
                if y < h - 1:
                    nbrs[i].append(i + w)
                if x > 0:
                    nbrs[i].append(i - 1)
                if x < w - 1:
                    nbrs[i].append(i + 1)
    return nbrs


def count_components(
    labels: np.ndarray,
    region: str = "wt",
    min_volume_mm3: float = 0.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[int, np.ndarray]:
    """Composantes connexes d'une région (wt/tc/et), filtre volume min (mm³).

    Retourne (n, cc_labels) où cc_labels ne contient que les composantes retenues.
    """
    mask = region_mask(labels, region)
    voxel_mm3 = float(np.prod(spacing))
    n, cc = _connected_components(mask)
    if min_volume_mm3 > 0 and n > 0:
        keep = np.zeros(n + 1, dtype=bool)
        counts = np.bincount(cc.ravel(), minlength=n + 1)
        for i in range(1, n + 1):
            keep[i] = counts[i] * voxel_mm3 >= min_volume_mm3
        cc = np.where(keep[cc], cc, 0)
        n = int(keep.sum())
    return n, cc


def region_mask(labels: np.ndarray, region: str) -> np.ndarray:
    region = region.lower()
    if region == "wt":
        return labels > 0
    if region == "tc":
        return np.isin(labels, (LABEL_NCR, LABEL_ET))
    if region == "et":
        return labels == LABEL_ET
    raise ValueError(f"région inconnue : {region} (attendu wt|tc|et)")
