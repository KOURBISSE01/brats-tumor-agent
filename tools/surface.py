"""Aire de surface discrète et sphéricité — pipeline : lissage gaussien + marching
tetrahedra vectorisé (numpy pur, aucun scipy/skimage).

Méthode :
1. le masque binaire est lissé (gaussien séparable, σ = 0.6 mm par défaut) ce qui
   supprime les marches d'escalier du champ ;
2. l'iso-surface {0.5} est extraite par marching tetrahedra (6 tétraèdres cohérents
   par cellule 2×2×2, interpolation linéaire du champ continu) ;
3. ψ = π^(1/3) (6V)^(2/3) / A, avec V = somme du champ lissé × volume voxel.

Pourquoi pas le simple comptage de faces (A6) ? A6 et la variation totale L1 qui
lui est équivalent surestiment les surfaces courbes d'un facteur ~1.5 selon
l'orientation (biais des marches d'escalier) : une sphère ressortait « moins
sphérique » qu'un cube. Avec lissage + marching tets :

    cube 10³      ψ ≈ 0.89   (cible géométrique 0.806, arrondi des arêtes par σ)
    sphère r=16   ψ ≈ 0.96   (cible 1.0)

L'ordre sphère > cube est respecté, et ψ reste comparable d'un patient à l'autre
(même biais systématique). ψ peut légèrement dépasser 1 sur de petites formes
(discretisation) — documenté dans le README.

Références : Lorensen & Cline 1987 (marching cubes), décomposition tétraédrique
standard 6 tets/cube le long de la diagonale principale.
"""
from __future__ import annotations

import numpy as np

# 6 tétraèdres le long de la diagonale 0→7 (décomposition cohérente du cube)
_TETS = np.array(
    [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ],
    dtype=np.int64,
)

# coins du cube : bit0 = x, bit1 = y, bit2 = z
_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int64,
)

DEFAULT_SIGMA_MM = 0.6
Z_CHUNK = 48  # tranches de cellules Z pour borner la mémoire (volumes cliniques 240³)


def _gauss1d(sigma: float) -> np.ndarray:
    r = max(int(np.ceil(3.0 * sigma)), 1)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-x * x / (2.0 * sigma * sigma))
    return k / k.sum()


def _smooth_field(field: np.ndarray, sigma_vox: tuple[float, float, float]) -> np.ndarray:
    """Gaussien séparable par convolution directe (numpy pur), float32."""
    out = field.astype(np.float32, copy=True)
    for axis in range(3):
        s = float(sigma_vox[axis])
        if s <= 0:
            continue
        k = _gauss1d(s).astype(np.float32)
        r = len(k) // 2
        pad_shape = list(out.shape)
        pad_shape[axis] += 2 * r
        padded = np.zeros(pad_shape, dtype=np.float32)
        src = [slice(None)] * 3
        src[axis] = slice(r, r + out.shape[axis])
        padded[tuple(src)] = out
        acc = np.zeros_like(out)
        for i, kv in enumerate(k):
            if kv == 0:
                continue
            s2 = [slice(None)] * 3
            s2[axis] = slice(i, i + out.shape[axis])
            acc += kv * padded[tuple(s2)]
        out = acc
    return out


def _tri_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)


def _marching_tets_area(field: np.ndarray, spacing: np.ndarray) -> float:
    """Aire de l'iso-surface {0.5} d'un champ continu (déjà lissé), par tranches Z."""
    m = np.pad(field, 1, mode="constant", constant_values=0.0)
    nz, ny, nx = (s - 1 for s in m.shape)
    if nz <= 0 or ny <= 0 or nx <= 0:
        return 0.0

    total = 0.0
    for z0 in range(0, nz, Z_CHUNK):
        z1 = min(z0 + Z_CHUNK, nz)
        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1), np.arange(ny), np.arange(nx), indexing="ij"
        )
        base = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)

        for tet in _TETS:
            f = np.empty((base.shape[0], 4), dtype=np.float32)
            pos = np.empty((base.shape[0], 4, 3), dtype=np.float32)
            for ci, corner in enumerate(tet):
                idx = base + _CORNERS[corner]
                f[:, ci] = m[idx[:, 0], idx[:, 1], idx[:, 2]]
                pos[:, ci] = idx * spacing

            above = f > np.float32(0.5)
            mixed = above.any(axis=1) & (~above).any(axis=1)
            if not mixed.any():
                continue
            f, pos, above = f[mixed], pos[mixed], above[mixed]
            acc = np.zeros(f.shape[0], dtype=np.float64)
            n_in = above.sum(axis=1)

            # --- 1 ou 3 coins dessus : triangle des 3 coupures
            for n_case in (1, 3):
                sc = n_in == n_case
                if not sc.any():
                    continue
                fc, pc = f[sc], pos[sc]
                for i in range(4):
                    cond = (fc[:, i] > 0.5) if n_case == 1 else (fc[:, i] <= 0.5)
                    if not cond.any():
                        continue
                    others = [j for j in range(4) if j != i]
                    pts = []
                    for o in others:
                        denom = fc[cond, o] - fc[cond, i]
                        t = np.where(np.abs(denom) > 1e-30, (0.5 - fc[cond, i]) / np.where(np.abs(denom) > 1e-30, denom, 1.0), 0.5)
                        pts.append(pc[cond, i] + t[:, None] * (pc[cond, o] - pc[cond, i]))
                    idxs = np.flatnonzero(sc)[cond]
                    acc[idxs] += _tri_area(pts[0], pts[1], pts[2])

            # --- 2 dessus / 2 dessous : quad → 2 triangles (une seule paire "dessus")
            sc = n_in == 2
            if sc.any():
                fc, pc = f[sc], pos[sc]
                for i in range(4):
                    for j in range(i + 1, 4):
                        pair = (fc[:, i] > 0.5) & (fc[:, j] > 0.5)
                        if not pair.any():
                            continue
                        k, l = [x for x in range(4) if x not in (i, j)]

                        def cut(a: int, b: int, fc=fc, pc=pc, pair=pair) -> np.ndarray:
                            denom = fc[pair, b] - fc[pair, a]
                            t = np.where(np.abs(denom) > 1e-30, (0.5 - fc[pair, a]) / np.where(np.abs(denom) > 1e-30, denom, 1.0), 0.5)
                            return pc[pair, a] + t[:, None] * (pc[pair, b] - pc[pair, a])

                        mac, mad = cut(i, k), cut(i, l)
                        mbc, mbd = cut(j, k), cut(j, l)
                        idxs = np.flatnonzero(sc)[pair]
                        acc[idxs] += _tri_area(mac, mad, mbd) + _tri_area(mac, mbd, mbc)

            total += float(acc.sum())

    return total


def surface_area(
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma_mm: float = DEFAULT_SIGMA_MM,
) -> float:
    """Aire (mm²) de la surface WT : masque booléen → lissage → marching tets."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0
    sp = np.asarray(spacing, dtype=np.float64)
    sigma_vox = tuple(sigma_mm / max(s, 1e-9) for s in sp)
    field = _smooth_field(mask.astype(np.float32), sigma_vox)  # type: ignore[arg-type]
    return _marching_tets_area(field, sp.astype(np.float32))


def sphericity(
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma_mm: float = DEFAULT_SIGMA_MM,
) -> float:
    """ψ = π^(1/3) (6V)^(2/3) / A — sphère ≈ 0.96, cube ≈ 0.89 (voir module docstring)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0
    sp = np.asarray(spacing, dtype=np.float64)
    voxel_mm3 = float(np.prod(sp))
    sigma_vox = tuple(sigma_mm / max(s, 1e-9) for s in sp)
    field = _smooth_field(mask.astype(np.float32), sigma_vox)  # type: ignore[arg-type]
    volume_mm3 = float(field.sum()) * voxel_mm3  # masse conservée par le gaussien
    area = _marching_tets_area(field, sp.astype(np.float32))
    if volume_mm3 <= 0 or area <= 0:
        return 0.0
    return float(np.pi ** (1.0 / 3.0) * (6.0 * volume_mm3) ** (2.0 / 3.0) / area)
