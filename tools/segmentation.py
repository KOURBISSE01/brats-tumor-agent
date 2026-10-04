"""Segmentation 3D BraTS.

- **Backend réel** : modèle TorchScript (DynUNet3D exporté) chargé si
  `torch` + poids disponibles → inférence sliding-window 128³, overlap 0.5,
  importance gaussienne (style nnU-Net) pour fusionner les patches.
- **Backend mock** : fantôme analytique déterministe (sphères concentriques
  Œdème ⊃ TC ⊃ ET au centre du volume) — tests, CI, démo sans GPU ni poids.

Les deux retournent une carte de labels {0,1,2,3} avec hiérarchie WT ⊇ TC ⊇ ET.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from tools.postprocess import (
    enforce_hierarchy,
    enforce_probability_hierarchy,
    region_mask,
)

PATCH = 128
OVERLAP = 0.5


# ---------------------------------------------------------------------------
# Chargement modèle
# ---------------------------------------------------------------------------
def _load_model(model_path: str | None):
    """Tente de charger un modèle TorchScript. Retourne (model|None, erreur|None)."""
    model_path = model_path or os.getenv("BRATS_MODEL_PATH")
    if not model_path:
        return None, "BRATS_MODEL_PATH non défini"
    p = Path(model_path)
    if not p.is_file():
        return None, f"poids introuvables : {model_path}"
    try:
        import torch  # type: ignore
    except ImportError:
        return None, "torch non installé"
    try:
        model = torch.jit.load(str(p), map_location="cpu")
        model.eval()
        return model, None
    except Exception as exc:  # format de poids non TorchScript
        return None, f"chargement modèle échoué : {exc}"


# ---------------------------------------------------------------------------
# Sliding window avec importance gaussienne
# ---------------------------------------------------------------------------
def gaussian_importance_map(patch: tuple[int, int, int]) -> np.ndarray:
    """Carte d'importance gaussienne centrée (évite les artefacts de couture)."""
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, patch[0]),
        np.linspace(-1.0, 1.0, patch[1]),
        np.linspace(-1.0, 1.0, patch[2]),
        indexing="ij",
    )
    sigma = 0.125  # nnU-Net : gaussienne étroite, bords peu pondérés
    w = np.exp(-((zz**2 + yy**2 + xx**2) / (2 * sigma**2)))
    return (w / w.max()).astype(np.float32)


def _steps(size: int, patch: int, overlap: float) -> list[int]:
    stride = max(int(patch * (1.0 - overlap)), 1)
    starts = list(range(0, max(size - patch, 0) + 1, stride))
    if not starts or starts[-1] + patch < size:
        starts.append(max(size - patch, 0))
    return sorted(set(starts))


def sliding_window_predict(
    volume: np.ndarray,
    predict_fn,
    patch: tuple[int, int, int] = (PATCH, PATCH, PATCH),
    overlap: float = OVERLAP,
) -> np.ndarray:
    """volume (C,D,H,W) → probas (3,D,H,W). predict_fn(batch(1,C,d,h,w)) → (1,3,d,h,w)."""
    volume = np.asarray(volume, dtype=np.float32)
    c, d, h, w = volume.shape
    pd, ph, pw = patch
    orig = (d, h, w)
    if d < pd or h < ph or w < pw:
        pad = ((0, 0), (0, max(pd - d, 0)), (0, max(ph - h, 0)), (0, max(pw - w, 0)))
        volume = np.pad(volume, pad, mode="edge")
        _, d, h, w = volume.shape

    importance = gaussian_importance_map((pd, ph, pw))
    acc = np.zeros((3, d, h, w), dtype=np.float32)
    weight = np.zeros((1, d, h, w), dtype=np.float32)

    for z in _steps(d, pd, overlap):
        for y in _steps(h, ph, overlap):
            for x in _steps(w, pw, overlap):
                patch_data = volume[:, z:z + pd, y:y + ph, x:x + pw]
                pred = predict_fn(patch_data[np.newaxis])          # (1,3,pd,ph,pw)
                pred = np.asarray(pred, dtype=np.float32)[0]
                acc[:, z:z + pd, y:y + ph, x:x + pw] += pred * importance
                weight[0, z:z + pd, y:y + ph, x:x + pw] += importance

    # division sûre : un plancher à 1e-8 écrase les importances gaussiennes
    # réelles (coins ≈ 5e-42) et fausse les bords — diviser seulement si w > 0
    out = np.zeros_like(acc)
    covered = weight[0] > 0
    out[:, covered] = acc[:, covered] / weight[0][covered]
    return out[:, :orig[0], :orig[1], :orig[2]]


# ---------------------------------------------------------------------------
# Fantôme mock (déterministe, sans dépendance)
# ---------------------------------------------------------------------------
def _mock_probabilities(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sphères gaussiennes concentriques : ET ⊂ TC ⊂ WT (hiérarchie garantie)."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, d),
        np.linspace(-1, 1, h),
        np.linspace(-1, 1, w),
        indexing="ij",
    )
    # centre légèrement décalé pour rester visible dans un slice central
    r2 = ((zz - 0.05) ** 2 + (yy - 0.05) ** 2 + xx**2)

    def blob(sigma: float) -> np.ndarray:
        return np.exp(-r2 / (2 * sigma**2)).astype(np.float32)

    p_wt = blob(0.45)
    p_tc = blob(0.30)
    p_et = blob(0.18)
    return p_wt, p_tc, p_et


def probs_to_labels(p_wt: np.ndarray, p_tc: np.ndarray, p_et: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Seuillage hiérarchique : ET > TC > Œdème(label 2) > fond."""
    labels = np.zeros(p_wt.shape, dtype=np.uint8)
    labels[p_wt > thr] = 2          # œdème = WT \ TC
    labels[p_tc > thr] = 1          # NCR/nécrose = TC \ ET
    labels[p_et > thr] = 3          # enhancing
    return labels


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------
def segment_volume(
    volume: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    model_path: str | None = None,
    patch: int = PATCH,
    overlap: float = OVERLAP,
) -> tuple[np.ndarray, dict]:
    """Retourne (labels uint8 {0,1,2,3}, meta{backend, warnings})."""
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim == 3:
        volume = volume[np.newaxis]
    if volume.ndim != 4:
        raise ValueError(f"volume attendu (C,D,H,W) ou (D,H,W), obtenu {volume.shape}")

    warnings: list[str] = []
    model, err = _load_model(model_path)

    if model is None:
        warnings.append(f"Backend MOCK actif ({err}) — fantôme synthétique, pas une vraie prédiction.")
        p_wt, p_tc, p_et = _mock_probabilities(volume.shape[1:])
        backend = "mock"
    else:
        def _predict(batch: np.ndarray) -> np.ndarray:
            import torch  # type: ignore
            with torch.no_grad():
                out = model(torch.from_numpy(batch))
            return out.numpy() if hasattr(out, "numpy") else np.asarray(out)

        probs = sliding_window_predict(volume, _predict, patch=(patch, patch, patch), overlap=overlap)
        p_wt, p_tc, p_et = probs[0], probs[1], probs[2]
        backend = "dynunet-torchscript"

    p_wt, p_tc, p_et = enforce_probability_hierarchy(p_wt, p_tc, p_et)
    labels = probs_to_labels(p_wt, p_tc, p_et)
    labels = enforce_hierarchy(labels)

    # garde-fou : aucun voxel hors WT possible par construction, mais on vérifie
    assert region_mask(labels, "wt").sum() >= region_mask(labels, "tc").sum() >= region_mask(labels, "et").sum()

    return labels, {"backend": backend, "warnings": warnings, "spacing": tuple(spacing)}
