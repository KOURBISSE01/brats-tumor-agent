"""I/O volumes — .npy natif, .nii/.nii.gz via nibabel (optionnel, import gardé)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_volume(path: str | Path, channel_first: bool = True) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Charge un volume. Retourne (array, spacing_mm).

    - .npy  : shape (D,H,W) ou (C,D,H,W) ; spacing par défaut (1,1,1) mm
    - .nii/.nii.gz : nibabel requis ; spacing déduit de l'affine
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"volume introuvable : {path}")
    suffix = "".join(path.suffixes).lower()

    if suffix.endswith(".npy"):
        arr = np.load(path)
        return _as_channel_first(arr, channel_first), (1.0, 1.0, 1.0)

    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError(
                "nibabel non installé — `pip install nibabel` pour lire les NIfTI, "
                "ou convertir en .npy"
            ) from exc
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
        return _as_channel_first(data, channel_first), spacing  # type: ignore[return-value]

    raise ValueError(f"format non supporté : {suffix} (attendu .npy / .nii / .nii.gz)")


def _as_channel_first(arr: np.ndarray, channel_first: bool) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[np.newaxis] if channel_first else arr
    if arr.ndim == 4 and channel_first:
        # (D,H,W,C) -> (C,D,H,W) si la dernière dim est le canal (4 modalités)
        if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
            return np.moveaxis(arr, -1, 0)
        return arr
    raise ValueError(f"shape inattendu : {arr.shape} (3D ou 4D attendu)")


def save_labels(labels: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels.astype(np.uint8, copy=False))
    return path


def load_labels(path: str | Path) -> np.ndarray:
    arr, _ = load_volume(path, channel_first=False)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.int32, copy=False)
