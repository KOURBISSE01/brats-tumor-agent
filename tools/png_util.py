"""Encodeur PNG minimal (graisse ou RGB, 8 bits) — stdlib uniquement (zlib+struct).
Permet de servir des slices IRM depuis l'API sans Pillow/matplotlib."""
from __future__ import annotations

import struct
import zlib

import numpy as np


def normalize_u8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros(a.shape, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


def render_slice_overlay(
    volume_slice: np.ndarray,
    labels_slice: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Slice IRM en niveaux de gris + overlay rouge des labels tumorales → (H,W,3) uint8."""
    base = normalize_u8(volume_slice).astype(np.float64)
    rgb = np.stack([base, base, base], axis=-1)
    mask = np.asarray(labels_slice) > 0
    if mask.any():
        # teintes selon la région : ET rouge vif, TC rouge sombre, Œdème orange
        et = np.asarray(labels_slice) == 3
        tc = np.asarray(labels_slice) == 1
        ed = np.asarray(labels_slice) == 2
        rgb[mask] = rgb[mask] * (1 - alpha)
        rgb[ed] += alpha * np.array([255.0, 160.0, 40.0])
        rgb[tc] += alpha * np.array([255.0, 60.0, 60.0])
        rgb[et] += alpha * np.array([255.0, 30.0, 30.0])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def encode_png(arr: np.ndarray) -> bytes:
    """(H,W) ou (H,W,1) → PNG grayscale · (H,W,3) → PNG RGB. Filtre ligne = 0."""
    a = np.asarray(arr)
    if a.ndim == 2:
        color_type = 0
        a = a[:, :, np.newaxis]
    elif a.ndim == 3 and a.shape[2] == 3:
        color_type = 2
    elif a.ndim == 3 and a.shape[2] == 1:
        color_type = 0
    else:
        raise ValueError(f"shape inattendu : {arr.shape}")
    if a.dtype != np.uint8:
        a = normalize_u8(a)

    h, w = a.shape[0], a.shape[1]
    # lignes : filtre (1 octet) + pixels
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def slice_png(volume_slice: np.ndarray, labels_slice: np.ndarray | None = None) -> bytes:
    if labels_slice is None:
        return encode_png(normalize_u8(volume_slice))
    return encode_png(render_slice_overlay(volume_slice, labels_slice))
