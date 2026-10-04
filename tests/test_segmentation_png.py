"""Segmentation (mock, sliding window, hiérarchie) + encodeur PNG."""
import numpy as np

from tools.png_util import encode_png, normalize_u8, render_slice_overlay, slice_png
from tools.segmentation import (
    gaussian_importance_map,
    probs_to_labels,
    segment_volume,
    sliding_window_predict,
)


def test_mock_backend_deterministic_and_hierarchical():
    vol = np.zeros((4, 32, 32, 32), dtype=np.float32)
    lab1, meta1 = segment_volume(vol)
    lab2, meta2 = segment_volume(vol)
    assert meta1["backend"] == "mock" and meta2["backend"] == "mock"
    assert any("MOCK" in w for w in meta1["warnings"])
    np.testing.assert_array_equal(lab1, lab2)          # déterministe
    assert set(np.unique(lab1)) <= {0, 1, 2, 3}
    assert (lab1 > 0).sum() >= (np.isin(lab1, (1, 3))).sum() >= (lab1 == 3).sum()


def test_probs_to_labels_thresholds():
    p_wt = np.full((4, 4, 4), 0.9, np.float32)
    p_tc = np.full((4, 4, 4), 0.6, np.float32)
    p_et = np.full((4, 4, 4), 0.4, np.float32)   # sous le seuil
    lab = probs_to_labels(p_wt, p_tc, p_et, thr=0.5)
    assert (lab == 1).all()      # TC sans ET → nécrose


def test_gaussian_importance_normalized():
    w = gaussian_importance_map((16, 16, 16))
    assert w.shape == (16, 16, 16)
    assert w.max() == np.float32(1.0)
    assert w[8, 8, 8] > w[0, 0, 0]     # centre > coin


def test_sliding_window_padding_small_volume():
    """Volume plus petit que le patch → padding edge + crop à la sortie."""
    vol = np.random.default_rng(3).random((1, 20, 20, 20)).astype(np.float32)

    def predict(batch):  # (1,C,d,h,w) → (1,3,d,h,w) constant
        b = batch.shape[0]
        return np.full((b, 3, *batch.shape[2:]), 0.5, dtype=np.float32)

    out = sliding_window_predict(vol, predict, patch=(32, 32, 32), overlap=0.5)
    assert out.shape == (3, 20, 20, 20)
    assert np.allclose(out, 0.5, atol=1e-5)


def test_segment_rejects_bad_shape():
    import pytest

    with pytest.raises(ValueError, match="attendu"):
        segment_volume(np.zeros((4, 4), dtype=np.float32))


# -- PNG -------------------------------------------------------------------
def test_encode_png_gray_magic():
    data = encode_png(np.arange(64, dtype=np.uint8).reshape(8, 8))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in data and b"IDAT" in data and b"IEND" in data


def test_encode_png_rgb():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    data = encode_png(rgb)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # color type 2 (RGB) à l'offset 25 du chunk IHDR (8 sig + 4 len + 4 tag + 16)
    assert data[25] == 2


def test_overlay_paints_tumor_red():
    gray = np.full((16, 16), 100.0)
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[4:8, 4:8] = 3
    rgb = render_slice_overlay(gray, labels)
    assert rgb.shape == (16, 16, 3)
    r_tumor = rgb[5, 5]
    r_bg = rgb[1, 1]
    assert r_tumor[0] > r_bg[0] + 40   # rouge plus intense sur la tumeur
    assert r_tumor[0] > r_tumor[1]     # composante rouge dominante


def test_normalize_u8_constant_array():
    out = normalize_u8(np.full((4, 4), 7.0))
    assert out.dtype == np.uint8
    assert out.max() == 0              # plage nulle → noir, pas de division par 0


def test_slice_png_both_modes():
    gray = np.random.default_rng(0).random((16, 16))
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[8:12, 8:12] = 1
    assert slice_png(gray)[:8] == b"\x89PNG\r\n\x1a\n"
    assert slice_png(gray, labels)[:8] == b"\x89PNG\r\n\x1a\n"
