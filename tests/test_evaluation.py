"""Dice WT/TC/ET — cas parfaits, décalés, vides."""
import numpy as np
import pytest

from tools.evaluation import dice_score, evaluate_dice


def test_perfect_prediction():
    lab = np.zeros((20, 20, 20), dtype=np.uint8)
    lab[5:15, 5:15, 5:15] = 3
    r = evaluate_dice(lab, lab)
    assert r.dice_wt == pytest.approx(1.0)
    assert r.dice_tc == pytest.approx(1.0)
    assert r.dice_et == pytest.approx(1.0)
    assert r.mean_dice == pytest.approx(1.0)


def test_both_empty_is_one():
    empty = np.zeros((10, 10, 10), dtype=np.uint8)
    r = evaluate_dice(empty, empty)
    assert r.dice_wt == 1.0  # convention BraTS
    assert r.mean_dice == 1.0


def test_disjoint_is_zero():
    a = np.zeros((20, 20, 20), dtype=np.uint8)
    b = np.zeros((20, 20, 20), dtype=np.uint8)
    a[0:10, 0:10, 0:10] = 2
    b[10:20, 10:20, 10:20] = 2
    r = evaluate_dice(a, b)
    assert r.dice_wt == pytest.approx(0.0)


def test_half_overlap_wt():
    a = np.zeros((20, 20, 20), dtype=np.uint8)
    b = np.zeros((20, 20, 20), dtype=np.uint8)
    a[0:10, :, :] = 2          # 2000 voxels
    b[5:15, :, :] = 2          # chevauchement 5:10 → 1000 voxels
    d = dice_score(a > 0, b > 0)
    # 2*1000 / (2000+2000) = 0.5
    assert d == pytest.approx(0.5)


def test_et_region_only():
    gt = np.zeros((16, 16, 16), dtype=np.uint8)
    pred = np.zeros((16, 16, 16), dtype=np.uint8)
    gt[4:12, 4:12, 4:12] = 1     # TC = nécrose seulement
    pred[4:12, 4:12, 4:12] = 1
    r = evaluate_dice(pred, gt)
    assert r.dice_tc == pytest.approx(1.0)
    assert r.dice_et == pytest.approx(1.0)   # vides des deux côtés
    assert r.dice_wt == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shapes"):
        evaluate_dice(np.zeros((8, 8, 8)), np.zeros((9, 9, 9)))
