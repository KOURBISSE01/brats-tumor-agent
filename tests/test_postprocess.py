"""Post-traitement : hiérarchie des probabilités, labels, composantes connexes."""
import numpy as np

from tools.postprocess import (
    count_components,
    enforce_hierarchy,
    enforce_probability_hierarchy,
    region_mask,
)


def test_probability_hierarchy_min_cumule():
    p_wt = np.full((4, 4, 4), 0.9, dtype=np.float32)
    p_tc = np.full((4, 4, 4), 0.7, dtype=np.float32)
    p_et = np.full((4, 4, 4), 0.8, dtype=np.float32)  # ET > TC : violation
    w, t, e = enforce_probability_hierarchy(p_wt, p_tc, p_et)
    assert np.all(e <= t) and np.all(t <= w)
    assert np.all(e == 0.7)   # min(0.8, 0.7)


def test_labels_hierarchy_wt_sup_tc_sup_et():
    lab = np.zeros((10, 10, 10), dtype=np.uint8)
    lab[2:8, 2:8, 2:8] = 2
    lab[3:7, 3:7, 3:7] = 1
    lab[4:6, 4:6, 4:6] = 3
    out = enforce_hierarchy(lab)
    assert region_mask(out, "wt").sum() >= region_mask(out, "tc").sum()
    assert region_mask(out, "tc").sum() >= region_mask(out, "et").sum()


def test_invalid_labels_zeroed():
    lab = np.full((6, 6, 6), 7, dtype=np.uint8)  # étiquette inconnue
    out = enforce_hierarchy(lab)
    assert out.max() == 0


def test_count_components_regions():
    lab = np.zeros((30, 30, 30), dtype=np.uint8)
    lab[2:6, 2:6, 2:6] = 2       # blob WT n°1 (œdème)
    lab[20:26, 20:26, 20:26] = 1  # blob WT n°2 (nécrose = TC)
    lab[21:24, 21:24, 21:24] = 3  # ET dans le blob 2 → même composante WT

    n_wt, _ = count_components(lab, "wt")
    n_tc, _ = count_components(lab, "tc")   # blobs: nécrose + ET reliés → 1
    n_et, _ = count_components(lab, "et")
    assert n_wt == 2
    assert n_tc == 1
    assert n_et == 1


def test_min_volume_filter():
    lab = np.zeros((40, 40, 40), dtype=np.uint8)
    lab[2:12, 2:12, 2:12] = 2        # gros : 1000 mm³
    lab[30:32, 30:32, 30:32] = 2     # petit : 8 mm³
    n_all, _ = count_components(lab, "wt", min_volume_mm3=0.0)
    n_big, _ = count_components(lab, "wt", min_volume_mm3=100.0)
    assert n_all == 2
    assert n_big == 1


def test_empty_mask_zero_components():
    n, cc = count_components(np.zeros((8, 8, 8), dtype=np.uint8), "wt")
    assert n == 0
    assert cc.sum() == 0
