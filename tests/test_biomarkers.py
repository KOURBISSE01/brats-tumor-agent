"""Biomarqueurs : volumes exacts, ratios, sphéricité (ordre sphère > cube)."""
import numpy as np
import pytest

from tools.biomarkers import compute_biomarkers, sphericity
from tools.surface import sphericity as surface_sphericity


def _cube_labels(size=10, label=3, offset=10, shape=30):
    lab = np.zeros((shape, shape, shape), dtype=np.uint8)
    lab[offset:offset + size, offset:offset + size, offset:offset + size] = label
    return lab


def test_volumes_cube_exact():
    # cube 10³ étiquette 3 (ET) en 1mm iso = 1000 mm³ = 1 cm³
    lab = _cube_labels()
    b = compute_biomarkers(lab)
    assert b.volume_et_cm3 == pytest.approx(1.0, rel=1e-6)
    assert b.volume_tc_cm3 == pytest.approx(1.0, rel=1e-6)   # ET ⊂ TC
    assert b.volume_wt_cm3 == pytest.approx(1.0, rel=1e-6)   # ET ⊂ WT
    assert b.volume_edema_cm3 == 0.0
    assert b.volume_necrosis_cm3 == 0.0


def test_regions_edema_necrosis():
    lab = np.zeros((30, 30, 30), dtype=np.uint8)
    lab[10:20, 10:20, 10:20] = 2   # bloc1000 voxels
    lab[12:16, 12:16, 12:16] = 1   # 64 voxels repassés en nécrose
    b = compute_biomarkers(lab)
    assert b.volume_edema_cm3 == pytest.approx(0.936, rel=1e-6)  # 1000-64
    assert b.volume_necrosis_cm3 == pytest.approx(0.064, rel=1e-6)
    assert b.volume_wt_cm3 == pytest.approx(1.0, rel=1e-6)
    assert b.volume_tc_cm3 == pytest.approx(0.064, rel=1e-6)


def test_ratios_and_empty_guards():
    lab = np.zeros((30, 30, 30), dtype=np.uint8)
    lab[10:20, 10:20, 10:20] = 2
    lab[10:15, 10:15, 10:15] = 3  # ET = 125 voxels dans WT = 1000
    b = compute_biomarkers(lab)
    assert b.ratio_et_wt == pytest.approx(0.125, rel=1e-6)
    assert b.ratio_tc_wt == pytest.approx(0.125, rel=1e-6)
    assert b.ratio_et_tc == pytest.approx(1.0, rel=1e-6)
    assert b.ratio_edema_wt == pytest.approx(0.875, rel=1e-6)

    empty = compute_biomarkers(np.zeros((8, 8, 8), dtype=np.uint8))
    assert empty.volume_wt_cm3 == 0.0
    assert empty.ratio_et_wt == 0.0   # division par zéro → 0.0
    assert empty.sphericity_wt == 0.0
    assert empty.n_components == 0


def test_n_components_two_blobs():
    lab = np.zeros((40, 40, 40), dtype=np.uint8)
    lab[2:8, 2:8, 2:8] = 2
    lab[30:36, 30:36, 30:36] = 2
    b = compute_biomarkers(lab)
    assert b.n_components == 2


def test_sphericity_order_sphere_over_cube():
    """Ordre géométrique : la sphère doit être plus sphérique que le cube."""
    cube = _cube_labels(size=16, label=2, offset=5, shape=26)
    n, r = 44, 16
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    c = n // 2
    sphere = np.zeros((n, n, n), dtype=np.uint8)
    sphere[(zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 <= r * r] = 2

    psi_cube = sphericity(cube)
    psi_sphere = sphericity(sphere)
    assert 0.80 < psi_cube < 0.95       # cube ≈ 0.89 (arrondi par σ)
    assert 0.90 < psi_sphere <= 1.10    # sphère ≈ 0.96
    assert psi_sphere > psi_cube        # l'ordre classique A6 inversait


def test_sphericity_anisotropic_spacing():
    """Spacing non unitaire : même masque, ψ proche (invariance d'échelle)."""
    lab = _cube_labels(size=10, label=2)
    psi_iso = surface_sphericity(lab > 0, (1.0, 1.0, 1.0))
    psi_aniso = surface_sphericity(lab > 0, (2.0, 0.5, 1.0))
    assert psi_aniso == pytest.approx(psi_iso, rel=0.15)
