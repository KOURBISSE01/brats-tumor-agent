"""Évaluation Dice sur les régions BraTS WT / TC / ET (+ moyenne = mean Dice).

Convention BraTS : si les deux masques sont vides → Dice = 1.0.
Dice = 2|A∩B| / (|A|+|B|) = 2×TP / (2×TP + FP + FN).
"""
from __future__ import annotations

import numpy as np

from agent.schemas import DiceReport
from tools.postprocess import region_mask


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return 1.0
    inter = int(np.count_nonzero(p & g))
    return 2.0 * inter / denom


def evaluate_dice(pred_labels: np.ndarray, gt_labels: np.ndarray) -> DiceReport:
    pred = np.asarray(pred_labels)
    gt = np.asarray(gt_labels)
    if pred.shape != gt.shape:
        raise ValueError(f"shapes divergentes : pred {pred.shape} vs gt {gt.shape}")

    d_wt = dice_score(region_mask(pred, "wt"), region_mask(gt, "wt"))
    d_tc = dice_score(region_mask(pred, "tc"), region_mask(gt, "tc"))
    d_et = dice_score(region_mask(pred, "et"), region_mask(gt, "et"))
    mean = (d_wt + d_tc + d_et) / 3.0
    return DiceReport(
        dice_wt=round(d_wt, 6),
        dice_tc=round(d_tc, 6),
        dice_et=round(d_et, 6),
        mean_dice=round(mean, 6),
    )
