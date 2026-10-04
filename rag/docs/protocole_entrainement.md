# Protocole d'entraînement DynUNet (BraTS2023)

## Architecture
DynUNet3D, ~31M de paramètres : encodeur128³→8³, canaux32-64-128-256-320, décodeur symétrique avec connexions saut. Têtes3 sigmoïdes (WT/TC/ET) — pas de softmax car les régions se chevauchent (hiérarchie). Deep supervision aux échelles1.0 /0.5 /0.25 avec poids décroissants.

## Entraînement
Perte Dice-Focal (γ=2) — Dice pour l'équilibre des régions rares (ET), Focal pour les faux positifs. AdamW lr=1e-4, planification cosine,150 époques avec early stopping patience50. Batch2 ×128³. Split70/10/20 (train/val/test), seed42. Augmentation : flips, rotations, crops, jitter d'intensité. Suivi W&B.

## Inférence
Sliding window128³, overlap0.5, pondération importance gaussienne (bords des patches peu pondérés, coutures invisibles), puis fusion par moyenne pondérée. Post-traitement : hiérarchie des probabilités, seuillage0.5, composantes connexes.

## Optimisations recommandées (audite)
1. Augmentation d'intensité MRI agressive (bias field, gamma, ghosting) — absente du protocole, premier levier sur ET.
2. Sur-échantillonnage des cas ET petits (Dice ET le plus faible et le plus variance :0.871 ±0.20).
3. Seuils de risque calibrés sur la validation, gelés avant test (le pas actuel calibre sur le test = fuite).
4. Entraînement des3 canaux avec contrainte hiérarchique explicite (p_et ≤ p_tc ≤ p_wt) en loss.
5. Test-time augmentation (flips) pour gagner0.5-1 point de Dice moyen.
6. Calibration des probabilités (temperature scaling) avant seuillage0.5.
