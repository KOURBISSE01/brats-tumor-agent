---
name: segmentation
description: Segmentation BraTS d'un cas — sliding window, hiérarchie WT ⊇ TC ⊇ ET, mode mock si pas de poids.
---

# Segmentation d'un patient BraTS

## Quand l'utiliser
Workflow rapide/standard : l'objectif est d'obtenir la carte de labels {0,1,2,3} et son résumé.

## Procédure
1. Charger le volume (4 modalités T1n/T1c/T2w/FLAIR empilées en (C,D,H,W)) via `tools.nifti_io.load_volume` ou directement un ndarray.
2. Appeler `tools.segmentation.segment_volume(volume, spacing, model_path)` :
   - poids TorchScript disponibles → inférence sliding-window 128³, overlap 0.5, importance gaussienne ;
   - sinon → **mode MOCK** : fantôme de sphères concentriques (ET ⊂ TC ⊂ WT), déterministe, adapté aux tests/CI, jamais à l'analyse clinique.
3. Vérifier la hiérarchie : `tools.postprocess.enforce_hierarchy` + `region_mask` — WT ≥ TC ≥ ET en voxels.
4. Post-traitement : composantes connexes 6-voisins (`count_components`), volume min configurable.

## Contraintes
- Toujours exposer `backend` (dynunet-torchscript | mock) dans le résultat : masquer un mock serait une faute de transparence.
- Échec de l'étape → isoler, logger dans `steps[]`, ne pas aborter tout le pipeline.
- Seuil de probabilité0.5 par défaut, min cumulé avant seuillage (p_et ≤ p_tc ≤ p_wt).

## Sorties attendues
`(labels uint8, meta{backend, warnings, spacing})` — puis biomarqueurs et risque dans le workflow standard.
