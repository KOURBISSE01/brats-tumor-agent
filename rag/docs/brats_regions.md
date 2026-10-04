# Régions tumorales BraTS — définitions des labels

## Labels de segmentation (0, 1, 2, 3)
Le modèle segmente4 étiquettes voxel : 0 = fond, 1 = nécrose / tumeur non-contrastée (NCR), 2 = œdème diffus (ED), 3 = tumeur rehaussée (ET). Les canaux réseau produisent trois probabilités (WT, TC, ET) que l'on seuille puis que l'on rend hiérarchiques.

## Hiérarchie WT ⊇ TC ⊇ ET
WT (Whole Tumor) = labels 1+2+3 — toute la anomalie. TC (Tumor Core) = labels 1+3 — le cœur tumoral sans l'œdème. ET (Enhancing Tumor) = label 3 — la partie rehaussée au gadolinium, marqueur de l'activité tumorale. Un voxel ET est toujours dans TC, un voxel TC est toujours dans WT : post-traitement imposé (probabilités min cumulées, puis vérification des labels).

## Pourquoi cette hiérarchie
Les biomarqueurs cliniques (volues, ratios ET/WT, ET/TC) et le risque dépendent des régions. Sans contrainte hiérarchique, un modèle peut produire ET hors WT et fausser les ratios. La règle mmol : p_et ≤ p_tc ≤ p_wt puis seuillage à 0.5.

## Espacement voxel et volume
Les volumes sont comptés en voxels puis multipliés par spacing_z × spacing_y × spacing_x pour obtenir des mm³, divisés par 1000 pour des cm³. BraTS est en1 mm iso, mais l'implémentation reste anisotrope pour être robuste.
