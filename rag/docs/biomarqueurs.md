# Biomarqueurs BraTS — les11 métriques calculées

## Les5 volumes (cm³)
volume_wt (tumeur entière), volume_tc (cœur tumoral), volume_et (rehaussée), volume_edema (œdème label 2), volume_necrosis (nécrose label 1). Volume = n_voxels × spacing /1000. Ce sont les entrées du modèle de risque.

## Les4 ratios
ratio_et_wt = ET/WT (fraction rehaussée dans la tumeur totale), ratio_et_tc = ET/TC (activité dans le cœur), ratio_tc_wt = TC/WT (compacité du cœur), ratio_edema_wt = œdème/WT (dominante œdémateuse). Dénominateur nul →0.0. Un ET/WT élevé avec un WT modeste signale une tumeur à fort rehaussement.

## Sphéricité et composantes connexes
sphericity_wt = π^(1/3)(6V)^(2/3)/A —1 = sphère parfaite, ≈0.89 pour un cube (estimation discrète : lissage gaussien0.6 mm + marching tetrahedra, voir tools/surface.py). n_components = nombre de composantes connexes WT (6-voisins), un marqueur de dissémination multi-focale. Le comptage de faces brut est proscrit : il inverse l'ordre sphère/cube.

## Seuils de volume utilisés
Dans le prototype BraTS du dossier, les seuils de risque utilisaient WT62 cm³ (quantile Q33), WT115 cm³ (Q66), ET11 cm³ (Q33), ET25 cm³ (Q66). Ces valeurs viennent des quantiles du jeu de test — calibration à refaire sur la validation (voir règles de risque).
