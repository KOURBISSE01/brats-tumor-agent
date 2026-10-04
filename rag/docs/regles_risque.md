# Classification de risque LOW / MED / HIGH

## Règle de décision (heuristique)
LOW si WT < 62 cm³ ET ET < 11 cm³ ET une seule composante connexe (confiance0.85). HIGH si WT > 115 cm³ OU ET > 25 cm³ OU ≥3 composantes connexes (confiance0.60 +0.15 × nombre de critères déclenchés, plafond1.0). MED sinon (confiance0.70). Les conditions LOW et HIGH sont mutuellement exclusives : on teste LOW d'abord.

## Origine des seuils — calibration Q33/Q66
Les valeurs62/115 (WT) et11/25 (ET) sont les quantiles Q33 et Q66 des volumes du jeu de test BraTS : le tiers inférieur des tumeurs est « petites », le tiers supérieur « grosses ». C'est une règle de stratification par volume, pas un seuil de malignité biologique.

## Avertissement méthodologique (data leakage)
Calibrer les seuils sur le jeu de TEST puis mesurer les métriques sur ce même jeu fausse l'évaluation : les seuils ont vu les données. Bonne pratique : fixer les seuils sur le jeu d'entraînement ou de validation, geler avant l'évaluation finale, idéalement par région (le ET varie bien plus que le WT — écart-type0.20 contre0.07 pour le Dice).

## Ce que le risque N'EST PAS
Le niveau LOW/MED/HIGH n'est pas un diagnostic de grade histologique, pas une prédiction de survie, pas une décision thérapeutique. C'est une priorisation quantitative à valider par un radiologue. L'affichage doit toujours rappeler : aide à la décision, pas un diagnostic.
