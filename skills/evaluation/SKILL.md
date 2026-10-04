---
name: evaluation
description: Évaluation Dice WT/TC/ET et comparaison de deux cas — interprétation correcte, pièges de calibration.
---

# Évaluation et comparaison

## Quand l'utiliser
- Présence d'un ground truth en workflow complet → Dice WT/TC/ET + moyenne.
- Workflow comparaison → deux patients côte à côte (volumes, ratios, risque).

## Procédure
1. `tools.evaluation.evaluate_dice(pred, gt)` : Dice par région (WT = labels>0, TC ∈ {1,3}, ET =3) + mean. Convention : deux masques vides →1.0.
2. Comparaison : table avec WT/ET/risque des deux cas, écarts absolus, jamais de classement sans rappeler l'incertitude ET (écart-type0.20).
3. Interpréter avec `rag/docs/interpretation_dice.md` : difficulté décroissante WT > TC > ET, bruit inter-observateurs ~0.05-0.10.

## Pièges à ne pas commettre
1. Ne jamais comparer des Dice si les seuils de risque ont été calibrés sur le même jeu de test (fuite) — le rappeler si le contexte le mentionne.
2. Ne pas conclure « modèle meilleur » sur un écart <0.01 sans test statistique.
3. Un bon Dice ne garantit pas un volume non biaisé — si des volumes prédit/vrai existent, rapporter la corrélation.

## Sorties attendues
`DiceReport{dice_wt, dice_tc, dice_et, mean_dice}` et, en comparaison, la section5 du rapport avec les deux colonnes.
