# Interprétation des Dice scores BraTS

## Lecture des métriques du projet
Dice WT =0.9425 ±0.07, Dice TC =0.9210 ±0.14, Dice ET =0.8713 ±0.20 (moyenne ± écart-type sur le jeu de test). L'ordre de difficulté est attendu : WT est la région la plus grande et la plus contrastée, ET est petite et dépend du rehaussement — variance triple pour ET.

## Dice parfait et cas limites
Dice =2|A∩B|/(|A|+|B|) ∈ [0,1]. Les deux masques vides → Dice =1.0 (convention BraTS). Une région absente des deux (ET=0 chez un patient sans rehaussement) ne pénalise pas. Un Dice ET de0.85 sur un ET de5 cm³ vaut mieux qu'un Dice WT de0.95 sur une masse infiltrante : toujours pondérer par le volume clinique.

## Comparaison inter-observateurs
Le bruit inter-annotateurs sur BraTS est souvent0.05-0.10 de Dice — un modèle à0.94 sur WT est proche de l'accord humain. Un écart de0.01 entre deux versions du modèle n'est significatif que s'il est stable sur les cas ET, là où la variance est0.20.

## Erreurs d'interprétation fréquentes
1. Comparer un Dice global sans décomposer WT/TC/ET — le WT masque les échecs sur l'ET.2. Annoncer « tous les benchmarks dépassés » sans intervalle de confiance ni test statistique (Wilcoxon sur les cas).3. Évaluer sur le jeu de test utilisé pour calibrer les seuils de risque — fuite de données.4. Oublier le volumétrie : un modèle peut avoir un bon Dice et un volume systématiquement biaisé (±10 %) — rapporter aussi la corrélation de volume prédit/vrai.
