---
name: rapport_radiologique
description: Rédaction du rapport d'analyse BraTS — chiffres uniquement issus des outils, ton clinique, mention obligatoire aide à la décision.
---

# Rapport d'analyse BraTS

## Quand l'utiliser
Workflow complet (et comparaison) : produire un rapport structuré pour un radiologue.

## Procédure
1. Congelés les faits : biomarqueurs (11), risque (niveau + confiance + critères), Dice si ground truth, backend. Ces valeurs sont la **seule** source de chiffres.
2. Structurer : volumes (tableau WT/TC/ET/œdème/nécrose) → ratios → forme/division → risque avec critères cités → Dice si dispo → références RAG.
3. Si LLM activé (OPENAI_API_KEY + use_llm) : prompt système restreint aux faits JSON, sinon rapport déterministe `agent.prompts.build_report` (toujours disponible).
4. Passer le rapport au contrôle de grounding : tout nombre à décimale doit exister dans les faits (tolérance1 % pour les arrondis d'affichage).

## Contraintes de style
- Phrases courtes, français clinique neutre, aucune spéculation étiologique.
- Mention finale obligatoire : « Aide à la décision — pas un diagnostic. Seuils heuristiques — valider avec un clinicien. »
- Métrique absente → « non disponible » (jamais de valeur inventée).
- Citer les références RAG par fichier + section (traçabilité).

## Contrôle qualité
- `grounded=False` → joindre l'avertissement avec les nombres fautifs.
- Étape en échec → le rapport le dit (section indisponible), il ne disparait pas silencieusement.
