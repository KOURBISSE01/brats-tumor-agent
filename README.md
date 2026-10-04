# BraTS Tumor Agent — reconstruction agentique

> Agent d'analyse d'IRM cérébrale **BraTS2023** : segmentation 3D → **11 biomarqueurs**
> → classification de risque **LOW / MED / HIGH** → rapport sourcé.
> Architectures agentiques : **RAG** (BM25), **MCP** (6 outils), **skills** (3 procédures).
> Déterministe, testé : **73 tests** · **0 dépendance obligatoire** hors `numpy` + `pydantic`.

```bash
git clone https://github.com/KOURBISSE01/brats-tumor-agent.git
cd brats-tumor-agent && .venv/bin/python -m pytest tests/ -q   # 73 passed
```

---

## Sommaire

1. [Analyse du projet d'origine (PDF)](#1-analyse-du-projet-dorigine-pdf)
2. [Architecture de l'agent](#2-architecture-de-lagent)
3. [Cœur de l'agent : workflows, biomarqueurs, risque](#3-cœur-de-lagent)
4. [Intégrations : API, MCP, skills, RAG](#4-intégrations)
5. [Optimisations apportées](#5-optimisations-apportées)
6. [Exemple de sortie](#6-exemple-de-sortie)
7. [Tests](#7-tests)
8. [Démarrage rapide & onboarding équipe](#8-démarrage-rapide--onboarding-équipe)
9. [Contribuer en équipe](#9-contribuer-en-équipe)
10. [Limites & suite](#10-limites--suite)

---

## 1. Analyse du projet d'origine (PDF)

### Ce que le projet fait

| Volet | Contenu du projet |
|---|---|
| Données | BraTS2023 — 1251 patients, 4 modalités (T1n, T1c, T2w, FLAIR), labels {0,1,2,3} |
| Régions | WT = 1+2+3 · TC = 1+3 · ET = 3 (hiérarchie WT ⊇ TC ⊇ ET) |
| Modèle | DynUNet3D, ~31M paramètres, encodeur 128³→8³ (canaux 32-64-128-256-320), deep supervision 1.0/0.5/0.25, 3 têtes sigmoïdes |
| Entraînement | DiceFocal γ=2, AdamW lr=1e-4, cosine, 150 époques (early stop p=50), batch 2×128³, aug Flip+Rotate+Crop+Jitter, split 70/10/20 seed 42, W&B |
| Inférence | Sliding window 128³ overlap 0.5, pondération importance gaussienne, post-traitement hiérarchique + composantes connexes |
| Métriques | Dice WT **0.9425 ±0.07** · TC **0.9210 ±0.14** · ET **0.8713 ±0.20** |
| Biomarqueurs | 11 métriques : 5 volumes (cm³) + 4 ratios + sphéricité + n composantes |
| Risque | LOW / MED / HIGH — seuils WT 62/115 et ET 11/25 (quantiles Q33/Q66), confiance 0.85 / 0.70 / 0.60+0.15×n |
| Livrables | API REST (8 endpoints), interface web (5 pages), suivi W&B |

### Points forts du projet d'origine
- Pipeline complet et reproductible (split seedé, suivi d'expériences).
- Post-traitement correct : hiérarchie des probabilités, suppression du bruit.
- Biomarqueurs bien choisis (les ratios ET/TC sont prédictifs du grade).
- Séparation claire entraînement / inférence / API.

### Problèmes identifiés (base des optimisations)
1. **Fuite de données (P0)** — les seuils de risque Q33/Q66 sont calibrés sur le
   **jeu de test**, puis les métriques sont mesurées sur ce même jeu. Calibrer
   sur la validation et geler avant test.
2. **Affirmation « tous benchmarks dépassés » (P1)** — sans test statistique
   (Wilcoxon par cas) ni IC, non soutenable face à un leaderboard public.
3. **Augmentation d'intensité absente (P1)** — pas de bias field / gamma /
   ghosting MRI : premier levier de gain sur l'ET (le plus difficile).
4. **ET sous-représenté (P1)** — Dice ET 0.871 ±0.20 : variance triple du WT.
   Sur-échantillonnage des petits ET recommandé.
5. **Pas de tests automatisés (P1)** — aucun pytest : les 11 biomarqueurs et les
   règles de risque n'étaient verrouillés par rien.
6. **Absents** : calibration des probabilités avant seuillage, TTA,
   estimation d'incertitude, anonymisation DICOM/DICOM-IR, gestion d'erreurs API.

---

## 2. Architecture de l'agent

```
brats-tumor-agent/
├── agent/                     ← CŒUR AGENT
│   ├── schemas.py             Pydantic : Workflow, Biomarkers, Risk, Dice, AgentResult
│   ├── orchestrator.py        planner par règles (+LLM), pipeline à contexte, grounding
│   ├── prompts.py             SYSTEM_PROMPT, template de rapport, LLM optionnel
│   └── skills_loader.py       chargement des SKILL.md par workflow
├── tools/                     ← OUTILS DÉTERMINISTES (numpy pur)
│   ├── nifti_io.py            .npy natif, .nii/.nii.gz via nibabel optionnel
│   ├── segmentation.py        sliding-window 128³ gaussien + backend MOCK déterministe
│   ├── postprocess.py         hiérarchie WT⊇TC⊇ET, composantes connexes (scipy ou BFS)
│   ├── surface.py             sphéricité : lissage σ 0.6 mm + marching tetrahedra
│   ├── biomarkers.py          les 11 biomarqueurs
│   ├── risk.py                LOW/MED/HIGH (règles Q33/Q66, confiances)
│   ├── evaluation.py          Dice WT/TC/ET + moyenne (convention BraTS)
│   └── png_util.py            encodeur PNG stdlib (gris + overlay rouge)
├── rag/
│   ├── knowledge_base.py      BM25 pur Python (RSJ, k1=1.5, b=0.75, accents OK)
│   └── docs/                  5 docs FR : régions, biomarqueurs, risque, protocole, dice
├── skills/
│   ├── segmentation/SKILL.md          procédure de segmenter (workflow rapide/standard)
│   ├── rapport_radiologique/SKILL.md  rédaction sourcée + contraintes de style
│   └── evaluation/SKILL.md            Dice + comparaison, pièges méthodologiques
├── mcp_server.py              serveur MCP stdio JSON-RPC2.0 — 0 dépendance, 6 outils
├── api/app.py                 FastAPI — les 8 endpoints du projet
├── tests/                     73 tests (outils, agent, RAG, MCP, PNG)
├── data/patients/             cas de démonstration
├── requirements.txt           cœur requis ; extras commentés (torch, nibabel, fastapi…)
└── .env.example               BRATS_MODEL_PATH, clé LLM optionnelle
```

### Flux d'exécution

```
texte utilisateur ─► classify_intent (règles, accents normalisés)
                          │  rapide | standard | complet | comparaison
                          ▼
        ┌── segmentation ──► biomarqueurs ──► risque ──► [dice si GT] ──► RAG ──► rapport ──► grounding
        │   (mock|DynUNet)   (11 métr.)      (règles)                  (BM25)   (déterministe│LLM)  (chiffres)
        └── chaque étape isolée : échec ⇒ steps[i].ok=false, pipeline continue
```

### 4 garanties de l'agent

1. **Règles d'abord** — le planificateur est un ensemble d'expressions
   régulières testées (11 cas en pytest) ; le LLM n'intervient que si
   `use_llm=True` + clé présente, sinon fallback déterministe.
2. **Grounding des nombres** — tout nombre à décimale du rapport doit exister
   dans les sorties d'outils (tolérance 1 % pour les arrondis d'affichage,
   scores RAG inclus). Sinon `grounded=false` + liste des nombres fautifs.
3. **Isolation par étape** — une étape plante ⇒ erreur capturée, `steps[]`
   journalisé, les autres étapes continuent, le rapport le dit.
4. **Transparence du backend** — sans poids TorchScript, le segmenteur tourne en
   **MOCK** (fantôme sphérique déterministe) et le signale dans les warnings :
   jamais de mock présenté comme une vraie prédiction.

---

## 3. Cœur de l'agent

### 3.1 Les 4 workflows

Workflow choisi par `classify_intent` (normalisation NFKD des accents, puis règles) :

| Workflow | Déclencheurs (exemples) | Étapes exécutées | Skill appliqué | RAG |
|---|---|---|---|---|
| `rapide` | « rapide », « synthèse », « résumé » | segmentation → biomarqueurs → risque → rapport | `segmentation` | non |
| `standard` | défaut (aucune correspondance) | segmentation → biomarqueurs → risque → rapport | `segmentation` | non |
| `complet` | « analyse **complète** », « rapport », « dice », « evaluation » | … → **dice** (si GT) → **RAG** → rapport | `rapport_radiologique` | **oui** (citations) |
| `comparaison` | « comparer », « versus », « deux patients », « cote a cote » | pipeline **×2 cas** → rapport comparatif | `evaluation` | **oui** |

### 3.2 Les 11 biomarqueurs

| # | Champ | Unité | Définition |
|---|---|---|---|
| 1 | `volume_wt_cm3` | cm³ | tumeur entière (labels 1+2+3) |
| 2 | `volume_tc_cm3` | cm³ | cœur tumoral (1+3) |
| 3 | `volume_et_cm3` | cm³ | rehaussement actif (3) |
| 4 | `volume_edema_cm3` | cm³ | œdème (2) |
| 5 | `volume_necrosis_cm3` | cm³ | nécrose (1) |
| 6 | `ratio_et_wt` | — | ET/WT (0 si dénominateur vide) |
| 7 | `ratio_et_tc` | — | ET/TC |
| 8 | `ratio_tc_wt` | — | TC/WT |
| 9 | `ratio_edema_wt` | — | Œdème/WT |
| 10 | `sphericity_wt` | — | ψ ∈ [0, ~1] : 1 = sphère parfaite |
| 11 | `n_components` | — | composantes connexes WT (volume mini paramétrable) |

### 3.3 Règles de risque

| Niveau | Conditions (testées dans l'ordre) | Confiance |
|---|---|---|
| **LOW** | WT < 62 cm³ **et** ET < 11 cm³ **et** 1 seule composante connexe | **0.85** |
| **HIGH** | WT > 115 cm³ **ou** ET > 25 cm³ **ou** ≥ 3 composantes | **0.60 + 0.15 × n_critères**, plafonné à **1.0** |
| **MED** | sinon | **0.70** |

> Bornes exclusives (WT = 62 exact ⇒ pas LOW). Seuils issus des quantiles
> Q33/Q66 du **jeu de test** du projet d'origine → **fuite de données** :
> chaque évaluation porte la note « PAS une recommandation clinique » et
> `rag/docs/regles_risque.md` documente le recalibrage à faire sur la validation.

---

## 4. Intégrations

### 4.1 API REST — 8 endpoints (FastAPI)

| Méthode | Chemin | Rôle |
|---|---|---|
| GET | `/api/health` | vivacité + backend modèle |
| GET | `/api/patients` | liste des IDs disponibles |
| GET | `/api/patients/{patient_id}` | métadonnées (modalités, shape, spacing) |
| GET | `/api/patients/{patient_id}/slice/{n}` | PNG (gris + overlay) d'une coupe |
| POST | `/api/inference/predict` | lance un job agent `{patient_id, text}` |
| GET | `/api/jobs/{job_id}` | état du job + steps + warnings |
| GET | `/api/jobs/{job_id}/slice/{n}` | coupe du résultat |
| GET | `/api/jobs/{job_id}/rapport` | rapport Markdown complet |

Les jobs sont en mémoire (dictionnaire verrouillé) : un redémarrage les perd.

### 4.2 MCP — 6 outils (stdio, JSON-RPC 2.0, protocole `2024-11-05`)

| Outil | Arguments principaux | Retour |
|---|---|---|
| `segment` | `volume_path`, `output_path` | backend, warnings, voxels WT |
| `extract_biomarkers` | `labels_path` | les 11 biomarqueurs JSON |
| `classify_risk` | `biomarkers` (ou via labels) | niveau, confiance, critères, note |
| `evaluate_dice` | `pred_path`, `gt_path` | Dice WT/TC/ET + moyenne |
| `query_knowledge` | `query`, `k` | top-k BM25 avec `source` + score |
| `render_slice` | `volume_path`, `output_path` (+ overlay optionnel) | fichier PNG |

Zéro dépendance externe (pas de SDK MCP) : le serveur lit/écrit des `.npy` et
rend des PNG avec l'encodeur stdlib — testé en sous-processus réel (9 tests).

### 4.3 Skills — 3 procédures (`SKILL.md`)

| Skill | Appliqué quand | Contenu |
|---|---|---|
| `segmentation` | rapide / standard | ordre des opérations, hiérarchie WT⊇TC⊇ET |
| `rapport_radiologique` | complet | plan du rapport, style, obligation de sourcer, garde-fous |
| `evaluation` | comparaison | protocole Dice, pièges méthodologiques (double vide, décalage) |

### 4.4 RAG — 5 documents FR (BM25, `k1=1.5`, `b=0.75`, RSJ idf)

| Fichier | Couverture |
|---|---|
| `brats_regions.md` | labels, WT/TC/ET, hiérarchie |
| `biomarqueurs.md` | les 11 métriques, interprétation des ratios |
| `regles_risque.md` | seuils, confiances, **avertissement fuite Q33/Q66** |
| `protocole_entrainement.md` | setup DynUNet + **6 optimisations recommandées** |
| `interpretation_dice.md` | lecture des scores, variances ET vs WT |

Tokenisation NFKD (accents gérés), stopwords FR, découpage sur les titres `## `.
Le workflow `complet` **cite ses sources** dans le rapport (traçabilité).

---

## 5. Optimisations apportées

### 5.1 Sphéricité — bug silencieux corrigé
L'estimation naïve par comptage de faces (A6) **inverse l'ordre géométrique** :
surestimant les surfaces courbes d'un facteur ~1.5 (biais d'orientation des
marches d'escalier), une sphère ressort ψ≈0.67 alors qu'un cube vaut 0.81.
La variation totale L1 a le même biais. **Solution implémentée** :
`tools/surface.py` = lissage gaussien σ=0.6 mm (supprime les marches d'escalier)
+ marching tetrahedra vectorisé (6 tétraèdres/cube, interpolation linéaire),
découpage en tranches Z pour la mémoire (240³ ≈ 18 s).

| Forme | A6 brut | Marching tets + σ | Cible |
|---|---|---|---|
| Cube 10³ | 0.806 | **0.887** | 0.806 (arrondi σ) |
| Sphère r=16 | 0.671 ✗ | **0.961** ✓ | 1.0 |
| Ordre sphère > cube | **non** ✗ | **oui** ✓ | obligatoire |

### 5.2 Sliding-window — division par zéro aux bords
Le plancher `maximum(weight, 1e-8)` écrasait les importances gaussiennes
réelles des coins (≈5e-42) et faussait les voxels de bord. Remplacé par une
division conditionnelle `weight > 0` (exacte, sans plafonnement artificiel).

### 5.3 Classification d'intention — accents
`« analyse complète »` ne matchait pas `complet` à cause du **è**. Normalisation
NFKD (accents retirés) avant les règles : 11/11 cas de test au lieu de 8/11.

### 5.4 Fuite de calibration documentée dans le RAG
`rag/docs/regles_risque.md` et `biomarqueurs.md` rappellent que les seuils
Q33/Q66 sont calibrés sur le test et doivent être recalibrés sur la validation —
chaque rapport de risque porte la note « PAS une recommandation clinique ».

### 5.5 Techniques agentiques demandées
- **RAG** — BM25 sans dépendance sur 5 documents FR de protocole ; le workflow
  complet interroge la base et **cite les sources** dans le rapport.
- **MCP** — serveur stdio zéro dépendance (initialize/tools/list/tools/call,
  notifications sans réponse) : 6 outils pilotables par Claude Code ou tout
  client MCP, passation par fichiers `.npy`.
- **Skills** — 3 `SKILL.md` (frontmatter + procédure + contraintes) chargés par
  workflow.

---

## 6. Exemple de sortie

Requête : `« analyse complète du patient »` (cas démo seed 42, backend MOCK) :

```
workflow: complet | grounded: True | skill: rapport_radiologique
risk: low 0.85 | wt: 19.484 cm³
rag: regles_risque.md (score 5.536) · biomarqueurs.md (score 4.566)
steps: segmentation✓ biomarqueurs✓ risque✓ rag✓ rapport✓
```

```markdown
## Rapport d'analyse — DEMO
**Workflow exécuté :** complet · **Backend modèle :** mock

### 1. Volumes (cm³)
| Région | Volume |
|---|---|
| WT (tumeur entière) | 19.484 |
| TC (cœur tumoral) | 5.774 |
| ET (rehaussement) | 1.244 |
| Œdème | 13.710 |
| Nécrose | 4.530 |

### 2. Ratios clés
- ET/WT = 0.0638 · ET/TC = 0.2154
- TC/WT = 0.2963 · Œdème/WT = 0.7037

### 3. Forme & division
- Sphéricité WT = 0.9570 (1 = sphère parfaite) · Composantes connexes = 1

### 4. Risque
**Niveau : low** (confiance 0.85)
Critères : WT 19.48 < 62 cm³; ET 1.24 < 11 cm³; 1 composante connexe unique

### Références (RAG)
- regles_risque.md · regles_risque (score 5.536)
- biomarqueurs.md · biomarqueurs (score 4.566)

---
*Aide à la décision — pas un diagnostic. Seuils heuristiques (Q33/Q66) — valider avec un clinicien.*
```

Chiffres **groundés** : chaque valeur décimale du rapport existe dans les
sorties d'outils — le contrôle est automatique (`grounded` booléen + `warnings`).

---

## 7. Tests

**73 tests · ~5,5 s · aucun réseau, aucun modèle requis**

| Fichier | # | Ce qu'il verrouille |
|---|---:|---|
| `test_orchestrator.py` | 18 | intents (accents), 4 workflows, isolation d'étape, grounding (faux positif/négatif), skills, sections du rapport, fallback LLM |
| `test_segmentation_png.py` | 10 | mock déterministe + hiérarchie, sliding-window (padding, bords), PNG (signature, color-type, overlay rouge) |
| `test_mcp.py` | 9 | sous-processus réel : initialize, tools/list, notifications, roundtrip segment→biomarqueurs, dice, RAG, erreurs, PNG |
| `test_rag.py` | 9 | 5 requêtes doc, accents/stopwords, requête vide |
| `test_risk.py` | 9 | LOW/HIGH/MED, bornes exclusives, plafond confiance 1.0, note fuite |
| `test_biomarkers.py` | 6 | volumes exacts au voxel, ratios, gardes division/0, ordre sphère>cube, spacing anisotrope |
| `test_evaluation.py` | 6 | Dice parfait/vide/décalé, convention BraTS, erreurs de shape |
| `test_postprocess.py` | 6 | hiérarchie probabilités & labels, composantes connexes (scipy et BFS) |

```bash
.venv/bin/python -m pytest tests/ -q          # 73 passed in ~5.5s
.venv/bin/python -m pytest tests/test_risk.py # cibler un fichier
```

---

## 8. Démarrage rapide & onboarding équipe

### 8.1 Installation

```bash
git clone https://github.com/KOURBISSE01/brats-tumor-agent.git
cd brats-tumor-agent
python3 -m venv .venv
.venv/bin/pip install numpy pydantic pytest          # cœur (obligatoire)
.venv/bin/pip install fastapi httpx uvicorn          # API (optionnel)
# extras : torch (vrai modèle), nibabel (NIfTI), scipy (CC rapide)

.venv/bin/python -m pytest tests/ -q                 # 73 passed
```

### 8.2 Lancer l'API

```bash
.venv/bin/uvicorn api.app:app --port 8000

curl localhost:8000/api/health
curl localhost:8000/api/patients/BraTS_demo_001
curl -X POST localhost:8000/api/inference/predict \
     -H 'Content-Type: application/json' \
     -d '{"patient_id":"BraTS_demo_001","text":"rapport complet"}'
# → {"job_id": "..."}  puis :
curl localhost:8000/api/jobs/<job_id>/rapport
curl localhost:8000/api/jobs/<job_id>/slice/32        # PNG
```

### 8.3 Brancher l'agent MCP (Claude Code / clients MCP)

```json
{ "mcpServers": { "brats": {
    "command": "/chemin/absolu/brats-tumor-agent/.venv/bin/python",
    "args": ["/chemin/absolu/brats-tumor-agent/mcp_server.py"] } } }
```

Outils exposés : `segment`, `extract_biomarkers`, `classify_risk`,
`evaluate_dice`, `query_knowledge`, `render_slice`.

### 8.4 Usage Python direct

```python
import sys, numpy as np
sys.path.insert(0, ".")
from agent.orchestrator import Case, run_agent

vol = np.random.default_rng(0).normal(100, 30, (4, 64, 64, 64)).astype(np.float32)
r = run_agent("rapport complet avec évaluation", Case("DEMO", vol))
print(r.risk.level, r.biomarkers.volume_wt_cm3, "cm³ | grounded:", r.grounded)
print(r.report)          # Markdown prêt à partager
```

### 8.5 Exemples de prompts à essayer

| Prompt | Workflow résultant |
|---|---|
| `« donne un résumé rapide »` | rapide |
| `« analyse complète du patient »` | complet (+ RAG cité) |
| `« compare les deux patients cote a cote »` | comparaison |
| `« évalue le dice contre la vérité terrain »` | complet + dice |

### 8.6 Activer le vrai modèle DynUNet (optionnel)

```bash
pip install torch                       # + exporter DynUNet en TorchScript
export BRATS_MODEL_PATH=/chemin/vers/dynunet.ts   # ou via .env
```

Sans poids : backend **MOCK** (fantôme sphérique déterministe) — idéal
démo/CI, **signalé dans chaque résultat** (`warnings` + section du rapport).

---

## 9. Contribuer en équipe

- **Branche par sujet** : `feat/…`, `fix/…`, `docs/…` depuis `main`, PR vers `main`.
- **Toute règle métier a un test** : seuil de risque, biomarqueur, intent
  d'utilisateur → test dans le fichier correspondant (`tests/test_*.py`).
- **Grounding** : si vous ajoutez un chiffre au rapport, sa source doit exister
  dans les sorties d'outils, sinon le test de grounding échoue (c'est voulu).
- **Docs RAG** : une nouvelle règle = un fichier `rag/docs/*.md` avec un titre
  `## ` par section + un test dans `test_rag.py` qui vérifie la requête.
- **Style** : identifiants en anglais, contenu utilisateur/rapports en français,
  docstrings courtes, zéro dépendance non déclarée dans `requirements.txt`.
- Vérifier avant PR :
  ```bash
  .venv/bin/python -m pytest tests/ -q    # 73 passed attendu
  ```

---

## 10. Limites & suite

- Le **mock** produit un fantôme déterministe : il ne segmente pas de vraies IRM.
- Sphéricité : ψ cube ≈ 0.89 (σ arrondit les arêtes), peut dépasser 1 sur
  petites formes (discretisation) — biais identique entre patients, comparaisons
  valides.
- API : jobs en mémoire (un redémarrage les perd) — persister en base si besoin.
- Front-end non inclus (le PDF décrivait 5 pages) : l'API couvre les 8 endpoints
  et sert les slices PNG prêts à afficher.
- Suite recommandée : recalibrage des seuils sur validation, augmentation
  d'intensité MRI, TTA, calibration des probabilités, tests Wilcoxon par cas.

---
*Projet reconstitué avec RAG + MCP + skills — aide à la décision, pas un diagnostic.*
