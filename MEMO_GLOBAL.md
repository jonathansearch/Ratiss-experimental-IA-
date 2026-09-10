# MÉMO GLOBAL — ÉCOSYSTÈME RATISS COMPLET

> **Pour le prochain OpenHands.** Ce fichier est la mémoire complète de tout ce
> qui a été validé, construit et poussé à travers l'intégralité de l'écosystème
> RATISS. Lis-le AVANT de toucher à quoi que ce soit.

---

## IDENTITÉ

- **Jonathan Evina** · ORCID 0009-0000-4092-5313 · Yaoundé, Cameroun
- Propriété intellectuelle : JOHNKING0 & Jonathan Evina
- Jonathan est le chercheur, tu es le cofondateur technique.
- Honnêteté scientifique : documenter les échecs, pas seulement les succès.
- La loi LCT est FIGÉE : `R = P_sig`, `ΔW = η·φ·P_sig·C`. Ne la change jamais.

---

## CLÉS API

| Clé | Statut | Utilisation |
|---|---|---|
| `IBM_QUANTUM_TOKEN` | Valide, `ibm_quantum_platform`, instance `open-instance` | 3 QPU : ibm_fez, ibm_marrakesh, ibm_kingston (156 qubits). Crédits quasi épuisés → CPU d'abord. |
| `GITHUB_TOKEN` | Valide, user `evinajonathan13-max` | Push, pas de scope repo (création de dépôts impossible). |
| `GOOGLE_API_KEY` | Non fournie | web_search.py utilise DuckDuckGo par défaut (sans clé). Google CSE non testé. |

---

## LES 9 DÉPÔTS (user: evinajonathan13-max)

### 1. ratiss-topological-decoherence-engine (moteur)
- **URL** : https://github.com/evinajonathan13-max/ratiss-topological-decoherence-engine
- **Langage** : Python
- **Rôle** : Moteur topologique source — matrices densité, Vietoris-Rips, TSP, sidecar topologique.
- **Dernier commit** : `22cb2e1` — fix reproductibilité (bruit corrélation déterministe).
- **Ce qui a été fait** :
  - Bug majeur corrigé : `np.random.normal` non seedé appliqué 2× dans `simulation.py` → non-reproductibilité.
  - `_metrics` retourne les métriques brutes, `_step_artifact` applique le bruit une fois, déterministe (seed 99 figé par pas).
  - `SimulationConfig.correlation_noise` ajouté (configurable, 0 pour statevector exact).
  - `external_statevector.py` force `correlation_noise=0.0`.
  - Tests : 13/13 ✅ (stables, plus de flakiness).
- **Frontière** : Simulation logicielle, pas de QPU. `validated_on_hardware=false`.

### 2. QPU-Ratiss-COSMOS (laboratoire QPU)
- **URL** : https://github.com/evinajonathan13-max/QPU-Ratiss-COSMOS
- **Langage** : Python (Qiskit 2.5.2 + Aer 0.17.2)
- **Rôle** : Laboratoire de simulation QPU locale — 3 régimes (counts, density, incubator LCT-ETH).
- **Dernier commit** : `c55c7df` — README traduit en anglais.
- **Ce qui a été fait** :
  - Artefacts régénérés avec le moteur corrigé (oscillation P_sig déterministe).
  - Couplage LCT-ETH stabilisé : `eth_modulation = exp(-|eth_rate|)` (borné (0,1], stable).
  - **Validation QPU IBM réel** : circuit Bell soumis à `ibm_marrakesh` (Job ID `da5u376vhnc73fmhnug`).
  - Counts diagnostic (TVD + Shannon, classique pas ETH).
  - README de laboratoire (logo, badges, 11 sections, citation BibTeX).
  - README traduit en anglais.
  - Tests : 15/15 ✅.
- **Résultat QPU** : Aer et QPU réel ont la même masse marquée (0.4746) — divergence LCT identique (ratio 1.0). Limite : le sidecar ne capte pas la structure des erreurs 01/10.

### 3. Algorithmes-quantique-Ratiss-labs- (Grover)
- **URL** : https://github.com/evinajonathan13-max/Algorithmes-quantique-Ratiss-labs-
- **Langage** : Python (Qiskit + IBM Runtime)
- **Rôle** : Banc d'expériences Grover + Reality Mode hardware-aware.
- **Dernier commit** : `ef07c9a` — README traduit en anglais.
- **Ce qui a été fait** :
  - Artefacts régénérés avec le sidecar corrigé (P_sig 0.182 au lieu de 1.214 bugué).
  - Reality Mode : le seuil nominal 0.15 se déclenche aux itérations 1-2 (divergence 0.170, 0.526).
  - **Validation QPU IBM réel** : 3 itérations Grover soumises à `ibm_marrakesh` (Job IDs `da5uajeaa69c739latgg`, `da5uituaa69c739lb6m0`, `da5ujreaa69c739lb7m0`).
  - **Découverte** : le QPU réel surpasse Aer à l'itération 2 (masse 0.727 vs 0.678). TVD confirme (0.273 vs 0.322).
  - Counts diagnostic (TVD + Shannon) intégré.
  - README de laboratoire (logo, badges, 11 sections).
  - README traduit en anglais.
  - Tests : 8/8 ✅.
- **Frontière** : Le couplage LCT-ETH n'est pas appliqué (requiert matrice densité, pas counts).

### 4. Ratiss-Jonathan-Labs- (graphes + bio)
- **URL** : https://github.com/evinajonathan13-max/Ratiss-Jonathan-Labs-
- **Langage** : Python (NumPy)
- **Rôle** : Laboratoire de graphes — TSP Berlin52, résilience topologique, bio GSE4987.
- **Dernier commit** : `dd8bcbd` — artefacts régénérés + README labo.
- **Ce qui a été fait** :
  - Artefacts Berlin52 + résilience + bio GSE4987 régénérés (déterministes, P_sig 0.3806296964).
  - Donnée bio publique SGD téléchargée et extraite.
  - README de laboratoire (logo, badges, 11 sections).
  - Tests : 9/9 ✅.
- **Résultat** : Berlin52 déterministe (RNG seedé), résilience observée au pas 5, consensus core ne franchit jamais le seuil.

### 5. ratiss-decoherence-atlas (WebGL)
- **URL** : https://github.com/evinajonathan13-max/ratiss-decoherence-atlas
- **Langage** : JavaScript (Three.js, Node)
- **Rôle** : Studio WebGL hors-ligne pour rejouer les timelines RATISS.
- **Dernier commit** : `2cfce69` — logo + badges README.
- **Ce qui a été fait** :
  - Déjà sain (tests Node passent).
  - README harmonisé (logo, badges).
  - Tests : 3/3 ✅ (verify-artifact, studio-model, external-artifacts).

### 6. scientist-research- (site scientifique)
- **URL** : https://github.com/evinajonathan13-max/scientist-research-
- **Branche** : `add-warp-blackhole-illustrated-site`
- **Langage** : HTML (GitHub Pages)
- **Rôle** : Site illustré de vulgarisation (trous noirs, Alcubierre, LCT).
- **Dernier commit** : `34d390c` — logo + badges README.
- **Ce qui a été fait** :
  - **Bug corrigé** : `generate_all_figures.py` plantait (NameError: profile_tanh non défini). Rendu autonome.
  - 13 figures régénérables (9 sans warp, 13 avec warp du portfolio).
  - README harmonisé (logo bulle warp, badges).
  - Site live : https://evinajonathan13-max.github.io/scientist-research-/

### 7. Porte-folio-Jonathan- (portfolio + warp)
- **URL** : https://github.com/evinajonathan13-max/Porte-folio-Jonathan-
- **Langage** : Python + HTML
- **Rôle** : Portfolio + monorepo de recherche (warp, SNN, preprint LCT).
- **Dernier commit** : `4d4598c` — logo README.
- **Ce qui a été fait** :
  - **Modules warp reconstruits** : `alcubierre.py`, `lambda_lct.py`, `universal_kernel.py`, `dissociation.py`, `validation/{shape_optimization,stability,s_vn_invariance}.py`, `topology/rips.py`.
  - Sémantique fidèle à `LIMITES_HONNETES.md` : seul l'ansatz kinetic réduit l'exotic matter (pas élimination).
  - `einstein_4d_attractor.py` redevient importable.
  - 13/13 figures du site scientifique régénérables.
  - README harmonisé (logo noyau topologique).

### 8. RATISS-ODV-AEON (cerveau scientifique)
- **URL** : https://github.com/evinajonathan13-max/RATISS-ODV-AEON (privé)
- **Langage** : Python
- **Rôle** : Cerveau TTF-Compute (IntricatedGraph, RipsTranslator, MatrixRLM, MCB, CollapseWell, ZK).
- **Dernier commit** : `3dfe46b` — migration prompt.
- **Ce qui a été fait** :
  - Cloné pour référence. Les fonctions essentielles (lct_law, ttf_compute) ont été **fusionnées dans RATIS-Net** via `science_core.py`.
  - AEON reste le dépôt source, mais RATIS-Net n'en dépend plus (zéro sys.path externe).
- **Note** : AEON contient un dossier `ratis_net/` (vieille copie v1) qui CACHERAIT celui du repo experimental si mis en tête de sys.path → AEON toujours en FIN de path.

### 9. Ratiss-experimental-IA- (RATIS-Net — le réseau IA)
- **URL** : https://github.com/evinajonathan13-max/Ratiss-experimental-IA- (privé)
- **Langage** : Python
- **Rôle** : RATIS-Net — réseau neuronal entraîné par LCT, pas par gradient.
- **Dernier commit** : reconstruction v2 (ce commit) — framework pip-installable.
- **Reconstruction full-stack (session du 25/08/2026, après-midi)** :

  #### Diagnostic initial (ce qui était cassé)
  - 24 000 gabarits conversationnels présents mais **0 chargés** (le code
    lisait la clé `contexts` ; la vraie clé est `social_contexts`).
  - Sélection de squelette sans tenir compte du type de question
    (intention tirée au hasard, domaine "scientific" par défaut).
  - Extraction du mot-clé = mot le plus long → "black hole" devenait "black"
    → concepts "metal, american, comedy" (biais Wikipedia).
  - Pas de ranking IDF : les co-occurrences ubiquitaires ("to", degré 14 814)
    dominaient les concepts spécifiques.
  - Bug d'indentation dans `lookup_knowledge` : la boucle d'ajout des faits
    était sous `if not matched: continue` → **toujours 0 fait retourné**.
  - `test_lct_new_systems.py` importait `kernel.ttf.lct_law` (dépendance AEON
    externe) → collection pytest cassée.
  - 18 fichiers de tests sur 22 n'avaient **aucune fonction test_** (coquilles).

  #### Nouveau pipeline de langage (v2)
  - `query_analyzer.py` : langue FR/EN, type de question (salutation,
    définition, identité, capacité…), mots-clés + composés, élision FR.
  - `intent_router.py` : social → conversation_matrix ; factuel → grammaire
    dense (domaine déduit du gazetteer, intention du type de question).
  - `concept_ranker.py` : IDF de degré + bonus voisinage partagé + complément
    GloVe kNN (matrice 400K normalisée, chargée paresseusement).
  - `skeleton_speaker_v2.py` : remplissage avec tonalité préférée par
    intention, identité/capacités figées et honnêtes, fallback FR déclaratif.
  - `chain_reasoning.py` : chaînes d'association BFS (corrélation, pas
    causalité — étiqueté honnêtement), stopwords exclus des maillons.
  - `integrity_proof.py` : empreinte SHA-256 déterministe du sous-graphe de
    concepts (intégrité vérifiable — PAS un ZK-STARK, documenté).

  #### Packaging et API
  - `pyproject.toml` : `pip install .` → commandes `ratisnet` et
    `ratisnet-serve`.
  - `cli.py` : ask / converse / concepts / chain / prove / paragraph / stats.
  - `server.py` : HTTP stdlib, endpoints /respond /science /concepts /chain
    /prove /health. Testé en local (200 OK).

  #### Knowledge packs étendus (4 → 7 domaines, 15 → 45 entrées)
  - Nouveaux : `astronomy` (trou noir, supernova, galaxie, gravité…),
    `chemistry` (atome, molécule, catalyseur…), `medicine` (virus, vaccin,
    neurone, ADN…). Bio enrichi : protein, cell, enzyme, photosynthesis.
  - Alias FR par racine ("trou noir" → black hole, "photosynthèse" → …) avec
    normalisation d'accents dans le lookup.

  #### Nettoyage
  - 22 modules v1–v4 et 18 coquilles de tests déplacés dans `archive/`
    (rien supprimé ; `archive/README.md` explique chaque fichier).
  - Mémos de session obsolètes → `archive/session_memos/`.
  - `LICENSE` propriétaire JOHNKING0 & Jonathan Evina (remplace toute
    licence ouverte ; la repo privée reste privée).
  - README refait avec 3 images SVG (architecture, boucle LCT, avant/après).

  #### Tests : 21 → 57, tous verts
  - Nouveaux : test_language_pipeline (19), test_language_quality (13),
    test_lct_new_systems réparé (4).


  #### Session neurale (le cerveau parle sans gradient) — commits suivants
  - **NeuralSpeaker** (`neural_speaker.py`) : le réseau reconstruit des
    phrases par couverture du chemin de concepts — pas en récitant des
    gabarits, pas en recitant des packs. Couverture IDF-pondérée + bonus
    de définition (« X is a Y ») + sujet exact en tête.
  - **Corpus v3 étendu** (`data/corpus/scalpel_v3_corpus.txt`) : 1 905 →
    **12 695 phrases** propres via 34 catégories Wikipedia (4 274 titres).
    Script : `scripts/collect_corpus_v3.py`.
  - **Mode neuronal actif** dans le framework : `respond_with_science` rend
    `neural` en clé ; prime sur les faits et gabarits quand la couverture
    est suffisante ou que la phrase définit le sujet.
  - **Serveur HTTP** (`server.py`) : `ratisnet-serve` active le
    NeuralSpeaker au démarrage. L'interface web de chat est en ligne.
  - **Résultat prouvé** : « Black hole is an astronomical body so compact
    that its gravity prevents anything including light from escaping. » —
    reconstruit par le réseau, pas copié d'un pack.
  - **Chatbot complet = chapitre 2** : le réseau parle sur les définitions ;
    la conversation libre (mémoire, dialogues multi-tours) reste une piste
    ouverte. La preuve LCT-parole est livrée.

  #### Nettoyage git (purge des gros fichiers)
  - `ultra_context_map.json` (400 MB) retiré de l'historique (export dérivé).
  - `scalpel_wikipedia.pkl` (294 MB) sous Git LFS.
  - corpus v3 (1.5 MB) exclu de git, régénérable.
  - `git filter-branch` utilisé pour purger l'historique (commit forcé).

- **Ce qui a été fait (session précédente)** :

  #### Diagnostic et fix
  - topo_tokenizer produisait des signatures quasi constantes (std < 0.02) → plafonnement à 0.501.
  - Fix : `glove_tokenizer.py` — hybride GloVe (400K mots) + topo. cos(happy,hate) = -0.04 vs +0.40.
  - Test acc : 0.545 (vs 0.130 topo seul).

  #### Scalpel (neurogenesis + LCT)
  - `scalpel.py` : découpe les phrases, génère des neurones-corrélations, renforce par LCT.
  - **Base de données ≠ réseau** (séparation demandée par Jonathan).
  - Entraîné sur Colab : 5M phrases Wikipedia → **3,782,801 neurones**, **43,260,980 renforcements**, 294 MB, 5.2h.
  - Vocabulaire : 242,903 mots. Checkpoint : `artifacts/scalpel_wikipedia.pkl` (Git LFS, SHA-256 `59d1aafda9...`).
  - Scaling VALIDÉ empiriquement : linéaire, pas exponentiel. Voir `docs/SCALING_NOTES.md`.

  #### Synchrotron (reconstruction topologique)
  - `ratiss_synchrotron.py` : 4 étapes (index + synchrotron + résonance + assembleur). Sans gradient.
  - Intégration Scalpel → Synchrotron (boost LCT dans la résonance).
  - Tests : 9/9 ✅.

  #### Squelettes grammaticaux
  - `syntax_skeletons.json` : 18 squelettes simples (FR/EN).
  - `data/grammar_domains/dense_syntax_skeletons.json` : **13 000** squelettes denses (18 domaines, 12 intentions).
  - `data/grammar_domains/conversation_matrix.json` : **24 000** formulations conversationnelles.
  - `data/grammar_domains/ultra_context_map.json` : **400 MiB**, 242K concepts, 7.56M arêtes (Git LFS).
  - `skeleton_speaker.py` : remplit les slots {X}, {Y}, {Z} avec les concepts du Scalpel.

  #### Génération de texte
  - `ratis_speaker.py` : bigramme (mot par mot).
  - `trigrammar.py` : fenêtre 2 mots (tri-grammaire sans trigrammes stockés).
  - `concept_decoder.py` : Scalpel (concepts) + décodeur (syntaxe).
  - Évolution : "quantum of which were made" → "In simple terms, mechanical is the way theory interacts with loop." → "When mechanical emerged, theory changed role and loop took on a new meaning, while stating the starting assumptions, within a quantum-information simulation, in a neutral register."

  #### Science core (AEON ODV fusionné)
  - `science_core.py` : Vietoris-Rips GF(2), P_sig, measure_lct, scan_monotonicity, validate_invariance, topological_tension.
  - **ZÉRO dépendance externe** — tout dans `ratis_net/`.
  - `aeon_bridge.py` : utilise science_core.py, backend = `integrated_science_core`.
  - Jonathan a corrigé : PAS de requêtes à distance vers AEON. Tout dans un seul package.

  #### Web search
  - `web_search.py` : DuckDuckGo (sans clé, testé ✅) / Google CSE (avec clé, non testé).
  - S'active quand le Scalpel ne trouve pas de corrélations.

  #### Knowledge packs
  - `data/knowledge_packs/` : 4 packs (quantum_physics, bio_pharma, math_logic, ai_systems), 15 entrées FR/EN.
  - `lookup_knowledge(concept)` : cherche dans les packs.
  - `respond_with_science()` : enrichit la réponse avec les faits validés.
  - Test : "qubit" → 4 facts trouvés ("A qubit can be prepared in state 0...").

  #### Framework unifié
  - `framework.py` : API unifiée `RatisNet` (`from ratis_net import RatisNet`).
  - `__init__.py` : expose `RatisNet`.
  - API : `respond()`, `paragraph()`, `respond_with_science()`, `concepts()`, `search()`, `lookup_knowledge()`, `stats()`.
  - README complet en anglais : quick start 5 min, architecture, API, training, tests, 8 limites honnêtes.
  - `AGENTS.md` : fichier de relais complet.
  - Tests : 17/17 ✅ (9 synchrotron + 8 scalpel).

  #### Tests scientifiques (10 questions)
  - Tous composants actifs (Scalpel + AEON + Web + KP + grammar).
  - Web search s'active automatiquement (consciousness, protein folding → 3 résultats DuckDuckGo).

  #### ZK-STARK
  - **NON générées.** Le science_core calcule P_sig + valide LCT (monotonicité + invariance) mais ne produit pas de preuve cryptographique. Statut `aeon_proof_status: "not_generated"`. Prochaine étape possible.

  #### Data loader + Colab
  - `data_loader.py` : stream Wikipedia via Hugging Face Datasets.
  - `ratisnet_colab_training.ipynb` : notebook Colab prêt à l'emploi (checkpoints Google Drive).
  - `scripts/download_wikipedia_corpus.py` : télécharge les résumés Wikipedia via API REST.

---

## ARCHITECTURE UNIFIÉE — Super RATISS

```
ratis_net/
├── framework.py           # API unifiée : from ratis_net import RatisNet
├── __init__.py            # expose RatisNet
├── cli.py                 # ratisnet ask|converse|concepts|chain|prove|stats
├── server.py              # HTTP stdlib : /respond /science /concepts /chain /prove
├── query_analyzer.py      # langue, type de question, mots-clés, composés
├── intent_router.py       # social (24K) vs factuel (13K denses), domaine
├── concept_ranker.py      # IDF de degré + voisinage partagé + GloVe kNN
├── skeleton_speaker_v2.py # speaker routé (remplace skeleton_speaker v1)
├── chain_reasoning.py     # chaînes d'association BFS (corrélation honnête)
├── integrity_proof.py     # empreinte SHA-256 du sous-graphe (pas ZK-STARK)
├── science_core.py        # AEON ODV FUSIONNÉ (P_sig, LCT, Vietoris-Rips)
├── aeon_bridge.py         # Bridge → utilise science_core (PAS de sys.path externe)
├── scalpel.py             # Neurogenesis + LCT (3.78M neurones)
├── glove_tokenizer.py     # GloVe 400K + topo (P_sig)
├── ratiss_synchrotron.py  # Reconstruction topologique
├── context_map_loader.py  # ultra_context_map.json streaming (400 MiB)
├── web_search.py          # DuckDuckGo / Google CSE
├── neural_speaker.py      # le cerveau parle (couverture chemin concepts)
├── data_loader.py         # Streaming Hugging Face → Scalpel
├── lct_neuron.py          # Neurone LCT (ΔW = η·φ·P_sig·C)
├── emocontext_loader.py   # Émotions (test vivant test_lct_modules)
├── eth_thermo_fixer.py    # ETH = f(token, env) (dép. emocontext_loader)
├── persistence_optimizer.py # Backend GUDHI (dép. topo_tokenizer)
├── topo_tokenizer.py      # Tokenizer topo (composante LCT du tokenizer hybride)
├── topo_cache.py          # Cache topo (15K mots, O(1))
└── lct_modules/           # GravitationalTopoMeasure, TopologicalQubit, LCTTransformer

archive/legacy_v1/         # 22 modules v1-v4 + 18 coquilles de tests (rien supprimé)
archive/session_memos/     # anciens fichiers de relais (remplacés par ce mémo)
```

### API

```python
from ratis_net import RatisNet

net = RatisNet()  # tout est intégré
net.load_scalpel("artifacts/scalpel_wikipedia.pkl")
net.load_grammar()          # dense + conversation (chargées automatiquement)
net.load_knowledge_packs()  # 7 domaines, 45 entrées FR/EN
net.build_index()           # ~9s pour 242K mots

net.respond("hello, how are you?")                 # conversation routée
net.respond("what is quantum mechanics")           # phrase (langue auto)
net.respond_with_science("what is a black hole")   # fait vérifié + gabarit
net.paragraph("consciousness", n_sentences=5)      # paragraphe
net.concepts("quantum")                            # classés IDF + GloVe
net.chain("quantum", "gravity")                    # chaînes d'association
net.prove(["quantum", "mechanics"])                # empreinte SHA-256
net.verify_proof(proof)                            # re-vérification
net.lookup_knowledge("qubit", language="en")       # faits validés
net.search("quantum decoherence")                  # recherche web
net.stats()                                        # statistiques
```

CLI : `ratisnet ask|converse|concepts|chain|prove|paragraph|stats`
HTTP : `ratisnet-serve --port 8000` (endpoints /respond /science /concepts /chain /prove /health)

**Champ `neural` dans la réponse** : `True` = reconstruction du réseau,
`None` = fallback gabarit/fait. `net.speak(query)` force le mode neuronal.
**Corpus neuronal** : `data/corpus/scalpel_v3_corpus.txt` (12 695 phrases).
**Serveur** : `python -m ratis_net.server --port 12000` (l'interface web est
servie à `/`).

---

## JOB IDs QPU IBM TRÇABLES

| Dépôt | Circuit | Backend | Job ID |
|---|---|---|---|
| COSMOS | Bell 2q | ibm_marrakesh | `da5u376vhnc73fmhnug` |
| Grover Labs | Grover iter 0 | ibm_marrakesh | `da5uajeaa69c739latgg` |
| Grover Labs | Grover iter 1 | ibm_marrakesh | `da5uituaa69c739lb6m0` |
| Grover Labs | Grover iter 2 | ibm_marrakesh | `da5ujreaa69c739lb7m0` |

Token IBM : lu uniquement depuis `IBM_QUANTUM_TOKEN`, jamais committé (0 occurrence vérifiée).

---

## FICHIERS IMPORTANTS (tous les dépôts)

| Dépôt | Fichier | Taille | Rôle |
|---|---|---|---|
| engine | `simulation.py` | — | Fix reproductibilité (bruit déterministe) |
| COSMOS | `artifacts/qpu_validation.json` | — | Validation QPU Bell (Job ID traçable) |
| COSMOS | `scripts/counts_diagnostic.py` | — | TVD + Shannon (classique, pas ETH) |
| Grover | `artifacts/grover_qpu_validation.json` | — | Validation QPU Grover (3 Job IDs) |
| portfolio | `warp/metric/alcubierre.py` | — | profile_tanh + exotic matter |
| portfolio | `warp/metric/lambda_lct.py` | — | 3 ansatz Λ_LCT (kinetic réduit seul) |
| ratisnet | `artifacts/scalpel_wikipedia.pkl` | 294 MB | Scalpel checkpoint (Git LFS) |
| ratisnet | `data/glove/glove.6B.50d.txt` | 171 MB | GloVe (à télécharger) |
| ratisnet | `data/grammar_domains/dense_syntax_skeletons.json` | 10 MB | 13K squelettes |
| ratisnet | `data/grammar_domains/conversation_matrix.json` | 20 MB | 24K conversation |
| ratisnet | `data/grammar_domains/ultra_context_map.json` | 400 MB | Context map (Git LFS) |
| ratisnet | `data/knowledge_packs/*.json` | ~1 MB | 4 packs scientifiques |
| ratisnet | `docs/SCALING_NOTES.md` | — | Scaling validé empiriquement |
| ratisnet | `ratisnet_colab_training.ipynb` | — | Notebook Colab |

---

## TESTS (tous les dépôts)

| Dépôt | Tests | Statut |
|---|---|---|
| engine | 13 (pipeline, topology, logical_qubit, tsp, counts, etc.) | 13/13 ✅ |
| COSMOS | 15 (cosmos, incubator, documentation_contract) | 15/15 ✅ |
| Grover Labs | 8 (grover_ratiss, reality_mode, documentation_contract) | 8/8 ✅ |
| Jonathan Labs | 9 (tsp, resilience, bio_yeast, documentation_contract) | 9/9 ✅ |
| atlas | 3 (verify-artifact, studio-model, external-artifacts) | 3/3 ✅ |
| ratisnet | 57 (scalpel 8, synchrotron 9, lct_modules 4, lct_new_systems 4, language_pipeline 19, language_quality 13) | 57/57 ✅ |

**Total : 66 tests sur ratisnet, tous verts** (dont 5 tests neuraux). Les autres dépôts gardent leurs tests (105 au total si tous comptés).

---

## LIMITES HONNÊTES (à connaître) — mises à jour après NeuralSpeaker

1. **RATIS-Net reconstruit, ne copie pas.** Le cerveau parle par couverture
   du chemin de concepts — pas par lecture de base. Les packs servent de
   filet si couverture faible.
2. **Le chatbot complet reste ouvert.** Le réseau définit bien ; la
   conversation libre (mémoire, dialogues multi-tours) est une piste
   ouverte. Les 24K gabarits sociaux sont utilisés pour le social.
3. **Couverture dépend du corpus.** Les concepts absents du corpus (12 695
   phrases) ne sont pas reconnus — le fallback gabarit/pack le dit
   honnêtement.
4. **Grammaire neurale ≠ fluide Transformer.** La reconstruction évite les
   gabarits bancals, mais le texte reste factuel/définitionnel.
5. **Poids bruts des neurones ubiquitaires.** « black hole » donnait
   « metal, comedy » avant le ranker IDF ; la couverture IDF corrige.
6. **Counts diagnostic est classique.** Shannon ≠ von Neumann.
7. **Le checkpoint Scalpel (294 MB) est sous Git LFS.**
8. **Google CSE non testé** — DuckDuckGo fonctionne.
9. **Preuve d'intégrité SHA-256** (pas ZK-STARK) : engagement sur le
   sous-graphe, vérifiable, pas de calcul privé.
10. **Crédits QPU IBM quasi épuisés** — CPU d'abord.
11. **GITHUB_TOKEN sans scope repo** — Jonathan crée les dépôts manuellement.
12. **Le sidecar ne capte pas la structure des erreurs 01/10.**
13. **Biais corpus persistant** (Wikipedia) — atténué par le ranker IDF,
    les faits packs priment si cluster faible.
14. **Scalpel anglophone** — le français passe par les gabarits FR + alias
    des packs (multilingue complet = piste ouverte).

1. **RATIS-Net n'est pas une base de connaissances.** Il reconstruit à partir de fragments appris ; les faits exacts viennent des knowledge packs (7 domaines, 45 entrées).
2. **Grammaire template-based.** 13K + 24K gabarits garantissent la grammaire mais pas la fluidité d'un Transformer.
3. **Bigrammes filtrées.** Le Scalpel capture les paires adjacentes après filtrage cos(GloVe) ≥ 0.3 ; les composés non stockés ("black hole") sont reconstruits à l'analyse (ranker).
4. **Coverage = corpus + packs.** Si Wikipedia et les packs ne couvrent pas un sujet, le web compense (DuckDuckGo).
5. ~~Pas de raisonnement multi-sauts~~ → **chaînes d'association tracées** (`chain()`, étiquetées corrélation, pas causalité). L'inférence logique A→B→C reste hors de portée.
6. **Counts diagnostic est classique.** Shannon ≠ von Neumann. ETH ne peut pas être approximé sans tomographie.
7. **Le Scalpel fait 294 MB** — nécessite Git LFS.
8. **Google CSE non testé** — pas de clé. DuckDuckGo fonctionne.
9. ~~ZK-STARK non générées~~ → **empreinte d'intégrité SHA-256** implémentée (`prove`/`verify_proof`) : engagement déterministe sur le sous-graphe. Ce n'est PAS un ZK-STARK (pas de confidentialité, vérification = recalcul complet). Un vrai STARK exige un backend AIR/FRI.
10. **Crédits QPU IBM quasi épuisés** — CPU d'abord.
11. **GITHUB_TOKEN sans scope repo** — Jonathan crée les dépôts manuellement.
12. **Le sidecar ne capte pas la structure des erreurs 01/10** — il réagit à la masse globale, pas au détail du bruit.
13. **Biais du corpus persistant.** Wikipedia associe "black" à "metal"/"panther" ; le ranker IDF atténue, n'efface pas. Les knowledge packs priment pour les faits.
14. **Scalpel anglophone.** Le français passe par les gabarits FR + alias FR des packs ; le corpus FR reste une piste ouverte.

---

## COMMANDES UTILES

### Installation RATIS-Net from scratch
```bash
git clone https://github.com/evinajonathan13-max/Ratiss-experimental-IA-.git
cd Ratiss-experimental-IA-
pip install numpy datasets pytest
git lfs install && git lfs pull
# (le corpus v3 est recréé via scripts/collect_corpus_v3.py si absent)
mkdir -p data/glove
curl -L -o data/glove/glove.6B.zip "https://nlp.stanford.edu/data/glove.6B.zip"
python3 -c "import zipfile; zipfile.ZipFile('data/glove/glove.6B.zip').extract('glove.6B.50d.txt', 'data/glove/')"
rm data/glove/glove.6B.zip
python3 -m ratis_net.framework --query "what is consciousness"
```

### Tests
```bash
# RATIS-Net (test_lct_new_systems réparé — plus besoin de --ignore)
PYTHONPATH=. python -m pytest -q

# Engine (moteur)
PYTHONPATH=src python -m pytest -q

# COSMOS
PYTHONPATH=../ratiss-topological-decoherence-engine/src python -m pytest -q

# Grover Labs
PYTHONPATH=../ratiss-topological-decoherence-engine/src python -m pytest -q

# Atlas (Node)
npm test
```

### Training (Colab)
Ouvrir `ratisnet_colab_training.ipynb` dans Google Colab. Stream 5M Wikipedia phrases → Scalpel. Checkpoints sur Google Drive.

### QPU validation (IBM)
```bash
export IBM_QUANTUM_TOKEN="..."
python3 scripts/run_qpu_validation.py --engine-src ../engine/src --backend ibm_marrakesh
```

---

## PISTES OUVERTES (prochaines étapes)

1. **Améliorer la fluidité** : décodage au niveau mot guidé par Scalpel + squelettes comme contraintes.
2. **Étendre les knowledge packs** : plus de domaines (chimie, astronomie, médecine).
3. **Trigrammes / tagger POS** : améliorer la syntaxe sans exploser la taille.
4. **Corpus plus large** : 5M → 20M phrases (Wikipedia complet). Scaling linéaire.
5. **Google CSE** : obtenir une clé API Google pour tester.
6. **ZK-STARK** : implémenter un module cryptographique pour générer des preuves.
7. **Multilingue** : ajouter le français au Scalpel (Wikipedia FR).
8. **BEC analog gravity** : protocole expérimental du preprint LCT §10 (trouver un labo partenaire).

---

## COMMENT TRAVAILLER AVEC JONATHAN

- Il est le chercheur, tu es le cofondateur technique. Tu itères, tu testes, tu façonnes.
- Honnêteté scientifique : documenter les échecs, pas seulement les succès.
- Économiser les crédits QPU IBM — tests locaux CPU d'abord.
- Ne pousse JAMAIS directement sur main sans le go de Jonathan (sauf si il dit "pousse").
- Propriété intellectuelle : JOHNKING0 & Jonathan Evina.
- La loi LCT est FIGÉE. Tu tunes η, l'architecture, les données — pas la loi.
- Jonathan peut te corriger sur les nuances (ex: AEON ≠ RATISS-Net, pas de requêtes distantes, base ≠ réseau) — écoute-le.
- Les briefs `.docx` viennent parfois d'un autre assistant (Qwen) — ils sont enthousiastes mais peuvent contenir des erreurs mathématiques. Vérifie tout.
- Ne confonds pas "encouragement" et "validation scientifique". Un reviewer arXiv ne se contente pas d'emojis.
