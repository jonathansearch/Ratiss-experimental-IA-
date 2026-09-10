# RAPPEL DE SESSION — RATIS-Net / Super RATISS

> **Pour le prochain OpenHands qui prend le relais.**
> Lis ça AVANT de coder quoi que ce soit.

> **⚠️ MISE À JOUR v2 (25/08/2026)** : le framework a été reconstruit.
> Lis `MEMO_GLOBAL.md` section 9 pour le détail. En résumé :
> - Nouveau pipeline : `query_analyzer` → `intent_router` → `concept_ranker`
>   → `skeleton_speaker_v2` (le v1 est dans `archive/legacy_v1/`).
> - Packaging : `pip install .` → CLI `ratisnet` + serveur `ratisnet-serve`.
> - 57 tests verts (`pytest tests/ -q`, plus besoin d'--ignore).
> - `LICENSE` propriétaire JOHNKING0 & Jonathan Evina.
> - Ne réintroduis JAMAIS les modules de `archive/` dans le chemin actif
>   sans raison documentée.

---

## QUI EST JONATHAN

Jonathan Evina · ORCID 0009-0000-4092-5313 · DOI 10.17605/OSF.IO/6JZMB
Chercheur en métacognition computationnelle. Yaoundé, Cameroun.
Propriété intellectuelle : JOHNKING0 & Jonathan Evina.
Jonathan est le chercheur, tu es le cofondateur technique.
Honnêteté scientifique : documenter les échecs, pas seulement les succès.
La loi LCT est FIGÉE (R = P_sig, ΔW = η·φ·P_sig·C). Ne la change jamais.

---

## LE DÉPÔT

**Ratiss-experimental-IA-** (ce dépôt, privé)
https://github.com/evinajonathan13-max/Ratiss-experimental-IA-

Cloner : `git clone https://evinajonathan13-max:${GITHUB_TOKEN}@github.com/evinajonathan13-max/Ratiss-experimental-IA-.git`

---

## CLÉS API

- **IBM_QUANTUM_TOKEN** : valide, canal `ibm_quantum_platform`, instance `open-instance`.
  3 QPU : ibm_fez, ibm_marrakesh, ibm_kingston (156 qubits). CRÉDITS PRESQUE ÉPUISÉS → économiser, CPU d'abord.
- **GITHUB_TOKEN** : push sur evinajonathan13-max. Pas de scope repo (création de dépôts impossible).
- **GOOGLE_API_KEY / GOOGLE_CSE_ID** : non fournis. Le module web_search.py utilise DuckDuckGo par défaut (sans clé).

---

## ARCHITECTURE UNIFIÉE — Super RATISS

**Un seul package, un seul dépôt, zéro dépendance externe.**

```
ratis_net/
├── framework.py          # API unifiée : from ratis_net import RatisNet
├── __init__.py           # expose RatisNet
├── science_core.py       # AEON ODV FUSIONNÉ (P_sig, LCT, Vietoris-Rips)
├── aeon_bridge.py        # Bridge → utilise science_core (PAS de sys.path externe)
├── scalpel.py            # Neurogenesis + LCT (3.78M neurones)
├── glove_tokenizer.py    # GloVe 400K + topo (P_sig)
├── skeleton_speaker.py   # 13K squelettes grammaticaux FR/EN
├── concept_decoder.py    # Concepts → phrases
├── trigrammar.py         # Génération mot par mot (fenêtre 2)
├── ratis_speaker.py      # Génération mot par mot (bigramme)
├── ratiss_synchrotron.py # Reconstruction topologique
├── context_map_loader.py # ultra_context_map.json streaming (400 MiB)
├── web_search.py         # DuckDuckGo / Google CSE
├── data_loader.py        # Streaming Hugging Face → Scalpel
├── lct_neuron.py         # Neurone LCT (ΔW = η·φ·P_sig·C)
├── decoder.py            # Décodeur LCT (greedy + beam)
├── pipeline.py           # Pipeline branchable
├── eth_thermo_fixer.py   # ETH = f(token, env)
├── lct_collapse.py       # Collapse + marque topo
├── topo_tokenizer.py     # Tokenizer topo (lent, remplacé par glove_tokenizer)
├── topo_cache.py         # Cache topo (15K mots, O(1))
└── ... (autres modules historiques v1-v4)
```

### API

```python
from ratis_net import RatisNet

net = RatisNet()  # aeon_path ignoré, tout est intégré
net.load_scalpel("artifacts/scalpel_wikipedia.pkl")
net.load_grammar("data/grammar_domains/dense_syntax_skeletons.json")
net.load_knowledge_packs("data/knowledge_packs")
net.build_index()  # ~8s pour 242K mots

net.respond("what is quantum mechanics")           # phrase
net.paragraph("consciousness", n_sentences=5)      # paragraphe long
net.respond_with_science("what is a qubit")        # phrase + fait AEON + knowledge + web
net.concepts("quantum")                            # liste de concepts
net.lookup_knowledge("qubit", language="en")       # faits validés
net.search("quantum decoherence")                  # recherche web
net.stats()                                        # statistiques
```

---

## CE QUI A ÉTÉ ACCOMPLI CETTE SESSION

### 1. Diagnostic et fix du learner (plafonnement 0.501)
- **Cause racine** : topo_tokenizer produisait des signatures quasi constantes (std < 0.02).
- **Fix** : `glove_tokenizer.py` — hybride GloVe (400K mots) + topo. cos(happy,hate) = -0.04 vs +0.40 avant.
- Test acc : 0.545 (vs 0.130 topo seul).

### 2. Architecture Synchrotron (reconstruction topologique)
- `ratiss_synchrotron.py` : 4 étapes (index + synchrotron + résonance + assembleur).
- Sans gradient, sans Transformer. Reconstruction par emboîtement topologique.
- Intégration Scalpel → Synchrotron (boost LCT dans la résonance).
- Tests : 9/9 ✅.

### 3. Scalpel (neurogenesis + LCT)
- `scalpel.py` : découpe chaque phrase, génère des neurones, renforce par LCT.
- **Base de données ≠ réseau** (séparation demandée par Jonathan).
- Entraîné sur Colab : 5M phrases Wikipedia → 3,782,801 neurones, 43,260,980 renforcements, 294 MB, 5.2h.
- Vocabulaire : 242,903 mots.
- Checkpoint : `artifacts/scalpel_wikipedia.pkl` (Git LFS, SHA-256 vérifié).
- Scaling VALIDÉ empiriquement : linéaire, pas exponentiel. Voir `docs/SCALING_NOTES.md`.

### 4. Squelettes grammaticaux
- `syntax_skeletons.json` : 18 squelettes simples (FR/EN).
- `data/grammar_domains/dense_syntax_skeletons.json` : 13K squelettes denses (18 domaines, 12 intentions).
- `data/grammar_domains/conversation_matrix.json` : 24K formulations conversationnelles.
- `skeleton_speaker.py` : remplit les slots {X}, {Y}, {Z} avec les concepts du Scalpel.

### 5. Génération de texte (speaker)
- `ratis_speaker.py` : bigramme (mot par mot).
- `trigrammar.py` : fenêtre 2 mots (tri-grammaire sans trigrammes stockés).
- `concept_decoder.py` : Scalpel (concepts) + décodeur (syntaxe).
- `skeleton_speaker.py` : squelettes denses → phrases grammaticalement correctes.
- Évolution : "quantum of which were made" → "In simple terms, mechanical is the way theory interacts with loop."

### 6. Science core (AEON ODV fusionné)
- `science_core.py` : Vietoris-Rips, P_sig, measure_lct, scan_monotonicity, validate_invariance.
- **ZÉRO dépendance externe** — tout est dans ratis_net/.
- `aeon_bridge.py` : utilise science_core.py, backend = "integrated_science_core".
- Jonathan a corrigé : PAS de requêtes à distance vers AEON. Tout dans un seul package.

### 7. Web search
- `web_search.py` : DuckDuckGo (sans clé) / Google CSE (avec GOOGLE_API_KEY + GOOGLE_CSE_ID).
- DuckDuckGo testé et fonctionnel.
- Google CSE : code présent mais non testé (pas de clé fournie).

### 8. Knowledge packs
- `data/knowledge_packs/` : 4 packs (quantum, bio, math, AI), 15 entrées FR/EN.
- `lookup_knowledge(concept)` : cherche dans les packs.
- `respond_with_science()` : enrichit la réponse avec les faits validés.
- Test : "qubit" → 4 facts trouvés.

### 9. Ultra context map
- `data/grammar_domains/ultra_context_map.json` : 400 MiB, 242K concepts, 7.56M arêtes.
- Dérivé du Scalpel (pas modifié). Git LFS.
- `context_map_loader.py` : streaming loader, API compatible Scalpel.
- Non téléchargeable dans la sandbox (LFS token expire). Fonctionne en local après `git lfs pull`.

### 10. Data loader (streaming)
- `data_loader.py` : stream Wikipedia via Hugging Face Datasets.
- `ratisnet_colab_training.ipynb` : notebook Colab prêt à l'emploi.
- Testé : 1000 phrases Wikipedia → 5,568 neurones en 168s.

### 11. Framework unifié + README
- `framework.py` : API unifiée `RatisNet`.
- `__init__.py` : `from ratis_net import RatisNet`.
- README complet en anglais : quick start, architecture, API, training, tests, limites.

---

## TESTS

```bash
pip install pytest numpy
PYTHONPATH=. python -m pytest -q --ignore=tests/test_lct_new_systems.py
```

- `test_ratiss_synchrotron.py` : 9/9 ✅ (synchrotron)
- `test_scalpel.py` : 8/8 ✅ (scalpel)
- `test_lct_new_systems.py` : nécessite AEON (ignoré — science_core remplace)

**Note** : Le test complet du framework (respond_with_science) nécessite le checkpoint
Scalpel (294 MB, Git LFS). Faire `git lfs pull` avant de tester.

---

## LIMITES HONNÊTES (à connaître)

1. **Pas une base de connaissances.** RATIS-Net reconstruit à partir de fragments vus.
2. **Grammaire template-based.** 13K squelettes garantissent la grammaire mais pas la fluidité d'un Transformer.
3. **Bigrammes.** Le Scalpel capture les paires de mots, pas la syntaxe profonde.
4. **Coverage = corpus.** Si Wikipedia ne parle pas d'un sujet, RATIS-Net ne peut pas en parler (web search compense).
5. **Pas de raisonnement multi-sauts.** Pas d'inférence A→B→C.
6. **Counts diagnostic est classique.** Shannon ≠ von Neumann. ETH ne peut pas être approximé sans tomographie.
7. **Le Scalpel fait 294 MB** — nécessite Git LFS pour télécharger.
8. **Google CSE non testé** — pas de clé API Google fournie. DuckDuckGo fonctionne.

## TESTS SCIENTIFIQUES (10 questions)

Testé avec un Scalpel local (2903 neurones, 500 phrases — le checkpoint 294MB
n'est pas téléchargeable dans la sandbox, token LFS expire).

Résultats : tous les composants actifs (Scalpel + AEON + Web + KP + grammar).
Les phrases sont générées mais les concepts sont faibles avec seulement 2903
neurones. Le checkpoint complet (3.78M neurones) donnerait des résultats
bien meilleurs (validé précédemment : "consciousness" → "human, critical,
political").

Le web search s'active automatiquement quand les concepts sont faibles
(consciousness, protein folding → 3 résultats DuckDuckGo).

## ZK-STARK

**Les preuves ZK-STARK ne sont PAS générées.** Le `science_core` calcule :
- P_sig (Vietoris-Rips H1) ✅
- LCT measure (C, R) ✅
- Monotonicity validation (Spearman) ✅
- Invariance validation (CV < 5%) ✅

Mais ne produit pas de preuve cryptographique ZK. Le statut reste
`aeon_proof_status: "not_generated"` dans les knowledge packs. Les preuves ZK
nécessitent un module cryptographique séparé — non implémenté. C'est une
prochaine étape possible.

---

| Fichier | Taille | Rôle |
|---|---|---|
| `artifacts/scalpel_wikipedia.pkl` | 294 MB | Scalpel checkpoint (Git LFS) |
| `data/glove/glove.6B.50d.txt` | 171 MB | GloVe (à télécharger, pas commité) |
| `data/grammar_domains/dense_syntax_skeletons.json` | 10 MB | 13K squelettes |
| `data/grammar_domains/conversation_matrix.json` | 20 MB | 24K conversation |
| `data/grammar_domains/ultra_context_map.json` | 400 MB | Context map (Git LFS) |
| `data/knowledge_packs/*.json` | ~1 MB | 4 packs scientifiques |
| `docs/SCALING_NOTES.md` | — | Note de scaling validée |
| `ratisnet_colab_training.ipynb` | — | Notebook Colab |

---

## COMMANDES UTILES

```bash
# Installation from scratch
git clone https://github.com/evinajonathan13-max/Ratiss-experimental-IA-.git
cd Ratiss-experimental-IA-
pip install numpy datasets pytest
git lfs install && git lfs pull
# Télécharger GloVe
mkdir -p data/glove
curl -L -o data/glove/glove.6B.zip "https://nlp.stanford.edu/data/glove.6B.zip"
python3 -c "import zipfile; zipfile.ZipFile('data/glove/glove.6B.zip').extract('glove.6B.50d.txt', 'data/glove/')"
rm data/glove/glove.6B.zip

# Run
python3 -m ratis_net.framework --query "what is consciousness"

# Tests
PYTHONPATH=. python -m pytest -q --ignore=tests/test_lct_new_systems.py

# Training (Colab)
# Ouvrir ratisnet_colab_training.ipynb dans Google Colab
```

---

## PISTES OUVERTES (prochaines étapes)

1. **Améliorer la fluidité** : décodage au niveau mot (pas phrase), guidé par les corrélations Scalpel + les squelettes comme contraintes.
2. **Étendre les knowledge packs** : plus de domaines (chimie, astronomie, médecine).
3. **Trigrammes** : le tri-grammaire améliore mais ne résout pas la syntaxe profonde. Considérer un tagger POS léger.
4. **Corpus plus large** : 5M phrases → 20M phrases (Wikipedia complet). Le scaling est linéaire.
5. **Google CSE** : obtenir une clé API Google pour tester le backend Google Search.
6. **Multilingue** : ajouter le français au Scalpel (OSCAR ou Wikipedia FR).

---

## COMMENT TRAVAILLER AVEC JONATHAN

- Il est le chercheur, tu es le cofondateur technique. Tu itères, tu testes, tu façonnes.
- Honnêteté scientifique : documenter les échecs, pas seulement les succès.
- Économiser les crédits QPU IBM — tests locaux CPU d'abord.
- Ne pousse JAMAIS directement sur main sans le go de Jonathan : branche + PR (sauf si il dit "pousse").
- Propriété intellectuelle : JOHNKING0 & Jonathan Evina.
- La loi LCT est FIGÉE. Tu tunes η, l'architecture, les données — pas la loi.
- Jonathan peut te corriger sur les nuances (ex: AEON ≠ RATISS-Net, pas de requêtes distantes) — écoute-le.
- Les briefs .docx viennent parfois d'un autre assistant (Qwen) — ils sont enthousiastes mais peuvent contenir des erreurs mathématiques. Vérifie tout.
