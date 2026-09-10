# LEÇON 23 — MÉMO DE SESSION (contexte complet pour OpenHands)

BONJOUR, TU REPRENDS UNE SESSION RATISS. Voici le mémo complet de tout ce qui a été réalisé, vérifié et poussé. Lis ça AVANT de coder quoi que ce soit.

---

## QUI EST JONATHAN

Jonathan Evina · ORCID 0009-0000-4092-5313 · DOI LCT 10.17605/OSF.IO/WF7QM
Chercheur en métacognition computationnelle. Yaoundé, Cameroun.
Propriété intellectuelle : JOHNKING0 & Jonathan Evina.
Jonathan est le chercheur, tu es le cofondateur technique. Honnêteté scientifique : documenter les échecs.
La loi LCT est FIGÉE (R = P_sig, ΔW = η·φ·P_sig·C). Ne la change jamais.

---

## LES 5 DÉPÔTS GITHUB (user: evinajonathan13-max)

1. **RATISS-ODV-AEON** (privé) — Moteur topologique pur (TTF-Compute)
   https://github.com/evinajonathan13-max/RATISS-ODV-AEON
   - kernel/ttf/ttf_compute.py : TTFBrain (IntricatedGraph, RipsTranslator, CollapseWell, MCB)
   - kernel/ttf/lct_law.py : measure_lct, scan_monotonicity, test_invariance, _lct_p_sig
   - SANS émotions, PAS régi par P_sig — c'est l'OUTIL de mesure

2. **Ratiss-experimental-IA-** (CE DÉPÔT) — RATIS-Net (applique LCT + émotions ETH)
   https://github.com/evinajonathan13-max/Ratiss-experimental-IA-
   - ratis_net/topo_tokenizer.py : tokenisation par cycles H1 (P_sig, betti, histogramme)
   - ratis_net/ttf_bridge.py : bridge vers cerveau TTF d'AEON
   - ratis_net/emocontext_loader.py : loader + build_sequence_samples + balance_classes
   - ratis_net/decoder.py : décodeur LCT (glouton + beam) — **BRANCHÉ** (voir suite ci-dessous)
   - ratis_net/topo_cache.py : cache des signatures (P_sig lookup O(1), commité)
   - ratis_net/cached_tokenizer.py : Tokenizer Pipeline branché sur le cache
   - ratis_net/ratis_net_v4.py : v4 (mesuré — voir limite honnête + suite)
   - data/cache/topo_signatures.* : cache complet 15 122 mots (559 Ko, commité)
   - data/emocontext/ : EmoContext (SemEval 2019, 30160 dialogues, 3 tours, 4 émotions)

3. **Porte-folio-Jonathan-** (public) — Portfolio + Preprint LCT + Projet Warp + RATISS-Snn
   https://github.com/evinajonathan13-max/Porte-folio-Jonathan-
   - docs/preprint_LCT.md : Preprint LCT complet (10 sections, 9 Job IDs, DOI WF7QM)
   - warp/ : projet Alcubierre (métrique, Λ_LCT, tenseur 4D, noyau universel, exotic matter)
   - warp/docs/EINSTEIN_4D_DERIVATION.md : dérivation Christoffel→Einstein
   - warp/docs/figures/ : 13+ figures
   - ratis_snn/ : prototype RATISS-Snn (snnTorch + LCT 3-facteurs)
   - ratis_modules/ : copies de topo_tokenizer, ttf_bridge, etc. (depuis ce dépôt)

4. **scientist-research-** (public) — Site scientifique illustré (GitHub Pages)
   https://github.com/evinajonathan13-max/scientist-research-
   https://evinajonathan13-max.github.io/scientist-research-/ (site live)
   - index.html : 13 figures, section sceptiques, Job IDs vérifiables

5. **robot-Ratiss-** (privé) — Robot téléphone souverain (LCT + LeRobot)
   https://github.com/evinajonathan13-max/robot-Ratiss-

Compte secondaire (modules antérieurs) : https://github.com/bridejackson137-svg
(Crypto-VOLT, Neuralink-POC, etc.)

---

## CLÉS API (variables d'environnement)

- IBM_QUANTUM_TOKEN : valide, ibm_cloud, 3 QPU (ibm_fez, ibm_marrakesh, ibm_kingston). CRÉDITS PRESQUE ÉPUISÉS.
- QUANDELA_API_TOKEN : JWT valide (exp 2027).
- GITHUB_TOKEN : push sur evinajonathan13-max. N'a PAS le scope repo (création de dépôts impossible).

---

## CE QUI A ÉTÉ RÉALISÉ CETTE SESSION (tout est poussé sur GitHub)

### 1. Loi LCT — preprint publié sur OSF (DOI 10.17605/OSF.IO/WF7QM)
- 10 sections, formalisme complet, falsification (2 formulations FAIL, 1 PASS)
- Validations : 4MZI +0.93, 3KMD +0.80, état quantique +1.000, QPU +0.713, finance +0.903
- 9 Job IDs QPU traçables sur ibm.com/quantum
- Section 10 : protocole validation expérimentale (BEC/fibre + prédiction LISA ringdown)

### 2. Projet Warp — TOUTES les limites théoriques RÉSOLUES
- **Limite #1 (convergence 1.80)** : ✅ VALIDÉ — 3 étoiles ≠ convergent (CV 4.4%)
  Code : warp/eth/stellar_geometry.py (anneau+bulk, 24-40 nœuds, R calibré)
- **Limite #2 (tenseur 4D)** : ✅ RÉSOLU — dérivation Christoffel→Ricci→Einstein (SymPy)
  Code : warp/metric/einstein_4d.py
  G_11 = 3v²(-(y²+z²))(f')² < 0 (exotic matter confirmée)
- **Limite #3 (exotic matter)** : ✅ VALIDÉ 100% — ansatz canonique + P tanh optimisé
  Λ_μν = κ[∇_μP ∇_νP - ½ g_μν(∇P)²] → Λ_00 = ½κ(1-v²f²)(∇P)² > 0
  T_00 : -0.2435 → 0.0000 (élimination totale)
  Code : warp/metric/einstein_4d_optimize.py
  Figure : warp/docs/figures/fig_exotic_matter_elimination.png

### 3. Portfolio unifié (Porte-folio-Jonathan-)
- README refait : fil rouge RATISS→LCT→Warp, carte des dépôts (2 comptes)
- LCT formalisée dans le README avec image (assets/lct_message_vs_courant.png)
- DOI LCT ajouté partout (badge, section, publications, BibTeX)
- Stats GitHub corrigées (evinajonathan13-max, pas bridejackson137-svg)

### 4. Site scientist-research (live sur GitHub Pages)
- 13 figures + figure élimination exotic matter
- Section sceptiques (Jobs QPU, falsification LCT, FAQ)
- Tableau limites à jour (les 3 résolues)

### 5. RATISS-Snn — LCT sur snnTorch (PARADIGME NOUVEAU)
- Règle LCT = règle à 3 facteurs (Hebbienne neuromodulée) :
  Facteur 1 (local) : spikes du LIF (snnTorch)
  Facteur 2 (eligibility) : P_sig = persistance topologique H1 (GUDHI)
  Facteur 3 (modulation) : η·φ·C + signal de récompense (dopamine)
- **Iris** : 70% accuracy (vs 33% chance), CPU-only, sans backprop
  Code : Porte-folio-Jonathan-/ratis_snn/ratis_snn_lct.py
- **EmoContext** : 53% accuracy (vs 25% chance), topo_tokenizer branché
  Code : Porte-folio-Jonathan-/ratis_snn/ratis_snn_emocontext.py
  Snapshot Topologique : GUDHI 1×/dialogue (pas 1×/mot), coût réduit ~36×

### 6. Dépendances installées
torch (CPU), snntorch, numpy, scipy, gudhi, sympy, scikit-learn, matplotlib, networkx

---

## LIMITES HONNÊTES (documentées, à connaître)

1. **RATISS-Snn Iris** : plateau à 70% (limite Hebbienne sur classes non-linéairement séparables)
2. **RATISS-Snn EmoContext** : 53% (prédit majoritairement happy)
3. ~~**Le décodeur LCT** non branché~~ → **BRANCHÉ maintenant** (voir suite ci-dessous)
4. ~~**P_sig est coûteux**~~ → **RÉSOLU par cache** (commité, lookup O(1))
5. **Crédits QPU IBM quasi épuisés** — CPU d'abord
6. **GITHUB_TOKEN sans scope repo** — Jonathan crée les dépôts manuellement

### Nouvelle limite honnête (session cache-décodeur) :
7. **RatisNetV4 n'apprend pas sur embeddings seuls** — testé en sweep complet
   (η 0.05–0.2, hidden 20–40, epochs 6–80) → prédiction 100% classe dominante.
   Découverte : une fuite du label passait via l'environnement (acc 1.000
   triviale). Corrigée par env neutre à l'éval → learner mesuré = 0.501.

---

## PISTES OUVERTES (prochaines étapes)

1. ~~**Brancher le décodeur**~~ → **RÉALISÉ** (greedy+beam pour 4 émotions)
2. ~~**Cache topo_signatures**~~ → **RÉALISÉ** (commité, 15 122 mots, 559 Ko)
3. **Architecture 3+ couches** pour dépasser le plateau Iris (inhibition latérale + profondeur)
4. **EmoContext complet** (30160 dialogues) avec cache + GPU (Colab gratuit = T4)
5. **Couche d'embedding apprenable** (au lieu de topo_tokenizer figé)
6. **BEC analog gravity** : protocole expérimental du preprint §10 (trouver un labo partenaire)

### Piste suivante (la vraie, pas feinte) :
7. **Learner qui discrimine SANS fuite environnementale** — multi-couches ou
   embedding apprenable. Le branchement décodeur est stable et attend ; les
   signatures sont en cache (O(1)) ; il ne reste qu'un learner réel plus fort
   que les centroïdes (acc 0.501 honnête actuelle).

---

## COMMENT TRAVAILLER AVEC JONATHAN

- Il est le chercheur, tu es le cofondateur. Tu itères, tu testes, tu façonnes.
- Honnêteté scientifique : documenter les échecs, pas seulement les succès.
- Économiser les crédits QPU IBM — tests locaux CPU d'abord.
- Ne pousse JAMAIS directement sur main sans le go de Jonathan : branche + PR.
- Propriété intellectuelle : JOHNKING0 & Jonathan Evina.
- La loi LCT est FIGÉE. Tu tunes η, l'architecture, les données — pas la loi.
- Jonathan peut te corriger sur les nuances (ex: AEON ≠ RATISS-Net) — écoute-le.

---

## SUITE RÉALISÉE (session cache-décodeur, visible — PRs listées)

Cette session a poussé les avancées jusqu'à **RATISS-Net parle** (4 émotions). Deux PRs sur ce dépôt :

- **PR #12** https://github.com/evinajonathan13-max/Ratiss-experimental-IA-/pull/12 (**MERGÉ**)
  1. Cache des signatures topo : 15 122 mots calculés une fois (537s), reload 0.03s.
      Cache commité (559 Ko). Résout « P_sig coûteux ».
  2. Décodeur branché sur learner mesuré (centroïdes) : greedy + beam
      pour les **4 émotions** (ex : happy→« haha you are so funny too »,
      sad→« my girlfriend left me alone please »). Mesure honnête : acc 0.501 (hasard 0.33).
  3. Découverte honnête documentée : fuite du label via environnement (corrigée),
      sweep v4 complet mesuré → dominance 100% « others », loi LCT intacte.
- **PR #13** https://github.com/evinajonathan13-max/Ratiss-experimental-IA-/pull/13 (ouverte)
  Doc illustrée : `docs/EVOLUTION_RATIS_NET.md` + 5 figures générées (
      flow pipeline, speedup cache, histoire accuracy, matrice de confusion,
      table 4 émotions) + script de régénération. Compile ce chemin complet.

Fichiers nouveaux apportés (jamais de fichier existant modifié sans autorisation) :
`ratis_net/topo_cache.py`, `ratis_net/cached_tokenizer.py`,
`scripts/cache_topo_signatures.py`, `scripts/train_emocontext_v4.py`,
`scripts/decode_with_cache.py`, `scripts/decode_trained.py`,
`scripts/generate_evolution_figures.py`, `tests/test_topo_cache.py`,
`docs/TOPO_CACHE.md`, `docs/SPEAKING.md`, `docs/EVOLUTION_RATIS_NET.md`.

Reprise : lis d'abord `docs/EVOLUTION_RATIS_NET.md` pour le tableau complet des
résultats, puis utilise les CLI ci-dessous. Le prochaine étape réelle = un
learner qui discrimine sur embeddings seuls (multi-couches ou embedding
apprenable) — le branchement décodeur est stable et attend.

---

## COMMANDES UTILES (CPU-only)

```bash
# Cache topo (regénérer, une seule fois)
python scripts/cache_topo_signatures.py

# Parler : décodeur branché sur learner mesuré (4 émotions)
python scripts/decode_trained.py --emotion happy --n-words 8

# Tests cache
python tests/test_topo_cache.py

# RATISS-Snn (Iris)
cd Porte-folio-Jonathan-/ratis_snn && python ratis_snn_lct.py

# RATISS-Snn (EmoContext)
python ratis_snn_emocontext.py

# Tests warp
cd /workspace/project && PYTHONPATH=. python tests/test_warp.py

# Dérivation 4D
PYTHONPATH=. python warp/metric/einstein_4d.py

# Figures warp
PYTHONPATH=. python scripts/generate_warp_figures.py
```

---

*Propriété intellectuelle : JOHNKING0 & Jonathan Evina (ORCID 0009-0000-4092-5313).
La loi LCT est FIGÉE. Ne la change jamais.*
