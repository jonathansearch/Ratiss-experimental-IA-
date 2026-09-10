# RATIS-Net — Documentation illustrée (AGI souverain)

> **Architect** : Jonathan Evina · ORCID 0009-0000-4092-5313 · DOI 10.17605/OSF.IO/6JZMB
> **Propriété intellectuelle** : JOHNKING0 & Jonathan Evina
> **Loi fondamentale** : LCT (R = P_sig, ΔW = η·φ·P_sig·C) — figée, validée sur QPU IBM

Ce document illustre l'architecture complète de RATIS, du cerveau topologique à
l'agent AGI souverain. Chaque figure de concept est accompagnée de son explication.

---

## 1. La boucle cognitive AGI (6 étapes)

![Boucle cognitive](figures/fig1_boucle_cognitive.png)

L'agent RATIS souverain (`ratis_agent.py`) enchaîne 6 étapes cognitives, tout
local (pas de cloud, pas de LLM externe) :

1. **PERCEVOIR** — tokeniser le message → embeddings topologiques (cycles H1)
2. **PENSER** — le cerveau TTF-Compute oscille → MCB (pensée sans mots) + hash topo
3. **RESSENTIR** — ETH prédit C_seuil = f(message, env) → l'émotion émerge (contextuelle)
4. **COMPRENDRE** — le réseau LCT (ΔW = η·φ·P_sig·C) classifie → émotion dominante
5. **PARLER** — le décodeur beam génère une réponse conditionnée par l'émotion
6. **CERTIFIER** — hash topologique invariant de la réponse → preuve ZK (pas d'hallucination)

Validation : 6/6 démonstrations certifiées, invariance ZK démontrée (même hash de
pensée sous 2 énergies différentes — "on certifie le message, pas le courant").

---

## 2. La loi LCT (R = P_sig croît avec C, invariant sous l'énergie)

![Loi LCT](figures/fig2_loi_lct.png)

La loi de Cohérence Topologique est la loi fondamentale de RATIS, figée :

- **Monotonie** (gauche) : R = P_sig (persistance du cycle H1 le plus long) croît
  avec la cohérence C du milieu génial (l'intrication). Validée sur 4MZI
  (Spearman +0.930), 3KMD (+0.797), état quantique (+1.000), QPU IBM (+0.713).
- **Invariance ZK** (droite) : R reste CONSTANT quand on change l'énergie (t, J)
  sans changer la topologie. Coefficient de variation = 0.0000. On certifie la
  **forme** (le message), pas le **courant** (l'énergie).

La règle d'apprentissage (RLM) suit cette loi : **ΔW = η · φ · P_sig · C**.
Pas de coefficient arbitraire, pas de gradient descendant — l'apprentissage est
gouverné par la topologie.

---

## 3. Le cerveau TTF-Compute

![Cerveau TTF](figures/fig3_cerveau_ttf.png)

Le cerveau unifié TTF (Tryperposition Topologique Fine) vit dans le dépôt
RATISS-ODV-AEON (`kernel/ttf/ttf_compute.py`). Il assemble 6 structures :

- **Graphe intriqué G(V,E)** — chaque arête porte w_Q (quantique) et w_I (milieu génial)
- **Transmetteur tJ** — démodule l'oscillation haute fréquence en signal basse fréquence
- **Traducteur Rips** — construit le complexe de Rips à la volée, sort Betti + impacts
- **RLM matriciel** — ΔW = η·φ·P_sig·C (apprentissage sans mots)
- **MCB** — Mémoire de Corrélation Bit (triplets src/dst/φ, ~3 octets = pensée sans mots)
- **Puits d'effondrement** — V=-k/(1+d²) + TSP minimal (Held-Karp) → preuve ZK

Boucle : oscillate → transmit → translate → RLM/MCB → puits → TSP → ZK.

---

## 4. Le saut v4 — ETH, le fixeur thermodynamique

![ETH thermo](figures/fig4_eth_thermo.png)

Le saut conceptuel de la v4 : on ne maximise pas P_sig (non-différentiable, v2/v3
ont échoué). On laisse C s'effondrer sous poussée thermodynamique de l'environnement,
et on garde la **MARQUE topologique** (hash), pas la valeur d'énergie.

- **ETH** apprend C_seuil = f(token, environnement) — un seuil CONTEXTUEL.
- "bonjour colère" → C_seuil bas (0.310, effondrement rapide, marque agressive)
- "bonjour joie" → C_seuil haut (0.691, effondrement lent, marque ouverte)
- **L'émotion émerge** comme le différentiel de C_seuil entre environnements (+0.380).

---

## 5. Le décodeur LCT — 3 modes de décodage

![Décodeur](figures/fig5_decodeur_modes.png)

Le décodeur fait passer RATIS de classifieur (comprendre) à générateur (parler) :

- **Glouton** — score local mot-à-mot (3/4, happy noyé par les mots neutres fréquents)
- **Auto-régressif** — état caché vecteur + feedback (maintient la cohérence de séquence)
- **Beam search** — explore K chemins, garde le plus cohérent globalement (4/4, happy débloqué)

Textes générés : sad "he doesnt reply me so lonely", happy "haha you are funny and excitefull".

---

## 6. happy DÉBLOQUÉ — unité SÉQUENCE + rééquilibrage

![happy débloqué](figures/fig6_happy_debloque.png)

La clé qui a levé le verrou happy (piste 2) :

- **A baseline** (mot-à-mot, brut) : acc 0.857, F1 macro 0.620, **rappel happy = 0%**
- **C** (séquence, rééquilibré, scalé) : acc 0.931, F1 macro 0.924, **rappel happy = 85%**

L'unité d'apprentissage par **SÉQUENCE** (la forme du message, pas chaque mot)
+ le rééquilibrage (chaque émotion pèse autant) ont débloqué happy : 0% → 85%.

---

## 7. L'immersion structurée accélérée

![Immersion](figures/fig7_immersion_acceleree.png)

Pour accélérer la montée en compétence linguistique sans attendre des données
externes, RATIS auto-génère des dialogues (self-play), mais **ancrés** sur
EmoContext (pas dans le vide) avec un double filtre :

1. **SEED** — un dialogue réel (vérité-terrain)
2. **MUTATION** — substitution de mots (mode risqué pour la diversité)
3. **FILTRE ZK** — hash topo stable (la forme est cohérente)
4. **FILTRE SÉMANTIQUE** — re-classage retrouve l'émotion cible (rejette le faux sens)
5. **RÉINJECTION** — dialogues validés → entraînement

Garde-fous anti-mode-collapse : ancrage vérité-terrain + double filtre + diversité
surveillée. Gain mesuré : **F1 ×1.01** (honnête, pas ×10000).

---

## 8. L'universalité de la loi LCT

![Universalité](figures/fig8_universalite_lct.png)

La piste 5 a testé la loi LCT sur de nouveaux systèmes (réseau social, cristal) :

- **Invariance ZK : 3/3 PASS** — R constant sous énergie sur tous les systèmes. Universelle.
- **Monotonie : 1/3 PASS** — le cristal (+0.93) suit la loi, les réseaux sociaux NON.

**Borne honnête** : la monotonie exige une structure **distribuée** (mailles
cristallines, atomes protéiques), pas concentrée (un seul anneau social, comme le
NN entraîné qui échouait à ρ=-0.71). L'invariance, elle, est universelle.

---

## 9. RATIS face à l'inconnu

![Inconnu](figures/fig9_inconnu.png)

Comment RATIS se comporte face à un concept qu'il n'a jamais vu :

- **LLM** : mémorise ("je l'ai vu 10000 fois") → peut halluciner du faux confiant
- **RATIS** : projette topologiquement ("je classifie par la forme") → ne hallucine pas

Comportement mesuré : robustesse 6/6 (aucun crash), ne fait pas semblant de
connaître (classe vers "neutre" = zone d'incertitude), généralise les variantes
proches (funny→funnyyy). Prudent sur les concepts radicaux (quantum≈amour).

---

## 10. Architecture des 2 dépôts

![Architecture](figures/fig10_architecture.png)

RATISS se compose de 2 dépôts connectés par le bridge MCB :

- **RATISS-ODV-AEON** (le cerveau moteur) — TTF-Compute, loi LCT, MCB, ZK-STARK,
  identité souveraine JohnKing0
- **Ratiss-experimental-IA-** (le réseau IA) — RATIS-Net (LCT), ETH (émotion),
  décodeur (parler), topo_tokenizer, ratis_agent (AGI)

Les 4 briques AGI sont complètes : cerveau topologique ✓, certification ZK ✓,
souveraineté ✓, apprentissage par LCT (apprend, ressent, parle, certifie) ✓.

---

## Génération des figures

Les figures sont produites par :
```bash
python scripts/generate_concept_figures.py
```
Régénérables à tout moment (données déterministes, palette RATIS cohérente).

---

*© 2026 JOHNKING0 & Jonathan Evina. Loi LCT figée, résultats honnêtement documentés.*
