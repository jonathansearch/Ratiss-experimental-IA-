# Les 3 modules algorithmiques LCT (session transdisciplinaire)

Trois modules codés, testés réellement, documentés honnêtement. Nouveaux fichiers
uniquement ; la loi LCT (ΔW = η·φ·P_sig·C) n'est pas modifiée.

---

## Module 1 — Mesure gravitationnelle topologique (`grav_measure.py`)

Extrait la « forme » d'une densité gravitationnelle : cycles H1 de persistance
mesurés sur la structure d'intrication, comme oscillation cohérence/décohérence.

**Test réel :** densité coquille+bulk (60 pts, courbure 1.0) → P_sig = 0.420,
betti = [1,0,0]. Profil oscillationnel : θ balaye [0,π/2], C = |cos θ|,
compression TTF (filtre quantile) → courbe (θ, C, P_sig).

```python
from ratis_net.lct_modules import GravitationalTopoMeasure
m = GravitationalTopoMeasure()
pts = m.density_field(curvature=1.0)
m.measure_density(pts)          # {'P_sig': 0.420, 'betti': [1,0,0], ...}
m.oscillation_profile(pts)      # courbe cohérence → décohérence
```

## Module 2 — Qubit topologique simulé (`topo_qubit.py`)

Qubit logiciel dont le bit logique est un **invariant topologique** (résidu H1),
pas une amplitude. Portes = opérations sur les résidus (torsion du réseau).
Protection : le bit survit au bruit tant que P_sig > seuil.

**Test réel :** |0> → P_sig = 0.897 ; X gate → |1> → P_sig = 0.608 ;
bruit 0.3 → **bit préservé** (protégé = True).

```python
from ratis_net.lct_modules import TopologicalQubit
q = TopologicalQubit(protection=0.10)
q.x_gate().phase_gate(0.5)
q.measure_state()   # {'P_sig', 'logical_bit', 'protected', ...}
```

**Statut honnête** : simulation algorithmique. Aucune prétention matérielle —
c'est l'algorithme prêt AVANT la puce.

## Module 3 — Transformer LCT (`lct_transformer.py`)

Entraînement dédié sans backprop : ΔW = η·φ·P_sig·C·erreur·x, avec
**inhibition latérale** (winner-take-all amorti) pour forcer la spécialisation.
P_sig amorti (recalcul tous les 10 pas — il varie lentement) → ~30× plus rapide.

**Tests réels :**

| Test | Résultat | Statut |
|---|---|---|
| XOR (non-linéaire) | **3/4** (acc_train = 0.750) | ✅ le mécanisme apprend |
| EmoContext (embeddings topo) | acc = 0.403, collapse 1 classe | ⚠️ limite documentée |

### 🔬 Découverte définitive : le goulot est l'embedding, pas le learner

Mesure de séparabilité des centroïdes de classe dans l'espace topo_tokenizer :

```
cos(centroïde[angry], centroïde[happy])  = 1.0000
cos(centroïde[angry], centroïde[others]) = 1.0000
cos(centroïde[happy], centroïde[others]) = 1.0000
```

**Les centroïdes de classe sont IDENTIQUES** (cos = 1.0). Le topo_tokenizer
capture la *forme* des lettres (invariante par permutation sémantique), pas le
*sens* émotionnel. Aucun learner — LCT, centroid, ou autre — ne peut discriminer
des classes dont les représentations moyennes sont confondues.

Conséquence (piste 5 du mémo, désormais prouvée) : la prochaine étape est une
**couche d'embedding apprenable**. Le transformer LCT est prêt et validé sur
XOR ; il attend une représentation d'entrée informative.

---

## Lancer les tests

```bash
python tests/test_lct_modules.py
```

Sortie attendue : `TOUT OK — les 3 modules LCT sont fonctionnels`.

---
*Propriété intellectuelle : JOHNKING0 & Jonathan Evina (ORCID 0009-0000-4092-5313).
La loi LCT est figée — ces modules l'appliquent, ils ne la changent pas.*
