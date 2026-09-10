# Cache des signatures topologiques (P_sig)

**Objectif :** les signatures topo sont déterministes (même seed/paramètres) → chaque mot se calcule **une seule fois**, puis l'entraînement devient un lookup O(1). Résout la limite « P_sig coûteux ».

## Fichiers (tous NOUVEAUX — aucun fichier existant modifié)

- `ratis_net/topo_cache.py` — classe `TopoCache` (dict mémoire + persistance npz/json)
- `scripts/cache_topo_signatures.py` — CLI : scanne `data/emocontext/train.txt` + `dev.txt`, construit le vocabulaire via `emocontext_loader.vocabulary`, pré-calcule, sauvegarde
- `tests/test_topo_cache.py` — script test (convention du dépôt : `python tests/test_topo_cache.py`)
- `data/cache/topo_signatures.npz` + `.json` — le cache commité (voc complet ≈ 0.5 Mo)

## Usage

```bash
# régénérer le cache complet
python scripts/cache_topo_signatures.py

# ou limiter pour un test rapide
python scripts/cache_topo_signatures.py --max-examples 2000 --top-k 3000
```

```python
from ratis_net.topo_cache import TopoCache
cache = TopoCache().load()
sig = cache.get("bonjour")   # lookup O(1), pur numpy
```

## Branchement décodeur LCT (piste suivante)

Le script `scripts/decode_with_cache.py` (nouveau) montre le décodeur LCT branché sur le cache :
- `cache: 15122 sigs | vocab filtré: 2500 mots`
- `generate_greedy(emotion="happy")` → `haha good joke ll kk`
- `generate_beam(...)` → dégénère en répétition (`ll ll ll`) quand le learner est un probe minimal — limite honnête connue (le réseau n'a pas de profondeur pour la cohérence de séquence)
- Le branchement **fonctionne** : le décodeur reçoit les signatures en O(1) du cache et produit du langage conditionné par émotion. Reste à brancher le learner sur le vrai SNN entraîné (RATISS-Snn).

## Chiffres réels (session d'intégration)

- Backend `cpu` (gudhi optionnel, même API)
- ~25 signatures/s → vocab complet EmoContext (15 122 mots) ≈ **10 min, une seule fois**
- Sample 3000 mots = 107 Ko ; full ≈ 0.5 Mo
- Test : persistence disque ≈ appel direct `topo_tokenizer` (np.allclose), idempotence du warmup, calcul paresseux des mots inconnus

---
*Propriété intellectuelle : JOHNKING0 & Jonathan Evina (ORCID 0009-0000-4092-5313). La loi LCT est figée — ce cache accélère son calcul, il ne la change pas.*
