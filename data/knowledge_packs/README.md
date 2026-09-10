# RATISS-Ready Knowledge Packs

Les packs de ce dossier sont des **JSON compacts, bilingues et sourcés**. Ils ne recopient pas des articles complets, ne contiennent pas de corpus massif et peuvent être chargés à la demande par domaine. Le manifeste `pack_index.json` décrit les fichiers disponibles et le contrat de relation commun.

| Pack | Champ `domain` | Portée | Limite importante |
|---|---|---|---|
| `quantum_physics_pack.json` | `quantum_physics` | États, mesures, portes et limites de matériel quantique | Pas une preuve QPU ni une validation d’hypothèse RATISS. |
| `math_logic_pack.json` | `math_logic` | Ensembles, axiomes, preuve et algèbre | Pas un assistant de preuve formelle. |
| `bio_pharma_pack.json` | `bio_pharma` | Biologie structurale et annotations moléculaires | Pas de diagnostic, posologie ou décision clinique. |
| `ai_systems_pack.json` | `ai_systems` | Gouvernance et gestion des risques de systèmes IA | Pas une certification de RATISS ou d’un autre système. |

## Contrat compact

Chaque entrée possède un concept racine `r` et une liste de relations `rel`. Une relation utilise `t` pour le concept cible, `k` pour le type de relation, `c` pour son contexte, `fr` et `en` pour les formulations bilingues, `src` pour les sources, et `evidence` pour le statut de la relation. Le champ `aeon_proof_status` vaut `not_generated` tant qu’aucune preuve interne n’a été effectivement produite.

> La présence d’une source dans un pack signifie que la phrase concise renvoie vers cette source. Elle ne transfère pas automatiquement les droits de réutilisation de la source et ne remplace jamais sa consultation directe.

## Chargement ciblé

```python
import json
from pathlib import Path

root = Path("data/knowledge_packs")
quantum = json.loads((root / "quantum_physics_pack.json").read_text(encoding="utf-8"))

for entry in quantum["entries"]:
    for relation in entry["rel"]:
        print(entry["r"], relation["k"], relation["t"])
```

Les packs sont volontairement séparés afin de garder une empreinte faible : une requête de mathématiques n’a pas à charger le contenu quantique, biomoléculaire ou IA.
