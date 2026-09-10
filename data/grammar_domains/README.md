# Matrices linguistiques bilingues RATISS-Net

Ce dossier contient deux banques de gabarits **FR/EN** conçues pour une sélection locale par domaine, intention, contexte social, couche émotionnelle et registre. Elles servent à la réalisation de surface : elles ne remplacent pas le Scalpel, la mémoire de corrélations, la récupération topologique, ni la vérification des connaissances.

| Fichier | Rôle | Index de sélection | Volume brut | Entrées |
|---|---|---|---:|---:|
| `dense_syntax_skeletons.json` | Formulations grammaticales structurées | `domain`, `intention`, `register` | Voir `metadata.uncompressed_bytes` | Voir `metadata.entry_count` |
| `conversation_matrix.json` | Formulations de dialogue social structurées | `social_context`, `intention`, `emotional_layer`, `register` | Voir `metadata.uncompressed_bytes` | Voir `metadata.entry_count` |
| `ultra_context_map.json` | Carte des corrélations de mots réellement enregistrées par le checkpoint Scalpel Wikipedia | `concept_root` → `co_occurs_with` → `surface_routes` | Environ 400 MiB | Voir `export.included_roots` et `export.included_directed_edges` |

## Contrat de données

Chaque entrée comporte un identifiant stable, une formulation française `fr`, une formulation anglaise `en`, une liste `placeholders` et les clés de sélection utilisées. Les variables sont conservées sous la forme `{X}`, `{Y}`, `{Z}`, `{PERSON}`, `{OBJECT}`, `{ACTION}`, `{TIME}`, `{PLACE}` ou `{EMOTION}`. Une entrée n’emploie que les variables réellement nécessaires à sa syntaxe.

> Après substitution des variables, le programme appelant doit vérifier l’accord, les pronoms, le registre et la sûreté contextuelle. Les matrices ne valident pas elles-mêmes les faits, les résultats scientifiques, une décision médicale, juridique ou financière, et ne doivent pas faire croire qu’un système possède une expérience émotionnelle humaine.

## Chargement local minimal

```python
import json
from pathlib import Path

root = Path("data/grammar_domains")
grammar = json.loads((root / "dense_syntax_skeletons.json").read_text(encoding="utf-8"))
conversation = json.loads((root / "conversation_matrix.json").read_text(encoding="utf-8"))

entry = grammar["domains"]["scientific"]["explain"][0]
print(entry["fr"])
```

Les métadonnées de chaque fichier précisent son format, son nombre d’entrées, son poids brut UTF-8, le contrat de variables, la méthode d’assemblage et les limites d’usage. Les contenus sont des gabarits originaux assemblés localement ; ce ne sont ni un corpus de conversations récoltées, ni un modèle linguistique, ni une promesse de couverture universelle de la grammaire.

## Ultra Context Map

`ultra_context_map.json` est une vue JSON indexée par concept racine, construite en lecture seule à partir de `artifacts/scalpel_wikipedia.pkl`. Chaque arête conserve exactement cinq éléments : le terme voisin, le poids LCT, `P_sig`, la cohérence et le compteur de renforcements. La carte ajoute une fenêtre de contexte à deux mots à partir des voisins les plus forts, ainsi que des chemins de sélection vers les deux banques de gabarits de ce dossier.

> Le checkpoint Scalpel enregistre des **paires de mots**, et non des trigrammes observés. La fenêtre à deux mots est donc une recombinaison locale exploitable par `ratis_net.trigrammar`, pas une affirmation qu’un trigramme historique a été stocké. La carte est volumineuse : un programme doit éviter de la charger entièrement si une recherche ciblée ou un lecteur streaming suffit.
