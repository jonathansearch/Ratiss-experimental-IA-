# Archive — Ce qui n'est plus dans le chemin actif

> Rien ici n'est supprimé : tout est **archivé, daté et expliqué**.
> Le code actif est dans `ratis_net/` (framework v2). Ces fichiers restent
> disponibles pour référence historique et pour comprendre les itérations
> passées, mais ils ne sont plus importés par le framework ni par les tests.

## legacy_v1/modules/ — premières itérations du réseau (v1 → v4)

| Module | Rôle historique | Pourquoi archivé |
|---|---|---|
| `lct_network.py`, `lct_network_v2.py`, `lct_network_v3.py` | Premières boucles LCT (classification directe) | Remplacées par Scalpel + science_core |
| `ratis_net_v4.py` | Classification émotion (EmoContext) | Hors du cœur : RATIS-Net est désormais un framework de langage |
| `pipeline.py` | Chaîne tokenizer→réseau→décodeur v4 | Le framework v2 route différemment (query_analyzer → intent_router) |
| `decoder.py` | Décodeur bigramme LCT | Supplanté par squelettes grammaticaux (13K+24K gabarits) |
| `ratis_speaker.py`, `trigrammar.py`, `concept_decoder.py` | Génération mot par mot | Qualité insuffisante vs squelettes ; gardés comme preuves d'étape |
| `ratis_agent.py`, `dialogue_engine.py` | Boucle cognitive + dialogue | Merged dans le framework v2 |
| `accelerated_immersion.py` | Entraînement accéléré v4 | data_loader (streaming HF) le remplace |
| `eth_thermo_fixer.py` ⚠️ **restauré** | ETH = f(token, env) | Requis par `emocontext_loader` (test vivant) |
| `lct_collapse.py` | Collapse topologique v4 | science_core expose les mesures LCT directement |
| `topo_gradient.py`, `topo_proxy.py` | Gradients topologiques | La loi LCT n'utilise pas de gradient (figée) |
| `shadow_tomography.py` | Tomographie quantique | Appartient au labo QPU, pas au framework langage |
| `cached_tokenizer.py` | Variante de tokenizer | topo_cache.py suffit |
| `skeleton_speaker.py` (v1) | Premier speaker à squelettes | Remplacé par `skeleton_speaker_v2.py` (routage + IDF + social) |
| `persistence_optimizer.py` ⚠️ **restauré** | Backend GUDHI pour la persistance | Requis par `topo_tokenizer.py` |
| `emocontext_loader.py` ⚠️ **restauré** | Données d'émotions | Requis par `tests/test_lct_modules.py` |

## legacy_v1/tests/ — coquilles de tests v1–v4

Ces fichiers ciblaient les modules archivés. La plupart ne contiennent
**aucune fonction `test_*`** (des scripts de démonstration, pas des tests) :
ils donnaient l'illusion d'une couverture. Les vrais tests actifs sont dans
`tests/` (57 tests, tous verts).

## legacy_v1/data/ — données des modules archivés

`emocontext/` ⚠️ **restauré** dans `data/` : utilisé par le test
`test_lct_transformer_emocontext`.

## session_memos/ — mémos de session passées

`LIRE_AVEC_URGENCE.md`, `LIRE_AVEC_URGENCE_v2.md`, `LECON_23_MEMO_SESSION.md` :
fichiers de relais d'anciennes sessions, **remplacés par `MEMO_GLOBAL.md`**
(à la racine) qui consolide tout l'écosystème.
