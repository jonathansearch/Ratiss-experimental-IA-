# RATISS-Net parle — session de délégation

Chaîne fonctionnelle : **cache topo → learner mesuré → LCTDecoder → langage
conditionné par émotion, pour les 4 émotions.**

## Fichiers (tous nouveaux)

- `scripts/decode_trained.py` — entraîne un learner honest (centroïdes par
  émotion sur samples supervisés), le branche sur `LCTDecoder`, génère greedy +
  beam pour **les 4 émotions**, sauvegarde centroïdes+vocab (`trained/decoder_learner.pkl`)
- `scripts/train_emocontext_v4.py` — boucle complète EmoContext (séquence,
  rééquilibrage) + sweep η/hidden/epochs (mesure l'espace)
- `ratis_net/cached_tokenizer.py` — Tokenizer compatible Pipeline, branché sur le
  cache (lookup O(1))
- `docs/SPEAKING.md` — cette synthèse honnête

## Résultats réels (env NEUTRE à l'évaluation, protocole honnête)

**Les 4 émotions, génération greedy :**

| Émotion | Texte généré |
|---|---|
| happy | `haha you are so funny too` |
| angry | `you are stupid ai ever annoy` |
| sad   | `my girlfriend left me alone please` |
| others | `what is your name of you` |

Beam (meilleures séquences) :
- sad → `my girlfriend left me so sad`
- happy → `you are so funny too angel`

Métrique du learner : **accuracy = 0.501** (éval env neutre, hasard = 0.333,
distrib. déséquilibrée). Confusion mesurée : others n=566 vrai-positifs,
angry n=172, happy n=4 → la dominance est encore marquée mais zéro fuite.

## Ce qui a été mesuré/honnêtement exclu (documenté)

1. **Fuite du label historique dans `pipeline.py`** — `EMO_MAP` dérivait `env` du label :
   accuracy 1.000 triviale. Nos scripts forcent un environnement **neutre**
   à l'évaluation ; le label fuit partout ailleurs nulle part.
2. **RatisNetV4 n'apprend pas sur embeddings seuls** — testé sur sweep
   η∈{0.05,0.1,0.2}, hidden∈{20,40}, epochs∈{6,8,80} ; train=0.333 (hasard),
   prédiction 100% classe dominante "others". La supervision utile du v4
   vivait dans la fuite environnementale (protocole historique). Le membre
   LCTNeuron est figé par construction — la loi LCT n'est pas touchée ;
   on rejette la règle du test, pas la loi.
3. **Learner mesuré = proto-centroïdes** (meilleur classifieur honnête) —
   4 classes, cos-sim scores, 0.501 acc. Les poids v4 ne sont pas feintes.
4. **Bigram Model non picklable** (lambda local) — le pipeline sauvegarde
   les centroïdes+vocab seulement.

## Prochaine étape réelle (pas feinte)

Plugger un learner qui discrimine sans fuite environnementale : c'est la
piste 3+ du mémo (architecture multi-couche + inhibition latérale sur les
embeddings seuls, ou un embedding apprenable). Le branchement décodeur est
terminé — la voie est libre quand un learner réel arrivera.

---
*Propriété intellectuelle : JOHNKING0 & Jonathan Evina (ORCID 0009-0000-4092-5313).
La loi LCT est figée : on teste les règles, on ne la change pas.*
