"""ratis_net.framework — Framework IA unifié RATIS-Net.

Unifie tous les modules en une seule API cohérente :

  RatisNet (framework)
    ├── Scalpel          (réseau de corrélations LCT, 3.78M neurones)
    ├── Synchrotron      (reconstruction topologique)
    ├── SkeletonSpeaker  (génération par squelettes grammaticaux)
    ├── ConceptDecoder   (concepts → phrases)
    ├── TriGrammarSpeaker(génération mot par mot, fenêtre 2)
    └── GloveTokenizer   (embeddings hybrides)

Usage :
    from ratis_net.framework import RatisNet

    net = RatisNet()
    net.load_scalpel("artifacts/scalpel_wikipedia.pkl")
    net.load_grammar("data/grammar_domains/dense_syntax_skeletons.json")

    print(net.respond("what is quantum mechanics"))
    print(net.respond("explain consciousness", language="fr"))
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ratis_net.glove_tokenizer import GloveTokenizer
from ratis_net.scalpel import ScalpelLayer
from ratis_net.skeleton_speaker_v2 import SkeletonSpeakerV2
from ratis_net.aeon_bridge import AeonBridge
from ratis_net.web_search import WebSearchModule
from ratis_net.chain_reasoning import find_chains
from ratis_net.integrity_proof import prove as _prove_subgraph
from ratis_net.integrity_proof import verify as _verify_subgraph


class RatisNet:
    """Framework IA unifié : Scalpel + AEON + Web + Squelettes.

    Super RATISS : fusion des deux cerveaux.
      - RATIS-Net (langage) : Scalpel + Squelettes + GloVe
      - AEON ODV (science)  : topologie, LCT, preuves
      - Web (temps réel)    : DuckDuckGo / Google CSE

    API simple :
      net.respond("what is quantum") → phrase fluide + fait scientifique
      net.concepts("quantum") → liste de concepts
      net.paragraph("consciousness") → paragraphe long
      net.search("quantum decoherence") → résultats web
    """

    def __init__(self, dim: int = 12, n_glove: int = 8, seed: int = 42,
                 aeon_path: str | Path | None = None,
                 engine_path: str | Path | None = None):
        # aeon_path/engine_path ignorés : tout est intégré dans science_core.py
        self.tokenizer = GloveTokenizer(dim=dim, n_glove=n_glove)
        self.scalpel = ScalpelLayer(self.tokenizer, eta=0.1,
                                     coherence_threshold=0.3, seed=seed)
        self.speaker = SkeletonSpeakerV2(self.scalpel, seed=seed)
        self.aeon = AeonBridge()  # toujours disponible (intégré)
        self.web = WebSearchModule()
        self._loaded = False
        self._index_built = False
        self._knowledge_packs: dict = {}
        self._neural: Any | None = None  # NeuralSpeaker (cerveau qui parle)

    def load_scalpel(self, path: str | Path = "artifacts/scalpel_wikipedia.pkl",
                     verbose: bool = True) -> None:
        """Charge le checkpoint Scalpel (294 MB, 3.78M neurones)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scalpel checkpoint not found: {path}")
        self.scalpel.load(path)
        if verbose:
            print(f"Scalpel: {self.scalpel.network_size():,} neurons, "
                  f"{self.scalpel.total_reinforcements:,} reinforcements")
        self._loaded = True

    def load_grammar(self, dense_path: str | Path | None = None,
                     conversation_path: str | Path | None = None,
                     verbose: bool = True) -> None:
        """Charge les gabarits grammaticaux (13K + 24K formulations)."""
        # Le SkeletonSpeaker charge automatiquement les fichiers s'ils existent
        # dans data/grammar_domains/. Cette méthode force le rechargement.
        if dense_path:
            import json
            with open(dense_path, encoding="utf-8") as f:
                self.speaker.dense_grammar = json.load(f)
        if conversation_path:
            import json
            with open(conversation_path, encoding="utf-8") as f:
                self.speaker.conversation_matrix = json.load(f)
        if verbose:
            n_dense = 0
            if self.speaker.dense_grammar:
                domains = self.speaker.dense_grammar.get("domains", {})
                n_dense = sum(len(entries)
                              for d in domains.values()
                              for entries in d.values())
            n_conv = 0
            if self.speaker.conversation_matrix:
                contexts = self.speaker.conversation_matrix.get("social_contexts", {})
                n_conv = sum(len(entries)
                             for c in contexts.values() if isinstance(c, dict)
                             for entries in c.values() if isinstance(entries, list))
            print(f"Grammar: {n_dense:,} dense + {n_conv:,} conversation templates")

    def build_index(self, verbose: bool = True) -> None:
        """Construit l'index inversé pour les lookups O(1)."""
        if not self._loaded:
            raise RuntimeError("Load Scalpel first with load_scalpel()")
        self.speaker.build_index(verbose=verbose)
        self._index_built = True

    def load_neural_speaker(self,
                            corpus_path: str | Path | None = None,
                            verbose: bool = True) -> None:
        """Charge le speaker neuronal (reconstruction depuis le corpus appris).

        Le réseau v1 active un chemin de concepts ; le corpus ordonné fournit
        les phrases candidates ; la couverture IDF choisit la meilleure.
        """
        from ratis_net.neural_speaker import NeuralSpeaker
        if corpus_path is None:
            corpus_path = (Path(__file__).resolve().parents[1]
                           / "data" / "corpus" / "scalpel_v3_corpus.txt")
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            if verbose:
                print(f"Neural speaker: corpus absent ({corpus_path})")
            return
        graph: dict[str, list] = {}
        for (a, b), n in self.scalpel.neurons.items():
            graph.setdefault(a, []).append((b, n.weight, n.p_sig,
                                            n.n_reinforcements))
            graph.setdefault(b, []).append((a, n.weight, n.p_sig,
                                            n.n_reinforcements))
        corpus = [l.strip() for l in open(corpus_path, encoding="utf-8")
                  if l.strip()]
        self._neural = NeuralSpeaker(graph, corpus)
        if verbose:
            print(f"Neural speaker: {len(corpus):,} phrases apprises, "
                  f"{len(graph):,} mots dans le graphe")

    def concepts(self, word: str, n: int = 10) -> list[str]:
        """Extrait les n concepts les plus pertinents autour d'un mot
        (classement IDF + voisinage partagé + complément GloVe)."""
        if not self._index_built:
            self.build_index(verbose=False)
        return self.speaker._extract_concepts(word.lower(), n=n)

    def load_knowledge_packs(self, path: str | Path | None = None,
                             verbose: bool = True) -> None:
        """Charge les knowledge packs (quantum, bio, math, AI).

        Chaque pack contient des relations scientifiques FR/EN avec source
        et statut de preuve. Le framework peut les consulter pour enrichir
        les réponses avec des faits validés.
        """
        import json
        if path is None:
            path = Path(__file__).resolve().parents[1] / "data" / "knowledge_packs"
        path = Path(path)
        if not path.exists():
            if verbose:
                print(f"Knowledge packs: not found at {path}")
            return
        # Charger l'index
        index_path = path / "pack_index.json"
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                self._knowledge_packs["_index"] = json.load(f)
        # Charger chaque pack
        for pack_file in path.glob("*_pack.json"):
            domain = pack_file.stem.replace("_pack", "")
            with open(pack_file, encoding="utf-8") as f:
                self._knowledge_packs[domain] = json.load(f)
        if verbose:
            total_entries = sum(len(v.get("entries", []))
                                for v in self._knowledge_packs.values()
                                if isinstance(v, dict))
            print(f"Knowledge packs: {len(self._knowledge_packs)-1} domains, "
                  f"{total_entries} entries")

    def lookup_knowledge(self, concept: str, language: str = "en") -> list[dict]:
        """Cherche un concept dans les knowledge packs.

        Retourne les relations trouvées : [{term, relation, context, text}, ...]
        """
        import unicodedata

        def _norm(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFKD", s.lower().strip())
                           if not unicodedata.combining(c))

        results = []
        concept_lower = concept.lower().strip()
        concept_norm = _norm(concept)
        if len(concept_lower) < 3:
            return results
        for domain, pack in self._knowledge_packs.items():
            if domain == "_index" or not isinstance(pack, dict):
                continue
            for entry in pack.get("entries", []):
                root = entry.get("r", "").lower()
                aliases = [_norm(a) for a in entry.get("aliases", [])]
                # exact, alias FR, ou inclusion de mot complet (évite "va" → "vaccine")
                matched = (concept_lower == root or
                           concept_norm in aliases or
                           (len(concept_lower) >= 4 and concept_lower in root) or
                           (len(root) >= 4 and root in concept_lower))
                if not matched:
                    continue
                for rel in entry.get("rel", []):
                    lang_key = language if language in rel else "en"
                    results.append({
                        "domain": domain,
                        "root": entry.get("r", ""),
                        "term": rel.get("t", ""),
                        "relation": rel.get("k", ""),
                        "context": rel.get("c", ""),
                        "text": rel.get(lang_key, rel.get("en", "")),
                        "source": pack.get("sources", ["unknown"]),
                        "aeon_proof_status": "not_generated",
                    })
        return results

    def search(self, query: str, n: int = 3) -> list[dict]:
        """Recherche web temps réel (DuckDuckGo ou Google CSE)."""
        results = self.web.search(query, n=n)
        return [r.to_dict() for r in results]

    def respond(self, query: str, language: str | None = None) -> str:
        """Génère une réponse fluide à une requête.

        Pipeline Super RATISS :
          1. RATIS-Net extrait les concepts (Scalpel).
          2. AEON valide/calcul le fait scientifique (P_sig, LCT).
          3. Le squelette grammatical habille le tout.
        """
        if not self._index_built:
            self.build_index(verbose=False)
        result = self.speaker.generate_response(query, language=language)
        return result["sentence"]

    def respond_with_science(self, query: str, language: str | None = None) -> dict[str, Any]:
        """Réponse complète : phrase + fait scientifique AEON + concepts.

        C'est le pipeline Super RATISS complet :
          1. Extraction des concepts (Scalpel)
          2. Calcul scientifique (AEON : P_sig du cluster de concepts)
          3. Génération de la phrase (Squelettes)
          4. Recherche web si les concepts sont inconnus
        """
        if not self._index_built:
            self.build_index(verbose=False)

        # 1. Concepts via Scalpel
        result = self.speaker.generate_response(query, language=language)
        concepts = result.get("concepts", [])
        lang = result.get("language", language or "en")

        # Les actes sociaux (salutation, gratitude...) ne déclenchent ni
        # faits scientifiques ni recherche web : la réponse est le social.
        social_qtypes = {"greeting", "farewell", "gratitude", "identity"}
        if result.get("qtype") in social_qtypes:
            return {
                "query": query,
                "sentence": result["sentence"],
                "paragraph": result["paragraph"],
                "concepts": concepts[:10],
                "aeon_fact": None,
                "knowledge_facts": [],
                "web_results": [],
                "aeon_backend": self.aeon.backend_name,
                "web_backend": self.web.backend,
                "neural": None,
            }

        # 2. Fait scientifique via AEON (science_core intégré)
        aeon_fact = self.aeon.query(concepts, scalpel=self.scalpel)

        # 2b. Knowledge packs : faits validés, composés de la requête d'abord
        # ("black hole" est une racine de pack ; "black" seul ne l'est pas)
        knowledge_facts = []
        seen_facts: set[str] = set()
        lookup_terms = (result.get("compounds", []) +
                        result.get("keywords", []) + concepts[:3])
        for term in lookup_terms:
            for fact in self.lookup_knowledge(term, language=lang):
                key = fact["text"][:80]
                if key not in seen_facts:
                    seen_facts.add(key)
                    knowledge_facts.append(fact)
            if len(knowledge_facts) >= 5:
                break

        # 2c. Le cerveau parle : reconstruction neuronale depuis le corpus
        # appris. Si le réseau reconnaît une phrase couvrant le chemin de
        # concepts, elle prime sur le gabarit (et sur le fait si la
        # couverture est forte ET la phrase est une définition du sujet).
        neural_sentence = None
        if self._neural is not None:
            neural = self._neural.speak(query)
            if neural.get("ok"):
                neural_sentence = neural

        # 3. Si les concepts sont faibles ET pas de knowledge pack, chercher web
        web_results = []
        if len(concepts) < 3 and len(knowledge_facts) < 2:
            web_results = self.search(query, n=3)
            for wr in web_results:
                if wr.get("snippet"):
                    words = wr["snippet"].split()[:5]
                    concepts.extend([w for w in words if len(w) > 3])

        # 4. Construire la réponse enrichie
        # Priorité : cerveau neuronal > faits vérifiés > gabarit.
        # Le neuronal prime quand il reconnaît une phrase avec une couverture
        # suffisante — c'est le réseau qui parle, pas une base curée.
        enriched_sentence = result["sentence"]
        if neural_sentence is not None:
            enriched_sentence = neural_sentence["sentence"]
        elif knowledge_facts:
            facts_text = " ".join(f["text"].strip() for f in knowledge_facts[:2])
            if facts_text and facts_text[-1] not in ".!?":
                facts_text += "."
            related = [c for c in concepts[:5]
                       if c.lower() not in facts_text.lower()]
            # Requête multi-mots : un concept « associé » doit être relié à
            # au moins 2 mots-clés dans le graphe — sinon le biais de corpus
            # passerait à côté du fait (« black hole » → metal, comedy).
            kws = [k for k in (result.get("keywords") or []) if len(k) > 2]
            if len(kws) >= 2 and self.speaker._ranker is not None:
                neigh = self.speaker._ranker._neighbors
                related = [c for c in related
                           if sum(1 for k in kws
                                  if c in neigh.get(k, ())) >= 2]
            related = related[:3]
            # Une association isolée est du bruit ; un cluster (≥2) est un
            # signal. Sans cluster, pas de complément — le fait reste seul.
            if len(related) < 2:
                related = []
            if related:
                tail = ("Associated in the learned corpus: "
                        if lang == "en" else "Associés dans le corpus appris : ")
                facts_text += " " + tail + ", ".join(related) + "."
            enriched_sentence = facts_text

        return {
            "query": query,
            "sentence": enriched_sentence,
            "paragraph": result["paragraph"],
            "concepts": concepts[:10],
            "aeon_fact": aeon_fact.to_dict(),
            "knowledge_facts": knowledge_facts[:5],
            "web_results": web_results,
            "aeon_backend": self.aeon.backend_name,
            "web_backend": self.web.backend,
            "neural": neural_sentence,
        }

    def respond_full(self, query: str, language: str | None = None) -> dict[str, Any]:
        """Génère une réponse détaillée (avec concepts, squelette, paragraphe)."""
        if not self._index_built:
            self.build_index(verbose=False)
        return self.speaker.generate_response(query, language=language)

    def chain(self, source: str, target: str, max_hops: int = 3,
              max_chains: int = 3) -> list[dict[str, Any]]:
        """Chaînes d'association entre deux concepts (corrélation, pas causalité)."""
        if not self._index_built:
            self.build_index(verbose=False)
        chains = find_chains(source, target, self.speaker._index,
                             max_hops=max_hops, max_chains=max_chains)
        return [c.to_dict() for c in chains]

    def speak(self, query: str) -> dict[str, Any]:
        """Le cerveau parle : reconstruction neuronale depuis le corpus appris.

        À la différence de respond() (faits validés + gabarits), speak()
        retourne la phrase que le réseau RECONNAIT comme la plus proche de
        la requête — couverture IDF du chemin de concepts activé.
        """
        if self._neural is None:
            self.load_neural_speaker(verbose=False)
        if self._neural is None:
            return {"mode": "neural", "ok": False, "sentence": "",
                    "reason": "corpus neural absent"}
        return self._neural.speak(query)

    def prove(self, concepts: list[str]) -> dict[str, Any]:
        """Empreinte SHA-256 du sous-graphe de concepts (intégrité, pas ZK)."""
        proof = _prove_subgraph(concepts, self.scalpel)
        return proof.to_dict()

    def verify_proof(self, proof: dict[str, Any]) -> bool:
        """Vérifie qu'une empreinte se reproduit depuis le checkpoint chargé."""
        from ratis_net.integrity_proof import IntegrityProof
        p = IntegrityProof(concepts=proof["concepts"],
                           n_edges=proof["n_edges"],
                           total_weight=proof["total_weight"],
                           p_sig_max=proof["p_sig_max"],
                           digest=proof["digest"])
        return _verify_subgraph(p, self.scalpel)

    def paragraph(self, theme: str, n_sentences: int = 5,
                  language: str = "en") -> str:
        """Génère un paragraphe long sur un thème."""
        if not self._index_built:
            self.build_index(verbose=False)
        return self.speaker.generate_paragraph(theme, n_sentences=n_sentences,
                                                language=language)

    def converse(self, query: str, language: str | None = None) -> str:
        """Mode conversation : réponse + paragraphe court (3 phrases)."""
        if not self._index_built:
            self.build_index(verbose=False)
        result = self.speaker.generate_response(query, language=language)
        return result["paragraph"]

    def stats(self) -> dict[str, Any]:
        """Statistiques du framework."""
        return {
            "scalpel_neurons": self.scalpel.network_size(),
            "scalpel_reinforcements": self.scalpel.total_reinforcements,
            "scalpel_neurogenesis": self.scalpel.total_neurogenesis,
            "vocab_size": len(self.speaker._index) if self._index_built else 0,
            "grammar_loaded": self.speaker.dense_grammar is not None,
            "conversation_loaded": self.speaker.conversation_matrix is not None,
            "glove_available": self.tokenizer.backend() != "topo_fallback",
            "aeon_backend": self.aeon.backend_name,
            "aeon_available": self.aeon.available,
            "web_backend": self.web.backend,
            "web_available": self.web.available,
            "knowledge_packs": len(self._knowledge_packs),
        }

    def __repr__(self) -> str:
        n = self.scalpel.network_size()
        return f"RatisNet(neurons={n:,}, loaded={self._loaded})"


# ─────────────────────────────────────────────────────────────────────────────
# CLI : python -m ratis_net.framework
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="RATIS-Net unified framework")
    ap.add_argument("--scalpel", default="artifacts/scalpel_wikipedia.pkl")
    ap.add_argument("--grammar", default="data/grammar_domains/dense_syntax_skeletons.json")
    ap.add_argument("--query", default=None, help="Single query to respond to")
    ap.add_argument("--paragraph", default=None, help="Theme for paragraph generation")
    ap.add_argument("--language", default="en", choices=["en", "fr"])
    args = ap.parse_args()

    print("=== RATIS-Net Unified Framework ===\n")
    net = RatisNet()
    net.load_scalpel(args.scalpel)
    net.load_grammar(args.grammar)
    net.build_index()
    print()

    stats = net.stats()
    print(f"Stats: {stats}\n")

    if args.query:
        print(f"Q: {args.query}")
        print(f"R: {net.respond(args.query, language=args.language)}\n")
    elif args.paragraph:
        print(f"Theme: {args.paragraph}")
        print(f"{net.paragraph(args.paragraph, n_sentences=5, language=args.language)}\n")
    else:
        # Demo complet
        queries_en = [
            "what is quantum mechanics",
            "explain consciousness",
            "what is love",
            "how does the brain work",
            "what is science",
        ]
        print("=== Responses (EN) ===\n")
        for q in queries_en:
            print(f"Q: {q}")
            print(f"R: {net.respond(q, language='en')}")
            print()

        print("=== Paragraphs ===\n")
        for theme in ["quantum", "consciousness", "love"]:
            print(f"[{theme}]")
            print(net.paragraph(theme, n_sentences=4, language="en"))
            print()

        print("=== Français ===\n")
        for q in ["qu'est-ce que la science", "explique l'amour"]:
            print(f"Q: {q}")
            print(f"R: {net.respond(q, language='fr')}")
            print()
