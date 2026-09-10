"""ratis_net.skeleton_speaker — Generation par squelettes syntaxiques.

Au lieu de generer mot par mot au hasard, le speaker choisit un SQUELETTE
grammatical (ex: "{concept1} is a branch of {concept2} that studies {concept3}")
et remplit les trous avec les concepts extraits par le Scalpel.

Le Scalpel fournit le fond (concepts), le squelette fournit la forme (grammaire).
Resultat : des phrases grammaticalement correctes qui parlent du sujet demande.

Avantages :
  - Leger : 18 squelettes stockes, pas des milliards de phrases.
  - Controle : la grammaire est garantie par le squelette.
  - Flexible : squelettes pour definir, expliquer, decrire, questionner, etc.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ratis_net.scalpel import ScalpelLayer
except ImportError:
    from scalpel import ScalpelLayer


# Mots structurels qui ne doivent pas remplir les slots concept
STOPWORDS = {"the", "a", "an", "of", "in", "and", "to", "is", "was", "were",
             "that", "this", "it", "for", "on", "with", "as", "by", "at",
             "from", "or", "be", "has", "have", "had", "but", "not", "he",
             "she", "his", "her", "its", "their", "they", "which", "who",
             "are", "been", "also", "than", "then", "so", "if", "can", "will",
             "would", "could", "should", "may", "might", "must", "do", "does",
             "did", "no", "yes", "all", "any", "some", "each", "every", "such",
             "one", "two", "three", "first", "second", "last", "more", "most",
             "less", "much", "many", "few", "other", "same", "different"}


class SkeletonSpeaker:
    """Genere des phrases en remplissant des squelettes avec les concepts Scalpel.

    1. Le Scalpel extrait un cluster de concepts autour du theme.
    2. On choisit un squelette adapte au theme (par keywords).
    3. On remplit les slots {concept1}, {concept2}, {concept3} avec les
       concepts les plus pertinents (poids LCT le plus eleve).
    4. Le resultat est une phrase grammaticalement correcte.
    """

    def __init__(self, scalpel: ScalpelLayer,
                 skeletons_path: str | Path | None = None,
                 dense_path: str | Path | None = None,
                 conversation_path: str | Path | None = None,
                 seed: int = 42):
        self.scalpel = scalpel
        self.rng = random.Random(seed)
        self._index: dict[str, list[tuple[str, float, float]]] = {}
        self._indexed = False

        base = Path(__file__).resolve().parent

        # Charger la base dense (13K squelettes) si disponible
        if dense_path is None:
            dense_path = base.parent / "data" / "grammar_domains" / "dense_syntax_skeletons.json"
        self.dense_grammar = None
        if Path(dense_path).exists():
            with open(dense_path, encoding="utf-8") as f:
                self.dense_grammar = json.load(f)

        # Charger la matrice conversationnelle (24K) si disponible
        if conversation_path is None:
            conversation_path = base.parent / "data" / "grammar_domains" / "conversation_matrix.json"
        self.conversation_matrix = None
        if Path(conversation_path).exists():
            with open(conversation_path, encoding="utf-8") as f:
                self.conversation_matrix = json.load(f)

        # Fallback : squelettes simples
        if skeletons_path is None:
            skeletons_path = base / "syntax_skeletons.json"
        with open(skeletons_path, encoding="utf-8") as f:
            self.skeletons = json.load(f)

    def build_index(self, verbose: bool = True) -> None:
        import time as _time
        t0 = _time.time()
        for (a, b), neuron in self.scalpel.neurons.items():
            entry = (b, neuron.weight, neuron.p_sig)
            if a in self._index:
                self._index[a].append(entry)
            else:
                self._index[a] = [entry]
            entry_b = (a, neuron.weight, neuron.p_sig)
            if b in self._index:
                self._index[b].append(entry_b)
            else:
                self._index[b] = [entry_b]
        for k in self._index:
            self._index[k].sort(key=lambda x: x[1], reverse=True)
        self._indexed = True
        if verbose:
            print(f"  Index: {len(self._index):,} mots en {_time.time()-t0:.1f}s")

    def _corrs(self, word: str) -> list[tuple[str, float, float]]:
        if self._indexed:
            return self._index.get(word, [])
        return self.scalpel.get_correlations(word)

    def _extract_concepts(self, theme: str, n: int = 10) -> list[str]:
        """Extrait les n concepts les plus corrélés au theme (hors stopwords)."""
        corrs = self._corrs(theme)
        concepts = []
        for word, weight, _ in corrs:
            if word not in STOPWORDS and word != theme and len(word) > 1:
                concepts.append(word)
                if len(concepts) >= n:
                    break
        return concepts

    def _select_skeleton_dense(self, theme: str, domain: str | None = None,
                                intention: str | None = None) -> dict | None:
        """Choisit un squelette dans la base dense (13K entrees)."""
        if self.dense_grammar is None:
            return None
        domains = self.dense_grammar.get("domains", {})
        # Mapper le theme vers un domaine
        if domain is None:
            domain = self._theme_to_domain(theme)
        if domain not in domains:
            # fallback : scientific
            domain = "scientific" if "scientific" in domains else list(domains.keys())[0]
        intentions = domains[domain]
        if intention is None:
            intention = self.rng.choice(list(intentions.keys()))
        if intention not in intentions:
            intention = list(intentions.keys())[0]
        entries = intentions[intention]
        if entries:
            return self.rng.choice(entries)
        return None

    def _theme_to_domain(self, theme: str) -> str:
        """Mappe un mot-thème vers un domaine de la base dense."""
        domain_map = {
            "quantum": "quantum_information", "physics": "scientific",
            "science": "scientific", "mathematics": "mathematics",
            "math": "mathematics", "love": "emotional_reflection",
            "emotion": "emotional_reflection", "brain": "medicine",
            "neural": "medicine", "gravity": "scientific",
            "chemistry": "scientific", "biology": "scientific",
            "history": "history", "music": "arts", "art": "arts",
            "philosophy": "philosophy", "consciousness": "philosophy",
            "technology": "technology", "robot": "robotics",
            "ai": "software_development", "code": "software_development",
            "law": "law", "economy": "economy", "climate": "environment",
        }
        return domain_map.get(theme.lower(), "scientific")

    def _select_skeleton(self, theme: str) -> dict:
        """Choisit un squelette adapte au theme (dense en priorite, fallback simple)."""
        # 1. Base dense (13K entrees)
        dense_entry = self._select_skeleton_dense(theme)
        if dense_entry:
            return dense_entry
        # 2. Fallback : squelettes simples
        matching = [s for s in self.skeletons if theme.lower() in s.get("keywords", [])]
        if matching:
            return self.rng.choice(matching)
        general = [s for s in self.skeletons if s["intent"] in ("explain", "describe", "define")]
        return self.rng.choice(general) if general else self.rng.choice(self.skeletons)

    def generate_sentence(self, theme: str, language: str = "en") -> str:
        """Genere une phrase sur un theme en remplissant un squelette."""
        skeleton = self._select_skeleton(theme)
        concepts = self._extract_concepts(theme, n=10)

        if len(concepts) < 3:
            while len(concepts) < 3:
                concepts.append(theme)

        # Detecter le format de template (dense utilise {X}/{Y}/{Z}, simple utilise {concept1})
        template_key = f"template_{language}" if f"template_{language}" in skeleton else language
        if template_key not in skeleton:
            template_key = "en" if "en" in skeleton else "template_en"
        template = skeleton.get(template_key, skeleton.get("en", skeleton.get("template_en", "")))

        # Detecter les slots : {X}, {Y}, {Z} (dense) ou {concept1}, {concept2} (simple)
        import re
        slots_found = re.findall(r"\{(\w+)\}", template)
        if not slots_found:
            return template

        # Remplir les slots avec les concepts (non stopwords, non repetes)
        filled = template
        used = set()
        concept_idx = 0
        for slot in slots_found:
            # Chercher le concept suivant non utilise
            chosen = None
            for c in concepts[concept_idx:]:
                if c not in used and c not in STOPWORDS:
                    chosen = c
                    concept_idx = concepts.index(c) + 1
                    break
            if chosen is None:
                # fallback : theme ou concept avec stopword
                for c in concepts[concept_idx:]:
                    if c not in used:
                        chosen = c
                        concept_idx = concepts.index(c) + 1
                        break
            if chosen is None:
                chosen = theme
            used.add(chosen)
            filled = filled.replace(f"{{{slot}}}", chosen, 1)

        return filled

    def generate_paragraph(self, theme: str, n_sentences: int = 5,
                           language: str = "en") -> str:
        """Genere un paragraphe de plusieurs phrases sur le meme theme."""
        sentences = []
        for i in range(n_sentences):
            self.rng = random.Random(42 + i * 7)  # diversifier
            sent = self.generate_sentence(theme, language)
            sentences.append(sent)
        return " ".join(sentences)

    def generate_response(self, query: str, language: str = "en") -> dict[str, Any]:
        """Genere une reponse a une requete utilisateur.

        Extrait le mot-cle principal de la requete, choisit un squelette et
        remplit les concepts.
        """
        # Extraction simple du mot-cle = le mot le plus long non-stopword
        words = query.lower().strip().split()
        keyword = None
        for w in sorted(words, key=len, reverse=True):
            if w not in STOPWORDS and w in self._index if self._indexed else True:
                keyword = w
                break
        if keyword is None:
            keyword = words[0] if words else "science"

        sentence = self.generate_sentence(keyword, language)
        paragraph = self.generate_paragraph(keyword, n_sentences=3, language=language)

        # Extraire les concepts pour la transparence
        concepts = self._extract_concepts(keyword, n=8)
        skeleton = self._select_skeleton(keyword)

        # Extraire l'intention (dense: selection.intention, simple: intent)
        if "selection" in skeleton:
            intent = skeleton["selection"].get("intention", "unknown")
            skel_id = skeleton.get("id", "unknown")
        else:
            intent = skeleton.get("intent", "unknown")
            skel_id = skeleton.get("id", "unknown")

        return {
            "query": query,
            "keyword": keyword,
            "concepts": concepts,
            "skeleton_id": skel_id,
            "skeleton_intent": intent,
            "sentence": sentence,
            "paragraph": paragraph,
        }


if __name__ == "__main__":
    from pathlib import Path
    from ratis_net.glove_tokenizer import GloveTokenizer

    tok = GloveTokenizer(dim=12, n_glove=8)
    scalpel = ScalpelLayer(tok, eta=0.1, coherence_threshold=0.3)
    scalpel.load(Path("artifacts/scalpel_wikipedia.pkl"))
    print(f"Scalpel: {scalpel.network_size():,} neurons\n")

    speaker = SkeletonSpeaker(scalpel)
    speaker.build_index()

    print("=== Skeleton Speaker : concepts + squelettes syntaxiques ===\n")

    themes = ["quantum", "science", "love", "gravity", "brain", "music", "history", "consciousness"]
    for theme in themes:
        result = speaker.generate_response(f"what is {theme}")
        print(f"[{theme}]")
        print(f"  Concepts: {result['concepts'][:5]}")
        print(f"  Squelette: {result['skeleton_id']} ({result['skeleton_intent']})")
        print(f"  Phrase: {result['sentence']}")
        print()

    print("=== Paragraphes longs ===\n")
    for theme in ["quantum", "love", "consciousness"]:
        print(f"[{theme}]")
        para = speaker.generate_paragraph(theme, n_sentences=4)
        print(f"  {para}\n")
