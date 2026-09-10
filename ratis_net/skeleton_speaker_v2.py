"""ratis_net.skeleton_speaker_v2 — Génération par squelettes, pipeline routé.

Différences avec skeleton_speaker v1 (conservé pour compatibilité) :

  1. La requête passe par query_analyzer (langue, type, mots-clés, composés).
  2. intent_router choisit la source : matrice conversationnelle (social) ou
     grammaire dense (factuel, intention déduite du type de question).
  3. concept_ranker classe les concepts par IDF de degré + voisinage partagé
     + complément GloVe kNN — fini les co-occurrences ubiquitaires en tête.
  4. Les slots {PERSON} de la matrice conversationnelle reçoivent un
     interlocuteur neutre ("my friend"), pas un concept.

Le Scalpel reste la source des associations ; ce module ne change ni la loi
LCT ni le checkpoint — il corrige la LECTURE du réseau.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

try:
    from ratis_net.scalpel import ScalpelLayer
except ImportError:
    from scalpel import ScalpelLayer

from ratis_net.query_analyzer import (QueryAnalysis, STOPWORDS_EN,
                                      STOPWORDS_FR, analyze)
from ratis_net.concept_ranker import ConceptRanker
from ratis_net.intent_router import KEYWORD_DOMAIN, Route, route

STOPWORDS = STOPWORDS_EN | STOPWORDS_FR

# Interlocuteur pour les slots {PERSON} de la matrice conversationnelle
_PERSON_FILLER = {"en": "my friend", "fr": "mon ami"}
# Remplissage neutre pour les slots abstraits sans concept ({X}/{Y}/{Z})
_GENERIC_FILLER = {"en": "this", "fr": "ce sujet"}

# Réponses figées : l'identité et les capacités ne s'inventent pas, elles
# se déclarent. (Honnêteté : pas de fausse personnalité, pas de promesse.)
_IDENTITY = {
    "en": ("I am RATIS-Net — a neural network trained by the Law of "
           "Topological Coherence (LCT) rather than gradient descent, "
           "created by Jonathan Evina and JOHNKING0. I answer by reading "
           "learned word correlations, curated knowledge facts, and web "
           "search; I do not think like a human."),
    "fr": ("Je suis RATIS-Net — un réseau neuronal entraîné par la Loi de "
           "Cohérence Topologique (LCT) plutôt que par descente de gradient, "
           "créé par Jonathan Evina et JOHNKING0. Je réponds en lisant des "
           "corrélations apprises, des faits validés et la recherche web ; "
           "je ne pense pas comme un humain."),
}
_CAPABILITY = {
    "en": ("I can discuss concepts from my learned corpus, give curated "
           "scientific facts, trace association chains between ideas, and "
           "search the web when my knowledge is missing. I cannot guarantee "
           "truth: I show correlations, not certainties."),
    "fr": ("Je peux discuter des concepts de mon corpus appris, donner des "
           "faits scientifiques validés, tracer des chaînes d'association "
           "entre idées et chercher sur le web quand il me manque un savoir. "
           "Je ne garantis pas la vérité : je montre des corrélations, "
           "pas des certitudes."),
}


class SkeletonSpeakerV2:
    """Speaker routé : conversation pour le social, grammaire dense sinon.

    API compatible avec SkeletonSpeaker (generate_sentence, generate_paragraph,
    generate_response, build_index, dense_grammar, conversation_matrix,
    _index, _corrs, STOPWORDS).
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
        self._ranker: ConceptRanker | None = None

        base = Path(__file__).resolve().parent

        if dense_path is None:
            dense_path = base.parent / "data" / "grammar_domains" / "dense_syntax_skeletons.json"
        self.dense_grammar = None
        if Path(dense_path).exists():
            with open(dense_path, encoding="utf-8") as f:
                self.dense_grammar = json.load(f)

        if conversation_path is None:
            conversation_path = base.parent / "data" / "grammar_domains" / "conversation_matrix.json"
        self.conversation_matrix = None
        if Path(conversation_path).exists():
            with open(conversation_path, encoding="utf-8") as f:
                self.conversation_matrix = json.load(f)

        if skeletons_path is None:
            skeletons_path = base / "syntax_skeletons.json"
        with open(skeletons_path, encoding="utf-8") as f:
            self.skeletons = json.load(f)

    # ── Index inversé du Scalpel ────────────────────────────────────────────
    def build_index(self, verbose: bool = True) -> None:
        import time as _time
        t0 = _time.time()
        for (a, b), neuron in self.scalpel.neurons.items():
            self._index.setdefault(a, []).append((b, neuron.weight, neuron.p_sig))
            self._index.setdefault(b, []).append((a, neuron.weight, neuron.p_sig))
        for k in self._index:
            self._index[k].sort(key=lambda x: x[1], reverse=True)
        self._indexed = True
        self._ranker = ConceptRanker(self._index)
        if verbose:
            print(f"  Index: {len(self._index):,} mots en {_time.time()-t0:.1f}s")

    def _corrs(self, word: str) -> list[tuple[str, float, float]]:
        if self._indexed:
            return self._index.get(word, [])
        return self.scalpel.get_correlations(word)

    def _extract_concepts(self, theme: str, n: int = 10) -> list[str]:
        """Concepts autour d'un thème, classés par pertinence (IDF+GloVe)."""
        if self._ranker is not None:
            return self._ranker.rank([theme.lower()], n=n)
        # Secours : poids brut (v1)
        out = []
        for word, _w, _p in self._corrs(theme):
            if word not in STOPWORDS and word != theme and len(word) > 1:
                out.append(word)
                if len(out) >= n:
                    break
        return out

    def concepts_for(self, analysis: QueryAnalysis, n: int = 10) -> list[str]:
        """Concepts pour une requête complète (mots-clés + composés)."""
        if self._ranker is None:
            return self._extract_concepts(analysis.primary, n=n)
        kws = analysis.all_concepts[:4]
        if not kws:
            return []
        return self._ranker.rank(kws, n=n)

    # ── Sélection de squelette ──────────────────────────────────────────────
    def _dense_entries(self, domain: str, intention: str) -> list[dict]:
        if self.dense_grammar is None:
            return []
        domains = self.dense_grammar.get("domains", {})
        intentions = domains.get(domain) or domains.get("scientific", {})
        return intentions.get(intention) or intentions.get(
            "explain") or (list(intentions.values())[0] if intentions else [])

    def _social_entries(self, context: str, intent: str) -> list[dict]:
        if self.conversation_matrix is None:
            return []
        social = self.conversation_matrix.get("social_contexts", {})
        ctx = social.get(context) or social.get("casual_chat", {})
        entries = ctx.get(intent) or ctx.get("share") or []
        if not entries:
            for v in ctx.values():
                if isinstance(v, list) and v:
                    entries = v
                    break
        return entries if isinstance(entries, list) else []

    # Tonalités préférées par intention sociale (évite "frustrated" en
    # salutation ; les layers sont des choix d'étiquettes, pas des émotions).
    _LAYER_PREF = {
        "greet": {"calm", "joy", "gratitude"},
        "share": {"calm", "joy"},
        "encourage": {"calm", "gratitude"},
        "reflect": {"calm"},
        "organize": {"calm"},
        "listen": {"calm", "concern"},
        "clarify": {"calm", "curiosity"},
        "apologize": {"concern", "calm"},
        "negotiate": {"calm"},
    }

    def _pick(self, entries: list[dict], language: str,
              intent: str | None = None) -> dict | None:
        """Choix d'une entrée : registre neutre, langue dispo, tonalité adaptée."""
        if not entries:
            return None
        neutral = [e for e in entries if e.get("register") in
                   ("neutral", "casual", "standard")]
        pool = neutral or entries
        pool = [e for e in pool if e.get(language) or e.get(f"template_{language}")] or pool
        preferred = self._LAYER_PREF.get(intent or "")
        if preferred:
            layered = [e for e in pool if e.get("emotional_layer") in preferred]
            pool = layered or pool
        return self.rng.choice(pool)

    def _select_entry(self, route_: Route, language: str,
                      n_concepts: int = 3) -> dict | None:
        if route_.social:
            return self._pick(self._social_entries(route_.context or "casual_chat",
                                                    route_.conv_intent or "greet"),
                              language, intent=route_.conv_intent)
        entries = self._dense_entries(route_.domain, route_.intention)
        # Pool de concepts réduit → préférer les squelettes à peu de slots
        # (évite "X organise X afin d'interpréter X" quand un seul concept).
        if n_concepts < 3 and entries:
            max_slots = max(n_concepts, 1) + 1
            few = [e for e in entries
                   if len(re.findall(r"\{(\w+)\}",
                                     self._template_of(e, language))) <= max_slots]
            entries = few or entries
        return self._pick(entries, language, intent=route_.intention)

    # ── Remplissage ─────────────────────────────────────────────────────────
    _META_MARKERS = re.compile(
        r"(registre|register|tonalité|tone\b|rythme de l'échange|pace of the "
        r"exchange|hypothèses de départ|starting assumptions|observations des "
        r"interprétations|observations from interpretations|niveau de preuve|"
        r"level of evidence|cadre|contexte|frame|context|simulation|réflexion|"
        r"reflection|lecture|reading|situation|pratique|practice|analyse|"
        r"analysis|échange|exchange|collaboration|introduction|soutien|support|"
        r"familial|family|créatif|creative|recherche|research|vérifier|"
        r"verification|apprentissage|learning)", re.IGNORECASE)

    @staticmethod
    def _strip_meta_tail(template: str) -> str:
        """Supprime la queue de métalangage après le dernier slot {…}.

        Vérifié empiriquement sur les 13 000 gabarits denses et les 24 000
        formulations conversationnelles : tout le texte situé après le
        dernier slot est du métalangage (« in a neutral register », « avec
        une tonalité calme », « dans un cadre scientifique »…), jamais du
        contenu. On garde la ponctuation finale du template.
        """
        matches = list(re.finditer(r"\{(\w+)\}", template))
        if not matches:
            return template
        core = template[:matches[-1].end()]
        tail = template[matches[-1].end():]
        if not tail.strip():
            return core + "."
        # Garder les segments de tête non-méta (« aujourd'hui » est du
        # contenu) ; couper dès le premier segment méta (registre, tonalité,
        # cadre, niveau de preuve, rythme de l'échange…).
        kept: list[str] = []
        for seg in tail.split(","):
            if SkeletonSpeakerV2._META_MARKERS.search(seg):
                break
            kept.append(seg)
        kept_tail = ",".join(kept).strip().rstrip(" ,;:")
        if kept_tail:
            if kept_tail[-1] not in ".!?…":
                kept_tail += "."
            sep = "" if kept_tail[:1] in ".,;:!?)]" else " "
            return core + sep + kept_tail
        m = re.search(r"[.!?…]+\s*$", tail)
        return core + (m.group(0).strip() if m else ".")

    @staticmethod
    def _polish(text: str) -> str:
        """Finitions typographiques : espaces doubles, « , » orphelines,
        majuscule initiale, point final."""
        t = re.sub(r"\s+", " ", text).strip()
        t = re.sub(r"\s+([,.])", r"\1", t)
        if t and t[0].islower():
            t = t[0].upper() + t[1:]
        if t and t[-1] not in ".!?…»":
            t += "."
        return t

    @staticmethod
    def _template_of(entry: dict, language: str) -> str:
        for key in (f"template_{language}", language, "template_en", "en"):
            if entry.get(key):
                return SkeletonSpeakerV2._strip_meta_tail(entry[key])
        return ""

    def _fill(self, template: str, concepts: list[str],
              language: str, theme: str) -> str:
        slots = re.findall(r"\{(\w+)\}", template)
        if not slots:
            return template
        filled = template
        used: set[str] = set()
        pool = [c for c in concepts if c not in STOPWORDS and len(c) > 1]
        for slot in slots:
            if slot == "PERSON":
                filled = filled.replace("{PERSON}", _PERSON_FILLER.get(language, "my friend"), 1)
                continue
            chosen = next((c for c in pool if c not in used), None)
            if chosen is None:
                chosen = theme or _GENERIC_FILLER.get(language, "this")
            used.add(chosen)
            filled = filled.replace(f"{{{slot}}}", chosen, 1)
        return self._polish(filled)

    # ── API publique (compatible v1) ────────────────────────────────────────
    def generate_sentence(self, theme: str, language: str = "en") -> str:
        analysis = analyze(f"what is {theme}", language=language)
        analysis.keywords = [theme]
        analysis.compounds = []
        route_ = route(analysis)
        if route_.social:  # un thème nu n'est jamais social
            route_ = Route(social=False, domain=KEYWORD_DOMAIN.get(theme, "scientific"),
                           intention="explain")
        entry = self._select_entry(route_, language)
        concepts = self._extract_concepts(theme, n=8)
        if entry is None:
            return self._fallback_sentence(theme, concepts, language)
        return self._fill(self._template_of(entry, language), concepts,
                          language, theme)

    def generate_paragraph(self, theme: str, n_sentences: int = 5,
                           language: str = "en") -> str:
        sentences = []
        seen: set[str] = set()
        intentions = ["define", "explain", "describe", "qualify", "synthesize",
                      "report", "narrate"]
        for i in range(n_sentences):
            analysis = analyze(f"what is {theme}", language=language)
            analysis.keywords = [theme]
            intention = intentions[i % len(intentions)]
            route_ = Route(social=False,
                           domain=KEYWORD_DOMAIN.get(theme, "scientific"),
                           intention=intention)
            rng = random.Random(1000 + i)
            entries = self._dense_entries(route_.domain, route_.intention)
            entry = rng.choice(entries) if entries else None
            concepts = self._extract_concepts(theme, n=10)
            # varier les concepts entre les phrases
            concepts = concepts[i % 3:] + concepts[:i % 3]
            if entry is None:
                sent = self._fallback_sentence(theme, concepts, language)
            else:
                sent = self._fill(self._template_of(entry, language), concepts,
                                  language, theme)
            if sent not in seen:
                sentences.append(sent)
                seen.add(sent)
        return " ".join(sentences)

    def _fallback_sentence(self, theme: str, concepts: list[str],
                           language: str) -> str:
        """Squelette minimal si aucune entrée ne correspond."""
        c = concepts[0] if concepts else theme
        if language == "fr":
            return f"{theme.capitalize()} est lié à {c} dans le corpus appris."
        return f"{theme.capitalize()} is related to {c} in the learned corpus."

    def generate_response(self, query: str, language: str | None = None) -> dict[str, Any]:
        """Réponse routée : social → matrice conversationnelle ; factuel → dense."""
        analysis = analyze(query, language=language)
        # Identité et capacités : réponses figées et honnêtes, pas de gabarit
        if analysis.qtype == "identity":
            sent = _IDENTITY.get(analysis.language, _IDENTITY["en"])
            return {"query": query, "keyword": "ratis-net",
                    "qtype": analysis.qtype, "language": analysis.language,
                    "keywords": analysis.keywords, "compounds": [],
                    "concepts": [], "skeleton_id": "identity",
                    "skeleton_intent": "identity", "route": "fixed:identity",
                    "sentence": sent, "paragraph": sent}
        if analysis.qtype == "capability":
            sent = _CAPABILITY.get(analysis.language, _CAPABILITY["en"])
            return {"query": query, "keyword": "capabilities",
                    "qtype": analysis.qtype, "language": analysis.language,
                    "keywords": analysis.keywords, "compounds": [],
                    "concepts": [], "skeleton_id": "capability",
                    "skeleton_intent": "report", "route": "fixed:capability",
                    "sentence": sent, "paragraph": sent}
        route_ = route(analysis)
        if route_.social:
            # Social : pas de concepts de corpus — les slots reçoivent un
            # interlocuteur neutre ({PERSON}) ou "this" pour {X}/{Y}/{Z}.
            concepts = []
        elif analysis.language == "fr":
            # Le Scalpel est anglophone : pour le français, on remplit les
            # squelettes FR avec les mots de la requête (vrais mots FR),
            # pas avec des concepts anglais traduits à l'aveugle.
            concepts = analysis.all_concepts[:10]
            if len(concepts) < 2:
                # Un seul concept FR → les gabarits le répéteraient dans
                # chaque slot. Réponse déclarative honnête à la place.
                theme0 = concepts[0] if concepts else ""
                sent = (f"« {theme0} » : le corpus appris est anglophone et "
                        f"la base de connaissances validées ne couvre pas "
                        f"encore ce terme en français — la recherche web "
                        f"peut compléter.") if theme0 else (
                        "Je n'ai pas identifié de concept précis dans votre "
                        "phrase — pouvez-vous la reformuler ?")
                return {"query": query, "keyword": theme0,
                        "qtype": analysis.qtype, "language": "fr",
                        "keywords": analysis.keywords,
                        "compounds": analysis.compounds,
                        "concepts": concepts, "skeleton_id": "fr_honest_fallback",
                        "skeleton_intent": "define", "route": "dense:fr_fallback",
                        "sentence": sent, "paragraph": sent}
        else:
            concepts = self.concepts_for(analysis, n=10)
        entry = self._select_entry(route_, analysis.language,
                                   n_concepts=len(concepts))
        theme = analysis.primary or (analysis.keywords[0] if analysis.keywords else "")

        if entry is None:
            sentence = self._fallback_sentence(theme, concepts, analysis.language)
            skel_id, intent = "fallback", "none"
        else:
            sentence = self._fill(self._template_of(entry, analysis.language),
                                  concepts, analysis.language, theme)
            skel_id = entry.get("id", "unknown")
            intent = route_.conv_intent if route_.social else route_.intention

        paragraph = self.generate_paragraph(theme, n_sentences=3,
                                            language=analysis.language) if theme else sentence

        return {
            "query": query,
            "keyword": theme,
            "qtype": analysis.qtype,
            "language": analysis.language,
            "keywords": analysis.keywords,
            "compounds": analysis.compounds,
            "concepts": concepts,
            "skeleton_id": skel_id,
            "skeleton_intent": intent,
            "route": "conversation" if route_.social else f"dense:{route_.domain}",
            "sentence": sentence,
            "paragraph": paragraph,
        }
