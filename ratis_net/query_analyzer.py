"""ratis_net.query_analyzer — Analyse des requêtes utilisateur.

Transforme une requête brute en structure exploitable par le routeur :

  1. Détection de langue (FR / EN) par mots-outils.
  2. Classification du type de question (salutation, définition, explication,
     comparaison, identité, capacité, gratitude, opinion, commande, énoncé).
  3. Extraction des mots-clés : ponctuation nettoyée, stopwords filtrés,
     composés détectés ("black hole", "machine learning" — le Scalpel filtre
     les paires à cos(GloVe) < 0.3, donc les composés sont reconstruits à
     l'analyse, pas cherchés comme paire stockée).

Ce module ne produit aucune réponse : il prépare la décision du routeur
(intent_router) et le remplissage des squelettes (skeleton_speaker).
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

# ── Stopwords (forme seulement ; jamais de concepts) ─────────────────────────
STOPWORDS_EN = {
    "the", "a", "an", "of", "in", "and", "to", "is", "was", "were", "that",
    "this", "it", "for", "on", "with", "as", "by", "at", "from", "or", "be",
    "has", "have", "had", "but", "not", "he", "she", "his", "her", "its",
    "their", "they", "which", "who", "are", "been", "also", "than", "then",
    "so", "if", "can", "will", "would", "could", "should", "may", "might",
    "must", "do", "does", "did", "no", "yes", "all", "any", "some", "each",
    "every", "such", "one", "two", "three", "first", "second", "last", "more",
    "most", "less", "much", "many", "few", "other", "same", "different",
    "what", "how", "why", "when", "where", "whom", "whose", "me", "you",
    "your", "yours", "my", "mine", "we", "us", "our", "ours", "i", "am",
    "tell", "explain", "define", "describe", "give", "about", "into", "up",
    "out", "over", "under", "between", "through", "during", "before", "after",
}

STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à", "au",
    "aux", "en", "dans", "sur", "sous", "par", "pour", "avec", "sans", "est",
    "sont", "était", "étaient", "être", "avoir", "a", "ont", "ce", "cet",
    "cette", "ces", "qui", "que", "quoi", "dont", "où", "quand", "comment",
    "pourquoi", "quel", "quelle", "quels", "quelles", "je", "tu", "il",
    "elle", "nous", "vous", "ils", "elles", "me", "te", "se", "mon", "ton",
    "son", "ma", "ta", "sa", "mes", "tes", "ses", "notre", "votre", "leur",
    "leurs", "ne", "pas", "plus", "moins", "très", "tout", "tous", "toute",
    "toutes", "c'est", "qu'est-ce", "peux", "peut", "pouvez", "dis", "moi",
    "explique", "définis", "décris", "donne", "parle", "ça", "cela", "ceci", "va", "vas", "vont", "allez", "est-ce",
}

# ── Marqueurs de langue ──────────────────────────────────────────────────────
_FR_MARKERS = STOPWORDS_FR - STOPWORDS_EN
_EN_MARKERS = {"the", "is", "are", "what", "how", "why", "you", "your", "me"}

# ── Patterns de type de question ─────────────────────────────────────────────
# ── Mots sociaux : portent l'acte de la requête, pas un sujet de corpus ─────
_SOCIAL_WORDS = {
    "hello", "hi", "hey", "greetings", "morning", "evening", "afternoon",
    "goodbye", "bye", "farewell", "thanks", "thank", "thx", "please",
    "name", "yourself", "friend", "sorry", "welcome", "bonjour", "salut",
    "bonsoir", "merci", "adieu", "bientôt", "au revoir", "ciao",
    # mots de remplissage (intensifieurs sans contenu conceptuel)
    "very", "really", "quite", "much", "just", "actually", "basically",
}

_GREETINGS_EN = {"hello", "hi", "hey", "good morning", "good evening",
                 "good afternoon", "greetings", "yo", "hiya"}
_GREETINGS_FR = {"bonjour", "salut", "bonsoir", "coucou", "hello", "hey"}
_FAREWELL_EN = {"goodbye", "bye", "see you", "farewell", "good night"}
_FAREWELL_FR = {"au revoir", "adieu", "à bientôt", "bonne nuit", "ciao"}
_GRATITUDE_EN = {"thanks", "thank you", "thank", "thx"}
_GRATITUDE_FR = {"merci", "remercie"}


@dataclass
class QueryAnalysis:
    """Résultat de l'analyse d'une requête."""
    raw: str
    language: str = "en"
    qtype: str = "statement"          # greeting|farewell|gratitude|identity|
                                      # capability|definition|explanation|
                                      # comparison|yesno|opinion|command|statement
    keywords: list[str] = field(default_factory=list)   # mots-clés unitaires
    compounds: list[str] = field(default_factory=list)  # composés ("black hole")
    is_question: bool = False

    @property
    def primary(self) -> str:
        """Concept principal : premier composé, sinon premier mot-clé."""
        if self.compounds:
            return self.compounds[0]
        return self.keywords[0] if self.keywords else ""

    @property
    def all_concepts(self) -> list[str]:
        """Tous les mots porteurs de sens (mots-clés décomposés inclus)."""
        out: list[str] = list(self.keywords)
        for comp in self.compounds:
            for w in comp.split():
                if w not in out:
                    out.append(w)
        return out


def _strip_punct(text: str) -> str:
    table = str.maketrans({c: " " for c in string.punctuation if c not in "'-"})
    # normaliser l'apostrophe typographique pour la tokenisation
    return text.replace("\u2019", "'").replace("\u2018", "'").translate(table)


def detect_language(text: str) -> str:
    """Détecte FR ou EN par vote des mots-outils et marqueurs sociaux."""
    words = _strip_punct(text.lower()).split()
    fr = sum(1 for w in words if w in _FR_MARKERS)
    en = sum(1 for w in words if w in _EN_MARKERS)
    # marqueurs univoques : salutations FR, accents dans le texte
    fr += sum(1 for w in words if w in _GREETINGS_FR - {"hello", "hey"})
    if any(c in text for c in "éèêëàâîïôùûç"):
        fr += 1
    return "fr" if fr > en else "en"


def _classify(text: str, words: list[str], language: str) -> str:
    """Classifie le type de question."""
    low = _strip_punct(text.lower()).strip()
    joined = " ".join(words)

    greeting_words = {"hello", "hi", "hey", "yo", "hiya", "bonjour", "salut",
                      "bonsoir", "coucou"}
    if (words and words[0] in greeting_words) or \
       joined.startswith("how are you") or joined.startswith("comment vas") or \
       "ca va" in joined or "ça va" in joined:
        return "greeting"
    farewells = _FAREWELL_FR if language == "fr" else _FAREWELL_EN
    if any(f in joined for f in farewells):
        return "farewell"
    gratitudes = _GRATITUDE_FR if language == "fr" else _GRATITUDE_EN
    if any(g in joined for g in gratitudes):
        return "gratitude"

    identity_markers = (("your", "name"), ("who", "are", "you"), ("qui", "es"),
                        ("ton", "nom"), ("votre", "nom"), ("t'appelles",))
    identity_phrases = ("qui es-tu", "qui etes-vous", "qui êtes-vous",
                        "comment tu t'appelles", "comment t'appelles")
    if any(all(m in words for m in mk) for mk in identity_markers) or \
       any(p in joined for p in identity_phrases):
        return "identity"

    capability_markers = (("can", "you"), ("could", "you"), ("peux", "tu"),
                          ("pouvez", "vous"), ("what", "can", "you"))
    if any(all(m in words for m in mk) for mk in capability_markers):
        return "capability"

    comparison_markers = {"difference", "versus", "vs", "compare",
                          "différence", "comparer", "comparaison"}
    if any(m in words for m in comparison_markers) or \
       ("between" in words) or ("entre" in words and "différence" in joined):
        return "comparison"

    definition_markers_en = ("what",)   # what is X / what are X
    definition_markers_fr = ("qu'est-ce", "quel", "quelle", "définis", "define")
    if language == "fr":
        if any(m in joined for m in definition_markers_fr):
            return "definition"
    else:
        if words[:1] == ["what"] or "define" in words or "meaning" in words:
            return "definition"

    explanation_markers = {"why", "how", "explain", "pourquoi", "comment",
                           "explique", "explain", "cause", "causes", "works"}
    if words and words[0] in explanation_markers:
        return "explanation"

    if low.endswith("?") or (words and words[0] in {"is", "are", "does", "do",
                                                    "can", "est-ce"}):
        return "yesno"
    if any(m in words for m in ("think", "opinion", "believe", "penses",
                                 "avis", "crois")):
        return "opinion"
    if low.endswith("?"):
        return "yesno"
    return "statement"


def _extract_keywords(text: str, language: str) -> tuple[list[str], list[str]]:
    """Extrait mots-clés et composés (bigrammes de mots de contenu adjacents).

    Les mots purement sociaux ("hello", "thank", "your", "name"...) sont
    exclus des mots-clés : ils portent l'ACTE de la requête (saluer,
    remercier, demander l'identité), pas un sujet à explorer dans le corpus.
    """
    stop = STOPWORDS_FR | STOPWORDS_EN | _SOCIAL_WORDS
    words = []
    for w in _strip_punct(text.lower()).split():
        # élision française : "l'énergie" → "énergie", "qu'un" → "un"
        if "'" in w:
            head, _, tail = w.partition("'")
            if head in {"l", "d", "qu", "j", "n", "s", "t", "c", "m"} and tail:
                w = tail
        if len(w) > 1 and not w.isdigit():
            words.append(w)
    content = [w for w in words if w not in stop]
    # Composés : paires adjacentes de mots de contenu dans la requête
    compounds: list[str] = []
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        if a not in stop and b not in stop and len(a) > 2 and len(b) > 2:
            compounds.append(f"{a} {b}")
    return content, compounds


def analyze(query: str, language: str | None = None) -> QueryAnalysis:
    """Analyse complète d'une requête utilisateur."""
    raw = query.strip()
    lang = language or detect_language(raw)
    words = _strip_punct(raw.lower()).split()
    qtype = _classify(raw, words, lang)
    keywords, compounds = _extract_keywords(raw, lang)
    return QueryAnalysis(
        raw=raw,
        language=lang,
        qtype=qtype,
        keywords=keywords,
        compounds=compounds,
        is_question=raw.endswith("?") or qtype not in ("statement", "command"),
    )
