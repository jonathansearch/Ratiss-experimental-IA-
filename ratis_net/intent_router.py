"""ratis_net.intent_router — Routage de la requête vers la bonne source.

Le Scalpel ne décide pas DE QUOI on parle : il fournit des associations.
Le routeur décide quelle ressource répond :

  - Salutations / gratitude / identité → matrice conversationnelle
    (social_contexts.casual_chat / introduction, 24 000 formulations).
  - Questions factuelles (definition, explanation, comparison...) →
    grammaire dense (18 domaines, 12 intentions) + knowledge packs.
  - Requête sans concept connu → recherche web (complément honnête).

Ce module ne génère pas de texte : il retourne un Route (où puiser le
squelette, avec quelle intention, quel registre). Le SkeletonSpeaker remplit.
"""
from __future__ import annotations

from dataclasses import dataclass

from ratis_net.query_analyzer import QueryAnalysis

# ── Type de question → intention de la grammaire dense ───────────────────────
QTYPE_TO_INTENTION = {
    "definition": "define",
    "explanation": "explain",
    "comparison": "compare",
    "yesno": "qualify",
    "opinion": "argue",
    "statement": "synthesize",
    "command": "instruct",
    "capability": "report",
}

# ── Type de question → (contexte social, intention conversationnelle) ────────
QTYPE_TO_SOCIAL = {
    "greeting": ("casual_chat", "greet"),
    "farewell": ("casual_chat", "share"),
    "gratitude": ("casual_chat", "encourage"),
    "identity": ("introduction", "reflect"),
    "capability": ("collaboration", "organize"),
}

# ── Mots-clés → domaine dense (complété par Gazetteer du speaker) ────────────
KEYWORD_DOMAIN = {
    "quantum": "quantum_information", "qubit": "quantum_information",
    "decoherence": "quantum_information", "entanglement": "quantum_information",
    "physics": "scientific", "science": "scientific", "gravity": "scientific",
    "chemistry": "scientific", "biology": "scientific", "evolution": "scientific",
    "mathematics": "mathematics", "math": "mathematics", "algebra": "mathematics",
    "geometry": "mathematics", "topology": "mathematics",
    "love": "emotional_reflection", "emotion": "emotional_reflection",
    "fear": "emotional_reflection", "joy": "emotional_reflection",
    "brain": "medicine", "neuron": "medicine", "medicine": "medicine",
    "dna": "medicine", "protein": "medicine", "virus": "medicine",
    "history": "history", "war": "history", "empire": "history",
    "music": "arts", "art": "arts", "painting": "arts", "cinema": "arts",
    "philosophy": "philosophy", "consciousness": "philosophy",
    "ethics": "philosophy", "metaphysics": "philosophy",
    "technology": "technology", "computer": "technology",
    "robot": "robotics", "robotics": "robotics",
    "ai": "software_development", "code": "software_development",
    "software": "software_development", "programming": "software_development",
    "law": "law", "justice": "law", "court": "law",
    "economy": "economy", "market": "economy", "finance": "economy",
    "climate": "environment", "environment": "environment",
    "photosynthesis": "environment", "ecosystem": "environment",
    "education": "education", "school": "education", "learning": "education",
    "machine": "technology", "black": "scientific", "hole": "scientific",
    "star": "scientific", "galaxy": "scientific", "universe": "scientific",
    "atom": "scientific", "molecule": "scientific", "cell": "medicine",
}


@dataclass
class Route:
    """Décision de routage pour une requête."""
    social: bool
    context: str | None = None      # ex: casual_chat (matrice conversationnelle)
    conv_intent: str | None = None  # ex: greet
    domain: str = "scientific"      # domaine de la grammaire dense
    intention: str = "explain"      # intention de la grammaire dense
    keywords: list[str] | None = None
    compounds: list[str] | None = None


def route(analysis: QueryAnalysis,
           domain_lookup: dict[str, str] | None = None) -> Route:
    """Décide où puiser le squelette de réponse."""
    lookup = domain_lookup or KEYWORD_DOMAIN
    qtype = analysis.qtype

    # 1. Social : matrice conversationnelle
    if qtype in QTYPE_TO_SOCIAL:
        context, conv_intent = QTYPE_TO_SOCIAL[qtype]
        return Route(social=True, context=context, conv_intent=conv_intent,
                     keywords=analysis.keywords, compounds=analysis.compounds)

    # 2. Factuel : grammaire dense, domaine déduit des mots-clés
    domain = "scientific"
    for word in analysis.all_concepts:
        if word in lookup:
            domain = lookup[word]
            break
    intention = QTYPE_TO_INTENTION.get(qtype, "explain")
    return Route(social=False, domain=domain, intention=intention,
                 keywords=analysis.keywords, compounds=analysis.compounds)
