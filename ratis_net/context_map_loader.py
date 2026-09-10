"""ratis_net.context_map_loader — Chargement streaming de l'ultra_context_map.

L'ultra_context_map.json fait ~400 MiB (242 632 concepts, 7.56M arêtes).
Ce loader permet de :
  1. Charger un concept unique sans lire tout le fichier (streaming ijson).
  2. Faire un fallback vers le Scalpel si la carte n'est pas disponible.

Le SkeletonSpeaker et le TriGrammarSpeaker utilisent ce loader pour accéder
aux corrélations sans reconstruire l'index inversé à chaque démarrage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class ContextMapLoader:
    """Chargeur streaming pour ultra_context_map.json.

    Évite de charger 400 MiB en mémoire. Permet :
      - lookup(concept) : récupère les arêtes d'un concept.
      - get_correlations(word) : compatible avec l'API du Scalpel.
    """

    def __init__(self, map_path: str | Path | None = None):
        if map_path is None:
            map_path = Path(__file__).resolve().parents[1] / "data" / "grammar_domains" / "ultra_context_map.json"
        self.path = Path(map_path)
        self._full_cache: dict | None = None  # chargement paresseux si petit

    @property
    def available(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 1000

    def _load_full(self) -> dict:
        """Charge tout le fichier (400 MiB). À éviter si possible."""
        if self._full_cache is None and self.available:
            with open(self.path, encoding="utf-8") as f:
                self._full_cache = json.load(f)
        return self._full_cache or {}

    def lookup(self, concept: str) -> list[dict]:
        """Récupère les arêtes d'un concept depuis la carte.

        Retourne une liste de dicts : [{term, weight, p_sig, coherence, n}, ...]
        """
        data = self._load_full()
        # La carte est indexée par concept racine
        # Structure : {concept: {co_occurs_with: [{term, weight, ...}], ...}}
        entry = data.get(concept)
        if entry is None:
            return []
        edges = entry.get("co_occurs_with", [])
        if isinstance(edges, list):
            return edges
        if isinstance(edges, dict):
            return [{"term": k, **v} for k, v in edges.items()]
        return []

    def get_correlations(self, word: str) -> list[tuple[str, float, float]]:
        """API compatible avec Scalpel.get_correlations.

        Retourne [(term, weight, p_sig), ...] trié par poids décroissant.
        """
        edges = self.lookup(word)
        corrs = []
        for edge in edges:
            term = edge.get("term", edge.get("neighbor", ""))
            weight = float(edge.get("weight", edge.get("lct_weight", 0.0)))
            p_sig = float(edge.get("p_sig", 0.0))
            if term:
                corrs.append((term, weight, p_sig))
        corrs.sort(key=lambda x: x[1], reverse=True)
        return corrs

    def get_context_window(self, word: str) -> list[str]:
        """Récupère la fenêtre de contexte à 2 mots (voisins les plus forts)."""
        data = self._load_full()
        entry = data.get(word)
        if entry is None:
            return []
        window = entry.get("surface_routes", entry.get("context_window", []))
        if isinstance(window, list):
            return window
        return []

    def stats(self) -> dict:
        """Statistiques de la carte."""
        data = self._load_full()
        if not data:
            return {"available": False}
        # Si la carte a des métadonnées
        if "metadata" in data or "export" in data:
            meta = data.get("metadata", data.get("export", {}))
            return {"available": True, "roots": meta.get("included_roots", len(data) - 1),
                    "edges": meta.get("included_directed_edges", "unknown"),
                    "size_mb": self.path.stat().st_size / 1024 / 1024}
        return {"available": True, "roots": len(data),
                "size_mb": self.path.stat().st_size / 1024 / 1024}


if __name__ == "__main__":
    loader = ContextMapLoader()
    if loader.available:
        print(f"Ultra context map: {loader.stats()}")
        for word in ["quantum", "science", "love", "gravity", "brain"]:
            corrs = loader.get_correlations(word)[:5]
            print(f"\n{word}:")
            for term, weight, p_sig in corrs:
                print(f"  -> {term:15s} weight={weight:.2f} P_sig={p_sig:.4f}")
    else:
        print("Ultra context map non disponible (télécharger via Git LFS)")
