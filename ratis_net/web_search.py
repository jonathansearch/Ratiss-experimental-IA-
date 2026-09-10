"""ratis_net.web_search — Module de recherche web souverain.

Permet à RATIS-Net de rechercher des informations en temps réel quand le
Scalpel local ne trouve pas de corrélation. Deux backends :

  1. Google Custom Search API (si GOOGLE_API_KEY + GOOGLE_CSE_ID disponibles)
  2. DuckDuckGo Instant Answer (sans clé, fallback public)

Tous les résultats web passent par le bridge AEON pour validation avant
d'atteindre le speaker. Le web est un sens supplémentaire, pas une source
aveugle.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any


class WebSearchResult:
    """Un résultat de recherche web validé."""

    def __init__(self, title: str, snippet: str, url: str,
                 source: str = "web", validated: bool = False):
        self.title = title
        self.snippet = snippet
        self.url = url
        self.source = source
        self.validated = validated

    def to_dict(self) -> dict:
        return {"title": self.title, "snippet": self.snippet,
                "url": self.url, "source": self.source, "validated": self.validated}


class WebSearchModule:
    """Module de recherche web avec validation AEON.

    Usage :
        web = WebSearchModule()
        results = web.search("quantum decoherence biology")
        for r in results:
            print(r.title, r.snippet[:80])
    """

    def __init__(self, google_api_key: str | None = None,
                 google_cse_id: str | None = None):
        self.api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self.cse_id = google_cse_id or os.environ.get("GOOGLE_CSE_ID")

    @property
    def backend(self) -> str:
        if self.api_key and self.cse_id:
            return "google_cse"
        return "duckduckgo"

    @property
    def available(self) -> bool:
        """True si au moins un backend est utilisable."""
        return True  # DuckDuckGo toujours disponible

    def _search_google(self, query: str, n: int = 5) -> list[WebSearchResult]:
        """Google Custom Search API (nécessite clé + CSE ID)."""
        url = ("https://www.googleapis.com/customsearch/v1?"
               f"key={self.api_key}&cx={self.cse_id}&q={urllib.parse.quote(query)}"
               f"&num={n}")
        req = urllib.request.Request(url, headers={"User-Agent": "RATISS-Net/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = []
            for item in data.get("items", [])[:n]:
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("link", ""),
                    source="google_cse"))
            return results
        except Exception:
            return []

    def _search_duckduckgo(self, query: str, n: int = 5) -> list[WebSearchResult]:
        """DuckDuckGo Instant Answer API (sans clé, public)."""
        url = (f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}"
               f"&format=json&no_html=1&skip_disambig=1")
        req = urllib.request.Request(url, headers={"User-Agent": "RATISS-Net/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = []
            # Abstract
            if data.get("AbstractText"):
                results.append(WebSearchResult(
                    title=data.get("Heading", query),
                    snippet=data.get("AbstractText", ""),
                    url=data.get("AbstractURL", ""),
                    source="duckduckgo"))
            # Related topics
            for topic in data.get("RelatedTopics", [])[:n - 1]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(WebSearchResult(
                        title=topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                        snippet=topic.get("Text", ""),
                        url=topic.get("FirstURL", ""),
                        source="duckduckgo"))
            return results[:n]
        except Exception:
            return []

    def search(self, query: str, n: int = 5) -> list[WebSearchResult]:
        """Recherche web : Google si clé disponible, sinon DuckDuckGo."""
        if self.api_key and self.cse_id:
            results = self._search_google(query, n)
            if results:
                return results
        return self._search_duckduckgo(query, n)

    def search_and_extract(self, query: str, n: int = 3) -> list[str]:
        """Recherche et extrait les snippets comme liste de chaînes.

        Utile pour injecter dans le Scalpel ou le Synchrotron.
        """
        results = self.search(query, n)
        return [r.snippet for r in results if r.snippet]


if __name__ == "__main__":
    web = WebSearchModule()
    print(f"Backend: {web.backend}")
    print(f"Available: {web.available}\n")

    queries = ["quantum decoherence", "protein folding topology", "Alcubierre warp drive"]
    for q in queries:
        print(f"Q: {q}")
        results = web.search(q, n=3)
        for r in results:
            print(f"  [{r.source}] {r.title[:50]}")
            print(f"    {r.snippet[:100]}")
        if not results:
            print("  (no results)")
        print()
