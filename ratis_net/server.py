"""ratis_net.server — Serveur HTTP RATIS-Net (bibliothèque standard seule).

    ratisnet-serve --port 8000
    python -m ratis_net.server --port 8000

Endpoints JSON :
  GET  /health                      → {"status": "ok", ...stats}
  POST /respond    {"q": "..."}     → {"sentence": "...", ...}
  POST /science    {"q": "..."}     → réponse enrichie (faits + preuve LCT)
  POST /concepts   {"word": "..."}  → {"concepts": [...]}
  POST /chain      {"from","to"}    → {"chains": [...]}
  POST /prove      {"concepts":[]}  → empreinte d'intégrité SHA-256
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_MODEL = None  # chargé une fois au démarrage


def _model():
    global _MODEL
    if _MODEL is None:
        from ratis_net import RatisNet
        base = Path(__file__).resolve().parents[1]
        net = RatisNet()
        net.load_scalpel(os.environ.get(
            "RATISNET_SCALPEL", str(base / "artifacts" / "scalpel_wikipedia.pkl")),
            verbose=False)
        data = Path(os.environ.get("RATISNET_DATA", base / "data"))
        net.load_grammar(data / "grammar_domains" / "dense_syntax_skeletons.json",
                         data / "grammar_domains" / "conversation_matrix.json",
                         verbose=False)
        net.load_knowledge_packs(data / "knowledge_packs", verbose=False)
        net.build_index(verbose=False)
        net.load_neural_speaker(verbose=False)  # le cerveau parle
        _MODEL = net
    return _MODEL


class Handler(BaseHTTPRequestHandler):
    server_version = "RatisNet/0.2"

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt, *args):  # noqa: N802 — silence stderr par défaut
        if os.environ.get("RATISNET_ACCESS_LOG"):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("", "/index.html"):
            page = Path(__file__).resolve().parent / "static" / "index.html"
            if page.exists():
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "ui not found"}, 404)
        elif self.path.rstrip("/") == "/health":
            net = _model()
            self._json({"status": "ok", **net.stats()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        net = _model()
        body = self._body()
        route = self.path.rstrip("/")
        try:
            if route == "/respond":
                q = body.get("q", "")
                lang = body.get("language")
                kw = {} if lang is None else {"language": lang}
                self._json(net.speaker.generate_response(q, **kw))
            elif route == "/science":
                q = body.get("q", "")
                lang = body.get("language")
                kw = {} if lang is None else {"language": lang}
                self._json(net.respond_with_science(q, **kw))
            elif route == "/concepts":
                self._json({"concepts": net.concepts(
                    body.get("word", ""), n=int(body.get("n", 10)))})
            elif route == "/chain":
                self._json({"chains": net.chain(
                    body.get("from", ""), body.get("to", ""),
                    max_hops=int(body.get("hops", 3)))})
            elif route == "/prove":
                self._json(net.prove(list(body.get("concepts", []))))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # jamais de traceback brut au client
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RATIS-Net HTTP server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    _model()  # charge avant d'écouter
    print(f"RATIS-Net listening on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
