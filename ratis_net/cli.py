"""ratis_net.cli — Ligne de commande RATIS-Net.

    ratisnet ask "what is quantum mechanics"
    ratisnet converse "hello, how are you?"
    ratisnet concepts --word quantum --n 10
    ratisnet chain --from quantum --to gravity
    ratisnet prove --concepts quantum,mechanics
    ratisnet stats

Le checkpoint Scalpel et les grammaires sont localisés automatiquement :
variables d'environnement RATISNET_SCALPEL / RATISNET_DATA, sinon chemins
relatifs au dépôt (artifacts/, data/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _default_data_dir() -> Path:
    return Path(os.environ.get("RATISNET_DATA",
                               Path(__file__).resolve().parents[1] / "data"))


def _default_scalpel() -> str:
    return os.environ.get(
        "RATISNET_SCALPEL",
        str(Path(__file__).resolve().parents[1] / "artifacts" / "scalpel_wikipedia.pkl"))


def build_model(args: argparse.Namespace):
    from ratis_net import RatisNet
    net = RatisNet()
    net.load_scalpel(args.scalpel, verbose=False)
    data = _default_data_dir()
    net.load_grammar(data / "grammar_domains" / "dense_syntax_skeletons.json",
                     data / "grammar_domains" / "conversation_matrix.json",
                     verbose=False)
    net.load_knowledge_packs(data / "knowledge_packs", verbose=False)
    net.build_index(verbose=False)
    return net


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ratisnet",
                                 description="RATIS-Net — LCT-trained network CLI")
    ap.add_argument("--scalpel", default=_default_scalpel(),
                    help="Chemin du checkpoint Scalpel")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="Question factuelle (réponse enrichie)")
    p_ask.add_argument("query")
    p_ask.add_argument("--language", default=None, choices=["en", "fr"])
    p_ask.add_argument("--json", action="store_true")

    p_conv = sub.add_parser("converse", help="Conversation simple")
    p_conv.add_argument("query")
    p_conv.add_argument("--language", default=None, choices=["en", "fr"])

    p_c = sub.add_parser("concepts", help="Concepts associés à un mot")
    p_c.add_argument("--word", required=True)
    p_c.add_argument("--n", type=int, default=10)

    p_ch = sub.add_parser("chain", help="Chaînes d'association entre deux concepts")
    p_ch.add_argument("--from", dest="source", required=True)
    p_ch.add_argument("--to", dest="target", required=True)
    p_ch.add_argument("--hops", type=int, default=3)

    p_p = sub.add_parser("prove", help="Empreinte d'intégrité d'un sous-graphe")
    p_p.add_argument("--concepts", required=True,
                     help="Concepts séparés par des virgules")

    p_par = sub.add_parser("paragraph", help="Paragraphe sur un thème")
    p_par.add_argument("theme")
    p_par.add_argument("--sentences", type=int, default=4)
    p_par.add_argument("--language", default="en", choices=["en", "fr"])

    sub.add_parser("stats", help="Statistiques du réseau")

    args = ap.parse_args(argv)
    lang = getattr(args, "language", None)

    if args.cmd == "stats" and args.cmd != "stats":
        pass

    net = build_model(args)

    if args.cmd == "ask":
        kw = {} if lang is None else {"language": lang}
        out = net.respond_with_science(args.query, **kw)
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=1))
        else:
            print(out["sentence"])
            if out["knowledge_facts"]:
                print("\nFaits vérifiés :")
                for f in out["knowledge_facts"][:3]:
                    print(f"  - {f['text']}")
            print(f"\nConcepts: {', '.join(out['concepts'][:8])}")
    elif args.cmd == "converse":
        kw = {} if lang is None else {"language": lang}
        print(net.respond(args.query, **kw))
    elif args.cmd == "concepts":
        print(", ".join(net.concepts(args.word, n=args.n)))
    elif args.cmd == "chain":
        for c in net.chain(args.source, args.target, max_hops=args.hops):
            print(" <-> ".join(c["path"]),
                  f"(min {c['min_weight']:.4f}, {c['hops']} hops)")
    elif args.cmd == "prove":
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
        print(json.dumps(net.prove(concepts), ensure_ascii=False, indent=1))
    elif args.cmd == "paragraph":
        print(net.paragraph(args.theme, n_sentences=args.sentences,
                            language=args.language or "en"))
    elif args.cmd == "stats":
        print(json.dumps(net.stats(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
