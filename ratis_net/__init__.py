"""RATIS-Net — Neural network trained by LCT (Law of Topological Coherence).

Proof-of-concept : un reseau de neurones qui apprend par la loi LCT
(deltaW = eta * phi * P_sig * C) au lieu du gradient descendant.

Au lieu de minimiser une loss par backpropagation, le reseau maximise P_sig
(la persistance topologique de sa matrice de poids) -- il apprend en devenant
topologiquement robuste. C est l apprentissage par coherence topologique.

Framework unifie :
    from ratis_net import RatisNet

    net = RatisNet()
    net.load_scalpel("artifacts/scalpel_wikipedia.pkl")
    net.load_grammar("data/grammar_domains/dense_syntax_skeletons.json")
    net.build_index()

    print(net.respond("what is quantum mechanics"))
"""
from ratis_net.framework import RatisNet

__all__ = ["RatisNet"]
