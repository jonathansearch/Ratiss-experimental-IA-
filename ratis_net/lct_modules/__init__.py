"""ratis_net.lct_modules — Les 3 modules algorithmiques LCT (session transdisciplinaire).

1. grav_measure : mesure topologique gravitationnelle (cycles H1, cohérence/décohérence)
2. topo_qubit : simulation algorithmique d'un qubit topologique
3. lct_transformer : entraînement dédié LCT (ΔW = η·φ·P_sig·C) avec inhibition latérale
"""
from .grav_measure import GravitationalTopoMeasure
from .topo_qubit import TopologicalQubit
from .lct_transformer import LCTTransformer

__all__ = ["GravitationalTopoMeasure", "TopologicalQubit", "LCTTransformer"]
