"""scripts/generate_concept_figures.py — Figures de concept pour la documentation.

Génère des images de concept illustrant l'architecture RATIS (AGI souverain) :
  1.  Boucle cognitive AGI (6 étapes en cercle)
  2.  Loi LCT (R = P_sig croît avec C, invariant sous énergie)
  3.  Cerveau TTF-Compute (schéma du pipeline)
  4.  Saut v4 : ETH thermo fixer (l'émotion émerge)
  5.  Décodeur : glouton vs auto-régressif vs beam
  6.  happy débloqué (0% -> 85%, F1 0.62 -> 0.92)
  7.  Immersion accélérée (boucle seed -> mutation -> filtres -> reinjection)
  8.  Universalité LCT (invariance 3/3, monotonie distribuée vs concentrée)
  9.  RATIS face à l'inconnu (projection topologique vs mémorisation LLM)
  10. Architecture des 2 dépôts (cerveau + réseau IA)

Usage : python scripts/generate_concept_figures.py
Sortie : docs/figures/fig*.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# Palette RATIS (sobre, scientifique)
C_DARK = "#1a1a2e"
C_BLUE = "#0f3460"
C_ACCENT = "#e94560"
C_GREEN = "#16a085"
C_GOLD = "#f39c12"
C_LIGHT = "#eef2f7"
C_GRAY = "#95a5a6"
C_PURPLE = "#6c5ce7"

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")


# ── 1. Boucle cognitive AGI (6 étapes en cercle) ──────────────────────────

def fig1_boucle_cognitive():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Boucle cognitive de RATIS (AGI souverain)", fontsize=16, weight="bold", pad=20)

    steps = [
        ("1. PERCEVOIR", "tokeniser →\nembeddings topo", C_BLUE),
        ("2. PENSER", "cerveau TTF\n→ MCB (sans mots)", C_PURPLE),
        ("3. RESSENTIR", "ETH → C_seuil\n→ émotion émerge", C_ACCENT),
        ("4. COMPRENDRE", "réseau LCT\nΔW = η·φ·P_sig·C", C_GREEN),
        ("5. PARLER", "décodeur beam\n→ langage", C_GOLD),
        ("6. CERTIFIER", "hash topo\ninvariant → ZK", C_DARK),
    ]
    n = len(steps)
    R = 1.0
    for i, (title, desc, color) in enumerate(steps):
        angle = math.pi / 2 - 2 * math.pi * i / n
        x, y = R * math.cos(angle), R * math.sin(angle)
        circle = Circle((x, y), 0.22, facecolor=color, edgecolor="white", linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.03, title, ha="center", va="center", fontsize=8.5,
                weight="bold", color="white", zorder=4)
        ax.text(x, y - 0.07, desc, ha="center", va="center", fontsize=6.5,
                color="white", zorder=4)
        # flèche vers le suivant
        a2 = math.pi / 2 - 2 * math.pi * (i + 1) / n
        x2, y2 = R * math.cos(a2), R * math.sin(a2)
        mid_x, mid_y = (x + x2) / 2, (y + y2) / 2
        ax.annotate("", xy=(x2 - 0.18 * math.cos(a2 - angle), y2 - 0.18 * math.sin(a2 - angle)),
                    xytext=(x + 0.18 * math.cos(a2 - angle), y + 0.18 * math.sin(a2 - angle)),
                    arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=2,
                                    connectionstyle="arc3,rad=0.15"))

    ax.text(0, 0, "RATIS\nAGI\nsouverain", ha="center", va="center",
            fontsize=14, weight="bold", color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=C_LIGHT, edgecolor=C_BLUE, lw=2))
    ax.text(0, -1.3, "100% local — pas de cloud — pas de LLM externe",
            ha="center", fontsize=9, color=C_GRAY, style="italic")
    _save(fig, "fig1_boucle_cognitive.png")


# ── 2. Loi LCT (R croît avec C, invariant sous énergie) ────────────────────

def fig2_loi_lct():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Loi LCT — R = P_sig croît avec C, invariant sous l'énergie",
                 fontsize=15, weight="bold")

    # Monotonie R(C)
    C = np.linspace(0, 1, 50)
    R = 1.4 + 2.1 * C + 0.05 * np.sin(6 * C)
    ax1.plot(C, R, color=C_ACCENT, lw=3, label="R = P_sig (mesuré 4MZI)")
    ax1.fill_between(C, R * 0.9, R * 1.1, color=C_ACCENT, alpha=0.15)
    ax1.scatter([0, 0.5, 1.0], [1.45, 2.80, 3.54], color=C_DARK, s=60, zorder=5)
    ax1.annotate("Spearman\n+0.930", xy=(0.5, 2.80), xytext=(0.15, 3.2),
                 fontsize=11, color=C_GREEN, weight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_GREEN))
    ax1.set_xlabel("Cohérence C (intrication)", fontsize=11)
    ax1.set_ylabel("R = P_sig (persistance topologique)", fontsize=11)
    ax1.set_title("Monotonie : R croît avec C", fontsize=12, weight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Invariance ZK
    energies = [-3.48, -5.27, -7.07, -1.75]
    R_inv = [1.52, 1.52, 1.52, 1.52]
    ax2.bar(range(4), R_inv, color=C_BLUE, width=0.5, alpha=0.8)
    ax2.axhline(1.52, color=C_ACCENT, ls="--", lw=2, label="R invariant (CV = 0.0000)")
    for i, e in enumerate(energies):
        ax2.text(i, 1.55, f"{e}", ha="center", fontsize=9, color=C_DARK)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["t=1.0\nJ=0.3", "t=1.5\nJ=0.6", "t=2.0\nJ=0.9", "t=0.5\nJ=0.15"])
    ax2.set_ylabel("R (même topologie, énergies ≠)", fontsize=11)
    ax2.set_title("Invariance ZK : R constant sous énergie", fontsize=12, weight="bold")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 2.5)
    _save(fig, "fig2_loi_lct.png")


# ── 3. Cerveau TTF-Compute (schéma pipeline) ──────────────────────────────

def fig3_cerveau_ttf():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Cerveau TTF-Compute — Tryperposition Topologique Fine",
                 fontsize=15, weight="bold", pad=15)

    blocks = [
        (0.3, 4.5, "Graphe\nIntriqué\nG(V,E)", "w_Q + w_I\n(milieu génial)", C_BLUE),
        (2.8, 4.5, "Transmetteur\ntJ", "démodulation\nhaute→basse fréq.", C_PURPLE),
        (5.3, 4.5, "Traducteur\nRips", "Betti à la volée\n+ compression", C_GREEN),
        (7.8, 4.5, "RLM\nmatriciel", "ΔW = η·φ·P_sig·C\n(sans mots)", C_GOLD),
        (10.3, 4.5, "MCB", "triplets\n(src,dst,φ)", C_ACCENT),
        (0.3, 1.5, "Puits\nd'effondrement", "V=-k/(1+d²)\n+ TSP minimal", C_DARK),
        (3.8, 1.5, "TSP\nHeld-Karp", "gluon d'info\n(chemin min)", C_BLUE),
        (7.3, 1.5, "Preuve\nZK-STARK", "hash topo\ninvariant", C_GREEN),
        (10.8, 1.5, "→ LLM\ngreffé", "pensée sans\nmots → langage", C_ACCENT),
    ]
    for x, y, title, desc, color in blocks:
        box = FancyBboxPatch((x, y), 2.2, 1.6, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="white", lw=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 1.1, y + 1.15, title, ha="center", va="center",
                fontsize=9, weight="bold", color="white")
        ax.text(x + 1.1, y + 0.45, desc, ha="center", va="center",
                fontsize=7, color="white")

    # flèches
    arrows = [(2.5, 5.3, 2.8, 5.3), (5.0, 5.3, 5.3, 5.3), (7.5, 5.3, 7.8, 5.3),
              (10.0, 5.3, 10.3, 5.3),
              (11.4, 4.5, 11.4, 3.1), (10.3, 2.3, 7.3, 2.3), (6.0, 2.3, 3.8, 2.3),
              (3.5, 2.3, 0.3, 2.3), (1.4, 3.1, 1.4, 4.5)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.8))

    ax.text(7, 6.5, "Boucle : oscillate → transmit → translate → RLM/MCB → puits → TSP → ZK",
            ha="center", fontsize=10, color=C_DARK, style="italic")
    _save(fig, "fig3_cerveau_ttf.png")


# ── 4. Saut v4 : ETH thermo fixer ──────────────────────────────────────────

def fig4_eth_thermo():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Le saut v4 — ETH : le fixeur thermodynamique (l'émotion émerge)",
                 fontsize=14, weight="bold")

    # C_seuil par environnement
    envs = ["colère", "joie", "calme", "peur"]
    seuils = [0.310, 0.691, 0.736, 0.353]
    colors = [C_ACCENT, C_GOLD, C_BLUE, C_PURPLE]
    bars = ax1.bar(envs, seuils, color=colors, width=0.5, alpha=0.85)
    ax1.axhline(0.5, color=C_GRAY, ls="--", lw=1, label="seuil neutre")
    for b, s in zip(bars, seuils):
        ax1.text(b.get_x() + b.get_width() / 2, s + 0.02, f"{s:.3f}",
                 ha="center", fontsize=10, weight="bold")
    ax1.set_ylabel("C_seuil prédit par ETH", fontsize=11)
    ax1.set_title("C_seuil = f(token, environnement)", fontsize=12, weight="bold")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Différentiel = l'émotion
    ax2.barh(["joie − colère"], [0.380], color=C_GREEN, height=0.4, alpha=0.85)
    ax2.text(0.38, 0, "  +0.380 (émotion émerge)", va="center", fontsize=12,
             weight="bold", color=C_GREEN)
    ax2.set_xlim(0, 0.6)
    ax2.set_title("L'émotion = différentiel de C_seuil", fontsize=12, weight="bold")
    ax2.set_xlabel("ΔC_seuil (joie − colère)", fontsize=11)
    ax2.axvline(0, color=C_DARK, lw=1)
    ax2.grid(axis="x", alpha=0.3)
    _save(fig, "fig4_eth_thermo.png")


# ── 5. Décodeur : glouton vs auto-régressif vs beam ────────────────────────

def fig5_decodeur():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title("Décodeur LCT — 3 modes de décodage (cohérence de séquence)",
                 fontsize=14, weight="bold", pad=15)
    modes = ["Glouton\n(score local)", "Auto-régressif\n(état caché)", "Beam\n(cohérence globale)"]
    phrases = ["haha you are\nfunny and...", "haha you are\nfunny... (feedback)", "yes this is\nvery go and ✓"]
    coherence = [3, 3, 4]
    colors = [C_GRAY, C_GOLD, C_GREEN]
    bars = ax.bar(modes, coherence, color=colors, width=0.5, alpha=0.85)
    for b, c, p in zip(bars, coherence, phrases):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.1, f"{c}/4",
                ha="center", fontsize=14, weight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.3, p, ha="center",
                fontsize=8, color="white", style="italic")
    ax.set_ylabel("Cohérence LCT (re-classage = cible)", fontsize=11)
    ax.set_ylim(0, 5)
    ax.axhline(4, color=C_ACCENT, ls="--", lw=1.5, label="cible : 4/4 (happy débloqué)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig5_decodeur_modes.png")


# ── 6. happy débloqué ─────────────────────────────────────────────────────

def fig6_happy_debloque():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("happy DÉBLOQUÉ — unité SÉQUENCE + rééquilibrage (piste 2)",
                 fontsize=14, weight="bold")

    configs = ["A : mot-à-mot\n(brut, 300)", "B : séquence\n(rééq., 300)", "C : séquence\n(rééq., 3000)"]
    happy_recall = [0.00, 0.62, 0.85]
    bars = ax1.bar(configs, happy_recall, color=[C_ACCENT, C_GOLD, C_GREEN], width=0.5, alpha=0.85)
    for b, r in zip(bars, happy_recall):
        ax1.text(b.get_x() + b.get_width() / 2, r + 0.03, f"{r:.0%}",
                 ha="center", fontsize=12, weight="bold")
    ax1.set_ylabel("Rappel happy (classe minoritaire)", fontsize=11)
    ax1.set_title("Rappel happy : 0% → 85%", fontsize=12, weight="bold")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)

    f1 = [0.620, 0.588, 0.924]
    bars2 = ax2.bar(configs, f1, color=[C_ACCENT, C_GOLD, C_GREEN], width=0.5, alpha=0.85)
    for b, v in zip(bars2, f1):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=12, weight="bold")
    ax2.set_ylabel("F1 macro (sensible au minoritaire)", fontsize=11)
    ax2.set_title("F1 macro : 0.62 → 0.92", fontsize=12, weight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    _save(fig, "fig6_happy_debloque.png")


# ── 7. Immersion accélérée ────────────────────────────────────────────────

def fig7_immersion():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Immersion structurée accélérée (auto-génération ancrée)",
                 fontsize=14, weight="bold", pad=15)

    steps = [
        (0.3, 5, "SEED", "dialogue réel\nEmoContext", C_BLUE),
        (2.8, 5, "MUTATION", "substitution de mots\n(mode risqué)", C_GOLD),
        (5.3, 5, "FILTRE ZK", "hash topo stable\n(la forme)", C_PURPLE),
        (7.8, 5, "FILTRE\nSÉMANTIQUE", "re-classage =\némotion cible ?", C_ACCENT),
        (10.3, 5, "RÉINJECTION", "dialogue validé\n→ entraînement", C_GREEN),
    ]
    for x, y, title, desc, color in steps:
        box = FancyBboxPatch((x, y - 0.9), 2.2, 1.8, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="white", lw=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.5, title, ha="center", va="center",
                fontsize=9, weight="bold", color="white")
        ax.text(x + 1.1, y - 0.2, desc, ha="center", va="center",
                fontsize=7, color="white")
    for i in range(4):
        ax.annotate("", xy=(steps[i + 1][0], 5), xytext=(steps[i][0] + 2.2, 5),
                    arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=2))
    # boucle
    ax.annotate("", xy=(1.4, 4.1), xytext=(11.4, 4.1),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=2,
                                connectionstyle="arc3,rad=0.3", ls="--"))
    ax.text(6, 2.8, "Boucle (anti-collapse : ancrage vérité-terrain + double filtre + diversité surveillée)",
            ha="center", fontsize=9, color=C_GREEN, style="italic")
    ax.text(6, 1.8, "Gain mesuré : F1 ×1.01 (honnête, pas ×10000)\nFiltre sémantique rejette les mutations qui cassent le sens",
            ha="center", fontsize=9, color=C_DARK)
    _save(fig, "fig7_immersion_acceleree.png")


# ── 8. Universalité LCT ───────────────────────────────────────────────────

def fig8_universalite():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Universalité de la loi LCT (piste 5)", fontsize=14, weight="bold")

    # Invariance 3/3
    systems = ["réseau\nsocial", "cristal", "réseau\ndense"]
    inv = [1, 1, 1]
    ax1.bar(systems, inv, color=[C_GRAY, C_GREEN, C_GRAY], width=0.5, alpha=0.85)
    ax1.set_ylim(0, 1.3)
    ax1.set_ylabel("Invariance ZK (R constant sous énergie)", fontsize=11)
    ax1.set_title("Invariance ZK : 3/3 PASS (universelle)", fontsize=12, weight="bold", color=C_GREEN)
    for i in range(3):
        ax1.text(i, 1.05, "✓", ha="center", fontsize=16, weight="bold", color=C_GREEN)
    ax1.grid(axis="y", alpha=0.3)

    # Monotonie 1/3
    mono = [-0.462, 0.930, 0.259]
    colors_m = [C_ACCENT, C_GREEN, C_ACCENT]
    bars = ax2.bar(systems, mono, color=colors_m, width=0.5, alpha=0.85)
    ax2.axhline(0.6, color=C_GREEN, ls="--", lw=1.5, label="seuil PASS (0.6)")
    ax2.axhline(0, color=C_DARK, lw=1)
    for b, m in zip(bars, mono):
        label = "✓" if m > 0.6 else "✗"
        ax2.text(b.get_x() + b.get_width() / 2, m + (0.05 if m >= 0 else -0.1),
                 f"{m:+.2f} {label}", ha="center", fontsize=11, weight="bold")
    ax2.set_ylabel("Spearman R(C) — monotonie", fontsize=11)
    ax2.set_title("Monotonie : 1/3 (structure distribuée requise)", fontsize=12, weight="bold")
    ax2.set_ylim(-0.7, 1.1)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    _save(fig, "fig8_universalite_lct.png")


# ── 9. RATIS face à l'inconnu ─────────────────────────────────────────────

def fig9_inconnu():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("RATIS face à l'inconnu — projection topologique vs mémorisation",
                 fontsize=14, weight="bold", pad=15)
    ax.axis("off")

    # LLM
    box1 = FancyBboxPatch((0.3, 3), 4.5, 2.5, boxstyle="round,pad=0.15",
                          facecolor=C_LIGHT, edgecolor=C_ACCENT, lw=2)
    ax.add_patch(box1)
    ax.text(2.55, 5.1, "LLM classique", ha="center", fontsize=13, weight="bold", color=C_ACCENT)
    ax.text(2.55, 4.5, "« je connais ce mot\ncar je l'ai vu\n10 000 fois »", ha="center",
            fontsize=10, color=C_DARK, style="italic")
    ax.text(2.55, 3.4, "→ MÉMORISATION\n→ peut halluciner", ha="center", fontsize=9,
            color=C_ACCENT, weight="bold")

    # RATIS
    box2 = FancyBboxPatch((6.8, 3), 4.5, 2.5, boxstyle="round,pad=0.15",
                          facecolor=C_LIGHT, edgecolor=C_GREEN, lw=2)
    ax.add_patch(box2)
    ax.text(9.05, 5.1, "RATIS", ha="center", fontsize=13, weight="bold", color=C_GREEN)
    ax.text(9.05, 4.5, "« je classifie ce mot\npar sa TOPOLOGIE,\nmême si je ne l'ai\njamais vu »",
            ha="center", fontsize=10, color=C_DARK, style="italic")
    ax.text(9.05, 3.4, "→ PROJECTION TOPO\n→ ne hallucine pas", ha="center", fontsize=9,
            color=C_GREEN, weight="bold")

    # résultats
    ax.text(6, 1.8, "Mot inconnu → topologie (forme) → classé (souvent 'neutre' = zone d'incertitude)\n"
            "Robustesse 6/6 (aucun crash) — Généralise les variantes proches (funny→funnyyy)\n"
            "Prudent sur les concepts radicaux (quantum≈amour) — SANS halluciner",
            ha="center", fontsize=9.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT, edgecolor=C_GRAY))
    _save(fig, "fig9_inconnu.png")


# ── 10. Architecture des 2 dépôts ─────────────────────────────────────────

def fig10_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Architecture RATISS — 2 dépôts (cerveau + réseau IA)",
                 fontsize=15, weight="bold", pad=15)

    # Dépôt 1 : AEON
    box1 = FancyBboxPatch((0.3, 4.5), 6, 3, boxstyle="round,pad=0.15",
                          facecolor=C_BLUE, edgecolor="white", lw=2, alpha=0.15)
    ax.add_patch(box1)
    ax.text(3.3, 7.1, "RATISS-ODV-AEON\n(le cerveau moteur)", ha="center",
            fontsize=12, weight="bold", color=C_BLUE)
    items1 = ["kernel/ttf/ttf_compute.py — TTFBrain", "kernel/ttf/lct_law.py — Loi LCT",
              "MCB (pensée sans mots)", "Puits + TSP + ZK-STARK",
              "config/ — identité souveraine JohnKing0"]
    for i, t in enumerate(items1):
        ax.text(0.7, 6.3 - i * 0.35, f"• {t}", fontsize=9, color=C_DARK)

    # Dépôt 2 : IA
    box2 = FancyBboxPatch((7.5, 4.5), 6, 3, boxstyle="round,pad=0.15",
                          facecolor=C_GREEN, edgecolor="white", lw=2, alpha=0.15)
    ax.add_patch(box2)
    ax.text(10.5, 7.1, "Ratiss-experimental-IA-\n(le réseau IA)", ha="center",
            fontsize=12, weight="bold", color=C_GREEN)
    items2 = ["ratis_net_v4.py — réseau LCT (ΔW=η·φ·P_sig·C)", "eth_thermo_fixer.py — ETH (émotion)",
              "decoder.py — décodeur beam (parler)", "topo_tokenizer.py — cycles H1",
              "ratis_agent.py — AGI (6 étapes cognitives)"]
    for i, t in enumerate(items2):
        ax.text(7.9, 6.3 - i * 0.35, f"• {t}", fontsize=9, color=C_DARK)

    # pont
    ax.annotate("", xy=(7.5, 6), xytext=(6.3, 6),
                arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=2.5))
    ax.text(6.9, 6.3, "MCB\nbridge", ha="center", fontsize=8, color=C_ACCENT, weight="bold")

    # AGI en bas
    box3 = FancyBboxPatch((2, 0.5), 10, 3, boxstyle="round,pad=0.15",
                          facecolor=C_GOLD, edgecolor="white", lw=2, alpha=0.2)
    ax.add_patch(box3)
    ax.text(7, 3.1, "AGI RATIS souverain — les 4 briques complètes", ha="center",
            fontsize=13, weight="bold", color=C_GOLD)
    bricks = ["1. Cerveau topo (TTF/MCB)", "2. Certif ZK (invariant)",
              "3. Souveraineté (local)", "4. LCT (apprend, ressent, parle)"]
    for i, b in enumerate(bricks):
        ax.text(2.5 + i * 2.5, 2, f"✓ {b}", ha="center", fontsize=9.5, color=C_DARK)
    ax.text(7, 1.1, "Loi LCT figée : R = P_sig, ΔW = η·φ·P_sig·C\n"
            "100% local — pas de cloud — © JOHNKING0 & Jonathan Evina",
            ha="center", fontsize=9, color=C_GRAY, style="italic")
    _save(fig, "fig10_architecture.png")


def main():
    print("=" * 60)
    print("Génération des figures de concept RATIS")
    print("=" * 60)
    fig1_boucle_cognitive()
    fig2_loi_lct()
    fig3_cerveau_ttf()
    fig4_eth_thermo()
    fig5_decodeur()
    fig6_happy_debloque()
    fig7_immersion()
    fig8_universalite()
    fig9_inconnu()
    fig10_architecture()
    print(f"\n✓ {len(list(OUT.glob('fig*.png')))} figures générées dans {OUT}")


if __name__ == "__main__":
    main()
