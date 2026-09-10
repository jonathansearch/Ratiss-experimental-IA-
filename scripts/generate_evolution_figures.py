"""Génère les figures de l'évolution RATISS-Net (session cache-décodeur)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "evolution"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1216", "axes.facecolor": "#0f1216",
    "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white",
    "ytick.color": "white", "axes.edgecolor": "#3a4150",
    "font.size": 11,
})


def fig_cache_speedup():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    methods = ["Snapshot\nTopologique\n(mémo)", "Cache disque\n(commité)", "Cache mémoire\n(lookup dict)"]
    times = [0.04, 0.03, 0.000002]
    colors = ["#f2a93b", "#5cb85c", "#4d9de0"]
    bars = ax.barh(methods, times, color=colors, edgecolor="#2a3240")
    ax.set_xscale("log")
    ax.set_xlabel("Temps par signature (s, log)")
    ax.set_title("P_sig : le cache élimine le calcul persistant (~36× → ~10⁴×)")
    for b, t in zip(bars, times):
        ax.text(t * 1.6, b.get_y() + b.get_height() / 2, f"{t:.1e}s", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cache_speedup.png", dpi=160)
    plt.close(fig)


def fig_generation_table():
    pairs = [
        ("happy (greedy)", "haha you are so funny too"),
        ("happy (beam)", "you are so funny too angel"),
        ("angry (greedy)", "you are stupid ai ever annoy"),
        ("angry (beam)", "fuck you are not talk to"),
        ("sad (greedy)", "my girlfriend left me alone please"),
        ("sad (beam)", "my girlfriend left me so sad"),
        ("others (greedy)", "what is your name of you"),
        ("others (beam)", "what are you know what is"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    celltext = [[e, t] for e, t in pairs]
    table = ax.table(cellText=celltext, colLabels=["Condition", "Séquence générée"],
                     loc="center", cellLoc="left", colWidths=[0.22, 0.75])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#3a4150")
        if r == 0:
            cell.set_facecolor("#1f2634")
        elif "happy" in celltext[r - 1][0]:
            cell.set_facecolor("#1a2b1a")
        elif "angry" in celltext[r - 1][0]:
            cell.set_facecolor("#2b1a1a")
        elif "sad" in celltext[r - 1][0]:
            cell.set_facecolor("#1a1a2b")
        else:
            cell.set_facecolor("#141b27")
    ax.set_title("RATISS-Net parle — les 4 émotions (greedy + beam)", pad=20)
    fig.tight_layout()
    fig.savefig(OUT / "fig_generation_table.png", dpi=160)
    plt.close(fig)


def fig_confusion():
    counts = np.array([[172, 0, 106], [4, 0, 94], [0, 0, 566]])
    labels = ["angry", "happy", "others"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(counts, cmap="viridis")
    fig.colorbar(im, label="Nb dialogues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prédicted"); ax.set_ylabel("Vrai")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                    color="white", fontsize=12)
    ax.set_title("Matrice de confusion (eval env neutre, acc=0.501)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_confusion.png", dpi=160)
    plt.close(fig)


def fig_accuracy_history():
    kurves = [
        ("Baseline (fuite label)\npipeline historique", 1.000),
        ("Eval neutre (honest)\ncentroïdes+v4", 0.501),
        ("Hasard (3 classes)", 0.333),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#c94c4c", "#5cb85c", "#f2a93b"]
    bars = ax.barh([k[0] for k in kurves], [k[1] for k in kurves], color=colors)
    ax.set_xlim(0, 1.1); ax.set_xlabel("Accuracy test")
    ax.set_title("Le 1.000 du pipeline était une fuite ; on mesure honnêtement")
    for b, (n, a) in zip(bars, kurves):
        ax.text(a + 0.02, b.get_y() + b.get_height() / 2, f"a={a:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_accuracy_history.png", dpi=160)
    plt.close(fig)


def fig_flow():
    import matplotlib.patches as mp
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    steps = [
        ("data/cache/.npz\n15 122 sigs", "#1a2b1a"),
        ("TopoCache.load()\n0.03s lookup", "#253a2c"),
        ("embedding_fn(w)\nper-token O(1)", "#2d4a33"),
        ("CentroidLearner\nproto-classes", "#3a4150"),
        ("BigramModel\nEmoContext", "#4a3a50"),
        ("LCTDecoder\ngreedy + beam", "#5c4a50"),
        ("Langage\ngénéré", "#3a4a5c"),
    ]
    for i, (txt, color) in enumerate(steps):
        x = 0.01 + i * 0.145
        ax.add_patch(mp.FancyBboxPatch((x, 0.32), 0.13, 0.4,
                                       boxstyle="round,pad=0.008",
                                       facecolor=color, edgecolor="#3a4150"))
        ax.text(x + 0.065, 0.52, txt, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.145, 0.52), xytext=(x + 0.135, 0.52),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    ax.set_title("Pipeline décodeur — cache → centroïdes → bigram → LCTDecoder",
                 pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_flow.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_cache_speedup()
    fig_generation_table()
    fig_confusion()
    fig_accuracy_history()
    fig_flow()
    print(f"figures générées dans {OUT}")