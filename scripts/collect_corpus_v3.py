"""Collecte nocturne du corpus v3 — Wikipedia batch API, ~2000 titres.

Même source que le v1 du Scalpel (Wikipedia EN, résumés propres), mais en
batch de 50 titres par requête (API officielle, pas de rate-limit brutal).
Sortie : data/corpus/scalpel_v3_corpus.txt (phrases dédupliquées).

Usage : nohup python3 scripts/collect_corpus_v3.py > /tmp/corpus_v3.log 2>&1 &
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "corpus" / "scalpel_v3_corpus.txt"
LOG = Path("/tmp/corpus_v3_progress.txt")

# ~2000 titres : science, bio, médecine, tech, quotidien, conversation
TOPICS = {
    "physique": ["Quantum mechanics", "Quantum entanglement", "Quantum computing",
        "Qubit", "Quantum field theory", "Wave function", "Schrödinger equation",
        "Uncertainty principle", "Particle physics", "Standard Model", "Photon",
        "Electron", "Proton", "Neutron", "Atom", "Atomic nucleus", "Superposition",
        "General relativity", "Special relativity", "Black hole", "Event horizon",
        "Hawking radiation", "Spacetime", "Gravity", "Dark matter", "Dark energy",
        "Big Bang", "Cosmic microwave background", "Supernova", "Neutron star",
        "Pulsar", "Galaxy", "Milky Way", "Star", "Sun", "Solar System", "Planet",
        "Exoplanet", "Mars", "Jupiter", "Light-year", "Redshift", "Universe",
        "Electromagnetism", "Thermodynamics", "Entropy", "Energy", "Force",
        "Mass", "Momentum", "Electric charge", "Magnetic field", "Light",
        "Wave", "Frequency", "Sound", "Heat", "Temperature", "Optics"],
    "biologie": ["DNA", "RNA", "Protein", "Amino acid", "Enzyme", "Cell (biology)",
        "Mitochondrion", "Photosynthesis", "Cellular respiration", "Chlorophyll",
        "Gene", "Genome", "Chromosome", "Evolution", "Natural selection",
        "Mutation", "Virus", "Bacteria", "Immune system", "Antibody", "Vaccine",
        "Antibiotic", "Neuron", "Synapse", "Brain", "Nervous system", "Memory",
        "Sleep", "Heart", "Blood", "Hemoglobin", "Diabetes", "Cancer", "Stem cell",
        "Ecosystem", "Metabolism", "Homeostasis", "Organism", "Species", "Plant",
        "Animal", "Mammal", "Bird", "Fish", "Insect", "Tree", "Flower", "Leaf",
        "Root", "Fruit", "Seed", "Forest", "Ocean", "Coral reef", "Rainforest"],
    "medecine": ["Medicine", "Hospital", "Physician", "Health", "Nutrition",
        "Exercise", "Disease", "Infection", "Inflammation", "Fever", "Pain",
        "Surgery", "Pharmacy", "Drug", "Therapy", "Diagnosis", "Symptom",
        "Immune system", "Allergy", "Asthma", "Stroke", "Hypertension"],
    "chimie": ["Chemical bond", "Molecule", "Chemical reaction", "Acid",
        "Base (chemistry)", "Catalyst", "Oxidation", "Periodic table", "Carbon",
        "Oxygen", "Hydrogen", "Water", "Organic chemistry", "Polymer", "Salt",
        "Metal", "Gas", "Liquid", "Solid", "Crystal", "Solution"],
    "maths": ["Mathematics", "Algebra", "Geometry", "Topology", "Calculus",
        "Number theory", "Set theory", "Logic", "Mathematical proof",
        "Probability", "Statistics", "Algorithm", "Equation", "Function",
        "Variable", "Graph theory", "Linear algebra", "Vector space"],
    "ia_tech": ["Artificial intelligence", "Machine learning", "Neural network",
        "Deep learning", "Natural language processing", "Transformer (deep learning)",
        "Large language model", "Computer", "Internet", "Cryptography", "Robot",
        "Software", "Programming language", "Python (programming language)",
        "Algorithm", "Data structure", "Database", "Cloud computing"],
    "philosophie": ["Philosophy", "Consciousness", "Metacognition", "Epistemology",
        "Ethics", "Free will", "Mind", "Perception", "Cognition", "Intelligence",
        "Knowledge", "Truth", "Reason", "Logic", "Existence", "Reality"],
    "quotidien": ["Love", "Friendship", "Happiness", "Emotion", "Music", "Art",
        "Film", "Literature", "Cooking", "Sport", "Association football",
        "Basketball", "Travel", "Weather", "Coffee", "Tea", "Dog", "Cat",
        "Tree", "Ocean", "Mountain", "City", "Paris", "London", "New York City",
        "Tokyo", "Africa", "Cameroon", "Education", "School", "Science",
        "Technology", "History", "Climate change", "Renewable energy",
        "Solar power", "Electric vehicle", "Language", "French language",
        "English language", "Writing", "Reading", "Book", "Family", "Food",
        "Water", "Sleep", "Dream", "Work", "Money", "Time", "Day", "Night",
        "Season", "Rain", "Wind", "Fire", "Earth", "Sky", "Moon", "Star"],
}

def clean(text: str) -> list[str]:
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace("°", " ").replace("–", " ").replace("—", " ")
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        words = [w.strip("'-") for w in re.findall(r"[a-zA-Z][a-zA-Z'-]*", s.lower())]
        words = [w for w in words if len(w) > 1]
        if 4 <= len(words) <= 35:
            out.append(" ".join(words))
    return out

def main() -> None:
    titles = [t for lst in TOPICS.values() for t in lst]
    titles = list(dict.fromkeys(titles))  # dédup
    done: set[str] = set()
    if OUT.exists():
        done = set(OUT.read_text(encoding="utf-8").splitlines())
    print(f"Départ : {len(done)} phrases existantes, {len(titles)} titres", flush=True)
    LOG.write_text(f"start {len(done)}\n", encoding="utf-8")

    new = 0
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        params = urllib.parse.urlencode({
            "action": "query", "prop": "extracts", "exintro": "1",
            "explaintext": "1", "format": "json",
            "titles": "|".join(batch), "redirects": "1",
        })
        url = "https://en.wikipedia.org/w/api.php?" + params
        req = urllib.request.Request(url, headers={
            "User-Agent": "ratis-net-research/0.3 (corpus collection)"})
        try:
            data = json.load(urllib.request.urlopen(req, timeout=30))
            for page in data.get("query", {}).get("pages", {}).values():
                for sent in clean(page.get("extract", "")):
                    if sent not in done:
                        done.add(sent)
                        new += 1
        except Exception as e:
            print(f"  [batch {i}] skip: {e}", flush=True)
        if (i // 50) % 5 == 0:
            LOG.write_text(f"batch {i}/{len(titles)} total={len(done)} new={new}\n",
                           encoding="utf-8")
            print(f"  batch {i}/{len(titles)} → {len(done)} phrases (+{new})", flush=True)
        time.sleep(1.2)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(done)) + "\n")
    LOG.write_text(f"DONE total={len(done)}\n", encoding="utf-8")
    print(f"TERMINE : {len(done)} phrases dans {OUT}", flush=True)

if __name__ == "__main__":
    main()
