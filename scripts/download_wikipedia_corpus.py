"""Download Wikipedia article summaries as a real corpus for the Scalpel.

Uses the Wikipedia REST API (free, no auth) to fetch article summaries.
Each summary is a real encyclopedic sentence. We collect ~2000 articles
from a curated list of popular topics across science, history, arts, etc.
"""
import json
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error

OUT = Path(__file__).resolve().parent.parent / "data" / "corpus" / "wikipedia_summaries.txt"

# Curated list of ~500 popular Wikipedia article titles (broad coverage)
TITLES = [
    "Quantum mechanics", "General relativity", "Black hole", "String theory",
    "Particle physics", "Standard Model", "Electromagnetism", "Thermodynamics",
    "Entropy", "Information theory", "Chaos theory", "Fractal",
    "Mathematics", "Calculus", "Algebra", "Geometry", "Topology", "Number theory",
    "Probability", "Statistics", "Linear algebra", "Differential equation",
    "Physics", "Chemistry", "Biology", "Astronomy", "Cosmology", "Astrophysics",
    "Neuroscience", "Cognitive science", "Computer science", "Artificial intelligence",
    "Machine learning", "Neural network", "Deep learning", "Natural language processing",
    "Algorithm", "Data structure", "Cryptography", "Quantum computing",
    "Philosophy", "Logic", "Ethics", "Epistemology", "Metaphysics",
    "Psychology", "Sociology", "Economics", "Anthropology", "Linguistics",
    "History", "Archaeology", "Geography", "Geology", "Climate", "Ecology",
    "Evolution", "Genetics", "DNA", "Protein", "Cell biology",
    "Medicine", "Pharmacology", "Anatomy", "Immunology", "Virology",
    "Technology", "Engineering", "Robotics", "Nanotechnology", "Biotechnology",
    "Internet", "World Wide Web", "Operating system", "Programming language",
    "Software engineering", "Database", "Computer graphics", "Human-computer interaction",
    "Music", "Art", "Literature", "Poetry", "Theatre", "Film", "Photography",
    "Architecture", "Sculpture", "Painting", "Dance", "Opera", "Ballet",
    "Agriculture", "Industry", "Manufacturing", "Transport", "Aviation",
    "Space exploration", "Satellite", "Rocket", "Energy", "Renewable energy",
    "Fossil fuel", "Nuclear power", "Solar power", "Wind power",
    "Philosophy of science", "Scientific method", "Experiment", "Hypothesis",
    "Theory", "Law", "Empiricism", "Rationalism", "Pragmatism",
    "World War I", "World War II", "Cold War", "French Revolution",
    "American Revolution", "Industrial Revolution", "Renaissance",
    "Enlightenment", "Middle Ages", "Ancient Rome", "Ancient Greece",
    "Ancient Egypt", "Byzantine Empire", "Ottoman Empire", "Mongol Empire",
    "Chinese dynasty", "British Empire", "Colonialism", "Imperialism",
    "Democracy", "Republic", "Monarchy", "Socialism", "Capitalism",
    "Communism", "Fascism", "Liberalism", "Conservatism", "Anarchism",
    "United Nations", "European Union", "NATO", "World Bank",
    "Africa", "Asia", "Europe", "North America", "South America",
    "Australia", "Antarctica", "Arctic", "Pacific Ocean", "Atlantic Ocean",
    "Indian Ocean", "Sahara Desert", "Amazon rainforest", "Himalayas",
    "Mount Everest", "Nile", "Amazon River", "Mediterranean Sea",
    "Human", "Brain", "Heart", "Lung", "Liver", "Kidney",
    "Eye", "Ear", "Skeleton", "Muscle", "Nervous system",
    "Immune system", "Digestive system", "Respiratory system",
    "Consciousness", "Memory", "Learning", "Perception", "Emotion",
    "Language", "Speech", "Writing", "Reading", "Communication",
    "Education", "School", "University", "Research", "Academia",
    "Patent", "Copyright", "Trademark", "Intellectual property",
    "Internet protocol", "TCP/IP", "HTTP", "HTML", "JavaScript",
    "Python (programming language)", "Java (programming language)",
    "C (programming language)", "Linux", "Unix", "Windows",
    "Smartphone", "Computer", "Microprocessor", "Transistor",
    "Semiconductor", "Silicon", "Circuit design", "Embedded system",
    "Vacuum tube", "Telecommunication", "Radio", "Television",
    "Telephone", "Telegraphy", "Printing press", "Compass",
    "Wheel", "Steam engine", "Internal combustion engine",
    "Electric motor", "Battery", "Solar cell", "LED",
    "Laser", "Fiber optics", "Microscope", "Telescope",
    "Spectroscopy", "X-ray", "MRI", "Ultrasound", "Tomography",
    "Vaccine", "Antibiotic", "Insulin", "Aspirin", "Anesthesia",
    "Surgery", "Transplant", "Gene therapy", "Stem cell",
    "Cancer", "Diabetes", "Alzheimer's disease", "Heart disease",
    "Malaria", "Tuberculosis", "Influenza", "HIV", "COVID-19",
    "Virus", "Bacteria", "Fungus", "Parasite", "Infection",
    "Antibiotic resistance", "Epidemiology", "Public health",
    "Nutrition", "Vitamin", "Mineral", "Protein", "Carbohydrate",
    "Fat", "Metabolism", "Hormone", "Enzyme", "Metabolite",
    "Photosynthesis", "Respiration", "Mitochondria", "Chloroplast",
    "Ribosome", "Chromosome", "Gene", "Genome", "Mutation",
    "Natural selection", "Speciation", "Extinction", "Biodiversity",
    "Ecosystem", "Food chain", "Symbiosis", "Parasitism",
    "Plant", "Animal", "Fungi", "Protist", "Bacteria",
    "Insect", "Fish", "Bird", "Mammal", "Reptile",
    "Amphibian", "Dinosaur", "Fossil", "Evolutionary biology",
    "Earth", "Moon", "Sun", "Solar System", "Planet",
    "Mercury (planet)", "Venus", "Mars", "Jupiter", "Saturn",
    "Uranus", "Neptune", "Pluto", "Asteroid", "Comet",
    "Meteorite", "Galaxy", "Milky Way", "Andromeda Galaxy",
    "Big Bang", "Cosmic microwave background", "Dark matter",
    "Dark energy", "Gravitational wave", "Neutron star", "Pulsar",
    "Quasar", "Supernova", "White dwarf", "Red giant",
    "Exoplanet", "Habitable zone", "Fermi paradox", "SETI",
    "Space Shuttle", "International Space Station", "Apollo program",
    "Voyager program", "Hubble Space Telescope", "James Webb Space Telescope",
    "Atom", "Electron", "Proton", "Neutron", "Quark",
    "Photon", "Boson", "Fermion", "Higgs boson", "Antimatter",
    "Nuclear fission", "Nuclear fusion", "Radioactivity",
    "Isotope", "Periodic table", "Chemical element", "Chemical bond",
    "Molecule", "Chemical reaction", "Catalyst", "Polymer",
    "Organic chemistry", "Biochemistry", "Physical chemistry",
    "Analytical chemistry", "Acid", "Base", "pH", "Solution",
    "Crystal", "Glass", "Metal", "Alloy", "Plastic",
    "Rubber", "Ceramic", "Composite material", "Nanomaterial",
    "Graphene", "Carbon nanotube", "Buckminsterfullerene",
    "Superconductivity", "Magnetism", "Electricity", "Voltage",
    "Current", "Resistance", "Capacitor", "Inductor", "Transformer",
    "Motor", "Generator", "Turbine", "Pump", "Compressor",
    "Refrigeration", "Air conditioning", "Heating", "Ventilation",
    "Bridge", "Dam", "Tunnel", "Skyscraper", "Road",
    "Rail transport", "Ship", "Boat", "Submarine", "Aircraft",
    "Helicopter", "Rocket", "Spacecraft", "Satellite",
    "Wheel", "Axle", "Gear", "Bearing", "Spring",
    "Clock", "Watch", "Calendar", "Time zone", "Daylight saving",
    "Equinox", "Solstice", "Eclipse", "Tide", "Season",
    "Weather", "Cloud", "Rain", "Snow", "Storm", "Hurricane",
    "Tornado", "Earthquake", "Volcano", "Tsunami", "Flood",
    "Drought", "Wildfire", "Climate change", "Global warming",
    "Ozone depletion", "Air pollution", "Water pollution",
    "Deforestation", "Desertification", "Recycling", "Conservation",
    "Sustainable development", "Renewable resource", "Fossil fuel",
    "Oil", "Coal", "Natural gas", "Uranium",
    "Gold", "Silver", "Copper", "Iron", "Aluminium",
    "Titanium", "Tungsten", "Lead", "Zinc", "Nickel",
    "Diamond", "Graphite", "Salt", "Sand", "Clay",
    "Limestone", "Granite", "Marble", "Slate",
    "River", "Lake", "Ocean", "Glacier", "Groundwater",
    "Water cycle", "Evaporation", "Condensation", "Precipitation",
    "Soil", "Rock", "Mineral", "Fossil fuel",
    "Mountain", "Valley", "Canyon", "Plateau", "Plain",
    "Desert", "Wetland", "Cave", "Island", "Peninsula",
    "Continent", "Tectonic plate", "Fault", "Earthquake",
    "Volcano", "Geothermal", "Geyser", "Hot spring",
    "Cartography", "Map", "Globe", "Compass", "GPS",
    "Surveying", "Navigation", "Astronomy",
    "Light", "Color", "Sound", "Wave", "Frequency",
    "Amplitude", "Wavelength", "Spectrum", "Refraction",
    "Reflection", "Diffraction", "Interference", "Polarization",
    "Optics", "Lens", "Mirror", "Prism", "Holography",
    "Acoustics", "Doppler effect", "Resonance", "Harmonic",
    "Thermodynamics", "Heat", "Temperature", "Entropy",
    "Enthalpy", "Free energy", "Phase transition", "Melting",
    "Boiling", "Condensation", "Sublimation", "Evaporation",
    "Pressure", "Vacuum", "Fluid mechanics", "Viscosity",
    "Surface tension", "Buoyancy", "Bernoulli's principle",
]


def fetch_summary(title: str) -> str | None:
    """Fetch a Wikipedia article summary via the REST API."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    req = urllib.request.Request(url, headers={"User-Agent": "RATISS-Net/1.0 (research)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            extract = data.get("extract", "")
            return extract if extract and len(extract) > 20 else None
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    phrases = []
    failed = 0
    for i, title in enumerate(TITLES):
        summary = fetch_summary(title)
        if summary:
            # split into sentences, keep those with >= 4 words
            for sent in summary.replace(". ", ".\n").split("\n"):
                sent = sent.strip()
                if len(sent.split()) >= 4:
                    phrases.append(sent)
        else:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(TITLES)} articles, {len(phrases)} phrases, {failed} failed")
        time.sleep(0.05)  # be polite to the API

    with open(OUT, "w", encoding="utf-8") as f:
        for p in phrases:
            f.write(p + "\n")
    print(f"\nWrote {len(phrases)} phrases to {OUT}")
    print(f"Failed: {failed}/{len(TITLES)}")


if __name__ == "__main__":
    main()
