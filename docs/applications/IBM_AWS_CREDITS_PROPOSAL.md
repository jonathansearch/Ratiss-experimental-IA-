# Compute-Credits Proposals — IBM Quantum & AWS Research (RATISS / LCT)

> FR (usage interne Jonathan) : deux dossiers, une seule page de base. La section
> « Technical summary » (EN) est la version prête à coller dans les deux
> formulaires. Ne rien inventer : tous les chiffres cités existent dans ce
> dépôt ou dans le preprint OSF (DOI 10.17605/OSF.IO/WF7QM).

---

## One-page base (commun aux deux dossiers)

**Project.** RATISS is a sovereign topological AI stack built on the
**Law of Topological Coherence (LCT)**: `ΔW = η · φ · P_sig · C`, a learning
rule with **no gradient descent, no loss function, no backpropagation**. The
learning signal is the persistent-homology signature `P_sig` of the weight
matrix, modulated by coherence `C` and phase `φ`.

**Verified evidence (already acquired, all traceable).**

| Claim | Evidence | Where |
|---|---|---|
| Monotonicity of R = P_sig on real QPU hardware | Spearman **ρ = +0.713** across energy levels | IBM QPU (ibm_marrakesh), Job IDs `d9u47t0u5hac73agnhj0`, `d9u48aj43mgs73esfle0`, `d9u48o498n5s7392c0jg` |
| Invariance under energy on QPU (20 qubits, 4096 shots) | Pearson ≈ 1, CV = 0.0000 | ibm_marrakesh, Job `d9u42dt35hes73fje2bg` |
| Monotonicity on additional systems | 4MZI **+0.93**, 3KMD **+0.80**, finance **+0.903** | OSF preprint §5 |
| Quantum state tomography (exact) | ρ = **+1.000** | Statevector CPU |
| Emotion-conditioned language (greedy+beam, 4 emotions) | e.g. happy → "haha you are so funny too" | `docs/EVOLUTION_RATIS_NET.md` |
| Honest re-evaluation (no label leak) | v4 baseline 0.333, centroid learner **0.501** | Same doc, confusion matrix |

**Publication.** LCT preprint (10 sections), DOI **10.17605/OSF.IO/WF7QM**,
ORCID **0009-0000-4092-5313**. 9 publicly traceable QPU Job IDs.

**Bottleneck (honest).** Classical validation is done; credits are nearly
exhausted (≈ 6 min left) and the Ryzen 5 saturates on full-corpus persistent
homology (GUDHI). RATISS already runs LCT on real hardware — it needs
**power**, not funds.

---

## 1. IBM Quantum → program: Quantum Credits / Startup Credits

**Requested.** 5–10 QPU hours on utility-class devices (100+ qubits; current
access: ibm_fez / ibm_marrakesh / ibm_kingston).

**Goal.** Scale the falsifiable LCT measurements (R = P_sig monotonicity +
invariance) from 20-qubit ansatz to 100–156 qubit circuits, across all three
accessible devices, with publicly traceable Job IDs (9 already published).

**Fit with their criteria.** Tests classical-method limits with a novel idea:
learning derived from topological persistence on quantum hardware — not a
backprop-based LLM. We falsify our own formulation (2 of 3 formulations
documented FAIL); that is the program's spirit.

**Technical summary (EN, ~500 words for the application).**

> RATISS implements learning by the Law of Topological Coherence (LCT):
> ΔW = η · φ · P_sig · C, where P_sig is the persistent-homology signature
> of the weight matrix. On IBM QPUs we already measured monotone increase
> of R = P_sig with coherence (Spearman ρ = +0.713 at 20 qubits) and its
> invariance under energy re-scaling (CV = 0.0000) using 4096-shot ensembles.
> All Job IDs are public. The remaining scientific question is scalability:
> does the monotone window persist at 100+ qubits, where classical emulation
> collapses? We request utility-class circuits on ibm_fez / ibm_marrakesh /
> ibm_kingston (the three devices already authorized to our account) to sweep
> t–J ansatz parameters across 4 energy levels and 3 coherence regimes.
> Deliverables: (i) Job IDs published, (ii) OSF preprint update with
> falsification tables, (iii) open-source release of the sweep code in
> Ratiss-experimental-IA-. IBM Resources required: QPU hours only. No HPC
> infrastructure desired. The project's weight is scientific, not commercial:
> an AGI architecture whose learning signal is a measured topological
> quantity, falsifiable by construction.

---

## 2. AWS Cloud Credits for Research → program: Research Credits / Science-as-a-Service

**Requested.** Compute credits (target: high-memory EC2) — the measured need is
RAM for GUDHI persistent homology over the full 30,160-dialogue EmoContext
corpus and the 15,122-word topo vocabulary (cache already 559 MB... committed
.npz — 15,122 signatures computed once on CPU in 537 s).

**Goal.** (i) Expand the committed topological cache to the full corpus at
scale; (ii) train/evaluate the next learners (multi-layer, lateral inhibition,
learnable embedding — v5) beyond the honest centroid baseline 0.501;
(iii) cross-provider replication of QPU sweeps via Amazon Braket on
non-IBM backends, as independent replication evidence for the preprint.

**Milestones (deliverable-like, as they demand).**

| # | Milestone | Compute | Deliverable |
|---|---|---|---|
| M1 | Full-corpus topo cache (30,160 dialogues, normed lookup) | EC2 high-memory (r6i/r6a.32xlarge, ~1 TB RAM) | `data/cache/full_topo_signatures.npz` |
| M2 | v5 learner (multi-layer) > centroid 0.501 baseline | EC2 + optional SageMaker | accuracy table, honest fail documented |
| M3 | Cross-provider replication (Braket vs IBM QPU sweeps) | Braket | Job-set comparison table for OSF update |

**Technical summary (EN, ~500 words for the application).**

> RATISS is a Science-as-a-Service topological AI: the LCT rule replaces
> gradient descent with persistent-homology signatures as the learning
> signal. The community artifact we ship is the topological signature cache
> (deterministic, seeded, published) so that any researcher can run
> topological embeddings in O(1) without GUDHI. AWS services specifically
> requested: high-memory EC2 (r6i/r6a) for full-corpus Rips filtrations
> (GUDHI), SageMaker for the v5 learner sweeps, and Braket for
> cross-provider quantum replication of the IBM QPU measurements (Spearman
> ρ = +0.713 traceable Job IDs). This is deliberately per-service: EC2 for
> classical persistent homology, Braket for quantum cross-validation. The
> community receives the cache, the learner, and the replication table—all
> open source in Ratiss-experimental-IA-, with honest fail documentation as
> a working norm.

---

## Prochaine étape (Action)

1. Jonathan remplit le formulaire IBM avec la section EN (≈ 500 mots) ci-dessus.
2. Jonathan remplit le formulaire AWS Research Credits avec la section EN +
   milestones ci-dessus.
3. On revient avec un chiffre « hameçon » précis : le Spearman **+0.713 sur QPU
   réel** (Job IDs ci-dessus) est le meilleur hook — c'est le mesurable que ces
   programmes cherchent.

---
*Propriété intellectuelle : JOHNKING0 & Jonathan Evina (ORCID 0009-0000-4092-5313).
La loi LCT est figée — ces propositions la falsifient, ne la changent pas.*
