# CONTEXT.md — bib_rag Domain Glossary

> **Why this file exists** (borrowed from Matt Pocock's `grill-with-docs`): A shared language between you and the agent. Variables, files, agent responses, and search queries all use the same vocabulary → fewer tokens, fewer mistakes, fewer hallucinations.
>
> **Rules**:
> 1. If a term below is used anywhere in `bib_rag` (queries, writer output, debate writer, agentic graph) — **use exactly this spelling**.
> 2. If you need a new term, add it here **before** using it elsewhere.
> 3. If a term drifts in meaning, update this file in the same commit.

---

## 1. Core Concept: Eph–ephrin System

| Term | Definition | Use it for |
|------|-----------|-----------|
| **Eph receptor** | Receptor tyrosine kinase (RTK); 14 vertebrate members split into **EphA** (A1–A8, A10) and **EphB** (B1–B4, B6). Binds ephrin ligands on neighboring cells. | "EphA2 activation drives…" (not "Eph kinase") |
| **ephrin** | Ligand for Eph receptors. Two classes: **ephrin-A** (GPI-anchored, 5 members A1–A5) and **ephrin-B** (transmembrane, 3 members B1–B3). A-class binds EphA, B-class binds EphB — with cross-binding exceptions. | "ephrin-B reverse signaling…" (not "ephrin ligand") |
| **bidirectional signaling** | Both Eph (forward, in receptor cell) and ephrin (reverse, in ligand cell) signal simultaneously upon cell-cell contact. **Never** call this "two-way signaling" or "dual signaling". | "Bidirectional signaling coordinates…" |
| **cis-interaction** | Eph–ephrin binding on the **same cell surface** (lateral). Often inhibits trans signaling. | "Cis-attenuation of EphB…" |
| **trans-interaction** | Eph–ephrin binding **between two cells** (the canonical mode for boundary formation). | "Trans-binding triggers repulsion…" |

## 2. The Segregation Problem (Core Phenomenon)

| Term | Definition |
|------|-----------|
| **cell segregation** | Sorted arrangement of two cell populations into distinct, non-mixed territories. (Not "cell sorting" — that word is ambiguous in computer science.) |
| **boundary formation** | The line or zone where two segregated populations meet. (Not "border" in isolation — too vague.) |
| **border sharpening** | Progressive straightening and refinement of a boundary over time, from wavy/diffuse to sharp/linear. |
| **compartmental boundary** | Boundary between two embryonic tissue compartments (e.g., rhombomere boundaries, somite boundaries). |
| **heterotypic repulsion** | Repulsion **between** two different cell types — the mechanism by which Eph–ephrin drives segregation. |
| **homotypic repulsion** | Repulsion **among same-type cells** — usually suppressed by N-cadherin to enable cohesion within a compartment. |

## 3. The Three Competing Models (Use These Names)

When the literature argues about mechanism, it falls into one of these camps. **Always name the model explicitly**, never "the prevailing view" or "alternative explanation":

### Model A: **repulsion-driven segregation** (Taylor et al. 2017 + many)
- Different Eph/ephrin levels on the two cell types → heterotypic repulsion > homotypic repulsion → cells sort.
- **Cadherin role**: N-cadherin suppresses homotypic repulsion (enables within-type cohesion); not the driver.
- **Use it for**: "We adopt the repulsion model (Taylor et al., 2017) to interpret…"

### Model B: **differential adhesion** (older, Steinberg; largely rejected for Eph context)
- Different cadherin levels → cadherin-mediated adhesion differentials sort cells.
- **Use it for**: "The differential adhesion model alone cannot explain…" (this is the model you reject)

### Model C: **cortical tension** (O'Neill et al. 2016, Calzolari et al. 2014)
- Actomyosin-generated surface tension differs between cell types → cell rounding & sorting from energy minimization.
- **Use it for**: "Cortical tension measurements (O'Neill et al., 2016) suggest…"

### Model D: **endocytosis / cell-shape remodeling**
- Eph–ephrin endocytosis (both cells) physically pulls cell bodies apart. (Decuzzi, Kadir et al.)
- **Use it for**: "Endocytic pulling (Kadir et al., …) provides a mechanical basis for…"

## 4. Cell-Surface Mechanics (Cadherin, N-cadherin)

| Term | Definition |
|------|-----------|
| **N-cadherin** | The specific classical cadherin studied in this system. **Not "cadherin"** unless type-agnostic. |
| **cadherin-mediated adhesion** | Type-agnostic; use when citing generic cadherin papers. |
| **adhesion differential** | Difference in adhesion strength between two cell types. Used in Model B. |
| **Eph–cadherin crosstalk** | Bidirectional regulation between Eph signaling and cadherin adhesion. (Not "Eph-cadherin interaction".) |
| **adherens junction** | The structural complex anchoring cadherin to actin. |

## 5. Methodological Vocabulary (use exactly)

| Term | Use it for |
|------|-----------|
| **expression pattern** | Spatial distribution of a protein/mRNA in a tissue. (Not "expression map".) |
| **in situ hybridization / ISH** | Spatial mRNA detection. |
| **immunostaining** | Antibody-based protein localization. |
| **knockdown (KD)** | siRNA / shRNA / morpholino reduction. |
| **knockout (KO)** | Genetic deletion. **Never "knock-down"** for KO. |
| **conditional KO (cKO)** | Tissue/timing-specific deletion. |
| **CRISPR-Cas9** | Specify variant if relevant (Cas9, Cas12a, dCas9-KRAB for repression). |
| **co-culture** | Two cell types grown together in vitro. |
| **cell mixing assay** | Dissociated cells allowed to re-aggregate; quantifies sorting. |
| **time-lapse microscopy** | Live imaging. Specify frame interval. |
| **FRET** | Förster resonance energy transfer — proximity reporter. |
| **FRAP** | Fluorescence recovery after photobleaching — membrane mobility. |
| **super-resolution** | STED / SIM / STORM / PALM — when below diffraction limit matters. |

## 6. Tissues & Developmental Stages

| Term | Use it for |
|------|-----------|
| **neural crest** | Migratory embryonic cell population. (Not "neural crest cells" in shorthand.) |
| **rhombomere** | Hindbrain segmental unit (r1–r7). |
| **somite** | Mesodermal segment of the paraxial mesoderm. |
| **embryonic day E[n]** | Mouse staging. (e.g., "E8.5" not "embryonic day 8.5".) |
| **HH[n] (Hamburger-Hamilton)** | Chick staging. |
| **segmental plate** | Unsegmented presomitic mesoderm. |
| **otic vesicle** | Inner-ear primordium. |
| **forebrain / midbrain / hindbrain** | Anatomical regions; use the lowercase words. |
| **Drosophila germband extension** | Specific to invertebrate boundary studies. |

## 7. Cell Types (use these exact names)

- **HEK293 / HEK293T** — not "293 cells"
- **MDCK** — Madin-Darby canine kidney
- **HeLa** — not "hela" or "HeLa cells" on second mention
- **NIH-3T3** — not "3T3"
- **primary neuron** — not "primary neurons" on first use
- **cortical neuron** — specifies origin
- **hippocampal neuron** — specifies origin

## 8. Common Anti-Patterns (don't write these)

❌ "Eph kinase" → ✅ "Eph receptor"
❌ "two-way signaling" → ✅ "bidirectional signaling"
❌ "border formation" alone → ✅ "boundary formation" or specify "compartmental boundary"
❌ "cell sorting" → ✅ "cell segregation"
❌ "the prevailing view" → ✅ name the model: "the repulsion model (Taylor et al., 2017)"
❌ "decreased adhesion" → ✅ "reduced heterotypic adhesion" or "adhesion differential" (specify the cell pair)
❌ "EphB-ephrinB" → ✅ "EphB–ephrin-B" (en-dash, hyphen in ephrin-B)
❌ "adhesion molecule" → ✅ "adhesion receptor" (cadherins are receptors, not molecules)
❌ "morpholino" lowercase in title case → ✅ "Morpholino" (it's a brand class)
❌ "Eph–ephrin interaction" → ✅ "Eph–ephrin binding" (interaction is vague; binding specifies the biochemical event)

## 9. Acronyms (expand on first use, then acronym)

- **RTK** — receptor tyrosine kinase
- **GPI** — glycosylphosphatidylinositol (anchor)
- **GPCR** — G-protein-coupled receptor
- **PDZ** — PSD-95/Dlg/ZO-1 domain
- **SH2** — Src homology 2
- **PI3K** — phosphoinositide 3-kinase
- **MAPK** — mitogen-activated protein kinase
- **aPKC** — atypical protein kinase C
- **ECM** — extracellular matrix
- **KD** — knockdown (context-dependent; for DNA-binding affinity it's "dissociation constant")

## 10. Citation Style (default: APA)

In-text: `(Taylor et al., 2017)` or `Taylor et al. (2017) demonstrated that…`
- Two authors: `(Smith & Jones, 2020)`
- Three+ authors: `(Smith et al., 2020)` (always use "et al." for 3+)
- First citation of a model name: spell out: "the repulsion model (Taylor et al., 2017)"

---

_Last updated: 2026-06-07 (added by hindsight/manual). Update this file in the same commit as any new term._
