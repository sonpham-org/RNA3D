# Stanford RNA 3D Folding Competition - Part 2

## Problem Statement

**Goal:** Predict the 3D structure of RNA molecules from their nucleotide sequence.

Given an RNA sequence (e.g., `GGGCCACAGU...`), predict the (x, y, z) coordinates of each
residue's **C1' atom** (the sugar carbon - RNA's equivalent of a protein's alpha carbon).
You must provide **5 different structural predictions** per target. The best of the 5 is scored.

**Metric:** TM-score (Template Modeling score), range 0.0-1.0, **higher is better**.
- Measures topological similarity between predicted and experimental structures
- Alignment done with US-align (rotation/translation invariant)
- Only residues matching by numbering are rewarded (no shuffling tricks)
- Final score = mean TM-score across all test targets, using each target's best-of-5

---

## Data Structure

### Inputs: `test_sequences.csv`
| Column | Type | Description |
|--------|------|-------------|
| `target_id` | str | Unique ID (e.g., R1107) |
| `sequence` | str | Full RNA sequence (ACGU), chains concatenated |
| `temporal_cutoff` | str | PDB release date cutoff |
| `description` | str | Target name/organism |
| `all_sequences` | str | FASTA of individual chain sequences |
| `stoichiometry` | str | Chain composition (e.g., `ChainA:1;ChainB:2`) |

### Output: `submission.csv`
| Column | Type | Description |
|--------|------|-------------|
| `ID` | str | `{target_id}_{residue_number}` (1-indexed) |
| `resname` | str | Nucleotide (A/C/G/U) |
| `resid` | int | Residue position (1-indexed) |
| `x_1..x_5` | float | X-coordinates for 5 predictions |
| `y_1..y_5` | float | Y-coordinates for 5 predictions |
| `z_1..z_5` | float | Z-coordinates for 5 predictions |

One row per residue. Coordinates in Angstroms, clipped to [-999.999, 9999.999].

### Training Data Available
- **train_sequences.csv** (~5,135 sequences) + **train_labels.csv** (C1' coords)
- **validation_sequences.csv** + **validation_labels.csv** (additional templates)
- **MSA directory** - precomputed multiple sequence alignments (FASTA per target)
- **All-atom structures** (~108 GB, separate dataset)
- **External data** allowed if publicly available before Sep 30, 2024 (CASP16 cutoff)

### Key Properties
- Sequence lengths: ~30 to >1000 nucleotides
- Multi-chain targets common (RNA complexes)
- ~108-119 test targets
- Some targets have close templates in training set, others are novel

---

## Competition Constraints
- **Notebook-only:** Must submit via Kaggle Notebooks
- **No internet** during scoring run
- **GPU runtime limit** applies (typically ~9 hours for GPU notebooks)
- **All weights/data must be pre-uploaded** as Kaggle datasets
- **Deadline:** March 25, 2026
- **5 submissions per day**

---

## Competitive Landscape

### Part 1 Results (completed Sep 2025)
| Rank | Team | TM-score (Private LB) |
|------|-------|----------------------|
| 1st | john | 0.577 |
| 2nd | odat | 0.564 |
| 3rd | Eigen | 0.542 |
| Baseline | AlphaFold3 | ~0.48 |
| Baseline | Vfold (human experts) | 0.461 |
| Post-comp | RNAPro (NVIDIA) | **0.640** |
| Post-comp | Agentic tree search | **0.635** |

### Part 1 Key Finding
**Template-based methods dominated.** 95% of Part 1 targets had usable templates.
- john (1st): TBM + DRFold2, TBM-only scored 0.593
- odat (2nd): Embedding-based template discovery
- Eigen (3rd): Protenix variant retrained for RNA

### Part 2 Differences
- **Harder targets** - explicitly includes template-free RNA structures
- Need both TBM (for easy targets) AND de novo (for hard targets)
- Ensembling across methods even more critical

### Medal Thresholds (depends on # teams)
- **Silver:** Top 5% of teams (target for us)
- **Bronze:** Top 10%
- **Gold:** Top 10 + floor(N/500)

---

## Our Current State
- **Score: 0.360** using RNAPro + TBM hybrid
- Approach: TBM generates templates → fed to RNAPro → merge results
- Key weaknesses identified (see STRATEGY.md)

---

## Critical Research Findings

### What matters most (from ablations & competition results):
1. **Template quality** - #1 factor. Better templates = dramatically better scores
2. **Ensembling** - Combining multiple methods beats any single method
3. **RibonanzaNet2** - RNA foundation model, important but secondary to templates
4. **MSAs** - Help when available, but not all targets have good MSAs
5. **Secondary structure** - Correct 2D structure input can beat AF3 (trRosettaRNA2)

### Available Models/Tools
| Model | Type | Strengths | Weaknesses |
|-------|------|-----------|------------|
| RNAPro | DL (AF3-based) | Best single model, templates+MSA+RiboNet2 | 512 token limit, heavy |
| TBM | Template-based | No length limit, fast | Only works with templates |
| Protenix | DL (AF3 repro) | Open source, flexible | Generic, not RNA-optimized |
| DRFold2 | DL for RNA | Used by 1st place | Separate tool |
| trRosettaRNA2 | DL for RNA | Top CASP16 automated server | Needs good 2D structure |
| RhoFold+ | DL + language model | Pre-trained on 23.7M sequences | Smaller model |
| Physics-based | FARFAR2, BRiQ | Good for refinement | Very slow |

### Template Generation Approaches
1. **Sequence alignment** (our current TBM) - align test to train, transfer coords
2. **MMseqs2 search** - fast homology search against PDB
3. **john's 1st-place TBM** - the gold standard for template generation
4. **Embedding-based** (odat's approach) - find templates by learned similarity
