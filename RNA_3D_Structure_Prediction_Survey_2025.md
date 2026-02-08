# State of the Art: RNA 3D Structure Prediction (2024-2026)

## A Comprehensive Survey

---

## Table of Contents

1. [Introduction and Context](#1-introduction-and-context)
2. [Deep Learning Approaches](#2-deep-learning-approaches)
3. [Template-Based Methods](#3-template-based-methods)
4. [Physics-Based / Energy Minimization Methods](#4-physics-based--energy-minimization-methods)
5. [Ensemble / Hybrid Strategies](#5-ensemble--hybrid-strategies)
6. [Model Quality Assessment (MQA)](#6-model-quality-assessment-mqa)
7. [Key Insights and Practical Guidance](#7-key-insights-and-practical-guidance)
8. [Competitions and Blind Benchmarks](#8-competitions-and-blind-benchmarks)
9. [Open-Source Tools and Resources](#9-open-source-tools-and-resources)
10. [Outlook and Open Problems](#10-outlook-and-open-problems)

---

## 1. Introduction and Context

### Why RNA 3D Structure Prediction Is Hard

RNA 3D structure prediction remains an **unsolved grand challenge**, substantially harder
than protein structure prediction for several fundamental reasons:

- **Data scarcity**: As of July 2024, the PDB contains ~200,000 protein structures but only
  ~7,956 RNA structures. After filtering for non-redundant, high-resolution (<3.5 A)
  entries, only ~2,416 RNA structures remain. RNA-only structures comprise less than 1%
  of all PDB entries.

- **Conformational heterogeneity**: RNA sequences can fold into **multiple conformations**,
  whereas most protein sequences converge to a single stable fold. This makes the
  prediction target itself ambiguous.

- **Structural complexity**: RNA secondary structure is defined by hydrogen bonds between
  nucleobases (not backbone atoms as in proteins), and the phosphate backbone conformation
  is strongly constrained by base-pairing. Non-canonical base pairs, pseudoknots, and
  tertiary motifs (e.g., A-minor interactions, G-quadruplexes) further complicate prediction.

- **Fewer evolutionary sequences**: RNA families often have shallower MSAs compared to
  protein families, limiting the co-evolutionary signal available to deep learning methods.

The field is currently estimated to be in a stage analogous to protein structure prediction
**before AlphaFold2** -- significant progress has been made, but no single method dominates
across all targets.

---

## 2. Deep Learning Approaches

### 2.1 AlphaFold3 (DeepMind/Isomorphic Labs, 2024)

**Paper**: Abramson et al., Nature 2024
**Access**: AlphaFold Server (https://alphafoldserver.com)

AlphaFold3 (AF3) extends the AlphaFold framework to predict structures of proteins, DNA,
RNA, small molecules, ions, and their complexes. It uses a diffusion-based structure module
(replacing the structure module of AF2) and can model chemical modifications.

**RNA Capabilities**:
- Can predict RNA-only structures, protein-RNA complexes, and RNA-ligand complexes
- Produces plausible models for RNAs up to ~2000 nucleotides in length
- Adequate performance for small-to-medium sized RNAs (RMSD comparable to specialized tools)

**Known Limitations**:
- Models often contain **severe steric clashes** and **backbone breaks** in the
  phosphodiester chain, with probability increasing with RNA length
- Predicted structures tend to be **excessively spherical** (too-low anisotropy),
  especially for longer RNAs
- Does not consistently reproduce non-Watson-Crick interactions crucial for structural
  stability
- Struggles with orphan RNA families lacking supplementary contextual information
- In the CASP16 blind assessment, AF3 did **not outperform human-assisted methods** on
  RNA targets
- Thousands of predictions may be required to obtain models free of geometric problems

**Key Reference**: "Has AlphaFold3 achieved success for RNA?" (Acta Cryst D, 2025) --
a critical analysis showing that while AF3 represents progress, it has not achieved for
RNA what AF2 achieved for proteins.

---

### 2.2 RoseTTAFold2NA (Baker Lab, UW, 2023)

**Paper**: Baek et al., Nature Methods 2024
**Code**: https://github.com/uw-ipd/RoseTTAFold2NA

RoseTTAFold2NA extends the RoseTTAFold2 platform to predict structures of protein-DNA
and protein-RNA complexes, as well as RNA tertiary structures.

**Architecture**: Extends the three-track (1D sequence, 2D pairwise, 3D structure)
architecture of RoseTTAFold to handle nucleic acid sequences. Trained on all RNA,
protein-RNA, and protein-DNA complexes in the PDB.

**Strengths**:
- Rapid 3D structure prediction with confidence estimates
- Good performance on protein-nucleic acid complexes
- Confident predictions have considerably higher accuracy than previous methods
- Useful for designing sequence-specific RNA/DNA-binding proteins

**Limitations**:
- Performance on RNA-only structures is less competitive than specialized RNA methods
- Outperformed by AF3 on protein-nucleic acid benchmarks (significant gains in TM-score
  and lDDT), though the advantage is more limited for RNA multimers

---

### 2.3 RhoFold+ (CUHK, 2024)

**Paper**: Shen et al., Nature Methods 2024
**Code**: https://github.com/ml4bio/RhoFold
**Server**: https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/

RhoFold+ is currently one of the **leading automated RNA 3D structure prediction methods**.

**Architecture**:
- Integrates **RNA-FM**, an RNA language model pretrained on ~23.7 million RNA sequences
  using a BERT-like masked-nucleotide prediction objective
- End-to-end pipeline from sequence to 3D coordinates
- Uses a geometry-aware Invariant Point Attention (IPA) mechanism for 3D refinement
- Predicts secondary structure and interhelical angles as auxiliary outputs

**Training Strategy** (addressing data scarcity):
- Initial training on PDB data only
- Self-distillation: uses initially trained model to generate pseudo-structural labels
- Retraining with 25% PDB data + 75% distillation data

**Performance**:
- Retrospective evaluations on RNA-Puzzles and CASP15 demonstrate superiority over
  existing methods, including human expert groups
- Fully automated, end-to-end, requiring only sequence as input
- Produces per-residue confidence scores

---

### 2.4 trRosettaRNA / trRosettaRNA2 (Yang Lab, Shandong University)

**Paper**: Wang et al., Nature Communications 2023
**Server**: https://yanglab.qd.sdu.edu.cn/trRosettaRNA/

**Architecture**:
- **Two-stage pipeline**: (1) Transformer network (RNAformer) predicts 1D and 2D
  geometric restraints; (2) Rosetta energy minimization generates full-atom 3D models
- RNAformer uses 48 transformer blocks cycled 4 times
- Combines learned deep-learning potentials with physics-based Rosetta energy terms

**trRosettaRNA2** (2024-2025):
- Updated version with improved accuracy and efficiency
- Uses **fewer parameters and computational resources** than alternatives
- **Ranked 1st among automated servers** (as "Yang-Server") in CASP16 RNA prediction,
  surpassing AlphaFold3
- Key innovation: **highly sensitive to secondary structure input quality** -- optimizing
  the 2D structure input yields more accurate predictions than AF3
- Flexibility in leveraging diverse secondary structure inputs enables exploration of
  RNA conformational space

**Significance**: trRosettaRNA2 is described as representing a "solid step towards the
AlphaFold moment for RNA."

---

### 2.5 NuFold (Kihara Lab, Purdue)

**Paper**: Kagaya et al., Nature Communications 2025
**Code**: https://github.com/kiharalab/NuFold

**Architecture**:
- End-to-end deep learning approach predicting all-atom RNA 3D structures from sequences
- Features a **nucleobase center representation** enabling flexible ribose ring conformations
- Predicts torsion angles for main chain, chi angles, and ribose ring (reproducing
  sugar-puckering formations)
- Three components: Preprocessing, EvoFormer (MSA-based co-evolutionary extraction),
  and Structure Module

**Performance**:
- Clearly outperforms energy-based methods
- Comparable to state-of-the-art deep learning methods
- Particular advantage in building **correct local geometries**
- Capable of predicting RNA multimer complex structures by linking input sequences
- KiharaLab ranked **3rd overall in CASP16 RNA** (top human predictor group)

---

### 2.6 DeepFoldRNA (Zhang Lab)

**Paper**: Pearce et al., Nature Communications 2023

**Architecture**:
- Deep self-attention neural networks predict geometric restraints (distances,
  orientations, torsion angles) from MSA and secondary structure
- Restraints are used to guide structure assembly through energy minimization

**Performance**:
- **Best overall automated method** in systematic 2024 benchmarking (Bahai et al.,
  PLOS Comp Bio 2024)
- Lowest average RMSD (12.84 A) on a newly compiled dataset
- ~50% correct predictions (within 5 A) on a 65-target benchmark

---

### 2.7 DRfold / DRfold2 (Zhang Lab)

**Paper**: Li et al., Nature Communications 2023
**Server**: https://aideepmed.com/DRfold2/

**Architecture**:
- Self-attention transformer networks learn coarse-grained RNA structures directly
  from sequence
- Uses a P, C4', N coarse-grained model for training efficiency
- Predicted conformations are optimized by a separately trained deep-geometric potential
  through gradient-descent simulation

**DRfold2** (2024-2025):
- Built on a novel **composite language model** that captures co-evolutionary patterns
  and secondary structure information
- Improved accuracy over original DRfold

**Performance**:
- **Second-best automated method** in 2024 benchmarking (after DeepFoldRNA)
- Outperforms previous approaches by >73.3% in TM-score on non-redundant datasets

---

### 2.8 RNAPro (NVIDIA, 2025-2026)

**Code**: https://github.com/NVIDIA-Digital-Bio/RNAPro
**Weights**: https://huggingface.co/nvidia/RNAPro-Public-Best-500M

**Architecture**:
- Combines an **AF3-like co-folding architecture (Protenix)** with RNA-specific modules
- Uses a frozen pretrained **RibonanzaNet2** (100M-parameter RNA foundation model) as
  encoder for sequence and pairwise features
- Features are injected into RNA-post-trained Protenix with learned gating
- Includes: input embedder, MSA module, gating module, template module, Pairformer
  blocks, and diffusion-based structure module

**Key Innovation**: First model to successfully combine:
1. RNA foundation model (RibonanzaNet2) for rich sequence features
2. Structural MSA information
3. Template-based modeling
4. AF3-like diffusion architecture

Developed in collaboration with Stanford Das Lab and winners of the Stanford RNA 3D
Folding Kaggle competition. Released January 2026 under NVIDIA Clara.

---

### 2.9 RNAgrail (2024)

**Venue**: NeurIPS MLSB Workshop 2024

**Architecture**:
- Combines **graph neural networks (GNN)** with **denoising diffusion probabilistic
  models (DDPM)**
- Uses a coarse-grained 5-bead RNA representation for computational efficiency
- RNA structure is corrupted via diffusion (Gaussian noise on coordinates) and the
  network learns to reverse the process

**Performance**:
- Outperformed AlphaFold3 by 12% in mean RMSD, 24% in mean eRMSD, and 40% in mean INF
- Preserves **100% of Watson-Crick-Franklin base pair interactions** in predictions
  (unique among current methods)
- Open-source with training code and data

---

### 2.10 Protenix (ByteDance, 2025-2026)

**Code**: https://github.com/bytedance/Protenix

A comprehensive **open-source reproduction of AlphaFold3** by ByteDance's AI4Science Team.

- Achieves state-of-the-art performance across protein-ligand, protein-protein, and
  protein-nucleic acid predictions
- Matches AF3 performance on RNA structure prediction (competitive lDDT and TM-scores)
- Surpasses RoseTTAFold2NA
- **Protenix-v1** (released 2026-02-05): Added support for template/RNA MSA features
- Full access to trained weights, training code, and inference code
- Apache 2.0 license

---

### 2.11 Chai-1 and Boltz-1 (2024)

**Chai-1**: https://github.com/chaidiscovery/chai-lab
**Boltz-1**: https://github.com/jwohlwend/boltz

Both are open-source biomolecular structure prediction models that support RNA:
- **Chai-1**: Multi-modal foundation model supporting proteins, small molecules, DNA,
  RNA, and glycosylations. Claims better performance than AF3 on some benchmarks.
- **Boltz-1**: First fully open-source model to approach AF3 accuracy. Released under
  MIT license with full training and inference code.

Both allow input of arbitrary nucleic acid sequences and can model protein-RNA complexes.

---

### 2.12 RNA Language Models (Foundation Models)

Several RNA language models serve as feature extractors or embeddings for downstream
structure prediction:

| Model | Parameters | Training Data | Key Feature |
|-------|-----------|---------------|-------------|
| **RNA-FM** | - | 23.7M ncRNA sequences | BERT-like, used by RhoFold+ |
| **RiNALMo** | 650M | 36M ncRNA sequences | Largest RNA LM; generalizes to unseen families |
| **RNA-MSM** | - | MSA-based training | Captures co-evolutionary signals |
| **RNAErnie** | - | Motif-aware pretraining | Incorporates RNA motifs as biological priors |
| **RNABERT** | - | - | Early RNA language model |
| **RibonanzaNet2** | 100M | Crowdsourced chemical mapping | Used by RNAPro; SOTA in 2D prediction |

These models learn rich representations of RNA sequences in a self-supervised manner.
When fine-tuned on downstream tasks, they can significantly improve secondary structure
prediction, solvent accessibility estimation, and 3D structure inference.

---

## 3. Template-Based Methods

### 3.1 How Template-Based Modeling Works for RNA

Template-based (or homology-based) RNA modeling follows a general workflow:

1. **Template identification**: Search for homologous RNAs with known structures using
   sequence alignment (BLAST, nhmmer), covariance models (Infernal/Rfam), or structural
   alignment tools
2. **Alignment**: Align the query sequence to the template, incorporating secondary
   structure information (tools: R-Coffee, LocARNA, Infernal)
3. **Model building**: Transfer coordinates from the template structure to the aligned
   query sequence
4. **Loop and gap modeling**: Model regions without template coverage using fragment
   libraries or de novo methods
5. **Refinement**: Energy minimization and clash resolution

### 3.2 Template Selection Best Practices

- **Sequence identity threshold**: RNA loops differing by <25% in sequence identity
  fold into very similar structures -- this threshold guides template selection
- **Rfam-based search**: When BLAST fails, the Rfam database can perform sophisticated
  searches against its ncRNA family archive and return alignments. Pre-calculated Rfam
  alignments may require manual refinement
- **Structure-aware alignment**: Software that accounts for covariation in RNA secondary
  structure provides more accurate alignments than sequence-only methods
- **Multiple templates**: Multiple templates can seed parallel modeling runs, though
  selecting a subset optimizes computational resources

### 3.3 Key Template-Based Tools

**ModeRNA**: A tool for comparative modeling of RNA 3D structure. Takes a template
structure and an alignment as input, and builds a model by transferring coordinates.

**Vfold / Vfold-Pipeline**: A hierarchical, hybrid method:
- Uses Vfold2D and VfoldMCPX for 2D structure prediction
- **Motif-based template assembly**: Uses the Vfold3D program to assemble 3D templates
  for motifs identified in the 2D structure
- VfoldLA decomposes structures into helices and loop strands when motif templates
  are unavailable
- **Top performer at CASP16**: The Vfold group was the #1 human expert predictor,
  integrating templates with AF3 predictions and physics-based models

**3dRNA**: Uses secondary structure elements (SSEs) as building blocks:
- Decomposes known structures into 1-3mer fragments
- Combinatorially reassembles fragments to minimize molecular interaction energy
- Uses 3dRNAscore for candidate ranking
- Strong on pseudoknots and large RNAs with tertiary contacts

### 3.4 Critical Insight: Template Availability Dominates Accuracy

CASP16 assessment clearly demonstrated that **3D accuracy generally depends on the
availability of closely related 3D structure templates in the PDB**. When accounting
for template availability, there has not been a notable increase in nucleic acid
modeling accuracy between previous blind challenges and CASP16.

---

## 4. Physics-Based / Energy Minimization Methods

### 4.1 Rosetta RNA Tools

The Rosetta software suite provides several RNA-specific tools:

**FARFAR2 (Fragment Assembly of RNA with Full-Atom Refinement 2)**:
- **Core approach**: Assembles short (1-3 nucleotide) fragments from existing RNA
  crystal structures
- Accepts RNA sequence, secondary structure (dot-bracket notation), and optional
  template structures
- Two stages: (1) Low-resolution fragment assembly with base-pair step sampling;
  (2) All-atom refinement with Rosetta energy function
- Implements "base pair step" sampling allowing quartet substitution of helical residues
- In 16 of 21 RNA-Puzzles revisited, FARFAR2 recovers native-like structures more
  accurate than originally submitted models
- Best for RNAs up to ~200 nucleotides
- Available on ROSIE web server

**Stepwise Assembly / Stepwise Monte Carlo**:
- Deterministic or stochastic buildup of RNA structures one residue at a time
- Can reach **atomic accuracy for small motifs** (<=12 residues)
- Bottleneck: complete conformational sampling for larger RNAs
- Greater computational expense; not yet straightforward for public server use

**RNA_denovo (rna_denovo)**:
- The underlying Rosetta application for fragment-based RNA structure assembly
- Input: sequence + secondary structure; output: 3D models ranked by Rosetta score
- Can incorporate experimental restraints (SHAPE, DMS, hydroxyl radical probing)

### 4.2 Coarse-Grained Models

**SimRNA**:
- Monte Carlo simulation using a coarse-grained RNA representation
- Simulation modes: isothermal, simulated annealing, Replica Exchange Monte Carlo (REMC)
- Explores RNA conformational space efficiently
- Computationally intensive but captures thermodynamic folding landscapes
- One of six methods benchmarked in the 2024 NAR comparative analysis

**IsRNA (Iterative Simulated RNA)**:
- 4/5-bead coarse-grained representation per nucleotide
- Extracts correlated energy functions from known structures through iterative MD
  simulations
- Accounts for both native and non-native interactions
- Recent extension: **IsRNAcirc** (2024) for circular RNA 3D structure prediction

**BRiQ (Base Rotameric and Quantum-mechanical)**:
- Knowledge-based energy function with quantum mechanics corrections for base-base
  interactions
- Atom-level refinement capability
- Performance: improves 81% of Rosetta-SWM structures (RMSD <2A), 100% of RNA-Puzzle
  structures (RMSD <4A), 83% of FARFAR2 structures (RMSD <6A)
- Used by top-performing AIchemy_RNA2 group in CASP15
- Available at: https://github.com/Jian-Zhan/RNA-BRiQ

### 4.3 Molecular Dynamics Approaches

All-atom MD simulations using force fields such as:
- **AMBER** (ff99, OL3 for RNA): Most widely used for RNA simulations
- **CHARMM** (CHARMM27/36 for nucleic acids): Accurate ribose moiety representation
- **GROMACS**: Popular open-source MD engine with nucleic acid support

MD is primarily used for **refinement and dynamics** rather than ab initio prediction:
- Structure refinement after initial model generation
- Conformational sampling around a predicted structure
- Studying RNA dynamics and flexibility
- Validating predicted structures through stability assessment

---

## 5. Ensemble / Hybrid Strategies

### 5.1 Combining Multiple Prediction Methods

The CASP16 results clearly showed that **the best performing groups used ensemble/hybrid
approaches**:

**Vfold (Top CASP16 performer)**:
- Integrates knowledge from templates, AlphaFold3 predictions, and physics-based models
- Hierarchical pipeline combining 2D prediction, template assembly, and refinement

**GuangzhouRNA-human (2nd at CASP16)**:
- Human-expert integration of multiple methods with manual assessment

**KiharaLab / NuFold (3rd at CASP16)**:
- End-to-end deep learning combined with human expertise for difficult targets

### 5.2 General Ensemble Strategy

A practical ensemble approach for RNA 3D prediction:

1. **Generate diverse candidates**: Run multiple methods (e.g., AF3, trRosettaRNA2,
   RhoFold+, DeepFoldRNA, FARFAR2)
2. **Score and rank**: Use MQA tools (ARES, lociPARSE, Rosetta score) to rank models
3. **Select and refine**: Choose top-ranked models; refine with BRiQ or MD
4. **Consensus analysis**: Identify structural features that are consistent across
   methods (higher confidence)

### 5.3 Input Optimization Strategy

An important finding from CASP16 and recent work (ICLR 2025):
- **Secondary structure input quality** is critical for methods like trRosettaRNA2
- Optimizing the 2D structure input resulted in more accurate predictions than AF3
- Different 2D structure predictors (RNAfold, CONTRAfold, EternaFold, experimental data)
  can be tried to find the best input
- RNA-specific MSA generation pipelines (rMSA) significantly impact prediction quality

### 5.4 The rMSA Pipeline for MSA Generation

**rMSA** (https://github.com/pylelab/rMSA):
- Hierarchical pipeline for sensitive search and accurate alignment of RNA homologs
- Merges sequences from NCBI nucleotide (NT) and RNAcentral databases
- First stage: blastn search against RNAcentral and NT
- Alignments refined by nhmmer using profile HMMs
- Critical for feeding deep learning methods with high-quality evolutionary information

**rMSA2**: Updated version with expanded database support.

---

## 6. Model Quality Assessment (MQA)

Choosing the best model from a pool of predictions is as important as generating
predictions themselves. Key MQA tools for RNA:

### 6.1 ARES (Atomic Rotationally Equivariant Scorer)

**Paper**: Townshend et al., Science 2021

- Deep neural network that predicts RMSD from the unknown native structure
- Takes only 3D coordinates and chemical element types as input
- No preprogrammed biological concepts (helices, base pairs, etc.)
- Top-scoring models included the correct structure for **81% of benchmark RNAs**
  (vs. <50% for other methods)
- Learns effectively from small datasets

### 6.2 lociPARSE (2024)

**Paper**: Baral et al., JCIM 2024

- Locality-aware Invariant Point Attention architecture
- Predicts **local Distance Difference Test (lDDT)** scores per nucleotide
- Superposition-free assessment (does not require global alignment)
- **Significantly outperforms** ARES, RNA3DCNN, and statistical potentials (rsRNASP,
  cgRNASP, DFIRE-RNA, RASP) across multiple metrics
- Inspired by AlphaFold2's IPA module

### 6.3 RNArank (2025)

- Deep learning approach extracting multi-modal features
- Y-shaped residual neural network architecture
- Predicts inter-nucleotide contact maps and distance deviation maps
- Consistently outperforms traditional methods on CASP15 and CASP16
- Handles both local and global quality assessment

### 6.4 RNAdvisor 2.0 (2025)

- Unified platform integrating multiple metrics, scoring functions, and meta-metrics
- Includes RNA-specific metrics: INF (Interaction Network Fidelity), MCQ (Mean of
  Circular Quantities)
- Also supports standard metrics: RMSD, TM-score, GDT-TS
- Web server for easy access

### 6.5 Statistical Potentials

- **3dRNAscore**: All-atom distance potential
- **RASP**: RNA All-atom Statistical Potential
- **rsRNASP / cgRNASP**: Residue-specific / coarse-grained RNA statistical potentials
- **DFIRE-RNA**: Distance-scaled finite ideal-gas reference state
- Generally outperformed by deep learning MQA methods but useful as fast filters

### 6.6 Practical MQA Recommendation

Based on 2024-2025 benchmarking:
1. Use **lociPARSE** for per-residue accuracy estimation
2. Use **ARES** for global RMSD estimation
3. Use **Rosetta score** as a physics-based sanity check
4. Consider **RNArank** for the latest deep learning-based assessment
5. Use **RNAdvisor 2.0** to get a comprehensive multi-metric evaluation

---

## 7. Key Insights and Practical Guidance

### 7.1 Short vs Long RNA

| Characteristic | Short RNA (< ~100 nt) | Long RNA (> ~120 nt) |
|---|---|---|
| Best approaches | Deep learning (RhoFold+, trRosettaRNA2), Rosetta stepwise assembly | Template-based + DL ensembles, Vfold |
| AF3 performance | Adequate, RMSD comparable to specialized tools | Declines significantly; steric clashes and backbone breaks increase |
| Physics-based | Stepwise assembly can reach atomic accuracy (<=12 nt) | FARFAR2 up to ~200 nt; computationally prohibitive beyond |
| Key challenge | Non-canonical interactions, G-quadruplexes | Complex tertiary interactions, pseudoknots |
| Benchmark note | Every RNA >120 nt is classified as "difficult" in benchmarks | No method achieves TM-score >0.8 for novel long RNAs (CASP16) |

### 7.2 Single-Chain vs Multi-Chain

| Aspect | Single-Chain RNA | Multi-Chain / Complexes |
|---|---|---|
| Best methods | RhoFold+, trRosettaRNA2, DeepFoldRNA | AF3, Chai-1, Boltz-1, Protenix, RoseTTAFold2NA |
| Template role | Important but not always necessary | Critical -- most methods rely heavily on templates |
| CASP16 performance | Poor for novel structures (no TM >0.8) | Even more challenging; limited improvement |
| Data availability | ~7,956 RNA structures in PDB | Fewer multi-chain RNA complex structures |
| Protein-RNA complexes | N/A | RF2NA, AF3 show substantial improvement over previous methods |

### 7.3 Role of MSAs for RNA Prediction

MSAs are **critically important** for RNA structure prediction, but the situation differs
from proteins:

- **Fewer homologs**: RNA families often have shallower MSAs than protein families
- **Co-evolutionary signal**: When present, co-evolutionary information from MSAs
  strongly informs secondary and tertiary structure
- **RNA-specific MSA tools** (rMSA, rMSA2) are necessary -- general protein MSA tools
  are insufficient
- **RNA-MSM** (2024): MSA-based RNA language model that explicitly captures co-evolutionary
  signals, producing attention maps directly correlated with secondary structure
- **ICLR 2025 finding**: RNA-specific structural MSA pipelines with engineered evolutionary
  features significantly improve 3D prediction accuracy
- **Template-free targets**: Good performance on OLE RNA at CASP16 suggested that structural
  information **can** be extracted from MSAs even without templates

**Practical implication**: Investing effort in building high-quality MSAs (using rMSA,
searching RNAcentral and NT databases, using Rfam covariance models) pays off
significantly for prediction accuracy.

### 7.4 Templates vs De Novo Prediction

**CASP16 established definitively**: Template availability is the **single most important
factor** determining prediction accuracy.

- When close templates exist: Template-based methods (Vfold, ModeRNA) achieve the best
  results
- When no templates exist: Deep learning methods (trRosettaRNA2, DeepFoldRNA, RhoFold+)
  provide the best chance, but accuracy remains limited
- For **orphan RNAs** (no structural homologs): ML methods are only slightly better than
  fragment-assembly methods, and all methods perform poorly
- **Accounting for template availability, there has NOT been a notable increase in nucleic
  acid modeling accuracy between previous blind challenges and CASP16** -- this is a
  sobering finding

**Practical strategy**:
1. Always search for templates first (Rfam, PDB, DALI)
2. If templates found: Use template-based modeling, refine with BRiQ or MD
3. If no templates: Generate ensemble of DL predictions, score with MQA, refine best
4. For all cases: Combine template and DL information when possible (as Vfold does)

### 7.5 Summary of Method Strengths

| Method | Best For | Key Advantage |
|---|---|---|
| **trRosettaRNA2** | General RNA, automated prediction | #1 server at CASP16; flexible 2D input |
| **RhoFold+** | Single-chain RNA from sequence | End-to-end; strong language model |
| **DeepFoldRNA** | Automated prediction | Best overall in systematic benchmark |
| **NuFold** | Local geometry accuracy | Correct sugar-puckering; multimer support |
| **AF3/Protenix** | Protein-RNA complexes | Broad biomolecular scope |
| **Vfold** | Expert-guided prediction | #1 at CASP16 (human); template+physics+ML |
| **FARFAR2** | Small-medium RNA refinement | Physics-based; handles experimental restraints |
| **RNAPro** | Comprehensive prediction | Combines FM + MSA + templates + diffusion |
| **RNAgrail** | Base-pair preservation | 100% WCF base pair recovery |
| **BRiQ** | Structure refinement | Quantum-corrected energy; excellent refinement |

---

## 8. Competitions and Blind Benchmarks

### 8.1 CASP16 RNA Results (2024)

The 16th CASP included RNA structure prediction for the second time:
- **42 targets** spanning RNA functions from dopamine binding to nanocage formation
- **65 groups** from 46 labs participated
- **Top performers** (all human expert groups): Vfold, GuangzhouRNA-human, KiharaLab
- **Top automated server**: Yang-Server (trRosettaRNA2), ranked 4th overall, 1st among
  servers, surpassing AlphaFold3
- **Key finding**: No predictions of previously unseen natural RNA structures achieved
  TM-scores above 0.8
- **Functional assessment**: Blind predictions often lack accuracy in regions of highest
  functional importance (binding sites, catalytic centers)

### 8.2 Stanford RNA 3D Folding Kaggle Competition

**Phase 1** (2024):
- First demonstration that fully automated ML models could match human experts
- Winners: Kagglers john, odat, and team Eigen
- Top TM-align scores: 0.671, 0.653, 0.615 (vs. Vfold baseline 0.461)
- Led to development of RNAPro (NVIDIA collaboration)

**Phase 2** (2025, ongoing):
- Harder targets including RNAs with no available templates
- Revised evaluation framework rewarding higher accuracy
- $75,000 prize pool
- Built around fine-tuning RibonanzaNet2 (100M-parameter RNA foundation model)
- Competition period: Feb-Sep 2025

### 8.3 RNA-Puzzles

Long-running community-wide blind prediction experiment:
- Focused specifically on RNA 3D structure
- Provides retrospective evaluation benchmarks used by most methods
- RhoFold+ and trRosettaRNA demonstrate competitive or superior performance on
  RNA-Puzzles targets

---

## 9. Open-Source Tools and Resources

### 9.1 Structure Prediction Tools

| Tool | URL | License | Notes |
|---|---|---|---|
| RhoFold+ | github.com/ml4bio/RhoFold | Apache 2.0 | End-to-end from sequence |
| trRosettaRNA | yanglab.qd.sdu.edu.cn/trRosettaRNA | - | Web server available |
| NuFold | github.com/kiharalab/NuFold | - | Multimer support |
| DeepFoldRNA | zhanggroup.org | - | Best automated benchmark |
| DRfold2 | aideepmed.com/DRfold2 | - | Server available |
| RNAPro | github.com/NVIDIA-Digital-Bio/RNAPro | Open | NVIDIA Clara model |
| RNAgrail | zenodo.org/records/13757098 | Open | GNN + diffusion |
| Protenix | github.com/bytedance/Protenix | Apache 2.0 | Open AF3 reproduction |
| Chai-1 | github.com/chaidiscovery/chai-lab | - | Multi-modal FM |
| Boltz-1 | github.com/jwohlwend/boltz | MIT | Open AF3-like model |
| RF2NA | github.com/uw-ipd/RoseTTAFold2NA | - | Protein-nucleic acid |
| FARFAR2 | rosettacommons.org | Academic | Fragment assembly |
| SimRNA | genesilico.pl/SimRNA | Academic | Coarse-grained MC |
| BRiQ | github.com/Jian-Zhan/RNA-BRiQ | - | Refinement tool |

### 9.2 MSA and Feature Generation

| Tool | URL | Purpose |
|---|---|---|
| rMSA/rMSA2 | github.com/pylelab/rMSA | RNA MSA generation |
| Infernal | eddylab.org/infernal | Covariance model search (Rfam) |
| RNAcentral | rnacentral.org | Comprehensive ncRNA database |
| Rfam 15 | rfam.org | RNA families database (2025 update) |

### 9.3 Quality Assessment

| Tool | URL | Type |
|---|---|---|
| ARES | - | DL-based global RMSD prediction |
| lociPARSE | - | DL-based local lDDT prediction |
| RNArank | - | DL-based local + global QA |
| RNAdvisor 2.0 | - | Multi-metric platform |
| 3dRNAscore | - | Statistical potential |

### 9.4 RNA Language Models

| Model | URL | Parameters |
|---|---|---|
| RNA-FM | github.com/ml4bio/RNA-FM | - |
| RiNALMo | - | 650M |
| RibonanzaNet2 | ribonanza.stanford.edu | 100M |
| RNA-MSM | github.com/yikunpku/RNA-MSM | - |

### 9.5 Unified Pipeline Tool

**ABCFold** (2025): https://github.com/ -- enables easier running and comparison of
AlphaFold 3, Boltz-1, and Chai-1 from a unified interface.

---

## 10. Outlook and Open Problems

### 10.1 Remaining Challenges

1. **No "AlphaFold moment" for RNA yet**: Unlike proteins, no single method dominates.
   The best RNA predictions at CASP16 are still far below protein prediction accuracy.

2. **Non-canonical interactions**: Current methods struggle with pseudoknots, non-WCF
   base pairs, A-minor interactions, and other tertiary motifs critical for function.

3. **Conformational dynamics**: RNA molecules exist as conformational ensembles, not
   single structures. Predicting the ensemble, not just one conformation, is increasingly
   recognized as the true target.

4. **Long RNAs**: Performance degrades significantly beyond ~200 nucleotides. The
   combination of data scarcity, conformational complexity, and computational cost makes
   long RNA prediction a frontier challenge.

5. **Functional accuracy**: Even when global folds are correct, predictions often fail
   in functionally important regions (binding sites, catalytic centers).

### 10.2 Promising Directions

1. **RNA foundation models**: Larger language models (RiNALMo at 650M params, RibonanzaNet2
   at 100M) trained on tens of millions of sequences are beginning to compensate for
   structural data scarcity.

2. **Hybrid architectures**: RNAPro's combination of foundation models + MSA + templates +
   diffusion represents the most comprehensive approach to date.

3. **Diffusion models**: RNAgrail and the AF3/Protenix diffusion-based structure modules
   show promise for generating diverse, physically valid conformations.

4. **Input optimization**: The finding that secondary structure input quality dramatically
   affects prediction accuracy (trRosettaRNA2 at CASP16) suggests significant gains are
   possible through better preprocessing.

5. **Crowdsourced data**: The Ribonanza/Eterna paradigm of using crowdsourced chemical
   mapping data to train models (RibonanzaNet2) may help address the data scarcity problem.

6. **Open science**: The release of open-source AF3-like models (Protenix, Boltz-1, Chai-1)
   and RNA-specific tools (RNAPro) is accelerating progress by enabling community
   experimentation.

### 10.3 Practical Recommendations (2025-2026)

For a researcher wanting to predict RNA 3D structure today:

1. **Start with template search**: Search Rfam and PDB for homologs. If close templates
   exist, use template-based modeling (Vfold-Pipeline, ModeRNA) as a strong baseline.

2. **Run multiple DL methods**: Generate predictions from trRosettaRNA2, RhoFold+,
   DeepFoldRNA, and AF3/Protenix. Each method may excel on different targets.

3. **Optimize inputs**: Try multiple secondary structure predictions as input. Build
   high-quality MSAs using rMSA.

4. **Score and select**: Use lociPARSE and ARES to rank predictions. Use RNAdvisor 2.0
   for comprehensive evaluation.

5. **Refine best models**: Apply BRiQ refinement to top candidates. Consider short MD
   simulations to assess stability.

6. **For protein-RNA complexes**: Use AF3, Protenix, or Chai-1, which handle multi-chain
   inputs natively.

7. **For short motifs** (<=12 nt): Consider Rosetta stepwise assembly for atomic accuracy.

8. **Validate predictions**: Check for steric clashes, backbone continuity, and base-pair
   geometry. Use experimental data (SHAPE, DMS) as restraints when available.

---

## Key References

### Landmark Papers
- Abramson et al. (2024). "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature.
- Shen et al. (2024). "Accurate RNA 3D structure prediction using a language model-based deep learning approach." Nature Methods. (RhoFold+)
- Wang et al. (2023). "trRosettaRNA: automated prediction of RNA 3D structure with transformer network." Nature Communications.
- Kagaya et al. (2025). "NuFold: end-to-end approach for RNA tertiary structure prediction." Nature Communications.
- Pearce et al. (2023). "Integrating end-to-end learning with deep geometrical potentials for ab initio RNA structure prediction." Nature Communications. (DeepFoldRNA)
- Baek et al. (2024). "Accurate prediction of protein-nucleic acid complexes using RoseTTAFoldNA." Nature Methods.
- Watkins & Das (2020). "FARFAR2: Improved de novo Rosetta prediction of complex global RNA folds." Structure.

### Benchmarking and Assessment
- Bahai et al. (2024). "Systematic benchmarking of deep-learning methods for tertiary RNA structure prediction." PLOS Computational Biology.
- Kretsch et al. (2025). "Assessment of Nucleic Acid Structure Prediction in CASP16." Proteins.
- Bernard et al. (2024). "State-of-the-RNArt: benchmarking current methods for RNA 3D structure prediction." NAR Genomics and Bioinformatics.
- Barciszewski et al. (2024). "Comparative analysis of RNA 3D structure prediction methods." Nucleic Acids Research.
- Fang et al. (2025). "Has AlphaFold3 achieved success for RNA?" Acta Crystallographica D.

### Quality Assessment
- Townshend et al. (2021). "Geometric deep learning of RNA structure." Science. (ARES)
- Baral et al. (2024). "lociPARSE: A Locality-aware Invariant Point Attention Model for Scoring RNA 3D Structures." JCIM.

### Tools and Models
- RNAPro: github.com/NVIDIA-Digital-Bio/RNAPro
- Protenix: github.com/bytedance/Protenix
- RhoFold+: github.com/ml4bio/RhoFold
- NuFold: github.com/kiharalab/NuFold

---

*Survey compiled: February 2026*
*This document reflects the state of the field as of early 2026, incorporating results
from CASP16 (2024), the Stanford RNA 3D Folding Kaggle Competition (2024-2025), and
publications through January 2026.*
