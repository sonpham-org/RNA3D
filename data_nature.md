# Nature of RNA 3D Competition Data

## 1. What is RNA 3D Structure Prediction?

### The Physical Problem

RNA (ribonucleic acid) molecules fold from linear chains of nucleotides (A, C, G, U) into complex three-dimensional structures that determine their biological function. Unlike proteins, which fold into relatively stable globular shapes driven by a hydrophobic core, RNA structures are dominated by:

- **Base pairing**: Watson-Crick pairs (A-U, G-C) and wobble pairs (G-U) form the secondary structure (helices, stems)
- **Base stacking**: Aromatic ring stacking with ~3.3A spacing provides thermodynamic stability
- **Tertiary contacts**: Non-canonical base pairs, A-minor interactions, pseudoknots, and metal ion coordination create the 3D fold
- **Backbone flexibility**: RNA has 7 backbone torsion angles per nucleotide (vs. 3 for proteins), giving far more conformational freedom

### What C1' Atoms Represent

Each RNA nucleotide has three chemical groups:
1. **Phosphate group** (P) -- connects nucleotides via the sugar-phosphate backbone
2. **Ribose sugar** (5 carbons: C1' through C5') -- the structural scaffold
3. **Nitrogenous base** (A, C, G, or U) -- encodes sequence information and mediates pairing

The **C1' atom** is the carbon at position 1 of the ribose sugar ring. It connects the sugar to the base (via an N-glycosidic bond to N1 for pyrimidines C/U or N9 for purines A/G). C1' serves as RNA's equivalent of a protein's alpha-carbon (CA) for coarse-grained structure representation because:

- It captures the overall backbone trace and topology
- It reflects both backbone conformation and base positioning
- Adjacent C1'-C1' distances are ~5.9A in A-form helices, ~6.5A in extended regions
- Watson-Crick base pair C1'-C1' distances are ~10.4A

By predicting only C1' coordinates (one 3D point per nucleotide), the competition reduces the full-atom problem (~33 heavy atoms per nucleotide) to a tractable coarse-grained representation that still captures the essential 3D topology.

### TM-score Metric

The competition uses TM-score (Template Modeling score) to evaluate predictions. For nucleic acids, the formula uses C3' atoms internally in US-align, but the competition maps C1' coordinates for scoring.

**TM-score formula:**

```
TM-score = (1/L) * sum_i [ 1 / (1 + (d_i / d_0)^2) ]
```

Where:
- `L` = length of the target structure (number of residues)
- `d_i` = distance between i-th aligned residue pair after optimal superposition
- `d_0` = length-dependent normalization: `d_0 = 0.6 * sqrt(L - 0.5) - 2.5` (for L >= 30)

**d0 values at key lengths:**

| Sequence Length (L) | d0 (Angstroms) | Interpretation |
|---------------------|----------------|----------------|
| 30 | 1.26A | Extremely tight -- need sub-1.3A accuracy per residue |
| 50 | 1.72A | Still very tight |
| 100 | 3.48A | Moderate -- need reasonable local accuracy |
| 118 | 4.01A | Moderate |
| 200 | 5.98A | Forgiving -- rough topology gives partial credit |
| 500 | 8.91A | Very forgiving -- rough shape enough |
| 720 | 9.24A | Large targets: just need correct topology |
| 1000 | 10.44A | Getting global fold roughly right scores well |

**Key implication**: Short targets require near-atomic precision; long targets primarily reward getting the overall fold topology correct. This has major strategic consequences for how prediction effort should be allocated.

**Score thresholds:**
- TM-score < 0.17: random/unrelated structures
- TM-score >= 0.45: structures share the same global topology
- TM-score >= 0.5: generally the same fold
- TM-score = 1.0: perfect structural match

---

## 2. Training Data: ~5,135 RNA Sequences

### Overview

The competition provides a training set of approximately 5,135 RNA sequences with experimentally determined 3D structure labels (C1' atom coordinates). These are derived from structures in the Protein Data Bank (PDB) released before a temporal cutoff date (September 30, 2024, aligned with CASP16).

Additionally, a **validation set** with separate sequences and labels is provided. The Phase 1 TBM notebook merges training and validation data to create a larger template database for search.

### Data Format

**train_sequences.csv columns:**

| Column | Description |
|--------|-------------|
| `target_id` | Unique identifier (e.g., R1107) |
| `sequence` | Full RNA sequence (ACGU only), all chains concatenated |
| `temporal_cutoff` | PDB release date before which the structure was available |
| `description` | Target name, organism, additional info (e.g., ligand SMILES) |
| `all_sequences` | FASTA-formatted individual chain sequences (RNA and protein) |
| `stoichiometry` | Chain composition (e.g., `ChainA:1;ChainB:2`) |

**train_labels.csv columns:**

| Column | Description |
|--------|-------------|
| `ID` | `{target_id}_{residue_number}` (1-indexed) |
| `resname` | Nucleotide identity (A, C, G, or U) |
| `resid` | Residue position (1-indexed) |
| `x_1`, `y_1`, `z_1` | C1' atom coordinates in Angstroms |

Labels provide only RNA C1' coordinates. Even for RNA-protein complexes, only the RNA portion has coordinate labels.

### Length Distribution

Based on the competition data and PDB RNA structure statistics:

| Length Range | Typical Count | Category | Prediction Difficulty |
|-------------|--------------|----------|----------------------|
| 30-50 nt | Many | Short aptamers, riboswitches | High precision needed (d0 ~1.3-1.7A) |
| 50-100 nt | Common | tRNAs, small ribozymes | Moderate |
| 100-200 nt | Common | Riboswitches, small RNAs | Moderate |
| 200-500 nt | Less common | Group I introns, large ribozymes | Template-dependent |
| 500-1000 nt | Rare | rRNA domains, large complexes | RNAPro's 512-token limit applies |
| >1000 nt | Very rare | rRNA, GOLLD/ROOL RNAs | Extremely challenging |

The RNA3DB dataset analysis (from PDB as of Jan 2024) shows:
- 21,005 total RNA chains in PDB
- After length and resolution filtering: 11,176 chains remain
- Clustered at 99% sequence identity: 1,645 distinct RNA sequences
- Median chains per cluster: 2.0 (many are crystallized multiple times)
- 216 distinct Rfam structural families represented

### Multi-Chain vs Single-Chain

RNA structures in the PDB come in several forms:

1. **Single-chain RNA** (RNA only): Self-folding molecules like riboswitches, aptamers, self-cleaving ribozymes
2. **Multi-chain RNA** (RNA-RNA complexes): Ribosomal RNA subunits, multi-strand RNA origami
3. **RNA-protein complexes**: ~36%+ of test targets (from computational biology analysis). Examples include ribosomes, spliceosomes, RNPs, CRISPR-Cas complexes

The `stoichiometry` field encodes chain composition. Examples from test data:
- `ChainA:1` -- single RNA chain
- `ChainA:1;ChainB:2` -- one RNA + two copies of another chain
- Multiple protein chains + one RNA chain (e.g., R1189: RsmZ RNA with 6 RsmA protein molecules)

**Critical insight**: The `all_sequences` FASTA contains BOTH RNA and protein chain sequences. The `sequence` column is the concatenated RNA-only sequence. For RNA-protein complexes, protein chains provide structural context that shapes the RNA fold, but our current pipeline ignores them entirely.

### Coordinate Ranges

C1' atom coordinates are provided in Angstroms, clipped to [-999.999, 9999.999]. Typical structures occupy a volume of:
- Small RNAs (30-100 nt): ~30-60A diameter
- Medium RNAs (100-500 nt): ~60-150A diameter
- Large complexes (>500 nt): ~100-300A+ diameter

---

## 3. Test Data: ~108-119 Targets

### Part 1 Test Set (Completed Sep 2025)

The Part 1 test set was designed with 20 CASP16 RNA targets on the initial public leaderboard, refined to 8 after a data refresh, with 20 targets on the private leaderboard (10 RNA-Puzzles targets + 10 new PDB structures).

**Key Part 1 finding**: 19 of 20 targets (95%) had potential templates with TM-align > 0.45 in the PDB. This is dramatically higher than CASP15 (25% of RNA targets had templates) and CASP16 general (56% templated). The high template availability explains why template-based methods dominated Part 1.

### Part 2 Test Set (Current Competition)

Part 2 explicitly introduces **harder targets** including:
- Template-free RNA structures (no close homologs in PDB)
- Longer sequences
- More complex multi-chain assemblies
- Alternative conformations

The test set contains approximately 108-119 targets spanning:

| Category | Estimated Fraction | Approach Needed |
|----------|--------------------|-----------------|
| Template-rich (good homologs) | ~50-60% | TBM or RNAPro with templates |
| Template-poor (distant homologs) | ~20-30% | RNAPro de novo + MSA |
| Template-free (novel folds) | ~10-20% | Pure de novo methods |
| RNA-protein complexes | ~36%+ | Co-folding with protein context |

### Example Test Targets (from RNAPro examples)

From the test_sequences.csv provided with RNAPro:

| Target | Length | Type | Description |
|--------|--------|------|-------------|
| R1107 | 68 nt | RNA-protein | CPEB3 ribozyme (Human) + U1 snRNP A protein |
| R1108 | 69 nt | RNA-protein | CPEB3 ribozyme (Chimpanzee) + protein |
| R1116 | 151 nt | Single RNA | Poliovirus cloverleaf RNA with tRNA scaffold |
| R1117v2 | 29 nt | Riboswitch + ligand | PreQ1 class I type III riboswitch (with SMILES) |
| R1126 | 363 nt | Single RNA | RNA origami traptamer |
| R1128 | 228 nt | Single RNA | Paranemic crossover RNA triangle |
| R1136 | 363 nt | Aptamer + ligands | Broccoli-pepper aptamer FRET tile (3 ligands) |
| R1138 | 720 nt | Single RNA | 6-helix bundle RNA with clasp (co-transcriptional) |
| R1149 | 120 nt | Single RNA | SARS-CoV-2 SL5 (alternative conformations) |
| R1156 | 133 nt | Single RNA | BtCoV-HKU5 SL5 |
| R1189 | 118 nt | RNA + 6 proteins | RsmZ RNA with 3 RsmA protein dimers |
| R1190 | 118 nt | RNA + 4 proteins | RsmZ RNA with 2 RsmA protein dimers |

**Notable patterns:**
- Several targets are from CASP16 (released 2022-2024 PDB structures)
- Ligand-bound targets include SMILES strings in descriptions (riboswitches, aptamers)
- Alternative conformation targets test whether models can predict multiple states
- RNA-protein targets have protein sequences in `all_sequences` but only RNA in `sequence`

---

## 4. Data Characteristics

### Sequence Composition (ACGU)

RNA sequences contain only four nucleotides:
- **A (Adenine)** -- purine, pairs with U
- **C (Cytosine)** -- pyrimidine, pairs with G
- **G (Guanine)** -- purine, pairs with C
- **U (Uracil)** -- pyrimidine, pairs with A

Typical composition in structured RNAs:
- G+C content is generally higher in stable structures (~50-70% GC)
- Purine-pyrimidine ratio is roughly balanced due to Watson-Crick pairing constraints
- tRNAs: high G+C content (~55-65%)
- Riboswitches: variable, sometimes A/U-rich in single-stranded regions
- rRNA: moderate G+C (~50-55%)

The 4-letter alphabet (vs. 20 amino acids in proteins) means sequence-based homology detection is harder -- random sequences share 25% identity by chance, compared to 5% for proteins. This makes template discovery from sequence alone more challenging.

### Chain Stoichiometry Patterns

From the training and test data, stoichiometry patterns include:

| Pattern | Example | Meaning |
|---------|---------|---------|
| `ChainA:1` | Simple RNA | Single RNA molecule |
| `ChainA:1;ChainB:1` | Heterodimer | Two different chains |
| `ChainA:1;ChainB:2` | 1+2 complex | One chain + two copies of another |
| `ChainA:6;ChainB:1` | Many copies | Six copies of protein + one RNA |
| Complex patterns | Large assemblies | Multiple protein + RNA chains |

The stoichiometry field is critical for:
1. Knowing how many independent chains to predict
2. Identifying symmetric chains (same sequence, multiple copies)
3. Understanding the biological assembly context

### Backbone Geometry Constraints

RNA backbone has characteristic distances that can be used as geometric constraints:

| Measurement | Distance | Context |
|------------|----------|---------|
| C1'-C1' adjacent (i, i+1) | ~5.9A | A-form helix |
| C1'-C1' adjacent (i, i+1) | ~6.5A | Extended/loop regions |
| C1'-C1' (i, i+2) | ~10.2A | Helical rise |
| W-C base pair C1'-C1' | ~10.4A | Across the helix |
| Base stacking distance | ~3.3-3.4A | Between stacked bases |
| P-P adjacent | ~5.8-6.2A | Phosphate backbone |

---

## 5. Physical Determinants of RNA 3D Structure (Augmentation Guide)

Understanding what physically determines each nucleotide's 3D position is essential for designing data augmentation strategies. This section answers: what can we safely perturb during training, and what will produce unrealistic structures?

### 5.1 What Determines the 3D Position of Each Nucleotide?

RNA 3D structure is determined by a hierarchy of forces, roughly ordered by contribution:

**Level 1: Secondary Structure (~60-70% of structural determination)**

Secondary structure -- the pattern of Watson-Crick and wobble base pairs -- is the dominant determinant of RNA 3D shape. Approximately 66% of nucleotides in structured RNAs form regular helical (double-stranded) regions. These helices adopt a predictable A-form geometry with:
- ~2.8A rise per base pair along the helix axis
- ~32.7 degrees twist per base pair
- ~11 base pairs per full helical turn
- C1'-C1' distance across a W-C pair: ~10.4A
- Helix diameter: ~20A (major groove: ~11A, minor groove: ~4A)

The A-form helix is so geometrically regular that knowing the base-pairing pattern alone constrains ~66% of all C1' positions to within ~1-2A of their true location. This is why secondary structure prediction is often called the "solved" part of the problem.

**Level 2: Junction Topology and Coaxial Stacking (~15-20%)**

Where multiple helices meet (junctions, multi-way junctions, pseudoknots), their relative orientations are determined by:
- **Coaxial stacking**: Adjacent helices stack end-to-end, creating longer continuous helical stacks. This is the dominant organizing principle for 3D architecture. At junctions, which helices stack on which is sequence-dependent and critically determines the overall shape.
- **Junction conformation**: The number of helices and the lengths of single-stranded linkers between them constrain possible geometries. A 3-way junction (like in tRNA) has fewer possible arrangements than a 4-way or 5-way junction.
- **Topological constraints**: Secondary structure alone restricts the conformational space by over an order of magnitude (10x+). The connectivity of the chain through paired and unpaired regions acts as a powerful constraint on which 3D arrangements are physically possible.

**Level 3: Tertiary Interactions (~10-15%)**

Sequence-specific long-range contacts lock the 3D fold into its final form:
- **A-minor interactions**: The most common RNA tertiary motif. Adenines dock their minor-groove face into the minor groove of a receptor helix. These are highly sequence-specific -- mutating the adenine or the receptor base pair destroys the contact.
- **Tetraloop-receptor interactions**: GNRA tetraloops (especially GAAA) dock into 11-nt receptor motifs. Extremely specific: the G:U wobble closing pair of the receptor is critical even though it is 7A from the contact site. A G:U to G:C mutation alone destabilizes the interaction by +4.33 kcal/mol.
- **Ribose zippers**: 2'-OH hydrogen bond networks between adjacent strands, not strongly sequence-specific.
- **Base triples and triple helices**: Specific nucleotides in loop regions form hydrogen bonds with existing base pairs.
- **Kissing loops**: Two hairpin loops form Watson-Crick pairs between their loop nucleotides.
- **Pseudoknots**: Loop nucleotides base-pair with a distant region, creating a knot-like topology.

**Level 4: Metal Ions and Solvent (~5-10%)**

- **Mg2+ ions**: Essential for RNA tertiary folding. Mg2+ is the most abundant divalent cation in cells and the most frequently identified metal in RNA structures. It stabilizes tertiary structure through both diffuse electrostatic screening and site-specific binding. "Y-clamp" and "Mg2+ clamp" motifs anchor distant RNA strands together (analogous to disulfide bridges in proteins). 814 magnesium clamps and 238 Y-clamps have been identified across known RNA structures. Without sufficient Mg2+, RNA folds into partially folded intermediates rather than the native structure.
- **Monovalent ions (K+, Na+)**: Provide charge screening of the phosphate backbone. Required for secondary structure stability.
- **Water**: Structured water molecules bridge hydrogen bonds in the minor groove and at tertiary contacts.

**Level 5: Protein Binding and External Context (~variable, 0-30%)**

- **Protein-induced folding**: Many RNAs only adopt their final 3D structure upon binding protein partners. In ribosome assembly, protein binding induces long-range RNA conformational changes -- secondary assembly proteins can reshape RNA structure at sites distant from their binding sites. For RNA-protein complexes in our test data (~36% of targets), ignoring the protein context means ignoring a major structural determinant.
- **Co-transcriptional folding**: Some RNAs fold differently depending on the kinetics of transcription. The competition target R1138 ("Young conformer of 6-helix bundle") is explicitly described as a co-transcriptional product.
- **Ligand binding**: Riboswitches change conformation upon ligand binding. Several test targets (R1117v2, R1136) have bound ligands specified by SMILES.

### 5.2 What Sequence Changes PRESERVE 3D Structure?

These mutations are "safe" for data augmentation -- they produce realistic sequence-structure pairs:

**Safe: Compensatory mutations in Watson-Crick base pairs**

If a G-C pair is mutated to an A-U pair (or C-G, U-A, or G-U wobble), the helix geometry is preserved. The C1' positions change by <0.5A. This is the most well-validated augmentation strategy, extensively confirmed by evolutionary analysis (covariation).

Rules for compensatory mutations:
- G:C <-> C:G (swap strand): Preserves geometry almost exactly
- A:U <-> U:A (swap strand): Preserves geometry almost exactly
- G:C <-> A:U: Slightly different geometry (A:U has 2 H-bonds vs 3 for G:C, helix may be slightly less stable), but C1' positions change <1A
- G:U wobble <-> A:C or U:G: Wobble pairs have slightly shifted geometry (~0.5-1A displacement), but are structurally tolerated in most positions
- Any W-C pair <-> any W-C pair: Generally safe if both positions are mutated simultaneously

**Critical caveat:** Some base pairs participate in tertiary contacts. A G:U wobble that serves as a tetraloop receptor recognition element CANNOT be changed to G:C without destroying the tertiary contact (even though the helix is fine).

**Safe: Mutations in unpaired single-stranded regions (with caveats)**

Unpaired loop and linker nucleotides evolve much faster than paired nucleotides. In eukaryotic rRNA, loops evolve at ~2-3x the rate of stems. General rules:
- Interior of large loops (>6 nt): Most positions tolerate any mutation
- Hairpin loop positions not involved in tertiary contacts: Tolerable
- Single-stranded linkers between helices: Tolerable (but length matters)

**Caveats for loop mutations:**
- Tetraloop sequences (GNRA, UNCG, CUUG) are highly conserved because they form specific structures. Mutating GAAA to GCAA changes loop geometry.
- Nucleotides that form tertiary contacts (A-minor adenines, base triples) are NOT safely mutable even if they appear "unpaired" in secondary structure.
- Up to ~40% sequence dissimilarity in loop regions, structure is generally preserved. Beyond ~40%, structural divergence occurs.

**Safe: Sequence-independent backbone modifications**

- Coordinate noise injection (~0.5-1.5A Gaussian noise on C1' positions): Simulates crystal packing effects and molecular dynamics fluctuations. Structures with ~3A RMSD from the PDB structure are physically realistic (equivalent to MD-derived conformations).
- Small rigid-body rotations of helical segments (~5-10 degrees): Simulates natural flexibility
- Thermal fluctuations of unpaired regions (~1-3A displacement): Loops are inherently flexible

### 5.3 What Sequence Changes DESTROY 3D Structure?

These mutations produce unrealistic training examples and should be avoided:

**Destructive: Breaking Watson-Crick pairs without compensation**

Mutating one strand of a base pair without the other (e.g., G:C to A:C) creates a mismatch that disrupts the helix. Effects:
- Single mismatch in a long helix (>10 bp): Tolerable, causes local distortion (~2-3A displacement at mismatch, ~1A for neighbors)
- Multiple adjacent mismatches: Helix melts locally, creating a bulge or internal loop. C1' positions shift 3-10A.
- Mismatch near helix end: Helix may unravel ("fraying"), losing terminal base pairs
- More than ~30% mismatches in a helix: Complete helix destruction

**Destructive: Mutating tertiary contact nucleotides**

- A-minor adenines: These MUST be adenine. A-to-G/C/U mutation eliminates the minor groove docking interaction. Effect: loss of tertiary contact, potentially unfolding a domain.
- Tetraloop-receptor recognition elements: The GNRA tetraloop and its 11-nt receptor are exquisitely specific. Even the G:U wobble closing pair of the receptor (7A from the contact!) cannot be changed to G:C.
- Kissing loop complementary bases: Must remain Watson-Crick complementary.

**Destructive: Changing sequence length in structured regions**

- Insertions/deletions in helices: Shift the register, breaking all downstream base pairs (catastrophic)
- Insertions/deletions in junction linkers: Change junction geometry, altering helix orientations (moderately destructive)
- Adding nucleotides to tetraloops: A tetraloop MUST be exactly 4 nucleotides

**Destructive: Removing metal-binding sites**

If a specific Mg2+ binding site involves non-bridging phosphate oxygens from specific nucleotides, mutating those positions may not change the base but the backbone dynamics change. However, since we only predict C1' positions and metals are not explicitly modeled, this is less relevant for our competition.

### 5.4 How Much Structure is Determined by Each Level?

| Structural Level | % of C1' positions determined | Predictable from? | Augmentation safety |
|-----------------|------------------------------|-------------------|---------------------|
| Secondary structure (helices) | ~60-70% | Sequence (RNAfold, covariation) | Compensatory mutations safe |
| Junction topology | ~15-20% | Secondary structure + sequence | Linker length changes risky |
| Tertiary contacts | ~10-15% | MSA covariation, templates | Most mutations destructive |
| Metal ions / solvent | ~5-10% | Known binding motifs | Not directly augmentable |
| Protein context | 0-30% (if RNA-protein) | Protein sequence, co-folding | Must include protein for complexes |

**Key insight for augmentation:** The safest augmentations are:
1. Compensatory base pair mutations in helices (affects ~60-70% of nucleotides, well-understood rules)
2. Conservative mutations in large unpaired loops (affects ~15-20% of nucleotides, but must avoid tertiary contact positions)
3. Coordinate noise injection (~0.5-3A, simulates physical flexibility)
4. Rigid-body perturbation of helical segments (simulates junction flexibility)

The riskiest augmentations are:
1. Any mutation at positions involved in tertiary contacts (often not obvious from sequence alone)
2. Insertions/deletions anywhere in the structure
3. Breaking base pairs without compensation
4. Removing protein context from RNA-protein complexes

### 5.5 Practical Augmentation Implications for RNAPro Fine-Tuning

Given that RNAPro is trained with 256-token crops and uses diffusion-based structure generation, the most promising augmentation strategies are:

1. **Compensatory mutation augmentation (HIGH value, SAFE):** For each training structure, generate variants by swapping W-C pairs (G:C <-> A:U, C:G <-> U:A) while keeping coordinates identical. This multiplies the effective training set by 2-4x for helical regions. Requires knowing the secondary structure (available from RNAfold or Rfam annotations).

2. **Coordinate noise augmentation (MODERATE value, SAFE):** Add Gaussian noise (sigma=0.5-1.5A) to C1' coordinates. This is equivalent to label smoothing and prevents overfitting to specific crystal conformations. Already proven effective: MD-derived "drifted structures" with ~3A RMSD are used as augmentation in RhoFold+ training.

3. **Homolog substitution (HIGH value, requires care):** Replace training sequences with close homologs from Rfam alignments (>60% sequence identity) while keeping the same 3D coordinates. This exploits the fact that structural conservation exceeds sequence conservation.

4. **Cropping augmentation (MODERATE value, BUILT-IN):** RNAPro already uses random 256-token crops during training. Varying the crop boundaries provides some augmentation, though it doesn't increase sequence diversity.

5. **Secondary structure-aware mutation (HIGH value, MODERATE risk):** Predict secondary structure, then mutate unpaired positions randomly while applying compensatory mutations to paired positions. This requires accurate secondary structure prediction and careful avoidance of tertiary contact positions.

---

## 6. Template Coverage

### The Critical Role of Templates

Template-based modeling (TBM) is by far the most impactful approach for RNA 3D structure prediction. This is because:

1. **Structural conservation exceeds sequence conservation** in RNA. Two RNAs with only 40% sequence identity may share the same 3D fold (same Rfam family).

2. **The structural universe is small.** RNA3DB identifies only 216 distinct Rfam structural families across all 21,005 PDB RNA chains. Many "novel" test targets actually belong to known families.

3. **Competition validation**: In Part 1, 95% of targets had usable templates (TM-align > 0.45). The 1st-place winner (john) scored 0.591 using TBM alone, surpassing deep learning approaches.

### Template Sources

| Source | Size | Coverage | Notes |
|--------|------|----------|-------|
| Competition training data | ~5,135 sequences | Primary source | Curated, C1' labels provided |
| Competition validation data | Additional sequences | Supplement | Merged with training for search |
| PDB (pre Sep 2024 cutoff) | ~5,000+ RNA structures | Comprehensive | External data allowed |
| RNA3DB clusters | 1,645 distinct sequences | Non-redundant | 99% identity clusters |
| Rfam families | 216 structural families | Family-level | Covariance model search |

### Template Quality Gap

The gap between our TBM (0.359) and john's TBM (0.591) comes from:

1. **Search method**: We use naive global pairwise alignment (BioPython). John likely uses PDB-wide search with structural alignment tools.
2. **Template database**: We only search competition training data. John searches the full PDB.
3. **Alignment quality**: We use fixed scoring parameters. Better local/profile alignment would help.
4. **Coordinate transfer**: We use linear interpolation for gaps. Better gap modeling (loop fragments) would improve quality.
5. **Multi-chain handling**: We concatenate all chains. Chain-level alignment preserves structural context.

### Template Availability for Part 2

Part 2 explicitly includes template-free targets, but many targets will still have templates:

| Template Quality | Fraction (estimated) | Best Approach |
|-----------------|---------------------|---------------|
| High (alignment > 0.7) | ~40-50% | Direct TBM, high confidence |
| Medium (alignment 0.4-0.7) | ~20-30% | RNAPro with template + de novo backup |
| Low (alignment 0.2-0.4) | ~15-20% | De novo primary, template as weak guide |
| None (alignment < 0.2) | ~10-15% | Pure de novo (RNAPro, DRFold2, RhoFold+) |

---

## 7. MSA Data

### What MSAs Provide

Multiple Sequence Alignments (MSAs) contain evolutionary information critical for structure prediction. For each target RNA, an MSA collects homologous sequences from databases and aligns them. Co-varying positions (compensatory mutations) reveal base-pairing and structural contacts.

### Available MSA Data

The competition provides precomputed MSA files (FASTA format) per target in an MSA directory. These are generated using RNA-specific search tools:

| Tool | Database | Method |
|------|----------|--------|
| nhmmer | Rfam + nt/nr | Profile HMM search |
| INFERNAL | Rfam CMs | Covariance model search (structure-aware) |
| rMSA pipeline | Multiple databases | Iterative search (nhmmer + BLASTN) |

### MSA Quality Variation

MSA quality varies dramatically across RNA families:

| MSA Depth (Neff) | Quality | Implication |
|-------------------|---------|-------------|
| >1000 | Excellent | Strong co-evolution signal, reliable contact prediction |
| 100-1000 | Good | Useful for structure prediction |
| 10-100 | Moderate | Some signal, but noisy |
| <10 | Poor | Essentially single-sequence prediction |
| 1 (no homologs) | None | Must rely entirely on model/templates |

**Key statistics** (from rMSA pipeline analysis):
- Median MSA sequences per Rfam family: ~2,184
- rMSA achieves 20% higher F1-score for secondary structure prediction vs. baseline tools
- Many small, synthetic, or novel RNAs have very shallow MSAs
- Ribosomal RNA families have extremely deep MSAs (millions of sequences)
- Viral RNAs and synthetic constructs often have sparse MSAs

### How RNAPro Uses MSAs

RNAPro's MSA module (4 blocks) processes up to 16,384 MSA sequences:
1. OuterProductMean: converts MSA row features to pairwise features
2. MSA attention: captures coevolution patterns
3. PairStack: integrates MSA information with pair representation

When MSAs are unavailable or shallow, this module provides little information, and the model relies more on templates and RibonanzaNet2 embeddings.

---

## 8. Key Statistics and Data Insights

### RNA Structure Prediction in Context

| Metric | Proteins | RNA |
|--------|----------|-----|
| Known 3D structures in PDB | ~200,000+ | ~5,000-16,000 (nucleic-acid containing) |
| Distinct sequences (non-redundant) | ~100,000+ | ~1,645 (RNA3DB 99% identity) |
| Structural families | ~1,500 SCOP/CATH folds | ~216 Rfam families |
| Alphabet size | 20 amino acids | 4 nucleotides |
| Random sequence identity | 5% | 25% |
| Backbone torsions per residue | 3 (phi, psi, omega) | 7 (alpha through zeta + chi) |
| Typical MSA depth | 1,000-100,000+ | 10-10,000 (much sparser) |
| State of the art (best TM-score) | >0.9 (AlphaFold2/3) | ~0.5-0.65 (competition level) |

### Competition-Specific Statistics

| Metric | Value |
|--------|-------|
| Training sequences | ~5,135 |
| Validation sequences | Additional (merged for template search) |
| Test targets | ~108-119 |
| Sequence length range | ~29 to >720+ nt |
| Predictions per target | 5 (best-of-5 scored) |
| Scoring metric | TM-score (0-1) |
| Part 1 winner score | 0.577 (john, TBM + DRFold2) |
| Part 1 TBM-only score | 0.591 (john) |
| RNAPro retrospective Part 1 | 0.640-0.648 |
| Agentic tree search (3 models) | 0.635 |
| Our current score | 0.361 |
| Part 1 template coverage | 95% (19/20 targets had templates) |

### CASP16 RNA Prediction Statistics

| Metric | Value |
|--------|-------|
| Total RNA targets assessed | 42 |
| Participating groups | 65 from 46 labs |
| Targets with templates (CASP16) | 56% (19/34) |
| Max TM-score on novel targets | <0.8 |
| Best automated server | trRosettaRNA2 |
| Best human expert groups | Vfold, GuangzhouRNA-human, KiharaLab |
| Template-free success story | OLE RNA (one exception) |

### RNA3DB Database Summary

| Metric | Value |
|--------|-------|
| Total RNA chains in PDB (Jan 2024) | 21,005 |
| After length/resolution filtering | 11,176 |
| Sequence clusters (99% identity) | 1,645 |
| Rfam structural families | 216 |
| Proposed training split | 1,152 seqs (169 families, 9,832 structures) |
| Proposed test split | 493 seqs (47 families, 1,344 structures) |

### Implications for Fine-Tuning and Data Augmentation

1. **Data scarcity is the primary challenge.** With only ~5,135 training sequences (vs. ~200K+ protein structures), RNA structure prediction is fundamentally data-limited.

2. **High redundancy within families.** Many training examples belong to the same Rfam family and share similar structures. Effective training/validation splits must account for structural similarity (as RNA3DB does).

3. **Sequence identity is misleading for RNA.** Two RNA sequences with 50% identity may share identical structures, while two with 90% identity may fold differently due to key mutations. Structural clustering (RNA3DB approach) is essential.

4. **Short sequences dominate.** Many training/test targets are <200 nt. For these, d0 is small, and prediction accuracy must be near-atomic.

5. **Template availability dominates performance.** For targets with good templates, TBM approaches score dramatically better than de novo. The gap between template-available and template-free prediction remains large.

6. **Protein context matters for complexes.** ~36%+ of targets are RNA-protein complexes. The protein chains shape the RNA fold and ignoring them loses critical structural context. RNAPro (AF3-based) natively supports multi-molecule inputs.

7. **Ligands provide functional context.** Some targets (riboswitches) have bound ligands specified via SMILES. Including ligand information could improve predictions for these targets.

8. **Alternative conformations exist.** Some targets (e.g., R1138, R1149) have known alternative conformations. The 5-prediction-slot format was designed to capture this diversity.

---

## 9. Data Flow in the Competition Pipeline

```
test_sequences.csv
    |
    |-- target_id, sequence, description, all_sequences, stoichiometry
    |
    v
Template Search (TBM)
    |-- Search train_sequences.csv + validation_sequences.csv + PDB
    |-- Align test sequence to training sequences
    |-- Transfer coordinates from best-matching templates
    |-- Generate 5 template candidates per target
    |
    v
RNAPro Inference (or other model)
    |-- Input: sequence + templates (optional) + MSA (optional)
    |-- Template embedder: convert template C1' coords to distance features
    |-- MSA module: extract coevolution signals
    |-- RibonanzaNet2: RNA-specific embeddings (if enabled)
    |-- PairformerStack (48 blocks, N_cycle recycling passes)
    |-- DiffusionModule (24 blocks, N_step denoising steps, N_sample samples)
    |-- ConfidenceHead: ranking_score for sample selection
    |
    v
submission.csv
    |-- One row per residue: ID, resname, resid, x_1..x_5, y_1..y_5, z_1..z_5
    |-- 5 predictions per target (best-of-5 scored by TM-score)
```

---

*Document compiled: February 8, 2026*
*Sources: Competition data, PROBLEM.md, RESEARCH_PLAN.md, RNAPro source code, Phase 1 TBM notebook, RNA3DB paper, CASP16 assessment, PDB statistics*
