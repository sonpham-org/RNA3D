# RNAPro Fine-Tuning Plan

## Goal
Fine-tune RNAPro (488M params) on competition training data (~5,135 RNA structures) to improve TM-score from 0.361 to 0.45+.

## Hardware
- **Primary**: NVIDIA RTX 4090 (24GB VRAM)
- **Note**: AMD Radeon 8060s (RDNA 4) cannot be used — no confirmed ROCm support, PyTorch cannot mix CUDA + ROCm backends

---

## Phase 1: Confidence-Only Fine-Tuning (Lowest Risk, Fastest)

**What**: Train only the ConfidenceHead (4 pairformer blocks, ~20-30M params) while freezing everything else.

**Why**: The ranking_score is broken for single-chain RNA (iptm=0, disorder=0, so ranking = 0.2*ptm). Better confidence scores = better sample selection from the 5 diffusion samples.

**Settings**:
| Parameter | Value |
|-----------|-------|
| `train_confidence_only` | `True` |
| Trainable params | ~20-30M (ConfidenceHead only) |
| Crop size | 384 tokens |
| Batch size | 1 (gradient accumulation = 4) |
| Learning rate | 3e-4 |
| Epochs | 10-20 |
| Time estimate | 3-6 hrs/epoch |
| Loss | pLDDT + PAE + PDE + resolved (alpha_pae = 1.0) |
| Precision | bf16 mixed precision |

**Training stages** (from RNAPro loss config):
- alpha_diffusion = 0 (frozen)
- alpha_distogram = 0 (frozen)
- alpha_confidence = 1.0
- alpha_pae = 1.0

**Validation**: Rfam family-based split (NOT random — high redundancy within families inflates generalization estimates).

---

## Phase 2: LoRA Fine-Tuning (Medium Risk, Higher Potential)

**What**: Add LoRA adapters (rank 4-8) to Q/K/V attention projections in PairformerStack and DiffusionTransformer.

**Why**: Adapt the structural prediction to RNA-specific features without catastrophic forgetting.

**Settings**:
| Parameter | Value |
|-----------|-------|
| LoRA rank | 4-8 |
| Trainable params | ~2-10M |
| Crop size | 256 tokens |
| Batch size | 1 (gradient accumulation = 4) |
| Learning rate | 1e-4 |
| Epochs | 5-15 |
| Target modules | Q/K/V in PairformerStack + DiffusionTransformer attention |
| Loss | SmoothLDDT + MSE + Distogram (Stage 1 config) |
| Precision | bf16 mixed precision |

**Key considerations**:
- Start from best Phase 1 checkpoint
- Monitor validation loss closely — RNA training set is small (overfitting risk)
- Save checkpoints every epoch for ensemble potential
- Try both `public-best` and `private-best` base checkpoints

---

## Phase 3: Validate and Upload

1. Run inference on held-out validation set
2. Compare TM-scores: base vs Phase 1 vs Phase 2
3. Upload best checkpoint(s) as Kaggle dataset
4. Test in submission notebook with `enable_internet: false`

---

## Data Augmentation Strategy

### Tier 1: Coordinate Noise (SAFE, built-in)
- Add Gaussian noise (sigma=0.5-1.5A) to C1' coordinates
- Already proven effective (RhoFold+ uses MD-derived drifted structures at ~3A RMSD)
- Equivalent to label smoothing, prevents overfitting to crystal conformations
- **Implementation**: Apply random_transform (SE(3) augmentation already in featurizer.py)

### Tier 2: Compensatory Mutations (SAFE, high value)
- Swap Watson-Crick base pairs: G:C <-> A:U, C:G <-> U:A
- C1' positions change by <0.5A — structure is preserved
- Multiplies effective training set by 2-4x for helical regions (~60-70% of nucleotides)
- **Requires**: Secondary structure from RNAfold (partition function, threshold >= 0.95 for ~95% PPV)
- **Critical caveat**: Some base pairs participate in tertiary contacts (e.g., G:U wobble in tetraloop receptors) — high-confidence pairs only

### Tier 3: Loop Mutations (MODERATE risk)
- Mutate unpaired single-stranded regions (loops, linkers)
- Safe for interior of large loops (>6 nt), risky for tetraloops and tertiary contact positions
- Limit to ~40% sequence dissimilarity in loop regions
- **Requires**: Accurate identification of unpaired positions AND tertiary contacts

### Tier 4: Homolog Substitution (HIGH value, requires care)
- Replace training sequences with close Rfam homologs (>60% sequence identity)
- Structural conservation exceeds sequence conservation in RNA
- **Requires**: Rfam alignment data, careful per-family validation

---

## Data Leakage Warnings

Two dangerous methods found in existing RNAPro code — **DO NOT use during fine-tuning**:

1. **`_create_masked_template_features()`** (rna_dataset_allatom.py ~line 940)
   - Uses GROUND TRUTH coordinates as masked templates
   - Would leak answer into input features during training

2. **`augment_with_test_samples()`** (rna_dataset_allatom.py ~line 1195)
   - Adds test sequences to training set with masked coordinates
   - Direct train/test contamination

**Safe to use**: `_create_template_features_ca_precomputed()` with 50% template dropout (already implemented).

---

## Validation Split Strategy

**MUST use Rfam family-based clustering**, not random split:
- RNA families share high structural similarity despite sequence divergence
- Random split inflates generalization estimates due to within-family redundancy
- RNA3DB proposes: 169 families (1,152 seqs) for training, 47 families (493 seqs) for test
- Minimum: ensure no Rfam family appears in both train and validation

---

## Checkpoints to Try

| Checkpoint | Description | Priority |
|-----------|-------------|----------|
| `private-best` | Best on private leaderboard | Primary (current default) |
| `public-best` | Best on public leaderboard | Try for ensemble |
| Phase 1 output | Confidence-only fine-tuned | After Phase 1 |
| Phase 2 output | LoRA fine-tuned | After Phase 2 |

**Ensemble strategy**: Run inference with multiple checkpoints, use confidence scores to select best per target. Free diversity with no training cost.

---

## Secondary Structure Prediction for Augmentation

| Method | PPV | Sensitivity | Speed | Use Case |
|--------|-----|------------|-------|----------|
| RNAfold MFE | ~66% | ~66% | Fast | Too inaccurate for augmentation |
| RNAfold partition (threshold >= 0.95) | ~95% | ~40% | Fast | **Best for augmentation** — high precision, low recall is fine |
| Rfam CM (INFERNAL) | ~99% | ~90%+ | Moderate | Best when family is known |
| EternaFold | ~70-75% | ~70-75% | Fast | Slight improvement over RNAfold |

**Recommendation**: Use RNAfold partition function with base-pair probability threshold >= 0.95. Only ~40% of true pairs are identified, but those identified are ~95% correct. False positive base pairs (incorrectly marking unpaired as paired) are more damaging than missing true pairs (treating paired as unpaired).

---

## Expected Timeline

| Phase | Duration | Expected Impact |
|-------|----------|-----------------|
| Phase 1 (confidence-only) | 2-3 days | +0.01-0.03 (better sample selection) |
| Phase 2 (LoRA) | 3-5 days | +0.02-0.05 (better structure prediction) |
| Phase 3 (validate + upload) | 1 day | Verify gains |
| **Total** | **~1 week** | **+0.03-0.08 expected** |

---

*Plan compiled: February 8, 2026*
*Based on: 6-agent research team findings (data-explorer, ml-researcher-training, ml-researcher-augmentation, bio-statistician, rna-specialist, data-leakage-investigator)*
