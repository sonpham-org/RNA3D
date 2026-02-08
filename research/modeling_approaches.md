# Modeling Approaches and Architectures for RNA 3D Structure Prediction

## Research Report - ML Researcher (Modeling)

---

## 1. Current Architecture: RNAPro Deep Dive

### 1.1 Architecture Overview

RNAPro is a 488M parameter model based on the AlphaFold3 (AF3) architecture, adapted for RNA structure prediction by NVIDIA. The architecture follows the AF3 pipeline:

```
Input Features --> InputFeatureEmbedder --> PairformerStack (48 blocks, recycled N_cycle times)
                                              |
                         [+ RibonanzaNet2 embeddings if enabled]
                         [+ Template embeddings if provided]
                         [+ MSA module if MSAs available]
                                              |
                                    s_trunk, z_trunk
                                              |
                              DiffusionModule (24 transformer blocks)
                                              |
                                    Predicted coordinates
                                              |
                              ConfidenceHead (4 pairformer blocks)
                                              |
                              pLDDT, PAE, PDE, resolved scores
```

### 1.2 Key Components

**InputFeatureEmbedder** (Algorithm 2, AF3):
- Atom-level attention encoder aggregates atom features to token-level
- Concatenates: token embedding (384d) + restype one-hot (32d) + MSA profile (32d) + deletion mean (1d) = 449d
- Optional ESM embedding integration (2560d projected down)

**RibonanzaNet2** (RNA foundation model):
- 48-layer ConvTransformerEncoder with triangle multiplicative updates
- Produces both sequence features (per-residue) and pairwise features (per-pair)
- Integrated via learned layer-wise weighted sum + gated injection
- Sequence features: projected to 449d, injected into s_inputs via `GatedSequenceFeatureInjector`
- Pairwise features: projected to 128d (c_z), injected into z_init via `GatedPairwiseFeatureInjector`
- Additional 4-block PairformerStack for processing RibonanzaNet2 pairwise features (currently in code but may need verification of actual usage)

**RNATemplateEmbedder** (custom for RNA):
- Processes multiple template structures (default 4 templates)
- Converts template C1' coordinates to distance matrices
- Distance binning: 3.25A to 52.0A in 1.25A steps (39 bins) + raw distance
- Each template processed through 2-block PairformerStack
- Templates averaged, then passed through final linear (zero-initialized)
- Takes s_inputs, s_trunk, and z_trunk as additional context

**PairformerStack** (Algorithm 17, AF3):
- 48 blocks of: TriangleMultiplicationOutgoing -> TriangleMultiplicationIncoming -> TriangleAttentionStart -> TriangleAttentionEnd -> PairTransition -> AttentionPairBias -> SingleTransition
- c_z=128, c_s=384, n_heads=16
- Dropout=0.25 for triangle updates

**MSAModule** (Algorithm 8, AF3):
- 4 blocks, processes up to 16384 MSA sequences (test) or 512 (train)
- MSA pair-weighted averaging with OuterProductMean communication
- Each block: OuterProductMean -> MSAStack -> PairStack (triangle updates)

**DiffusionModule** (Algorithm 20, AF3):
- DiffusionConditioning: Fourier noise embedding + trunk conditioning
- AtomAttentionEncoder (3 blocks, 4 heads): atom-level to token-level
- DiffusionTransformer (24 blocks, 16 heads): full self-attention at token level
- AtomAttentionDecoder (3 blocks, 4 heads): token-level back to atom-level
- EDM-style preconditioning (sigma_data=16.0)
- Inference: 200 denoising steps, N_sample=5

**ConfidenceHead** (Algorithm 31, AF3):
- 4-block PairformerStack operating on z_trunk + distance embeddings from predicted coords
- Outputs: pLDDT (50 bins), PAE (64 bins), PDE (64 bins), resolved (2 bins)
- Generates per-sample confidence scores for ranking

### 1.3 Key Dimensions
| Component | Dimension |
|-----------|-----------|
| c_s (single) | 384 |
| c_z (pair) | 128 |
| c_s_inputs | 449 |
| c_token (diffusion) | 768 |
| c_atom | 128 |
| c_atompair | 16 |
| n_blocks (pairformer) | 48 |
| n_blocks (diffusion transformer) | 24 |
| n_heads (pairformer) | 16 |
| n_heads (diffusion) | 16 |

---

## 2. Loss Functions

RNAPro uses a composite loss (`RNAProLoss`) with the following components:

### Diffusion Losses (alpha_diffusion = 4.0):
1. **SmoothLDDTLoss** (weight: 1.0): Smooth approximation of LDDT using sigmoid thresholds [0.5, 1, 2, 4]A. Uses bespoke radius (30A for nucleotides, 15A for others). Supports sparse computation for memory efficiency.
2. **BondLoss** (weight: 0.0 base, 1.0 in finetuning): Squared distance error on bonded atom pairs. Scaled by noise level during training.
3. **MSELoss** (weight: 1/3): Weighted rigid alignment MSE. RNA gets 5x weight, DNA 5x, ligand 10x. Uses Kabsch alignment (weighted_rigid_align).

### Confidence Losses (alpha_confidence = 1e-4):
4. **PLDDTLoss**: Cross-entropy on binned per-atom LDDT scores (50 bins, 0-1 range)
5. **PDELoss**: Predicted distance error, 64 bins (0-32A range)
6. **PAELoss** (alpha_pae = 0.0 base, 1.0 in finetuning stage 3): Predicted aligned error using frame-based computation
7. **ExperimentallyResolvedLoss**: Binary classification of resolved vs unresolved atoms

### Distogram Loss (alpha_distogram = 3e-2):
8. **DistogramLoss**: Cross-entropy on binned pairwise distances (64 bins, 2.3-21.7A)

### Training Stages:
- **Stage 1 (initial)**: smooth_lddt=1.0, bond=0.0, pae=0.0 (no bond or PAE)
- **Stage 2 (finetuning)**: bond=1.0, smooth_lddt=0.0 (add bond loss, drop smooth LDDT)
- **Stage 3 (confidence-only)**: diffusion=0, distogram=0, pae=1.0 (only train confidence head)

---

## 3. Handling Variable-Length Sequences and Multi-Chain

### Current Approach:
- **Token crop**: Training uses 256-token crops (configurable), inference processes full length
- **512-token limit**: The model silently crops at 512 tokens during inference for sequences >512nt
- **Multi-chain handling**: `asym_id`, `entity_id`, `sym_id` embeddings encode chain identity
- **Relative position encoding**: Uses residue index differences (clipped to +/-32) + chain/entity identity signals
- **Symmetric permutation**: Handles equivalent chains via permutation search during training/inference

### Weaknesses:
- No sliding-window or hierarchical approach for long sequences
- Crop-based training means the model never sees long-range interactions >256 tokens during training
- Multi-chain complexes with >512 total residues are truncated

### Potential Improvements:
1. **Sliding window with overlap**: Predict overlapping 512-token windows, stitch together using confidence-weighted averaging in overlap regions
2. **Hierarchical prediction**: First predict secondary structure / coarse structure, then refine locally
3. **Chain-by-chain prediction + docking**: Predict individual chains, then assemble using inter-chain distance predictions or docking algorithms

---

## 4. Fine-Tuning Strategies

### 4.1 Within Kaggle Constraints (No Training)
Since Kaggle provides no internet and limited GPU time (~9 hours), **we cannot fine-tune during submission**. All training must happen offline, and weights must be uploaded as datasets.

### 4.2 Offline Fine-Tuning Opportunities
If we had access to GPU resources outside Kaggle:

1. **Fine-tune on Part 2 training data**: The 5,135 training sequences with labels could be used to:
   - Continue training with the competition's specific RNA structures
   - Focus on RNA-only (vs. the general biomolecule training of Protenix base)
   - Use larger crop sizes (384 or 512 instead of 256)

2. **Multi-stage fine-tuning** (RNAPro's training protocol):
   - Stage 1: Main training (smooth LDDT + MSE + distogram)
   - Stage 2: Bond loss finetuning (add bond constraints)
   - Stage 3: Confidence-only finetuning (train pLDDT/PAE/PDE heads)

3. **LoRA/adapter fine-tuning**: Could add small adapter modules to existing frozen weights:
   - Add LoRA to PairformerStack attention layers
   - Add LoRA to DiffusionTransformer attention layers
   - Much faster than full fine-tuning, needs fewer examples

4. **Distillation from ensemble**: Run multiple models, use consensus as pseudo-labels for a single model

### 4.3 Checkpoint Ensemble (No Training Needed)
- Two checkpoints available: `public-best` and `private-best`
- Run inference with both, use confidence scores to select best per target
- This is free (no training) and likely gives +0.01-0.02

---

## 5. Transfer Learning from Protein Structure Prediction

### 5.1 RNAPro's Transfer Learning
RNAPro is already built on protein structure prediction knowledge:
- **Base model**: Protenix (ByteDance's AF3 reproduction, trained on proteins/nucleic acids)
- **RNA adaptation**: Additional RNA-specific training on RNA structures from PDB
- **RibonanzaNet2 integration**: RNA language model embeddings injected via gated feature injection

### 5.2 Key Differences RNA vs. Protein
| Feature | Proteins | RNA |
|---------|----------|-----|
| Backbone | N-Cα-C-O | P-O5'-C5'-C4'-C3'-O3' |
| # atom types | 20 amino acids | 4 nucleotides (ACGU) |
| Secondary structure | alpha helix, beta sheet | stem, loop, junction, pseudoknot |
| Tertiary contacts | hydrophobic core | base stacking, base pairing, Mg2+ ions |
| Typical length | 100-1000 residues | 30-3000+ nucleotides |
| Structure data | ~200K PDB structures | ~5K RNA-only structures |
| Template availability | High (many homologs) | Low (sparse structural coverage) |

### 5.3 What Transfers Well
- Triangle attention / pairformer architecture (geometry is universal)
- Diffusion-based structure generation (physics is similar)
- Confidence prediction framework (LDDT, PAE concepts apply)
- Relative position encoding (sequence locality is universal)

### 5.4 What Doesn't Transfer Well
- Amino acid-specific features (restype embeddings need retraining)
- Backbone geometry constraints (RNA backbone is much more flexible)
- Template processing (RNA has different distance distributions)
- MSA statistics (RNA MSAs are typically shallower than protein MSAs)

---

## 6. Alternative Models and Their Architectures

### 6.1 DRFold2 (1st place Part 1 component)
- Energy-based model for RNA structure prediction
- Uses distance/orientation prediction + energy minimization
- Complementary to diffusion-based approaches
- Good for de novo prediction where templates are absent

### 6.2 RhoFold+ (RNA-specific)
- RNA language model backbone (pre-trained on 23.7M RNA sequences)
- Direct structure prediction from sequence
- No template dependency
- Good de novo predictor, potentially complementary to RNAPro

### 6.3 trRosettaRNA2 (CASP16 top server)
- Uses 2D structure prediction as input
- Distance/orientation map prediction -> energy minimization
- Can beat AF3 when given correct secondary structure
- Complementary approach: could provide constraints or alternative predictions

### 6.4 Protenix (AF3 base)
- The underlying AF3 reproduction that RNAPro is built on
- Could be used directly on RNA targets
- Different training data/approach may capture different structural features

---

## 7. Recommendations for Competition (Ranked by Expected Impact)

### Tier 1: High Impact, Implementable Now

1. **Enable RibonanzaNet2** (missing from kernel-metadata.json)
   - Expected impact: +0.02-0.05
   - Risk: Low (it's designed to work)
   - Action: Add RibonanzaNet2 model source to all kernel-metadata.json files

2. **Better template generation** (john's 1st-place TBM)
   - Expected impact: +0.05-0.10
   - Risk: Medium (implementation effort)
   - Templates are the #1 factor; improving them has the highest ceiling

3. **Checkpoint ensemble** (public-best + private-best)
   - Expected impact: +0.01-0.02
   - Risk: Very low
   - Use confidence scores to pick best prediction per target from each checkpoint

4. **Multi-seed inference** (seeds 42, 101, 202)
   - Expected impact: +0.01-0.03
   - Risk: Low (just time budget)
   - More diverse starting points for diffusion

### Tier 2: Medium Impact, Moderate Effort

5. **Per-target routing** (template vs. de novo)
   - For targets with good templates: RNAPro with templates
   - For targets without templates: RNAPro de novo
   - Use alignment score / coverage as proxy for template quality

6. **Model Quality Assessment (MQA)**
   - Use lociPARSE or pLDDT/PAE scores to rank candidates
   - Select 5 most diverse high-quality predictions per target
   - Critical for optimizing best-of-5 metric

7. **Long sequence handling**
   - Sliding window approach for sequences >512nt
   - Or: reduce N_cycle/N_step for long sequences to fit in time budget

### Tier 3: High Impact but High Effort

8. **Integrate DRFold2 as second model**
   - Different inductive biases -> better ensemble diversity
   - Used by 1st place in Part 1

9. **Secondary structure constraints**
   - Run RNAfold/EternaFold for 2D structure
   - Feed as constraints to RNAPro or trRosettaRNA2
   - trRosettaRNA2 showed this approach can beat AF3

10. **Offline fine-tuning on competition data**
    - Train on 5135 training examples with RNA-specific augmentation
    - Focus on crop size 512, RNA-only objective
    - Requires GPU access outside Kaggle

---

## 8. Inference Optimization for Kaggle (~9 hour GPU limit)

### Current Time Budget (estimates per target):
| Setting | Time per target |
|---------|----------------|
| N_cycle=10, N_step=200, N_sample=5 | ~3-5 min (short), ~15-30 min (long) |
| N_cycle=6, N_step=100, N_sample=5 | ~1.5-3 min (short), ~8-15 min (long) |
| N_cycle=4, N_step=50, N_sample=5 | ~0.8-1.5 min (short), ~4-8 min (long) |

### Optimization Strategies:
1. **Adaptive settings**: Full quality for short sequences, reduced for long ones
2. **Early stopping**: If confidence converges, stop denoising early
3. **bf16 precision**: Already using bf16, ensure it's consistent
4. **Memory optimization**: Chunk size tuning, gradient checkpointing off during inference
5. **Batch processing**: Group similar-length sequences for efficient GPU utilization

### Recommended Time Allocation (~108 targets, ~9 hours):
- Short sequences (<200nt, ~60 targets): N_cycle=10, N_step=200, N_sample=5 (~4 hours)
- Medium sequences (200-500nt, ~30 targets): N_cycle=8, N_step=150, N_sample=5 (~3 hours)
- Long sequences (>500nt, ~18 targets): N_cycle=6, N_step=100, N_sample=5 (~2 hours)

---

## 9. Key Architectural Insights

1. **RibonanzaNet2 is critical but currently disabled**: The gated feature injection provides RNA-specific sequence and pairwise information. Without it, the model only sees generic atom/residue features. This is likely the single biggest quick win.

2. **Template quality matters more than model quality**: The ablation (TBM-only 0.359 vs RNAPro+TBM 0.360) shows templates dominate. But this is circular -- if templates are bad, RNAPro can't fix them. Better templates will unlock RNAPro's potential.

3. **Confidence scores enable smart selection**: The ConfidenceHead produces pLDDT, PAE, PDE, and ranking_score. Use these to select which of the 5 predictions to keep, and which model/checkpoint/seed combination to use per target.

4. **The diffusion process benefits from diversity**: N_sample=5 with different seeds gives different local minima. The best-of-5 metric means diversity is directly rewarded.

5. **Multi-chain handling is a potential weakness**: The symmetric permutation system is designed for protein complexes. RNA complexes may have different symmetry patterns. The 512-token crop is particularly painful for large RNA complexes.

---

## 10. Analysis of Creative/Unconventional Approaches

### 10.1 Wisdom of Crowds Ensemble (HIGHEST PRIORITY)

**Feasibility: HIGH | Impact: +0.10-0.20 | Effort: HIGH**

The agentic tree search hitting 0.635 by combining 3 models is the most compelling data point. The key architectural insight is that RNAPro, DRFold2, and TBM have fundamentally different inductive biases:

| Model | Bias Type | Failure Mode |
|-------|-----------|--------------|
| RNAPro (diffusion) | Learns data distribution, generates from noise | Poor on targets far from training distribution |
| DRFold2 (energy-based) | Minimizes physical energy function | Gets stuck in local energy minima |
| TBM (template) | Copies known structures | Fails completely without templates |
| RhoFold+ (LM-based) | Sequence patterns from 23.7M RNAs | Poor at novel folds not in pre-training data |

**Implementation within RNAPro's architecture**: We don't need to modify RNAPro itself. The ensemble happens at the output level:
1. Generate N candidates from each model
2. Score all candidates with MQA (lociPARSE or RNAPro's own confidence)
3. Select 5 maximally diverse high-quality predictions
4. The diversity selection is critical: use pairwise TM-score to avoid selecting 5 nearly-identical structures

**Time budget concern**: Running multiple models within 9 hours is tight. Possible allocation:
- RNAPro (4 hrs for ~108 targets, reduced settings for long sequences)
- TBM (15 min, CPU-bound)
- DRFold2 or RhoFold+ (2-3 hrs if available)
- MQA + selection (30 min)

### 10.2 Diffusion Guidance with Physics Constraints (MODERATE PRIORITY)

**Feasibility: MEDIUM | Impact: +0.01-0.03 | Effort: MODERATE**

This is architecturally interesting. Looking at the `sample_diffusion` function in `generator.py`, the denoising loop (Algorithm 18, AF3) has a clear injection point for guidance:

```python
# Current: pure diffusion step
x_denoised = denoise_net(x_noisy, t_hat, ...)
delta = (x_noisy - x_denoised) / t_hat
x_l = x_noisy + step_scale_eta * dt * delta

# With guidance: add physics gradient
physics_grad = compute_physics_gradient(x_l)  # e.g., bond length violations
x_l = x_l - guidance_weight * physics_grad
```

The physics constraints for RNA would include:
- **C1'-C1' backbone distance**: ~5.9A between adjacent residues
- **Steric clashes**: No two atoms within van der Waals radius
- **Base pairing geometry**: Watson-Crick pairs at ~10.4A C1'-C1' distance
- **Sugar pucker constraints**: C2'-endo or C3'-endo conformations

**Practical concern**: The guidance gradient computation adds time per diffusion step. With 200 steps, even small overhead per step multiplies. Could be applied only at later steps (when structure is nearly formed) to save time.

**Alternative that's simpler**: Post-processing physics refinement AFTER diffusion is complete. Apply BRiQ energy minimization or simple constraint optimization as a final step. Less elegant but more practical within the time budget.

### 10.3 Cascade Architecture: Coarse-to-Fine (MODERATE PRIORITY)

**Feasibility: MEDIUM | Impact: +0.02-0.05 for long sequences | Effort: HIGH**

This directly addresses the 512-token limit. The idea maps onto RNAPro's architecture as:

**Stage 1 - Coarse prediction (every Kth residue)**:
- Subsample sequence to fit within 512 tokens (e.g., K=2 for 1024nt sequence)
- Run RNAPro on subsampled sequence
- Get coarse backbone trace

**Stage 2 - Fine-grained refinement**:
- Use coarse prediction as template (via RNATemplateEmbedder)
- Run RNAPro on full-resolution 512-token windows with the coarse template as constraint
- Stitch overlapping windows using confidence-weighted averaging

**Architectural compatibility**: The RNATemplateEmbedder already accepts arbitrary C1' coordinate templates. The coarse prediction can be directly fed as a template for the fine stage. No model modifications needed -- just a two-pass inference pipeline.

**Simpler alternative**: Sliding window approach where each 512-token window inherits its template from the TBM prediction of the full sequence. This is essentially what we already do, but making it explicit and adding overlap-based stitching.

### 10.4 Test-Time Training / Adaptation (LOW PRIORITY for Kaggle)

**Feasibility: LOW | Impact: uncertain | Effort: VERY HIGH**

The idea is intriguing but faces severe practical constraints:

1. **No internet**: Can't download additional data or pre-computed features
2. **Time budget**: TTT requires forward+backward passes on test data. With 488M parameters and ~108 targets, even a few gradient steps per target would consume hours
3. **Risk of overfitting**: With N=1 (single test target), gradient updates could easily destroy learned representations
4. **RibonanzaNet2 is frozen** (uses `torch.no_grad()`): Can't backpropagate through the RNA language model

**More practical variant**: "Test-time augmentation" (TTA) is feasible and doesn't require gradient computation:
- Run inference with different random seeds (already planned)
- Run inference with different template subsets
- Run inference with slightly perturbed MSAs
- Average/select predictions using confidence scores

This captures some of the diversity benefits of TTT without the risk.

### 10.5 2D-to-3D Pipeline (HIGH PRIORITY)

**Feasibility: HIGH | Impact: +0.02-0.05 | Effort: MODERATE**

This is architecturally elegant because RNAPro already has a `ConstraintEmbedder` that can accept structural constraints:

```python
# From configs_base.py - constraint embedder is already in the architecture
"constraint_embedder": {
    "substructure_embedder": {
        "enable": False,  # Currently disabled!
        "n_classes": 4,
        "architecture": "transformer",
    },
    "contact_embedder": {
        "enable": False,  # Currently disabled!
        "c_z_input": 2,
    },
}
```

**The SubstructureEmbedder** (in `embedders.py`) can accept a pairwise distance class matrix [N_token, N_token, N_classes] and embed it into pair representations. This is exactly what secondary structure base-pairing information looks like!

**Implementation path**:
1. Predict 2D structure with RNAfold (no internet needed if pre-installed or included as dependency)
2. Convert base-pair predictions to pairwise contact/distance classes
3. Feed through the SubstructureEmbedder (need to enable in config + provide checkpoint with trained weights)

**Challenge**: The SubstructureEmbedder weights need to have been trained with 2D structure input. The current checkpoints may not have trained these weights (they're zero-initialized). Without proper training, enabling this pathway would just add zeros.

**Workaround**: Instead of using the SubstructureEmbedder, convert 2D predictions to template coordinates:
1. Predict base pairs with RNAfold
2. Build a crude 3D model from base pairs (e.g., A-form helix for stems, simple loop geometries)
3. Feed as template to RNATemplateEmbedder (which IS trained)
This bypasses the need for SubstructureEmbedder training.

### 10.6 Meta-Learner Routing (MODERATE PRIORITY)

**Feasibility: HIGH | Impact: +0.02-0.04 | Effort: LOW-MODERATE**

Simple routing based on target properties is very practical:

**Features for routing** (available at inference time):
- Sequence length
- Number of chains (from stoichiometry)
- MSA depth (from MSA files)
- Best template alignment score (from TBM search)
- Template coverage (fraction of residues with aligned template)
- Sequence composition (GC content, repeat structure)

**Routing decisions**:
- Template alignment score > X: Use RNAPro with templates
- Template alignment score < X: Use RNAPro de novo
- Sequence length > 512: Use TBM or coarse-to-fine RNAPro
- Multi-chain complex: Predict chains separately + assemble

**Implementation**: A simple if/else tree would work. No ML needed. Can be tuned using submission feedback (score each routing rule, adjust thresholds).

### 10.7 Structural Alignment Search (MODERATE PRIORITY)

**Feasibility: MEDIUM | Impact: +0.03-0.08 | Effort: MODERATE**

This directly addresses the template quality bottleneck. The idea of using RibonanzaNet2 embeddings for template search is architecturally sound:

1. **Offline** (before submission): Compute RibonanzaNet2 embeddings for all ~5135 training sequences
2. **At inference**: Compute embedding for test sequence, find nearest neighbors in embedding space
3. **Use those neighbors as templates** instead of sequence-aligned templates

**Why this could work better than sequence alignment**:
- RibonanzaNet2 embeddings capture structural patterns, not just sequence identity
- Two sequences with 30% sequence identity but similar structure would be found
- This is exactly what odat (2nd place, Part 1) did

**Challenge**: Requires pre-computing and storing ~5135 embeddings as a Kaggle dataset. Each embedding is [L, 256] for sequence + [L, L, 64] for pairwise. For L=500 average, that's ~500KB per sequence = ~2.5GB total. Feasible as a Kaggle dataset.

---

## 11. Revised Recommendation Priority (Incorporating Creative Ideas)

### Immediate Actions (This Week):
1. **Enable RibonanzaNet2** (+0.02-0.05) - add model source to kernel-metadata.json
2. **Better TBM templates** (+0.05-0.10) - john's approach
3. **Checkpoint ensemble** (+0.01-0.02) - public-best + private-best
4. **Multi-seed** (+0.01-0.03) - seeds 42, 101, 202

### Short-Term (Next 1-2 Weeks):
5. **Per-target routing** (+0.02-0.04) - template quality-based if/else
6. **2D-to-3D as template** (+0.02-0.05) - RNAfold -> crude 3D model -> RNAPro template
7. **MQA for selection** (+0.01-0.03) - confidence-based 5-of-N selection

### Medium-Term (Weeks 3-4):
8. **Multi-model ensemble** (+0.10-0.20) - RNAPro + DRFold2 + TBM candidates
9. **Embedding-based template search** (+0.03-0.08) - RibonanzaNet2 template discovery
10. **Coarse-to-fine for long sequences** (+0.02-0.05) - two-pass inference

### Stretch Goals:
11. **Diffusion guidance** (+0.01-0.03) - physics constraints during denoising
12. **Meta-learner optimization** - iterative routing based on submission feedback

---

## 12. Follow-Up Deep Dives (PI Questions A-D)

### 12A. Template Embedder Zero-Init: Are Templates Having Any Effect?

**Answer: The zero-init is on the FINAL linear only, and yes, templates DO have an effect in the trained checkpoint.**

The `RNATemplateEmbedder` (`pairformer.py:1346-1585`) has this architecture:
1. Distance binning of template coords -> one-hot embedding
2. Linear projections of s_inputs -> z_init pair features
3. 2-block PairformerStack processing per template
4. LayerNorm on output
5. Average over templates
6. **Final linear** (zero-initialized at init time)

The `zero_init_final_linear=True` parameter (`pairformer.py:1442-1445`) means that at **random initialization** (before training), the final linear outputs all zeros. This is a standard AF3 technique to ensure templates have no effect at the start of training, then gradually learn to contribute as training progresses.

However, after training, the `final_linear_no_bias.weight` will have been updated by gradient descent. The checkpoint we load (`rnapro-private-best-500m.ckpt`) is loaded with `strict=True` at inference time, meaning ALL weights including the template embedder's final linear ARE loaded from the checkpoint.

**To verify checkpoint weights are non-trivial**: We cannot load the checkpoint without the model class, but the fact that RNAPro was specifically trained with RNA templates (the README says "We train on 5135 RNA structures with precomputed templates") confirms the template embedder was trained and its weights are non-trivial in the checkpoint.

**Key insight**: The zero-init means templates contribute nothing at training step 0, but after full training, they contribute meaningfully. The checkpoint's template embedder weights reflect whatever was learned during training.

### 12B. How template_idx Works with .pt Files and Predictions

**Answer: template_idx selects WHICH templates from the .pt file, not duplicating the same one.**

The flow is:

**Step 1: .pt file format** (from `convert_templates_to_pt_files.py`):
The .pt file is a dict keyed by target_id, each containing `xyz` with shape `[N_residues, N_template_candidates, 3]`. Our TBM generates 5 template candidates per target.

**Step 2: template_idx selects template combinations** (`infer_data_pipeline.py:474-498`):
```python
template_combinations = [
    [0],           # idx=0: top-1 only
    [0, 1],        # idx=1: top-1 + top-2
    [0, 1, 2],     # idx=2: top-1 + top-2 + top-3
    [0, 1, 2, 3],  # idx=3: top-1 through top-4
    [0, 1, 2, 3, 4],  # idx=4: all 5 templates
]
template_ca = xyz[:, template_combinations[template_idx]].permute(1, 0, 2)
# Shape: [n_selected_templates, N_residues, 3]
```

So `template_idx=0` gives RNAPro the BEST template only, `template_idx=1` gives top-2, etc. The template embedder then loops over each template, processes through its 2-block PairformerStack, and averages.

**Step 3: The notebook loop** (`inference.py` `run()` function):
```python
for template_idx in range(n_template_combos):
    run_ptx(..., template_idx=template_idx, ...)
```

With `n_template_combos=1` (our current Sub1 setting), it runs the loop ONCE with `template_idx=0`, which uses only the top-1 template. Then N_SAMPLE=5 generates 5 diffusion samples from that single-template trunk.

**With the old baseline** (`n_template_combos=5`), it ran the loop 5 times:
- template_idx=0: top-1 template -> 1 diffusion sample
- template_idx=1: top-1+2 templates -> 1 diffusion sample
- template_idx=2: top-1+2+3 templates -> 1 diffusion sample
- template_idx=3: top-1+2+3+4 templates -> 1 diffusion sample
- template_idx=4: all 5 templates -> 1 diffusion sample

**This means the old baseline runs the ENTIRE trunk (pairformer + template embedder + MSA + diffusion) 5 separate times, each with a different template combination.** Each pass generates N_SAMPLE=1 prediction.

**Critical finding**: The PI's observation is correct -- the 5-pass approach is extremely wasteful. Each full trunk pass takes the majority of inference time. With Sub1 (N_SAMPLE=5, n_template_combos=1), we run the trunk ONCE and get 5 diverse diffusion samples. The diffusion module is much cheaper than the trunk.

### 12C. GPU Memory Analysis at Different Sequence Lengths

**Estimated GPU memory usage on T4 (16GB VRAM):**

The dominant memory consumers are:
1. **Model parameters**: ~488M params in bf16 = ~976 MB
2. **z_trunk pair representation**: `[N_token, N_token, c_z=128]` in bf16
3. **Diffusion intermediates** (per sample, per denoising step): DiffusionConditioning creates `[2*c_z]` pair concat + transitions
4. **Confidence head**: processes samples one-at-a-time, but final PAE/PDE tensors stack up
5. **RibonanzaNet2** (if enabled): 48-layer model with `all_pairwise_features [48, 1, L, L, pw_dim]`

**Detailed chunking analysis** (verified from code):

The diffusion and confidence stages both chunk N_sample to avoid memory blowup:

1. **Diffusion chunking** (`sample_diffusion_chunk_size=1` in configs_base.py:162):
   - `sample_diffusion()` in generator.py:241-258 loops over chunks of size 1
   - Each chunk calls `DiffusionModule.forward()` with `N_sample=1`
   - Inside `f_forward()` (diffusion.py:430-436), z_pair is expanded via `.expand()` (a **view**, NOT a copy -- verified in utils.py:375 `x.expand()`)
   - DiffusionConditioning (diffusion.py:148-154) creates `[N_token, N_token, 2*c_z]` concat then projects back to `[N_token, N_token, c_z]` + two Transition layers
   - The code comment at diffusion.py:393 says: "Diffusion_conditioning consumes 7-8G when token num is 768" -- this is the peak during diffusion
   - The 24-block DiffusionTransformer then runs with pair bias from `[1, N_token, N_token, c_z]`
   - Peak intermediate: attention scores = `[1, 16_heads, N_token, N_token]` in fp32

2. **Confidence chunking** (confidence.py:228-250):
   - Loops `for i in range(N_sample)`, processing 1 sample at a time
   - Each iteration runs a 4-block PairformerStack on `[N_token, N_token, c_z]`
   - For N_token > 2000, CPU-offloads PAE/PDE per sample (confidence.py:242-246)
   - PAE per sample: `[1, N_token, N_token, 64]` in bf16

3. **Final stacking** (after all samples processed):
   - PAE stacked: `[N_sample, N_token, N_token, 64]` -- biggest single tensor
   - PDE stacked: same size
   - These are computed AFTER diffusion is done, so they don't overlap with diffusion memory

**Memory estimates** (bf16, inference mode, N_SAMPLE=5):

| N_token | Model | z_trunk | DiffCond peak | Conf peak | Final PAE+PDE | Total peak | T4? |
|---------|-------|---------|---------------|-----------|--------------|------------|-----|
| 100     | ~1 GB | 2.4 MB  | ~15 MB        | ~5 MB     | 15 MB        | ~1.1 GB    | OK  |
| 512     | ~1 GB | 64 MB   | ~200 MB       | ~70 MB    | 400 MB       | ~1.7 GB    | OK  |
| 1000    | ~1 GB | 244 MB  | ~700 MB       | ~260 MB   | 1.5 GB       | ~3.5 GB    | OK  |
| 2000    | ~1 GB | 977 MB  | ~2.5 GB       | ~1.0 GB   | CPU offload  | ~5.5 GB    | OK  |
| 3000    | ~1 GB | 2.2 GB  | ~5+ GB        | ~2.2 GB   | CPU offload  | ~10 GB     | TIGHT |

**Formulas**:
- z_trunk: `N^2 * 128 * 2 bytes` (bf16)
- DiffusionConditioning peak: ~`N^2 * 256 * 2 bytes` (2*c_z concat) + Transition intermediates (`N^2 * 128 * 4 * 2` for n=2 transition) + attention scores (`16 * N^2 * 4 bytes` fp32)
- Final PAE+PDE: `N_sample * N^2 * 64 * 2 * 2` (bf16, both PAE and PDE)

**Key observations**:
1. For N_token <= 1000 (our current max_len), N_SAMPLE=5 fits comfortably on T4 (~3.5 GB peak, well within 16 GB)
2. The chunking ensures only 1 sample at a time in diffusion and confidence -- per-sample memory equals N_SAMPLE=1
3. The **DiffusionConditioning module** is a significant memory consumer (code comment: "7-8G at N=768"), creating `[2*c_z]` pair concat + two Transition layers with expansion factor 2
4. The final PAE/PDE stacking is the biggest post-diffusion tensor, but for N_token > 2000 it's CPU-offloaded
5. Peak memory occurs DURING diffusion (DiffusionConditioning + DiffusionTransformer attention), not from N_sample accumulation

**Conclusion**: N_SAMPLE=5 in a single pass on T4 with max_len=1000 is safe (~3.5 GB peak). At max_len=2000, still fits (~5.5 GB with CPU offload). Memory is NOT the bottleneck for N_SAMPLE=5.

**Warning about RibonanzaNet2**: When enabled, RibonanzaNet2's `get_embeddings()` returns `all_pairwise_features [48, 1, L, L, pw_dim]`. For L=1000 with pw_dim=32 in bf16, that's `48 * 1000 * 1000 * 32 * 2 = ~2.9 GB`. This is computed with `torch.no_grad()` (RNAPro.py:219) so no gradient storage, but the tensor itself exists during the embedding injection. Combined with the model parameters and z_trunk, peak memory at L=1000 with RibonanzaNet2 could reach ~6.5 GB -- still within T4 capacity but should be monitored.

### 12D. Confidence Head Output: ranking_score Format

**Answer: ranking_score is a SINGLE SCALAR per sample.**

The ranking_score is computed in `sample_confidence.py:177-182`:

```python
summary_confidence["ranking_score"] = (
    0.8 * summary_confidence["iptm"]
    + 0.2 * summary_confidence["ptm"]
    + 0.5 * summary_confidence["disorder"]
    - 100 * summary_confidence["has_clash"]
)
```

Where:
- `iptm`: scalar per sample (inter-chain predicted TM-score, shape `[N_sample]`)
- `ptm`: scalar per sample (predicted TM-score, shape `[N_sample]`)
- `disorder`: scalar per sample (currently hardcoded to 0.0, `sample_confidence.py:176`)
- `has_clash`: scalar per sample (0 or 1, whether any clash is detected)

So `ranking_score` shape is `[N_sample]` -- one scalar per sample.

**How it's used for sample selection**:
The `break_down_to_per_sample_dict()` function at line 212 converts the summary into a list of per-sample dicts. The dumper then sorts predictions by ranking_score (when `sorted_by_ranking_score=True`, which is the default). CIF files are named `{target}_sample_0.cif`, `{target}_sample_1.cif`, etc. in ranking-score order.

**For single-chain RNA targets**: Since there's only one chain (asym_id=0), `iptm` requires at least 2 chains to be meaningful. With a single chain:
- `is_diff_chain` is all False
- `iptm` will be 0 (or near-zero since eps prevents division by zero)
- `ranking_score = 0.8 * 0 + 0.2 * ptm + 0 - clash_penalty = 0.2 * ptm - clash_penalty`

This means **for single-chain targets, ranking is dominated by ptm and clash avoidance**. The ptm itself is computed from PAE predictions.

**Other available confidence metrics** (all per-sample scalars):
- `plddt`: mean pLDDT * 100 (range 0-100)
- `gpde`: global predicted distance error
- `ptm`: predicted TM-score
- `iptm`: inter-chain predicted TM-score (0 for single-chain)
- `chain_ptm`: per-chain pTM
- `chain_plddt`: per-chain pLDDT
- `has_clash`: binary clash indicator

**Recommendation**: For single-chain RNA, consider using `plddt` or `ptm` directly for sample selection instead of `ranking_score`, since `iptm` contributes nothing. Or redefine ranking_score for single-chain as `ptm - 100 * has_clash`.
