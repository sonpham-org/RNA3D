# Research Plan: Stanford RNA 3D Folding Part 2

## Goal: 0.361 -> 0.55+ TM-score (Silver Medal)

**Current score:** 0.361 (RNAPro + TBM hybrid, N_SAMPLE=5)
**Target:** 0.55+ TM-score
**Deadline:** March 25, 2026 (~46 days remaining)
**Constraints:** Notebook-only, no internet during scoring, ~9hr GPU limit (2x T4 16GB)

---

## Root Cause Analysis: Why We're at 0.361

The gap between us (0.361) and the target (0.55+) comes from five compounding problems:

1. **Bad templates (biggest factor):** Our TBM uses naive global pairwise alignment (BioPython) and scores 0.359 alone. john's 1st-place TBM scored 0.591 using the same concept. That's a 0.234 gap in template quality alone.

2. **Circular template pipeline:** Our 5 "templates" fed to RNAPro are all variations of the SAME TBM alignment result (best, jitter, hinge, chain perturbation, smooth wiggle). RNAPro gets no diverse structural hypotheses.

3. **Missing RibonanzaNet2:** Our kernel-metadata.json is missing the RibonanzaNet2 model source (`shujun717/ribonanzanet2/pyTorch/alpha/1`). RNAPro runs WITHOUT its RNA foundation model features, falling back to generic AF3 behavior.

4. **Wasteful inference:** We run the full RNAPro trunk 5 separate times (template_idx 0-4), each producing only 1 diffusion sample. This is ~5x more expensive than necessary.

5. **No diversity:** Single seed, single checkpoint, single model, single template source. All 5 predictions are near-identical.

6. **Ignoring protein context in RNA-protein complexes:** ~36%+ of targets may be RNA-protein complexes. Our pipeline treats everything as RNA-only, but RNAPro (being AF3-based) natively supports protein+RNA inputs. Providing protein chain sequences as context could significantly improve RNA structure prediction for these targets.

---

## Priority Stack (Ordered by Impact/Effort Ratio)

### TIER 0: Immediate Fixes (Day 1-2, +0.03 to +0.08 expected)

These are bugs/oversights that should be fixed before any new experiments.

#### 0A. Add RibonanzaNet2 to kernel-metadata.json
- **What:** Add `"shujun717/ribonanzanet2/pyTorch/alpha/1"` to `model_sources` in all kernel-metadata.json files
- **Why:** RNAPro was designed to use RibonanzaNet2 as its RNA foundation model encoder. Without it, the model is generic AF3, not RNA-specialized. The gated injection mechanism (learnable sigmoid gates on 48-layer features) provides both sequence and pairwise RNA-specific representations.
- **Effort:** 1 minute (edit JSON)
- **Expected impact:** +0.02-0.05 TM-score
- **Risk:** None

#### 0B. Fix inference: 1 pass with N_SAMPLE=5 instead of 5 passes x 1 sample
- **What:** Change to N_SAMPLE=5, run once per target with template_idx=4 (all 5 templates). Remove the loop that runs 5 separate inference passes.
- **Why:** Current approach recomputes the full Pairformer trunk (48 blocks) 5 times. With N_SAMPLE=5 in 1 pass: 1 trunk computation + 5 diffusion samples. Saves ~60% GPU time AND gets better diversity (different noise initializations).
- **Effort:** Modify inference loop in notebook
- **Expected impact:** +0.01-0.03 TM-score + massive time savings
- **Risk:** Low. May need to adjust `sample_diffusion_chunk_size` for memory.

#### 0C. Multi-seed inference
- **What:** Use `--seeds 42,101,202` instead of `--seeds 42`
- **Why:** Each seed gives an independent trunk computation with different dropout/noise. 3 seeds x 5 samples = 15 candidates. Select best 5 by ranking_score.
- **Effort:** Change 1 parameter
- **Expected impact:** +0.01-0.03 TM-score
- **Risk:** 3x time cost for RNAPro inference (but offset by 0B savings)

#### 0D. Use both checkpoints
- **What:** Run inference with both `rnapro-private-best-500m.ckpt` and `rnapro-public-best-500m.ckpt`
- **Why:** Different checkpoints may excel on different target types. Per-target selection by ranking_score.
- **Effort:** Add second checkpoint to dataset, run inference twice
- **Expected impact:** +0.01-0.02 TM-score
- **Risk:** Need to upload public-best checkpoint as Kaggle dataset. 2x time cost.
- **Source:** HuggingFace `nvidia/RNAPro-Public-Best-500M`

### TIER 1: Template Revolution (Days 3-10, +0.10 to +0.20 expected)

Template quality is the #1 factor. This tier is the single biggest lever.

#### 1A. Implement john's 1st-place TBM approach [HIGHEST PRIORITY]
- **What:** Study and replicate john's TBM method from his public notebook: https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach
- **Why:** His TBM-only scored 0.591 vs our 0.359. Same concept, vastly different execution. The gap is almost certainly in: template selection algorithm, alignment quality, coordinate transfer method, and multi-template usage.
- **Key differences to investigate:**
  - Does he use local vs global alignment?
  - Does he search against more databases (PDB, not just competition training data)?
  - Does he handle multi-chain targets differently?
  - Does he use structural alignment (US-align/RNA-align) rather than sequence alignment?
  - Does he have a per-target quality filter?
- **Effort:** High (requires careful study and reimplementation)
- **Expected impact:** +0.10-0.20 TM-score (the single biggest improvement possible)
- **Risk:** Medium. His approach may depend on resources we don't have.

#### 1B. Expand template database with PDB RNA structures
- **What:** Pre-extract C1' coordinates from all RNA-containing PDB structures released before Sep 30, 2024. Upload as Kaggle dataset.
- **Why:** Our current template database is ~5,135 competition training sequences. PDB has thousands more RNA structures. RNA3DB alone has 1,645 distinct RNA sequences across 21,005 crystal structures and 216 Rfam families.
- **Process:**
  1. Download RNA structures from PDB (respecting temporal cutoff)
  2. Extract C1' atom coordinates
  3. Build searchable sequence/structure database
  4. Upload as Kaggle dataset (~1-5 GB)
- **Effort:** Medium (offline data processing + upload)
- **Expected impact:** +0.03-0.05 TM-score (more templates for more targets)
- **Risk:** Low. More templates can only help.

#### 1C. Multi-method template search pipeline
- **What:** Combine multiple template search methods and feed genuinely diverse templates to RNAPro
- **Methods to combine:**
  1. john's TBM approach (best sequence-based)
  2. MMseqs2 structural search (pre-computed dataset available: `rhijudas/rna-3d-folding-templates`)
  3. Our current BioPython alignment (as fallback)
- **Template usage:** Instead of 5 variations of 1 template, feed 5 templates from different sources/structures
- **Effort:** Medium
- **Expected impact:** +0.03-0.05 TM-score
- **Risk:** Low

#### 1D. Chain-level alignment for multi-chain targets
- **What:** Align chains individually instead of concatenating the whole sequence
- **Why:** Current approach concatenates all chains into one long sequence for alignment. This misaligns chains in multi-chain complexes. Better: match chain stoichiometry, align per-chain, assemble.
- **Effort:** Medium (refactor TBM code)
- **Expected impact:** +0.01-0.03 TM-score (depends on fraction of multi-chain targets)
- **Risk:** Low

#### 1E. RNA-protein co-folding for complex targets [HIGH PRIORITY -- NEW FINDING]
- **What:** For the ~36%+ of test targets that are RNA-protein complexes, provide protein chain sequences as co-folding context to RNAPro
- **Why:** RNAPro is built on AF3/Protenix, which natively supports multi-molecule (protein+RNA) inputs. The data pipeline already tracks protein/DNA/RNA/ligand molecule types (verified in `infer_data_pipeline.py:371`). Currently we feed ONLY RNA sequences, ignoring protein chains that shape the RNA fold. For RNA-protein complexes, the protein context can dramatically constrain and improve RNA structure prediction.
- **Implementation:**
  1. Parse `all_sequences` and `stoichiometry` to identify protein chains in test targets
  2. Include protein chain sequences in the RNAPro input JSON (as `proteinChain` entries alongside `rnaSequence`)
  3. RNAPro will co-fold protein+RNA, but we only extract RNA C1' coordinates for submission
  4. For template search: also find protein-RNA complex templates from PDB
- **Effort:** Medium (modify input JSON generation, may need protein templates)
- **Expected impact:** +0.03-0.08 TM-score (affects 36%+ of targets; protein context constrains RNA fold)
- **Risk:** Medium (increases sequence length -> more compute; protein templates needed; verify RNAPro handles mixed inputs correctly at inference)
- **Note:** Must check if test `all_sequences` contains protein chains. If test data only provides RNA sequences, we would need to identify protein partners from PDB metadata.

### TIER 2: Model Diversity and Ensembling (Days 8-18, +0.05 to +0.15 expected)

Adding a second model and smart selection is the second biggest lever.

#### 2A. Add DRFold2 as second model
- **What:** Integrate DRFold2 (used by 1st-place winner) as an independent prediction pipeline
- **Why:** DRFold2 is complementary to RNAPro. john used it for de novo predictions. Different architectures make different errors, so ensembling helps.
- **Requirements:** Need DRFold2 weights on Kaggle, must fit within GPU/time budget
- **Key questions (pending from ml-tools-inference):**
  - Can it run on T4 16GB?
  - What's inference time per target?
  - Does it need MSAs?
- **Effort:** High (new model integration)
- **Expected impact:** +0.05-0.10 TM-score
- **Risk:** High (may not fit in time/memory budget)

#### 2B. Consensus scoring for sample selection
- **What:** Generate 15-20 candidates per target from multiple methods. Compute pairwise TM-score (US-align). Select 5 most "central" structures (highest average agreement).
- **Why:** The agentic tree search approach hit 0.635 using this exact strategy. The consensus structure is most likely correct. Diverse outliers cover alternative conformations.
- **Implementation:**
  1. Generate candidates: RNAPro (template, 5 samples) + RNAPro (de novo, 5 samples) + TBM (5 templates) = 15 candidates
  2. Run US-align pairwise: 15 * 14 / 2 = 105 comparisons per target
  3. For each candidate, compute average TM-score against all others
  4. Select: 1 highest-consensus + 4 diverse alternatives (maximize min pairwise distance)
- **Effort:** Medium (US-align is fast, runs on CPU)
- **Expected impact:** +0.05-0.10 TM-score
- **Risk:** Low (US-align is well-tested, no GPU needed)

#### 2C. Per-target routing
- **What:** Classify each test target by template quality, then route to optimal method
- **Routing logic:**
  - Best alignment score > 0.7: TBM primary, RNAPro with templates
  - Score 0.4-0.7: RNAPro with templates + de novo, select best
  - Score < 0.4: De novo primary (RNAPro no-template + alternative models)
  - Score < 0.2: Pure de novo, multiple models, maximum samples
- **Template quality signal:** alignment score, coverage, MSA depth (Neff)
- **Effort:** Medium
- **Expected impact:** +0.03-0.08 TM-score
- **Risk:** Low

#### 2D. RNAPro de novo mode for template-free targets
- **What:** Run RNAPro with `--use_template None --use_msa true` for targets without good templates
- **Why:** Part 2 explicitly includes template-free targets. For these, forcing bad templates may hurt.
- **Effort:** Low (already implemented in sub2_denovo, just needs scoring)
- **Expected impact:** +0.02-0.05 TM-score (on template-free targets)
- **Risk:** Low

### TIER 3: Structural Refinement and Constraints (Days 15-25, +0.02 to +0.05 expected)

#### 3A. Improve RNA geometric constraints
- **What:** Update our constraint values based on RNA crystallography literature
- **Current values:** C1'-C1' = 5.95A, i+2 distance = 10.2A
- **Literature values to verify:**
  - C1'-C1' consecutive: ~5.9A (A-form helix) to ~6.5A (extended)
  - P-P consecutive: ~5.8-6.2A
  - Base pair C1'-C1': ~10.4A (Watson-Crick)
  - Stacking distance: ~3.3-3.4A
- **Effort:** Low (update constants + add new constraint types)
- **Expected impact:** +0.01-0.02 TM-score
- **Risk:** Low

#### 3B. Secondary structure-informed gap modeling
- **What:** Instead of linear interpolation for alignment gaps, use secondary structure prediction to model gap regions
- **Why:** Linear interpolation creates unphysical extended chains for gaps. If we know a gap region is a hairpin loop, we can model it as a loop.
- **Tools:** RNAfold, CONTRAfold, or EternaFold for 2D prediction (can be pre-computed)
- **Effort:** Medium
- **Expected impact:** +0.01-0.03 TM-score
- **Risk:** Medium (depends on 2D prediction quality)

#### 3C. Diverse 2D structure inputs for 5 prediction slots
- **What:** Use different secondary structure prediction tools to generate diverse 3D predictions
- **Slots:**
  1. RNAfold 2D -> constrain prediction
  2. CONTRAfold 2D -> constrain prediction
  3. Best template 2D -> constrain prediction
  4. De novo (no 2D constraint)
  5. Consensus best
- **Why:** Different 2D tools make different errors. This maximizes diversity for best-of-5.
- **Effort:** Medium (need to integrate 2D tools + constraint injection into RNAPro)
- **Expected impact:** +0.02-0.05 TM-score
- **Risk:** Medium (RNAPro constraint embedders are disabled; may need code changes)

### TIER 4: Advanced Optimizations (Days 20-35, +0.01 to +0.05 expected)

#### 4A. Adaptive compute allocation
- **What:** Two-pass approach: fast scan (N_step=50, N_cycle=6) for all targets, then full settings (N_step=200, N_cycle=10) for uncertain targets
- **Why:** Some targets are easy (good template, high confidence). Spending full compute on them wastes time. Redirect compute to hard targets.
- **Effort:** Medium
- **Expected impact:** More models/samples within time budget -> indirect score improvement
- **Risk:** Low

#### 4B. Long sequence handling
- **What:** For sequences >512 tokens: predict individual chains/domains separately, then assemble
- **Why:** RNAPro was trained with 256-token crops. Its ability to model long-range interactions is limited. Predicting shorter segments may give better local quality.
- **Assembly:** Use inter-chain contacts from MSA coevolution for docking
- **Effort:** High
- **Expected impact:** +0.01-0.03 TM-score (depends on fraction of long targets)
- **Risk:** Medium

#### 4C. Confidence-only fine-tuning for better ranking
- **What:** Fine-tune RNAPro's confidence head on competition training data to better predict TM-score
- **Why:** Better ranking = better sample selection from N candidates. RNAPro supports `train_confidence_only=True`.
- **Process:** Fine-tune offline, upload weights as Kaggle dataset
- **Effort:** High (offline training + validation)
- **Expected impact:** +0.01-0.03 TM-score
- **Risk:** Medium (overfitting with 5135 samples, 488M params)

#### 4D. Add RhoFold+ as third model
- **What:** RNA-specific model with language model backbone, pre-trained on 23.7M sequences
- **Why:** Complementary to RNAPro (no template dependency), good de novo predictor
- **Effort:** High (new model integration)
- **Expected impact:** +0.02-0.05 TM-score (mainly on template-free targets)
- **Risk:** High (GPU/time budget constraints)

---

## Experiment Roadmap

### Week 1 (Days 1-7): Foundation Fixes
| Day | Submissions | Focus |
|-----|------------|-------|
| 1 | Sub 1: Fix RibonanzaNet2 + N_SAMPLE=5 | Tier 0A + 0B combined |
| 1 | Sub 2: De novo mode (re-test sub2) | Tier 2D baseline |
| 2 | Sub 3: Multi-seed (42,101,202) + RibonanzaNet2 | Tier 0C |
| 2 | Sub 4: Public-best checkpoint | Tier 0D |
| 3 | Sub 5: Evaluate score changes, decide direction | Analysis |
| 4-7 | Study john's TBM notebook, begin implementation | Tier 1A prep |

**Expected score after Week 1:** 0.38-0.42

### Week 2 (Days 8-14): Template Revolution
| Day | Submissions | Focus |
|-----|------------|-------|
| 8-10 | Sub 6-10: john's TBM implementation + testing | Tier 1A |
| 11-12 | Sub 11-12: PDB template expansion | Tier 1B |
| 13-14 | Sub 13-14: Multi-method template pipeline | Tier 1C |

**Expected score after Week 2:** 0.42-0.50

### Week 3 (Days 15-21): Ensembling
| Day | Submissions | Focus |
|-----|------------|-------|
| 15-17 | Sub 15-19: Consensus scoring implementation | Tier 2B |
| 18-19 | Sub 20-22: Per-target routing | Tier 2C |
| 20-21 | Sub 23-24: DRFold2 integration (if feasible) | Tier 2A |

**Expected score after Week 3:** 0.48-0.55

### Week 4 (Days 22-28): Refinement
| Day | Submissions | Focus |
|-----|------------|-------|
| 22-24 | Sub 25-29: Best performing pipeline optimization | Tier 4A |
| 25-26 | Sub 30-32: Long sequence handling | Tier 4B |
| 27-28 | Sub 33-34: Secondary structure experiments | Tier 3B, 3C |

**Expected score after Week 4:** 0.52-0.58

### Weeks 5-6 (Days 29-46): Final Push
- Lock best configuration
- Maximum compute per target
- Target-specific tuning for outliers
- Polish submission format

---

## Key Resources Needed

| Resource | Source | Status |
|----------|--------|--------|
| RibonanzaNet2 checkpoint | `shujun717/ribonanzanet2/pyTorch/alpha/1` (Kaggle Models) | Need to add to metadata |
| RNAPro Public-Best checkpoint | HuggingFace `nvidia/RNAPro-Public-Best-500M` | Need to upload to Kaggle |
| john's TBM notebook | https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach | Need to study |
| MMseqs2 templates | `rhijudas/rna-3d-folding-templates` (Kaggle Dataset) | Available |
| DRFold2 weights | TBD (pending resource-finder report) | Unknown |
| RhoFold+ weights | TBD (pending resource-finder report) | Unknown |
| PDB RNA C1' database | Need to build (RNA3DB: 1,645 seqs, 21,005 structures) | Need to process |
| US-align binary | Pre-compiled, upload to Kaggle | Need to build |

---

## Risk Mitigation

1. **Time budget overflow:** If multi-model approach exceeds 9hr, reduce N_step (200->100), N_cycle (10->6), or process fewer targets with full settings.

2. **DRFold2/RhoFold+ infeasible:** Fall back to RNAPro-only with consensus scoring. RNAPro (template) + RNAPro (de novo) + TBM already gives 15+ candidates per target.

3. **john's TBM too complex:** Start with MMseqs2 template search (pre-computed dataset available) as faster alternative. Add john's method incrementally.

4. **Template-free targets:** These are the hardest. De novo RNAPro + diverse seeds/checkpoints + consensus scoring is our best approach. Adding RhoFold+ would specifically help here.

5. **Overfitting fine-tuned models:** Use held-out validation set (competition provides one) for all hyperparameter decisions.

---

## Success Metrics

| Milestone | TM-score | Key Change |
|-----------|----------|------------|
| Current | 0.361 | Baseline |
| After Tier 0 fixes | 0.40+ | RibonanzaNet2 + N_SAMPLE=5 + multi-seed |
| After Tier 1 templates | 0.48+ | john's TBM + PDB expansion |
| After Tier 2 ensembling | 0.53+ | Consensus scoring + per-target routing |
| Silver medal target | 0.55+ | Full pipeline optimized |

---

## Decision Points

1. **After Sub 1-2 (Day 1):** If RibonanzaNet2 gives >0.03 improvement, confirms our model was broken. Prioritize Tier 0 fixes.
2. **After Sub 5-6 (Day 3):** If de novo beats template mode, our templates are actively hurting. Prioritize de novo for all targets.
3. **After john's TBM (Day 10):** If TBM-only scores >0.50, templates are solved. Shift focus to ensembling (Tier 2).
4. **After Week 3 (Day 21):** If score <0.48, consider radical approach change (pure TBM like john, or agentic tree search).

---

## Team Findings Summary

### From ml-modeling:
- RNAPro architecture: 488M params, 48 PairformerStack blocks, 24 DiffusionTransformer blocks
- Training crop_size=256 tokens (model never trained on long sequences)
- RibonanzaNet2 provides gated injection of RNA-specific features (critical missing piece)
- Template embedder uses zero-initialized final linear (may not be well-trained)
- Confidence head outputs ranking_score (for sample selection)
- N_SAMPLE=5 in single pass is much more efficient than 5 passes x 1 sample

### From creative-thinker:
- Consensus scoring (self-play) can replace external MQA tools
- Diversity > precision for best-of-5 scoring metric
- Per-target routing is essential for Part 2 (mixed template/template-free targets)
- Two-pass approach (fast scan + targeted deep compute) maximizes throughput
- The gap is NOT model quality -- it's template quality + diversity + ensembling

### From ml-dataset:
- Template selection is the single biggest lever
- john's TBM scored 0.591 alone (vs our 0.359) -- same concept, vastly different execution
- RNA3DB: 1,645 distinct RNA sequences, 21,005 crystal structures, 216 Rfam families
- rMSA pipeline: 20% higher F1-score for secondary structure prediction
- Multi-method template search (john TBM + MMseqs2 + embedding-based) is the way forward
- Per-target routing thresholds: >0.7 (TBM), 0.4-0.7 (template+de novo), <0.4 (de novo)

### From computational-biologist:
- RNA backbone C1'-C1' distance varies: ~5.9A (A-form helix) to ~6.5A (extended regions)
- Current constraint values (5.95A) are reasonable for helical regions but too tight for loops/bulges
- Secondary structure provides strong geometric constraints: WC base pairs fix C1'-C1' at ~10.4A, stacking at ~3.3A
- Gap regions should use loop/bulge geometry rather than linear interpolation
- Multi-chain RNA interfaces involve base-pairing and stacking interactions between chains
- BRiQ energy minimization can refine ML predictions but is slow; feasible for top candidates only
- RNA has 7 backbone torsion angles per nucleotide (vs 3 for proteins) -- more conformational freedom
- Fragment-based gap modeling (using known loop structures from PDB) is more biophysically realistic

### From ml-tools-inference:
- DRFold2 is the key second model (used by 1st place, complementary to RNAPro)
- john's TBM likely uses: (a) PDB-wide search (not just competition data), (b) chain-level alignment, (c) structural quality filters
- Time budget estimate: TBM ~15min + RNAPro ~4hr + DRFold2 ~3hr + selection ~30min = fits in 9hr
- US-align runs in milliseconds per pair -- consensus scoring is feasible (105 pairs x 108 targets = ~11,000 comparisons, <10 min total)
- RhoFold+ inference is fast (~1-2 min per target) but needs separate GPU memory
- Multi-model pipeline is feasible if we optimize RNAPro inference (N_step=100, N_cycle=6 for non-critical targets)

### From resource-finder:
- RibonanzaNet2 checkpoint confirmed on Kaggle: `shujun717/ribonanzanet2/pyTorch/alpha/1`
- RNAPro source code dataset: `theoviel/rnapro-src` (already in our metadata)
- MMseqs2 pre-computed templates: `rhijudas/rna-3d-folding-templates` (ready to use)
- DRFold2: GitHub repo exists (robustsp/DRFold2), weights need to be uploaded to Kaggle
- RhoFold+: GitHub repo available (ml4bio/RhoFold), weights downloadable
- john's Part 1 notebook is public and accessible
- PDB RNA structures: ~5,000+ RNA-containing structures available before Sep 2024 cutoff
- US-align binary available for compilation and upload

---

## Verification Status (PI Direct Code Verification)

The following critical claims were verified directly from the RNAPro source code:

| Claim | Status | Evidence |
|-------|--------|----------|
| Training crop_size=256 | **VERIFIED** | `configs_data.py:78` -- `"train_crop_size": 256` |
| Template embedder zero-init final linear | **VERIFIED** | `pairformer.py:1442-1445` -- `nn.init.zeros_(self.final_linear_no_bias.weight)` with `zero_init_final_linear=True` default |
| Constraint embedders exist but disabled | **VERIFIED** | `configs_base.py:234-257` -- all 4 embedders have `"enable": False` |
| Constraint injection via z_init += z_constraint | **VERIFIED** | `RNAPro.py:226-261` -- checks `"constraint_feature" in input_feature_dict`, adds to pair representation |
| ranking_score = 0.8*iptm + 0.2*ptm + 0.5*disorder - 100*has_clash | **VERIFIED** | `sample_confidence.py:177-181` -- scalar per sample |
| Template_idx selects cumulative template combinations | **VERIFIED** | `infer_data_pipeline.py:476-487` -- idx 0=[0], 1=[0,1], 2=[0,1,2], 3=[0,1,2,3], 4=[0,1,2,3,4] |
| RibonanzaNet2 model source path | **VERIFIED** | `README.md:99` -- `shujun717/ribonanzanet2/pyTorch/alpha/1` |
| RNAPro checkpoints on HuggingFace | **VERIFIED** | `README.md:145,178` -- `nvidia/RNAPro-Public-Best-500M` and `nvidia/RNAPro-Private-Best-500M` |
| RibonanzaNet2 missing from our metadata | **VERIFIED** | All 5 kernel-metadata.json files have `"model_sources": []` |
| Distance binning: 3.25-52A, 1.25A steps | **VERIFIED** | `pairformer.py:1363-1365,1396-1399` |
| MSA cropping aligns with target sequence | **VERIFIED** | `infer_data_pipeline.py:176-256` -- MSA crops match target crop |

### Verifier Results (15 verified, 7 minor corrections, 0 wrong):

**Minor corrections applied to this plan:**
- john's TBM-only: 0.591 (was 0.593)
- RNAPro retrospective: 0.648 (was 0.640)
- john's hybrid: 0.577 (unchanged)
- All tool availability claims verified as real

**New finding integrated:**
- ~36%+ of test targets are RNA-protein complexes (from computational-biologist)
- Added as Tier 1E: RNA-protein co-folding priority item

### Plan Impact of Verification:
No critical claims were found to be wrong. Minor score corrections do not change the strategy. The RNA-protein complex finding is significant and has been added as a Tier 1 priority. The core plan (fix RibonanzaNet2 + improve templates + ensemble + co-folding) is confirmed sound.

---

*Plan compiled by PI. Finalized: Feb 7, 2026.*
*8-agent research team: 6 researchers + 1 verifier + 1 PI.*
*15/15 code claims verified. 7 minor external corrections applied. 0 claims wrong.*
*Ready for execution.*
